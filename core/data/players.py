"""
Players Repository - Data access for player-related collections.

Handles:
- stats_nba_players: Per-game player statistics
- players_nba: Player directory/metadata
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from .base import BaseRepository

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


class PlayerStatsRepository(BaseRepository):
    """Repository for stats_nba_players collection (per-game player stats)."""

    collection_name = 'nba_player_stats'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["player_stats"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_game(self, game_id: str) -> List[Dict]:
        """Get all player stats for a game."""
        return self.find({'game_id': game_id})

    def find_by_player(
        self,
        player_id: str,
        season: str = None,
        before_date: str = None,
        limit: int = 0
    ) -> List[Dict]:
        """
        Get stats for a specific player.

        Args:
            player_id: ESPN player ID (string)
            season: Optional season filter
            before_date: Only games before this date
            limit: Max games to return
        """
        # Handle both string and int player_id for compatibility
        query = {'player_id': {'$in': [player_id, int(player_id) if player_id.isdigit() else player_id]}}
        if season:
            query['season'] = season
        if before_date:
            query['date'] = {'$lt': before_date}
        return self.find(query, sort=[('date', -1)], limit=limit)

    def find_by_team_and_date(
        self,
        team: str,
        game_date: str,
        season: str = None
    ) -> List[Dict]:
        """Get all player stats for a team on a specific date."""
        query = {'team': team, 'date': game_date}
        if season:
            query['season'] = season
        return self.find(query)

    def find_starters(
        self,
        game_id: str = None,
        team: str = None,
        game_date: str = None
    ) -> List[Dict]:
        """Get starters for a game."""
        query = {'starter': True}
        if game_id:
            query['game_id'] = game_id
        if team:
            query['team'] = team
        if game_date:
            query['date'] = game_date
        return self.find(query)

    def find_player_game_stat(
        self,
        player_id: str,
        game_id: str
    ) -> Optional[Dict]:
        """Get a specific player's stats for a specific game."""
        return self.find_one({
            'player_id': {'$in': [player_id, int(player_id) if player_id.isdigit() else player_id]},
            'game_id': game_id
        })

    def find_recent_games(
        self,
        player_id: str,
        before_date: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get player's recent games before a date."""
        query = {
            'player_id': {'$in': [player_id, int(player_id) if player_id.isdigit() else player_id]},
            'date': {'$lt': before_date},
            'active': True
        }
        return self.find(query, sort=[('date', -1)], limit=limit)

    # --- Aggregation Methods ---

    def get_player_season_averages(
        self,
        player_id: str,
        season: str
    ) -> Optional[Dict]:
        """Calculate season averages for a player."""
        pipeline = [
            {'$match': {
                'player_id': {'$in': [player_id, int(player_id) if player_id.isdigit() else player_id]},
                'season': season,
                'active': True
            }},
            {'$group': {
                '_id': '$player_id',
                'games': {'$sum': 1},
                'avg_pts': {'$avg': '$stats.pts'},
                'avg_reb': {'$avg': '$stats.reb'},
                'avg_ast': {'$avg': '$stats.ast'},
                'avg_min': {'$avg': '$stats.min'},
                'avg_fg_pct': {'$avg': {'$cond': [
                    {'$gt': ['$stats.fg_att', 0]},
                    {'$divide': ['$stats.fg_made', '$stats.fg_att']},
                    0
                ]}}
            }}
        ]
        result = self.aggregate(pipeline)
        return result[0] if result else None

    def get_games_with_stats(self) -> List[str]:
        """Get list of game IDs that have player stats."""
        return self.distinct('game_id')

    # --- Update Methods ---

    def upsert_player_stat(
        self,
        player_id: str,
        game_id: str,
        stat_data: Dict
    ) -> bool:
        """Insert or update player stats for a game."""
        result = self.update_one(
            {'player_id': player_id, 'game_id': game_id},
            {'$set': stat_data},
            upsert=True
        )
        return result.acknowledged


class PlayersRepository(BaseRepository):
    """Repository for players_nba collection (player directory)."""

    collection_name = 'nba_players'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["players"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_player_id(self, player_id: str) -> Optional[Dict]:
        """Find player by ID."""
        # Handle both string and int for compatibility
        return self.find_one({
            'player_id': {'$in': [player_id, int(player_id) if player_id.isdigit() else player_id]}
        })

    def find_by_ids(self, player_ids: List[str]) -> List[Dict]:
        """Find multiple players by IDs."""
        # Include both string and int variants
        expanded_ids = []
        for pid in player_ids:
            expanded_ids.append(pid)
            if isinstance(pid, str) and pid.isdigit():
                expanded_ids.append(int(pid))
        return self.find({'player_id': {'$in': expanded_ids}})

    def find_by_name(self, name: str, exact: bool = False) -> List[Dict]:
        """Find players by name."""
        if exact:
            query = {'player_name': name}
        else:
            query = {'player_name': {'$regex': name, '$options': 'i'}}
        return self.find(query)

    def find_by_team(self, team: str, active_only: bool = True) -> List[Dict]:
        """Find players on a team."""
        query = {'team': team}
        if active_only:
            query['active'] = True
        return self.find(query, sort=[('player_name', 1)])

    def search_players(
        self,
        query_str: str,
        limit: int = 20
    ) -> List[Dict]:
        """Search players by name (partial match)."""
        return self.find(
            {'player_name': {'$regex': query_str, '$options': 'i'}},
            sort=[('player_name', 1)],
            limit=limit
        )

    # --- Update Methods ---

    def upsert_player(self, player_id: str, player_data: Dict) -> bool:
        """Insert or update player metadata."""
        result = self.update_one(
            {'player_id': player_id},
            {'$set': player_data},
            upsert=True
        )
        return result.acknowledged

    def update_last_game(self, player_id: str, game_date: str) -> bool:
        """Update player's last game date."""
        result = self.update_one(
            {'player_id': player_id},
            {'$set': {'last_game_date': game_date, 'updated_at': datetime.utcnow()}}
        )
        return result.modified_count > 0

    def set_active_status(self, player_id: str, active: bool) -> bool:
        """Set player's active status."""
        result = self.update_one(
            {'player_id': player_id},
            {'$set': {'active': active, 'updated_at': datetime.utcnow()}}
        )
        return result.modified_count > 0

    def bulk_update_active_status(
        self,
        active_ids: List[str],
        inactive_ids: List[str]
    ) -> None:
        """Bulk update active/inactive status for players."""
        if active_ids:
            self.update_many(
                {'player_id': {'$in': active_ids}},
                {'$set': {'active': True, 'updated_at': datetime.utcnow()}}
            )
        if inactive_ids:
            self.update_many(
                {'player_id': {'$in': inactive_ids}},
                {'$set': {'active': False, 'updated_at': datetime.utcnow()}}
            )
