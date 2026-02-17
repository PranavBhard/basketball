"""
Rosters Repository - Data access for nba_rosters and teams_nba collections.

Handles:
- nba_rosters: Team rosters by season with player status
- teams_nba: Team metadata
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from .base import BaseRepository

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


class RostersRepository(BaseRepository):
    """Repository for nba_rosters collection (team rosters by season)."""

    collection_name = 'nba_rosters'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["rosters"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_roster(self, team: str, season: str) -> Optional[Dict]:
        """Get roster for a team and season."""
        return self.find_one({'team': team, 'season': season})

    def find_by_season(self, season: str) -> List[Dict]:
        """Get all team rosters for a season."""
        return self.find({'season': season})

    def find_player_on_roster(
        self,
        player_id: str,
        season: str
    ) -> Optional[Dict]:
        """Find which team a player is on for a given season."""
        return self.find_one({
            'season': season,
            'roster.player_id': player_id
        })

    def get_active_players(
        self,
        team: str,
        season: str
    ) -> List[Dict]:
        """Get active (is_playing=True) players for a team."""
        roster_doc = self.find_roster(team, season)
        if not roster_doc or 'roster' not in roster_doc:
            return []
        return [p for p in roster_doc['roster'] if p.get('is_playing', True)]

    def get_starters(
        self,
        team: str,
        season: str
    ) -> List[Dict]:
        """Get designated starters for a team."""
        roster_doc = self.find_roster(team, season)
        if not roster_doc or 'roster' not in roster_doc:
            return []
        return [p for p in roster_doc['roster'] if p.get('is_starter', False)]

    def get_injured_players(
        self,
        team: str,
        season: str
    ) -> List[Dict]:
        """Get players marked as not playing (injured/out)."""
        roster_doc = self.find_roster(team, season)
        if not roster_doc or 'roster' not in roster_doc:
            return []
        return [p for p in roster_doc['roster'] if not p.get('is_playing', True)]

    # --- Update Methods ---

    def upsert_roster(
        self,
        team: str,
        season: str,
        roster: List[Dict]
    ) -> bool:
        """Insert or update a team roster."""
        result = self.update_one(
            {'team': team, 'season': season},
            {'$set': {
                'team': team,
                'season': season,
                'roster': roster,
                'updated_at': datetime.utcnow()
            }},
            upsert=True
        )
        return result.acknowledged

    def update_player_status(
        self,
        team: str,
        season: str,
        player_id: str,
        is_playing: bool = None,
        is_starter: bool = None
    ) -> bool:
        """Update a player's status on the roster."""
        update_fields = {}
        if is_playing is not None:
            update_fields['roster.$.is_playing'] = is_playing
        if is_starter is not None:
            update_fields['roster.$.is_starter'] = is_starter

        if not update_fields:
            return False

        result = self.update_one(
            {'team': team, 'season': season, 'roster.player_id': player_id},
            {'$set': update_fields}
        )
        return result.modified_count > 0

    def update_player_lineup_flags(
        self,
        team: str,
        season: str,
        player_id: str,
        *,
        injured: Optional[bool] = None,
        starter: Optional[bool] = None,
    ) -> bool:
        """
        Update prediction-roster flags for a player in `nba_rosters`.

        IMPORTANT:
        - Prediction workflows (and lineup tools) use `roster[].injured` and `roster[].starter`
          in `nba_rosters` as the source of truth.
        - This differs from the older `is_playing` / `is_starter` fields used in some legacy paths.
        """
        update_fields: Dict[str, Any] = {}
        if injured is not None:
            update_fields["roster.$.injured"] = bool(injured)
        if starter is not None:
            update_fields["roster.$.starter"] = bool(starter)
        if not update_fields:
            return False

        update_fields["updated_at"] = datetime.utcnow()
        result = self.update_one(
            {"team": team, "season": season, "roster.player_id": str(player_id)},
            {"$set": update_fields},
            upsert=False,
        )
        return result.modified_count > 0

    def add_player_to_roster(
        self,
        team: str,
        season: str,
        player: Dict
    ) -> bool:
        """Add a player to a team roster."""
        result = self.update_one(
            {'team': team, 'season': season},
            {'$push': {'roster': player}}
        )
        return result.modified_count > 0

    def remove_player_from_roster(
        self,
        team: str,
        season: str,
        player_id: str
    ) -> bool:
        """Remove a player from a team roster."""
        result = self.update_one(
            {'team': team, 'season': season},
            {'$pull': {'roster': {'player_id': player_id}}}
        )
        return result.modified_count > 0

    def transfer_player(
        self,
        player_id: str,
        from_team: str,
        to_team: str,
        season: str,
        player_data: Dict = None
    ) -> bool:
        """Transfer a player from one team to another."""
        # Get player data from source roster if not provided
        if player_data is None:
            from_roster = self.find_roster(from_team, season)
            if from_roster and 'roster' in from_roster:
                player_data = next(
                    (p for p in from_roster['roster'] if p.get('player_id') == player_id),
                    None
                )
            if not player_data:
                return False

        # Update player's team
        player_data['team'] = to_team

        # Remove from source, add to destination
        self.remove_player_from_roster(from_team, season, player_id)
        return self.add_player_to_roster(to_team, season, player_data)

    def bulk_update_playing_status(
        self,
        team: str,
        season: str,
        playing_ids: List[str],
        not_playing_ids: List[str]
    ) -> None:
        """Bulk update is_playing status for multiple players."""
        roster_doc = self.find_roster(team, season)
        if not roster_doc or 'roster' not in roster_doc:
            return

        # Update roster in memory
        for player in roster_doc['roster']:
            pid = player.get('player_id')
            if pid in playing_ids:
                player['is_playing'] = True
            elif pid in not_playing_ids:
                player['is_playing'] = False

        # Save updated roster
        self.update_one(
            {'team': team, 'season': season},
            {'$set': {'roster': roster_doc['roster'], 'updated_at': datetime.utcnow()}}
        )


class TeamsRepository(BaseRepository):
    """Repository for teams_nba collection (team metadata)."""

    collection_name = 'nba_teams'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["teams"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_abbreviation(self, abbrev: str) -> Optional[Dict]:
        """Find team by abbreviation (e.g., 'LAL')."""
        return self.find_one({'abbreviation': abbrev})

    def find_by_name(self, name: str) -> Optional[Dict]:
        """Find team by full name."""
        return self.find_one({'name': name})

    def get_all_teams(self) -> List[Dict]:
        """Get all teams."""
        return self.find({}, sort=[('name', 1)])

    def get_team_abbreviations(self) -> List[str]:
        """Get all team abbreviations."""
        return self.distinct('abbreviation')

    # --- Update Methods ---

    def upsert_team(self, abbrev: str, team_data: Dict) -> bool:
        """Insert or update team metadata."""
        result = self.update_one(
            {'abbreviation': abbrev},
            {'$set': team_data},
            upsert=True
        )
        return result.acknowledged
