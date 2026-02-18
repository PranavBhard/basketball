"""
Games Repository - Data access for stats_nba collection.

Handles all game-related database operations including:
- Game lookups by ID, date, team, season
- Game statistics and predictions storage
- Pregame odds and venue data
"""

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime
from sportscore.db.base_repository import BaseRepository

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


class GamesRepository(BaseRepository):
    """Repository for stats_nba collection (game data)."""

    collection_name = 'stats_nba'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["games"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_game_id(self, game_id: str) -> Optional[Dict]:
        """Find a game by its ESPN game ID."""
        return self.find_one({'game_id': game_id})

    def find_by_date(self, date: str, game_type: str = None) -> List[Dict]:
        """
        Find all games on a specific date.

        Args:
            date: Date string in YYYY-MM-DD format
            game_type: Optional filter (e.g., 'regseason', 'playoff')
        """
        query = {'date': date}
        if game_type:
            query['game_type'] = game_type
        return self.find(query, sort=[('gametime', 1)])

    def find_by_date_range(
        self,
        start_date: str,
        end_date: str,
        completed_only: bool = False
    ) -> List[Dict]:
        """Find games within a date range."""
        query = {'date': {'$gte': start_date, '$lte': end_date}}
        if completed_only:
            query['homeWon'] = {'$exists': True}
        return self.find(query, sort=[('date', 1), ('gametime', 1)])

    def find_by_season(
        self,
        season: str,
        team: str = None,
        completed_only: bool = False,
        limit: int = 0
    ) -> List[Dict]:
        """
        Find games for a season.

        Args:
            season: Season string (e.g., '2024-2025')
            team: Optional team abbreviation to filter by
            completed_only: Only return games with results
            limit: Max games to return (0 for all)
        """
        query = {'season': season}
        if team:
            query['$or'] = [
                {'homeTeam.name': team},
                {'awayTeam.name': team}
            ]
        if completed_only:
            query['homeWon'] = {'$exists': True}
        return self.find(query, sort=[('date', 1)], limit=limit)

    def find_team_games(
        self,
        team: str,
        before_date: str = None,
        season: str = None,
        limit: int = 10,
        home_only: bool = False,
        away_only: bool = False
    ) -> List[Dict]:
        """
        Find recent games for a team.

        Args:
            team: Team abbreviation (e.g., 'LAL')
            before_date: Only games before this date
            season: Filter by season
            limit: Max games to return
            home_only: Only home games
            away_only: Only away games
        """
        if home_only:
            query = {'homeTeam.name': team}
        elif away_only:
            query = {'awayTeam.name': team}
        else:
            query = {'$or': [
                {'homeTeam.name': team},
                {'awayTeam.name': team}
            ]}

        if before_date:
            query['date'] = {'$lt': before_date}
        if season:
            query['season'] = season

        # Only completed games
        query['homeWon'] = {'$exists': True}

        return self.find(query, sort=[('date', -1)], limit=limit)

    def find_head_to_head(
        self,
        team1: str,
        team2: str,
        before_date: str = None,
        limit: int = 10
    ) -> List[Dict]:
        """Find head-to-head matchups between two teams."""
        query = {
            '$or': [
                {'homeTeam.name': team1, 'awayTeam.name': team2},
                {'homeTeam.name': team2, 'awayTeam.name': team1}
            ],
            'homeWon': {'$exists': True}
        }
        if before_date:
            query['date'] = {'$lt': before_date}
        return self.find(query, sort=[('date', -1)], limit=limit)

    def find_games_with_venue(
        self,
        venue_guid: str = None,
        has_venue: bool = True
    ) -> List[Dict]:
        """Find games with venue information."""
        if venue_guid:
            query = {'venue_guid': venue_guid}
        elif has_venue:
            query = {'venue_guid': {'$exists': True, '$ne': None}}
        else:
            query = {'$or': [
                {'venue_guid': {'$exists': False}},
                {'venue_guid': None}
            ]}
        return self.find(query)

    # --- Aggregation Methods ---

    def get_seasons(self, completed_only: bool = True) -> List[str]:
        """Get list of all seasons with games."""
        query = {}
        if completed_only:
            query['homeWon'] = {'$exists': True}
        return sorted(self.distinct('season', query), reverse=True)

    def get_team_record(
        self,
        team: str,
        season: str = None,
        before_date: str = None
    ) -> Dict[str, int]:
        """Get team win/loss record."""
        match_stage = {'homeWon': {'$exists': True}}
        if season:
            match_stage['season'] = season
        if before_date:
            match_stage['date'] = {'$lt': before_date}

        pipeline = [
            {'$match': match_stage},
            {'$facet': {
                'home_wins': [
                    {'$match': {'homeTeam.name': team, 'homeWon': True}},
                    {'$count': 'count'}
                ],
                'home_losses': [
                    {'$match': {'homeTeam.name': team, 'homeWon': False}},
                    {'$count': 'count'}
                ],
                'away_wins': [
                    {'$match': {'awayTeam.name': team, 'homeWon': False}},
                    {'$count': 'count'}
                ],
                'away_losses': [
                    {'$match': {'awayTeam.name': team, 'homeWon': True}},
                    {'$count': 'count'}
                ]
            }}
        ]
        result = self.aggregate(pipeline)
        if not result:
            return {'wins': 0, 'losses': 0}

        data = result[0]
        home_wins = data['home_wins'][0]['count'] if data['home_wins'] else 0
        home_losses = data['home_losses'][0]['count'] if data['home_losses'] else 0
        away_wins = data['away_wins'][0]['count'] if data['away_wins'] else 0
        away_losses = data['away_losses'][0]['count'] if data['away_losses'] else 0

        return {
            'wins': home_wins + away_wins,
            'losses': home_losses + away_losses,
            'home_wins': home_wins,
            'home_losses': home_losses,
            'away_wins': away_wins,
            'away_losses': away_losses
        }

    # --- Update Methods ---

    def upsert_game(self, game_id: str, game_data: Dict) -> bool:
        """Insert or update a game by game_id."""
        result = self.update_one(
            {'game_id': game_id},
            {'$set': game_data},
            upsert=True
        )
        return result.acknowledged

    def save_prediction(self, game_id: str, prediction: Dict) -> bool:
        """Save prediction for a game."""
        result = self.update_one(
            {'game_id': game_id},
            {'$set': {'last_prediction': prediction}},
            upsert=True
        )
        return result.acknowledged

    def update_pregame_lines(self, game_id: str, lines: Dict) -> bool:
        """Update pregame betting lines for a game."""
        result = self.update_one(
            {'game_id': game_id},
            {'$set': {'pregame_lines': lines}}
        )
        return result.modified_count > 0

    def mark_injured_players(
        self,
        game_id: str,
        home_injured: List[str],
        away_injured: List[str]
    ) -> bool:
        """Update injured player lists for a game."""
        result = self.update_one(
            {'game_id': game_id},
            {'$set': {
                'home_injured_players': home_injured,
                'away_injured_players': away_injured
            }}
        )
        return result.modified_count > 0

    # --- Utility Methods ---

    def game_exists(self, game_id: str) -> bool:
        """Check if a game exists in the database."""
        return self.exists({'game_id': game_id})

    def is_completed(self, game_id: str) -> bool:
        """Check if a game has been completed (has result)."""
        game = self.find_one({'game_id': game_id}, {'homeWon': 1})
        return game is not None and 'homeWon' in game
