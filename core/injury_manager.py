"""
Injury Manager Module

Computes and manages injured player data for games.

An injured player is defined as:
- On the team's roster (player has played for team before and after game date)
- Did not play in the game (stats.min == 0 or missing)
- Has at least 1 prior game for this team before the game date
- Last prior game was within 25 days of the game date

This module provides the core logic shared between CLI and web interfaces.

League-aware: Uses normalized collection names (games, player_stats) that work
with LeagueDbProxy for multi-league support.
"""

from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pymongo.database import Database


# Days threshold for recency check
RECENCY_THRESHOLD_DAYS = 25


class InjuryManager:
    """
    Manages injury detection and updates for NBA games.

    Usage:
        from bball_app.core.injury_manager import InjuryManager
        from bball_app.core.mongo import Mongo

        mongo = Mongo()
        injury_mgr = InjuryManager(mongo.db)

        # Update injuries for all games
        stats = injury_mgr.update_all_games()

        # Get injury stats
        stats = injury_mgr.get_injury_stats()
    """

    def __init__(self, db: Database):
        """
        Initialize InjuryManager.

        Args:
            db: MongoDB database instance
        """
        self.db = db

        # Precomputed maps (populated on demand)
        self._player_team_dates: Dict[str, Dict[str, List[str]]] = {}
        self._team_players: Dict[str, Set[str]] = {}
        self._game_team_players_played: Dict[Tuple[str, str], Set[str]] = {}
        self._player_last_game_info: Dict[str, dict] = {}
        self._maps_built = False

    def _build_precomputed_maps(self, progress_callback: callable = None) -> int:
        """
        Build precomputed maps from player_stats for efficient injury detection.

        Args:
            progress_callback: Optional callback(current, total, message) for progress

        Returns:
            Total player records processed
        """
        if self._maps_built:
            return 0

        self._player_team_dates = {}
        self._team_players = {}
        self._game_team_players_played = {}
        self._player_last_game_info = {}

        # Single scan over player_stats collection
        player_stats_cursor = self.db.player_stats.find(
            {'stats.min': {'$gt': 0}},  # Only players who actually played
            {'player_id': 1, 'team': 1, 'date': 1, 'game_id': 1, 'season': 1}
        )

        total_records = 0
        for player_record in player_stats_cursor:
            total_records += 1

            player_id = player_record.get('player_id')
            team = player_record.get('team')
            game_date = player_record.get('date')
            game_id = player_record.get('game_id')
            season = player_record.get('season')

            if not player_id or not team or not game_date or not game_id:
                continue

            # Build player_team_dates
            if player_id not in self._player_team_dates:
                self._player_team_dates[player_id] = {}
            if team not in self._player_team_dates[player_id]:
                self._player_team_dates[player_id][team] = []

            if game_date not in self._player_team_dates[player_id][team]:
                self._player_team_dates[player_id][team].append(game_date)

            # Build team_players
            if team not in self._team_players:
                self._team_players[team] = set()
            self._team_players[team].add(player_id)

            # Build game_team_players_played
            key = (game_id, team)
            if key not in self._game_team_players_played:
                self._game_team_players_played[key] = set()
            self._game_team_players_played[key].add(player_id)

            # Track last game info for each player
            if player_id not in self._player_last_game_info:
                self._player_last_game_info[player_id] = {
                    'date': game_date, 'team': team, 'season': season
                }
            else:
                current_last_date = self._player_last_game_info[player_id]['date']
                if game_date > current_last_date:
                    self._player_last_game_info[player_id] = {
                        'date': game_date, 'team': team, 'season': season
                    }

            if progress_callback and total_records % 10000 == 0:
                progress_callback(total_records, 0, f'Building maps: {total_records} records...')

        # Sort all date lists
        for player_id in self._player_team_dates:
            for team in self._player_team_dates[player_id]:
                self._player_team_dates[player_id][team].sort()

        self._maps_built = True
        return total_records

    def _get_roster_for_game(
        self,
        team: str,
        game_date: str,
        game_season: str
    ) -> Set[str]:
        """
        Get the roster (active players) for a team on a given game date.

        A player is considered on the roster if:
        - They have played for this team before the game date
        - Either they played for this team after the game date, OR
          their last game for any team was for this team in this season

        Args:
            team: Team name
            game_date: Game date (YYYY-MM-DD)
            game_season: Season string

        Returns:
            Set of player IDs on the roster
        """
        roster = set()

        if team not in self._team_players:
            return roster

        for player_id in self._team_players[team]:
            if player_id not in self._player_team_dates:
                continue
            if team not in self._player_team_dates[player_id]:
                continue

            dates = self._player_team_dates[player_id][team]
            if not dates:
                continue

            first_date = dates[0]
            last_date = dates[-1]

            # Player was active for this team during the game date
            if first_date <= game_date <= last_date:
                roster.add(player_id)
            elif first_date <= game_date:
                # Player's first game is before game_date but last is also before
                # Check if their last game (for any team) was for this team in this season
                if player_id in self._player_last_game_info:
                    last_game_info = self._player_last_game_info[player_id]
                    if (last_game_info['team'] == team and
                        last_game_info['season'] == game_season):
                        roster.add(player_id)

        return roster

    def _get_injured_players_for_game(
        self,
        game_id: str,
        team: str,
        game_date: str,
        game_season: str
    ) -> List[str]:
        """
        Get list of injured players for a team in a specific game.

        Args:
            game_id: Game ID
            team: Team name
            game_date: Game date (YYYY-MM-DD)
            game_season: Season string

        Returns:
            List of injured player IDs
        """
        roster = self._get_roster_for_game(team, game_date, game_season)
        played = self._game_team_players_played.get((game_id, team), set())

        injured_candidates = roster - played
        injured_players = []

        for player_id in injured_candidates:
            if player_id not in self._player_team_dates:
                continue
            if team not in self._player_team_dates[player_id]:
                continue

            dates = self._player_team_dates[player_id][team]
            if not dates:
                continue

            # Get prior game dates (dates < game_date)
            prior_dates = [d for d in dates if d < game_date]

            # Check: at least 1 prior game
            if len(prior_dates) == 0:
                continue

            # Check: last prior game within threshold
            last_prior_date = prior_dates[-1]
            date_diff = (
                datetime.strptime(game_date, '%Y-%m-%d').date() -
                datetime.strptime(last_prior_date, '%Y-%m-%d').date()
            ).days

            if date_diff <= RECENCY_THRESHOLD_DAYS:
                injured_players.append(player_id)

        return injured_players

    def compute_injuries_for_game(self, game: dict) -> Tuple[List[str], List[str]]:
        """
        Compute injured players for a single game.

        Args:
            game: Game document with game_id, date, season, homeTeam.name, awayTeam.name

        Returns:
            Tuple of (home_injured_players, away_injured_players)
        """
        if not self._maps_built:
            self._build_precomputed_maps()

        game_id = game.get('game_id')
        game_date = game.get('date')
        game_season = game.get('season')
        home_team = game.get('homeTeam', {}).get('name')
        away_team = game.get('awayTeam', {}).get('name')

        if not all([game_id, game_date, home_team, away_team]):
            return [], []

        home_injured = self._get_injured_players_for_game(
            game_id, home_team, game_date, game_season
        )
        away_injured = self._get_injured_players_for_game(
            game_id, away_team, game_date, game_season
        )

        return home_injured, away_injured

    def update_all_games(
        self,
        game_ids: List[str] = None,
        season: str = None,
        dry_run: bool = False,
        progress_callback: callable = None
    ) -> dict:
        """
        Compute and update injured players for all games.

        Args:
            game_ids: Optional list of specific game IDs to process
            season: Optional season filter (e.g., '2024-2025')
            dry_run: If True, don't update database
            progress_callback: Optional callback(stage, current, total, message)

        Returns:
            Dict with update statistics
        """
        # Stage 1: Build precomputed maps
        if progress_callback:
            progress_callback('build', 0, 1, 'Building player maps from player stats...')

        def build_progress(current, total, message):
            if progress_callback:
                progress_callback('build', current, 0, message)

        total_records = self._build_precomputed_maps(build_progress)

        if progress_callback:
            progress_callback('build', 1, 1, f'Built maps from {total_records} player records')

        # Stage 2: Get games to process
        query = {
            'game_id': {'$exists': True, '$ne': None},
            'homeWon': {'$exists': True},
            'homeTeam.points': {'$exists': True}
        }

        if game_ids:
            query['game_id'] = {'$in': game_ids}

        if season:
            query['season'] = season

        if progress_callback:
            progress_callback('fetch', 0, 1, 'Fetching games...')

        games = list(self.db.games.find(
            query,
            {'game_id': 1, 'date': 1, 'homeTeam.name': 1, 'awayTeam.name': 1, 'season': 1}
        ))

        if not games:
            return {
                'games_processed': 0,
                'games_updated': 0,
                'games_skipped': 0,
                'errors': 0,
                'dry_run': dry_run
            }

        if progress_callback:
            progress_callback('fetch', 1, 1, f'Found {len(games)} games')

        # Stage 3: Process games
        updated_count = 0
        skipped_count = 0
        error_count = 0
        total_home_injured = 0
        total_away_injured = 0

        for idx, game in enumerate(games):
            game_id = game.get('game_id')

            if progress_callback and (idx + 1) % 100 == 0:
                progress_callback('process', idx + 1, len(games),
                    f'Processing game {idx + 1}/{len(games)}...')

            try:
                home_injured, away_injured = self.compute_injuries_for_game(game)
                total_home_injured += len(home_injured)
                total_away_injured += len(away_injured)

                if not dry_run:
                    result = self.db.games.update_one(
                        {'game_id': game_id},
                        {'$set': {
                            'homeTeam.injured_players': home_injured,
                            'awayTeam.injured_players': away_injured
                        }}
                    )

                    if result.modified_count > 0:
                        updated_count += 1
                    else:
                        skipped_count += 1
                else:
                    if len(home_injured) + len(away_injured) > 0:
                        updated_count += 1
                    else:
                        skipped_count += 1

            except Exception as e:
                error_count += 1

        if progress_callback:
            progress_callback('process', len(games), len(games), 'Complete!')

        return {
            'games_processed': len(games),
            'games_updated': updated_count,
            'games_skipped': skipped_count,
            'errors': error_count,
            'total_home_injured': total_home_injured,
            'total_away_injured': total_away_injured,
            'dry_run': dry_run
        }

    def get_injury_stats(self) -> dict:
        """
        Get statistics about injury data in the database.

        Returns:
            Dict with injury statistics
        """
        # Count games with injury data
        games_with_injuries = self.db.games.count_documents({
            'homeTeam.injured_players': {'$exists': True, '$ne': []},
        })

        # Count games missing injury data
        games_without_injuries = self.db.games.count_documents({
            '$or': [
                {'homeTeam.injured_players': {'$exists': False}},
                {'homeTeam.injured_players': []}
            ],
            'homeWon': {'$exists': True}  # Only count completed games
        })

        total_games = self.db.games.count_documents({
            'homeWon': {'$exists': True}
        })

        # Get seasons with injury data
        seasons_pipeline = [
            {'$match': {'homeTeam.injured_players': {'$exists': True, '$ne': []}}},
            {'$group': {'_id': '$season'}},
            {'$sort': {'_id': -1}}
        ]
        seasons = [doc['_id'] for doc in self.db.games.aggregate(seasons_pipeline)]

        # Get date range of games with injuries
        date_range = None
        oldest = self.db.games.find_one(
            {'homeTeam.injured_players': {'$exists': True}},
            sort=[('date', 1)]
        )
        newest = self.db.games.find_one(
            {'homeTeam.injured_players': {'$exists': True}},
            sort=[('date', -1)]
        )
        if oldest and newest:
            date_range = {
                'min': oldest.get('date'),
                'max': newest.get('date')
            }

        return {
            'total_games': total_games,
            'games_with_injuries': games_with_injuries,
            'games_without_injuries': games_without_injuries,
            'seasons': seasons,
            'date_range': date_range,
            'coverage_percent': round(100 * games_with_injuries / total_games, 1) if total_games > 0 else 0
        }

    def get_games_with_injuries(
        self,
        season: str = None,
        date: str = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[dict]:
        """
        Get games with their injury data for audit/viewing.

        Args:
            season: Optional season filter
            date: Optional date filter (YYYY-MM-DD)
            limit: Maximum number of games to return
            offset: Number of games to skip

        Returns:
            List of game documents with injury info
        """
        query = {
            'homeWon': {'$exists': True}  # Completed games only
        }

        if season:
            query['season'] = season
        if date:
            query['date'] = date

        cursor = self.db.games.find(
            query,
            {
                'game_id': 1,
                'date': 1,
                'season': 1,
                'homeTeam.name': 1,
                'homeTeam.injured_players': 1,
                'homeTeam.points': 1,
                'awayTeam.name': 1,
                'awayTeam.injured_players': 1,
                'awayTeam.points': 1,
                'homeWon': 1
            }
        ).sort('date', -1).skip(offset).limit(limit)

        return list(cursor)

    def get_player_name(self, player_id: str) -> Optional[str]:
        """
        Get player name from player_id.

        Args:
            player_id: Player ID

        Returns:
            Player name or None if not found
        """
        player = self.db.player_stats.find_one(
            {'player_id': player_id},
            {'player_name': 1}
        )
        return player.get('player_name') if player else None

    def get_player_names_batch(self, player_ids: List[str]) -> Dict[str, str]:
        """
        Get player names for multiple player IDs.

        Args:
            player_ids: List of player IDs

        Returns:
            Dict mapping player_id -> player_name
        """
        if not player_ids:
            return {}

        cursor = self.db.player_stats.find(
            {'player_id': {'$in': player_ids}},
            {'player_id': 1, 'player_name': 1}
        )

        result = {}
        for doc in cursor:
            result[doc['player_id']] = doc.get('player_name', 'Unknown')

        return result

    def clear_injury_data(self, season: str = None) -> int:
        """
        Clear injury data from games.

        Args:
            season: Optional season filter. If None, clears all.

        Returns:
            Number of games updated
        """
        query = {}
        if season:
            query['season'] = season

        result = self.db.games.update_many(
            query,
            {'$unset': {
                'homeTeam.injured_players': '',
                'awayTeam.injured_players': ''
            }}
        )

        return result.modified_count


def get_injury_manager(db: Database = None) -> InjuryManager:
    """
    Get an InjuryManager instance.

    Args:
        db: Optional MongoDB database. If None, creates new connection.

    Returns:
        InjuryManager instance
    """
    if db is None:
        from bball_app.core.mongo import Mongo
        db = Mongo().db

    return InjuryManager(db)
