"""
Elo Rating Cache Module

Manages cached Elo ratings in MongoDB for fast prediction-time lookups.
Elo ratings are computed from historical game data and stored per (team, date, season).

MongoDB Collection: nba_cached_elo_ratings
Document schema:
{
    "team": "Boston Celtics",
    "game_date": "2024-01-15",
    "season": "2023-2024",
    "elo": 1623.5,
    "created_at": ISODate("2024-01-16T10:30:00Z")
}

Indexes:
- (team, game_date, season) - unique, for fast lookups
- (season) - for cache management by season
- (game_date) - for finding latest ratings

Uses data layer repositories for game data access.
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from pymongo import ASCENDING, DESCENDING
from pymongo.database import Database

from bball_app.core.data import GamesRepository

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


# MongoDB collection name
ELO_CACHE_COLLECTION = 'nba_cached_elo_ratings'

# Default Elo parameters
DEFAULT_STARTING_ELO = 1500
DEFAULT_K_FACTOR = 20
DEFAULT_HOME_ADVANTAGE = 100


class EloCache:
    """
    Manages Elo rating computation and caching in MongoDB.

    Usage:
        from bball_app.core.stats.elo_cache import EloCache
        from bball_app.core.mongo import Mongo

        mongo = Mongo()
        elo_cache = EloCache(mongo.db)

        # Compute and cache elo for all games
        stats = elo_cache.compute_and_cache_all()

        # Get elo for a specific game
        elo = elo_cache.get_elo_for_game("Boston Celtics", "2024-01-15", "2023-2024")
    """

    def __init__(
        self,
        db: Database,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
        starting_elo: float = DEFAULT_STARTING_ELO,
        k_factor: float = DEFAULT_K_FACTOR,
        home_advantage: float = DEFAULT_HOME_ADVANTAGE
    ):
        """
        Initialize EloCache.

        Args:
            db: MongoDB database instance
            starting_elo: Initial Elo rating for new teams (default: 1500)
            k_factor: K-factor for Elo updates (default: 20)
            home_advantage: Home court advantage in Elo points (default: 100)
        """
        self.db = db
        self.league = league
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["elo_cache"]
        self.collection = db[effective or ELO_CACHE_COLLECTION]
        self.starting_elo = starting_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage

        # Initialize repository for game data access
        self._games_repo = GamesRepository(db, league=league)

        # In-memory cache for fast lookups (populated by preload())
        self._memory_cache = None  # {(team, game_date, season): elo}
        self._team_latest_cache = None  # {team: [(game_date, elo), ...]} sorted by date desc
        self._preloaded = False

        # Ensure indexes exist
        self._ensure_indexes()

    @property
    def _exclude_game_types(self) -> list:
        """Get excluded game types from league config, with fallback."""
        return self.league.exclude_game_types if self.league else ['preseason', 'allstar']

    def _ensure_indexes(self):
        """Create required indexes if they don't exist."""
        # Unique compound index for lookups
        self.collection.create_index(
            [("team", ASCENDING), ("game_date", ASCENDING), ("season", ASCENDING)],
            unique=True,
            name="team_date_season_unique"
        )
        # Index for season-based queries
        self.collection.create_index(
            [("season", ASCENDING)],
            name="season_idx"
        )
        # Index for finding latest ratings
        self.collection.create_index(
            [("game_date", DESCENDING)],
            name="game_date_desc_idx"
        )
        # Index for team lookups
        self.collection.create_index(
            [("team", ASCENDING), ("game_date", DESCENDING)],
            name="team_date_desc_idx"
        )

    def preload(self, seasons: List[str] = None):
        """
        Preload all elo ratings into memory for fast batch processing.

        This converts per-row DB queries into a single bulk load,
        dramatically improving training data generation performance.

        Args:
            seasons: Optional list of seasons to preload. If None, loads all.
        """
        from collections import defaultdict

        query = {}
        if seasons:
            query['season'] = {'$in': seasons}

        print(f"  Preloading elo cache into memory...")
        docs = list(self.collection.find(query, {'team': 1, 'game_date': 1, 'season': 1, 'elo': 1, '_id': 0}))

        # Build exact match cache: {(team, game_date, season): elo}
        self._memory_cache = {}
        for doc in docs:
            key = (doc['team'], doc['game_date'], doc['season'])
            self._memory_cache[key] = doc['elo']

        # Build team latest cache for fallback lookups: {team: [(game_date, elo), ...]}
        # Sorted by date descending for each team
        team_records = defaultdict(list)
        for doc in docs:
            team_records[doc['team']].append((doc['game_date'], doc['elo']))

        self._team_latest_cache = {}
        for team, records in team_records.items():
            # Sort by date descending
            self._team_latest_cache[team] = sorted(records, key=lambda x: x[0], reverse=True)

        self._preloaded = True
        print(f"  Loaded {len(docs):,} elo records for {len(self._team_latest_cache)} teams")

    def compute_elo_ratings(
        self,
        games: List[dict],
        progress_callback: callable = None
    ) -> Tuple[Dict[Tuple[str, str, str], float], Dict[str, float]]:
        """
        Compute Elo ratings from a list of games.

        Args:
            games: List of game documents sorted chronologically.
                   Each game must have: homeTeam.name, awayTeam.name, date, season, homeWon
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Tuple of (elo_history, current_ratings)
            - elo_history: Dict mapping (team, game_date, season) -> pre-game elo
            - current_ratings: Dict mapping team -> current elo rating
        """
        elo = defaultdict(lambda: self.starting_elo)
        elo_history = {}

        # Filter out games without required fields
        valid_games = [
            g for g in games
            if all([
                g.get('season'),
                g.get('homeTeam', {}).get('name'),
                g.get('awayTeam', {}).get('name'),
                g.get('date'),
                'homeWon' in g
            ])
        ]

        if len(valid_games) < len(games):
            print(f"  Filtered out {len(games) - len(valid_games)} games with missing fields")

        # Sort chronologically
        sorted_games = sorted(valid_games, key=lambda g: g.get('date', ''))
        total = len(sorted_games)

        for idx, game in enumerate(sorted_games):
            home = game['homeTeam']['name']
            away = game['awayTeam']['name']
            game_date = game['date']
            season = game['season']

            # Store pre-game Elo for both teams
            elo_history[(home, game_date, season)] = elo[home]
            elo_history[(away, game_date, season)] = elo[away]

            # Calculate expected win probability for home team
            home_elo_adj = elo[home] + self.home_advantage
            expected_home = 1 / (1 + 10 ** ((elo[away] - home_elo_adj) / 400))

            # Update based on result
            actual_home = 1 if game['homeWon'] else 0
            elo_change = self.k_factor * (actual_home - expected_home)

            elo[home] += elo_change
            elo[away] -= elo_change

            # Progress callback
            if progress_callback and (idx + 1) % 500 == 0:
                progress_callback(idx + 1, total)

        if progress_callback:
            progress_callback(total, total)

        return elo_history, dict(elo)

    def cache_elo_ratings(
        self,
        elo_history: Dict[Tuple[str, str, str], float],
        batch_size: int = 1000,
        progress_callback: callable = None
    ) -> int:
        """
        Store computed Elo ratings in MongoDB cache.

        Args:
            elo_history: Dict mapping (team, game_date, season) -> elo rating
            batch_size: Number of documents to insert per batch
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            Number of documents upserted
        """
        from pymongo import UpdateOne

        total = len(elo_history)
        upserted = 0
        operations = []
        now = datetime.utcnow()

        for idx, ((team, game_date, season), elo) in enumerate(elo_history.items()):
            operations.append(
                UpdateOne(
                    {"team": team, "game_date": game_date, "season": season},
                    {"$set": {
                        "team": team,
                        "game_date": game_date,
                        "season": season,
                        "elo": elo,
                        "updated_at": now
                    }},
                    upsert=True
                )
            )

            # Execute batch
            if len(operations) >= batch_size:
                result = self.collection.bulk_write(operations)
                upserted += result.upserted_count + result.modified_count
                operations = []

                if progress_callback:
                    progress_callback(idx + 1, total)

        # Execute remaining operations
        if operations:
            result = self.collection.bulk_write(operations)
            upserted += result.upserted_count + result.modified_count

        if progress_callback:
            progress_callback(total, total)

        return upserted

    def compute_and_cache_all(
        self,
        seasons: List[str] = None,
        progress_callback: callable = None
    ) -> dict:
        """
        Compute and cache Elo ratings for all games (or specific seasons).

        Args:
            seasons: Optional list of seasons to process (e.g., ['2023-2024', '2024-2025'])
                    If None, processes all seasons.
            progress_callback: Optional callback(stage, current, total, message) for progress

        Returns:
            Dict with cache statistics
        """
        # Build query
        query = {'game_type': {'$nin': self._exclude_game_types}}
        if seasons:
            query['season'] = {'$in': seasons}

        # Stage 1: Fetch games
        if progress_callback:
            progress_callback('fetch', 0, 1, 'Fetching games from database...')

        # Fetch without MongoDB sort to avoid memory limit issues, sort in Python
        games = self._games_repo.find(query)
        games.sort(key=lambda g: g.get('date', ''))
        game_count = len(games)

        if progress_callback:
            progress_callback('fetch', 1, 1, f'Fetched {game_count} games')

        if game_count == 0:
            return {
                'games_processed': 0,
                'ratings_cached': 0,
                'teams': 0,
                'seasons': [],
                'date_range': None
            }

        # Stage 2: Compute Elo
        if progress_callback:
            progress_callback('compute', 0, game_count, 'Computing Elo ratings...')

        def compute_progress(current, total):
            if progress_callback:
                progress_callback('compute', current, total, f'Computing Elo: {current}/{total} games')

        elo_history, current_ratings = self.compute_elo_ratings(games, compute_progress)

        # Stage 3: Cache to MongoDB
        if progress_callback:
            progress_callback('cache', 0, len(elo_history), 'Caching Elo ratings to MongoDB...')

        def cache_progress(current, total):
            if progress_callback:
                progress_callback('cache', current, total, f'Caching: {current}/{total} ratings')

        ratings_cached = self.cache_elo_ratings(elo_history, progress_callback=cache_progress)

        # Gather statistics
        seasons_in_data = sorted(set(g['season'] for g in games if g.get('season')))
        dates = [g['date'] for g in games if g.get('date')]

        return {
            'games_processed': game_count,
            'ratings_cached': ratings_cached,
            'teams': len(current_ratings),
            'seasons': seasons_in_data,
            'date_range': {
                'min': min(dates) if dates else None,
                'max': max(dates) if dates else None
            },
            'current_ratings': current_ratings
        }

    def get_elo_for_game(self, team: str, game_date: str, season: str) -> Optional[float]:
        """
        Get cached pre-game Elo rating for a team.

        Args:
            team: Team name (e.g., "Boston Celtics")
            game_date: Game date string (YYYY-MM-DD)
            season: Season string (e.g., "2023-2024")

        Returns:
            Elo rating if cached, None otherwise
        """
        # Use in-memory cache if preloaded
        if self._preloaded and self._memory_cache is not None:
            return self._memory_cache.get((team, game_date, season))

        # Fall back to DB query
        doc = self.collection.find_one({
            "team": team,
            "game_date": game_date,
            "season": season
        })
        return doc['elo'] if doc else None

    def get_elo_for_game_with_fallback(
        self,
        team: str,
        game_date: str,
        season: str
    ) -> float:
        """
        Get cached Elo rating with fallback to most recent or default.

        Args:
            team: Team name
            game_date: Game date string (YYYY-MM-DD)
            season: Season string

        Returns:
            Elo rating (cached, most recent for team, or default 1500)
        """
        # Try exact match first
        elo = self.get_elo_for_game(team, game_date, season)
        if elo is not None:
            return elo

        # Use in-memory cache for fallback if preloaded
        if self._preloaded and self._team_latest_cache is not None:
            team_records = self._team_latest_cache.get(team)
            if team_records:
                # Find most recent elo before game_date (records are sorted desc)
                for rec_date, rec_elo in team_records:
                    if rec_date < game_date:
                        return rec_elo
            # No prior record found for this team
            return self.starting_elo

        # Fall back to DB query
        doc = self.collection.find_one(
            {"team": team, "game_date": {"$lt": game_date}},
            sort=[("game_date", DESCENDING)]
        )
        if doc:
            return doc['elo']

        # Final fallback: default starting elo
        return self.starting_elo

    def get_current_elo(self, team: str) -> float:
        """
        Get the most recent Elo rating for a team.

        Args:
            team: Team name

        Returns:
            Most recent Elo rating, or default 1500 if not found
        """
        doc = self.collection.find_one(
            {"team": team},
            sort=[("game_date", DESCENDING)]
        )
        return doc['elo'] if doc else self.starting_elo

    def get_all_current_elos(self) -> Dict[str, float]:
        """
        Get the most recent Elo rating for all teams.

        Returns:
            Dict mapping team name -> current elo rating
        """
        pipeline = [
            {"$sort": {"game_date": -1}},
            {"$group": {
                "_id": "$team",
                "elo": {"$first": "$elo"},
                "game_date": {"$first": "$game_date"}
            }},
            {"$sort": {"elo": -1}}
        ]

        results = list(self.collection.aggregate(pipeline))
        return {doc['_id']: doc['elo'] for doc in results}

    def get_cache_stats(self) -> dict:
        """
        Get statistics about the Elo cache.

        Returns:
            Dict with cache statistics
        """
        total_docs = self.collection.count_documents({})

        if total_docs == 0:
            return {
                'total_ratings': 0,
                'teams': 0,
                'seasons': [],
                'date_range': None,
                'last_updated': None
            }

        # Get unique teams and seasons
        teams = self.collection.distinct('team')
        seasons = sorted(self.collection.distinct('season'))

        # Get date range
        oldest = self.collection.find_one(sort=[("game_date", ASCENDING)])
        newest = self.collection.find_one(sort=[("game_date", DESCENDING)])

        # Get last update time
        last_updated_doc = self.collection.find_one(sort=[("updated_at", DESCENDING)])
        last_updated = last_updated_doc.get('updated_at') if last_updated_doc else None

        return {
            'total_ratings': total_docs,
            'teams': len(teams),
            'team_names': sorted(teams),
            'seasons': seasons,
            'date_range': {
                'min': oldest['game_date'] if oldest else None,
                'max': newest['game_date'] if newest else None
            },
            'last_updated': last_updated
        }

    def get_team_elo_history(
        self,
        team: str,
        season: str = None,
        limit: int = 100
    ) -> List[dict]:
        """
        Get Elo rating history for a team.

        Args:
            team: Team name
            season: Optional season filter
            limit: Maximum number of records to return

        Returns:
            List of {game_date, season, elo} dicts, sorted by date descending
        """
        query = {"team": team}
        if season:
            query["season"] = season

        cursor = self.collection.find(
            query,
            {"_id": 0, "game_date": 1, "season": 1, "elo": 1}
        ).sort("game_date", DESCENDING).limit(limit)

        return list(cursor)

    def clear_cache(self, seasons: List[str] = None) -> int:
        """
        Clear Elo cache.

        Args:
            seasons: Optional list of seasons to clear. If None, clears all.

        Returns:
            Number of documents deleted
        """
        if seasons:
            result = self.collection.delete_many({"season": {"$in": seasons}})
        else:
            result = self.collection.delete_many({})

        return result.deleted_count


# Convenience function for quick access
def get_elo_cache(db: Database = None) -> EloCache:
    """
    Get an EloCache instance.

    Args:
        db: Optional MongoDB database. If None, creates new connection.

    Returns:
        EloCache instance
    """
    if db is None:
        from bball_app.core.mongo import Mongo
        db = Mongo().db

    return EloCache(db)
