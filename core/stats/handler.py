"""
StatHandlerV2 - Core Feature Calculation Engine

Enhanced stat handler with exponential weighting and absolute values.

Improvements over StatHandler:
- Exponential decay weighting for historical averages
- Option to return absolute values alongside differentials
- Cleaner interface for rest/fatigue calculations
- Pace and volatility features
- Games played so far (reliability signal)
- Era normalization support
"""

import bisect
import math
import numpy as np
from datetime import date, datetime, timedelta
from collections import defaultdict
from bball_app.core.utils.collection import import_collection
from bball_app.core.utils.db_queries import (
    getTeamSeasonGamesFromDate,
    getTeamLastNMonthsSeasonGames,
    getTeamLastNDaysSeasonGames
)
from bball_app.core.mongo import Mongo
from bball_app.core.features.parser import parse_feature_name
from bball_app.core.features.registry import FeatureRegistry
from bball_app.core.data import (
    GamesRepository, PlayerStatsRepository, PlayersRepository,
    RostersRepository
)
from bball_app.core.league_config import LeagueConfig, load_league_config


class StatHandlerV2:
    """
    Enhanced stat handler with exponential weighting and absolute value support.
    """
    
    def __init__(
        self,
        statistics: list,
        include_absolute: bool = False,
        use_exponential_weighting: bool = False,
        exponential_lambda: float = 0.1,
        preloaded_games: list = None,
        league_averages: dict = None,
        db = None,
        lazy_load: bool = False,
        league: LeagueConfig = None
    ):
        """
        Initialize StatHandlerV2.

        Args:
            statistics: List of stat tokens to compute
            include_absolute: If True, return absolute values alongside diffs for key stats
            use_exponential_weighting: If True, apply exponential decay to averages
            exponential_lambda: Decay rate for exponential weighting
            preloaded_games: Optional preloaded game data to avoid redundant DB calls
            league_averages: Optional dict of {season: {stat: value}} for era normalization
            db: Optional MongoDB database connection for venue lookups
            lazy_load: If True and preloaded_games is None, skip loading all games upfront.
                      Games will be queried on-demand from DB as needed (for prediction use cases).
                      If False, loads all games upfront (for training use cases).
            league: LeagueConfig for league-specific collection access. Defaults to NBA if None.
        """
        # Initialize league config
        if league is None:
            league = load_league_config("nba")
        self.league = league
        self.statistics = statistics
        self.include_absolute = include_absolute
        self.use_exponential_weighting = use_exponential_weighting
        self.exponential_lambda = exponential_lambda
        self.league_averages = league_averages or {}
        self.db = db
        self.lazy_load = lazy_load
        self._venue_cache = {}  # Cache venue locations to avoid repeated DB queries

        # Initialize repositories (only if db is provided) - league-aware
        if db is not None:
            self._games_repo = GamesRepository(db, league=self.league)
            self._players_repo = PlayerStatsRepository(db, league=self.league)
            self._players_dir_repo = PlayersRepository(db, league=self.league)
            self._rosters_repo = RostersRepository(db, league=self.league)
        else:
            self._games_repo = None
            self._players_repo = None
            self._players_dir_repo = None
            self._rosters_repo = None
        
        # Injury feature caches (preloaded to avoid per-game DB queries)
        self._injury_player_stats_cache = {}  # dict[(team, season, date_str)] -> dict[player_id] -> stats
        self._injury_max_mpg_cache = {}  # dict[(team, season, date_str)] -> float
        self._injury_rotation_mpg_cache = {}  # dict[(team, season, date_str)] -> float
        self._injury_preloaded_players = {}  # dict[(team, season)] -> list of player records
        self._injury_cache_loaded = False
        
        # Cache for get_team_games_before_date to avoid repeated iterations
        self._team_games_cache = {}  # dict[(team, season, date_str)] -> list of games

        # Cache for H2H games to avoid repeated DB queries
        self._h2h_games_cache = {}  # dict[(team1, team2, date_str, n_games)] -> list of games

        # Cache for time-period (days_N / months_N) lookups
        self._team_time_period_cache = {}  # dict[(team, season, date_str, period_key)] -> list of games

        # Team-indexed structures for O(log N) bisect lookups (built after games load)
        self._team_games_index = {}  # {season: {team: [(date_str, game_doc), ...]}}
        self._team_dates_index = {}  # {season: {team: [date_str, ...]}}

        # Load all games into memory (or use preloaded data, or skip for lazy loading)
        if preloaded_games is not None:
            self.all_games = preloaded_games
            self.games_home = self.all_games[0]
            self.games_away = self.all_games[1]
            self._build_team_index()
        elif lazy_load:
            # Don't load all games upfront - will query on-demand
            self.all_games = None
            self.games_home = None
            self.games_away = None
        else:
            # Default behavior: load all games using league-aware collection name
            games_collection = self.league.collections.get("games", "stats_nba")
            self.all_games = import_collection(games_collection)
            self.games_home = self.all_games[0]
            self.games_away = self.all_games[1]
            self._build_team_index()

    @property
    def _exclude_game_types(self) -> list:
        """Get excluded game types from league config, with fallback."""
        return self.league.exclude_game_types if self.league else ['preseason', 'allstar']

    def _build_team_index(self):
        """
        Build secondary index: {season: {team: [(date_str, game_doc), ...]}}
        sorted by date_str for O(log N) bisect lookups.

        Iterates games_home once — each game appears once keyed by home team,
        so we append to both home and away team lists.
        """
        team_index = {}   # {season: {team: [(date_str, game_doc), ...]}}
        dates_index = {}  # {season: {team: [date_str, ...]}}

        for season, date_dict in self.games_home.items():
            season_teams = team_index.setdefault(season, defaultdict(list))
            for date_str, teams_dict in date_dict.items():
                for home_team, game in teams_dict.items():
                    # Append to home team's list
                    season_teams[home_team].append((date_str, game))
                    # Append to away team's list
                    away_team = game.get('awayTeam', {}).get('name')
                    if away_team:
                        season_teams[away_team].append((date_str, game))

        # Sort each team's list by date_str (YYYY-MM-DD sorts lexicographically)
        # and build parallel dates-only list for bisect
        for season, teams in team_index.items():
            dates_index[season] = {}
            for team, pairs in teams.items():
                pairs.sort(key=lambda p: p[0])
                dates_index[season][team] = [p[0] for p in pairs]

        self._team_games_index = team_index
        self._team_dates_index = dates_index

    def _get_team_games_in_range(
        self, team: str, season: str,
        begin_date_str: str = None, end_date_str: str = None,
        exclude_game_types: list = None
    ) -> list:
        """
        Fast O(log G) lookup of a team's games in [begin_date_str, end_date_str).

        Uses bisect on the pre-sorted _team_dates_index.

        Args:
            team: Team name
            season: Season string
            begin_date_str: Inclusive start date (YYYY-MM-DD). None = from beginning.
            end_date_str: Exclusive end date (YYYY-MM-DD). None = to end.
            exclude_game_types: Game types to filter out. None = use self._exclude_game_types.

        Returns:
            List of game docs sorted by date.
        """
        dates = self._team_dates_index.get(season, {}).get(team)
        if not dates:
            return []

        pairs = self._team_games_index[season][team]

        # Compute slice bounds via bisect
        lo = bisect.bisect_left(dates, begin_date_str) if begin_date_str else 0
        hi = bisect.bisect_left(dates, end_date_str) if end_date_str else len(dates)

        if exclude_game_types is None:
            exclude_game_types = self._exclude_game_types

        if not exclude_game_types:
            return [pair[1] for pair in pairs[lo:hi]]

        exclude_set = set(exclude_game_types)
        return [
            game for _, game in pairs[lo:hi]
            if game.get('game_type', 'regseason') not in exclude_set
        ]

    def avg(self, ls: list) -> float:
        """Simple average."""
        return sum(ls) / float(len(ls)) if ls else 0
    
    def std(self, ls: list) -> float:
        """Standard deviation."""
        if len(ls) < 2:
            return 0
        mean = self.avg(ls)
        variance = sum((x - mean) ** 2 for x in ls) / float(len(ls) - 1)
        return math.sqrt(variance)
    
    def preload_venue_cache(self):
        """Preload all venue locations into memory cache."""
        if self.db is None:
            return
        
        venues_coll = self.league.collections.get("venues", "nba_venues")
        venues = list(self.db[venues_coll].find(
            {},
            {'venue_guid': 1, 'location.lat': 1, 'location.lon': 1, 'location.long': 1}
        ))
        
        for venue in venues:
            venue_guid = venue.get('venue_guid')
            location = venue.get('location', {})
            if venue_guid and location:
                lat = location.get('lat')
                # Support both 'lon' and 'long' keys from DB
                lon = location.get('lon') if 'lon' in location else location.get('long')
                if lat is not None and lon is not None:
                    self._venue_cache[venue_guid] = (lat, lon)
        
        print(f"Preloaded {len(self._venue_cache)} venue locations into cache")
    
    def preload_injury_cache(self, games: list = None):
        """
        Preload player stats for injury feature calculations to avoid per-game DB queries.

        OPTIMIZED: Uses batch queries per season instead of per (team, season) to reduce
        DB round-trips from O(teams × seasons) to O(seasons).

        Args:
            games: Optional list of game documents. If None, will query all games from DB.
                  Should include 'homeTeam.name', 'awayTeam.name', 'season', 'date' fields.
        """
        if self.db is None:
            return

        if games is None:
            # Query all games from DB (exclude preseason/allstar)
            games = self._games_repo.find(
                {
                    'homeTeam.points': {'$exists': True, '$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={'homeTeam.name': 1, 'awayTeam.name': 1, 'season': 1, 'date': 1}
            )

        # Extract unique teams and seasons
        teams_by_season = defaultdict(set)
        for game in games:
            home_team = game.get('homeTeam', {}).get('name')
            away_team = game.get('awayTeam', {}).get('name')
            season = game.get('season')
            if home_team and season:
                teams_by_season[season].add(home_team)
            if away_team and season:
                teams_by_season[season].add(away_team)

        total_team_seasons = sum(len(teams) for teams in teams_by_season.values())
        print(f"Preloading injury cache for {total_team_seasons} team-season combinations "
              f"across {len(teams_by_season)} seasons...")

        # OPTIMIZED: Batch load by season (1 query per season instead of per team-season)
        # This reduces ~350 queries to ~1 query for single-season CBB processing
        total_records = 0
        for season, teams in teams_by_season.items():
            # Single query for all teams in this season
            all_records = list(self._players_repo.find(
                {
                    'season': season,
                    'team': {'$in': list(teams)},
                    'stats.min': {'$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'player_id': 1,
                    'team': 1,
                    'season': 1,
                    'date': 1,
                    'stats.min': 1
                }
            ))

            # Index by (team, season) and sort by date
            records_by_team = defaultdict(list)
            for record in all_records:
                team = record.get('team')
                if team:
                    records_by_team[team].append(record)

            # Sort each team's records by date and store
            for team, records in records_by_team.items():
                sorted_records = sorted(records, key=lambda r: r.get('date', ''))
                self._injury_preloaded_players[(team, season)] = sorted_records
                total_records += len(sorted_records)

        print(f"  Preloaded {total_records:,} player records for {total_team_seasons} team-seasons "
              f"({len(teams_by_season)} batch queries)")
        self._injury_cache_loaded = True

    def precompute_season_severity(self, games: list = None) -> dict:
        """
        Precompute season injury severity for ALL (team, season, date) combinations.

        Uses incremental computation - iterates chronologically through each team's
        games and maintains running totals. This is O(G) total instead of O(G²)
        for on-demand computation.

        IMPORTANT: Must be called AFTER preload_injury_cache() so that player stats
        are available for rotation MPG calculations.

        Args:
            games: List of game documents. Should include 'homeTeam', 'awayTeam',
                   'season', 'date' fields. If None, uses preloaded games.

        Returns:
            Dict mapping (team, season, date) -> severity value
        """
        if not self._injury_cache_loaded:
            print("Warning: precompute_season_severity called before injury cache loaded. "
                  "Season severity cache will be empty.")
            return {}

        # Build list of games if not provided
        if games is None:
            if self.games_home is None or self.games_away is None:
                print("Warning: No games available for season severity precomputation.")
                return {}
            games = []
            for season_data in self.games_home.values():
                for date_data in season_data.values():
                    for game in date_data.values():
                        games.append(game)

        # Group games by (team, season) - each team appears in games as home or away
        team_season_games = defaultdict(list)
        for game in games:
            home_team = game.get('homeTeam', {}).get('name')
            away_team = game.get('awayTeam', {}).get('name')
            season = game.get('season')
            game_date = game.get('date')

            # Only include completed games
            if game.get('homeWon') is None:
                continue

            if home_team and season and game_date:
                team_season_games[(home_team, season)].append({
                    'date': game_date,
                    'is_home': True,
                    'injured_players': game.get('homeTeam', {}).get('injured_players', [])
                })
            if away_team and season and game_date:
                team_season_games[(away_team, season)].append({
                    'date': game_date,
                    'is_home': False,
                    'injured_players': game.get('awayTeam', {}).get('injured_players', [])
                })

        print(f"Precomputing season severity for {len(team_season_games)} team-season combinations...")

        severity_cache = {}
        EPS = 1e-6

        for (team, season), team_games in team_season_games.items():
            # Sort games chronologically
            sorted_games = sorted(team_games, key=lambda g: g['date'])

            # Build cumulative player stats for this team-season
            # We need to track minutes per player as games progress
            player_cumulative = defaultdict(lambda: {'total_min': 0.0, 'games': 0})

            # Get all player records for this team-season (already preloaded)
            player_records = self._injury_preloaded_players.get((team, season), [])

            # Index player records by date for efficient lookup
            records_by_date = defaultdict(list)
            for record in player_records:
                rec_date = record.get('date', '')
                records_by_date[rec_date].append(record)

            # Running totals for season severity
            running_min_lost = 0.0
            running_rotation_mpg = 0.0

            for game_info in sorted_games:
                game_date = game_info['date']
                injured_ids = [str(pid) for pid in game_info['injured_players']] if game_info['injured_players'] else []

                # Store severity BEFORE this game (based on prior games)
                if running_rotation_mpg > 0:
                    severity = running_min_lost / (running_rotation_mpg + EPS)
                else:
                    severity = 0.0
                severity_cache[(team, season, game_date)] = severity

                # Compute rotation MPG at this point (using cumulative stats)
                # Rotation players are those with MPG >= 10
                game_rotation_mpg = 0.0
                for player_id, stats in player_cumulative.items():
                    if stats['games'] > 0:
                        mpg = stats['total_min'] / stats['games']
                        if mpg >= 10.0:
                            game_rotation_mpg += mpg

                # Compute min_lost for this game (MPG of injured rotation players)
                game_min_lost = 0.0
                if injured_ids:
                    for player_id in injured_ids:
                        if player_id in player_cumulative:
                            stats = player_cumulative[player_id]
                            if stats['games'] > 0:
                                mpg = stats['total_min'] / stats['games']
                                if mpg >= 10.0:  # Only rotation players
                                    game_min_lost += mpg

                # Update running totals
                running_rotation_mpg += game_rotation_mpg
                running_min_lost += game_min_lost

                # Update cumulative player stats with this game's data
                for record in records_by_date.get(game_date, []):
                    player_id = str(record.get('player_id', ''))
                    minutes = record.get('stats', {}).get('min', 0.0)
                    if minutes > 0:
                        player_cumulative[player_id]['total_min'] += minutes
                        player_cumulative[player_id]['games'] += 1

        print(f"  Precomputed {len(severity_cache)} season severity values")

        # Store in instance cache as well for direct access
        if not hasattr(self, '_season_injury_severity_cache'):
            self._season_injury_severity_cache = {}
        self._season_injury_severity_cache.update(severity_cache)

        return severity_cache

    def _get_venue_location(self, venue_guid: str) -> tuple:
        """
        Get venue location (lat, lon) from cache or database.
        
        Returns:
            Tuple of (lat, lon) or (None, None) if not found
        """
        
        # Check cache first
        if venue_guid in self._venue_cache:
            return self._venue_cache[venue_guid]
        
        # If no DB connection, return None
        if self.db is None:
            return None, None
        
        # Query database (fallback if not preloaded)
        venues_coll = self.league.collections.get("venues", "nba_venues")
        venue = self.db[venues_coll].find_one(
            {'venue_guid': venue_guid},
            {'location.lat': 1, 'location.lon': 1, 'location.long': 1}
        )
        
        if venue and venue.get('location'):
            lat = venue['location'].get('lat')
            # Support both 'lon' and 'long'
            lon = venue['location'].get('lon') if 'lon' in venue['location'] else venue['location'].get('long')
            self._venue_cache[venue_guid] = (lat, lon)
            return lat, lon
        
        # Cache None to avoid repeated queries
        self._venue_cache[venue_guid] = (None, None)
        return None, None
    
    def _calculate_travel_distance(self, team: str, year: int, month: int, day: int,
                                   season: str, n_days: int, target_venue_guid: str = None) -> float:
        """
        Calculate total travel distance for a team over the last n_days, including travel to the target game.

        Uses haversine formula to compute distance between venues.
        Uses cached games_home/games_away to avoid DB queries.

        Args:
            team: Team name
            year, month, day: Target game date
            season: NBA season string
            n_days: Number of days to look back
            target_venue_guid: Venue GUID of the target game (included as final destination)
        """
        import logging
        logger = logging.getLogger(__name__)
        if self.db is None:
            logger.debug(f"[TRAVEL] DB is None, returning 0.0 for travel distance ({team}, last {n_days} days)")
            return 0.0
        
        target_date = date(year, month, day)
        start_date = target_date - timedelta(days=n_days)
        
        # Get games in the date range - use helper method if lazy loading, otherwise use preloaded structure
        if self.games_home is None or self.games_away is None:
            # Lazy loading: use _get_team_games_last_n_days helper
            all_team_games = self._get_team_games_last_n_days(team, year, month, day, season, n_days)
            games = [g for g in all_team_games if g.get('game_type', 'regseason') not in self._exclude_game_types]
        else:
            # Use team index for O(log N) bisect lookup
            begin_str = start_date.strftime('%Y-%m-%d')
            end_str = f"{year}-{month:02d}-{day:02d}"
            games = self._get_team_games_in_range(team, season, begin_date_str=begin_str, end_date_str=end_str)
        
        if not games:
            logger.debug(f"[TRAVEL] No games for {team} in last {n_days} days before {year}-{month:02d}-{day:02d} ({season})")
            return 0.0
        
        total_distance = 0.0
        prev_venue_guid = None
        prev_lat = None
        prev_lon = None
        games_with_guid = 0
        games_with_coords = 0
        
        for game in games:
            # Try multiple possible keys for venue identifier
            current_venue_guid = (
                game.get('venue_guid') or
                game.get('venueGuid') or
                game.get('arena_guid') or
                game.get('arenaId') or
                (game.get('venue') or {}).get('venue_guid') or
                (game.get('venue') or {}).get('guid')
            )
            if not current_venue_guid:
                logger.debug(f"[TRAVEL] Missing venue id for game {game.get('game_id')} on {game.get('date')} for {team}. Available keys: {list(game.keys())}")
                continue
            games_with_guid += 1
            
            # Get venue location
            lat, lon = self._get_venue_location(current_venue_guid)
            
            if lat is None or lon is None:
                logger.debug(f"[TRAVEL] No coords for venue {current_venue_guid} (team={team})")
                continue
            games_with_coords += 1
            
            # If we have a previous venue, calculate distance
            if prev_venue_guid and prev_lat is not None and prev_lon is not None:
                # Only calculate distance if it's a different venue
                if current_venue_guid != prev_venue_guid:
                    distance = self._haversine_distance(prev_lat, prev_lon, lat, lon)
                    total_distance += distance
            
            # Update previous venue
            prev_venue_guid = current_venue_guid
            prev_lat = lat
            prev_lon = lon

        # Include distance to target game venue (if provided)
        if target_venue_guid and prev_venue_guid:
            target_lat, target_lon = self._get_venue_location(target_venue_guid)
            if target_lat is not None and target_lon is not None:
                if target_venue_guid != prev_venue_guid:
                    distance = self._haversine_distance(prev_lat, prev_lon, target_lat, target_lon)
                    total_distance += distance
                    games_with_coords += 1  # Count target venue
                    logger.debug(f"[TRAVEL] Added {distance:.1f} miles to target venue {target_venue_guid}")
        elif target_venue_guid and not prev_venue_guid:
            # No previous games, but we have target venue - travel is 0 (no prior reference point)
            logger.debug(f"[TRAVEL] Target venue provided but no prior games to calculate distance from")

        logger.debug(f"[TRAVEL] {team} last{n_days}: games={len(games)}, with_guid={games_with_guid}, with_coords={games_with_coords}, total_miles={total_distance:.1f}")
        return total_distance
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth.
        
        Returns distance in miles.
        """
        # Earth radius in miles
        R = 3958.8
        
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def weighted_avg(self, values: list, dates: list, reference_date: date) -> float:
        """
        Compute exponentially weighted average.
        
        Args:
            values: List of stat values
            dates: List of game dates (as strings 'YYYY-MM-DD')
            reference_date: The matchup date
            
        Returns:
            Weighted average with more recent games weighted higher
        
        Uses exponential decay: weight = exp(-lambda * days_ago)
        """
        if not values or not dates or len(values) != len(dates):
            return 0.0
        
        if not self.use_exponential_weighting:
            return self.avg(values)
        
        total_weighted = 0.0
        total_weight = 0.0
        
        for val, date_str in zip(values, dates):
            try:
                game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                days_ago = (reference_date - game_date).days
                
                if days_ago < 0:
                    continue  # Skip future dates
                
                weight = math.exp(-self.exponential_lambda * days_ago)
                total_weighted += val * weight
                total_weight += weight
            except (ValueError, TypeError):
                continue
        
        return total_weighted / total_weight if total_weight > 0 else 0.0
    
    def get_team_games_before_date(self, team: str, year: int, month: int, day: int, season: str) -> list:
        """
        Get all regular season games for a team before a given date.
        
        Returns:
            List of game documents sorted by date
        """
        # Use cache to avoid repeated iterations
        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = (team, season, date_str)
        
        if cache_key in self._team_games_cache:
            return self._team_games_cache[cache_key]
        
        target_date = date(year, month, day)
        games = []
        
        # If using lazy loading and games aren't preloaded, query from DB
        if self.games_home is None or self.games_away is None:
            if self.db is None:
                raise ValueError("DB connection required for lazy loading. Provide db parameter to StatHandlerV2.")
            
            # Query games for this team and season before the target date
            target_date_str = target_date.strftime('%Y-%m-%d')
            
            query = {
                'season': season,
                'date': {'$lt': target_date_str},
                'game_type': {'$nin': self._exclude_game_types},
                '$or': [
                    {'homeTeam.name': team},
                    {'awayTeam.name': team}
                ],
                'homeTeam.points': {'$exists': True, '$gt': 0},
                'awayTeam.points': {'$exists': True, '$gt': 0}
            }
            
            # Query all fields - we need stat fields from homeTeam/awayTeam which are nested
            # For predictions, we're only querying ~100-200 games, so this is fine
            db_games = self._games_repo.find(query, sort=[('date', 1)])
            
            # Log when query returns 0 games (only first occurrence per team/season/date to diagnose "no games" issue)
            if len(db_games) == 0:
                if not hasattr(self, '_debug_logged_queries'):
                    self._debug_logged_queries = set()
                cache_key = (team, season, target_date_str)
                if cache_key not in self._debug_logged_queries:
                    self._debug_logged_queries.add(cache_key)
                    # Check if games exist for this team/season (without date filter)
                    test_query = {
                        'season': season,
                        '$or': [{'homeTeam.name': team}, {'awayTeam.name': team}],
                        'homeTeam.points': {'$exists': True, '$gt': 0}
                    }
                    sample_games = self._games_repo.find(test_query, limit=3)
                    import logging
                    if sample_games:
                        sample_dates = [g.get('date') for g in sample_games]
                        logging.warning(
                            f"get_team_games_before_date: 0 games for {team} in {season} before {target_date_str}. "
                            f"Found {len(sample_games)} games in season (dates: {sample_dates})"
                        )
                    else:
                        logging.warning(
                            f"get_team_games_before_date: 0 games for {team} in {season}. "
                            f"No games found for this team/season combination."
                        )
            
            # Convert to format matching preloaded structure (add _id as string)
            # Validate that games have the necessary structure
            for game in db_games:
                game['_id'] = str(game['_id'])
                # Verify game has the required structure
                if 'homeTeam' in game and 'awayTeam' in game and 'date' in game:
                    games.append(game)
        else:
            # Use team index for O(log N) bisect lookup
            games = self._get_team_games_in_range(team, season, end_date_str=date_str)

        # Cache the result
        self._team_games_cache[cache_key] = games
        return games

    def _get_team_games_last_n_months(self, team: str, year: int, month: int, day: int, season: str, n_months: int) -> list:
        """
        Get team games in the last N months (for lazy loading compatibility).
        
        Returns list of games sorted by date.
        """
        if self.games_home is None or self.games_away is None:
            # Query from DB
            if self.db is None:
                raise ValueError("DB connection required for lazy loading.")
            
            from dateutil import relativedelta
            from datetime import datetime as dt
            target_date = dt(year, month, day).date()
            begin_date = target_date - relativedelta.relativedelta(months=n_months)
            
            query = {
                'season': season,
                'date': {'$gte': begin_date.strftime('%Y-%m-%d'), '$lt': target_date.strftime('%Y-%m-%d')},
                'game_type': {'$nin': self._exclude_game_types},
                '$or': [
                    {'homeTeam.name': team},
                    {'awayTeam.name': team}
                ]
            }
            
            # Query all fields - we need stat fields from homeTeam/awayTeam
            games = self._games_repo.find(query, sort=[('date', 1)])
            for game in games:
                game['_id'] = str(game['_id'])
            return games
        else:
            # Use team index for O(log N) bisect lookup with caching
            from dateutil import relativedelta as rd
            end_str = f"{year}-{month:02d}-{day:02d}"
            begin_dt = date(year, month, day) - rd.relativedelta(months=n_months)
            begin_str = begin_dt.strftime('%Y-%m-%d')
            cache_key = (team, season, end_str, f'months_{n_months}')
            if cache_key in self._team_time_period_cache:
                return self._team_time_period_cache[cache_key]
            games = self._get_team_games_in_range(team, season, begin_date_str=begin_str, end_date_str=end_str)
            self._team_time_period_cache[cache_key] = games
            return games

    def _get_team_games_last_n_days(self, team: str, year: int, month: int, day: int, season: str, n_days: int) -> list:
        """
        Get team games in the last N days (for lazy loading compatibility).

        Returns list of games sorted by date.
        """
        if self.games_home is None or self.games_away is None:
            # Query from DB
            if self.db is None:
                raise ValueError("DB connection required for lazy loading.")

            target_date = date(year, month, day)
            begin_date = target_date - timedelta(days=n_days)

            query = {
                'season': season,
                'date': {'$gte': begin_date.strftime('%Y-%m-%d'), '$lt': target_date.strftime('%Y-%m-%d')},
                'game_type': {'$nin': self._exclude_game_types},
                '$or': [
                    {'homeTeam.name': team},
                    {'awayTeam.name': team}
                ]
            }

            # Query all fields - we need stat fields from homeTeam/awayTeam
            games = self._games_repo.find(query, sort=[('date', 1)])
            for game in games:
                game['_id'] = str(game['_id'])
            return games
        else:
            # Use team index for O(log N) bisect lookup with caching
            end_str = f"{year}-{month:02d}-{day:02d}"
            begin_dt = date(year, month, day) - timedelta(days=n_days)
            begin_str = begin_dt.strftime('%Y-%m-%d')
            cache_key = (team, season, end_str, f'days_{n_days}')
            if cache_key in self._team_time_period_cache:
                return self._team_time_period_cache[cache_key]
            games = self._get_team_games_in_range(team, season, begin_date_str=begin_str, end_date_str=end_str)
            self._team_time_period_cache[cache_key] = games
            return games
    
    def get_schedule_context(self, team: str, year: int, month: int, day: int, season: str) -> dict:
        """
        Compute schedule-related features for a team.
        
        Returns:
            Dict with games_last_3_days, games_last_5_days, is_b2b
        """
        target_date = date(year, month, day)
        games = self.get_team_games_before_date(team, year, month, day, season)
        
        games_last_3_days = 0
        games_last_5_days = 0
        is_b2b = False
        
        # Check last few games
        recent_games = [g for g in games if 
                       (target_date - datetime.strptime(g['date'], '%Y-%m-%d').date()).days <= 5]
        
        if recent_games:
            last_game_date = datetime.strptime(recent_games[-1]['date'], '%Y-%m-%d').date()
            days_since_last = (target_date - last_game_date).days
            
            # Check for back-to-back
            if days_since_last == 1:
                is_b2b = True
            
            # Count games in last 3 and 5 days
            for game in recent_games:
                game_date = datetime.strptime(game['date'], '%Y-%m-%d').date()
                days_ago = (target_date - game_date).days
                if days_ago <= 3:
                    games_last_3_days += 1
                if days_ago <= 5:
                    games_last_5_days += 1
        
        return {
            'games_last_3_days': games_last_3_days,
            'games_last_5_days': games_last_5_days,
            'is_b2b': is_b2b
        }
    
    def get_games_played_so_far(self, team: str, year: int, month: int, day: int, season: str) -> int:
        """Get the number of regular season games played by a team before a given date."""
        games = self.get_team_games_before_date(team, year, month, day, season)
        return len(games)
    
    def getStatAvgDiffs(
        self,
        HOME: str,
        AWAY: str,
        season: str,
        year: int = None,
        month: int = None,
        day: int = None,
        point_regression: bool = False,
        include_absolute: bool = False,
        is_key_stat_func: callable = None,
        calc_weight_map: dict = None
    ):
        """
        DEPRECATED: This method is no longer used for feature calculation.
        All feature calculation goes through calculate_feature().
        Returns an empty list since self.statistics is always empty.

        Args:
            HOME: Home team name
            AWAY: Away team name
            season: NBA season string
            year, month, day: Game date
            point_regression: If True, return per-team features instead of diffs
            include_absolute: If True, include absolute values for key stats
            is_key_stat_func: Function to determine if a stat is a key stat
            calc_weight_map: DEPRECATED - no longer used

        Returns:
            Empty list (self.statistics is always empty in new architecture)
        """
        # In the new architecture, self.statistics is always empty.
        # All feature calculation goes through calculate_feature() instead.
        # Return empty list immediately since there are no stat tokens to process.
        if not self.statistics:
            return []

        # This code path is unreachable - self.statistics is always empty
        # Keeping the return for completeness
        return []
    
    def _get_stat(self, team: str, games: list, stat_name: str, reference_date: date, calc_weight: str = None) -> float:
        """
        Get a stat value for a team from a list of games.
        
        Args:
            team: Team name
            games: List of game documents
            stat_name: Stat name to compute
            reference_date: Reference date for weighted averages
            calc_weight: Optional 'raw' or 'avg'. If 'raw', aggregate first then calculate.
                        If 'avg' or None, calculate per-game then average.
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if not games:
            logger.debug(f"[FEAT] _get_stat: No games provided for {team}, stat '{stat_name}' -> 0.0")
            return 0.0
        
        # If calc_weight is 'raw', use aggregate-first approach
        if calc_weight == 'raw':
            computed_from_aggregate = self._compute_derived_stat_from_aggregate(stat_name, games, team)
            if computed_from_aggregate is not None:
                logger.debug(f"[FEAT] _get_stat: Stat '{stat_name}' computed from aggregated raw stats (raw calc_weight) for {team}")
                return computed_from_aggregate
            else:
                logger.debug(f"[FEAT] _get_stat: Could not compute '{stat_name}' from aggregate for {team}")
                return 0.0
        
        # Otherwise, use per-game then average approach (default)
        values = []
        dates = []
        found_count = 0
        
        for game in games:
            if game['homeTeam']['name'] == team:
                team_data = game['homeTeam']
            elif game['awayTeam']['name'] == team:
                team_data = game['awayTeam']
            else:
                continue
            
            # First try to get the stat directly from the DB
            if stat_name in team_data:
                value = team_data.get(stat_name)
                if value is not None:
                    values.append(value)
                    dates.append(game['date'])
                    found_count += 1
            else:
                # If not found, try to compute it from raw stats (derived stats)
                # Pass the full game document for 'wins' calculation
                computed_value = self._compute_derived_stat(stat_name, team_data, game)
                if computed_value is not None:
                    values.append(computed_value)
                    dates.append(game['date'])
                    found_count += 1
        
        if not values:
            # Try to compute from aggregated raw stats if this is a derived stat (fallback)
            computed_from_aggregate = self._compute_derived_stat_from_aggregate(stat_name, games, team)
            if computed_from_aggregate is not None:
                logger.debug(f"[FEAT] _get_stat: Stat '{stat_name}' computed from aggregated raw stats (fallback) for {team}")
                return computed_from_aggregate
            
            # Downgrade to debug to avoid noisy warnings for non-DB fields (e.g., 'b2b')
            logger.debug(f"[FEAT] _get_stat: Stat '{stat_name}' not found in DB for {team} (checked {len(games)} games, found in {found_count}). Sample dates: {[g.get('date') for g in games[:3]]}")
            return 0.0
        
        if found_count < len(games):
            logger.debug(f"[FEAT] _get_stat: Stat '{stat_name}' found in {found_count}/{len(games)} games for {team}")
        
        if self.use_exponential_weighting:
            return self.weighted_avg(values, dates, reference_date)
        else:
            return self.avg(values)
    
    def _compute_derived_stat(self, stat_name: str, team_data: dict, game: dict = None) -> float:
        """
        Compute a derived stat from raw stats in team_data for a single game.
        
        Args:
            stat_name: Name of the derived stat to compute
            team_data: Team data dictionary from a game document
            game: Optional full game document (needed for 'wins' calculation)
            
        Returns:
            Computed stat value, or None if it can't be computed
        """
        if stat_name == 'three_perc':
            three_made = team_data.get('three_made', 0)
            three_att = team_data.get('three_att', 0)
            if three_att > 0:
                return (three_made / three_att) * 100  # Convert to percentage
            return None
        
        elif stat_name == 'effective_fg_perc':
            fg_made = team_data.get('FG_made', 0)
            fg_att = team_data.get('FG_att', 0)
            three_made = team_data.get('three_made', 0)
            if fg_att > 0:
                return ((fg_made + 0.5 * three_made) / fg_att) * 100  # Convert to percentage
            return None
        
        elif stat_name == 'true_shooting_perc':
            points = team_data.get('points', 0)
            fg_att = team_data.get('FG_att', 0)
            ft_att = team_data.get('FT_att', 0)
            denominator = 2 * (fg_att + 0.44 * ft_att)
            if denominator > 0:
                return (points / denominator) * 100  # Convert to percentage
            return None
        
        elif stat_name == 'TO_metric':
            to = team_data.get('TO', 0)
            fg_att = team_data.get('FG_att', 0)
            ft_att = team_data.get('FT_att', 0)
            denominator = fg_att + 0.44 * ft_att + to
            if denominator > 0:
                return 100 * to / denominator
            return None
        
        elif stat_name == 'ast_to_ratio':
            assists = team_data.get('assists', 0)
            to = team_data.get('TO', 0)
            if to > 0:
                return assists / to
            return None
        
        elif stat_name == 'three_rate':
            three_att = team_data.get('three_att', 0)
            fg_att = team_data.get('FG_att', 0)
            if fg_att > 0:
                return float(three_att) / float(fg_att)
            return None

        elif stat_name == 'ft_rate':
            ft_att = team_data.get('FT_att', 0)
            fg_att = team_data.get('FG_att', 0)
            if fg_att > 0:
                return float(ft_att) / float(fg_att)
            return None

        elif stat_name == 'opp_effective_fg_perc' and game is not None:
            # Compute opponent eFG% from opponent raw stats for this single game
            team_name = team_data.get('name')
            if not team_name:
                return None
            if game.get('homeTeam', {}).get('name') == team_name:
                opp = game.get('awayTeam', {})
            else:
                opp = game.get('homeTeam', {})
            fg_made = opp.get('FG_made', 0)
            fg_att = opp.get('FG_att', 0)
            three_made = opp.get('three_made', 0)
            if fg_att > 0:
                return ((fg_made + 0.5 * three_made) / fg_att) * 100
            return None
        
        elif stat_name == 'wins' and game is not None:
            # Calculate if this team won the game
            # Need to know if team is home or away
            team_name = team_data.get('name')
            if team_name is None:
                return None
            
            is_home = game.get('homeTeam', {}).get('name') == team_name
            home_won = game.get('homeWon', False)
            
            # Team wins if: (is home and home won) OR (is away and home didn't win)
            if (is_home and home_won) or (not is_home and not home_won):
                return 1.0
            else:
                return 0.0
        
        # Return None if this is not a derived stat we know how to compute
        return None
    
    def _compute_derived_stat_from_aggregate(self, stat_name: str, games: list, team: str) -> float:
        """
        Compute a derived stat from aggregated raw stats across multiple games.
        This is used as a fallback when individual game values aren't available.
        Also handles advanced stats that need to be computed from aggregates (off_rtg, def_rtg, etc.).
        
        Args:
            stat_name: Name of the derived stat to compute
            games: List of game documents
            team: Team name
            
        Returns:
            Computed stat value, or None if it can't be computed
        """
        if not games:
            return None
        
        # Handle wins - count wins across games
        if stat_name == 'wins':
            wins = 0
            for game in games:
                is_home = game.get('homeTeam', {}).get('name') == team
                home_won = game.get('homeWon', False)
                if (is_home and home_won) or (not is_home and not home_won):
                    wins += 1
            return float(wins)
        
        # Handle attempt-rate stats (ratio of totals)
        if stat_name in ['three_rate', 'ft_rate']:
            total_three_att = 0.0
            total_fg_att = 0.0
            total_ft_att = 0.0
            for game in games:
                if game.get('homeTeam', {}).get('name') == team:
                    team_data = game.get('homeTeam', {})
                elif game.get('awayTeam', {}).get('name') == team:
                    team_data = game.get('awayTeam', {})
                else:
                    continue
                # NOTE: DB schema uses 'FG_att'/'FT_att'/'three_att'
                total_three_att += float(team_data.get('three_att', 0) or 0)
                total_fg_att += float(team_data.get('FG_att', 0) or 0)
                total_ft_att += float(team_data.get('FT_att', 0) or 0)

            if total_fg_att <= 0:
                return 0.0

            if stat_name == 'three_rate':
                return float(total_three_att) / float(total_fg_att)
            else:
                return float(total_ft_att) / float(total_fg_att)

        # Handle advanced stats that need full aggregation
        if stat_name in ['off_rtg', 'def_rtg', 'assists_ratio', 'TO_metric', 'ast_to_ratio', 'effective_fg_perc', 'true_shooting_perc', 'three_perc',
                         'opp_effective_fg_perc', 'opp_true_shooting_perc', 'opp_three_perc', 'opp_assists_ratio', 'opp_points']:
            # Aggregate team stats and opponent stats
            team_agg = defaultdict(float)
            team_against_agg = defaultdict(float)
            
            for game in games:
                if game['homeTeam']['name'] == team:
                    team_data = game['homeTeam']
                    opp_data = game['awayTeam']
                elif game['awayTeam']['name'] == team:
                    team_data = game['awayTeam']
                    opp_data = game['homeTeam']
                else:
                    continue
                
                # Aggregate team stats
                for key, value in team_data.items():
                    if isinstance(value, (int, float)):
                        team_agg[key] += value
                
                # Aggregate opponent stats (for defensive stats)
                for key, value in opp_data.items():
                    if isinstance(value, (int, float)):
                        team_against_agg[key] += value
            
            # Use existing advanced stats computation
            adv_stats = self._compute_advanced_stats(team_agg, team_against_agg, len(games))
            
            # Map stat names (handle aliases and opponent eFG)
            if stat_name == 'effective_fg_perc':
                return adv_stats.get('efg', None)
            if stat_name == 'assists_ratio':
                # Calculate assists_ratio: 100 * (assists / (((total_min / 5) * FG_made) - FG_made))
                # Note: total_min = 48 min per game unless "OT" is true, then 53 min
                assists = team_agg.get('assists', 0)
                fg_made = team_agg.get('FG_made', 0)
                if fg_made > 0:
                    # Calculate total_min across all games
                    total_min = 0
                    for game in games:
                        # Check if game has OT
                        is_ot = game.get('OT', False)
                        total_min += 53 if is_ot else 48
                    assists_ratio = 100 * (assists / (((total_min / 5) * fg_made) - fg_made)) if ((total_min / 5) * fg_made) - fg_made > 0 else 0
                    return assists_ratio
                return None
            if stat_name == 'opp_effective_fg_perc':
                opp_fg_att = team_against_agg.get('FG_att', 0)
                if opp_fg_att > 0:
                    opp_efg = ((team_against_agg.get('FG_made', 0) + 0.5 * team_against_agg.get('three_made', 0)) / opp_fg_att) * 100
                    return opp_efg
                return None
            
            # Compute opponent true shooting percentage (what opponents shot against this team)
            if stat_name == 'opp_true_shooting_perc':
                opp_pts = team_against_agg.get('points', 0)
                opp_fga = team_against_agg.get('FG_att', 0)
                opp_fta = team_against_agg.get('FT_att', 0)
                if opp_fga + 0.44 * opp_fta > 0:
                    opp_ts = (opp_pts / (2 * (opp_fga + 0.44 * opp_fta))) * 100
                    return opp_ts
                return None
            
            # Compute opponent three point percentage (what opponents shot from 3 against this team)
            if stat_name == 'opp_three_perc':
                opp_three_att = team_against_agg.get('three_att', 0)
                if opp_three_att > 0:
                    opp_three_pct = (team_against_agg.get('three_made', 0) / opp_three_att) * 100
                    return opp_three_pct
                return None
            
            # Compute opponent assists ratio (what opponents' assists ratio was against this team)
            if stat_name == 'opp_assists_ratio':
                # Calculate using same formula: 100 * (assists / (((total_min / 5) * FG_made) - FG_made))
                opp_ast = team_against_agg.get('assists', 0)
                opp_fg_made = team_against_agg.get('FG_made', 0)
                if opp_fg_made > 0:
                    # Calculate total_min across all games
                    total_min = 0
                    for game in games:
                        is_ot = game.get('OT', False)
                        total_min += 53 if is_ot else 48
                    opp_ast_ratio = 100 * (opp_ast / (((total_min / 5) * opp_fg_made) - opp_fg_made)) if ((total_min / 5) * opp_fg_made) - opp_fg_made > 0 else 0
                    return opp_ast_ratio
                return None
            
            # Compute TO_metric: 100 * TO / (FG_att + 0.44*FT_att + TO)
            if stat_name == 'TO_metric':
                to = team_agg.get('TO', 0)
                fga = team_agg.get('FG_att', 0)
                fta = team_agg.get('FT_att', 0)
                denominator = fga + 0.44 * fta + to
                if denominator > 0:
                    return 100 * to / denominator
                return None
            
            # Compute ast_to_ratio: assists/TO
            if stat_name == 'ast_to_ratio':
                assists = team_agg.get('assists', 0)
                to = team_agg.get('TO', 0)
                if to > 0:
                    return assists / to
                return None
            
            # Compute three_perc: (three_made / three_att) * 100
            if stat_name == 'three_perc':
                three_made = team_agg.get('three_made', 0)
                three_att = team_agg.get('three_att', 0)
                if three_att > 0:
                    return (three_made / three_att) * 100
                return None
            
            # Compute true_shooting_perc: (points / (2 * (FG_att + 0.44 * FT_att))) * 100
            if stat_name == 'true_shooting_perc':
                points = team_agg.get('points', 0)
                fg_att = team_agg.get('FG_att', 0)
                ft_att = team_agg.get('FT_att', 0)
                denominator = 2 * (fg_att + 0.44 * ft_att)
                if denominator > 0:
                    return (points / denominator) * 100
                return None
            
            # Compute opponent points (what opponents scored against this team)
            if stat_name == 'opp_points':
                opp_pts = team_against_agg.get('points', 0)
                if len(games) > 0:
                    # Return points per game
                    return opp_pts / len(games)
                return None
            if stat_name in adv_stats:
                return adv_stats.get(stat_name, None)
        
        # Aggregate raw stats for percentage calculations
        total_three_made = 0
        total_three_att = 0
        total_fg_made = 0
        total_fg_att = 0
        total_points = 0
        total_ft_att = 0
        
        for game in games:
            if game['homeTeam']['name'] == team:
                team_data = game['homeTeam']
            elif game['awayTeam']['name'] == team:
                team_data = game['awayTeam']
            else:
                continue
            
            total_three_made += team_data.get('three_made', 0)
            total_three_att += team_data.get('three_att', 0)
            total_fg_made += team_data.get('FG_made', 0)
            total_fg_att += team_data.get('FG_att', 0)
            total_points += team_data.get('points', 0)
            total_ft_att += team_data.get('FT_att', 0)
        
        if stat_name == 'three_perc':
            if total_three_att > 0:
                return (total_three_made / total_three_att) * 100
            return None
        
        elif stat_name == 'true_shooting_perc':
            denominator = 2 * (total_fg_att + 0.44 * total_ft_att)
            if denominator > 0:
                return (total_points / denominator) * 100
            return None
        
        return None
    
    def _get_all_stats_batch(self, team: str, games: list, stat_names: list, reference_date: date, is_side_stat: bool = False, window_key: tuple = None, calc_weight_map: dict = None) -> dict:
        """
        Batch compute multiple stats for a team by iterating through games only once.
        This is much faster than calling _get_stat multiple times.
        
        Args:
            team: Team name
            games: List of game documents
            stat_names: List of stat names to compute
            reference_date: Reference date for weighted averages
            is_side_stat: If True, compute side-specific stats (only count games where team is home/away)
            calc_weight_map: Optional dict mapping stat_name -> 'raw' or 'avg'. 
                            If 'raw', aggregate first then calculate (for complex stats like efg).
                            If 'avg' or not specified, calculate per-game then average.
            
        Returns:
            Dict mapping stat_name -> computed value
        """
        if not games or not stat_names:
            if games:  # Have games but no stat names requested
                return {stat: 0.0 for stat in stat_names}
            else:  # No games
                import logging
                logger = logging.getLogger(__name__)
                window_str = f" {window_key}" if window_key else ""
                logger.debug(f"[FEAT] _get_all_stats_batch: No games provided for {team} (window{window_str})")
                return {stat: 0.0 for stat in stat_names}
        
        if calc_weight_map is None:
            calc_weight_map = {}
        
        # Separate stats by calc_weight method
        raw_stats = [stat for stat in stat_names if calc_weight_map.get(stat) == 'raw']
        avg_stats = [stat for stat in stat_names if calc_weight_map.get(stat) != 'raw']
        
        # For 'raw' stats, compute directly from aggregate (skip per-game calculation)
        results = {}
        import logging
        logger = logging.getLogger(__name__)
        window_str = f" (window {window_key})" if window_key else ""
        
        for stat_name in raw_stats:
            # Compute from aggregated raw stats directly
            computed_from_aggregate = self._compute_derived_stat_from_aggregate(stat_name, games, team)
            if computed_from_aggregate is not None:
                results[stat_name] = computed_from_aggregate
                logger.debug(f"[FEAT] {stat_name}: Computed from aggregated raw stats (raw calc_weight) for {team}{window_str}")
            else:
                results[stat_name] = 0.0
                logger.debug(f"[FEAT] {stat_name}: Could not compute from aggregate for {team}{window_str}")
        
        # For 'avg' stats (or default), use per-game then average approach
        if avg_stats:
            # Initialize storage for each stat
            stat_values = {stat: [] for stat in avg_stats}
            stat_dates = {stat: [] for stat in avg_stats}
            stat_found_count = {stat: 0 for stat in avg_stats}  # Track how many games had this stat
            
            # Single pass through games to extract all stat values
            for game in games:
                is_home_team = game['homeTeam']['name'] == team
                is_away_team = game['awayTeam']['name'] == team
                
                # For side stats, filter to only games where team is home (when called for home stats)
                # or away (when called for away stats). The caller should pass filtered games, but
                # we'll be defensive and check here too.
                if is_side_stat:
                    # Side stats: only count games where team played at the specified side
                    # This depends on how the function is called - if it's for "home_side", only count home games
                    # The caller should already filter, but we verify the team is on the correct side
                    if not (is_home_team or is_away_team):
                        continue
                
                if is_home_team:
                    team_data = game['homeTeam']
                elif is_away_team:
                    team_data = game['awayTeam']
                else:
                    continue
                
                game_date = game['date']
                
                # Extract all stat values for this game
                for stat_name in avg_stats:
                    # First try to get the stat directly from the DB
                    if stat_name in team_data:
                        value = team_data.get(stat_name)
                        if value is not None:
                            stat_values[stat_name].append(value)
                            stat_dates[stat_name].append(game_date)
                            stat_found_count[stat_name] += 1
                    else:
                        # If not found, try to compute it from raw stats (derived stats)
                        # Pass the full game document for 'wins' calculation
                        computed_value = self._compute_derived_stat(stat_name, team_data, game)
                        if computed_value is not None:
                            stat_values[stat_name].append(computed_value)
                            stat_dates[stat_name].append(game_date)
                            stat_found_count[stat_name] += 1
            
            # Compute averages for all avg_stats
            for stat_name in avg_stats:
                values = stat_values[stat_name]
                dates = stat_dates[stat_name]
                found_count = stat_found_count[stat_name]
                
                if not values:
                    # Try to compute from aggregated raw stats if this is a derived stat (fallback)
                    computed_from_aggregate = self._compute_derived_stat_from_aggregate(stat_name, games, team)
                    if computed_from_aggregate is not None:
                        results[stat_name] = computed_from_aggregate
                        logger.debug(f"[FEAT] {stat_name}: Computed from aggregated raw stats (fallback) for {team}{window_str}")
                    else:
                        results[stat_name] = 0.0
                        # Downgrade missing-field notice to debug to avoid noisy logs for non-DB fields
                        if len(games) > 0 and found_count == 0:
                            logger.debug(f"[FEAT] {stat_name}: Field not found in DB for {team} ({found_count}/{len(games)} games){window_str}. All games checked: {[g.get('date') for g in games[:5]]}")
                        elif found_count < len(games):
                            logger.debug(f"[FEAT] {stat_name}: Found in {found_count}/{len(games)} games for {team}{window_str}")
                elif self.use_exponential_weighting:
                    results[stat_name] = self.weighted_avg(values, dates, reference_date)
                else:
                    results[stat_name] = self.avg(values)
        
        return results
    
    def _get_side_stat(self, team: str, games: list, stat_name: str, reference_date: date) -> float:
        """Get a side-specific stat (home/away) for a team."""
        if not games:
            return 0.0
        
        values = []
        dates = []
        
        for game in games:
            if game['homeTeam']['name'] == team:
                team_data = game['homeTeam']
            elif game['awayTeam']['name'] == team:
                team_data = game['awayTeam']
            else:
                continue
            
            # Get side-specific value
            side_value = team_data.get(stat_name, 0)
            if side_value is not None:
                values.append(side_value)
                dates.append(game['date'])
        
        if self.use_exponential_weighting:
            return self.weighted_avg(values, dates, reference_date)
        else:
            return self.avg(values)
    
    # All enhanced features are calculated directly via calculate_feature()
    
    def _calculate_pace(self, team: str, games: list) -> float:
        """Calculate average pace (possessions per game) for a team."""
        if not games:
            return 0.0
        
        total_possessions = 0.0
        for game in games:
            if game['homeTeam']['name'] == team:
                team_data = game['homeTeam']
            elif game['awayTeam']['name'] == team:
                team_data = game['awayTeam']
            else:
                continue
            
            # Simple pace calculation: FGA - OReb + TO + 0.44 * FTA
            fga = team_data.get('FG_att', 0)
            oreb = team_data.get('off_reb', 0)
            to = team_data.get('TO', 0)
            fta = team_data.get('FT_att', 0)
            
            possessions = fga - oreb + to + 0.44 * fta
            total_possessions += possessions
        
        return total_possessions / len(games) if games else 0.0
    
    def get_era_normalized_features(self, HOME: str, AWAY: str, season: str, year: int, month: int, day: int) -> dict:
        """
        Compute era-normalized features (relative to league average).
        
        Returns:
            Dict with home/away/diff features
        """
        if season not in self.league_averages:
            return {}
        
        league_avg = self.league_averages[season]
        
        # Get team season averages
        home_games = self.get_team_games_before_date(HOME, year, month, day, season)
        away_games = self.get_team_games_before_date(AWAY, year, month, day, season)
        
        reference_date = date(year, month, day)
        
        home_ppg = self._get_stat(HOME, home_games, 'points', reference_date)
        away_ppg = self._get_stat(AWAY, away_games, 'points', reference_date)
        
        home_off_rtg = self._get_stat(HOME, home_games, 'off_rtg', reference_date)
        away_off_rtg = self._get_stat(AWAY, away_games, 'off_rtg', reference_date)
        
        lg_ppg = league_avg.get('points', 100.0)
        lg_off_rtg = league_avg.get('off_rtg', 100.0)
        
        features = {
            'homePpgRel': home_ppg / lg_ppg if lg_ppg > 0 else 1.0,
            'awayPpgRel': away_ppg / lg_ppg if lg_ppg > 0 else 1.0,
            'homeOffRtgRel': home_off_rtg / lg_off_rtg if lg_off_rtg > 0 else 1.0,
            'awayOffRtgRel': away_off_rtg / lg_off_rtg if lg_off_rtg > 0 else 1.0,
            'ppgRelDiff': (home_ppg / lg_ppg if lg_ppg > 0 else 1.0) - (away_ppg / lg_ppg if lg_ppg > 0 else 1.0),
            'offRtgRelDiff': (home_off_rtg / lg_off_rtg if lg_off_rtg > 0 else 1.0) - (away_off_rtg / lg_off_rtg if lg_off_rtg > 0 else 1.0)
        }
        
        return features
    
    def _aggregate_games(self, games: list, team: str) -> tuple:
        """
        Aggregate stats from a list of games for a team.
        
        Returns:
            Tuple of (team_agg, team_against_agg, team_side_agg, team_side_against_agg, home_games, away_games)
        """
        team_agg = defaultdict(float)
        team_against_agg = defaultdict(float)
        team_side_agg = defaultdict(float)
        team_side_against_agg = defaultdict(float)
        home_games = []
        away_games = []
        
        for game in games:
            if game['homeTeam']['name'] == team:
                team_data = game['homeTeam']
                against_data = game['awayTeam']
                home_games.append(game)
            elif game['awayTeam']['name'] == team:
                team_data = game['awayTeam']
                against_data = game['homeTeam']
                away_games.append(game)
            else:
                continue
            
            # Aggregate team stats
            for key, value in team_data.items():
                if isinstance(value, (int, float)):
                    team_agg[key] += value
            
            # Aggregate against stats
            for key, value in against_data.items():
                if isinstance(value, (int, float)):
                    team_against_agg[key] += value
            
            # Aggregate side-specific stats
            if 'homeTeam' in game and game['homeTeam']['name'] == team:
                for key, value in team_data.items():
                    if isinstance(value, (int, float)) and '_side' in key:
                        team_side_agg[key] += value
                for key, value in against_data.items():
                    if isinstance(value, (int, float)) and '_side' in key:
                        team_side_against_agg[key] += value
        
        return team_agg, team_against_agg, team_side_agg, team_side_against_agg, home_games, away_games
    
    def _compute_advanced_stats(self, team_agg: dict, team_against_agg: dict, games_played: int) -> dict:
        """Compute advanced statistics from aggregated stats."""
        if games_played == 0:
            return {}
        
        # Calculate possessions
        fga = team_agg.get('FG_att', 0)
        oreb = team_agg.get('off_reb', 0)
        to = team_agg.get('TO', 0)
        fta = team_agg.get('FT_att', 0)
        possessions = fga - oreb + to + 0.44 * fta
        
        # Points per game
        ppg = team_agg.get('points', 0) / games_played if games_played > 0 else 0
        
        # Offensive rating
        off_rtg = (team_agg.get('points', 0) / possessions * 100) if possessions > 0 else 0
        
        # Defensive rating
        opp_possessions = team_against_agg.get('FG_att', 0) - team_against_agg.get('off_reb', 0) + team_against_agg.get('TO', 0) + 0.44 * team_against_agg.get('FT_att', 0)
        def_rtg = (team_against_agg.get('points', 0) / opp_possessions * 100) if opp_possessions > 0 else 0
        
        # Effective FG%
        fg_made = team_agg.get('FG_made', 0)
        efg = ((fg_made + 0.5 * team_agg.get('three_made', 0)) / fga * 100) if fga > 0 else 0
        
        # Assist ratio
        ast = team_agg.get('assists', 0)
        ast_ratio = (ast / (fga + 0.44 * fta + ast + to) * 100) if (fga + 0.44 * fta + ast + to) > 0 else 0
        
        # TO metric
        to_metric = (to / possessions * 100) if possessions > 0 else 0
        
        return {
            'ppg': ppg,
            'off_rtg': off_rtg,
            'def_rtg': def_rtg,
            'efg': efg,
            'ast_ratio': ast_ratio,
            'to_metric': to_metric
        }
    
    def _get_stat_diff(self, stat: str, home_games: list, away_games: list, 
                      home_agg: dict, home_against: dict, away_agg: dict, away_against: dict,
                      home_side_agg: dict, home_side_against: dict, away_side_agg: dict, away_side_against: dict,
                      home_side_games: int, away_side_games: int) -> float:
        """
        Calculate stat differential between home and away teams.
        """
        # Check if it's a known advanced stat
        if stat in ['off_rtg', 'def_rtg', 'efg', 'ast_ratio', 'to_metric']:
            home_adv = self._compute_advanced_stats(home_agg, home_against, len(home_games))
            away_adv = self._compute_advanced_stats(away_agg, away_against, len(away_games))
            
            home_val = home_adv.get(stat, 0)
            away_val = away_adv.get(stat, 0)
            
            # Calculate differential (home - away)
            return home_val - away_val
        
        # Check if it's a side-specific stat
        side_stat = '_side' in stat
        if side_stat:
            stat_clean = stat.replace('_side', '')
            
            # Get side-specific values
            home_side_val = home_side_agg.get(stat, 0) / home_side_games if home_side_games > 0 else 0
            away_side_val = away_side_agg.get(stat, 0) / away_side_games if away_side_games > 0 else 0
            
            # Get against values
            home_side_against_val = home_side_against.get(stat, 0) / home_side_games if home_side_games > 0 else 0
            away_side_against_val = away_side_against.get(stat, 0) / away_side_games if away_side_games > 0 else 0
            
            home_diff = home_side_val - home_side_against_val
            away_diff = away_side_val - away_side_against_val
            return home_diff - away_diff
        
        # Known stat in aggregation
        if side_stat:
            home_diff = (home_side_agg[stat] / home_side_games) - (home_side_against[stat] / home_side_games)
            away_diff = (away_side_agg[stat] / away_side_games) - (away_side_against[stat] / away_side_games)
        else:
            home_diff = (home_agg[stat] / len(home_games)) - (home_against[stat] / len(home_games))
            away_diff = (away_agg[stat] / len(away_games)) - (away_against[stat] / len(away_games))
        
        return home_diff - away_diff
    
    # =========================================================================
    # DIRECT FEATURE CALCULATION (NEW ARCHITECTURE)
    # =========================================================================
    
    def calculate_feature(
        self,
        feature_name: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        per_calculator=None,
        target_venue_guid: str = None
    ) -> float:
        """
        Calculate a single feature value directly from its name.

        This is the new direct architecture that eliminates stat tokens.
        Features are calculated directly based on their name components.

        Args:
            feature_name: Feature name  (e.g., 'points|season|avg|diff')
            home_team: Home team name
            away_team: Away team name
            season: NBA season string
            year, month, day: Game date
            per_calculator: Optional PERCalculator instance (for PER features)
            target_venue_guid: Venue GUID of the target game (for travel features)

        Returns:
            Feature value (float), or 0.0 if calculation fails
        """
        # Parse feature name
        components = parse_feature_name(feature_name)
        if not components:
            # Not a standard feature - might be special (elo, rest, etc.)
            return self._calculate_special_feature(feature_name, home_team, away_team, season, year, month, day, per_calculator)

        stat_name = components.stat_name
        time_period = components.time_period
        calc_weight = components.calc_weight
        perspective = components.home_away_diff  # 'home', 'away', or 'diff'
        is_side = components.is_side

        # ---------------------------------------------------------------------
        # Special features that parse but need special handling
        # ---------------------------------------------------------------------
        # Elo features use cached ratings from MongoDB
        if stat_name == 'elo':
            return self._calculate_special_feature(feature_name, home_team, away_team, season, year, month, day, per_calculator)

        # Rest features are handled specially
        if stat_name == 'rest':
            return self._calculate_special_feature(feature_name, home_team, away_team, season, year, month, day, per_calculator)

        # Vegas betting lines and derived features - pull from game document's 'vegas' field
        if stat_name in ('vegas_ML', 'vegas_spread', 'vegas_ou', 'vegas_implied_prob', 'vegas_edge'):
            return self._calculate_special_feature(feature_name, home_team, away_team, season, year, month, day, per_calculator)

        # ---------------------------------------------------------------------
        # Derived / matchup-scalar features
        # ---------------------------------------------------------------------
        # pace_interaction: harmonic mean of both teams' pace
        if stat_name == 'pace_interaction' or stat_name in ['pace_interaction_season', 'pace_interaction_games_10']:
            # Format: use time_period from parsed components
            
            tp = time_period if stat_name == 'pace_interaction' else ('season' if stat_name.endswith('_season') else 'games_10')
            return self._calculate_pace_interaction_feature(
                tp, perspective, home_team, away_team, season, year, month, day
            )

        # est_possessions: estimated possessions for the matchup
        if stat_name == 'est_possessions' or stat_name in ['est_possessions_season', 'est_possessions_games_10']:
            tp = time_period if stat_name == 'est_possessions' else ('season' if stat_name.endswith('_season') else 'games_10')
            return self._calculate_est_possessions_feature(
                tp, perspective, home_team, away_team, season, year, month, day
            )

        # exp_points_off, exp_points_def, exp_points_matchup
        if stat_name in ['exp_points_off', 'exp_points_def', 'exp_points_matchup'] or \
           stat_name.startswith('exp_points_off_') or stat_name.startswith('exp_points_def_') or stat_name.startswith('exp_points_matchup_'):
            # Format: use stat_name and time_period separately
            
            if stat_name in ['exp_points_off', 'exp_points_def', 'exp_points_matchup']:
                base_stat = stat_name
                tp = time_period
            else:
                
                if '_season' in stat_name:
                    base_stat = stat_name.replace('_season', '')
                    tp = 'season'
                else:
                    base_stat = stat_name.replace('_games_10', '')
                    tp = 'games_10'
            return self._calculate_exp_points_feature(
                base_stat, tp, perspective, home_team, away_team, season, year, month, day
            )

        # Handle PER features
        if stat_name.startswith('player_'):
            return self._calculate_per_feature(feature_name, home_team, away_team, season, year, month, day, per_calculator)

        # Handle composite time periods (blend, delta, blend-delta)
        if time_period.startswith('blend:') or time_period.startswith('delta:'):
            return self._calculate_composite_time_period_feature(
                stat_name, time_period, calc_weight, perspective, is_side,
                home_team, away_team, season, year, month, day, per_calculator
            )

        # Shot-mix attempt rates (explicitly defined as ratio-of-totals with location splits)
        # For std, fall through to the generic path to get per-game values
        if stat_name in ['three_rate', 'ft_rate'] and calc_weight != 'std':
            return self._calculate_attempt_rate_feature(
                stat_name, time_period, perspective, home_team, away_team, season, year, month, day
            )

        # ---------------------------------------------------------------------
        # GB-4 Matchup & Chemistry features
        # ---------------------------------------------------------------------
        # Margin features (point differential per game)
        if stat_name == 'margin':
            return self._calculate_margin_feature(
                time_period, calc_weight, perspective, is_side,
                home_team, away_team, season, year, month, day
            )

        # H2H win percentage feature
        if stat_name == 'h2h_win_pct':
            return self._calculate_h2h_win_pct_feature(
                time_period, calc_weight, perspective, is_side, home_team, away_team, season, year, month, day
            )

        # H2H margin feature (point margin in H2H games)
        if stat_name == 'margin_h2h':
            return self._calculate_margin_h2h_feature(
                time_period, calc_weight, perspective, is_side, home_team, away_team, season, year, month, day
            )

        # H2H games count (sample size indicator)
        if stat_name == 'h2h_games_count':
            return self._calculate_h2h_games_count_feature(
                time_period, is_side, home_team, away_team, season, year, month, day
            )

        # Conference win percentage (league-specific: requires conference field on teams)
        if stat_name == 'conf_wins':
            return self._calculate_conf_wins_feature(
                time_period, calc_weight, perspective,
                home_team, away_team, season, year, month, day
            )

        # Same conference binary (league-specific)
        if stat_name == 'same_conf':
            home_conf = self._get_team_conference(home_team)
            away_conf = self._get_team_conference(away_team)
            return 1.0 if (home_conf and away_conf and home_conf == away_conf) else 0.0

        # Conference average margin (league-specific)
        if stat_name == 'conf_margin':
            return self._calculate_conf_margin_feature(
                perspective, home_team, away_team, season, year, month, day
            )

        # Conference games played count (league-specific)
        if stat_name == 'conf_gp':
            return self._calculate_conf_gp_feature(
                perspective, home_team, away_team, season, year, month, day
            )

        # Postseason binary indicator
        if stat_name == 'postseason':
            return self._calculate_postseason_feature(home_team, away_team, year, month, day)

        # Close game win percentage
        if stat_name == 'close_win_pct':
            return self._calculate_close_win_pct_feature(
                time_period, perspective, home_team, away_team, season, year, month, day
            )

        # Handle enhanced features (pace, travel, b2b, first_of_b2b, games_played, road_games, days_rest)
        if stat_name in ['pace', 'travel', 'b2b', 'first_of_b2b', 'games_played', 'road_games', 'days_rest']:
            return self._calculate_enhanced_feature(feature_name, home_team, away_team, season, year, month, day, target_venue_guid)
        
        # Handle net features (stat_name ends with '_net')
        is_net = stat_name.endswith('_net')
        if is_net:
            base_stat_name = stat_name[:-4]  # Remove '_net'
            return self._calculate_net_feature(
                base_stat_name, time_period, calc_weight, perspective, is_side,
                home_team, away_team, season, year, month, day
            )
        
        # Handle regular stats (basic and rate)
        return self._calculate_regular_feature(
            stat_name, time_period, calc_weight, perspective, is_side,
            home_team, away_team, season, year, month, day
        )
    
    def _calculate_regular_feature(
        self,
        stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate a regular stat feature (basic or rate stat).

        Args:
            stat_name: Base stat name (e.g., 'points', 'efg', 'off_rtg')
            time_period: Time period (e.g., 'season', 'months_1', 'games_10', 'h2h_last10')
            calc_weight: Calculation method ('raw', 'avg', or 'std')
            perspective: 'home', 'away', or 'diff'
            is_side: Whether this is a side-split feature
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Feature value
        """
        reference_date = date(year, month, day)

        # Handle H2H time periods specially - both teams' stats come from the same H2H games
        if time_period.startswith('h2h_last'):
            n_games = int(time_period.replace('h2h_last', ''))
            h2h_games = self._get_h2h_games(home_team, away_team, season, year, month, day, n_games)

            if not h2h_games:
                return 0.0

            # For H2H, both teams use the same set of games
            # is_side filtering: only include games where team was home/away as specified
            if is_side:
                home_h2h_games = [g for g in h2h_games if g.get('homeTeam', {}).get('name') == home_team]
                away_h2h_games = [g for g in h2h_games if g.get('awayTeam', {}).get('name') == away_team]
            else:
                home_h2h_games = h2h_games
                away_h2h_games = h2h_games

            # Calculate stat for home team from H2H games
            home_value = self._calculate_team_stat(
                stat_name, home_team, home_h2h_games, calc_weight, reference_date
            )

            # Calculate stat for away team from H2H games
            away_value = self._calculate_team_stat(
                stat_name, away_team, away_h2h_games, calc_weight, reference_date
            )

            # Return based on perspective
            if perspective == 'home':
                return home_value
            elif perspective == 'away':
                return away_value
            else:  # 'diff'
                return home_value - away_value

        # Standard time periods - get games for each team independently
        home_games = self._get_games_for_time_period(home_team, season, year, month, day, time_period)
        away_games = self._get_games_for_time_period(away_team, season, year, month, day, time_period)

        # Filter to side-specific games if needed
        if is_side:
            home_games = [g for g in home_games if g.get('homeTeam', {}).get('name') == home_team]
            away_games = [g for g in away_games if g.get('awayTeam', {}).get('name') == away_team]

        # Calculate stat for home team
        home_value = self._calculate_team_stat(
            stat_name, home_team, home_games, calc_weight, reference_date
        )

        # Calculate stat for away team
        away_value = self._calculate_team_stat(
            stat_name, away_team, away_games, calc_weight, reference_date
        )

        # Return based on perspective
        if perspective == 'home':
            return home_value
        elif perspective == 'away':
            return away_value
        else:  # 'diff'
            return home_value - away_value
    
    def _calculate_team_stat(
        self,
        stat_name: str,
        team: str,
        games: list,
        calc_weight: str,
        reference_date: date
    ) -> float:
        """
        Calculate a stat value for a team from a list of games.
        
        Args:
            stat_name: Stat name (e.g., 'points', 'efg', 'off_rtg')
            team: Team name
            games: List of game documents
            calc_weight: 'raw', 'avg', or 'std'
            reference_date: Reference date for weighted averages
            
        Returns:
            Stat value
        """
        if not games:
            return 0.0
        
        # Map stat names to internal names using FeatureRegistry (SSoT)
        internal_stat_name = FeatureRegistry.get_db_field(stat_name)

        # Special handling for wins (count wins, not average)
        # Note: Check original stat_name, not internal_stat_name (which is 'homeWon' for wins)
        if stat_name == 'wins':
            wins = 0
            for game in games:
                is_home = game.get('homeTeam', {}).get('name') == team
                home_won = game.get('homeWon', False)
                if (is_home and home_won) or (not is_home and not home_won):
                    wins += 1
            if calc_weight == 'raw':
                return float(wins)
            else:  # 'avg' - wins per game (win percentage)
                return float(wins) / len(games) if games else 0.0
        
        # Determine if this is a rate/derived stat that needs per-game computation
        # Check by canonical name, not internal/DB name
        is_rate_stat = stat_name in FeatureRegistry.get_rate_stats()
        # Also check opponent stats which require aggregation
        opponent_stats = {
            'opp_effective_fg_perc', 'opp_true_shooting_perc', 'opp_three_perc',
            'opp_assists_ratio', 'opp_points'
        }
        # Derived stats that have per-game compute handlers in _compute_derived_stat
        per_game_derived_stats = {'three_rate', 'ft_rate'}
        is_rate_stat = is_rate_stat or internal_stat_name in opponent_stats or stat_name in per_game_derived_stats
        
        # Calculate based on calc_weight
        if calc_weight == 'raw':
            if is_rate_stat:
                # For rate stats with 'raw', aggregate first then calculate
                return self._compute_derived_stat_from_aggregate(internal_stat_name, games, team) or 0.0
            else:
                # For basic stats with 'raw', sum totals
                total = 0.0
                for game in games:
                    if game['homeTeam']['name'] == team:
                        team_data = game['homeTeam']
                    elif game['awayTeam']['name'] == team:
                        team_data = game['awayTeam']
                    else:
                        continue
                    value = team_data.get(internal_stat_name, 0)
                    if value is not None:
                        total += value
                return total
        else:  # calc_weight in ('avg', 'std')
            if is_rate_stat:
                # For rate stats with 'avg'/'std', calculate per-game then aggregate
                # BUT: Some rate stats (off_rtg, def_rtg, assists_ratio) require aggregation
                # and cannot be calculated per-game. For these, we aggregate first then calculate.
                # Opponent stats also require aggregation since they need team_against_agg.
                aggregation_required_stats = {
                    'off_rtg', 'def_rtg', 'assists_ratio',
                    'opp_effective_fg_perc', 'opp_true_shooting_perc', 'opp_three_perc',
                    'opp_assists_ratio', 'opp_points'
                }
                if internal_stat_name in aggregation_required_stats:
                    if calc_weight == 'std':
                        # No per-game values available for aggregation-required stats
                        return 0.0
                    # These stats require aggregation - calculate from aggregate
                    computed = self._compute_derived_stat_from_aggregate(internal_stat_name, games, team)
                    return computed if computed is not None else 0.0

                # For other rate stats, calculate per-game then avg/std
                values = []
                dates = []
                for game in games:
                    if game['homeTeam']['name'] == team:
                        team_data = game['homeTeam']
                    elif game['awayTeam']['name'] == team:
                        team_data = game['awayTeam']
                    else:
                        continue
                    computed = self._compute_derived_stat(internal_stat_name, team_data, game)
                    if computed is not None:
                        values.append(computed)
                        dates.append(game['date'])
                if values:
                    if calc_weight == 'std':
                        return self.std(values)
                    if self.use_exponential_weighting:
                        return self.weighted_avg(values, dates, reference_date)
                    else:
                        return self.avg(values)
                return 0.0
            else:
                # For basic stats with 'avg'/'std', extract per-game values then aggregate
                values = []
                dates = []
                for game in games:
                    if game['homeTeam']['name'] == team:
                        team_data = game['homeTeam']
                    elif game['awayTeam']['name'] == team:
                        team_data = game['awayTeam']
                    else:
                        continue
                    value = team_data.get(internal_stat_name, 0)
                    if value is not None:
                        values.append(value)
                        dates.append(game['date'])
                if values:
                    if calc_weight == 'std':
                        return self.std(values)
                    if self.use_exponential_weighting:
                        return self.weighted_avg(values, dates, reference_date)
                    else:
                        return self.avg(values)
                return 0.0
    
    def _get_games_for_time_period(
        self,
        team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        time_period: str
    ) -> list:
        """
        Get games for a team based on time period.
        
        Args:
            team: Team name
            season: Season string
            year, month, day: Game date
            time_period: 'season', 'months_N', 'games_N', 'days_N', or 'none'
            
        Returns:
            List of game documents
        """
        if time_period == 'season':
            return self.get_team_games_before_date(team, year, month, day, season)
        elif time_period.startswith('months_'):
            n_months = int(time_period.replace('months_', ''))
            return self._get_team_games_last_n_months(team, year, month, day, season, n_months)
        elif time_period.startswith('games_close'):
            # Special time period for close game stats (e.g., 'games_close5')
            # These are handled by dedicated calculators that filter for close games
            # Return empty list here since this is a special case
            return []
        elif time_period.startswith('games_'):
            n_games = int(time_period.replace('games_', ''))
            all_games = self.get_team_games_before_date(team, year, month, day, season)
            return all_games[-n_games:] if len(all_games) >= n_games else all_games
        elif time_period.startswith('days_'):
            n_days = int(time_period.replace('days_', ''))
            return self._get_team_games_last_n_days(team, year, month, day, season, n_days)
        else:  # 'none' or unknown
            return []
    
    def _calculate_net_feature(
        self,
        base_stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate a net feature (team_stat - team_allowed_stat).

        Args:
            base_stat_name: Base stat name without '_net' (e.g., 'points', 'efg')
            time_period, calc_weight, perspective, is_side: Feature components
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Net feature value
        """
        reference_date = date(year, month, day)

        # Map base stat to team stat and opponent defensive stat
        net_stat_map = {
            'efg': ('effective_fg_perc', 'opp_effective_fg_perc'),
            'ts': ('true_shooting_perc', 'opp_true_shooting_perc'),
            'three_pct': ('three_perc', 'opp_three_perc'),
            'off_rtg': ('off_rtg', 'def_rtg'),  # team's off_rtg - opponent's def_rtg
            'def_rtg': ('def_rtg', 'off_rtg'),  # team's def_rtg - opponent's off_rtg
            'assists_ratio': ('assists_ratio', 'opp_assists_ratio'),
            'points': ('points', 'opp_points'),
        }

        if base_stat_name not in net_stat_map:
            return 0.0

        team_stat_name, opp_def_stat_name = net_stat_map[base_stat_name]

        # Handle H2H time periods specially - both teams' stats come from the same H2H games
        if time_period.startswith('h2h_last'):
            n_games = int(time_period.replace('h2h_last', ''))
            h2h_games = self._get_h2h_games(home_team, away_team, season, year, month, day, n_games)

            if not h2h_games:
                return 0.0

            # For H2H, both teams use the same set of games
            if is_side:
                home_h2h_games = [g for g in h2h_games if g.get('homeTeam', {}).get('name') == home_team]
                away_h2h_games = [g for g in h2h_games if g.get('awayTeam', {}).get('name') == away_team]
            else:
                home_h2h_games = h2h_games
                away_h2h_games = h2h_games

            # Calculate home team net from H2H games
            home_team_stat = self._calculate_team_stat(team_stat_name, home_team, home_h2h_games, calc_weight, reference_date)
            home_opp_stat = self._calculate_team_stat(opp_def_stat_name, home_team, home_h2h_games, calc_weight, reference_date)
            home_net = home_team_stat - home_opp_stat

            # Calculate away team net from H2H games
            away_team_stat = self._calculate_team_stat(team_stat_name, away_team, away_h2h_games, calc_weight, reference_date)
            away_opp_stat = self._calculate_team_stat(opp_def_stat_name, away_team, away_h2h_games, calc_weight, reference_date)
            away_net = away_team_stat - away_opp_stat

            # Return based on perspective
            if perspective == 'home':
                return home_net
            elif perspective == 'away':
                return away_net
            else:  # 'diff'
                return home_net - away_net

        # Standard time periods - get games for each team independently
        home_games = self._get_games_for_time_period(home_team, season, year, month, day, time_period)
        away_games = self._get_games_for_time_period(away_team, season, year, month, day, time_period)

        # Filter to side-specific games if needed
        if is_side:
            home_games = [g for g in home_games if g.get('homeTeam', {}).get('name') == home_team]
            away_games = [g for g in away_games if g.get('awayTeam', {}).get('name') == away_team]

        # Calculate home team net
        home_team_stat = self._calculate_team_stat(team_stat_name, home_team, home_games, calc_weight, reference_date)
        home_opp_stat = self._calculate_team_stat(opp_def_stat_name, home_team, home_games, calc_weight, reference_date)
        home_net = home_team_stat - home_opp_stat

        # Calculate away team net
        away_team_stat = self._calculate_team_stat(team_stat_name, away_team, away_games, calc_weight, reference_date)
        away_opp_stat = self._calculate_team_stat(opp_def_stat_name, away_team, away_games, calc_weight, reference_date)
        away_net = away_team_stat - away_opp_stat

        # Return based on perspective
        if perspective == 'home':
            return home_net
        elif perspective == 'away':
            return away_net
        else:  # 'diff'
            return home_net - away_net
    
    def _calculate_composite_time_period_feature(
        self,
        stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        per_calculator=None
    ) -> float:
        """
        Calculate a feature with a composite time period (blend, delta, or blend-delta).

        Dispatches to _compute_blend for blend time periods, or recursively
        calls calculate_feature for delta components.

        Args:
            stat_name: Stat name (e.g., 'margin', 'wins', 'efg_net')
            time_period: Composite time period (e.g., 'blend:games_5:0.70/games_10:0.30',
                        'delta:games_5-season', 'delta:blend:games_5:0.70/games_10:0.30-season')
            calc_weight, perspective, is_side: Feature components
            home_team, away_team, season, year, month, day: Game context
            per_calculator: Optional PERCalculator instance

        Returns:
            Feature value (float)
        """
        if time_period.startswith('blend:') and not time_period.startswith('delta:'):
            # Pure blend: weighted combination of time periods
            parsed = FeatureRegistry.parse_blend_time_period(time_period)
            if parsed is None:
                return 0.0
            return self._compute_blend(
                stat_name, parsed['components'], calc_weight, perspective, is_side,
                home_team, away_team, season, year, month, day, per_calculator
            )

        if time_period.startswith('delta:'):
            # Delta or blend-delta
            parsed = FeatureRegistry.parse_delta_time_period(time_period)
            if parsed is None:
                return 0.0

            side_suffix = '|side' if is_side else ''

            if parsed['type'] == 'blend_delta':
                # Recent is a blend — reconstruct blend time_period string for recursion
                recent_tp = 'blend:' + '/'.join(
                    f"{tp}:{w}" for tp, w in parsed['recent']['components']
                )
            else:
                recent_tp = parsed['recent']

            baseline_tp = parsed['baseline']

            # Recursively calculate recent and baseline features
            recent_feature = f"{stat_name}|{recent_tp}|{calc_weight}|{perspective}{side_suffix}"
            baseline_feature = f"{stat_name}|{baseline_tp}|{calc_weight}|{perspective}{side_suffix}"

            recent_value = self.calculate_feature(
                recent_feature, home_team, away_team, season, year, month, day, per_calculator
            )
            baseline_value = self.calculate_feature(
                baseline_feature, home_team, away_team, season, year, month, day, per_calculator
            )

            return recent_value - baseline_value

        return 0.0

    def _compute_blend(
        self,
        stat_name: str,
        components: list,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        per_calculator=None
    ) -> float:
        """
        Compute a blended value from weighted time period components.

        Args:
            stat_name: Stat name (e.g., 'margin', 'wins')
            components: List of (time_period, weight) tuples
            calc_weight, perspective, is_side: Feature components
            home_team, away_team, season, year, month, day: Game context
            per_calculator: Optional PERCalculator instance

        Returns:
            Weighted blend value
        """
        side_suffix = '|side' if is_side else ''

        # Normalize weights if they don't sum to 1.0
        total_weight = sum(w for _, w in components)
        if total_weight <= 0:
            return 0.0

        blend_value = 0.0
        for tp, weight in components:
            component_feature = f"{stat_name}|{tp}|{calc_weight}|{perspective}{side_suffix}"
            component_value = self.calculate_feature(
                component_feature, home_team, away_team, season, year, month, day, per_calculator
            )
            blend_value += (weight / total_weight) * component_value

        return blend_value
    
    def _calculate_enhanced_feature(
        self,
        feature_name: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        target_venue_guid: str = None
    ) -> float:
        """
        Calculate an enhanced feature directly from its name.

        Args:
            feature_name: Full feature name  (e.g., 'pace|season|avg|diff')
            home_team, away_team, season, year, month, day: Game context
            target_venue_guid: Venue GUID of the target game (for travel features)
            
        Returns:
            Feature value
        """
        components = parse_feature_name(feature_name)
        if not components:
            return 0.0
        
        stat_name = components.stat_name
        time_period = components.time_period
        calc_weight = components.calc_weight
        perspective = components.home_away_diff
        reference_date = date(year, month, day)
        
        # Calculate games_played features
        if stat_name == 'games_played':
            if time_period == 'season':
                home_games = self.get_team_games_before_date(home_team, year, month, day, season)
                away_games = self.get_team_games_before_date(away_team, year, month, day, season)
                home_value = float(len(home_games))
                away_value = float(len(away_games))
            elif time_period.startswith('days_'):
                n_days = int(time_period.replace('days_', ''))
                home_games = self._get_team_games_last_n_days(home_team, year, month, day, season, n_days)
                away_games = self._get_team_games_last_n_days(away_team, year, month, day, season, n_days)
                home_value = float(len(home_games))
                away_value = float(len(away_games))
            else:
                return 0.0
            
            if perspective == 'home':
                return home_value
            elif perspective == 'away':
                return away_value
            else:  # 'diff'
                return home_value - away_value
        
        # Calculate road_games features (count of away games in period)
        elif stat_name == 'road_games':
            if time_period == 'season':
                home_games = self.get_team_games_before_date(home_team, year, month, day, season)
                away_games = self.get_team_games_before_date(away_team, year, month, day, season)
            elif time_period.startswith('days_'):
                n_days = int(time_period.replace('days_', ''))
                home_games = self._get_team_games_last_n_days(home_team, year, month, day, season, n_days)
                away_games = self._get_team_games_last_n_days(away_team, year, month, day, season, n_days)
            else:
                return 0.0

            # Count games where the team was the away team
            home_value = float(sum(1 for g in home_games if g.get('awayTeam', {}).get('name') == home_team))
            away_value = float(sum(1 for g in away_games if g.get('awayTeam', {}).get('name') == away_team))

            if perspective == 'home':
                return home_value
            elif perspective == 'away':
                return away_value
            else:  # 'diff'
                return home_value - away_value

        # Calculate pace features
        elif stat_name == 'pace':
            # Support season and games_N windows (broadest useful set).
            if time_period == 'season' or time_period.startswith('games_'):
                home_games = self._get_games_for_time_period(home_team, season, year, month, day, time_period)
                away_games = self._get_games_for_time_period(away_team, season, year, month, day, time_period)
            else:
                return 0.0
            
            home_pace = self._calculate_pace(home_team, home_games)
            away_pace = self._calculate_pace(away_team, away_games)
            
            if perspective == 'home':
                return home_pace
            elif perspective == 'away':
                return away_pace
            else:  # 'diff'
                return home_pace - away_pace
        
        # Calculate days_rest (days since last game) features
        elif stat_name == 'days_rest':
            home_rest = float(self._get_days_rest_capped(home_team, year, month, day, season))
            away_rest = float(self._get_days_rest_capped(away_team, year, month, day, season))

            if perspective == 'home':
                return home_rest
            elif perspective == 'away':
                return away_rest
            else:
                return home_rest - away_rest

        # Calculate b2b (back-to-back) features
        elif stat_name == 'b2b':
            # Recommended convention: b2b if played previous calendar day (days_rest == 1)
            home_rest = self._get_days_rest_capped(home_team, year, month, day, season)
            away_rest = self._get_days_rest_capped(away_team, year, month, day, season)

            home_b2b = 1.0 if home_rest == 1 else 0.0
            away_b2b = 1.0 if away_rest == 1 else 0.0

            if perspective == 'home':
                return home_b2b
            elif perspective == 'away':
                return away_b2b
            else:  # 'diff'
                return home_b2b - away_b2b

        # Calculate first_of_b2b (team plays again tomorrow)
        elif stat_name == 'first_of_b2b':
            home_first_b2b = 1.0 if self._team_plays_tomorrow(home_team, year, month, day, season) else 0.0
            away_first_b2b = 1.0 if self._team_plays_tomorrow(away_team, year, month, day, season) else 0.0

            if perspective == 'home':
                return home_first_b2b
            elif perspective == 'away':
                return away_first_b2b
            else:  # 'diff'
                return home_first_b2b - away_first_b2b

        # Calculate travel features
        elif stat_name == 'travel':
            if time_period.startswith('days_'):
                n_days = int(time_period.replace('days_', ''))
                home_travel = self._calculate_travel_distance(home_team, year, month, day, season, n_days, target_venue_guid)
                away_travel = self._calculate_travel_distance(away_team, year, month, day, season, n_days, target_venue_guid)

                if perspective == 'home':
                    return home_travel
                elif perspective == 'away':
                    return away_travel
                else:  # 'diff'
                    return home_travel - away_travel
        
        return 0.0

    # =========================================================================
    # NEW DERIVED FEATURES (PACE INTERACTION / POSSESSIONS / EXP POINTS / REST)
    # =========================================================================

    def _get_days_rest_capped(self, team: str, year: int, month: int, day: int, season: str, cap: int = 7) -> int:
        """
        Days since last game for team before (year, month, day), capped.
        If no prior game exists, returns cap (default 7).
        """
        target_date = date(year, month, day)
        games = self.get_team_games_before_date(team, year, month, day, season)

        last_game_date = None
        # games are usually in chronological order, but be defensive
        for g in sorted(games, key=lambda x: x.get('date', ''), reverse=True):
            try:
                g_date = datetime.strptime(g.get('date', ''), '%Y-%m-%d').date()
            except Exception:
                continue
            if g_date < target_date:
                last_game_date = g_date
                break

        if last_game_date is None:
            return cap

        days_rest = (target_date - last_game_date).days
        if days_rest < 0:
            days_rest = 0
        return min(int(days_rest), int(cap))

    def _team_plays_tomorrow(self, team: str, year: int, month: int, day: int, season: str) -> bool:
        """
        Check if a team plays the day after the given date (first of back-to-back).
        Excludes preseason/allstar games based on league config.

        Returns:
            True if team has a game tomorrow, False otherwise
        """
        from datetime import timedelta
        target_date = date(year, month, day)
        tomorrow = target_date + timedelta(days=1)
        tomorrow_str = tomorrow.strftime('%Y-%m-%d')

        # Check if using lazy loading (query DB) or preloaded data
        if self.games_home is None or self.games_away is None:
            if self.db is None:
                return False

            # Query for a game tomorrow involving this team
            query = {
                'season': season,
                'date': tomorrow_str,
                'game_type': {'$nin': self._exclude_game_types},
                '$or': [
                    {'homeTeam.name': team},
                    {'awayTeam.name': team}
                ]
            }
            games_tomorrow = self._games_repo.find(query, limit=1)
            return len(games_tomorrow) > 0
        else:
            # Check preloaded data for tomorrow
            if season in self.games_home:
                if tomorrow_str in self.games_home[season]:
                    for team_name, game in self.games_home[season][tomorrow_str].items():
                        if team_name == team:
                            game_type = game.get('game_type', 'regseason')
                            if game_type not in self._exclude_game_types:
                                return True

            if season in self.games_away:
                if tomorrow_str in self.games_away[season]:
                    for team_name, game in self.games_away[season][tomorrow_str].items():
                        if team_name == team:
                            game_type = game.get('game_type', 'regseason')
                            if game_type not in self._exclude_game_types:
                                return True

            return False

    def _get_league_pace_baseline(self, season: str) -> float:
        """
        Fallback baseline used when both home/away pace are missing/0.
        Computes a season-level average pace across team-games (home and away),
        cached per season. Falls back to 100.0 if DB/data is unavailable.
        """
        if not hasattr(self, '_league_pace_baseline_cache'):
            self._league_pace_baseline_cache = {}
        if season in self._league_pace_baseline_cache:
            return self._league_pace_baseline_cache[season]

        baseline = 100.0
        try:
            games = None
            if hasattr(self, 'all_games') and self.all_games:
                games = [g for g in self.all_games if g.get('season') == season]
            elif self._games_repo is not None:
                query = {'season': season, 'game_type': {'$nin': self._exclude_game_types}}
                projection = {
                    'homeTeam.name': 1, 'awayTeam.name': 1,
                    'homeTeam.FG_att': 1, 'homeTeam.off_reb': 1, 'homeTeam.TO': 1, 'homeTeam.FT_att': 1,
                    'awayTeam.FG_att': 1, 'awayTeam.off_reb': 1, 'awayTeam.TO': 1, 'awayTeam.FT_att': 1,
                }
                games = self._games_repo.find(query, projection=projection)
            if games:
                total = 0.0
                count = 0
                for g in games:
                    for side in ['homeTeam', 'awayTeam']:
                        td = g.get(side, {}) or {}
                        fga = float(td.get('FG_att', 0) or 0)
                        oreb = float(td.get('off_reb', 0) or 0)
                        to = float(td.get('TO', 0) or 0)
                        fta = float(td.get('FT_att', 0) or 0)
                        poss = fga - oreb + to + 0.44 * fta
                        total += poss
                        count += 1
                if count > 0:
                    baseline = total / count
        except Exception:
            baseline = 100.0

        self._league_pace_baseline_cache[season] = float(baseline)
        return float(baseline)

    def _calculate_pace_interaction_feature(
        self,
        time_period: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        pace_interaction|<time_period>|harmonic_mean|<perspective>
        Harmonic mean of both teams' pace. Returns same value for home/away, diff is 0.
        Robust fallback behavior per spec.
        """
        p_home = self.calculate_feature(f'pace|{time_period}|avg|home', home_team, away_team, season, year, month, day, None) or 0.0
        p_away = self.calculate_feature(f'pace|{time_period}|avg|away', home_team, away_team, season, year, month, day, None) or 0.0

        p_home = float(p_home or 0.0)
        p_away = float(p_away or 0.0)

        interaction = 0.0
        if p_home > 0 and p_away > 0:
            interaction = (2.0 * p_home * p_away) / (p_home + p_away)
        elif (p_home > 0) != (p_away > 0):
            interaction = max(p_home, p_away)
        else:
            # both zero/missing - try fallback to season if using games_N
            if time_period != 'season':
                season_interaction = self._calculate_pace_interaction_feature(
                    'season', perspective, home_team, away_team, season, year, month, day
                )
                if season_interaction > 0:
                    interaction = float(season_interaction)
            if interaction == 0.0:
                interaction = self._get_league_pace_baseline(season)

        # Return based on perspective
        # 'none' is the canonical perspective for matchup-level metrics
        # 'home'/'away' return the same value (for backwards compatibility)
        # 'diff' returns 0 since home == away
        if perspective == 'diff':
            return 0.0
        # 'none', 'home', 'away' all return the matchup interaction value
        return interaction

    def _calculate_est_possessions_feature(
        self,
        time_period: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        est_possessions|<time_period>|raw|<perspective>
        Estimated possessions for the matchup (pace_interaction value).
        v1: same scalar for home and away.
        """
        interaction = self._calculate_pace_interaction_feature(
            time_period, perspective, home_team, away_team, season, year, month, day
        )
        if perspective in ['home', 'away', 'none']:
            return float(interaction)
        # diff is zero in v1 since both sides share same estimate
        return 0.0

    def _calculate_exp_points_feature(
        self,
        base_stat: str,
        time_period: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        exp_points_off|<time_period>|raw|<perspective>
        exp_points_def|<time_period>|raw|<perspective>
        exp_points_matchup|<time_period>|raw|<perspective>
        Built from off/def ratings and est_possessions.
        """
        # Get estimated possessions 
        est_home = self._calculate_est_possessions_feature(
            time_period, 'home', home_team, away_team, season, year, month, day
        ) or 0.0
        est_away = self._calculate_est_possessions_feature(
            time_period, 'away', home_team, away_team, season, year, month, day
        ) or 0.0

        off_home = self.calculate_feature(f'off_rtg|{time_period}|raw|home', home_team, away_team, season, year, month, day, None) or 0.0
        off_away = self.calculate_feature(f'off_rtg|{time_period}|raw|away', home_team, away_team, season, year, month, day, None) or 0.0
        def_home = self.calculate_feature(f'def_rtg|{time_period}|raw|home', home_team, away_team, season, year, month, day, None) or 0.0
        def_away = self.calculate_feature(f'def_rtg|{time_period}|raw|away', home_team, away_team, season, year, month, day, None) or 0.0

        exp_off_home = (float(off_home) / 100.0) * float(est_home)
        exp_off_away = (float(off_away) / 100.0) * float(est_away)
        exp_def_home = (float(def_home) / 100.0) * float(est_home)
        exp_def_away = (float(def_away) / 100.0) * float(est_away)

        if base_stat == 'exp_points_off':
            if perspective == 'home':
                return exp_off_home
            if perspective == 'away':
                return exp_off_away
            return exp_off_home - exp_off_away

        if base_stat == 'exp_points_def':
            if perspective == 'home':
                return exp_def_home
            if perspective == 'away':
                return exp_def_away
            return exp_def_home - exp_def_away

        # matchup expected points
        home_exp = 0.5 * exp_off_home + 0.5 * exp_def_away
        away_exp = 0.5 * exp_off_away + 0.5 * exp_def_home
        if perspective == 'home':
            return home_exp
        if perspective == 'away':
            return away_exp
        return home_exp - away_exp

    def _calculate_attempt_rate_feature(
        self,
        stat_name: str,
        time_period: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        three_rate / ft_rate:
          ratio of totals over the window:
            three_rate = sum(three_att) / sum(FG_att)
            ft_rate    = sum(FT_att) / sum(FG_att)

        Split handling (v1 per spec): for the home-team value, only count games where that
        team played at home; for the away-team value, only count games where that team played away.
        """
        if time_period not in ['season'] and not time_period.startswith('games_'):
            return 0.0

        def _ratio_for(team: str, games: list, require_side: str) -> float:
            # require_side: 'home' or 'away' (location of the team in the game)
            if require_side == 'home':
                games = [g for g in games if g.get('homeTeam', {}).get('name') == team]
            elif require_side == 'away':
                games = [g for g in games if g.get('awayTeam', {}).get('name') == team]

            total_fg_att = 0.0
            total_three_att = 0.0
            total_ft_att = 0.0

            for g in games:
                if g.get('homeTeam', {}).get('name') == team:
                    td = g.get('homeTeam', {}) or {}
                elif g.get('awayTeam', {}).get('name') == team:
                    td = g.get('awayTeam', {}) or {}
                else:
                    continue

                total_fg_att += float(td.get('FG_att', 0) or 0)
                total_three_att += float(td.get('three_att', 0) or 0)
                total_ft_att += float(td.get('FT_att', 0) or 0)

            if total_fg_att <= 0:
                return 0.0

            if stat_name == 'three_rate':
                return total_three_att / total_fg_att
            else:
                return total_ft_att / total_fg_att

        # Build team windows
        home_games = self._get_games_for_time_period(home_team, season, year, month, day, time_period)
        away_games = self._get_games_for_time_period(away_team, season, year, month, day, time_period)

        home_val = _ratio_for(home_team, home_games, require_side='home')
        away_val = _ratio_for(away_team, away_games, require_side='away')

        if perspective == 'home':
            return float(home_val)
        if perspective == 'away':
            return float(away_val)
        return float(home_val) - float(away_val)

    # =========================================================================
    # GB-4 Matchup & Chemistry Feature Calculations
    # =========================================================================

    def _calculate_margin_feature(
        self,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate margin (point differential) feature.

        For rolling windows (games_10, season): computes avg or std of margins.
        For H2H windows (h2h_last1, h2h_last3, h2h_last5): computes from H2H games.

        Args:
            time_period: 'season', 'games_N', or 'h2h_lastN'
            calc_weight: 'raw', 'avg', or 'std'
            perspective: 'home', 'away', or 'diff'
            is_side: Whether to filter by home/away games only
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Margin feature value
        """
        # Handle H2H time periods
        if time_period.startswith('h2h_last'):
            n_games = int(time_period.replace('h2h_last', ''))
            h2h_games = self._get_h2h_games(home_team, away_team, season, year, month, day, n_games)

            if not h2h_games:
                return 0.0

            # For H2H, margin is from home team's perspective (positive = home team won)
            margins = []
            for game in h2h_games:
                home_score = game.get('homeTeam', {}).get('points', 0) or 0
                away_score = game.get('awayTeam', {}).get('points', 0) or 0
                game_home_team = game.get('homeTeam', {}).get('name', '')

                # Calculate margin from the perspective of the current home_team
                if game_home_team == home_team:
                    margin = home_score - away_score
                else:
                    margin = away_score - home_score
                margins.append(margin)

            if calc_weight == 'raw' and margins:
                return float(margins[-1])  # Most recent H2H game margin
            elif calc_weight == 'avg':
                return self.avg(margins)
            elif calc_weight == 'std':
                return self.std(margins)
            return 0.0

        # Handle regular time periods (season, games_N)
        home_games = self._get_games_for_time_period(home_team, season, year, month, day, time_period)
        away_games = self._get_games_for_time_period(away_team, season, year, month, day, time_period)

        # Filter to side-specific games if needed
        if is_side:
            home_games = [g for g in home_games if g.get('homeTeam', {}).get('name') == home_team]
            away_games = [g for g in away_games if g.get('awayTeam', {}).get('name') == away_team]

        def _get_margins(team: str, games: list) -> list:
            """Extract margins for a team from their games."""
            margins = []
            for game in games:
                home_score = game.get('homeTeam', {}).get('points', 0) or 0
                away_score = game.get('awayTeam', {}).get('points', 0) or 0
                game_home_team = game.get('homeTeam', {}).get('name', '')

                if game_home_team == team:
                    margin = home_score - away_score
                else:
                    margin = away_score - home_score
                margins.append(margin)
            return margins

        home_margins = _get_margins(home_team, home_games)
        away_margins = _get_margins(away_team, away_games)

        def _calc_margin_stat(margins: list) -> float:
            if not margins:
                return 0.0
            if calc_weight == 'avg':
                return self.avg(margins)
            elif calc_weight == 'std':
                return self.std(margins)
            elif calc_weight == 'raw':
                return float(sum(margins))  # Total margin (unusual but supported)
            return 0.0

        home_val = _calc_margin_stat(home_margins)
        away_val = _calc_margin_stat(away_margins)

        if perspective == 'home':
            return home_val
        elif perspective == 'away':
            return away_val
        else:  # 'diff'
            return home_val - away_val

    def _calculate_h2h_win_pct_feature(
        self,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate head-to-head win percentage.

        Args:
            time_period: 'season' (current season only), 'last_3', 'last_5' (cross-season)
            calc_weight: 'raw' (simple pct) or 'beta' (Beta-prior-smoothed)
            perspective: 'home', 'away', or 'diff'
            is_side: If True, only count games where today's home was home AND today's away was away
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Win percentage (0.0 to 1.0), or diff between home and away pcts
            Returns 0.5 (neutral) if no H2H games found
        """
        # Determine if season-only or cross-season lookup
        season_only = (time_period == 'season')
        n_games = 100  # Default: get all

        if time_period.startswith('last_'):
            n_games = int(time_period.replace('last_', ''))

        # Get H2H games with optional side filter
        h2h_games = self._get_h2h_games(
            home_team, away_team, season, year, month, day,
            n_games=n_games, side_filter=is_side, home_team=home_team, away_team=away_team,
            season_only=season_only
        )

        if not h2h_games:
            return 0.5  # Default to neutral when no H2H data

        # Calculate home team's wins
        home_wins = 0
        for game in h2h_games:
            home_score = game.get('homeTeam', {}).get('points', 0) or 0
            away_score = game.get('awayTeam', {}).get('points', 0) or 0
            game_home_team = game.get('homeTeam', {}).get('name', '')

            # Did today's home_team win this game?
            if game_home_team == home_team:
                if home_score > away_score:
                    home_wins += 1
            else:  # home_team was playing away in this game
                if away_score > home_score:
                    home_wins += 1

        n = len(h2h_games)

        # Apply calc_weight method
        if calc_weight == 'beta':
            # Beta-prior-smoothed: (wins + α) / (games + α + β)
            # Using α=β=2 for symmetric prior (equivalent to adding 2 wins and 2 losses)
            alpha = 2.0
            beta = 2.0
            home_pct = (home_wins + alpha) / (n + alpha + beta)
        else:  # 'raw'
            home_pct = float(home_wins) / n

        away_pct = 1.0 - home_pct  # Away's pct is always 1 - home's

        if perspective == 'home':
            return home_pct
        elif perspective == 'away':
            return away_pct
        else:  # 'diff'
            return home_pct - away_pct

    def _calculate_h2h_games_count_feature(
        self,
        time_period: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Return the number of H2H games found (sample size indicator).

        Args:
            time_period: 'season' (current season only), 'last_3', 'last_5' (cross-season)
            is_side: If True, only count games where today's home was home AND today's away was away
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Number of H2H games found
        """
        # Determine if season-only or cross-season lookup
        season_only = (time_period == 'season')
        n_games = 100  # Default: get all

        if time_period.startswith('last_'):
            n_games = int(time_period.replace('last_', ''))

        h2h_games = self._get_h2h_games(
            home_team, away_team, season, year, month, day,
            n_games=n_games, side_filter=is_side, home_team=home_team, away_team=away_team,
            season_only=season_only
        )
        return float(len(h2h_games))

    # -----------------------------------------------------------------
    # Conference win percentage (league-specific)
    # -----------------------------------------------------------------

    def _calculate_conf_wins_feature(
        self,
        time_period: str,
        calc_weight: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
    ) -> float:
        """Conference win percentage: wins vs same-conference opponents / total conf games."""
        home_pct = self._get_conference_win_pct(home_team, season, year, month, day)
        away_pct = self._get_conference_win_pct(away_team, season, year, month, day)

        if perspective == 'home':
            return home_pct
        elif perspective == 'away':
            return away_pct
        else:  # diff
            return home_pct - away_pct

    def _get_conference_win_pct(self, team: str, season: str, year: int, month: int, day: int) -> float:
        """Get a team's conference win percentage up to (not including) the given date."""
        conference = self._get_team_conference(team)
        if not conference:
            return 0.5  # Neutral default if no conference data

        conf_teams = self._get_conference_teams(conference)

        games = self._get_games_for_time_period(team, season, year, month, day, 'season')

        conf_wins = 0
        conf_games = 0
        for game in games:
            game_home = game.get('homeTeam', {}).get('name', '')
            game_away = game.get('awayTeam', {}).get('name', '')

            # Also check team_id for CBB which uses id-based matching
            game_home_id = str(game.get('homeTeam', {}).get('team_id', ''))
            game_away_id = str(game.get('awayTeam', {}).get('team_id', ''))

            # Determine if this team is home or away in this game
            is_home = (game_home == team or game_home_id == team)

            # Determine opponent identifiers
            if is_home:
                opp_name = game_away
                opp_id = game_away_id
            else:
                opp_name = game_home
                opp_id = game_home_id

            # Check if opponent is in same conference
            if opp_name not in conf_teams and opp_id not in conf_teams:
                continue

            conf_games += 1
            home_pts = game.get('homeTeam', {}).get('points', 0) or 0
            away_pts = game.get('awayTeam', {}).get('points', 0) or 0
            if is_home and home_pts > away_pts:
                conf_wins += 1
            elif not is_home and away_pts > home_pts:
                conf_wins += 1

        if conf_games == 0:
            return 0.5  # Neutral default

        return conf_wins / conf_games

    def _calculate_conf_margin_feature(
        self,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
    ) -> float:
        """Average margin per game against same-conference opponents this season."""
        home_margin = self._get_conference_avg_margin(home_team, season, year, month, day)
        away_margin = self._get_conference_avg_margin(away_team, season, year, month, day)

        if perspective == 'home':
            return home_margin
        elif perspective == 'away':
            return away_margin
        else:  # diff
            return home_margin - away_margin

    def _get_conference_avg_margin(self, team: str, season: str, year: int, month: int, day: int) -> float:
        """Get a team's average margin in conference games up to (not including) the given date."""
        conference = self._get_team_conference(team)
        if not conference:
            return 0.0

        conf_teams = self._get_conference_teams(conference)
        games = self._get_games_for_time_period(team, season, year, month, day, 'season')

        total_margin = 0.0
        conf_games = 0
        for game in games:
            game_home = game.get('homeTeam', {}).get('name', '')
            game_away = game.get('awayTeam', {}).get('name', '')
            game_home_id = str(game.get('homeTeam', {}).get('team_id', ''))
            game_away_id = str(game.get('awayTeam', {}).get('team_id', ''))

            is_home = (game_home == team or game_home_id == team)
            if is_home:
                opp_name = game_away
                opp_id = game_away_id
            else:
                opp_name = game_home
                opp_id = game_home_id

            if opp_name not in conf_teams and opp_id not in conf_teams:
                continue

            conf_games += 1
            home_pts = game.get('homeTeam', {}).get('points', 0) or 0
            away_pts = game.get('awayTeam', {}).get('points', 0) or 0
            if is_home:
                total_margin += (home_pts - away_pts)
            else:
                total_margin += (away_pts - home_pts)

        if conf_games == 0:
            return 0.0

        return total_margin / conf_games

    def _get_team_conference(self, team: str):
        """Look up a team's conference from teams collection. Cached per session."""
        if not hasattr(self, '_conference_cache'):
            self._conference_cache = {}
        if team in self._conference_cache:
            return self._conference_cache[team]

        if self.db is None:
            return None

        teams_coll = self.league.collections.get("teams", "cbb_teams")
        team_doc = self.db[teams_coll].find_one(
            {'$or': [{'abbreviation': team}, {'team_id': team}]},
            {'conference': 1}
        )
        conference = team_doc.get('conference') if team_doc else None
        self._conference_cache[team] = conference
        return conference

    def _get_conference_teams(self, conference: str) -> set:
        """Get all team names/ids in a conference. Cached per session."""
        if not hasattr(self, '_conf_teams_cache'):
            self._conf_teams_cache = {}
        if conference in self._conf_teams_cache:
            return self._conf_teams_cache[conference]

        if self.db is None:
            return set()

        teams_coll = self.league.collections.get("teams", "cbb_teams")
        docs = self.db[teams_coll].find(
            {'conference': conference},
            {'abbreviation': 1, 'team_id': 1}
        )
        names = set()
        for d in docs:
            if d.get('abbreviation'):
                names.add(d['abbreviation'])
            if d.get('team_id'):
                names.add(str(d['team_id']))
        self._conf_teams_cache[conference] = names
        return names

    def _calculate_conf_gp_feature(
        self,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
    ) -> float:
        """Conference games played count for each team this season."""
        home_gp = self._get_conference_games_played(home_team, season, year, month, day)
        away_gp = self._get_conference_games_played(away_team, season, year, month, day)

        if perspective == 'home':
            return float(home_gp)
        elif perspective == 'away':
            return float(away_gp)
        else:  # diff
            return float(home_gp - away_gp)

    def _get_conference_games_played(self, team: str, season: str, year: int, month: int, day: int) -> int:
        """Count conference games a team has played this season before the given date."""
        conference = self._get_team_conference(team)
        if not conference:
            return 0

        conf_teams = self._get_conference_teams(conference)
        games = self._get_games_for_time_period(team, season, year, month, day, 'season')

        conf_games = 0
        for game in games:
            game_home = game.get('homeTeam', {}).get('name', '')
            game_away = game.get('awayTeam', {}).get('name', '')
            game_home_id = str(game.get('homeTeam', {}).get('team_id', ''))
            game_away_id = str(game.get('awayTeam', {}).get('team_id', ''))

            is_home = (game_home == team or game_home_id == team)
            if is_home:
                opp_name, opp_id = game_away, game_away_id
            else:
                opp_name, opp_id = game_home, game_home_id

            if opp_name in conf_teams or opp_id in conf_teams:
                conf_games += 1

        return conf_games

    def _calculate_postseason_feature(self, home_team: str, away_team: str,
                                       year: int, month: int, day: int) -> float:
        """Binary: 1 if this game is postseason/playoffs, 0 otherwise."""
        if not hasattr(self, '_game_type_cache'):
            self._game_type_cache = {}

        date_str = f"{year}-{month:02d}-{day:02d}"
        cache_key = (home_team, away_team, date_str)
        if cache_key in self._game_type_cache:
            game_type = self._game_type_cache[cache_key]
        elif self.db is not None:
            games_coll = self.league.collections.get("games", "nba_games")
            doc = self.db[games_coll].find_one(
                {'date': date_str, 'homeTeam.name': home_team, 'awayTeam.name': away_team},
                {'game_type': 1}
            )
            game_type = doc.get('game_type') if doc else None
            self._game_type_cache[cache_key] = game_type
        else:
            game_type = None

        return 1.0 if game_type in ('postseason', 'playoffs') else 0.0

    def _calculate_margin_h2h_feature(
        self,
        time_period: str,
        calc_weight: str,
        perspective: str,
        is_side: bool,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate point margin in H2H games.

        Args:
            time_period: 'season' (current season only), 'last_3', 'last_5' (cross-season)
            calc_weight: 'avg' (simple average), 'eb' (Empirical Bayes shrinkage), 'logw' (log-weighted)
            perspective: 'home', 'away', or 'diff'
            is_side: If True, only count games where today's home was home AND today's away was away
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Point margin for the specified team in H2H games.
            Positive = team outscored opponent on average, negative = team was outscored.
            Returns 0.0 if no H2H games found.
        """
        import math

        # Determine if season-only or cross-season lookup
        season_only = (time_period == 'season')
        n_games = 100  # Default: get all

        if time_period.startswith('last_'):
            n_games = int(time_period.replace('last_', ''))

        # Get H2H games with optional side filter
        h2h_games = self._get_h2h_games(
            home_team, away_team, season, year, month, day,
            n_games=n_games, side_filter=is_side, home_team=home_team, away_team=away_team,
            season_only=season_only
        )

        if not h2h_games:
            return 0.0  # No H2H games

        # Calculate total margin for today's home team
        total_margin = 0.0
        for game in h2h_games:
            game_home_score = game.get('homeTeam', {}).get('points', 0) or 0
            game_away_score = game.get('awayTeam', {}).get('points', 0) or 0
            game_home_team = game.get('homeTeam', {}).get('name', '')

            # Calculate margin from today's home team's perspective
            if game_home_team == home_team:
                # Today's home team was home in this game
                margin = game_home_score - game_away_score
            else:
                # Today's home team was away in this game
                margin = game_away_score - game_home_score

            total_margin += margin

        n = len(h2h_games)
        home_avg_margin = total_margin / n

        # Apply calc_weight method
        if calc_weight == 'eb':
            # Empirical Bayes shrinkage: (n / (n + k)) * avg_margin
            # k = shrinkage constant (higher = more shrinkage toward 0)
            k = 5.0  # Shrinkage constant
            home_margin = (n / (n + k)) * home_avg_margin
        elif calc_weight == 'logw':
            # Log-weighted: avg_margin * log1p(n)
            # Scales up the margin based on sample size reliability
            home_margin = home_avg_margin * math.log1p(n)
        else:  # 'avg'
            home_margin = home_avg_margin

        if perspective == 'home':
            return home_margin
        elif perspective == 'away':
            return -home_margin  # Away's margin is negative of home's
        else:  # 'diff'
            # diff = home - away = home - (-home) = 2 * home
            return 2.0 * home_margin

    def _calculate_close_win_pct_feature(
        self,
        time_period: str,
        perspective: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int
    ) -> float:
        """
        Calculate win percentage in close games (decided by <=5 points).

        Args:
            time_period: 'season' or 'games_close5' (last 5 close games)
            perspective: 'home', 'away', or 'diff'
            home_team, away_team, season, year, month, day: Game context

        Returns:
            Close game win percentage (0.0 to 1.0)
        """
        def _close_win_pct_for_team(team: str) -> float:
            # Get all season games for the team
            all_games = self.get_team_games_before_date(team, year, month, day, season)

            # Filter to close games (margin <= 5)
            close_games = []
            for game in all_games:
                home_score = game.get('homeTeam', {}).get('points', 0) or 0
                away_score = game.get('awayTeam', {}).get('points', 0) or 0
                margin = abs(home_score - away_score)
                if margin <= 5:
                    close_games.append(game)

            # If games_close5, take only last 5 close games
            if time_period == 'games_close5':
                close_games = close_games[-5:] if len(close_games) >= 5 else close_games

            if not close_games:
                return 0.5  # Default to neutral

            wins = 0
            for game in close_games:
                home_score = game.get('homeTeam', {}).get('points', 0) or 0
                away_score = game.get('awayTeam', {}).get('points', 0) or 0
                game_home_team = game.get('homeTeam', {}).get('name', '')

                if game_home_team == team:
                    if home_score > away_score:
                        wins += 1
                else:
                    if away_score > home_score:
                        wins += 1

            return float(wins) / len(close_games)

        home_val = _close_win_pct_for_team(home_team)
        away_val = _close_win_pct_for_team(away_team)

        if perspective == 'home':
            return home_val
        elif perspective == 'away':
            return away_val
        else:  # 'diff'
            return home_val - away_val

    def _get_h2h_games(
        self,
        team1: str,
        team2: str,
        season: str,
        year: int,
        month: int,
        day: int,
        n_games: int = 100,
        side_filter: bool = False,
        home_team: str = None,
        away_team: str = None,
        season_only: bool = True
    ) -> list:
        """
        Get head-to-head games between two teams before a given date.

        Uses preloaded data when available, falls back to DB with caching.

        Args:
            team1, team2: The two teams
            season: Current season string
            year, month, day: Game date (exclude games on or after this date)
            n_games: Maximum number of H2H games to return (default 100 = effectively all)
            side_filter: If True, only include games where home_team was home AND away_team was away
            home_team, away_team: Required if side_filter=True (today's home/away teams)
            season_only: If True, only returns games from current season. If False, searches all seasons.

        Returns:
            List of H2H game documents, most recent last
        """
        game_date = f"{year}-{month:02d}-{day:02d}"

        # Normalize cache key (order teams alphabetically for consistent keys)
        teams_key = tuple(sorted([team1, team2]))
        cache_key = (teams_key[0], teams_key[1], game_date, season, n_games, side_filter, home_team, away_team, season_only)

        # Check cache first
        if cache_key in self._h2h_games_cache:
            return self._h2h_games_cache[cache_key]

        # Try to use preloaded games data (NO DB CALLS)
        if self.games_home is not None and self.games_away is not None:
            h2h_games = []

            # Determine which seasons to search
            seasons_to_search = [season] if season_only else list(self.games_home.keys())

            for season_key in seasons_to_search:
                if season_key not in self.games_home:
                    continue
                season_games = self.games_home[season_key]
                for date_str, date_games in season_games.items():
                    # Skip games on or after target date
                    if date_str >= game_date:
                        continue

                    # Check if this date has an H2H game
                    for game_home_team, game in date_games.items():
                        # Skip preseason and allstar games
                        game_type = game.get('game_type', 'regseason')
                        if game_type in self._exclude_game_types:
                            continue

                        game_away_team = game.get('awayTeam', {}).get('name', '')

                        # Check if this is an H2H game between team1 and team2
                        is_h2h = (game_home_team == team1 and game_away_team == team2) or \
                                 (game_home_team == team2 and game_away_team == team1)

                        if not is_h2h:
                            continue

                        # Apply side filter if requested
                        if side_filter and home_team and away_team:
                            # Only include if today's home team was home AND today's away team was away
                            if game_home_team != home_team or game_away_team != away_team:
                                continue

                        h2h_games.append(game)

            # Sort by date descending, take n_games most recent
            h2h_games.sort(key=lambda g: g.get('date', ''), reverse=True)
            h2h_games = h2h_games[:n_games]

            # Reverse to get chronological order (oldest first, newest last)
            h2h_games.reverse()

            # Cache the result
            self._h2h_games_cache[cache_key] = h2h_games
            return h2h_games

        # Fall back to DB query if no preloaded data
        if self.db is None:
            return []

        try:
            games_collection = self.league.collections.get("games", "stats_nba") if self.league else "stats_nba"

            if side_filter and home_team and away_team:
                # Only games where today's home was home and today's away was away
                query = {
                    'homeTeam.name': home_team,
                    'awayTeam.name': away_team,
                    'date': {'$lt': game_date},
                    'game_type': {'$nin': self._exclude_game_types}
                }
                if season_only:
                    query['season'] = season
            else:
                # All H2H games
                query = {
                    '$or': [
                        {'homeTeam.name': team1, 'awayTeam.name': team2},
                        {'homeTeam.name': team2, 'awayTeam.name': team1}
                    ],
                    'date': {'$lt': game_date},
                    'game_type': {'$nin': self._exclude_game_types}
                }
                if season_only:
                    query['season'] = season

            h2h_games = list(self.db[games_collection].find(query).sort('date', -1).limit(n_games))

            # Reverse to get chronological order (oldest first, newest last)
            h2h_games.reverse()

            # Cache the result
            self._h2h_games_cache[cache_key] = h2h_games

            return h2h_games

        except Exception:
            return []

    def _calculate_special_feature(
        self,
        feature_name: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        per_calculator=None
    ) -> float:
        """
        Calculate special features (elo, rest, per_available).

        Args:
            feature_name: Full feature name
            home_team, away_team, season, year, month, day: Game context
            per_calculator: Optional PERCalculator instance

        Returns:
            Feature value
        """
        components = parse_feature_name(feature_name)
        if not components:
            return 0.0

        stat_name = components.stat_name
        perspective = components.home_away_diff

        if stat_name == 'elo':
            # Look up cached Elo ratings
            return self._get_cached_elo_feature(
                home_team, away_team, season, year, month, day, perspective
            )
        elif stat_name == 'rest':
            # Rest is handled separately in BballModel
            # For now, return 0.0 - this will be calculated in BballModel
            return 0.0
        elif stat_name == 'per_available':
            # PER available flag - handled separately
            return 0.0
        elif stat_name in ('vegas_ML', 'vegas_spread', 'vegas_ou', 'vegas_implied_prob', 'vegas_edge'):
            # Vegas betting lines and derived features - pull from game document's 'vegas' field
            return self._get_vegas_feature(
                stat_name, home_team, away_team, year, month, day, perspective
            )

        return 0.0

    def _get_cached_elo_feature(
        self,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        perspective: str
    ) -> float:
        """
        Get Elo feature value from cache.

        Args:
            home_team, away_team: Team names
            season: Season string
            year, month, day: Game date
            perspective: 'home', 'away', or 'diff'

        Returns:
            Elo feature value (or 0.0 if cache unavailable)
        """
        if self.db is None:
            return 0.0

        try:
            from bball_app.core.stats.elo_cache import EloCache

            # Lazy-initialize elo cache
            if not hasattr(self, '_elo_cache'):
                self._elo_cache = EloCache(self.db)

            game_date = f"{year}-{month:02d}-{day:02d}"

            # Get Elo for both teams with fallback to most recent or default
            home_elo = self._elo_cache.get_elo_for_game_with_fallback(
                home_team, game_date, season
            )
            away_elo = self._elo_cache.get_elo_for_game_with_fallback(
                away_team, game_date, season
            )

            if perspective == 'home':
                return home_elo
            elif perspective == 'away':
                return away_elo
            else:  # 'diff'
                return home_elo - away_elo

        except Exception as e:
            # If cache lookup fails, return 0.0 rather than breaking prediction
            import logging
            logging.warning(f"Elo cache lookup failed: {e}")
            return 0.0

    def _get_vegas_feature(
        self,
        stat_name: str,
        home_team: str,
        away_team: str,
        year: int,
        month: int,
        day: int,
        perspective: str
    ) -> float:
        """
        Get Vegas betting line feature from game document.

        Checks 'vegas' field first (populated from CSV), falls back to
        'pregame_lines' field (legacy ESPN data) if vegas not available.

        Uses preloaded games if available (training mode), falls back to
        DB query if not (prediction mode).

        Args:
            stat_name: One of:
                - 'vegas_ML': Moneyline odds (American format)
                - 'vegas_spread': Point spread
                - 'vegas_ou': Over/under total
                - 'vegas_implied_prob': Win probability from ML (formula: if ML<0: -ML/(-ML+100), else: 100/(ML+100))
                - 'vegas_edge': Implied prob minus spread-derived prob (spread_prob = 1/(1+exp(-spread/6.5)))
            home_team, away_team: Team abbreviations
            year, month, day: Game date
            perspective: 'home', 'away', 'diff', or 'none'

        Returns:
            Vegas line value (or 0.0 if not available)
        """
        try:
            game_date = f"{year}-{month:02d}-{day:02d}"
            game = None

            # Try preloaded games first (fast path for training)
            if self.games_home is not None:
                # Derive season from date (NBA season cutover is August)
                cutover = self.league.season_cutover_month if self.league else 8
                if month > cutover:
                    season = f"{year}-{year + 1}"
                else:
                    season = f"{year - 1}-{year}"

                # Look up game from preloaded data: games_home[season][date][home_team]
                if season in self.games_home:
                    if game_date in self.games_home[season]:
                        game = self.games_home[season][game_date].get(home_team)

            # Fall back to DB query if no preloaded data or game not found
            if game is None and self.db is not None:
                games_collection = self.league.collections.get("games", "stats_nba") if self.league else "stats_nba"
                game = self.db[games_collection].find_one(
                    {
                        'date': game_date,
                        'homeTeam.name': home_team,
                        'awayTeam.name': away_team,
                    },
                    {'vegas': 1, 'pregame_lines': 1}
                )

            if not game:
                return 0.0

            # Try 'vegas' field first, fall back to 'pregame_lines'
            vegas = game.get('vegas')
            pregame = game.get('pregame_lines')

            if stat_name == 'vegas_ML':
                # Try vegas field first
                if vegas:
                    home_ml = vegas.get('home_ML')
                    away_ml = vegas.get('away_ML')
                else:
                    home_ml = None
                    away_ml = None

                # Fall back to pregame_lines
                if home_ml is None and pregame:
                    home_ml = pregame.get('home_ml')
                if away_ml is None and pregame:
                    away_ml = pregame.get('away_ml')

                if perspective == 'home':
                    return float(home_ml) if home_ml is not None else 0.0
                elif perspective == 'away':
                    return float(away_ml) if away_ml is not None else 0.0
                else:  # 'diff'
                    if home_ml is not None and away_ml is not None:
                        return float(home_ml) - float(away_ml)
                    return 0.0

            elif stat_name == 'vegas_spread':
                # Try vegas field first (has home_spread/away_spread)
                if vegas:
                    if perspective == 'home':
                        spread = vegas.get('home_spread')
                        if spread is not None:
                            return float(spread)
                    elif perspective == 'away':
                        spread = vegas.get('away_spread')
                        if spread is not None:
                            return float(spread)

                # Fall back to pregame_lines (has single 'spread' from home perspective)
                if pregame:
                    spread = pregame.get('spread')
                    if spread is not None:
                        if perspective == 'home':
                            return float(spread)
                        elif perspective == 'away':
                            return -float(spread)  # Negate for away perspective

                return 0.0

            elif stat_name == 'vegas_ou':
                # Try vegas field first
                ou = None
                if vegas:
                    ou = vegas.get('OU')

                # Fall back to pregame_lines
                if ou is None and pregame:
                    ou = pregame.get('over_under')

                return float(ou) if ou is not None else 0.0

            elif stat_name == 'vegas_implied_prob':
                # Implied win probability from moneyline
                # Formula: if ML < 0: (-ML) / (-ML + 100), else: 100 / (ML + 100)
                if vegas:
                    home_ml = vegas.get('home_ML')
                    away_ml = vegas.get('away_ML')
                else:
                    home_ml = None
                    away_ml = None

                # Fall back to pregame_lines
                if home_ml is None and pregame:
                    home_ml = pregame.get('home_ml')
                if away_ml is None and pregame:
                    away_ml = pregame.get('away_ml')

                def ml_to_prob(ml):
                    if ml is None:
                        return None
                    ml = float(ml)
                    if ml < 0:
                        return (-ml) / (-ml + 100)
                    else:
                        return 100 / (ml + 100)

                home_prob = ml_to_prob(home_ml)
                away_prob = ml_to_prob(away_ml)

                if perspective == 'home':
                    return home_prob if home_prob is not None else 0.0
                elif perspective == 'away':
                    return away_prob if away_prob is not None else 0.0
                else:  # 'diff'
                    if home_prob is not None and away_prob is not None:
                        return home_prob - away_prob
                    return 0.0

            elif stat_name == 'vegas_edge':
                # Edge = implied_prob - spread_prob
                # spread_prob = 1 / (1 + exp(-spread / k)), k ≈ 6.5
                import math
                K = 6.5  # Spread-to-probability conversion factor

                # Get moneylines for implied prob
                if vegas:
                    home_ml = vegas.get('home_ML')
                    away_ml = vegas.get('away_ML')
                else:
                    home_ml = None
                    away_ml = None

                if home_ml is None and pregame:
                    home_ml = pregame.get('home_ml')
                if away_ml is None and pregame:
                    away_ml = pregame.get('away_ml')

                # Get spread
                home_spread = None
                if vegas:
                    home_spread = vegas.get('home_spread')
                if home_spread is None and pregame:
                    home_spread = pregame.get('spread')

                def ml_to_prob(ml):
                    if ml is None:
                        return None
                    ml = float(ml)
                    if ml < 0:
                        return (-ml) / (-ml + 100)
                    else:
                        return 100 / (ml + 100)

                def spread_to_prob(spread):
                    if spread is None:
                        return None
                    # Probability of winning given the spread
                    # Positive spread = underdog, negative spread = favorite
                    # spread_prob = 1 / (1 + exp(-spread / k))
                    # Note: spread is from home perspective (negative = home favored)
                    return 1 / (1 + math.exp(-float(spread) / K))

                home_implied = ml_to_prob(home_ml)
                away_implied = ml_to_prob(away_ml)
                home_spread_prob = spread_to_prob(home_spread)
                away_spread_prob = spread_to_prob(-float(home_spread)) if home_spread is not None else None

                if perspective == 'home':
                    if home_implied is not None and home_spread_prob is not None:
                        return home_implied - home_spread_prob
                    return 0.0
                elif perspective == 'away':
                    if away_implied is not None and away_spread_prob is not None:
                        return away_implied - away_spread_prob
                    return 0.0
                else:  # 'diff'
                    home_edge = (home_implied - home_spread_prob) if (home_implied is not None and home_spread_prob is not None) else None
                    away_edge = (away_implied - away_spread_prob) if (away_implied is not None and away_spread_prob is not None) else None
                    if home_edge is not None and away_edge is not None:
                        return home_edge - away_edge
                    return 0.0

            return 0.0

        except Exception as e:
            import logging
            logging.warning(f"Vegas feature lookup failed: {e}")
            return 0.0

    def _calculate_per_feature(
        self,
        feature_name: str,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        per_calculator=None
    ) -> float:
        """
        Calculate PER features.
        
        Args:
            feature_name: Full feature name (e.g., 'player_team_per|season|avg|diff')
            home_team, away_team, season, year, month, day: Game context
            per_calculator: PERCalculator instance
            
        Returns:
            Feature value
        """
        # PER features are handled separately in BballModel via _get_per_features
        # For now, return 0.0 - this will be calculated in BballModel
        return 0.0
    
    # =========================================================================
    # INJURY FEATURE CALCULATIONS
    # =========================================================================
    
    def get_injury_features(
        self,
        HOME: str,
        AWAY: str,
        season: str,
        year: int,
        month: int,
        day: int,
        game_doc: dict = None,
        per_calculator = None,
        recency_decay_k: float = 15.0,
        precomputed_season_severity: dict = None
    ) -> dict:
        """
        Calculate injury impact features for a game.

        Args:
            HOME: Home team name
            AWAY: Away team name
            season: Season string
            year, month, day: Game date
            game_doc: Optional game document (if None, will query)
            per_calculator: Optional PERCalculator instance
            recency_decay_k: Decay constant for recency weighting (default 15)
            precomputed_season_severity: Optional dict mapping (team, season, date) -> severity value.
                When provided, skips expensive _get_season_injury_severity() calculation.
                Used by chunked master training workflow for performance.

        Returns:
            Dict with injury features (home/away/diff)
        """
        if self.db is None:
            return {}
        
        # Ensure recency_decay_k has a default value if None
        if recency_decay_k is None:
            recency_decay_k = 15.0
        
        game_date = f"{year}-{month:02d}-{day:02d}"
        game_date_obj = date(year, month, day)
        
        # Get game document if not provided
        if game_doc is None and self._games_repo:
            game_doc = self._games_repo.find_one({
                'homeTeam.name': HOME,
                'awayTeam.name': AWAY,
                'season': season,
                'date': game_date
            })
        
        if not game_doc:
            return {}
        
        # Priority 1: Use injured_players from game_doc (for training data - historical injury data)
        home_injured_ids = game_doc.get('homeTeam', {}).get('injured_players', [])
        away_injured_ids = game_doc.get('awayTeam', {}).get('injured_players', [])
        
        # Convert to strings for queries
        home_injured_ids = [str(pid) for pid in home_injured_ids] if home_injured_ids else []
        away_injured_ids = [str(pid) for pid in away_injured_ids] if away_injured_ids else []
        
        # Priority 2: If injured_players is not present, use nba_rosters to check injured status (for prediction)
        # This happens when game_doc doesn't have injured_players populated (e.g., future games)
        # Only skip roster lookup if explicitly in batch training mode (not just because games are preloaded)
        # The _batch_training_mode flag is set during training workflows; prediction context injection
        # preloads games but should NOT skip roster lookups for future games.
        skip_roster_lookup = getattr(self, '_batch_training_mode', False)
        if not home_injured_ids and not away_injured_ids and not skip_roster_lookup:
            # Get player lists from stats_nba for each team
            home_player_ids = game_doc.get('homeTeam', {}).get('players', [])
            away_player_ids = game_doc.get('awayTeam', {}).get('players', [])

            # Convert to strings for queries
            home_player_ids = [str(pid) for pid in home_player_ids] if home_player_ids else []
            away_player_ids = [str(pid) for pid in away_player_ids] if away_player_ids else []

            if self._rosters_repo is not None:
                try:
                    # Get home team roster
                    home_roster_doc = self._rosters_repo.find_roster(HOME, season)
                    if home_roster_doc:
                        home_roster = home_roster_doc.get('roster', [])
                        # Build a map of player_id -> injured status (excluding disabled players)
                        # Disabled players are excluded entirely, matching build_player_lists_for_prediction behavior
                        home_roster_map = {
                            str(entry.get('player_id')): entry.get('injured', False)
                            for entry in home_roster
                            if not entry.get('disabled', False)
                        }
                        # If we have a concrete player list for this game, restrict to it.
                        # Otherwise (common for future games), use the roster's injured flags directly.
                        if home_player_ids:
                            home_injured_ids = [pid for pid in home_player_ids if home_roster_map.get(pid, False)]
                        else:
                            home_injured_ids = [pid for pid, is_inj in home_roster_map.items() if is_inj]

                    # Get away team roster
                    away_roster_doc = self._rosters_repo.find_roster(AWAY, season)
                    if away_roster_doc:
                        away_roster = away_roster_doc.get('roster', [])
                        # Build a map of player_id -> injured status (excluding disabled players)
                        # Disabled players are excluded entirely, matching build_player_lists_for_prediction behavior
                        away_roster_map = {
                            str(entry.get('player_id')): entry.get('injured', False)
                            for entry in away_roster
                            if not entry.get('disabled', False)
                        }
                        # If we have a concrete player list for this game, restrict to it.
                        # Otherwise (common for future games), use the roster's injured flags directly.
                        if away_player_ids:
                            away_injured_ids = [pid for pid in away_player_ids if away_roster_map.get(pid, False)]
                        else:
                            away_injured_ids = [pid for pid, is_inj in away_roster_map.items() if is_inj]
                except Exception as e:
                    # If we can't get rosters, fall back to empty lists (injury features will be 0)
                    pass
        
        # Get player names for injured players (SKIP in batch mode to avoid DB calls)
        # Only fetch names if not in preloaded/batch mode
        home_injured_names = []
        away_injured_names = []
        skip_name_lookup = getattr(self, '_batch_training_mode', False)
        if not skip_name_lookup and self._players_dir_repo is not None:
            try:
                # Get home team injured player names
                if home_injured_ids:
                    home_player_docs = self._players_dir_repo.find(
                        {'player_id': {'$in': home_injured_ids}},
                        projection={'player_id': 1, 'player_name': 1}
                    )
                    home_injured_names = [doc.get('player_name', 'Unknown') for doc in home_player_docs]

                # Get away team injured player names
                if away_injured_ids:
                    away_player_docs = self._players_dir_repo.find(
                        {'player_id': {'$in': away_injured_ids}},
                        projection={'player_id': 1, 'player_name': 1}
                    )
                    away_injured_names = [doc.get('player_name', 'Unknown') for doc in away_player_docs]
            except Exception as e:
                # Silently fail if we can't get player names - features still work
                pass
        
        # Calculate features for each team
        home_features = self._calculate_team_injury_features(
            HOME, season, game_date, game_date_obj, home_injured_ids, per_calculator, recency_decay_k
        )
        away_features = self._calculate_team_injury_features(
            AWAY, season, game_date, game_date_obj, away_injured_ids, per_calculator, recency_decay_k
        )
        
        # Build result dict
        features = {}
        
        # Get values directly from home/away features
        home_inj_per_value = home_features.get('injPerValue', 0.0)
        away_inj_per_value = away_features.get('injPerValue', 0.0)
        home_inj_top1_per = home_features.get('injTop1Per', 0.0)
        away_inj_top1_per = away_features.get('injTop1Per', 0.0)
        home_inj_top3_sum = home_features.get('injTop3PerSum', 0.0)
        away_inj_top3_sum = away_features.get('injTop3PerSum', 0.0)
        home_inj_min_lost = home_features.get('injMinLost', 0.0)
        away_inj_min_lost = away_features.get('injMinLost', 0.0)
        home_injury_severity = home_features.get('injurySeverity', 0.0)
        away_injury_severity = away_features.get('injurySeverity', 0.0)
        home_inj_rotation = home_features.get('injRotation', 0.0)
        away_inj_rotation = away_features.get('injRotation', 0.0)
        
        # Features
        # inj_per features
        features['inj_per|none|weighted_MIN|home'] = home_inj_per_value
        features['inj_per|none|weighted_MIN|away'] = away_inj_per_value
        features['inj_per|none|weighted_MIN|diff'] = home_inj_per_value - away_inj_per_value
        
        features['inj_per|none|top1_avg|home'] = home_inj_top1_per
        features['inj_per|none|top1_avg|away'] = away_inj_top1_per
        features['inj_per|none|top1_avg|diff'] = home_inj_top1_per - away_inj_top1_per
        
        features['inj_per|none|top3_sum|home'] = home_inj_top3_sum
        features['inj_per|none|top3_sum|away'] = away_inj_top3_sum
        features['inj_per|none|top3_sum|diff'] = home_inj_top3_sum - away_inj_top3_sum
        
        # inj_min_lost features
        features['inj_min_lost|none|raw|home'] = home_inj_min_lost
        features['inj_min_lost|none|raw|away'] = away_inj_min_lost
        features['inj_min_lost|none|raw|diff'] = home_inj_min_lost - away_inj_min_lost
        
        # inj_severity features (point-in-time)
        features['inj_severity|none|raw|home'] = home_injury_severity
        features['inj_severity|none|raw|away'] = away_injury_severity
        features['inj_severity|none|raw|diff'] = home_injury_severity - away_injury_severity

        # inj_severity|season - season-to-date weighted average injury severity
        # Formula: Σ(inj_min_lost_k) / Σ(team_rotation_mpg_k) over prior games
        # Interpretation: "What fraction of rotation minutes has this team lost to injuries this season?"
        # OPTIMIZATION: Use precomputed values if provided (avoids expensive per-row calculation)
        if precomputed_season_severity is not None:
            home_season_severity = precomputed_season_severity.get((HOME, season, game_date), 0.0)
            away_season_severity = precomputed_season_severity.get((AWAY, season, game_date), 0.0)
        else:
            home_season_severity = self._get_season_injury_severity(HOME, season, game_date)
            away_season_severity = self._get_season_injury_severity(AWAY, season, game_date)
        features['inj_severity|season|raw|home'] = home_season_severity
        features['inj_severity|season|raw|away'] = away_season_severity
        features['inj_severity|season|raw|diff'] = home_season_severity - away_season_severity

        # Player lists for injury features (for display in feature modal)
        features['_player_lists'] = {
            # injPerValue (weighted PER)
            'inj_per|none|weighted_MIN|home': home_features.get('injPerValue_players', []),
            'inj_per|none|weighted_MIN|away': away_features.get('injPerValue_players', []),
            # injTop1Per (top 1 PER)
            'inj_per|none|top1_avg|home': home_features.get('injTop1Per_players', []),
            'inj_per|none|top1_avg|away': away_features.get('injTop1Per_players', []),
            # injTop3PerSum (top 3 PER sum)
            'inj_per|none|top3_sum|home': home_features.get('injTop3PerSum_players', []),
            'inj_per|none|top3_sum|away': away_features.get('injTop3PerSum_players', []),
            # injMinLost (minutes lost)
            'inj_min_lost|none|raw|home': home_features.get('injMinLost_players', []),
            'inj_min_lost|none|raw|away': away_features.get('injMinLost_players', []),
            # injRotation (rotation count - uses same players as injMinLost)
            'inj_rotation_per|none|raw|home': home_features.get('injRotation_players', []),
            'inj_rotation_per|none|raw|away': away_features.get('injRotation_players', [])
        }
        
        # inj_rotation_per features
        features['inj_rotation_per|none|raw|home'] = home_inj_rotation
        features['inj_rotation_per|none|raw|away'] = away_inj_rotation
        features['inj_rotation_per|none|raw|diff'] = home_inj_rotation - away_inj_rotation
        
        # Injury blend feature: weighted combination of injury metrics
        # Formula: 0.45 * injurySeverity + 0.35 * injTop1Per + 0.20 * injRotation
        # New notation: inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|{perspective}
        blend_weight_severity = 0.45
        blend_weight_top1_per = 0.35
        blend_weight_rotation = 0.20
        
        # Home version
        features['inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|home'] = (
            blend_weight_severity * home_injury_severity +
            blend_weight_top1_per * home_inj_top1_per +
            blend_weight_rotation * home_inj_rotation
        )
        
        # Away version
        features['inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|away'] = (
            blend_weight_severity * away_injury_severity +
            blend_weight_top1_per * away_inj_top1_per +
            blend_weight_rotation * away_inj_rotation
        )
        
        # Diff version
        features['inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|diff'] = (
            blend_weight_severity * (home_injury_severity - away_injury_severity) +
            blend_weight_top1_per * (home_inj_top1_per - away_inj_top1_per) +
            blend_weight_rotation * (home_inj_rotation - away_inj_rotation)
        )

        # =====================================================================
        # NORMALIZED INJURY FEATURES (deconfounded from team quality)
        # =====================================================================
        # These features measure "proportion of talent missing" rather than
        # "raw talent missing", which avoids the confound where high injured
        # PER implies "good team" regardless of injury impact.

        EPS = 1e-6  # Epsilon to avoid division by zero

        # Get team PER baselines from per_calculator (if available)
        home_top3_sum = 0.0
        away_top3_sum = 0.0

        if per_calculator:
            try:
                home_per_features = per_calculator.compute_team_per_features(HOME, season, game_date)
                away_per_features = per_calculator.compute_team_per_features(AWAY, season, game_date)
                if home_per_features:
                    home_top3_sum = home_per_features.get('top3_sum', 0.0)
                if away_per_features:
                    away_top3_sum = away_per_features.get('top3_sum', 0.0)
            except Exception:
                pass  # If per_calculator fails, baselines stay at 0

        # B2: inj_per_share|none|top3_sum - fraction of top-team PER that is injured
        # Numerator: sum of top 3 injured players' PER (from injTop3PerSum)
        # Denominator: sum of top 3 roster players' PER (from per_calculator)
        # Formula: inj_per_share = inj_top3_sum / (top3_sum + eps), clipped to [0, 1.5]
        inj_per_share_home = min(1.5, max(0.0, home_inj_top3_sum / (home_top3_sum + EPS)))
        inj_per_share_away = min(1.5, max(0.0, away_inj_top3_sum / (away_top3_sum + EPS)))
        inj_per_share_diff = inj_per_share_home - inj_per_share_away

        features['inj_per_share|none|top3_sum|home'] = inj_per_share_home
        features['inj_per_share|none|top3_sum|away'] = inj_per_share_away
        features['inj_per_share|none|top3_sum|diff'] = inj_per_share_diff

        # B3: inj_per_share|none|top1_avg - binary indicator if top PER player is injured
        # 1.0 if the team's top PER player is injured, 0.0 otherwise
        home_top1_injured = 0.0
        away_top1_injured = 0.0

        if per_calculator:
            try:
                # Get top-1 player IDs from per_calculator results
                home_per1_player = home_per_features.get('per1_player', []) if home_per_features else []
                away_per1_player = away_per_features.get('per1_player', []) if away_per_features else []

                # Check if top-1 player is in the injured list
                if home_per1_player and len(home_per1_player) > 0:
                    home_top1_id = str(home_per1_player[0].get('player_id', ''))
                    if home_top1_id and home_top1_id in [str(pid) for pid in home_injured_ids]:
                        home_top1_injured = 1.0

                if away_per1_player and len(away_per1_player) > 0:
                    away_top1_id = str(away_per1_player[0].get('player_id', ''))
                    if away_top1_id and away_top1_id in [str(pid) for pid in away_injured_ids]:
                        away_top1_injured = 1.0
            except Exception:
                pass  # If lookup fails, default to 0.0

        features['inj_per_share|none|top1_avg|home'] = home_top1_injured
        features['inj_per_share|none|top1_avg|away'] = away_top1_injured
        features['inj_per_share|none|top1_avg|diff'] = home_top1_injured - away_top1_injured

        # B4: inj_per_weighted_share|none|weighted_MIN - normalized weighted PER lost
        # Numerator: MPG+recency weighted sum of injured players' PER (from injPerValue)
        #   Formula: Σ(PER_i * (mpg_i / max_mpg) * exp(-days_since / k)) for injured rotation players
        # Denominator: Team weighted PER mass using the SAME weighting scheme
        #   Formula: Σ(PER_i * (mpg_i / max_mpg) * exp(-days_since / k)) for ALL rotation players
        # This ensures numerator and denominator are in the same "sum-space" units.
        # Final: inj_per_weighted_share = inj_per_value / (team_weighted_per_mass + eps), clipped to [0, 1.5]
        home_weighted_per_mass = self._get_team_weighted_per_mass(
            HOME, season, game_date, game_date_obj, per_calculator, recency_decay_k
        )
        away_weighted_per_mass = self._get_team_weighted_per_mass(
            AWAY, season, game_date, game_date_obj, per_calculator, recency_decay_k
        )
        inj_per_weighted_share_home = min(1.5, max(0.0, home_inj_per_value / (home_weighted_per_mass + EPS)))
        inj_per_weighted_share_away = min(1.5, max(0.0, away_inj_per_value / (away_weighted_per_mass + EPS)))
        inj_per_weighted_share_diff = inj_per_weighted_share_home - inj_per_weighted_share_away

        features['inj_per_weighted_share|none|weighted_MIN|home'] = inj_per_weighted_share_home
        features['inj_per_weighted_share|none|weighted_MIN|away'] = inj_per_weighted_share_away
        features['inj_per_weighted_share|none|weighted_MIN|diff'] = inj_per_weighted_share_diff

        # =====================================================================
        # NEW STAR-BASED INJURY FEATURES (from player_feature_updates.md)
        # =====================================================================
        # These features use star_score (PER × MPG) instead of just PER
        # to better capture the impact of losing key players.

        # Helper to compute star-based injury features for a team
        def compute_star_injury_features(team, injured_ids, per_features):
            """Compute inj_star_share, inj_star_score_share, inj_top1_star_out for a team."""
            result = {
                'inj_star_share': 0.0,
                'inj_star_score_share': 0.0,
                'inj_top1_star_out': 0.0
            }

            if not per_features:
                return result

            # Get player data from per_features (includes per, mpg)
            players = per_features.get('players', [])
            if not players:
                return result

            # Compute star_score (PER × MPG) for each player
            star_scores = []
            for p in players:
                per = p.get('per', 0)
                mpg = p.get('mpg', 0)
                star_score = per * mpg
                star_scores.append({
                    'player_id': str(p.get('player_id', '')),
                    'star_score': star_score
                })

            # Sort by star_score descending
            star_scores_sorted = sorted(star_scores, key=lambda x: x['star_score'], reverse=True)

            if not star_scores_sorted:
                return result

            # Top-3 team players by star_score
            top3_team = star_scores_sorted[:3]
            top3_team_sum = sum(s['star_score'] for s in top3_team)

            # Top-1 player
            top1_player_id = star_scores_sorted[0]['player_id']
            top1_star_score = star_scores_sorted[0]['star_score']

            # Convert injured_ids to set of strings
            injured_set = {str(pid) for pid in injured_ids} if injured_ids else set()

            # inj_top1_star_out: Binary - is top-1 usage star injured?
            result['inj_top1_star_out'] = 1.0 if top1_player_id in injured_set else 0.0

            # inj_star_share: Injured star's share of top-3 star mass (0 if star not out)
            # Only counts if the TOP-1 player is injured
            if top1_player_id in injured_set and top3_team_sum > EPS:
                result['inj_star_share'] = top1_star_score / (top3_team_sum + EPS)

            # inj_star_score_share: Top-3 injured star mass as share of team top-3
            # Sum star_scores of injured players who are in top-3
            injured_top3_sum = sum(
                s['star_score'] for s in top3_team
                if s['player_id'] in injured_set
            )
            if top3_team_sum > EPS:
                result['inj_star_score_share'] = min(1.5, max(0.0, injured_top3_sum / (top3_team_sum + EPS)))

            return result

        # Compute for home and away teams
        home_star_inj = compute_star_injury_features(HOME, home_injured_ids, home_per_features if per_calculator else None)
        away_star_inj = compute_star_injury_features(AWAY, away_injured_ids, away_per_features if per_calculator else None)

        # inj_star_share|none|raw|{home,away,diff}
        features['inj_star_share|none|raw|home'] = home_star_inj['inj_star_share']
        features['inj_star_share|none|raw|away'] = away_star_inj['inj_star_share']
        features['inj_star_share|none|raw|diff'] = home_star_inj['inj_star_share'] - away_star_inj['inj_star_share']

        # inj_star_score_share|none|top3_sum|{home,away,diff}
        features['inj_star_score_share|none|top3_sum|home'] = home_star_inj['inj_star_score_share']
        features['inj_star_score_share|none|top3_sum|away'] = away_star_inj['inj_star_score_share']
        features['inj_star_score_share|none|top3_sum|diff'] = home_star_inj['inj_star_score_share'] - away_star_inj['inj_star_score_share']

        # inj_top1_star_out|none|raw|{home,away,diff}
        features['inj_top1_star_out|none|raw|home'] = home_star_inj['inj_top1_star_out']
        features['inj_top1_star_out|none|raw|away'] = away_star_inj['inj_top1_star_out']
        features['inj_top1_star_out|none|raw|diff'] = home_star_inj['inj_top1_star_out'] - away_star_inj['inj_top1_star_out']

        # Add injured player names (for display in UI)
        features['home_injured_players'] = home_injured_names
        features['away_injured_players'] = away_injured_names

        return features
    
    def _calculate_team_injury_features(
        self,
        team: str,
        season: str,
        game_date: str,
        game_date_obj: date,
        injured_player_ids: list,
        per_calculator,
        recency_decay_k: float
    ) -> dict:
        """
        Calculate injury features for a single team.
        
        Returns:
            Dict with: injPerValue, injTop1Per, injMinLost, injurySeverity, injRotation
        """
        if not injured_player_ids:
            return {
                'injPerValue': 0.0,
                'injTop1Per': 0.0,
                'injTop3PerSum': 0.0,
                'injMinLost': 0.0,
                'injurySeverity': 0.0,
                'injRotation': 0.0,
                # Player lists for each feature
                'injPerValue_players': [],
                'injTop1Per_players': [],
                'injTop3PerSum_players': [],
                'injMinLost_players': [],
                'injRotation_players': []
            }
        
        # Get player stats (season-to-date, up to but not including game_date)
        player_stats = self._get_player_season_stats(team, season, game_date, injured_player_ids)
        
        if not player_stats:
            return {
                'injPerValue': 0.0,
                'injTop1Per': 0.0,
                'injTop3PerSum': 0.0,
                'injMinLost': 0.0,
                'injurySeverity': 0.0,
                'injRotation': 0.0,
                # Player lists for each feature
                'injPerValue_players': [],
                'injTop1Per_players': [],
                'injTop3PerSum_players': [],
                'injMinLost_players': [],
                'injRotation_players': []
            }
        
        # Filter: only include players whose last game was for this team
        valid_players = []
        for player_id, stats in player_stats.items():
            last_game_team = stats.get('last_game_team')
            if last_game_team == team:
                valid_players.append((player_id, stats))
        
        if not valid_players:
            return {
                'injPerValue': 0.0,
                'injTop1Per': 0.0,
                'injTop3PerSum': 0.0,
                'injMinLost': 0.0,
                'injurySeverity': 0.0,
                'injRotation': 0.0,
                # Player lists for each feature
                'injPerValue_players': [],
                'injTop1Per_players': [],
                'injTop3PerSum_players': [],
                'injMinLost_players': [],
                'injRotation_players': []
            }
        
        # Get PER values if per_calculator is available
        if per_calculator:
            for player_id, stats in valid_players:
                # Get PER for this player (season-to-date)
                per = per_calculator.get_player_per_before_date(
                    player_id, team, season, game_date
                )
                stats['per'] = per if per else 0.0
        
        # Get max MPG on team (for normalization)
        max_mpg = self._get_max_mpg_on_team(team, season, game_date)
        if max_mpg == 0:
            max_mpg = 1.0  # Avoid division by zero
        
        # Calculate rotation players (mpg >= 10)
        rotation_players = [(pid, stats) for pid, stats in valid_players if stats.get('mpg', 0) >= 10]
        
        # Calculate teamRotationMPG (sum of MPG for all rotation players on team)
        team_rotation_mpg = self._get_team_rotation_mpg(team, season, game_date)
        
        # Get player names for injured players (SKIP in batch mode to avoid DB calls)
        player_names = {}
        skip_name_lookup = getattr(self, '_batch_training_mode', False)
        if not skip_name_lookup and self._players_dir_repo is not None:
            try:
                player_id_list = [str(pid) for pid, _ in valid_players]
                if player_id_list:
                    player_docs = self._players_dir_repo.find(
                        {'player_id': {'$in': player_id_list}},
                        projection={'player_id': 1, 'player_name': 1}
                    )
                    player_names = {str(doc['player_id']): doc.get('player_name', 'Unknown') for doc in player_docs}
            except Exception:
                pass
        
        # 1. injPerValue: Weighted average of injured players' PERs
        inj_per_values = []
        inj_per_value_players = []
        for player_id, stats in valid_players:
            per = stats.get('per', 0.0)
            mpg = stats.get('mpg', 0.0)
            last_played_date = stats.get('last_played_date')
            
            if per == 0.0 or mpg == 0.0 or last_played_date is None:
                continue
            
            # MPG weight (normalized 0-1)
            mpg_weight = mpg / max_mpg
            
            # Recency weight
            days_since = (game_date_obj - last_played_date).days
            recency_weight = math.exp(-days_since / recency_decay_k)
            
            # Weighted PER contribution
            weighted_per = per * mpg_weight * recency_weight
            inj_per_values.append(weighted_per)
            
            # Add to player list
            inj_per_value_players.append({
                'player_id': str(player_id),
                'player_name': player_names.get(str(player_id), 'Unknown'),
                'per': per,
                'mpg': mpg,
                'weighted_per': weighted_per
            })
        
        inj_per_value = sum(inj_per_values) if inj_per_values else 0.0
        
        # 2. injTop1Per: Highest injured player PER
        injured_pers_with_players = [(stats.get('per', 0.0), player_id, stats) for player_id, stats in valid_players if stats.get('per', 0.0) > 0]
        injured_pers = [per for per, _, _ in injured_pers_with_players]
        inj_top1_per = max(injured_pers) if injured_pers else 0.0
        
        # Find top 1 player
        inj_top1_per_players = []
        if injured_pers_with_players:
            top1_player = max(injured_pers_with_players, key=lambda x: x[0])
            inj_top1_per_players.append({
                'player_id': str(top1_player[1]),
                'player_name': player_names.get(str(top1_player[1]), 'Unknown'),
                'per': top1_player[0],
                'mpg': top1_player[2].get('mpg', 0.0)
            })
        
        # 2b. injTop3PerSum: Sum of top 3 injured player PERs
        inj_top3_per_sum_players = []
        if injured_pers_with_players:
            sorted_pers_with_players = sorted(injured_pers_with_players, key=lambda x: x[0], reverse=True)
            inj_top3_per_sum = sum([per for per, _, _ in sorted_pers_with_players[:3]])
            # Get top 3 players
            for per, player_id, stats in sorted_pers_with_players[:3]:
                inj_top3_per_sum_players.append({
                    'player_id': str(player_id),
                    'player_name': player_names.get(str(player_id), 'Unknown'),
                    'per': per,
                    'mpg': stats.get('mpg', 0.0)
                })
        else:
            inj_top3_per_sum = 0.0
        
        # 3. injMinLost: Sum of MPG for injured rotation players
        inj_min_lost = sum(stats.get('mpg', 0.0) for _, stats in rotation_players)
        inj_min_lost_players = [{
            'player_id': str(player_id),
            'player_name': player_names.get(str(player_id), 'Unknown'),
            'mpg': stats.get('mpg', 0.0),
            'per': stats.get('per', 0.0)
        } for player_id, stats in rotation_players]
        
        # 4. injurySeverity: injMinLost / teamRotationMPG (uses same players as injMinLost)
        injury_severity = inj_min_lost / team_rotation_mpg if team_rotation_mpg > 0 else 0.0
        
        # 5. injRotation: Count of injured rotation players (uses same players as injMinLost)
        inj_rotation = len(rotation_players)
        inj_rotation_players = [{
            'player_id': str(player_id),
            'player_name': player_names.get(str(player_id), 'Unknown'),
            'mpg': stats.get('mpg', 0.0),
            'per': stats.get('per', 0.0)
        } for player_id, stats in rotation_players]
        
        return {
            'injPerValue': inj_per_value,
            'injTop1Per': inj_top1_per,
            'injTop3PerSum': inj_top3_per_sum,
            'injMinLost': inj_min_lost,
            'injurySeverity': injury_severity,
            'injRotation': float(inj_rotation),
            # Player lists for each feature
            'injPerValue_players': inj_per_value_players,
            'injTop1Per_players': inj_top1_per_players,
            'injTop3PerSum_players': inj_top3_per_sum_players,
            'injMinLost_players': inj_min_lost_players,
            'injRotation_players': inj_rotation_players
        }
    
    def _get_player_season_stats(
        self,
        team: str,
        season: str,
        before_date: str,
        player_ids: list
    ) -> dict:
        """
        Get season-to-date stats for players (MPG, last game date/team).
        Only includes games where stats.min > 0.
        
        Returns:
            Dict mapping player_id -> {
                'mpg': float,
                'per': float (will be 0, filled by caller),
                'last_played_date': date,
                'last_game_team': str
            }
        """
        if not player_ids:
            return {}
        
        # Convert player_ids to set for fast lookup
        player_ids_set = set(str(pid) for pid in player_ids)
        
        # Check cache first
        cache_key = (team, season, before_date)
        if cache_key in self._injury_player_stats_cache:
            # Return cached stats, but filter to only requested player_ids
            cached_stats = self._injury_player_stats_cache[cache_key]
            return {pid: cached_stats[pid] for pid in player_ids_set if pid in cached_stats}
        
        # Use preloaded data if available, otherwise query DB
        # Note: We query ALL players on the team (not just requested ones) to maximize cache reuse
        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            # Filter preloaded records in memory (all players, not just requested ones)
            player_records = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get('date', '') < before_date
            ]
        else:
            # Fallback to DB query - TRACK THIS!
            if not hasattr(self, '_db_fallback_player_stats'):
                self._db_fallback_player_stats = 0
            self._db_fallback_player_stats += 1
            if self._db_fallback_player_stats <= 3:
                print(f"[DB FALLBACK] _get_player_season_stats #{self._db_fallback_player_stats}: team={team}, season={season}, cache_loaded={self._injury_cache_loaded}, key_exists={(team, season) in self._injury_preloaded_players if hasattr(self, '_injury_preloaded_players') else 'NO_ATTR'}")
            player_records = self._players_repo.find(
                {
                    'team': team,
                    'season': season,
                    'date': {'$lt': before_date},
                    'stats.min': {'$gt': 0},  # Only games where player actually played
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'player_id': 1,
                    'team': 1,
                    'date': 1,
                    'stats.min': 1
                },
                sort=[('date', 1)]
            )  # Sort by date ascending
        
        if not player_records:
            # Cache empty result
            self._injury_player_stats_cache[cache_key] = {}
            return {}
        
        # Aggregate by player
        player_agg = defaultdict(lambda: {
            'total_minutes': 0.0,
            'games_played': 0,
            'last_played_date': None,
            'last_game_team': None
        })
        
        for record in player_records:
            player_id = str(record.get('player_id'))
            game_date_str = record.get('date')
            minutes = record.get('stats', {}).get('min', 0.0)
            
            if minutes > 0:
                player_agg[player_id]['total_minutes'] += minutes
                player_agg[player_id]['games_played'] += 1
                
                # Track last game
                game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
                if (player_agg[player_id]['last_played_date'] is None or 
                    game_date > player_agg[player_id]['last_played_date']):
                    player_agg[player_id]['last_played_date'] = game_date
                    player_agg[player_id]['last_game_team'] = record.get('team')
        
        # Calculate MPG
        result = {}
        for player_id, agg in player_agg.items():
            if agg['games_played'] == 0:
                continue
            
            mpg = agg['total_minutes'] / agg['games_played']
            
            result[player_id] = {
                'mpg': mpg,
                'per': 0.0,  # Will be filled by caller if per_calculator provided
                'last_played_date': agg['last_played_date'],
                'last_game_team': agg['last_game_team']
            }
        
        # Cache the full result (for all players on team, not just requested ones)
        # This allows reuse for other player_id lists
        self._injury_player_stats_cache[cache_key] = result
        
        # Return only requested players
        return {pid: result[pid] for pid in player_ids_set if pid in result}
    
    def _get_max_mpg_on_team(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> float:
        """
        Get maximum MPG among all players on the team (season-to-date).
        """
        # Check cache first
        cache_key = (team, season, before_date)
        if cache_key in self._injury_max_mpg_cache:
            return self._injury_max_mpg_cache[cache_key]
        
        # Use preloaded data if available, otherwise query DB
        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            # Filter preloaded records in memory
            all_players = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get('date', '') < before_date
            ]
        else:
            # Fallback to DB query
            all_players = self._players_repo.find(
                {
                    'team': team,
                    'season': season,
                    'date': {'$lt': before_date},
                    'stats.min': {'$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'player_id': 1,
                    'stats.min': 1
                }
            )

        if not all_players:
            self._injury_max_mpg_cache[cache_key] = 0.0
            return 0.0
        
        # Aggregate by player
        player_mpg = defaultdict(lambda: {'total_min': 0.0, 'games': 0})
        for record in all_players:
            player_id = str(record.get('player_id'))
            minutes = record.get('stats', {}).get('min', 0.0)
            player_mpg[player_id]['total_min'] += minutes
            player_mpg[player_id]['games'] += 1
        
        # Calculate MPG for each player and find max
        max_mpg = 0.0
        for player_id, agg in player_mpg.items():
            if agg['games'] > 0:
                mpg = agg['total_min'] / agg['games']
                max_mpg = max(max_mpg, mpg)
        
        # Cache result
        self._injury_max_mpg_cache[cache_key] = max_mpg
        return max_mpg
    
    def _get_team_rotation_mpg(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> float:
        """
        Get sum of MPG for all rotation players (mpg >= 10) on the team.
        """
        # Check cache first
        cache_key = (team, season, before_date)
        if cache_key in self._injury_rotation_mpg_cache:
            return self._injury_rotation_mpg_cache[cache_key]

        # Use preloaded data if available, otherwise query DB
        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            # Filter preloaded records in memory
            all_players = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get('date', '') < before_date
            ]
        else:
            # Fallback to DB query - TRACK THIS!
            if not hasattr(self, '_db_fallback_rotation_mpg'):
                self._db_fallback_rotation_mpg = 0
            self._db_fallback_rotation_mpg += 1
            if self._db_fallback_rotation_mpg <= 3:
                print(f"[DB FALLBACK] _get_team_rotation_mpg #{self._db_fallback_rotation_mpg}: team={team}, season={season}, cache_loaded={self._injury_cache_loaded}, key_exists={(team, season) in self._injury_preloaded_players if hasattr(self, '_injury_preloaded_players') else 'NO_ATTR'}")
            all_players = self._players_repo.find(
                {
                    'team': team,
                    'season': season,
                    'date': {'$lt': before_date},
                    'stats.min': {'$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'player_id': 1,
                    'stats.min': 1
                }
            )

        if not all_players:
            self._injury_rotation_mpg_cache[cache_key] = 0.0
            return 0.0
        
        # Aggregate by player
        player_mpg = defaultdict(lambda: {'total_min': 0.0, 'games': 0})
        for record in all_players:
            player_id = str(record.get('player_id'))
            minutes = record.get('stats', {}).get('min', 0.0)
            player_mpg[player_id]['total_min'] += minutes
            player_mpg[player_id]['games'] += 1
        
        # Calculate MPG and sum for rotation players (mpg >= 10)
        total_rotation_mpg = 0.0
        for player_id, agg in player_mpg.items():
            if agg['games'] > 0:
                mpg = agg['total_min'] / agg['games']
                if mpg >= 10.0:
                    total_rotation_mpg += mpg
        
        # Cache result
        self._injury_rotation_mpg_cache[cache_key] = total_rotation_mpg
        return total_rotation_mpg

    def _get_team_weighted_per_mass(
        self,
        team: str,
        season: str,
        game_date: str,
        game_date_obj: date,
        per_calculator,
        recency_decay_k: float = 15.0
    ) -> float:
        """
        Compute team's weighted PER mass using the same weighting scheme as injPerValue.

        Formula: Σ(PER_i * (mpg_i / max_mpg) * exp(-days_since / recency_decay_k))
        over rotation players (MPG >= 10) whose last game was for this team.

        This provides the denominator for inj_per_weighted_share, ensuring
        numerator and denominator are in the same "sum-space" units.

        Args:
            team: Team name
            season: Season string
            game_date: Game date string (YYYY-MM-DD)
            game_date_obj: Game date as date object
            per_calculator: PERCalculator instance for getting player PERs
            recency_decay_k: Decay constant for recency weighting (default 15)

        Returns:
            float: Team weighted PER mass
        """
        if per_calculator is None:
            return 0.0

        # Check cache first
        cache_key = (team, season, game_date, recency_decay_k)
        if not hasattr(self, '_team_weighted_per_mass_cache'):
            self._team_weighted_per_mass_cache = {}
        if cache_key in self._team_weighted_per_mass_cache:
            return self._team_weighted_per_mass_cache[cache_key]

        # Get max MPG on team (same as used for injury numerator)
        max_mpg = self._get_max_mpg_on_team(team, season, game_date)
        if max_mpg == 0:
            max_mpg = 1.0  # Avoid division by zero

        # Get all player stats from cache (warm cache if needed)
        # We pass a dummy ID just to trigger the cache population
        player_stats_cache_key = (team, season, game_date)
        if player_stats_cache_key not in self._injury_player_stats_cache:
            # Warm the cache by calling _get_player_season_stats with a query that
            # triggers the full team data load
            self._warm_player_stats_cache(team, season, game_date)

        # Get all player stats from cache
        all_player_stats = self._injury_player_stats_cache.get(player_stats_cache_key, {})

        if not all_player_stats:
            self._team_weighted_per_mass_cache[cache_key] = 0.0
            return 0.0

        # Filter to rotation players (MPG >= 10) whose last game was for this team
        rotation_players = []
        for player_id, stats in all_player_stats.items():
            mpg = stats.get('mpg', 0.0)
            last_game_team = stats.get('last_game_team')
            if mpg >= 10.0 and last_game_team == team:
                rotation_players.append((player_id, stats))

        if not rotation_players:
            self._team_weighted_per_mass_cache[cache_key] = 0.0
            return 0.0

        # Calculate weighted PER mass
        weighted_per_mass = 0.0
        for player_id, stats in rotation_players:
            mpg = stats.get('mpg', 0.0)
            last_played_date = stats.get('last_played_date')

            if mpg == 0.0 or last_played_date is None:
                continue

            # Get PER for this player
            per = per_calculator.get_player_per_before_date(
                player_id, team, season, game_date
            )
            if per is None or per <= 0:
                continue

            # MPG weight (normalized 0-1, same as injury numerator)
            mpg_weight = mpg / max_mpg

            # Recency weight (exponential decay, same as injury numerator)
            days_since = (game_date_obj - last_played_date).days
            recency_weight = math.exp(-days_since / recency_decay_k)

            # Add weighted contribution
            weighted_per_mass += per * mpg_weight * recency_weight

        # Cache result
        self._team_weighted_per_mass_cache[cache_key] = weighted_per_mass
        return weighted_per_mass

    def _warm_player_stats_cache(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> None:
        """
        Warm the player stats cache for a team/season/date.
        This ensures _injury_player_stats_cache is populated for all team players.
        """
        cache_key = (team, season, before_date)
        if cache_key in self._injury_player_stats_cache:
            return  # Already cached

        # Use preloaded data if available, otherwise query DB
        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            player_records = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get('date', '') < before_date
            ]
        else:
            # Fallback to DB query
            player_records = self._players_repo.find(
                {
                    'team': team,
                    'season': season,
                    'date': {'$lt': before_date},
                    'stats.min': {'$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'player_id': 1,
                    'team': 1,
                    'date': 1,
                    'stats.min': 1
                },
                sort=[('date', 1)]
            )

        if not player_records:
            self._injury_player_stats_cache[cache_key] = {}
            return

        # Aggregate by player
        player_agg = defaultdict(lambda: {
            'total_minutes': 0.0,
            'games_played': 0,
            'last_played_date': None,
            'last_game_team': None
        })

        for record in player_records:
            player_id = str(record.get('player_id'))
            game_date_str = record.get('date')
            minutes = record.get('stats', {}).get('min', 0.0)

            if minutes > 0:
                player_agg[player_id]['total_minutes'] += minutes
                player_agg[player_id]['games_played'] += 1

                # Track last game
                game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
                if (player_agg[player_id]['last_played_date'] is None or
                    game_date > player_agg[player_id]['last_played_date']):
                    player_agg[player_id]['last_played_date'] = game_date
                    player_agg[player_id]['last_game_team'] = record.get('team')

        # Calculate MPG
        result = {}
        for player_id, agg in player_agg.items():
            if agg['games_played'] == 0:
                continue

            mpg = agg['total_minutes'] / agg['games_played']

            result[player_id] = {
                'mpg': mpg,
                'per': 0.0,
                'last_played_date': agg['last_played_date'],
                'last_game_team': agg['last_game_team']
            }

        # Cache the result
        self._injury_player_stats_cache[cache_key] = result

    def _get_season_injury_severity(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> float:
        """
        Compute season-to-date injury severity as weighted ratio-of-sums.

        Formula: Σ(inj_min_lost_k) / Σ(team_rotation_mpg_k + ε)

        Where k iterates over all prior team games in the season.
        This gives "fraction of rotation minutes lost to injury" averaged over the season,
        weighted by game rotation minutes (more stable than simple mean).

        IMPORTANT: Uses preloaded game data (self.games_home/games_away) when available
        to avoid DB calls during iteration. This is critical for the chunked async
        populate_master_training_cols workflow.

        Args:
            team: Team name
            season: Season string
            before_date: Game date string (YYYY-MM-DD) - compute for games before this date

        Returns:
            float: Season-to-date injury severity (0.0 if no prior games)
        """
        EPS = 1e-6

        # Check cache first
        cache_key = (team, season, before_date, 'season_inj_severity')
        if not hasattr(self, '_season_injury_severity_cache'):
            self._season_injury_severity_cache = {}
        if cache_key in self._season_injury_severity_cache:
            return self._season_injury_severity_cache[cache_key]

        # Get all prior games for this team in the season
        # CRITICAL: Use preloaded data when available to avoid DB calls
        prior_games = []

        if self.games_home is not None and self.games_away is not None:
            # Use team index for O(log N) bisect lookup (no DB calls!)
            # Pass empty exclude list — injury severity needs all game types, then filter completed
            all_games = self._get_team_games_in_range(team, season, end_date_str=before_date, exclude_game_types=[])
            prior_games = [g for g in all_games if g.get('homeWon') is not None]
        else:
            # Fallback to DB query - TRACK THIS!
            if not hasattr(self, '_db_fallback_season_severity'):
                self._db_fallback_season_severity = 0
            self._db_fallback_season_severity += 1
            if self._db_fallback_season_severity <= 3:
                print(f"[DB FALLBACK] _get_season_injury_severity #{self._db_fallback_season_severity}: team={team}, season={season}, games_home_set={self.games_home is not None}, games_away_set={self.games_away is not None}")

            if self._games_repo is None:
                self._season_injury_severity_cache[cache_key] = 0.0
                return 0.0

            prior_games = list(self._games_repo.find(
                {
                    'season': season,
                    'date': {'$lt': before_date},
                    '$or': [
                        {'homeTeam.name': team},
                        {'awayTeam.name': team}
                    ],
                    'homeWon': {'$exists': True},
                    'game_type': {'$nin': self._exclude_game_types}
                },
                projection={
                    'date': 1,
                    'homeTeam.name': 1,
                    'homeTeam.injured_players': 1,
                    'awayTeam.name': 1,
                    'awayTeam.injured_players': 1
                }
            ))

        if not prior_games:
            self._season_injury_severity_cache[cache_key] = 0.0
            return 0.0

        # Accumulate sums across all prior games
        total_inj_min_lost = 0.0
        total_rotation_mpg = 0.0

        for game in prior_games:
            game_date = game.get('date')
            if not game_date:
                continue

            # Determine if team is home or away
            home_team = game.get('homeTeam', {}).get('name')
            is_home = (home_team == team)

            # Get injured player IDs for this team in this game
            if is_home:
                injured_ids = game.get('homeTeam', {}).get('injured_players', [])
            else:
                injured_ids = game.get('awayTeam', {}).get('injured_players', [])

            injured_ids = [str(pid) for pid in injured_ids] if injured_ids else []

            # Get team rotation MPG for this game date (uses cache, no DB call if warmed)
            game_rotation_mpg = self._get_team_rotation_mpg(team, season, game_date)
            total_rotation_mpg += game_rotation_mpg

            if not injured_ids:
                # No injuries this game, inj_min_lost = 0
                continue

            # Get player stats for injured players (uses cache, no DB call if warmed)
            player_stats = self._get_player_season_stats(team, season, game_date, injured_ids)

            # Sum MPG for injured rotation players (same logic as inj_min_lost)
            game_inj_min_lost = 0.0
            for player_id, stats in player_stats.items():
                mpg = stats.get('mpg', 0.0)
                last_game_team = stats.get('last_game_team')
                # Only count rotation players whose last game was for this team
                if mpg >= 10.0 and last_game_team == team:
                    game_inj_min_lost += mpg

            total_inj_min_lost += game_inj_min_lost

        # Compute weighted ratio
        if total_rotation_mpg <= 0:
            result = 0.0
        else:
            result = total_inj_min_lost / (total_rotation_mpg + EPS)

        # Cache and return
        self._season_injury_severity_cache[cache_key] = result
        return result
