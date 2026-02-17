"""
Shared feature context for parallel training data generation.

Pre-loads all necessary data ONCE, then shares across worker threads.
This avoids each thread creating its own MongoDB connections and
loading the same 500K+ records.

This module is extracted from cli/populate_master_training_cols.py
and made league-aware using LeagueConfig.
"""

from typing import List, Dict, Optional, Any
import threading

from bball_app.core.league_config import LeagueConfig


class SharedFeatureContext:
    """
    Pre-loads all necessary data for feature calculation ONCE, then shares
    across multiple worker threads. This avoids each thread creating its own
    MongoDB connections and loading the same 500K+ records.

    Usage:
        # Main thread - create once
        context = SharedFeatureContext(feature_names, league_config)

        # Worker threads - use shared context (read-only)
        features = context.calculate_features_for_row(row_data)
    """

    def __init__(
        self,
        feature_names: List[str],
        league_config: LeagueConfig,
        preload_games: bool = True,
        preload_venues: bool = True,
        preload_per_cache: bool = True,
        preload_injury_cache: bool = True,
        preload_seasons: List[str] = None,
    ):
        """
        Initialize shared feature context by pre-loading all necessary data.

        Args:
            feature_names: List of feature names that will be calculated
            league_config: League configuration (collection names, etc.)
            preload_games: If True, preload games into memory
            preload_venues: If True, preload venue cache
            preload_per_cache: If True, preload PER player stats
            preload_injury_cache: If True, preload injury data
            preload_seasons: Optional list of seasons to preload (e.g., ['2023-2024']).
                           If None, loads all seasons.
        """
        self.preload_seasons = preload_seasons
        from bball_app.core.mongo import Mongo
        from bball_app.core.stats.handler import StatHandlerV2
        from bball_app.core.stats.per_calculator import PERCalculator
        from bball_app.core.data import GamesRepository, RostersRepository
        from bball_app.core.utils.collection import import_collection

        self.feature_names = feature_names
        self.league_config = league_config

        # Get collection names from league config
        games_collection = league_config.collections.get('games', 'stats_nba')
        teams_collection = league_config.collections.get('teams', 'nba_teams')

        # Infer what components are needed based on feature names
        self._needs_per = any(
            f.startswith("player_") or f.startswith("per_available") or
            f.split("|", 1)[0].lower().endswith("_per")
            for f in feature_names
        )
        self._needs_injuries = any(f.startswith("inj_") for f in feature_names)
        self._needs_elo = any(f.split("|", 1)[0].lower().startswith("elo") for f in feature_names)

        print("=" * 60)
        print(f"INITIALIZING SHARED FEATURE CONTEXT ({league_config.league_id.upper()})")
        print("=" * 60)

        # Single MongoDB connection
        print("Connecting to MongoDB...")
        self.mongo = Mongo()
        self.db = self.mongo.db
        print("Connected to MongoDB.")

        # Build team name normalization map (displayName -> abbreviation)
        self._team_name_map = {}
        self._normalization_count = 0
        self._normalization_examples = set()
        try:
            teams_coll = self.db[teams_collection]
            for team in teams_coll.find({}, {'displayName': 1, 'abbreviation': 1}):
                display_name = team.get('displayName', '')
                abbr = team.get('abbreviation', '')
                if display_name and abbr:
                    self._team_name_map[display_name] = abbr
                    self._team_name_map[display_name.lower()] = abbr
            print(f"Loaded {len(self._team_name_map) // 2} team name mappings.")
        except Exception as e:
            print(f"Warning: Could not load team name mappings: {e}")

        # Initialize repositories
        self._games_repo = GamesRepository(self.db, collection_name=games_collection)
        self._rosters_repo = RostersRepository(self.db, collection_name=league_config.collections.get('rosters', 'nba_rosters'))

        # Load game data once
        self.all_games = None
        if preload_games:
            # Build season-filtered query if seasons provided
            query = {'season': {'$exists': True}}
            if self.preload_seasons:
                query['season'] = {'$in': self.preload_seasons}
                print(f"Loading game data for seasons: {self.preload_seasons}")
            else:
                print("Loading ALL game data from database (no season filter)...")
            self.all_games = import_collection(games_collection, query=query)
            print("Loaded game data.")

        # Initialize stat handler with preloaded games
        print("Initializing shared stat handler...")
        self.stat_handler = StatHandlerV2(
            statistics=[],
            use_exponential_weighting=False,
            preloaded_games=self.all_games,
            db=self.db,
            lazy_load=(not preload_games),
            league=league_config,
        )
        # Set batch training mode to skip DB lookups for roster/player names
        # These lookups are not needed during training - we use preloaded data only
        self.stat_handler._batch_training_mode = True

        # Preload venue cache
        if preload_venues:
            print("Preloading venue cache...")
            try:
                self.stat_handler.preload_venue_cache()
            except Exception:
                pass

        # Initialize PER calculator if needed
        self.per_calculator = None
        if (self._needs_per or self._needs_injuries) and preload_per_cache:
            # Ensure league stats cache exists for each season (required for PER constants)
            from bball_app.core.stats.league_cache import ensure_season_cached
            seasons_to_cache = self.preload_seasons or []
            if seasons_to_cache:
                print(f"Ensuring league stats cached for {len(seasons_to_cache)} season(s)...")
                for season in seasons_to_cache:
                    ensure_season_cached(season, self.db, league=league_config)

            if self.preload_seasons:
                print(f"Initializing shared PER calculator (preloading seasons: {self.preload_seasons})...")
            else:
                print("Initializing shared PER calculator (preloading all seasons)...")
            self.per_calculator = PERCalculator(
                self.db,
                preload=preload_per_cache,
                league=league_config,
                preload_seasons=self.preload_seasons
            )

        # Preload injury cache if needed
        if self._needs_injuries and preload_injury_cache and self.all_games:
            print("Preloading injury cache (player stats by team-season)...")
            games_list = []
            games_home, games_away = self.all_games
            for season_data in games_home.values():
                for date_data in season_data.values():
                    for game in date_data.values():
                        games_list.append(game)
            self.stat_handler.preload_injury_cache(games_list)

            # Precompute season severity values (O(G) instead of O(G²))
            print("Precomputing season injury severity values...")
            self._precomputed_season_severity = self.stat_handler.precompute_season_severity(games_list)

        # Preload elo cache if elo features are needed
        if self._needs_elo:
            print("Preloading elo ratings cache...")
            try:
                from bball_app.core.stats.elo_cache import EloCache
                self._elo_cache = EloCache(self.db, league=league_config)
                self._elo_cache.preload(seasons=self.preload_seasons)
                # Inject into stat handler so it uses the preloaded cache
                self.stat_handler._elo_cache = self._elo_cache
            except Exception as e:
                print(f"Warning: Could not preload elo cache: {e}")

        # Initialize season severity cache if not already precomputed
        if not hasattr(self, '_precomputed_season_severity'):
            self._precomputed_season_severity = {}

        # Pre-load venue_guids for travel features (will be populated per batch)
        self.venue_guid_cache = {}

        # Thread lock for cache updates
        self._cache_lock = threading.Lock()

        print("=" * 60)
        print("SHARED CONTEXT READY - Workers will use cached data")
        print("=" * 60)

    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to abbreviation format.

        Handles:
        - Full names: "Milwaukee Bucks" -> "MIL"
        - Already abbreviated: "MIL" -> "MIL" (no change)
        - Case insensitive: "milwaukee bucks" -> "MIL"

        Returns original if no mapping found.
        """
        if not team_name:
            return team_name

        if team_name in self._team_name_map:
            abbr = self._team_name_map[team_name]
            self._normalization_count += 1
            if len(self._normalization_examples) < 5:
                self._normalization_examples.add(f"{team_name} -> {abbr}")
            return abbr

        if team_name.lower() in self._team_name_map:
            abbr = self._team_name_map[team_name.lower()]
            self._normalization_count += 1
            if len(self._normalization_examples) < 5:
                self._normalization_examples.add(f"{team_name} -> {abbr}")
            return abbr

        return team_name

    def print_normalization_stats(self):
        """Print statistics about team name normalizations."""
        if self._normalization_count > 0:
            print(f"\n[DIAGNOSTIC] Team name normalizations: {self._normalization_count}")
            print(f"  Examples: {list(self._normalization_examples)}")
            print("  This confirms team names in CSV needed conversion from full names to abbreviations.")
        else:
            print("\n[DIAGNOSTIC] No team name normalizations needed (all names already abbreviated).")

    def preload_venue_guids(self, game_ids: List[str]):
        """
        Pre-load venue GUIDs for a batch of games.
        Called once per batch, not per row.
        """
        if not game_ids:
            return

        games_collection = self.league_config.collections.get('games', 'stats_nba')
        games_coll = self.db[games_collection]

        games_with_venue = games_coll.find(
            {'game_id': {'$in': game_ids}},
            {'game_id': 1, 'venue_guid': 1}
        )
        with self._cache_lock:
            for g in games_with_venue:
                if g.get('venue_guid'):
                    self.venue_guid_cache[str(g['game_id'])] = g['venue_guid']

    def calculate_features_for_row(
        self,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        game_id: str = None,
        venue_guid: str = None,
        existing_row_data: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate features for a single row using the shared pre-loaded data.
        This method is thread-safe for read operations.

        Args:
            home_team: Home team abbreviation or full name
            away_team: Away team abbreviation or full name
            season: Season string (e.g., "2023-2024")
            year, month, day: Game date components
            game_id: Optional game ID for venue lookup
            venue_guid: Optional venue GUID for travel features
            existing_row_data: Optional dict of existing column values from the row

        Returns:
            Dict mapping feature names to their values
        """
        features_dict = {}
        game_date_str = f"{year}-{month:02d}-{day:02d}"

        # Normalize team names to abbreviations
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)

        # Use cached venue_guid if available
        if venue_guid is None and game_id:
            venue_guid = self.venue_guid_cache.get(str(game_id))

        # Calculate each feature
        for feature_name in self.feature_names:
            # Handle special non-pipe features
            if '|' not in feature_name:
                if feature_name == 'SeasonStartYear':
                    features_dict[feature_name] = int(year) if int(month) >= 10 else int(year) - 1
                elif feature_name == 'Year':
                    features_dict[feature_name] = int(year)
                elif feature_name == 'Month':
                    features_dict[feature_name] = int(month)
                elif feature_name == 'Day':
                    features_dict[feature_name] = int(day)
                elif feature_name == 'pred_margin':
                    features_dict[feature_name] = 0.0
                else:
                    features_dict[feature_name] = 0.0
                continue

            # Skip PER and injury features - handled separately below
            if feature_name.startswith('player_') or feature_name.startswith('per_available'):
                continue
            if feature_name.startswith('inj_'):
                continue

            # Regular stat feature - use shared stat handler
            try:
                value = self.stat_handler.calculate_feature(
                    feature_name, home_team, away_team, season,
                    year, month, day, self.per_calculator, venue_guid
                )
                features_dict[feature_name] = value if value is not None else 0.0
            except Exception:
                features_dict[feature_name] = 0.0

        # Add PER features if needed
        has_per_features = any(
            fname.startswith('player_') or fname.startswith('per_available')
            for fname in self.feature_names
        )
        if has_per_features and self.per_calculator:
            injured_players_dict = None
            try:
                # Use preloaded games cache - NO DB CALLS in row iterations
                game_doc = None
                if self.all_games is not None:
                    games_home, games_away = self.all_games
                    if season in games_home and game_date_str in games_home[season]:
                        if home_team in games_home[season][game_date_str]:
                            game_doc = games_home[season][game_date_str][home_team]
                if game_doc:
                    home_injured = game_doc.get('homeTeam', {}).get('injured_players', [])
                    away_injured = game_doc.get('awayTeam', {}).get('injured_players', [])
                    if home_injured or away_injured:
                        injured_players_dict = {
                            home_team: [str(pid) for pid in home_injured] if home_injured else [],
                            away_team: [str(pid) for pid in away_injured] if away_injured else []
                        }
            except Exception:
                pass

            try:
                per_features = self.per_calculator.get_game_per_features(
                    home_team=home_team,
                    away_team=away_team,
                    season=season,
                    game_date=game_date_str,
                    player_filters=None,
                    injured_players=injured_players_dict,
                    game_id=game_id  # Enable cross-team aggregation for traded players
                )
                if per_features:
                    for fname in self.feature_names:
                        if fname.startswith('player_') or fname.startswith('per_available'):
                            if fname in per_features:
                                features_dict[fname] = per_features[fname]
                            else:
                                features_dict[fname] = 0.0
            except Exception:
                for fname in self.feature_names:
                    if fname.startswith('player_') or fname.startswith('per_available'):
                        features_dict[fname] = 0.0

        # Add injury features if needed
        has_injury_features = any(fname.startswith('inj_') for fname in self.feature_names)
        if has_injury_features:
            try:
                # Use preloaded games cache - NO DB CALLS in row iterations
                game_doc = None
                if self.all_games is not None:
                    games_home, games_away = self.all_games
                    if season in games_home and game_date_str in games_home[season]:
                        if home_team in games_home[season][game_date_str]:
                            game_doc = games_home[season][game_date_str][home_team]

                injury_features = self.stat_handler.get_injury_features(
                    home_team, away_team, season, year, month, day,
                    game_doc=game_doc,
                    per_calculator=self.per_calculator,
                    precomputed_season_severity=self._precomputed_season_severity
                )
                if injury_features:
                    for fname in self.feature_names:
                        if fname.startswith('inj_'):
                            if fname in injury_features:
                                features_dict[fname] = injury_features[fname]
                            else:
                                features_dict[fname] = 0.0

                # Handle share features from existing raw values
                if existing_row_data and self.per_calculator:
                    self._calculate_share_features(
                        features_dict, existing_row_data,
                        home_team, away_team, season, game_date_str
                    )

            except Exception:
                for fname in self.feature_names:
                    if fname.startswith('inj_'):
                        features_dict[fname] = 0.0

        # Ensure all requested features have a value
        for fname in self.feature_names:
            if fname not in features_dict:
                features_dict[fname] = 0.0

        return features_dict

    def _calculate_share_features(
        self,
        features_dict: Dict[str, float],
        existing_row_data: Dict[str, float],
        home_team: str,
        away_team: str,
        season: str,
        game_date_str: str,
    ):
        """Calculate share features from existing raw values."""
        share_features_to_fix = [
            ('inj_per_share|none|top3_sum|home', 'inj_per|none|top3_sum|home'),
            ('inj_per_share|none|top3_sum|away', 'inj_per|none|top3_sum|away'),
            ('inj_per_weighted_share|none|weighted_MIN|home', 'inj_per|none|weighted_MIN|home'),
            ('inj_per_weighted_share|none|weighted_MIN|away', 'inj_per|none|weighted_MIN|away'),
        ]

        home_top3_sum = 0.0
        away_top3_sum = 0.0
        home_per_weighted = 0.0
        away_per_weighted = 0.0
        denominators_fetched = False

        for share_feature, raw_feature in share_features_to_fix:
            if share_feature not in self.feature_names:
                continue

            current_share = features_dict.get(share_feature, 0.0)
            existing_raw = existing_row_data.get(raw_feature, 0.0)

            if current_share == 0.0 and existing_raw != 0.0:
                if not denominators_fetched:
                    try:
                        home_per = self.per_calculator.compute_team_per_features(
                            home_team, season, game_date_str
                        )
                        away_per = self.per_calculator.compute_team_per_features(
                            away_team, season, game_date_str
                        )
                        if home_per:
                            home_top3_sum = home_per.get('top3_sum', 0.0)
                            home_per_weighted = home_per.get('per_weighted', 0.0)
                        if away_per:
                            away_top3_sum = away_per.get('top3_sum', 0.0)
                            away_per_weighted = away_per.get('per_weighted', 0.0)
                    except Exception:
                        pass
                    denominators_fetched = True

                EPS = 1e-6
                if 'top3_sum|home' in share_feature:
                    if home_top3_sum > 0:
                        features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (home_top3_sum + EPS)))
                elif 'top3_sum|away' in share_feature:
                    if away_top3_sum > 0:
                        features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (away_top3_sum + EPS)))
                elif 'weighted_MIN|home' in share_feature:
                    if home_per_weighted > 0:
                        features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (home_per_weighted + EPS)))
                elif 'weighted_MIN|away' in share_feature:
                    if away_per_weighted > 0:
                        features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (away_per_weighted + EPS)))

        # Calculate diff features from home/away
        if 'inj_per_share|none|top3_sum|diff' in self.feature_names:
            home_share = features_dict.get('inj_per_share|none|top3_sum|home', 0.0)
            away_share = features_dict.get('inj_per_share|none|top3_sum|away', 0.0)
            features_dict['inj_per_share|none|top3_sum|diff'] = home_share - away_share

        if 'inj_per_weighted_share|none|weighted_MIN|diff' in self.feature_names:
            home_share = features_dict.get('inj_per_weighted_share|none|weighted_MIN|home', 0.0)
            away_share = features_dict.get('inj_per_weighted_share|none|weighted_MIN|away', 0.0)
            features_dict['inj_per_weighted_share|none|weighted_MIN|diff'] = home_share - away_share
