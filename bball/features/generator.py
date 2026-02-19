"""
Shared Feature Generator for efficient feature computation.

This module provides a SharedFeatureGenerator class that generates features once
for a given game, avoiding redundant DB queries when multiple models need features
from the same game (e.g., ensemble base models).

Key benefits:
- Single BasketballFeatureComputer for regular stat features
- Single PER calculator instance
- InjuryFeatureCalculator for injury feature computation
- Generates superset of features needed by all models
- Each model extracts its subset from the shared feature dict

Uses data layer repositories for all database operations.
"""

from typing import Dict, List, Optional, Set
from datetime import datetime

from bball.stats.per_calculator import PERCalculator
from bball.features.compute import BasketballFeatureComputer
from bball.features.injury import InjuryFeatureCalculator
from bball.features.parser import parse_feature_name
from bball.data import GamesRepository, RostersRepository


class SharedFeatureGenerator:
    """
    Generates features for a game using shared infrastructure.

    Designed for efficiency when multiple models need features from the same game.
    Instead of each model creating its own stat handler and querying the DB,
    this class generates all needed features once.

    Usage:
        generator = SharedFeatureGenerator(db)

        # Collect all features needed by all models
        all_features = set(model1_features) | set(model2_features) | set(model3_features)

        # Generate features once
        feature_dict = generator.generate_features(
            feature_names=list(all_features),
            home_team="BOS",
            away_team="LAL",
            season="2024-2025",
            game_date="2025-01-15",
            player_filters=player_filters
        )

        # Extract subset for each model
        model1_values = [feature_dict.get(f, 0.0) for f in model1_features]
    """

    def __init__(self, db, preload_venues: bool = True, league=None):
        """
        Initialize the shared feature generator.

        Args:
            db: MongoDB database connection
            preload_venues: Whether to preload venue cache for travel features
            league: Optional LeagueConfig for league-aware collection routing
        """
        self.db = db
        self.league = league
        # Initialize repositories
        self._games_repo = GamesRepository(db, league=league)
        self._rosters_repo = RostersRepository(db, league=league)

        # BasketballFeatureComputer handles all regular stat features
        self._computer = BasketballFeatureComputer(db=db, league=league)

        # Preload venue cache for travel distance features
        if preload_venues:
            try:
                self._computer.preload_venue_cache()
            except Exception:
                pass

        # InjuryFeatureCalculator for injury features
        self._injury_calculator = InjuryFeatureCalculator(db=db, league=league)

        # Create shared PER calculator (no preload - queries on demand)
        self.per_calculator = PERCalculator(db, preload=False, league=league)

        # Track player lists for UI display
        self._per_player_lists: Dict = {}
        self._injury_player_lists: Dict = {}

        # Prediction context reference (set via set_prediction_context)
        self._prediction_context = None

    def set_prediction_context(self, context) -> None:
        """
        Inject preloaded prediction context to avoid per-feature DB calls.

        Args:
            context: PredictionContext instance with preloaded data:
                - games_home: Dict[season][date][team] -> game_doc
                - games_away: Dict[season][date][team] -> game_doc
                - player_stats: Dict[(team, season)] -> [player_records]
                - venue_cache: Dict[venue_guid] -> (lat, lon)
        """
        if context is None:
            print("[SharedFeatureGenerator] set_prediction_context called with None context")
            return

        self._prediction_context = context
        print(f"[SharedFeatureGenerator] Injecting context with {len(context.player_stats)} team-season keys")

        # Inject into BasketballFeatureComputer (regular features)
        self._computer.set_preloaded_data(
            context.games_home, context.games_away,
            venue_cache=context.venue_cache,
        )

        # Inject Elo cache (matches training path in shared_context.py)
        if hasattr(context, 'elo_cache') and context.elo_cache is not None:
            self._computer._elo_cache = context.elo_cache
            print(f"[SharedFeatureGenerator] Injected elo_cache into BasketballFeatureComputer")

        # Inject into InjuryFeatureCalculator
        if self._injury_calculator:
            self._injury_calculator.set_preloaded_data(context.games_home, context.games_away)
            # Inject injury preloaded players cache
            if context.player_stats:
                player_stats = dict(context.player_stats) if hasattr(context.player_stats, 'items') else context.player_stats
                self._injury_calculator._injury_preloaded_players = player_stats
                self._injury_calculator._injury_cache_loaded = True
                print(f"[SharedFeatureGenerator] Set _injury_calculator._injury_cache_loaded=True, {len(player_stats)} keys")

        # Inject into per_calculator
        if self.per_calculator:
            player_stats = dict(context.player_stats) if hasattr(context.player_stats, 'items') else context.player_stats
            self.per_calculator._player_stats_cache = player_stats
            self.per_calculator._preloaded = True
            print(f"[SharedFeatureGenerator] Set per_calculator._preloaded=True, {len(player_stats)} keys")

            # CRITICAL: Also build _team_stats_cache from games data for PER calculations
            # Without this, compute_team_per_features falls back to slow DB queries
            from collections import defaultdict
            team_stats_cache = defaultdict(list)

            for season_key in context.games_home:
                for date_key in context.games_home[season_key]:
                    for team_name, game_doc in context.games_home[season_key][date_key].items():
                        home_team_data = game_doc.get('homeTeam', {})
                        team_stats_cache[(team_name, season_key)].append({
                            'game_id': game_doc.get('game_id'),
                            'date': game_doc.get('date'),
                            'team_data': {
                                'assists': home_team_data.get('assists', 0),
                                'FG_made': home_team_data.get('FG_made', 0),
                                'FG_att': home_team_data.get('FG_att', 0),
                                'FT_made': home_team_data.get('FT_made', 0),
                                'FT_att': home_team_data.get('FT_att', 0),
                                'total_reb': home_team_data.get('total_reb', 0),
                                'off_reb': home_team_data.get('off_reb', 0),
                                'TO': home_team_data.get('TO', 0),
                            }
                        })

            for season_key in context.games_away:
                for date_key in context.games_away[season_key]:
                    for team_name, game_doc in context.games_away[season_key][date_key].items():
                        away_team_data = game_doc.get('awayTeam', {})
                        team_stats_cache[(team_name, season_key)].append({
                            'game_id': game_doc.get('game_id'),
                            'date': game_doc.get('date'),
                            'team_data': {
                                'assists': away_team_data.get('assists', 0),
                                'FG_made': away_team_data.get('FG_made', 0),
                                'FG_att': away_team_data.get('FG_att', 0),
                                'FT_made': away_team_data.get('FT_made', 0),
                                'FT_att': away_team_data.get('FT_att', 0),
                                'total_reb': away_team_data.get('total_reb', 0),
                                'off_reb': away_team_data.get('off_reb', 0),
                                'TO': away_team_data.get('TO', 0),
                            }
                        })

            # Sort each team's games by date
            for key in team_stats_cache:
                team_stats_cache[key].sort(key=lambda x: x['date'])

            self.per_calculator._team_stats_cache = team_stats_cache
            print(f"[SharedFeatureGenerator] Built _team_stats_cache with {len(team_stats_cache)} team-season keys")

    def generate_features(
        self,
        feature_names: List[str],
        home_team: str,
        away_team: str,
        season: str,
        game_date: str,
        player_filters: Dict,
        additional_features: Optional[Dict] = None,
        recency_decay_k: float = 15.0,
        venue_guid: Optional[str] = None,
        game_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Generate all requested features for a game.

        Args:
            feature_names: List of feature names to generate
            home_team: Home team name
            away_team: Away team name
            season: Season string (e.g., '2024-2025')
            game_date: Game date string (YYYY-MM-DD)
            player_filters: Dict with team names as keys:
                {team: {'playing': [player_ids], 'starters': [player_ids]}}
            additional_features: Optional dict of pre-computed features (e.g., pred_margin)
            recency_decay_k: Decay constant for recency weighting
            venue_guid: Optional venue GUID for travel feature calculations
            game_id: Optional game ID for training mode. When provided, uses cross-team
                aggregation to include traded players' full season history in PER calcs.

        Returns:
            Dict mapping feature names to their computed values
        """
        # Parse date
        pred_date = datetime.strptime(game_date, '%Y-%m-%d')
        year = pred_date.year
        month = pred_date.month
        day = pred_date.day

        features_dict: Dict[str, float] = {}

        # Reset player lists for this generation
        self._per_player_lists = {}
        self._injury_player_lists = {}

        # Known special features that are NOT pipe-delimited
        VALID_SPECIAL_FEATURES = {
            'pred_margin',
            'SeasonStartYear', 'Year', 'Month', 'Day',
        }

        # Categorize features for efficient processing
        regular_features: List[str] = []
        per_features: List[str] = []
        injury_features: List[str] = []
        special_features: List[str] = []

        import time as _time
        _start_categorize = _time.time()

        for feature_name in feature_names:
            if '|' not in feature_name:
                special_features.append(feature_name)
            elif feature_name.startswith('player_') or feature_name.startswith('per_available'):
                per_features.append(feature_name)
            elif feature_name.startswith('inj_'):
                injury_features.append(feature_name)
            else:
                regular_features.append(feature_name)

        print(f"[SharedFeatureGenerator] Categorized {len(feature_names)} features: "
              f"{len(regular_features)} regular, {len(per_features)} PER, "
              f"{len(injury_features)} injury, {len(special_features)} special")

        # 1. Handle special non-pipe features
        for feature_name in special_features:
            if feature_name == 'SeasonStartYear':
                features_dict[feature_name] = int(year) if int(month) >= 10 else int(year) - 1
            elif feature_name == 'Year':
                features_dict[feature_name] = int(year)
            elif feature_name == 'Month':
                features_dict[feature_name] = int(month)
            elif feature_name == 'Day':
                features_dict[feature_name] = int(day)
            elif feature_name == 'pred_margin':
                # Will be overwritten by additional_features if provided
                features_dict[feature_name] = 0.0
            else:
                features_dict[feature_name] = 0.0

        # 2. Calculate regular stat features via BasketballFeatureComputer
        _start_regular = _time.time()

        regular_results = self._computer.compute_matchup_features(
            regular_features, home_team, away_team, season,
            game_date, venue_guid=venue_guid,
        )
        features_dict.update(regular_results)

        _elapsed_regular = _time.time() - _start_regular
        print(f"[SharedFeatureGenerator] Regular features: {_elapsed_regular:.2f}s for {len(regular_features)} features")

        # 3. Calculate PER features if any are needed
        if per_features:
            _start_per = _time.time()
            per_feature_values = self._calculate_per_features(
                home_team, away_team, season, game_date,
                player_filters, per_features,
                game_id=game_id
            )
            features_dict.update(per_feature_values)
            _elapsed_per = _time.time() - _start_per
            print(f"[SharedFeatureGenerator] PER features: {_elapsed_per:.2f}s for {len(per_features)} features")

        # 4. Calculate injury features if any are needed
        if injury_features:
            _start_injury = _time.time()
            injury_feature_values = self._calculate_injury_features(
                home_team, away_team, season, year, month, day,
                injury_features, recency_decay_k
            )
            features_dict.update(injury_feature_values)
            _elapsed_injury = _time.time() - _start_injury
            print(f"[SharedFeatureGenerator] Injury features: {_elapsed_injury:.2f}s for {len(injury_features)} features")

        # 5. Merge additional features (e.g., pred_margin from points model)
        if additional_features:
            features_dict.update(additional_features)

        # 6. Ensure all requested features have a value (default to 0.0)
        for feature_name in feature_names:
            if feature_name not in features_dict:
                features_dict[feature_name] = 0.0

        return features_dict

    def _calculate_per_features(
        self,
        home_team: str,
        away_team: str,
        season: str,
        game_date: str,
        player_filters: Dict,
        per_feature_names: List[str],
        game_id: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Calculate PER (Player Efficiency Rating) features.

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season string
            game_date: Date string (YYYY-MM-DD)
            player_filters: Dict with player filter info per team
            per_feature_names: List of PER feature names to calculate
            game_id: Optional game ID for training mode. When provided, uses cross-team
                aggregation to include traded players' full season history.
        """
        result: Dict[str, float] = {}

        # Initialize all to 0.0 in case calculation fails
        for fname in per_feature_names:
            result[fname] = 0.0

        try:
            # Get injured players from player_filters (already sourced from rosters)
            injured_players_dict = None
            if player_filters:
                injured_players_dict = {
                    team: player_filters.get(team, {}).get('injured', [])
                    for team in [home_team, away_team]
                }

            # Calculate PER features using PERCalculator
            per_features = self.per_calculator.get_game_per_features(
                home_team=home_team,
                away_team=away_team,
                season=season,
                game_date=game_date,
                player_filters=player_filters,
                injured_players=injured_players_dict,
                game_id=game_id
            )

            if per_features:
                # Copy calculated values
                for fname in per_feature_names:
                    if fname in per_features:
                        result[fname] = per_features[fname]

                # Extract player lists for UI
                if '_player_lists' in per_features:
                    self._per_player_lists.update(per_features['_player_lists'])

        except Exception as e:
            import traceback
            print(f"[SharedFeatureGenerator] Error calculating PER features: {e}")
            traceback.print_exc()

        return result

    def _calculate_injury_features(
        self,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        injury_feature_names: List[str],
        recency_decay_k: float
    ) -> Dict[str, float]:
        """Calculate injury-related features."""
        result: Dict[str, float] = {}

        # Initialize all to 0.0 in case calculation fails
        for fname in injury_feature_names:
            result[fname] = 0.0

        try:
            game_date_str = f"{year}-{month:02d}-{day:02d}"

            # Try to get game document
            game_doc = None
            try:
                # Use repository to find game by teams and date
                games = self._games_repo.find({
                    'homeTeam.name': home_team,
                    'awayTeam.name': away_team,
                    'season': season,
                    'date': game_date_str
                }, limit=1)
                game_doc = games[0] if games else None
            except Exception:
                pass

            injury_features = self._injury_calculator.get_injury_features(
                home_team, away_team, season, year, month, day,
                game_doc=game_doc,
                per_calculator=self.per_calculator,
                recency_decay_k=recency_decay_k
            )

            if injury_features:
                # Copy calculated values
                for fname in injury_feature_names:
                    if fname in injury_features:
                        result[fname] = injury_features[fname]

                # Extract player lists for UI
                if '_player_lists' in injury_features:
                    self._injury_player_lists.update(injury_features['_player_lists'])

        except Exception as e:
            import traceback
            print(f"[SharedFeatureGenerator] Error calculating injury features: {e}")
            traceback.print_exc()

        return result

    def get_player_lists(self) -> Dict:
        """Get player lists from the most recent feature generation (for UI display)."""
        result = {}
        if self._per_player_lists:
            result.update(self._per_player_lists)
        if self._injury_player_lists:
            result.update(self._injury_player_lists)
        return result


def collect_unique_features(feature_lists: List[List[str]]) -> List[str]:
    """
    Collect unique features from multiple feature lists.

    Args:
        feature_lists: List of feature name lists from different models

    Returns:
        Sorted list of unique feature names
    """
    all_features: Set[str] = set()
    for feature_list in feature_lists:
        all_features.update(feature_list)
    return sorted(all_features)
