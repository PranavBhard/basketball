"""
Unified Prediction Service - Single Source of Truth for all prediction workflows.

This service provides a unified interface for making NBA game predictions,
used by all consumer layers (web UI, agent tooling, CLI).

Consumer layers should ONLY use this service for predictions - do NOT implement
prediction logic elsewhere.

Usage:
    from bball.services.prediction import PredictionService

    service = PredictionService()

    # Single game prediction (rosters are the source of truth for player lists)
    result = service.predict_matchup(
        home_team='LAL',
        away_team='BOS',
        game_date='2024-03-15'
    )

    # All games on a date
    results = service.predict_date('2024-03-15')
"""

import os
import pickle
import json
from datetime import datetime, date
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from bball.mongo import Mongo
from bball.models.bball_model import BballModel
from bball.models.points_regression import PointsRegressionTrainer
from bball.utils.players import build_player_lists_for_prediction
from bball.services.config_manager import ModelConfigManager
from bball.models.artifact_loader import ArtifactLoader
from bball.data import GamesRepository, ClassifierConfigRepository, PointsConfigRepository
from bball.market.kalshi import get_team_abbrev_map
from sportscore.services.base_prediction import BasePredictionContext, BasePredictionService
from sportscore.services.betting_report import prob_to_american_odds
from collections import defaultdict
import time

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


class PredictionContext(BasePredictionContext):
    """
    Scoped preload context for predictions.

    Preloads only the data needed for a specific prediction scope
    (season, date range, teams) and shares it across models.

    This avoids the slow initialization of loading ALL 500K+ games
    while still preventing per-feature DB queries during prediction.

    Usage:
        context = PredictionContext(db, season='2024-2025')
        model.set_prediction_context(context)
        # Now predictions use preloaded data (no DB calls)
    """

    def __init__(
        self,
        db,
        season: str,
        include_previous_season: bool = True,
        league: Optional["LeagueConfig"] = None,
        teams: Optional[List[str]] = None,
    ):
        """
        Initialize prediction context with scoped data preloading.

        Args:
            db: MongoDB database instance
            season: Primary season to preload (e.g., '2024-2025')
            include_previous_season: If True, also load previous season for
                rolling averages that span season boundaries
            teams: Optional list of team abbreviations to scope loading.
                When provided, only games and player stats for these teams
                are loaded (much faster for single-game predictions).
        """
        super().__init__(season=season)
        self.db = db
        self.include_previous_season = include_previous_season
        self.league = league
        self.teams = teams

        # Basketball-specific caches
        self.player_stats = defaultdict(list)  # {(team, season): [player_game_records]}
        self.player_stats_by_player = defaultdict(list)  # {(player_id, season): [player_game_records]} - for cross-team aggregation
        self.elo_cache = None  # EloCache instance (preloaded)

        # Stats
        self._games_loaded = 0
        self._player_records_loaded = 0
        self._load_time_ms = 0

        # Perform preload
        self._preload()

    def _get_seasons_to_load(self) -> List[str]:
        """Get list of seasons to preload."""
        seasons = [self.season]

        if self.include_previous_season:
            # Parse season like '2024-2025' to get previous
            try:
                start_year = int(self.season.split('-')[0])
                prev_season = f"{start_year - 1}-{start_year}"
                seasons.append(prev_season)
            except (ValueError, IndexError):
                pass

        return seasons

    def _preload(self):
        """Load scoped data into memory."""
        start_time = time.time()

        seasons = self._get_seasons_to_load()
        print(f"[PredictionContext] Preloading data for seasons: {seasons}")

        # 1. Load games for target seasons
        self._preload_games(seasons)

        # 2. Load player stats for target seasons
        self._preload_player_stats(seasons)

        # 3. Load venue cache (lightweight, always full)
        self._preload_venues()

        # 4. Load Elo cache for elo features
        self._preload_elo(seasons)

        self._load_time_ms = int((time.time() - start_time) * 1000)
        print(f"[PredictionContext] Preloaded {self._games_loaded} games, "
              f"{self._player_records_loaded} player records in {self._load_time_ms}ms")

    def _preload_games(self, seasons: List[str]):
        """Load games for the target seasons into nested dict structure."""
        query = {'season': {'$in': seasons}}

        if self.teams:
            query['$or'] = [
                {'homeTeam.name': {'$in': self.teams}},
                {'awayTeam.name': {'$in': self.teams}},
            ]

        # Only fetch fields needed for feature calculations
        projection = {
            'homeTeam': 1, 'awayTeam': 1, 'season': 1, 'date': 1,
            'homeWon': 1, 'venue': 1, 'venue_guid': 1,
            'game_type': 1, '_id': 1, 'game_id': 1,  # game_id is needed to link to player stats
            'vegas': 1, 'pregame_lines': 1,  # Vegas betting lines for vegas_* features
        }

        games_coll = self.league.collections["games"] if self.league is not None else "stats_nba"
        games = list(self.db[games_coll].find(query, projection))
        self._games_loaded = len(games)

        # Build nested dict structure matching BasketballFeatureComputer's expected format
        for game in games:
            season = game.get('season')
            date_str = str(game.get('date', ''))[:10]  # Ensure YYYY-MM-DD format

            home_team = game.get('homeTeam', {})
            away_team = game.get('awayTeam', {})

            home_name = home_team.get('name') if isinstance(home_team, dict) else None
            away_name = away_team.get('name') if isinstance(away_team, dict) else None

            if not season or not date_str or not home_name or not away_name:
                continue

            # games_home[season][date][home_team] = game
            if season not in self.games_home:
                self.games_home[season] = {}
            if date_str not in self.games_home[season]:
                self.games_home[season][date_str] = {}
            self.games_home[season][date_str][home_name] = game

            # games_away[season][date][away_team] = game
            if season not in self.games_away:
                self.games_away[season] = {}
            if date_str not in self.games_away[season]:
                self.games_away[season][date_str] = {}
            self.games_away[season][date_str][away_name] = game

    def _preload_player_stats(self, seasons: List[str]):
        """Load player stats for injury/PER features."""
        exclude_game_types = (
            self.league.exclude_game_types if self.league else ['preseason', 'allstar']
        )
        query = {
            'season': {'$in': seasons},
            'stats.min': {'$gt': 0},  # Only players who played
            'game_type': {'$nin': exclude_game_types},
        }

        if self.teams:
            query['team'] = {'$in': self.teams}

        # Fetch fields needed for PER and injury calculations
        projection = {
            'player_id': 1, 'player_name': 1, 'game_id': 1,
            'date': 1, 'season': 1, 'team': 1, 'home': 1,
            'starter': 1, 'stats': 1
        }

        player_stats_coll = self.league.collections["player_stats"] if self.league is not None else "stats_nba_players"
        records = list(self.db[player_stats_coll].find(query, projection))
        self._player_records_loaded = len(records)

        # Index by (team, season) for fast lookup
        for rec in records:
            key = (rec.get('team'), rec.get('season'))
            self.player_stats[key].append(rec)

            # Also index by (player_id, season) for cross-team aggregation
            # This enables traded player stats to be aggregated across all teams
            player_key = (str(rec.get('player_id')), rec.get('season'))
            self.player_stats_by_player[player_key].append(rec)

        # Sort each list by date for chronological access
        for key in self.player_stats:
            self.player_stats[key].sort(key=lambda x: str(x.get('date', '')))
        for key in self.player_stats_by_player:
            self.player_stats_by_player[key].sort(key=lambda x: str(x.get('date', '')))

    def _preload_venues(self):
        """Load venue locations for travel distance features."""
        venues_coll = self.league.collections["venues"] if self.league is not None else "nba_venues"
        venues = list(self.db[venues_coll].find({}, {
            'venue_guid': 1,
            'location.lat': 1,
            'location.lon': 1,
            'location.long': 1  # Some records use 'long' instead of 'lon'
        }))

        for v in venues:
            guid = v.get('venue_guid')
            loc = v.get('location', {})
            if guid and loc:
                lat = loc.get('lat')
                lon = loc.get('lon') or loc.get('long')
                if lat is not None and lon is not None:
                    self.venue_cache[guid] = (lat, lon)

        print(f"[PredictionContext] Loaded {len(self.venue_cache)} venue locations")

    def _preload_elo(self, seasons: List[str]):
        """Preload Elo cache for fast elo feature lookups."""
        try:
            from bball.stats.elo_cache import EloCache
            self.elo_cache = EloCache(self.db, league=self.league)
            self.elo_cache.preload(seasons=seasons)
        except Exception as e:
            print(f"[PredictionContext] Warning: Failed to preload Elo cache: {e}")

    def get_stats(self) -> Dict:
        """Return preload statistics."""
        stats = {
            'season': self.season,
            'games_loaded': self._games_loaded,
            'player_records_loaded': self._player_records_loaded,
            'venues_loaded': len(self.venue_cache),
            'load_time_ms': self._load_time_ms
        }
        if self.teams:
            stats['teams'] = self.teams
        return stats


@dataclass
class PredictionResult:
    """Structured result from a prediction."""
    home_team: str
    away_team: str
    game_date: str
    predicted_winner: str  # 'home' or 'away'
    home_win_prob: float  # 0-100
    away_win_prob: float  # 0-100
    home_odds: int  # American odds
    away_odds: int  # American odds
    home_points_pred: Optional[int] = None
    away_points_pred: Optional[int] = None
    point_diff_pred: Optional[float] = None
    features_dict: Dict[str, float] = field(default_factory=dict)
    feature_players: Dict[str, List] = field(default_factory=dict)
    home_injured_players: List[Dict] = field(default_factory=list)
    away_injured_players: List[Dict] = field(default_factory=list)
    game_id: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class MatchupInfo:
    """Information about a scheduled matchup."""
    home_team: str
    away_team: str
    game_id: str
    venue_guid: Optional[str] = None
    game_date: Optional[str] = None


class PredictionService(BasePredictionService):
    """
    Single Source of Truth for all prediction workflows.

    This service unifies prediction logic across:
    - Web UI (game_list, game_detail pages)
    - Agent tooling (matchup_predict.py)
    - CLI (train.py predict mode)

    All prediction requests should go through this service.
    """

    def __init__(self, db=None, league: Optional["LeagueConfig"] = None):
        """
        Initialize PredictionService.

        Args:
            db: MongoDB database instance. If None, creates new connection.
        """
        self.db = db if db is not None else Mongo().db
        self.league = league
        self.config_manager = ModelConfigManager(self.db, league=league)
        self._model_cache: Dict[str, BballModel] = {}
        self._points_model_cache: Dict[str, PointsRegressionTrainer] = {}
        # Initialize repositories
        self._games_repo = GamesRepository(self.db, league=league)
        self._classifier_config_repo = ClassifierConfigRepository(self.db, league=league)
        self._points_config_repo = PointsConfigRepository(self.db, league=league)
        # Prediction context cache - avoids repeated data loading per season
        self._context_cache: Dict[str, PredictionContext] = {}
        # Team abbreviation mapping (ESPN/Kalshi -> internal DB format)
        league_id = league.league_id if league else "nba"
        self._team_abbrev_map = get_team_abbrev_map(league_id)

    def _normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to internal database format.

        ESPN/Kalshi uses abbreviations like 'GSW', 'NYK', 'NOP' while
        the internal database uses 'GS', 'NY', 'NO'. This method converts
        external abbreviations to internal format.

        Args:
            team_name: Team abbreviation (e.g., 'GSW')

        Returns:
            Internal abbreviation (e.g., 'GS')
        """
        if not team_name:
            return team_name
        return self._team_abbrev_map.get(team_name.upper(), team_name)

    def _get_or_create_context(self, season: str, teams: Optional[List[str]] = None) -> PredictionContext:
        """
        Get cached prediction context or create a new one.

        Contexts are cached by season (and optionally teams) to avoid repeated loading.
        A single context can be shared across multiple predictions for the same scope.

        Args:
            season: Season string (e.g., '2024-2025')
            teams: Optional list of team abbreviations to scope loading.
                When provided, only data for these teams is loaded.

        Returns:
            PredictionContext with preloaded data for the season
        """
        cache_key = f"{season}|{'_'.join(sorted(teams))}" if teams else season
        if cache_key not in self._context_cache:
            self._context_cache[cache_key] = PredictionContext(
                db=self.db,
                season=season,
                include_previous_season=True,
                league=self.league,
                teams=teams,
            )
        return self._context_cache[cache_key]

    def clear_context_cache(self):
        """Clear the prediction context cache. Call this to free memory."""
        self._context_cache.clear()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def build_context(self, league_id: str, target_date: str, **kwargs) -> PredictionContext:
        """Build a prediction context for the given date."""
        game_date_obj = datetime.strptime(target_date, '%Y-%m-%d').date()
        season = self._get_season_from_date(game_date_obj)
        return self._get_or_create_context(season)

    def predict_game(self, game: Dict[str, Any], context: BasePredictionContext, **kwargs) -> Dict[str, Any]:
        """
        Generate a prediction for a single game (base class interface).

        Unpacks the game dict and delegates to predict_matchup().
        """
        result = self.predict_matchup(
            home_team=game['home_team'],
            away_team=game['away_team'],
            game_date=game.get('game_date', ''),
            game_id=game.get('game_id'),
            venue_guid=game.get('venue_guid'),
            **kwargs,
        )
        return result.to_dict()

    def predict_matchup(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        game_id: Optional[str] = None,
        include_points: bool = True,
        classifier_config: Optional[Dict] = None,
        points_config: Optional[Dict] = None,
        venue_guid: Optional[str] = None,
    ) -> PredictionResult:
        """
        Generate prediction for a single game.

        This is the primary prediction method - all interfaces should use this.
        Player lists (playing, starters, injured) come from rosters as the
        single source of truth.

        Args:
            home_team: Home team abbreviation (e.g., 'LAL')
            away_team: Away team abbreviation (e.g., 'BOS')
            game_date: Game date in 'YYYY-MM-DD' format
            game_id: Optional game ID for looking up game document
            include_points: Whether to include points model prediction
            classifier_config: Optional classifier config (uses selected if None)
            points_config: Optional points config (uses selected if None)

        Returns:
            PredictionResult with prediction details
        """
        # Normalize team names (ESPN/Kalshi uses different abbreviations than internal DB)
        # e.g., 'GSW' -> 'GS', 'NOP' -> 'NO', 'NYK' -> 'NY'
        home_team = self._normalize_team_name(home_team)
        away_team = self._normalize_team_name(away_team)

        # Parse date
        try:
            game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
        except ValueError:
            return PredictionResult(
                home_team=home_team,
                away_team=away_team,
                game_date=game_date,
                predicted_winner='',
                home_win_prob=0,
                away_win_prob=0,
                home_odds=0,
                away_odds=0,
                error=f'Invalid date format: {game_date}. Use YYYY-MM-DD format.'
            )

        season = self._get_season_from_date(game_date_obj)

        # Get game document if game_id provided
        game_doc = None
        if game_id:
            game_doc = self._games_repo.find_by_game_id(game_id)

        # Load configs
        classifier_config = classifier_config or self._get_selected_classifier_config()
        if not classifier_config:
            return self._error_result(home_team, away_team, game_date, game_id,
                                      'No classifier model config selected.')

        # Validate config is ready for predictions
        is_valid, error_msg = ModelConfigManager.validate_config_for_prediction(classifier_config)
        if not is_valid:
            return self._error_result(home_team, away_team, game_date, game_id, error_msg)

        # Get or create prediction context for this season (preloads data once).
        # Always load full season — SOS features need all opponents' game data,
        # not just the two teams playing.
        context = self._get_or_create_context(season)

        # Load classifier model with preloaded context
        model = self._load_classifier_model(classifier_config, context)
        if not model:
            return self._error_result(home_team, away_team, game_date, game_id,
                                      'Failed to load classifier model.')

        # Build player filters from rosters (single source of truth)
        player_filters = build_player_lists_for_prediction(
            home_team=home_team,
            away_team=away_team,
            season=season,
            db=self.db,
            league=self.league
        )

        # Get points prediction if enabled
        additional_features = {}
        points_prediction = None

        if include_points:
            points_config = points_config or self._get_selected_points_config()
            if points_config:
                points_prediction = self._get_points_prediction(
                    points_config, home_team, away_team, game_date,
                    game_date_obj, season, game_doc, game_id
                )

                # Check if classifier needs pred_margin
                if self._needs_pred_margin(classifier_config, model):
                    pred_margin = self._extract_pred_margin(points_prediction)
                    if pred_margin is not None:
                        additional_features['pred_margin'] = pred_margin

        # Get venue_guid for travel feature calculations
        # Prefer explicitly passed venue_guid (from predict_date matchup),
        # fall back to game_doc (from DB — set by web app or ESPN sync)
        if not venue_guid and game_doc:
            venue_guid = game_doc.get('venue_guid')

        # Make classifier prediction
        try:
            use_calibrated = classifier_config.get('use_time_calibration', False)
            prediction = model.predict_with_player_config(
                home_team, away_team, season, game_date, player_filters,
                use_calibrated=use_calibrated,
                additional_features=additional_features if additional_features else None,
                venue_guid=venue_guid
            )
        except Exception as e:
            return self._error_result(home_team, away_team, game_date, game_id,
                                      f'Prediction failed: {str(e)}')

        # Build result
        return self._build_prediction_result(
            home_team, away_team, game_date, season, game_id,
            prediction, points_prediction, model
        )

    def predict_date(
        self,
        game_date: str,
        include_points: bool = True,
        classifier_config: Optional[Dict] = None,
        points_config: Optional[Dict] = None,
        job_id: Optional[str] = None,
        on_prediction: Optional[callable] = None,
    ) -> List[PredictionResult]:
        """
        Generate predictions for all games on a date.

        Args:
            game_date: Date in 'YYYY-MM-DD' format
            include_points: Whether to include points model predictions
            classifier_config: Optional classifier config (uses selected if None)
            points_config: Optional points config (uses selected if None)
            job_id: Optional job ID for progress tracking (used by async bulk predictions)
            on_prediction: Optional callback called after each prediction with (result, matchup).
                          Use this to save predictions incrementally for real-time UI updates.

        Returns:
            List of PredictionResult objects, one per game
        """
        # Import job progress updater if job_id provided
        if job_id:
            from bball.services.jobs import update_job_progress

        # Parse date
        try:
            game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
        except ValueError:
            return [PredictionResult(
                home_team='',
                away_team='',
                game_date=game_date,
                predicted_winner='',
                home_win_prob=0,
                away_win_prob=0,
                home_odds=0,
                away_odds=0,
                error=f'Invalid date format: {game_date}. Use YYYY-MM-DD format.'
            )]

        # Update progress: fetching matchups
        if job_id:
            update_job_progress(job_id, 10, 'Fetching matchups...', league=self.league)

        # Fetch matchups for the date
        matchups = self._get_matchups_for_date(game_date_obj)
        if not matchups:
            return []

        # OPTIMIZATION: Pre-load configs once before the loop (avoids repeated DB queries)
        if classifier_config is None:
            classifier_config = self._get_selected_classifier_config()
        if include_points and points_config is None:
            points_config = self._get_selected_points_config()

        # Update progress: loading models
        if job_id:
            update_job_progress(job_id, 15, 'Loading models...', league=self.league)

        # OPTIMIZATION: Get or create prediction context for this season (preloads data once)
        season = self._get_season_from_date(game_date_obj)
        context = self._get_or_create_context(season)

        # OPTIMIZATION: Pre-load models once with context (they will be cached for subsequent calls)
        if classifier_config:
            self._load_classifier_model(classifier_config, context)
        if include_points and points_config:
            self._load_points_model(points_config)

        # Update progress: starting predictions
        if job_id:
            update_job_progress(job_id, 20, f'Running predictions for {len(matchups)} games...', league=self.league)

        # Generate prediction for each matchup
        results = []
        total_games = len(matchups)

        for i, matchup in enumerate(matchups):
            result = self.predict_matchup(
                home_team=matchup.home_team,
                away_team=matchup.away_team,
                game_date=game_date,
                game_id=matchup.game_id,
                include_points=include_points,
                classifier_config=classifier_config,
                points_config=points_config,
                venue_guid=matchup.venue_guid,
            )
            results.append(result)

            # Call callback to save prediction immediately (for real-time UI updates)
            if on_prediction:
                try:
                    on_prediction(result, matchup)
                except Exception as e:
                    print(f"Error in on_prediction callback for {matchup.game_id}: {e}")

            # Update progress after each game (scale from 20% to 90%)
            if job_id:
                progress = 20 + int(70 * (i + 1) / total_games)
                msg = f'Predicted {i + 1}/{total_games}: {matchup.away_team} @ {matchup.home_team}'
                update_job_progress(job_id, progress, msg, league=self.league)

        return results

    def get_selected_configs(self) -> Dict[str, Optional[Dict]]:
        """
        Get currently selected classifier and points configs.

        Returns:
            Dict with 'classifier' and 'points' keys
        """
        return {
            'classifier': self._get_selected_classifier_config(),
            'points': self._get_selected_points_config()
        }

    # =========================================================================
    # CONFIG LOADING
    # =========================================================================

    def _get_selected_classifier_config(self) -> Optional[Dict]:
        """Get the currently selected classifier config."""
        return self._classifier_config_repo.find_selected()

    def _get_selected_points_config(self) -> Optional[Dict]:
        """Get the currently selected points config."""
        return self._points_config_repo.find_selected()

    # =========================================================================
    # MODEL LOADING
    # =========================================================================

    def _load_classifier_model(self, config: Dict, context: Optional[PredictionContext] = None) -> Optional[BballModel]:
        """
        Load classifier model from config.

        Uses caching to avoid redundant loads.

        Args:
            config: Model configuration dict
            context: Optional PredictionContext with preloaded data
        """
        is_ensemble = config.get('ensemble', False)

        if is_ensemble:
            return self._load_ensemble_model(config, context)
        else:
            return self._load_regular_model(config, context)

    def _load_regular_model(self, config: Dict, context: Optional[PredictionContext] = None) -> Optional[BballModel]:
        """Load a regular (non-ensemble) classifier model."""
        try:
            # Cache key uses model_artifact_path (aligned with BaseArtifactLoader caching)
            # This naturally invalidates when a model is retrained (new run_id = new path)
            cache_key = config.get('model_artifact_path') or str(config.get('_id', ''))
            if cache_key and cache_key in self._model_cache:
                model = self._model_cache[cache_key]
                # Inject context even for cached models (context may change per season)
                if context:
                    model.set_prediction_context(context)
                return model

            # Create BballModel instance
            model = BballModel(
                classifier_features=[],
                points_features=[],
                include_elo=True,
                use_exponential_weighting=False,
                include_era_normalization=False,
                include_per_features=True,
                include_injuries=False,
                preload_data=False,
                league=self.league
            )
            model.db = self.db

            # Load model using ArtifactLoader (prioritizes artifacts, uses cache)
            classifier, scaler, feature_names = ArtifactLoader.create_model(config, use_artifacts=True)

            model.classifier_model = classifier
            model.scaler = scaler
            model.feature_names = feature_names
            model.classifier_features = feature_names

            # Inject prediction context to avoid per-feature DB calls
            if context:
                model.set_prediction_context(context)

            # Cache the model for reuse
            if cache_key:
                self._model_cache[cache_key] = model

            return model

        except Exception as e:
            print(f"Error loading regular model: {e}")
            return None

    def _load_ensemble_model(self, config: Dict, context: Optional[PredictionContext] = None) -> Optional[BballModel]:
        """Load an ensemble model from config."""
        try:
            ensemble_run_id = config.get('ensemble_run_id')
            if not ensemble_run_id:
                print("Error: Ensemble config missing ensemble_run_id")
                return None

            # Cache key uses ensemble_run_id (aligned with artifact-based caching in sportscore)
            # This naturally invalidates when ensemble is retrained (new run_id = new artifacts)
            cache_key = ensemble_run_id
            if cache_key and cache_key in self._model_cache:
                model = self._model_cache[cache_key]
                # Inject context even for cached models (context may change per season)
                if context:
                    model.set_prediction_context(context)
                return model

            # Get path to ensemble artifacts
            # bball/services/prediction.py -> 3 dirname -> basketball/ (project root)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            ensembles_dir = os.path.join(project_root, 'cli', 'models', 'ensembles')
            meta_model_path = os.path.join(ensembles_dir, f'{ensemble_run_id}_meta_model.pkl')

            if not os.path.exists(meta_model_path):
                print(f"Error: Ensemble meta model not found: {meta_model_path}")
                return None

            # Create BballModel instance for ensemble
            model = BballModel(
                classifier_features=[],
                points_features=[],
                include_elo=False,
                use_exponential_weighting=False,
                include_era_normalization=False,
                include_per_features=True,
                include_injuries=False,
                preload_data=False,
                league=self.league
            )
            model.db = self.db

            # Mark as ensemble
            model.is_ensemble = True
            model.ensemble_run_id = ensemble_run_id
            model.ensemble_config = config
            model.ensemble_base_models = config.get('ensemble_models', [])
            model.ensemble_meta_features = config.get('ensemble_meta_features', [])

            # Inject prediction context to avoid per-feature DB calls
            if context:
                model.set_prediction_context(context)

            # Cache the model for reuse
            if cache_key:
                self._model_cache[cache_key] = model

            return model

        except Exception as e:
            print(f"Error loading ensemble model: {e}")
            return None

    def _load_points_model(self, config: Dict) -> Optional[PointsRegressionTrainer]:
        """Load points regression model from config."""
        try:
            # Cache key: prefer stable identifiers, fall back to artifact paths.
            cache_key = (
                str(config.get('_id'))
                if config.get('_id') is not None
                else (config.get('model_path') or config.get('model_artifact_path') or '')
            )
            if cache_key and cache_key in self._points_model_cache:
                return self._points_model_cache[cache_key]

            trainer = PointsRegressionTrainer(db=self.db)

            # Support both schemas:
            # - New: model_artifact_path / scaler_artifact_path / features_path
            # - Legacy: model_path (plus conventional sibling files like *_scaler.pkl, *_features.json)
            model_path = config.get('model_artifact_path') or config.get('model_path')
            if not model_path or not os.path.exists(model_path):
                return None

            model_dir = os.path.dirname(model_path)
            model_stem = os.path.splitext(os.path.basename(model_path))[0]

            # If explicit paths are provided, trust them; else derive from model_path.
            scaler_path = config.get('scaler_artifact_path') or os.path.join(model_dir, f"{model_stem}_scaler.pkl")
            features_path = config.get('features_path') or os.path.join(model_dir, f"{model_stem}_features.json")

            # Required files: model + scaler + feature names (PointsRegressionTrainer expects these).
            if not os.path.exists(scaler_path) or not os.path.exists(features_path):
                # As a fallback, try PointsRegressionTrainer.load_model(model_name) if the files
                # live in its artifacts directory (older flows often store only model_path in Mongo).
                try:
                    trainer.load_model(model_stem)
                    if cache_key:
                        self._points_model_cache[cache_key] = trainer
                    return trainer
                except Exception:
                    return None

            with open(model_path, 'rb') as f:
                trainer.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                trainer.scaler = pickle.load(f)
            with open(features_path, 'r') as f:
                trainer.feature_names = json.load(f)

            # Derive target_type (home_away vs margin) to match trainer expectations.
            if isinstance(trainer.model, dict):
                trainer.target_type = 'home_away'
            else:
                trainer.target_type = 'margin'

            # Optional: perspective-split scalers/features for newer home/away models.
            home_scaler_path = os.path.join(model_dir, f"{model_stem}_home_scaler.pkl")
            away_scaler_path = os.path.join(model_dir, f"{model_stem}_away_scaler.pkl")
            home_features_path = os.path.join(model_dir, f"{model_stem}_home_features.json")
            away_features_path = os.path.join(model_dir, f"{model_stem}_away_features.json")

            if os.path.exists(home_scaler_path):
                with open(home_scaler_path, 'rb') as f:
                    trainer.home_scaler = pickle.load(f)
            if os.path.exists(away_scaler_path):
                with open(away_scaler_path, 'rb') as f:
                    trainer.away_scaler = pickle.load(f)
            if os.path.exists(home_features_path):
                with open(home_features_path, 'r') as f:
                    trainer.home_feature_names = json.load(f)
            if os.path.exists(away_features_path):
                with open(away_features_path, 'r') as f:
                    trainer.away_feature_names = json.load(f)

            if cache_key:
                self._points_model_cache[cache_key] = trainer
            return trainer

        except Exception as e:
            print(f"Error loading points model: {e}")
            return None

    # =========================================================================
    # MATCHUP FETCHING
    # =========================================================================

    def _get_matchups_for_date(self, game_date: date) -> List[MatchupInfo]:
        """
        Fetch matchups for a date from ESPN API.

        Args:
            game_date: Date to fetch matchups for

        Returns:
            List of MatchupInfo objects
        """
        try:
            # Use ESPNClient for API access
            from bball.data import ESPNClient

            espn_client = ESPNClient(league=self.league)
            scoreboard = espn_client.get_scoreboard_site(game_date)
            if not scoreboard:
                print(f"[PredictionService] get_scoreboard returned None/empty for {game_date}")
                return []

            matchups = []
            _venue_id_to_guid = {}   # ESPN numeric id -> UUID venue_guid
            _espn_ids_to_resolve = set()

            # ESPN scoreboard API returns events at top level (not nested under sports/leagues)
            events = scoreboard.get('events', [])
            print(f"[PredictionService] Scoreboard has {len(events)} events")

            for event in events:
                game_id = event.get('id')
                if not game_id:
                    continue

                # Extract teams from competitions[0].competitors
                competitions = event.get('competitions', [])
                if not competitions:
                    continue

                competitors = competitions[0].get('competitors', [])
                home_team = None
                away_team = None
                for comp in competitors:
                    team = comp.get('team', {})
                    abbrev = team.get('abbreviation', '').upper()
                    if comp.get('homeAway') == 'home':
                        home_team = abbrev
                    else:
                        away_team = abbrev

                if not home_team or not away_team:
                    continue

                # Get venue info from competition
                venue_guid = None
                venue_info = competitions[0].get('venue', {})
                espn_venue_id = venue_info.get('id')

                # Scoreboard returns numeric ESPN venue 'id', but our venue
                # cache is keyed by UUID 'venue_guid'. Resolve via venues
                # collection (same approach as web app index route).
                if espn_venue_id and espn_venue_id not in _venue_id_to_guid:
                    _espn_ids_to_resolve.add(str(espn_venue_id))

                if espn_venue_id:
                    venue_guid = str(espn_venue_id)  # placeholder; resolved below

                # Fallback to game summary if no venue at all
                if not venue_guid:
                    game_summary = espn_client.get_game_summary(game_id)
                    if game_summary:
                        game_info = game_summary.get('gameInfo', {})
                        venue = game_info.get('venue', {})
                        venue_guid = venue.get('guid') or venue.get('id')

                matchups.append(MatchupInfo(
                    home_team=home_team,
                    away_team=away_team,
                    game_id=game_id,
                    venue_guid=venue_guid,
                    game_date=game_date.strftime('%Y-%m-%d')
                ))

            # Batch-resolve ESPN numeric venue IDs to UUID venue_guids
            if _espn_ids_to_resolve:
                venues_coll = "nba_venues"
                if self.league:
                    venues_coll = self.league.collections.get("venues", venues_coll)
                venue_docs = self.db[venues_coll].find(
                    {'id': {'$in': list(_espn_ids_to_resolve)}},
                    {'id': 1, 'venue_guid': 1}
                )
                for vdoc in venue_docs:
                    if vdoc.get('id') and vdoc.get('venue_guid'):
                        _venue_id_to_guid[str(vdoc['id'])] = vdoc['venue_guid']

            # Replace ESPN IDs with proper venue_guids on each matchup
            for m in matchups:
                if m.venue_guid:
                    m.venue_guid = _venue_id_to_guid.get(m.venue_guid, m.venue_guid)

            return matchups

        except Exception as e:
            import traceback
            print(f"Error fetching matchups: {e}")
            traceback.print_exc()
            return []

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_season_from_date(self, d: date) -> str:
        """Get season string from date using league rules."""
        cutover = self.league.season_cutover_month if self.league is not None else 8
        if d.month > cutover:
            return f"{d.year}-{d.year + 1}"
        else:
            return f"{d.year - 1}-{d.year}"

    def _needs_pred_margin(self, config: Dict, model: BballModel) -> bool:
        """Check if the classifier needs pred_margin as a feature."""
        is_ensemble = config.get('ensemble', False)
        if is_ensemble:
            ensemble_meta_features = config.get('ensemble_meta_features') or []
            return 'pred_margin' in ensemble_meta_features
        else:
            return (hasattr(model, 'feature_names') and
                    model.feature_names and
                    'pred_margin' in model.feature_names)

    def _get_points_prediction(
        self,
        points_config: Dict,
        home_team: str,
        away_team: str,
        game_date: str,
        game_date_obj: date,
        season: str,
        game_doc: Optional[Dict],
        game_id: Optional[str]
    ) -> Optional[Dict]:
        """Get points prediction from points model."""
        points_trainer = self._load_points_model(points_config)
        if not points_trainer:
            return None

        try:
            # Build game doc if needed
            if not game_doc:
                game_doc = {
                    'game_id': game_id or '',
                    'date': game_date,
                    'year': game_date_obj.year,
                    'month': game_date_obj.month,
                    'day': game_date_obj.day,
                    'season': season,
                    'homeTeam': {'name': home_team},
                    'awayTeam': {'name': away_team}
                }

            return points_trainer.predict(game_doc, game_date)

        except Exception as e:
            print(f"Warning: Points prediction failed: {e}")
            return None

    def _extract_pred_margin(self, points_prediction: Optional[Dict]) -> Optional[float]:
        """Extract pred_margin from points prediction."""
        if not points_prediction:
            return None

        if 'point_diff_pred' in points_prediction and points_prediction['point_diff_pred'] is not None:
            return points_prediction['point_diff_pred']

        if 'home_points' in points_prediction and 'away_points' in points_prediction:
            home = points_prediction.get('home_points', 0) or 0
            away = points_prediction.get('away_points', 0) or 0
            return home - away

        return None

    def _calculate_odds(self, prob_percent: float) -> int:
        """Convert probability (0-100) to American odds."""
        return prob_to_american_odds(prob_percent / 100.0)

    def _build_prediction_result(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        season: str,
        game_id: Optional[str],
        prediction: Dict,
        points_prediction: Optional[Dict],
        model: BballModel
    ) -> PredictionResult:
        """Build PredictionResult from model prediction output."""
        home_win_prob = prediction.get('home_win_prob', 50)
        away_win_prob = 100 - home_win_prob

        # Get feature players from model
        feature_players = {}
        if hasattr(model, '_per_player_lists') and model._per_player_lists:
            feature_players.update(model._per_player_lists.copy())
        if hasattr(model, '_injury_player_lists') and model._injury_player_lists:
            feature_players.update(model._injury_player_lists.copy())

        # Get injured player info
        home_injured: List[str] = []
        away_injured: List[str] = []
        try:
            # Prefer to derive from roster flags (prediction-time truth).
            from bball.data import RostersRepository, PlayersRepository

            rosters_repo = RostersRepository(self.db)
            players_repo = PlayersRepository(self.db)

            home_roster_doc = rosters_repo.find_roster(home_team, season)
            away_roster_doc = rosters_repo.find_roster(away_team, season)

            home_injured_ids = []
            away_injured_ids = []

            if home_roster_doc:
                home_injured_ids = [
                    str(e.get('player_id', ''))
                    for e in home_roster_doc.get('roster', [])
                    if e.get('injured', False)
                ]
            if away_roster_doc:
                away_injured_ids = [
                    str(e.get('player_id', ''))
                    for e in away_roster_doc.get('roster', [])
                    if e.get('injured', False)
                ]

            if home_injured_ids:
                docs = players_repo.find({'player_id': {'$in': home_injured_ids}}, projection={'player_name': 1, 'player_id': 1})
                home_injured = [d.get('player_name') or str(d.get('player_id')) for d in docs]
            if away_injured_ids:
                docs = players_repo.find({'player_id': {'$in': away_injured_ids}}, projection={'player_name': 1, 'player_id': 1})
                away_injured = [d.get('player_name') or str(d.get('player_id')) for d in docs]
        except Exception:
            # Non-fatal: UI can still render features; injured player names are optional.
            pass

        result = PredictionResult(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            game_id=game_id,
            predicted_winner=prediction.get('predicted_winner', 'home' if home_win_prob > 50 else 'away'),
            home_win_prob=home_win_prob,
            away_win_prob=away_win_prob,
            home_odds=self._calculate_odds(home_win_prob),
            away_odds=self._calculate_odds(away_win_prob),
            features_dict=prediction.get('features_dict', {}),
            feature_players=feature_players,
            home_injured_players=home_injured,
            away_injured_players=away_injured,
        )

        # Add points predictions if available
        if points_prediction:
            if 'home_points' in points_prediction and points_prediction['home_points'] is not None:
                result.home_points_pred = round(points_prediction['home_points'])
            if 'away_points' in points_prediction and points_prediction['away_points'] is not None:
                result.away_points_pred = round(points_prediction['away_points'])
            if 'point_diff_pred' in points_prediction and points_prediction['point_diff_pred'] is not None:
                result.point_diff_pred = round(points_prediction['point_diff_pred'], 1)

        return result

    def _error_result(
        self,
        home_team: str,
        away_team: str,
        game_date: str,
        game_id: Optional[str],
        error_msg: str
    ) -> PredictionResult:
        """Create an error PredictionResult."""
        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date,
            game_id=game_id,
            predicted_winner='',
            home_win_prob=0,
            away_win_prob=0,
            home_odds=0,
            away_odds=0,
            error=error_msg
        )

    # =========================================================================
    # PREDICTION PERSISTENCE
    # =========================================================================

    def save_prediction(
        self,
        result: PredictionResult,
        game_id: str,
        game_date: date,
        home_team: str,
        away_team: str,
    ) -> Dict[str, Any]:
        """
        Save prediction to the model_predictions collection.

        This is the Single Source of Truth for persisting predictions.
        Predictions are stored in a dedicated collection (not on game documents)
        and retrieved via get_predictions_for_date().

        Args:
            result: PredictionResult from predict_game()
            game_id: ESPN game ID
            game_date: Game date
            home_team: Home team abbreviation
            away_team: Away team abbreviation

        Returns:
            Dict with the saved prediction document
        """
        from datetime import timezone
        utc = timezone.utc

        # Get collection name from league config
        predictions_coll = "nba_model_predictions"
        if self.league is not None:
            predictions_coll = self.league.collections.get("model_predictions", "nba_model_predictions")

        # Get season
        season = self._get_season_from_date(game_date)
        game_date_str = game_date.strftime('%Y-%m-%d') if isinstance(game_date, date) else str(game_date)

        # Build prediction document
        prediction_doc = {
            'game_id': game_id,
            'game_date': game_date_str,
            'season': season,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': result.predicted_winner,
            'home_win_prob': result.home_win_prob,
            'away_win_prob': result.away_win_prob,
            'home_odds': result.home_odds,
            'away_odds': result.away_odds,
            'predicted_at': datetime.now(utc).isoformat(),
            'features_dict': result.features_dict,
            'home_injured_players': result.home_injured_players,
            'away_injured_players': result.away_injured_players,
            'feature_players': result.feature_players
        }

        # Add points predictions if available
        if result.home_points_pred is not None:
            prediction_doc['home_points_pred'] = result.home_points_pred
        if result.away_points_pred is not None:
            prediction_doc['away_points_pred'] = result.away_points_pred
        if result.home_points_pred is not None and result.away_points_pred is not None:
            prediction_doc['point_total_pred'] = result.home_points_pred + result.away_points_pred
        if result.point_diff_pred is not None:
            prediction_doc['point_diff_pred'] = result.point_diff_pred

        # Upsert by game_id (only keep latest prediction per game)
        self.db[predictions_coll].update_one(
            {'game_id': game_id},
            {'$set': prediction_doc},
            upsert=True
        )

        return prediction_doc

    def get_predictions_for_date(self, game_date: str) -> Dict[str, Dict]:
        """
        Get all predictions for games on a specific date.

        Args:
            game_date: Date string in 'YYYY-MM-DD' format

        Returns:
            Dict mapping game_id to prediction document
        """
        predictions_coll = "nba_model_predictions"
        if self.league is not None:
            predictions_coll = self.league.collections.get("model_predictions", "nba_model_predictions")

        predictions = {}
        cursor = self.db[predictions_coll].find({'game_date': game_date})
        for doc in cursor:
            game_id = doc.get('game_id')
            if game_id:
                # Remove MongoDB _id for JSON serialization
                doc.pop('_id', None)
                predictions[game_id] = doc

        return predictions

    def get_prediction_for_game(self, game_id: str) -> Optional[Dict]:
        """
        Get prediction for a specific game.

        Args:
            game_id: ESPN game ID

        Returns:
            Prediction document or None if not found
        """
        predictions_coll = "nba_model_predictions"
        if self.league is not None:
            predictions_coll = self.league.collections.get("model_predictions", "nba_model_predictions")

        doc = self.db[predictions_coll].find_one({'game_id': game_id})
        if doc:
            doc.pop('_id', None)
        return doc
