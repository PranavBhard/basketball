"""
Matchup Predict Tool - Generate predictions for NBA matchups

This tool uses PredictionService (core/prediction_service.py) as the Single Source of Truth.
It works standalone (e.g., from agent processes) without needing the Flask app running.

NOTE: The predict() function delegates to PredictionService.predict_game().
Helper functions (load_model_from_config, load_points_model) are kept for backward
compatibility but are deprecated - use PredictionService directly instead.
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

from bball_app.core.mongo import Mongo
from bball_app.core.models.bball_model import BballModel
from bball_app.core.models.points_regression import PointsRegressionTrainer
from bball_app.core.utils import get_season_from_date
from bball_app.core.services.config_manager import ModelConfigManager
from bball_app.core.models.artifact_loader import ArtifactLoader
from bball_app.core.services.prediction import PredictionService


def load_model_from_config(config: Dict, db=None) -> Optional[BballModel]:
    """
    Load BballModel from a MongoDB config document.

    Handles both regular models and ensemble models.
    Uses core infrastructure (ArtifactLoader) directly.

    Args:
        config: MongoDB model_config_nba document
        db: Optional database instance

    Returns:
        BballModel instance with model loaded, or None if failed
    """
    if db is None:
        db = Mongo().db

    is_ensemble = config.get('ensemble', False)

    if is_ensemble:
        return _load_ensemble_model(config, db)
    else:
        return _load_regular_model(config, db)


def _load_regular_model(config: Dict, db) -> Optional[BballModel]:
    """Load a regular (non-ensemble) model from config."""
    try:
        # Create BballModel instance - features loaded from config below
        model = BballModel(
            classifier_features=[],  # Set from config
            points_features=[],
            include_elo=True,
            use_exponential_weighting=False,
            include_era_normalization=False,
            include_per_features=True,
            include_injuries=False,
            preload_data=False
        )
        # Set database connection
        model.db = db

        # Load model using ArtifactLoader (prioritizes artifacts)
        classifier, scaler, feature_names = ArtifactLoader.create_model(config, use_artifacts=True)

        model.classifier_model = classifier
        model.scaler = scaler
        model.feature_names = feature_names
        model.classifier_features = feature_names

        return model

    except Exception as e:
        print(f"Error loading regular model: {e}")
        return None


def _get_nba_app_path() -> str:
    """Get the nba_app base directory path."""
    # This file is at nba_app/agents/tools/matchup_predict.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_ensemble_model(config: Dict, db) -> Optional[BballModel]:
    """
    Load an ensemble model from config.

    Creates an BballModel marked as ensemble, which will use
    _predict_ensemble_with_player_config() for predictions.
    """
    try:
        ensemble_run_id = config.get('ensemble_run_id')
        if not ensemble_run_id:
            print("Error: Ensemble config missing ensemble_run_id")
            return None

        # Get absolute path to ensemble models directory
        nba_app_path = _get_nba_app_path()
        ensembles_dir = os.path.join(nba_app_path, 'cli', 'models', 'ensembles')
        meta_model_path = os.path.join(ensembles_dir, f'{ensemble_run_id}_meta_model.pkl')
        ensemble_cfg_path = os.path.join(ensembles_dir, f'{ensemble_run_id}_ensemble_config.json')

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
            include_per_features=True,  # Base models may need PER
            include_injuries=False,
            preload_data=False
        )
        # Set database connection
        model.db = db

        # Mark as ensemble - BballModel will handle loading meta model during prediction
        model.is_ensemble = True
        model.ensemble_run_id = ensemble_run_id
        model.ensemble_config = config
        model.ensemble_base_models = config.get('ensemble_models', [])
        model.ensemble_meta_features = config.get('ensemble_meta_features', [])

        return model

    except Exception as e:
        print(f"Error loading ensemble model: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_points_model(config: Dict, db=None) -> Optional[PointsRegressionTrainer]:
    """
    Load PointsRegressionTrainer from a MongoDB config document.

    Args:
        config: MongoDB model_config_points_nba document
        db: Optional database instance

    Returns:
        PointsRegressionTrainer instance with model loaded, or None if failed
    """
    if db is None:
        db = Mongo().db

    try:
        trainer = PointsRegressionTrainer(
            model_type=config.get('model_type', 'RandomForest'),
            db=db
        )

        # Load model using artifacts if available
        model_artifact_path = config.get('model_artifact_path')
        scaler_artifact_path = config.get('scaler_artifact_path')
        features_path = config.get('features_path')

        if model_artifact_path and os.path.exists(model_artifact_path):
            import pickle
            import json

            with open(model_artifact_path, 'rb') as f:
                trainer.model = pickle.load(f)

            if scaler_artifact_path and os.path.exists(scaler_artifact_path):
                with open(scaler_artifact_path, 'rb') as f:
                    trainer.scaler = pickle.load(f)

            if features_path and os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    trainer.feature_names = json.load(f)

            return trainer
        else:
            # Fallback to training from CSV if needed
            training_csv = config.get('training_csv')
            if training_csv and os.path.exists(training_csv):
                trainer.train_from_csv(training_csv)
                return trainer

        return None

    except Exception as e:
        print(f"Error loading points model: {e}")
        return None


def predict(
    home: str,
    away: str,
    game_id: Optional[str] = None,
    game_date: Optional[str] = None,
    home_injuries: List[str] = None,
    away_injuries: List[str] = None,
    home_starters: List[str] = None,
    away_starters: List[str] = None,
    db=None,
    league=None,
    games_collection: str = 'stats_nba'
) -> Dict:
    """
    Generate prediction for a matchup.

    Delegates to PredictionService.predict_game() (core/prediction_service.py).

    Args:
        home: Home team abbreviation (e.g., 'LAL')
        away: Away team abbreviation (e.g., 'BOS')
        game_id: Optional game ID
        game_date: Optional game date (YYYY-MM-DD). If not provided, uses today's date.
        home_injuries: Optional list of home team injured player IDs (strings)
        away_injuries: Optional list of away team injured player IDs (strings)
        home_starters: Optional list of home team starter player IDs (strings)
        away_starters: Optional list of away team starter player IDs (strings)
        db: Optional MongoDB database instance
        league: Optional LeagueConfig instance for league-aware predictions
        games_collection: Name of the games collection (default: 'stats_nba')

    Returns:
        Dict with prediction results including:
        - predicted_winner: 'home' or 'away'
        - home_win_prob: Probability of home win (0-100)
        - away_win_prob: Probability of away win (0-100)
        - model_home_odds: Model-derived American odds for home team (NOT market odds)
        - model_away_odds: Model-derived American odds for away team (NOT market odds)
        - home_points_pred: Predicted home points (if available)
        - away_points_pred: Predicted away points (if available)
        - features_dict: Feature values used in prediction
    """
    if db is None:
        db = Mongo().db

    # Use league-aware collection if available
    if league is not None:
        games_collection = league.collections.get('games', 'stats_nba')

    # Determine game date (default to today if not provided)
    if not game_date:
        if game_id:
            game_doc = db[games_collection].find_one({'game_id': game_id})
            if game_doc and game_doc.get('date'):
                game_date = game_doc['date']
        if not game_date:
            game_date = datetime.now().strftime('%Y-%m-%d')

    # Use PredictionService (SSoT for all prediction workflows)
    # Rosters are the single source of truth for player lists
    service = PredictionService(db=db, league=league)
    result = service.predict_game(
        home_team=home,
        away_team=away,
        game_date=game_date,
        game_id=game_id,
        include_points=True
    )

    # Convert PredictionResult to dict format for backward compatibility
    if result.error:
        return {'error': result.error}

    return {
        'predicted_winner': result.predicted_winner,
        'home_win_prob': result.home_win_prob,
        'away_win_prob': result.away_win_prob,
        # Model-derived odds (converted from win probabilities) - NOT market odds
        'model_home_odds': result.home_odds,
        'model_away_odds': result.away_odds,
        'features_dict': result.features_dict,
        'feature_players': result.feature_players,
        'home_injured_players': result.home_injured_players,
        'away_injured_players': result.away_injured_players,
        'home_points_pred': result.home_points_pred,
        'away_points_pred': result.away_points_pred,
        'point_diff_pred': result.point_diff_pred
    }
