#!/usr/bin/env python3
"""
Flask Web Application for NBA Predictions with Player Management

Routes:
    /                    - Game list page (today's games or date from URL)
    /game/<game_id>      - Game detail page with player management
    /api/update-player   - Update player status (playing/starter)
    /api/predict         - Generate prediction with player config
"""

import sys
import os
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file, g
# fetch_dates, parse_date_range moved to core/pipeline/sync_pipeline
from bball_app.core.data.espn_client import ESPNClient
from bball_app.config import config
from bball_app.core.mongo import Mongo
from bball_app.core.models import NBAModel
from bball_app.core.models import PointsRegressionTrainer
from bball_app.core.services.matchup_chat import Controller as MatchupChatController
from bball_app.core.services.matchup_chat.schemas import ControllerOptions

# Import unified core infrastructure
from bball_app.core.services.config_manager import ModelConfigManager
from bball_app.core.models.artifact_loader import ArtifactLoader
from bball_app.core.services.business_logic import ModelBusinessLogic
from bball_app.core.services.artifacts import ArtifactManager
from bball_app.core.stats.per_calculator import PERCalculator

from bball_app.core.training import (
    evaluate_model_combo,
    evaluate_model_combo_with_calibration,
    create_model_with_c,
    get_latest_training_csv,
    get_best_config,
    compute_feature_importance,
    DEFAULT_MODEL_TYPES,
    DEFAULT_C_VALUES,
    C_SUPPORTED_MODELS,
    MODEL_CACHE_FILE,
    MODEL_CACHE_FILE_NO_PER
)
from bball_app.core.features.sets import get_features_by_sets, get_all_features

# Backward compatibility functions for default features
def get_default_classifier_features():
    """Get default classifier features from the feature registry."""
    return get_all_features(model_type=None)

def get_default_points_features():
    """Get default points features (empty list - points model uses its own features)."""
    return []
from bball_app.core.utils import get_season_from_date as get_season_from_date_core
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hashlib
from bson import ObjectId
import threading
import uuid
from functools import lru_cache
import uuid
from functools import lru_cache

# Team abbreviation mapping (ESPN team ID to our team abbreviations)
TEAM_ABBREV_MAP = {
    '1': 'ATL', '2': 'BOS', '3': 'BKN', '4': 'CHA', '5': 'CHI',
    '6': 'CLE', '7': 'DAL', '8': 'DEN', '9': 'DET', '10': 'GS',
    '11': 'HOU', '12': 'IND', '13': 'LAC', '14': 'LAL', '15': 'MEM',
    '16': 'MIA', '17': 'MIL', '18': 'MIN', '19': 'NO', '20': 'NY',
    '21': 'OKC', '22': 'ORL', '23': 'PHI', '24': 'PHX', '25': 'POR',
    '26': 'SAC', '27': 'SA', '28': 'TOR', '29': 'UTA', '30': 'WAS'
}

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize database connection
mongo = Mongo()
db = mongo.db

config_manager = ModelConfigManager(db)

# Context processor to provide league config to all templates
from bball_app.core.league_config import load_league_config, get_available_leagues

@app.context_processor
def inject_league_context():
    """Inject league configuration into all templates."""
    # Default to NBA, but could be determined from URL path or session
    league_id = getattr(g, 'league_id', None) or request.args.get('league', 'nba')
    try:
        league = load_league_config(league_id)
    except Exception:
        league = load_league_config('nba')
        league_id = 'nba'

    # Get available leagues for the dropdown
    available_leagues = []
    for lg_id in get_available_leagues():
        try:
            lg = load_league_config(lg_id)
            available_leagues.append({
                'id': lg_id,
                'display_name': lg.display_name
            })
        except Exception:
            pass

    return {
        'league': league,
        'league_id': league_id,
        'available_leagues': available_leagues,
    }

@app.before_request
def set_league_context():
    """Set league context on flask.g for use in view functions."""
    league_id = request.args.get('league', 'nba')
    # Also check URL path for league prefix (e.g., /cbb/...)
    path_parts = request.path.strip('/').split('/')
    available = set(get_available_leagues())
    if path_parts and path_parts[0] in available:
        league_id = path_parts[0]

    try:
        g.league = load_league_config(league_id)
        g.league_id = league_id
    except Exception:
        g.league = load_league_config('nba')
        g.league_id = 'nba'


def league_api_route(rule, **options):
    """
    Decorator that registers both plain and league-prefixed API routes.
    Usage: @league_api_route('/api/something', methods=['GET'])
    Registers both /api/something and /<league_id>/api/something
    """
    def decorator(f):
        # Register the plain route
        endpoint = options.pop('endpoint', None) or f.__name__
        app.add_url_rule(rule, endpoint, f, **options)
        # Register the league-prefixed route
        prefixed_rule = '/<league_id>' + rule
        app.add_url_rule(prefixed_rule, endpoint + '_league', f, **options)
        return f
    return decorator


def get_master_training_path() -> str:
    """Get master training CSV path from current league config."""
    league = getattr(g, 'league', None)
    if league:
        return league.master_training_csv
    # Fallback to NBA config if no league context
    from bball_app.core.league_config import load_league_config
    return load_league_config('nba').master_training_csv


# Add Jinja2 filter for formatting gametime to Eastern time
from pytz import timezone, utc
@app.template_filter('gametime_et')
def gametime_et_filter(gametime):
    """Convert UTC datetime to Eastern time and format as '7:00 PM'."""
    if not gametime:
        return None
    
    # Handle both datetime objects and strings
    if isinstance(gametime, str):
        try:
            # Try parsing ISO format
            if 'T' in gametime:
                gametime = datetime.fromisoformat(gametime.replace('Z', '+00:00'))
            else:
                return None
        except (ValueError, AttributeError):
            return None
    
    # Ensure it's timezone-aware (assume UTC if naive)
    if gametime.tzinfo is None:
        gametime = utc.localize(gametime)
    
    # Convert to Eastern time
    eastern = timezone('US/Eastern')
    et_time = gametime.astimezone(eastern)
    
    # Format as "7:00 PM"
    return et_time.strftime('%-I:%M %p').lstrip('0')

# Cache for PER calculator (shared across requests)
_per_calculator = None

# Cache for classifier model (shared across requests)
_nba_model = None
_nba_model_config_hash = None

# Cache for points model trainer (shared across requests)
_points_trainer = None
_points_trainer_model_path = None

def get_per_calculator():
    """Get or create PER calculator instance."""
    global _per_calculator
    if _per_calculator is None:
        _per_calculator = PERCalculator(db=db, preload=True)
    return _per_calculator


def create_ensemble_model(ensemble_config: dict, league=None):
    """
    Create an ensemble model from ensemble configuration.

    Args:
        ensemble_config: MongoDB config document with ensemble=True
        league: LeagueConfig for league-specific collection access

    Returns:
        NBAModel instance with ensemble loaded, or None if failed
    """
    try:
        from bball_app.core.training.stacking_trainer import StackingTrainer
        import uuid

        # Get ensemble configuration
        ensemble_models = ensemble_config.get('ensemble_models', [])
        ensemble_meta_features = ensemble_config.get('ensemble_meta_features', [])
        ensemble_use_disagree = ensemble_config.get('ensemble_use_disagree', False)
        ensemble_use_conf = ensemble_config.get('ensemble_use_conf', False)

        # Determine stacking mode based on whether any meta features are requested
        use_any_meta = ensemble_use_disagree or ensemble_use_conf or len(ensemble_meta_features) > 0
        stacking_mode = 'informed' if use_any_meta else 'naive'

        print(f"Creating ensemble with {len(ensemble_models)} base models")
        print(f"Stacking mode: {stacking_mode}")
        print(f"Use disagree: {ensemble_use_disagree}, Use conf: {ensemble_use_conf}")
        print(f"Custom meta features: {ensemble_meta_features}")

        # Create stacking trainer directly (no ModelerAgent needed)
        session_id = str(uuid.uuid4())
        stacking_trainer = StackingTrainer(db=db, league=league)

        # Get time-based calibration config
        dataset_spec = {
            'begin_year': ensemble_config.get('begin_year'),
            'calibration_years': ensemble_config.get('calibration_years'),
            'evaluation_year': ensemble_config.get('evaluation_year'),
            'use_master': True
        }

        # Train stacked model
        print("Training ensemble using StackingTrainer...")
        result = stacking_trainer.train_stacked_model(
            base_run_ids=ensemble_models,
            dataset_spec=dataset_spec,
            session_id=session_id,
            meta_c_value=0.1,  # Default C value for meta-model
            stacking_mode=stacking_mode,
            meta_features=ensemble_meta_features,
            use_disagree=ensemble_use_disagree,
            use_conf=ensemble_use_conf
        )
        
        if result and 'run_id' in result:
            print(f"Ensemble trained successfully: {result['run_id']}")
            
            # Create a special NBAModel instance for ensembles
            ensemble_model = NBAModel(
                classifier_features=[],
                points_features=[],
                include_elo=False,
                use_exponential_weighting=False,
                include_era_normalization=False,
                include_per_features=False,
                preload_data=False
            )
            
            # Mark as ensemble and store run info
            ensemble_model.is_ensemble = True
            ensemble_model.ensemble_run_id = result['run_id']
            ensemble_model.ensemble_config = ensemble_config
            ensemble_model.ensemble_base_models = ensemble_models
            ensemble_model.ensemble_meta_features = ensemble_meta_features
            
            # Store the stacking result for prediction use
            ensemble_model.ensemble_result = result
            
            return ensemble_model
        else:
            print(f"Ensemble training failed: {result}")
            return None
            
    except Exception as e:
        import traceback
        print(f"Error creating ensemble model: {e}")
        traceback.print_exc()
        return None


def get_bball_model():
    """Get or create NBAModel instance using unified infrastructure."""
    global _nba_model, _nba_model_config_hash
    
    # Get current selected config using unified manager
    selected_config = config_manager.get_selected_config()
    current_config_hash = selected_config.get('config_hash') if selected_config else None
    
    # Check if cached model matches current config
    if _nba_model is not None:
        if _nba_model_config_hash is not None:
            if current_config_hash != _nba_model_config_hash:
                print(f"Model config changed (hash: {_nba_model_config_hash} -> {current_config_hash}). Reloading model...")
                _nba_model = None
                _nba_model_config_hash = None
        elif current_config_hash is not None:
            print(f"Switching from cached model to MongoDB config (hash: {current_config_hash}). Reloading model...")
            _nba_model = None
            _nba_model_config_hash = None
    
    if _nba_model is None:
        # Check if selected config is an ensemble
        if selected_config and selected_config.get('ensemble', False):
            print("Loading ensemble model...")
            _nba_model = create_ensemble_model(selected_config, league=getattr(g, 'league', None))
            if _nba_model:
                _nba_model_config_hash = current_config_hash
                return _nba_model
            else:
                print("Failed to create ensemble model, falling back to regular model")
        
        # Create regular model using unified factory
        _nba_model = NBAModel(
            classifier_features=get_default_classifier_features(),
            points_features=get_default_points_features(),
            include_elo=True,
            use_exponential_weighting=False,
            include_era_normalization=False,
            include_per_features=True,
            preload_data=False
        )
        
        # Load model using unified factory from config
        if selected_config and not selected_config.get('ensemble', False):
            try:
                model, scaler, feature_names = ArtifactLoader.create_model(selected_config, use_artifacts=True)
                _nba_model.classifier_model = model
                _nba_model.scaler = scaler
                _nba_model.feature_names = feature_names
                _nba_model.classifier_features = feature_names
                _nba_model_config_hash = current_config_hash
                print(f"✅ Loaded model from config: {selected_config.get('model_type')} (artifacts: {'fast' if selected_config.get('model_artifact_path') else 'trained'})")
                return _nba_model
            except Exception as e:
                print(f"❌ Failed to load model from config: {e}")
                import traceback
                traceback.print_exc()
        
        # Fallback to cached model (try with PER first, then without)
        try:
            _nba_model.load_cached_model(no_per=False)
            _nba_model_config_hash = None
        except:
            try:
                _nba_model.load_cached_model(no_per=True)
                _nba_model_config_hash = None
            except:
                pass  # Model not cached yet
    
    return _nba_model


def get_points_model_trainer(preloaded_selected_config: dict = None, points_config_collection: str = None):
    """
    Get PointsRegressionTrainer instance with selected config from MongoDB.

    Returns:
        PointsRegressionTrainer instance with loaded model, or None if no config selected
    """
    try:
        from bball_app.core.models import PointsRegressionTrainer
        global _points_trainer, _points_trainer_model_path

        # Determine collection name
        if points_config_collection is None:
            try:
                from flask import g as flask_g
                points_config_collection = flask_g.league.collections.get('model_config_points', 'model_config_points_nba')
            except Exception:
                points_config_collection = 'model_config_points_nba'

        # Accept pre-fetched config to avoid extra DB call
        selected_config = preloaded_selected_config
        if selected_config is None:
            selected_config = db[points_config_collection].find_one({'selected': True})
        if not selected_config:
            print("No selected points model config found")
            return None

        model_path = selected_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"Points model file not found: {model_path}")
            return None

        # If cached trainer matches current model_path, reuse it
        if _points_trainer is not None and _points_trainer_model_path == model_path:
            return _points_trainer

        # Otherwise (re)load trainer for this model_path
        model_name = os.path.basename(model_path).replace('.pkl', '')

        # Create trainer (injury features will be auto-detected from model's feature list when making predictions)
        trainer = PointsRegressionTrainer(db=db)
        trainer.load_model(model_name)

        _points_trainer = trainer
        _points_trainer_model_path = model_path

        print(f"Loaded points model: {model_name}")
        return _points_trainer

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error loading points model: {e}")
        return None


# =============================================================================
# MODEL CONFIG HELPER FUNCTIONS
# =============================================================================

def load_features_from_master_csv():
    """
    Load features from MASTER_TRAINING.csv and organize them into feature sets.
    
    This function uses the master CSV headers as the source of truth for available features.
    All features in the CSV (except metadata columns) will be included in the sidebar.
    If a feature is not in the CSV, it will not appear in the sidebar.
    
    To update available features, regenerate the master CSV using:
        python cli/generate_master_training.py
    
    Returns:
        tuple: (feature_sets_dict, feature_set_descriptions_dict)
    """
    try:
        import csv
        from pathlib import Path
        from collections import defaultdict

        # Get the master training CSV path from league config
        master_training_path = get_master_training_path()

        if not os.path.exists(master_training_path):
            print(f"Warning: Master training CSV not found at {master_training_path}")
            # Fallback to empty sets
            return {}, {}

        # Read CSV headers
        with open(master_training_path, 'r') as f:
            reader = csv.reader(f)
            headers = next(reader)

        # Filter out metadata columns (not features)
        metadata_cols = {'Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id', 'home_points', 'away_points'}
        features = sorted([h for h in headers if h not in metadata_cols])

        # Log for debugging
        print(f"[load_features_from_master_csv] Loaded {len(features)} features from {master_training_path}")
        
        # Group features into sets based on naming patterns
        feature_sets = defaultdict(list)
        
        for feature in features:
            feature_lower = feature.lower()
            
            # Check for point prediction features first (pred_*)
            if feature.startswith('pred_'):
                feature_sets['point_predictions'].append(feature)
                continue
            
            # Parse feature name to determine set
            if '|' in feature:
                parts = feature.split('|')
                stat_name = parts[0].lower()
                
                # Determine feature set based on stat name and patterns
                # Check in order of specificity (more specific first)
                if 'inj' in stat_name or 'inj' in feature_lower:
                    feature_sets['injuries'].append(feature)
                elif stat_name.startswith('player_') or 'team_per' in stat_name or 'starters_per' in stat_name or \
                     'per1' in stat_name or 'per2' in stat_name or 'per3' in stat_name or 'per_available' in feature_lower:
                    feature_sets['player_talent'].append(feature)
                elif 'elo' in stat_name:
                    feature_sets['elo_strength'].append(feature)
                elif 'rel' in feature_lower:
                    feature_sets['era_normalization'].append(feature)
                elif 'off_rtg' in stat_name or 'assists_ratio' in stat_name:
                    feature_sets['offensive_engine'].append(feature)
                elif 'def_rtg' in stat_name or 'blocks' in stat_name or 'reb_total' in stat_name or 'reb_' in stat_name or 'turnovers' in stat_name:
                    feature_sets['defensive_engine'].append(feature)
                elif 'efg' in stat_name or 'ts' in stat_name or 'three' in stat_name:
                    feature_sets['shooting_efficiency'].append(feature)
                elif 'points' in stat_name or 'wins' in stat_name:
                    feature_sets['outcome_strength'].append(feature)
                elif 'pace' in stat_name or 'std' in feature_lower:
                    feature_sets['pace_volatility'].append(feature)
                elif 'b2b' in stat_name or 'travel' in stat_name or 'rest' in feature_lower:
                    feature_sets['schedule_fatigue'].append(feature)
                elif 'games_played' in stat_name:
                    if 'days' in feature_lower or 'diff' in feature_lower:
                        feature_sets['schedule_fatigue'].append(feature)
                    else:
                        feature_sets['sample_size'].append(feature)
                else:
                    # Default: put in absolute_magnitude
                    feature_sets['absolute_magnitude'].append(feature)
            else:
                # Old format features - try to categorize
                if 'Per' in feature or ('per' in feature_lower and 'percent' not in feature_lower):
                    feature_sets['player_talent'].append(feature)
                elif 'Inj' in feature or 'inj' in feature_lower:
                    feature_sets['injuries'].append(feature)
                elif 'Pace' in feature or 'pace' in feature_lower:
                    feature_sets['pace_volatility'].append(feature)
                elif 'B2B' in feature or 'Travel' in feature or 'GamesLast' in feature or 'rest' in feature_lower:
                    feature_sets['schedule_fatigue'].append(feature)
                elif 'Rel' in feature or 'rel' in feature_lower:
                    feature_sets['era_normalization'].append(feature)
                elif 'Elo' in feature or 'elo' in feature_lower:
                    feature_sets['elo_strength'].append(feature)
                elif 'GamesPlayed' in feature:
                    feature_sets['sample_size'].append(feature)
                else:
                    feature_sets['absolute_magnitude'].append(feature)
        
        # Convert to regular dict and ensure all expected sets exist (even if empty)
        feature_sets_dict = dict(feature_sets)
        
        # Ensure all expected sets exist (even if empty)
        expected_sets = [
            'outcome_strength', 'shooting_efficiency', 'offensive_engine', 
            'defensive_engine', 'pace_volatility', 'schedule_fatigue', 
            'sample_size', 'elo_strength', 'era_normalization', 
            'player_talent', 'absolute_magnitude', 'injuries', 'point_predictions'
        ]
        for set_name in expected_sets:
            if set_name not in feature_sets_dict:
                feature_sets_dict[set_name] = []
        
        # Feature set descriptions
        feature_set_descriptions = {
            "outcome_strength": "Highest-level 'scoreboard' signals: points and wins across multiple time windows",
            "shooting_efficiency": "How efficiently teams turn possessions into points (shot quality & spacing)",
            "offensive_engine": "Possession-level offensive quality and ball movement (process-oriented metrics)",
            "defensive_engine": "Prevent opponent scoring / control possessions (defense + possession control)",
            "pace_volatility": "Tempo (possessions) and consistency (volatility) metrics",
            "schedule_fatigue": "Short-term fatigue and schedule density (rest, back-to-backs, recent games)",
            "sample_size": "Data reliability signal (how much information available per team)",
            "elo_strength": "Single high-signal summary of team strength (meta-feature over other signals)",
            "era_normalization": "Normalized relative to league average (removes era effects)",
            "player_talent": "Player-level talent aggregations (PER-based team metrics)",
            "absolute_magnitude": "Non-differential magnitude features (absolute team performance levels)",
            "injuries": "Injury impact features (PER value lost, rotation impact, severity)",
            "point_predictions": "Point prediction model outputs (pred_margin, pred_home_points, pred_away_points, pred_point_total)",
        }
        
        # Log feature counts per set for debugging
        print(f"[load_features_from_master_csv] Feature counts by set:")
        for set_name in sorted(feature_sets_dict.keys()):
            count = len(feature_sets_dict[set_name])
            if count > 0:
                print(f"  {set_name}: {count} features")
        
        return feature_sets_dict, feature_set_descriptions
    
    except Exception as e:
        import traceback
        print(f"Error loading features from master CSV: {e}")
        traceback.print_exc()
        # Fallback to empty sets
        return {}, {}


def _get_venv_python_executable() -> str:
    """
    Prefer the repo-local venv Python when available.

    This matters because the web app is sometimes launched with a system/Homebrew Python,
    but background jobs (feature regen / add-columns) rely on deps installed in the repo venv.
    """
    # web/app.py lives at: nba_app/web/app.py
    # venv lives at:        nba_app/venv/bin/python
    venv_python = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'venv', 'bin', 'python'))
    return venv_python if os.path.exists(venv_python) else sys.executable


def _spawn_master_training_job(cmd: list, job_id: str, job_type: str, league=None) -> int:
    """
    Spawn a non-blocking subprocess and attach logs + pid to the Mongo job record.

    Returns:
        pid of spawned process
    """
    import subprocess
    from datetime import datetime

    logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'job_logs'))
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{job_type}_{job_id}.log")

    # Redirect output to file to avoid PIPE deadlocks + make debugging easy
    with open(log_path, 'ab', buffering=0) as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(os.path.abspath(cmd[1])),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            close_fds=True
        )

    # Persist pid + log path for observability
    jobs_collection = g.league.collections.get('jobs', 'jobs_nba') if (league is None and hasattr(g, 'league') and g.league) else (league.collections.get('jobs', 'jobs_nba') if league else 'jobs_nba')
    db[jobs_collection].update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'metadata.pid': proc.pid,
            'metadata.log_path': log_path,
            'message': f"Subprocess started (pid={proc.pid}). Logging to {log_path}",
            'updated_at': datetime.utcnow()
        }}
    )

    return proc.pid


# Use core layer's single source of truth for feature set hash generation
generate_feature_set_hash = ModelConfigManager.generate_feature_set_hash


# Use core layer's single source of truth for config hash generation
generate_config_hash = ModelConfigManager.generate_config_hash


def _parse_training_config(config: dict) -> dict:
    """
    Parse and type-convert all training config fields from request JSON.

    This is a web-layer helper that handles string-to-type coercion from the
    frontend. Both save_model_config() and train_model_config() share this.

    Args:
        config: Raw request JSON dict

    Returns:
        Dict with parsed, type-safe values
    """
    model_types = config.get('model_types', [])
    features = config.get('features', [])
    feature_sets = config.get('feature_sets', [])
    c_values = config.get('c_values', [])

    # Bool coercion helper
    def _to_bool(val, default=False):
        if isinstance(val, str):
            return val.lower() in ('true', '1', 'yes')
        return bool(val) if val is not None else default

    use_time_calibration = _to_bool(config.get('use_time_calibration', False))
    calibration_method = config.get('calibration_method', 'isotonic') if use_time_calibration else None
    begin_year = config.get('begin_year') if use_time_calibration else None
    calibration_years = config.get('calibration_years') if use_time_calibration else None
    evaluation_year = config.get('evaluation_year') if use_time_calibration else None

    # Backward compat: calibration_year (singular) → list
    if not calibration_years and config.get('calibration_year'):
        calibration_years = [config.get('calibration_year')]

    # Normalize year fields to ints
    if begin_year is not None:
        try:
            begin_year = int(begin_year)
        except (ValueError, TypeError):
            begin_year = None
    if calibration_years is not None:
        try:
            if isinstance(calibration_years, str):
                calibration_years = [int(y.strip()) for y in calibration_years.split(',') if y.strip()]
            elif isinstance(calibration_years, list):
                calibration_years = [int(y) for y in calibration_years]
            else:
                calibration_years = [int(calibration_years)]
        except (ValueError, TypeError):
            calibration_years = None
    if evaluation_year is not None:
        try:
            evaluation_year = int(evaluation_year)
        except (ValueError, TypeError):
            evaluation_year = None

    include_injuries = _to_bool(config.get('include_injuries', False))
    recency_decay_k = None
    if include_injuries:
        rdk_raw = config.get('recency_decay_k')
        if rdk_raw is not None:
            try:
                recency_decay_k = float(rdk_raw)
            except (ValueError, TypeError):
                recency_decay_k = 15.0
        else:
            recency_decay_k = 15.0

    # Min games played
    try:
        min_games_played = int(config.get('min_games_played', 15))
    except (ValueError, TypeError):
        min_games_played = 15

    # Exclude seasons
    exclude_seasons = config.get('exclude_seasons')
    if exclude_seasons is not None:
        try:
            if isinstance(exclude_seasons, str):
                exclude_seasons = [int(y.strip()) for y in exclude_seasons.split(',') if y.strip()]
            elif isinstance(exclude_seasons, list):
                exclude_seasons = [int(y) for y in exclude_seasons]
            else:
                exclude_seasons = [int(exclude_seasons)]
            if not exclude_seasons:
                exclude_seasons = None
        except (ValueError, TypeError):
            exclude_seasons = None

    # use_master — always True for model-config UI workflow
    use_master = _to_bool(config.get('use_master', True), default=True)

    point_model_id = config.get('point_model_id')  # Can be None

    # Resolve features from feature_sets if needed
    if features:
        requested_features = features
    elif feature_sets:
        requested_features = get_features_by_sets(feature_sets)
    else:
        requested_features = []

    return {
        'model_types': model_types,
        'features': requested_features,
        'feature_sets': feature_sets,
        'raw_features': features,  # Original from request (for run_training_job)
        'c_values': c_values,
        'use_time_calibration': use_time_calibration,
        'calibration_method': calibration_method,
        'begin_year': begin_year,
        'calibration_years': calibration_years,
        'evaluation_year': evaluation_year,
        'use_master': use_master,
        'include_injuries': include_injuries,
        'recency_decay_k': recency_decay_k,
        'min_games_played': min_games_played,
        'exclude_seasons': exclude_seasons,
        'point_model_id': point_model_id,
    }


def sanitize_nan(value):
    """Convert NaN, Infinity, and -Infinity to None for JSON serialization."""
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        elif np.isinf(value):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def sanitize_training_results(training_results):
    """
    Remove non-serializable objects (like model instances) from training_results.
    
    Args:
        training_results: Dictionary returned from PointsRegressionTrainer.train()
        
    Returns:
        Sanitized dictionary with only JSON-serializable data
    """
    sanitized = {
        'model_type': training_results.get('model_type'),
        'final_metrics': {},
        'feature_names': list(training_results.get('feature_names', [])),
        'n_samples': int(training_results.get('n_samples', 0)),
        'n_features': int(training_results.get('n_features', 0))
    }
    
    # Sanitize final_metrics
    final_metrics = training_results.get('final_metrics', {})
    for key, value in final_metrics.items():
        if isinstance(value, (float, np.floating)):
            sanitized['final_metrics'][key] = sanitize_nan(value)
        elif isinstance(value, (int, np.integer)):
            sanitized['final_metrics'][key] = int(value)
        else:
            sanitized['final_metrics'][key] = value
    
    # Sanitize cv_results - remove model objects, keep only metrics
    if 'cv_results' in training_results:
        sanitized_cv_results = []
        for result in training_results['cv_results']:
            sanitized_result = {}
            for key, value in result.items():
                # Skip model objects (they're not JSON serializable)
                if key in ['home_model', 'away_model']:
                    continue
                # Sanitize NaN values and convert numpy types
                if isinstance(value, (float, np.floating)):
                    sanitized_result[key] = sanitize_nan(value)
                elif isinstance(value, (int, np.integer)):
                    sanitized_result[key] = int(value)
                else:
                    sanitized_result[key] = value
            sanitized_cv_results.append(sanitized_result)
        sanitized['cv_results'] = sanitized_cv_results
    
    return sanitized


def save_artifacts_for_trained_model(model_instance, config_doc: dict, df, scaler):
    """Helper function to save artifacts for trained models."""
    import uuid
    
    run_id = str(uuid.uuid4())  # Generate unique run_id for artifacts
    config_id = str(config_doc.get('_id')) if '_id' in config_doc else None
    
    # Get the actual model that was trained (base or calibrated)
    trained_model = model_instance.calibrated_model or model_instance.classifier_model
    
    # Get feature names from training data
    meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id', 'home_points', 'away_points']
    feature_names = [col for col in df.columns if col not in meta_cols]
    
    # Save artifacts
    artifact_result = save_model_artifacts(
        model=trained_model,
        scaler=scaler,
        feature_names=feature_names,
        run_id=run_id,
        config_id=config_id
    )
    
    if artifact_result['success']:
        print(f"✅ Model artifacts saved successfully for {config_id[:8] if config_id else run_id[:8]}")
    else:
        print(f"❌ Failed to save model artifacts: {artifact_result.get('error')}")


def save_model_artifacts(model, scaler, feature_names, run_id: str, config_id: str = None, config_collection: str = 'nba_model_config') -> dict:
    """
    Save trained model artifacts to disk and return file paths.
    
    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler
        feature_names: List of feature names
        run_id: Run identifier for filename
        config_id: Optional MongoDB config ID
        
    Returns:
        Dict with artifact paths
    """
    import os
    import pickle
    
    # Create models directory if it doesn't exist
    models_dir = 'cli/models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Generate artifact file paths
    model_path = f'{models_dir}/{run_id}_model.pkl'
    scaler_path = f'{models_dir}/{run_id}_scaler.pkl'
    features_path = f'{models_dir}/{run_id}_features.json'
    
    try:
        # Save model artifact
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ Saved model artifact: {model_path}")
        
        # Save scaler artifact
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Saved scaler artifact: {scaler_path}")
        
        # Save feature names
        import json
        with open(features_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✅ Saved feature names: {features_path}")
        
        # If we have a config_id, update MongoDB with artifact paths
        if config_id:
            from bson import ObjectId
            db[config_collection].update_one(
                {'_id': ObjectId(config_id)},
                {'$set': {
                    'model_artifact_path': model_path,
                    'scaler_artifact_path': scaler_path,
                    'features_path': features_path,
                    'run_id': run_id,  # Store run_id for artifact loading
                    'artifacts_saved_at': datetime.utcnow()
                }}
            )
            print(f"✅ Updated MongoDB config {config_id[:8]} with artifact paths")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'features_path': features_path,
            'success': True
        }
        
    except Exception as e:
        print(f"❌ Error saving model artifacts: {e}")
        return {
            'success': False,
            'error': str(e)
        }


def load_model_from_mongo_config(model_instance: NBAModel, config_doc: dict) -> bool:
    """
    Load and train model from a MongoDB config document.
    If time-based calibration is enabled, creates a calibrated model using incremented calibration years.
    Also saves model artifacts for fast loading in future ensemble training.
    
    Args:
        model_instance: NBAModel instance to load into
        config_doc: MongoDB config document from model_config_nba
        
    Returns:
        True if successful, False otherwise
    """
    try:
        training_csv = config_doc.get('training_csv')
        if not training_csv or not os.path.exists(training_csv):
            return False
        
        model_type = config_doc.get('model_type')
        best_c_value = config_doc.get('best_c_value')
        
        # Set classifier CSV path
        model_instance.classifier_csv = training_csv
        
        # Load data
        df = pd.read_csv(training_csv)
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away']
        target_col = 'HomeWon'
        feature_cols = [c for c in df.columns if c not in meta_cols + [target_col]]
        
        # Filter to only features in the config (if specified)
        config_features = config_doc.get('features', [])
        if config_features:
            # Only use features that are in both the CSV and the config
            feature_cols = [f for f in feature_cols if f in config_features]
        
        X = df[feature_cols].values
        y = df[target_col].values
        model_instance.feature_names = feature_cols
        
        # NEW ARCHITECTURE: Feature names are self-describing (e.g., 'off_rtg|season|avg|home')
        # StatHandlerV2 uses calculate_feature() which parses feature names directly - no stat tokens needed!
        # StatHandlerV2 is initialized with statistics=[] (empty list) in the new architecture
        
        # Set classifier_features to feature_cols for backward compatibility (though it's not used in prediction)
        # In the new architecture, feature_names is the source of truth for what to calculate
        model_instance.classifier_features = feature_cols
        
        # Auto-detect feature types from feature_names for display/logging purposes only
        # At prediction time, _build_features_dict calculates whatever is in feature_names regardless of flags
        has_per_features = any(fname.startswith('player_') or fname.startswith('per_available') for fname in feature_cols)
        has_injury_features = any(fname.startswith('inj_') for fname in feature_cols)
        has_elo_features = any('elo' in fname.lower() for fname in feature_cols)
        has_absolute_features = any('|home' in fname or '|away' in fname or '_home_abs' in fname or '_away_abs' in fname for fname in feature_cols)
        has_era_normalization = any('rel' in fname.lower() for fname in feature_cols)
        
        # Set flags for backward compatibility (they don't control prediction behavior - feature_names does)
        model_instance.include_per_features = has_per_features
        model_instance.include_injuries = has_injury_features
        model_instance.include_elo = has_elo_features
        model_instance.include_absolute = has_absolute_features
        model_instance.include_era_normalization = has_era_normalization
        model_instance.include_enhanced_features = True  # Always True in new architecture (enhanced features are just regular features)
        
        # Set recency_decay_k if injuries are present (needed for injury feature calculation)
        if has_injury_features:
            model_instance.recency_decay_k = config_doc.get('recency_decay_k', 15.0)
        
        print(f"Loaded model with {len(feature_cols)} features (new architecture: feature names are self-describing)")
        
        # Handle NaN values before scaling
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            # Drop rows with NaN
            X = X[~nan_mask]
            y = y[~nan_mask]
            df = df[~nan_mask].copy().reset_index(drop=True)
        
        # Standardize
        model_instance.scaler = StandardScaler()
        X_scaled = model_instance.scaler.fit_transform(X)
        
        # Check if time-based calibration is enabled
        use_time_calibration = config_doc.get('use_time_calibration', False)
        calibration_method = config_doc.get('calibration_method', 'isotonic') if use_time_calibration else None
        calibration_years = config_doc.get('calibration_years')
        evaluation_year = config_doc.get('evaluation_year')
        
        # Handle backward compatibility for single calibration_year
        if not calibration_years and config_doc.get('calibration_year'):
            calibration_years = [config_doc.get('calibration_year')]
        
        if use_time_calibration and calibration_years and evaluation_year:
            # Increment calibration years and evaluation year for prediction
            # e.g., [2022, 2023] -> [2023, 2024], evaluation_year 2024 -> 2025
            incremented_calibration_years = [year + 1 for year in calibration_years]
            incremented_evaluation_year = evaluation_year + 1
            
            print(f"Time-based calibration enabled. Incrementing years:")
            print(f"  Training calibration years: {calibration_years} -> Prediction calibration years: {incremented_calibration_years}")
            print(f"  Training evaluation year: {evaluation_year} -> Prediction evaluation year: {incremented_evaluation_year}")
            
            # Create calibrated model using incremented years
            from sklearn.calibration import CalibratedClassifierCV, IsotonicRegression
            # Note: numpy is already imported at the top of the file as np
            
            # Split data using incremented years
            df_copy = df.copy()
            earliest_cal_year = min(incremented_calibration_years)
            
            # Training set: all data before earliest calibration year
            train_mask = df_copy['Year'] < earliest_cal_year
            X_train = X_scaled[train_mask]
            y_train = y[train_mask]
            
            # Calibration set: data from incremented calibration years (combined)
            cal_mask = df_copy['Year'].isin(incremented_calibration_years)
            X_cal = X_scaled[cal_mask]
            y_cal = y[cal_mask]
            
            print(f"  Train set: {len(X_train)} games (Year < {earliest_cal_year})")
            print(f"  Calibration set: {len(X_cal)} games (Year in {incremented_calibration_years})")
            
            if len(X_train) == 0:
                print(f"  Warning: Training set is empty for incremented calibration. Using all data for training.")
                X_train = X_scaled
                y_train = y
            
            if len(X_cal) == 0:
                print(f"  Warning: Calibration set is empty for incremented years {incremented_calibration_years}. Cannot create calibrated model.")
                # Fall back to uncalibrated model
                model_instance.classifier_model = create_model_with_c(model_type, best_c_value)
                model_instance.classifier_model.fit(X_scaled, y)
                model_instance.calibrated_model = None
                return True
            
            # Train base model
            base_model = create_model_with_c(model_type, best_c_value)
            base_model.fit(X_train, y_train)
            model_instance.classifier_model = base_model
            
            # Get raw predictions on calibration set
            y_proba_raw_cal = base_model.predict_proba(X_cal)
            
            # Create calibrated model wrapper that handles both isotonic and sigmoid
            if calibration_method == 'isotonic':
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(y_proba_raw_cal[:, 1], y_cal)
                # Create a wrapper class for isotonic calibration that mimics CalibratedClassifierCV interface
                class IsotonicCalibratedModel:
                    def __init__(self, base_model, calibrator):
                        self.base_model = base_model
                        self.calibrator = calibrator
                    
                    def predict(self, X):
                        return self.base_model.predict(X)
                    
                    def predict_proba(self, X):
                        raw_proba = self.base_model.predict_proba(X)
                        calibrated_1 = self.calibrator.predict(raw_proba[:, 1])
                        # Ensure probabilities sum to 1
                        calibrated_1 = np.clip(calibrated_1, 0.0, 1.0)
                        return np.column_stack([1 - calibrated_1, calibrated_1])
                
                model_instance.calibrated_model = IsotonicCalibratedModel(base_model, calibrator)
                model_instance._calibration_method = 'isotonic'
                print(f"  Created isotonic calibrated model using {len(X_cal)} calibration games")
                
                # Save artifacts for calibrated model
                save_artifacts_for_trained_model(model_instance, config_doc, df, scaler)
                
            elif calibration_method == 'sigmoid':
                # Use sigmoid (Platt scaling) - create wrapper to avoid cv='prefit' compatibility issues
                from sklearn.linear_model import LogisticRegression
                sigmoid_calibrator = LogisticRegression()
                sigmoid_calibrator.fit(y_proba_raw_cal[:, 1].reshape(-1, 1), y_cal)
                
                class SigmoidCalibratedModel:
                    def __init__(self, base_model, calibrator):
                        self.base_model = base_model
                        self.calibrator = calibrator
                    
                    def predict(self, X):
                        return self.base_model.predict(X)
                    
                    def predict_proba(self, X):
                        raw_proba = self.base_model.predict_proba(X)
                        calibrated_1 = self.calibrator.predict_proba(raw_proba[:, 1].reshape(-1, 1))[:, 1]
                        # Ensure probabilities sum to 1
                        calibrated_1 = np.clip(calibrated_1, 0.0, 1.0)
                        return np.column_stack([1 - calibrated_1, calibrated_1])
                
                calibrated_cv = SigmoidCalibratedModel(base_model, sigmoid_calibrator)
                model_instance.calibrated_model = calibrated_cv
                model_instance._calibration_method = 'sigmoid'
                print(f"  Created sigmoid calibrated model using {len(X_cal)} calibration games")
                
                # Save artifacts for calibrated model
                save_artifacts_for_trained_model(model_instance, config_doc, df, scaler)
            else:
                print(f"  Warning: Unknown calibration method '{calibration_method}'. Using uncalibrated model.")
                model_instance.calibrated_model = None
        else:
            # No calibration - train regular model
            model_instance.classifier_model = create_model_with_c(model_type, best_c_value)
            model_instance.classifier_model.fit(X_scaled, y)
            model_instance.calibrated_model = None
        
        # Save model artifacts for fast ensemble loading
        save_artifacts_for_trained_model(model_instance, config_doc, df, scaler)
        
        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False


def get_season_from_date(game_date: date) -> str:
    """Get NBA season string from date."""
    if game_date.month > 8:  # Oct-Dec
        return f"{game_date.year}-{game_date.year + 1}"
    else:  # Jan-Jun
        return f"{game_date.year - 1}-{game_date.year}"


def extract_and_update_teams(game_summary: Dict, teams_collection: str = 'nba_teams'):
    """
    Extract team information from game summary and upsert to teams collection.
    Uses team id as the upsert query key.

    Args:
        game_summary: ESPN game summary response
        teams_collection: Name of the teams collection (e.g., 'nba_teams', 'cbb_teams')
    """
    header = game_summary.get('header', {})
    competitions = header.get('competitions', [])
    
    if not competitions:
        return
    
    for competition in competitions:
        competitors = competition.get('competitors', [])
        
        for competitor in competitors:
            team = competitor.get('team', {})
            team_id = team.get('id')
            
            if not team_id:
                continue
            
            # Extract team data - keep API key names, only include truthy values
            update_data = {
                'team_id': str(team_id),  # Required: used as upsert query key
                'last_update': datetime.utcnow()
            }
            
            # Only add fields if they are truthy (not null, not empty string, not False)
            # Keep all key names as they are in the API
            for key, value in team.items():
                # Skip null, empty strings, and False values
                if value is not None and value != '' and value is not False:
                    # Special handling for logos array - extract first logo URL
                    if key == 'logos' and isinstance(value, list) and len(value) > 0:
                        # Get the first logo's href
                        first_logo = value[0]
                        if isinstance(first_logo, dict) and first_logo.get('href'):
                            update_data['logo'] = first_logo['href']
                        # Also store the full logos array
                        update_data['logos'] = value
                    else:
                        # Include the value (can be string, number, dict, list, etc.)
                        update_data[key] = value
            
            # Special handling for abbreviation - convert to uppercase if it exists
            if 'abbreviation' in update_data and update_data['abbreviation']:
                update_data['abbreviation'] = str(update_data['abbreviation']).upper()
            
            # If logo wasn't set from logos array but logos array exists, try to extract it
            if 'logo' not in update_data and 'logos' in update_data:
                logos = update_data.get('logos', [])
                if isinstance(logos, list) and len(logos) > 0:
                    first_logo = logos[0]
                    if isinstance(first_logo, dict) and first_logo.get('href'):
                        update_data['logo'] = first_logo['href']
            
            # Upsert to teams collection using team_id as the query key
            db[teams_collection].update_one(
                {'team_id': str(team_id)},  # Query by team_id
                {'$set': update_data},
                upsert=True
            )


def extract_and_update_player_roster(game_summary: Dict, home_team: str, away_team: str):
    """
    Extract player roster information from game summary and update players_nba collection.
    This includes headshots, names, positions, etc.
    
    IMPORTANT: Match teams by ID (not abbreviation) to avoid misassignment.
    Get home/away team IDs from competitors array which has homeAway field.
    """
    boxscore = game_summary.get('boxscore', {})
    players = boxscore.get('players', [])
    
    if not players:
        return
    
    # Get home and away team IDs from competitors array (more reliable than abbreviations)
    header = game_summary.get('header', {})
    competitors = header.get('competitions', [{}])[0].get('competitors', [])
    home_team_id = None
    away_team_id = None
    
    for comp in competitors:
        comp_team = comp.get('team', {})
        comp_team_id = comp_team.get('id')
        if comp.get('homeAway') == 'home':
            home_team_id = comp_team_id
        else:
            away_team_id = comp_team_id
    
    for team_players in players:
        team_info = team_players.get('team', {})
        team_id = team_info.get('id')
        
        # Match team by ID (more reliable than abbreviation matching)
        matching_team = None
        if team_id == home_team_id:
            matching_team = home_team
        elif team_id == away_team_id:
            matching_team = away_team
        else:
            # Fallback to abbreviation matching if team IDs don't match (shouldn't happen normally)
            team_abbrev_api = team_info.get('abbreviation', '').upper()
            team_abbrev = TEAM_ABBREV_MAP.get(str(team_id)) or team_abbrev_api
            for game_team in [home_team, away_team]:
                if (team_abbrev == game_team or 
                    team_abbrev_api == game_team or
                    team_abbrev.upper() == game_team.upper() or
                    team_abbrev_api.upper() == game_team.upper()):
                    matching_team = game_team
                    break
        
        if not matching_team:
            continue
        
        # Get athletes from statistics
        statistics = team_players.get('statistics', [])
        for stat_group in statistics:
            athletes = stat_group.get('athletes', [])
            for athlete_data in athletes:
                athlete = athlete_data.get('athlete', {})
                player_id = athlete.get('id')
                
                if not player_id:
                    continue
                
                # Extract player info
                update_data = {
                    'team': matching_team,
                    'last_roster_update': datetime.utcnow()
                }
                
                # Headshot
                if athlete.get('headshot'):
                    headshot_url = athlete['headshot'].get('href')
                    if headshot_url:
                        update_data['headshot'] = headshot_url
                
                # Name
                if athlete.get('displayName'):
                    update_data['player_name'] = athlete['displayName']
                if athlete.get('shortName'):
                    update_data['short_name'] = athlete['shortName']
                
                # Position
                if athlete.get('position'):
                    if athlete['position'].get('displayName'):
                        update_data['pos_display_name'] = athlete['position']['displayName']
                    if athlete['position'].get('name'):
                        update_data['pos_name'] = athlete['position']['name']
                    if athlete['position'].get('abbreviation'):
                        update_data['pos_abbreviation'] = athlete['position']['abbreviation']
                
                # Jersey number
                if athlete.get('jersey'):
                    update_data['jersey'] = athlete['jersey']
                
                # Starter status (from athlete_data, not athlete)
                if 'starter' in athlete_data:
                    update_data['is_starter'] = athlete_data.get('starter', False)
                
                # Update players_nba
                db[g.league.collections.get("players", "nba_players")].update_one(
                    {'player_id': str(player_id)},
                    {'$set': update_data},
                    upsert=True
                )


def extract_and_update_injuries(game_summary: Dict, home_team: str, away_team: str):
    """
    Extract injury data from ESPN API game summary and update players_nba collection.
    
    Args:
        game_summary: Full game summary response from ESPN API
        home_team: Home team abbreviation
        away_team: Away team abbreviation
    """
    injuries = game_summary.get('injuries', [])
    if not injuries:
        return
    
    # First, reset all injury statuses for these teams
    db[g.league.collections.get("players", "nba_players")].update_many(
        {'team': {'$in': [home_team, away_team]}},
        {
            '$set': {
                'injured': False,
                'injury_status': None
            }
        }
    )
    
    for team_injury_group in injuries:
        team_info = team_injury_group.get('team', {})
        team_abbrev_api = team_info.get('abbreviation', '').upper()
        team_id = team_info.get('id')
        team_abbrev = TEAM_ABBREV_MAP.get(str(team_id)) or team_abbrev_api
        
        # Match team by abbreviation (try both API abbrev and mapped abbrev)
        # Also try matching without case sensitivity
        matching_team = None
        for game_team in [home_team, away_team]:
            if (team_abbrev == game_team or 
                team_abbrev_api == game_team or
                team_abbrev.upper() == game_team.upper() or
                team_abbrev_api.upper() == game_team.upper()):
                matching_team = game_team
                break
        
        # Only process injuries for teams in this game
        if not matching_team:
            continue
        
        injury_list = team_injury_group.get('injuries', [])
        
        for injury in injury_list:
            athlete = injury.get('athlete', {})
            player_id = athlete.get('id')
            status = injury.get('status', '')
            
            if not player_id:
                continue
            
            # Determine injury status
            # Check fantasyStatus for GTD (Game Time Decision)
            details = injury.get('details', {})
            fantasy_status = details.get('fantasyStatus', {})
            fantasy_desc = fantasy_status.get('description', '').upper()
            
            # Determine injury status: "Out", "GTD", or "Day-To-Day"
            if status.lower() == 'out':
                injury_status = 'Out'
                is_injured = True
            elif fantasy_desc == 'GTD' or 'GTD' in fantasy_desc:
                injury_status = 'GTD'
                is_injured = False  # GTD players are not marked as injured
            else:
                injury_status = status  # "Day-To-Day" or other
                is_injured = False
            
            # Get headshot from athlete info
            headshot_url = None
            if athlete.get('headshot'):
                headshot_url = athlete['headshot'].get('href')
            
            # Update players_nba with injury info and headshot
            update_data = {
                'injured': is_injured,
                'injury_status': injury_status,
                'injury_date': injury.get('date'),
                'injury_details': details,
                'team': matching_team,
                'last_injury_update': datetime.utcnow()
            }
            
            # Add headshot if available
            if headshot_url:
                update_data['headshot'] = headshot_url
            
            # Also update player name and other metadata if available
            if athlete.get('displayName'):
                update_data['player_name'] = athlete['displayName']
            if athlete.get('shortName'):
                update_data['short_name'] = athlete['shortName']
            if athlete.get('position'):
                if athlete['position'].get('displayName'):
                    update_data['pos_display_name'] = athlete['position']['displayName']
                if athlete['position'].get('name'):
                    update_data['pos_name'] = athlete['position']['name']
            
            db[g.league.collections.get("players", "nba_players")].update_one(
                {'player_id': str(player_id)},
                {'$set': update_data},
                upsert=True  # Create if doesn't exist
            )


@app.route('/')
@app.route('/<league_id>/')
def index(league_id=None):
    """Main page: show today's games or games for date in URL."""
    date_str = request.args.get('date')
    
    if date_str:
        try:
            game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        except ValueError:
            game_date = date.today()
    else:
        game_date = date.today()
    
    # Get games for this date - use league-aware ESPN client
    # Use site API which includes venue info (header API doesn't have venue IDs)
    espn_client = ESPNClient(league=g.league)
    scoreboard = espn_client.get_scoreboard_site(game_date)
    games = []

    # Debug: Check what we got from API
    if scoreboard:
        events_count = len(scoreboard.get('events', []))
        print(f"DEBUG: Scoreboard (site API) has {events_count} events for {game_date}")

    # Try API scoreboard first - site API has events directly at top level
    if scoreboard:
        for event in scoreboard.get('events', []):
            game_id = event.get('id')
            if not game_id:
                continue

            home_team = None
            away_team = None

            # Site API structure: teams are in competitions[0].competitors[]
            competitions = event.get('competitions', [])
            if competitions and len(competitions) > 0:
                comp_list = competitions[0].get('competitors', [])
                if len(comp_list) == 2:
                    for comp in comp_list:
                        team_info = comp.get('team', {})
                        abbrev = team_info.get('abbreviation', '').upper()
                        if comp.get('homeAway') == 'home':
                            home_team = abbrev
                        else:
                            away_team = abbrev

            # Fallback: Parse from shortName (e.g., "NO @ CHA")
            if not home_team or not away_team:
                short_name = event.get('shortName', '')
                if '@' in short_name:
                    parts = short_name.split('@')
                    if len(parts) == 2:
                        away_team = parts[0].strip().upper()
                        home_team = parts[1].strip().upper()

            if home_team and away_team:
                # Use game_date from route parameter (user-requested date)
                # Don't use event.date from API as it's in UTC and can cause timezone shifts
                # The route parameter is the authoritative date the user wants to see
                event_date = game_date

                # Extract season, year, month, day
                season = get_season_from_date(event_date)
                year = event_date.year
                month = event_date.month
                day = event_date.day
                date_str = event_date.strftime('%Y-%m-%d')

                # Extract espn_link
                espn_link = event.get('link')
                if not espn_link:
                    # Fallback: construct from game_id
                    espn_link = f"https://www.espn.com/nba/boxscore/_/gameId/{game_id}"

                # Extract gametime from ESPN API (UTC format like "2026-02-01T19:00Z")
                event_gametime = event.get('date')

                # Extract odds data for pregame_lines (site API has different structure)
                pregame_lines_update = {}
                # Site API has odds in competitions[0].odds[] array
                if competitions:
                    odds_list = competitions[0].get('odds', [])
                    if odds_list and len(odds_list) > 0:
                        event_odds = odds_list[0]
                        if event_odds.get('overUnder') is not None:
                            pregame_lines_update['over_under'] = event_odds['overUnder']
                        if event_odds.get('spread') is not None:
                            pregame_lines_update['spread'] = event_odds['spread']
                        # Moneylines may be in homeTeamOdds/awayTeamOdds
                        home_odds = event_odds.get('homeTeamOdds', {})
                        away_odds = event_odds.get('awayTeamOdds', {})
                        if home_odds and home_odds.get('moneyLine') is not None:
                            pregame_lines_update['home_ml'] = home_odds['moneyLine']
                        if away_odds and away_odds.get('moneyLine') is not None:
                            pregame_lines_update['away_ml'] = away_odds['moneyLine']

                # Build upsert document - only include non-None values
                # Use dot notation for nested fields to preserve existing data
                update_doc = {'$set': {}}

                # Required fields
                update_doc['$set']['game_id'] = game_id
                update_doc['$set']['homeTeam.name'] = home_team
                update_doc['$set']['awayTeam.name'] = away_team

                # Date fields
                update_doc['$set']['date'] = date_str
                update_doc['$set']['year'] = year
                update_doc['$set']['month'] = month
                update_doc['$set']['day'] = day
                update_doc['$set']['season'] = season

                # ESPN link
                if espn_link:
                    update_doc['$set']['espn_link'] = espn_link

                # Pregame lines (only if we have odds data)
                if pregame_lines_update:
                    for key, value in pregame_lines_update.items():
                        update_doc['$set'][f'pregame_lines.{key}'] = value

                # Gametime - parse from event.date (UTC format like "2026-02-01T19:00Z")
                if event_gametime:
                    try:
                        from bball_app.core.utils.date_utils import parse_gametime
                        parsed_gametime = parse_gametime(event_gametime)
                        if parsed_gametime:
                            update_doc['$set']['gametime'] = parsed_gametime
                    except Exception as e:
                        print(f"Warning: Could not parse gametime '{event_gametime}': {e}")

                # Venue GUID - extract from competition venue (already have competitions from above)
                # The scoreboard site API returns a numeric ESPN venue id (e.g. "1830"),
                # not the UUID. Look up the UUID from the venues collection.
                if competitions:
                    venue_info = competitions[0].get('venue', {})
                    venue_id = venue_info.get('id')
                    if venue_id:
                        venues_collection = g.league.collections.get('venues', 'nba_venues')
                        venue_doc = db[venues_collection].find_one(
                            {'id': str(venue_id)},
                            {'venue_guid': 1}
                        )
                        if venue_doc and venue_doc.get('venue_guid'):
                            update_doc['$set']['venue_guid'] = venue_doc['venue_guid']

                # Get league-aware games collection
                games_collection = g.league.collections.get('games', 'stats_nba')

                # Upsert to games collection (preserves existing fields with $set)
                db[games_collection].update_one(
                    {'game_id': game_id},
                    update_doc,
                    upsert=True
                )

                # Get game document with pregame_lines, points, injured_players, and gametime
                game_doc = db[games_collection].find_one({'game_id': game_id}, {
                    'pregame_lines': 1,
                    'homeTeam.points': 1,
                    'awayTeam.points': 1,
                    'homeTeam.injured_players': 1,
                    'awayTeam.injured_players': 1,
                    'gametime': 1
                })

                # Get prediction from model_predictions collection
                predictions_collection = g.league.collections.get('model_predictions', 'nba_model_predictions')
                prediction_doc = db[predictions_collection].find_one({'game_id': game_id})
                last_prediction = None
                if prediction_doc:
                    last_prediction = {k: v for k, v in prediction_doc.items() if k != '_id'}

                pregame_lines = None
                home_points = None
                away_points = None
                home_injured_player_ids = []
                away_injured_player_ids = []
                gametime = event_gametime

                if game_doc:
                    if game_doc.get('pregame_lines'):
                        pregame_lines = game_doc['pregame_lines']
                        if hasattr(pregame_lines, 'to_dict'):
                            pregame_lines = pregame_lines.to_dict()
                        elif not isinstance(pregame_lines, dict):
                            import json
                            pregame_lines = json.loads(json.dumps(pregame_lines, default=str))

                    home_team_obj = game_doc.get('homeTeam', {})
                    away_team_obj = game_doc.get('awayTeam', {})
                    if home_team_obj and home_team_obj.get('points') is not None:
                        home_points = home_team_obj.get('points')
                    if away_team_obj and away_team_obj.get('points') is not None:
                        away_points = away_team_obj.get('points')

                    if home_team_obj:
                        home_injured_player_ids = home_team_obj.get('injured_players', [])
                    if away_team_obj:
                        away_injured_player_ids = away_team_obj.get('injured_players', [])

                    if not gametime and game_doc.get('gametime'):
                        gametime = game_doc['gametime']

                # Look up player names for injured players
                home_injured_names = []
                away_injured_names = []

                if home_injured_player_ids and len(home_injured_player_ids) > 0:
                    try:
                        home_player_ids_str = [str(pid) for pid in home_injured_player_ids]
                        home_players = list(db[g.league.collections.get("players", "nba_players")].find(
                            {'player_id': {'$in': home_player_ids_str}},
                            {'player_name': 1, 'player_id': 1}
                        ))
                        home_injured_names = [p.get('player_name', f"Player {p.get('player_id')}") for p in home_players]
                    except Exception as e:
                        print(f"Error looking up home injured players for game {game_id}: {e}")
                        home_injured_names = []

                if away_injured_player_ids and len(away_injured_player_ids) > 0:
                    try:
                        away_player_ids_str = [str(pid) for pid in away_injured_player_ids]
                        away_players = list(db[g.league.collections.get("players", "nba_players")].find(
                            {'player_id': {'$in': away_player_ids_str}},
                            {'player_name': 1, 'player_id': 1}
                        ))
                        away_injured_names = [p.get('player_name', f"Player {p.get('player_id')}") for p in away_players]
                    except Exception as e:
                        print(f"Error looking up away injured players for game {game_id}: {e}")
                        away_injured_names = []

                # Check API response for scores if database doesn't have them
                if home_points is None or away_points is None:
                    # Site API: scores in competitions[0].competitors[]
                    if competitions:
                        comp_list = competitions[0].get('competitors', [])
                        for comp in comp_list:
                            comp_score = comp.get('score', '')
                            if comp_score and comp_score != '':
                                try:
                                    score_int = int(comp_score)
                                    if comp.get('homeAway') == 'home':
                                        home_points = score_int
                                    else:
                                        away_points = score_int
                                except (ValueError, TypeError):
                                    pass

                # Extract game status from ESPN API
                event_status = event.get('status', {})
                game_status = 'pre'
                game_completed = False
                game_period = None
                game_clock = None

                if isinstance(event_status, str):
                    if event_status.lower() in ('final', 'completed', 'post'):
                        game_status = 'post'
                        game_completed = True
                    elif event_status.lower() in ('active', 'in', 'in progress', 'live'):
                        game_status = 'in'
                    else:
                        game_status = 'pre'
                elif isinstance(event_status, dict):
                    status_type = event_status.get('type', {})
                    if isinstance(status_type, dict):
                        game_status = status_type.get('name', 'pre')
                        game_completed = status_type.get('completed', False)
                    if game_status == 'in':
                        game_period = event_status.get('period', None)
                        game_clock = event_status.get('displayClock', None)

                games.append({
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': date_str,
                    'last_prediction': last_prediction,
                    'pregame_lines': pregame_lines,
                    'home_points': home_points,
                    'away_points': away_points,
                    'home_injured_players': home_injured_names,
                    'away_injured_players': away_injured_names,
                    'gametime': gametime,
                    'status': game_status,
                    'completed': game_completed,
                    'period': game_period,
                    'clock': game_clock
                })
    
    # Fallback to database lookup if API returns no games
    if not games:
        try:
            date_str = game_date.strftime('%Y-%m-%d')
            # Get the correct collection name for current league
            league_config = getattr(g, 'league', None)
            collection_name = league_config.collections.get('games', 'stats_nba') if league_config else 'stats_nba'

            # Query games directly from database
            db_games = list(db[collection_name].find(
                {'date': date_str},
                {
                    'game_id': 1,
                    'homeTeam': 1,
                    'awayTeam': 1,
                    'date': 1,
                    'last_prediction': 1,
                    'pregame_lines': 1,
                    'gametime': 1
                }
            ).sort('gametime', 1))

            for game_doc in db_games:
                game_id = game_doc.get('game_id')
                if not game_id:
                    continue

                home_team_obj = game_doc.get('homeTeam', {})
                away_team_obj = game_doc.get('awayTeam', {})

                home_team = home_team_obj.get('name', '').upper() if home_team_obj else ''
                away_team = away_team_obj.get('name', '').upper() if away_team_obj else ''

                if not home_team or not away_team:
                    continue

                # Get prediction from model_predictions collection
                predictions_collection = g.league.collections.get('model_predictions', 'nba_model_predictions')
                prediction_doc = db[predictions_collection].find_one({'game_id': game_id})
                last_prediction = None
                if prediction_doc:
                    last_prediction = {k: v for k, v in prediction_doc.items() if k != '_id'}

                pregame_lines = game_doc.get('pregame_lines')
                home_points = home_team_obj.get('points') if home_team_obj else None
                away_points = away_team_obj.get('points') if away_team_obj else None
                gametime = game_doc.get('gametime')

                # Ensure dicts are plain dicts
                if pregame_lines and hasattr(pregame_lines, 'to_dict'):
                    pregame_lines = pregame_lines.to_dict()

                # Get injured player names
                home_injured_player_ids = home_team_obj.get('injured_players', []) if home_team_obj else []
                away_injured_player_ids = away_team_obj.get('injured_players', []) if away_team_obj else []
                home_injured_names = []
                away_injured_names = []

                players_collection = 'nba_players'  # default
                if league_config:
                    players_collection = league_config.collections.get('players', 'nba_players')

                if home_injured_player_ids:
                    try:
                        home_player_ids_str = [str(pid) for pid in home_injured_player_ids]
                        home_players = list(db[players_collection].find(
                            {'player_id': {'$in': home_player_ids_str}},
                            {'player_name': 1, 'player_id': 1}
                        ))
                        home_injured_names = [p.get('player_name', f"Player {p.get('player_id')}") for p in home_players]
                    except Exception as e:
                        print(f"Error looking up home injured players: {e}")

                if away_injured_player_ids:
                    try:
                        away_player_ids_str = [str(pid) for pid in away_injured_player_ids]
                        away_players = list(db[players_collection].find(
                            {'player_id': {'$in': away_player_ids_str}},
                            {'player_name': 1, 'player_id': 1}
                        ))
                        away_injured_names = [p.get('player_name', f"Player {p.get('player_id')}") for p in away_players]
                    except Exception as e:
                        print(f"Error looking up away injured players: {e}")

                # For database fallback, infer status from points
                # If both teams have points, game is likely completed
                game_completed = home_points is not None and away_points is not None
                game_status = 'post' if game_completed else 'pre'

                games.append({
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': date_str,
                    'last_prediction': last_prediction,
                    'pregame_lines': pregame_lines,
                    'home_points': home_points,
                    'away_points': away_points,
                    'home_injured_players': home_injured_names,
                    'away_injured_players': away_injured_names,
                    'gametime': gametime,
                    'status': game_status,
                    'completed': game_completed,
                    'period': None,
                    'clock': None
                })
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in database fallback: {e}")
    
    # Add team logos and colors for each game
    def get_logo_url(team_data):
        """Extract logo URL from team data, handling both 'logo' field and 'logos' array."""
        if team_data.get('logo'):
            return team_data['logo']
        logos = team_data.get('logos', [])
        if isinstance(logos, list) and len(logos) > 0:
            first_logo = logos[0]
            if isinstance(first_logo, dict) and first_logo.get('href'):
                return first_logo['href']
        return None
    
    # Fetch team data for all games
    teams_collection = g.league.collections.get('teams', 'nba_teams')
    for game in games:
        home_team = game['home_team']
        away_team = game['away_team']

        # Get team data from league-specific teams collection
        home_team_data = db[teams_collection].find_one({'abbreviation': home_team}) or {}
        away_team_data = db[teams_collection].find_one({'abbreviation': away_team}) or {}
        
        # Extract logos
        game['home_team_logo'] = get_logo_url(home_team_data)
        game['away_team_logo'] = get_logo_url(away_team_data)
        
        # Extract colors
        game['home_team_color'] = home_team_data.get('color', '667eea')
        game['home_team_alternate_color'] = home_team_data.get('alternateColor', '764ba2')
        game['away_team_color'] = away_team_data.get('color', '666666')
        game['away_team_alternate_color'] = away_team_data.get('alternateColor', '999999')
    
    prev_date = (game_date - timedelta(days=1)).strftime('%Y-%m-%d')
    next_date = (game_date + timedelta(days=1)).strftime('%Y-%m-%d')

    # Sort games by gametime
    games.sort(key=lambda g: g.get('gametime', '') or '')

    # Calculate game pull stats: games with homeWon / total games for this date
    date_str = game_date.strftime('%Y-%m-%d')
    games_collection = g.league.collections.get('games', 'stats_nba')
    # Total games = max of scheduled games from API and games in DB (to handle both cases)
    api_games_count = len(games)
    db_games_count = db[games_collection].count_documents({'date': date_str})
    total_games = max(api_games_count, db_games_count) if (api_games_count > 0 or db_games_count > 0) else 0
    # Games with homeWon = games that have been pulled (have stats)
    games_with_homewon = db[games_collection].count_documents({
        'date': date_str,
        'homeWon': {'$exists': True}
    })

    # Get Kalshi team abbreviation mappings from league config for frontend
    league_config = getattr(g, 'league', None)
    kalshi_abbrev_map = {}
    kalshi_reverse_map = {}
    if league_config:
        kalshi_abbrev_map = league_config.raw.get('market', {}).get('team_abbrev_map', {})
        kalshi_reverse_map = {v: k for k, v in kalshi_abbrev_map.items()}

    # Get chat message counts per game for notification badges
    chat_message_counts = {}
    try:
        sessions_collection = g.league.collections.get('matchup_sessions', 'nba_matchup_sessions')
        game_ids = [g_item['game_id'] for g_item in games if g_item.get('game_id')]
        if game_ids:
            # Aggregate to count user messages per game_id
            pipeline = [
                {'$match': {'game_id': {'$in': game_ids}}},
                {'$project': {
                    'game_id': 1,
                    'user_message_count': {
                        '$size': {
                            '$filter': {
                                'input': {'$ifNull': ['$messages', []]},
                                'as': 'msg',
                                'cond': {'$eq': ['$$msg.role', 'user']}
                            }
                        }
                    }
                }}
            ]
            for doc in db[sessions_collection].aggregate(pipeline):
                if doc.get('user_message_count', 0) > 0:
                    chat_message_counts[doc['game_id']] = doc['user_message_count']
    except Exception as e:
        print(f"Error fetching chat message counts: {e}")

    return render_template('game_list.html',
                         games=games,
                         game_date=game_date,
                         prev_date=prev_date,
                         next_date=next_date,
                         games_pulled=games_with_homewon,
                         total_games=total_games,
                         kalshi_abbrev_map=kalshi_abbrev_map,
                         kalshi_reverse_map=kalshi_reverse_map,
                         chat_message_counts=chat_message_counts)


@app.route('/game/<game_id>')
@app.route('/<league_id>/game/<game_id>')
def game_detail(game_id, league_id=None):
    """Game detail page with player management."""
    # Check if date is provided in query parameters (from game list page)
    date_param = request.args.get('date')
    game_date = None
    
    # Priority 1: Use date from URL parameter (from game list page)
    if date_param:
        try:
            game_date = datetime.strptime(date_param, '%Y-%m-%d').date()
            print(f"DEBUG game_detail: Using date from URL parameter: {date_param} -> {game_date}")
        except Exception as e:
            print(f"DEBUG game_detail: Error parsing date parameter: {e}")
            date_param = None
    
    # Get game info from database or ESPN API
    games_collection = g.league.collections.get('games', 'stats_nba')
    game = db[games_collection].find_one({'game_id': game_id})

    # Check if game exists and has required structure
    if not game or not game.get('homeTeam') or not game.get('awayTeam'):
        # Try to get from ESPN API
        game_summary = get_game_summary(game_id)
        if not game_summary:
            return f"Game {game_id} not found", 404
        
        # Extract basic info
        header = game_summary.get('header', {})
        competitors = header.get('competitions', [{}])[0].get('competitors', [])
        
        if len(competitors) != 2:
            return "Invalid game data", 404
        
        home_team = None
        away_team = None
        for comp in competitors:
            if comp.get('homeAway') == 'home':
                home_team = comp.get('team', {}).get('abbreviation', '').upper()
            else:
                away_team = comp.get('team', {}).get('abbreviation', '').upper()
        
        # Priority 2: Get date from ESPN API (if not already set from URL)
        if game_date is None:
            # Get date from ESPN API - handle timezone properly
            # ESPN dates are in ISO format and may include timezone info
            # Extract just the date part (YYYY-MM-DD) to avoid timezone issues
            espn_date_str = header.get('competitions', [{}])[0].get('date', '')
            print(f"DEBUG game_detail: ESPN API date string: {espn_date_str}")
            if espn_date_str:
                try:
                    # Extract just the date portion (first 10 characters: YYYY-MM-DD)
                    # This avoids timezone conversion issues
                    date_only = espn_date_str[:10] if len(espn_date_str) >= 10 else espn_date_str
                    game_date = datetime.strptime(date_only, '%Y-%m-%d').date()
                    print(f"DEBUG game_detail: Extracted date from ESPN: {date_only} -> {game_date}")
                except Exception as e:
                    print(f"DEBUG game_detail: Error parsing ESPN date: {e}")
                    game_date = date.today()
            else:
                game_date = date.today()
    else:
        # Game exists in DB with proper structure
        home_team = game.get('homeTeam', {}).get('name', '')
        away_team = game.get('awayTeam', {}).get('name', '')
        
        # Fallback to API if team names are missing
        if not home_team or not away_team:
            game_summary = get_game_summary(game_id)
            if game_summary:
                header = game_summary.get('header', {})
                competitors = header.get('competitions', [{}])[0].get('competitors', [])
                for comp in competitors:
                    if comp.get('homeAway') == 'home':
                        home_team = comp.get('team', {}).get('abbreviation', '').upper()
                    else:
                        away_team = comp.get('team', {}).get('abbreviation', '').upper()
        
        # Priority 2: Get game date from database (if not already set from URL)
        if game_date is None:
            if game.get('date'):
                try:
                    game_date = datetime.strptime(game['date'], '%Y-%m-%d').date()
                    print(f"DEBUG game_detail: Using date from database: {game['date']} -> {game_date}")
                except:
                    game_date = date.today()
            else:
                game_date = date.today()
    
    print(f"DEBUG game_detail: Final game_date being used: {game_date}")
    
    # Get season
    season = get_season_from_date(game_date)
    
    # Get players for both teams from nba_rosters (no API call on page load)
    home_players = get_team_players_for_game(home_team, season, game_date)
    away_players = get_team_players_for_game(away_team, season, game_date)
    
    # Get player game status (playing/starter) from database
    game_status = get_player_game_status(game_id)
    
    # Merge player data with game status and injured status
    # Starter status comes from nba_rosters (already set in get_team_players_for_game)
    # Injury status comes from players_nba
    for player in home_players:
        player_id = player['player_id']
        status = game_status.get(f"{home_team}:{player_id}", {})
        player['is_playing'] = status.get('is_playing', True)
        # Starter status from nba_rosters (was_starter already set from roster)
        player['is_starter'] = player.get('was_starter', False)
        
        # Use nba_rosters as source of truth for injured status (already set in get_team_players_for_game)
        # The 'injured' field from get_team_players_for_game already comes from nba_rosters
        injury_status = player.get('injury_status')  # From players_nba, for display only
        player['injury_status'] = injury_status  # Pass to template for display
        is_injured_from_roster = player.get('injured', False)  # This comes from nba_rosters (source of truth)
        
        # Use nba_rosters as source of truth for injured flag
        player['is_injured'] = is_injured_from_roster
        
        # Set GTD status based on injury_status from API (for display only)
        if injury_status == 'GTD':
            player['is_gtd'] = True
        else:
            player['is_gtd'] = False
    
    for player in away_players:
        player_id = player['player_id']
        status = game_status.get(f"{away_team}:{player_id}", {})
        player['is_playing'] = status.get('is_playing', True)
        # Use game_status if available, otherwise fall back to nba_rosters (was_starter)
        player['is_starter'] = status.get('is_starter', player.get('was_starter', False))
        
        # Use nba_rosters as source of truth for injured status (already set in get_team_players_for_game)
        # The 'injured' field from get_team_players_for_game already comes from nba_rosters
        injury_status = player.get('injury_status')  # From players_nba, for display only
        player['injury_status'] = injury_status  # Pass to template for display
        is_injured_from_roster = player.get('injured', False)  # This comes from nba_rosters (source of truth)
        
        # Use nba_rosters as source of truth for injured flag
        player['is_injured'] = is_injured_from_roster
        
        # Set GTD status based on injury_status from API (for display only)
        if injury_status == 'GTD':
            player['is_gtd'] = True
        else:
            player['is_gtd'] = False
    
    # Sort players:
    # 1. Starters first
    # 2. Among starters: sort by position (guard, forward, center)
    # 3. Among non-starters: sort by MPG (descending)
    def sort_key(p):
        is_starter = p.get('is_starter', False)
        if is_starter:
            # Starters: sort by position only
            return (0, get_position_sort_order(p.get('pos_name', '')), 0)
        else:
            # Non-starters: sort by MPG descending
            return (1, 0, -p.get('stats', {}).get('mpg', 0.0))
    
    home_players.sort(key=sort_key)
    away_players.sort(key=sort_key)
    
    # Get team data (logo, colors) from league-specific teams collection
    teams_collection = g.league.collections.get('teams', 'nba_teams')
    home_team_data = db[teams_collection].find_one({'abbreviation': home_team}) or {}
    away_team_data = db[teams_collection].find_one({'abbreviation': away_team}) or {}
    
    # Extract logo URL from logos array if logo field doesn't exist
    def get_logo_url(team_data):
        """Extract logo URL from team data, handling both 'logo' field and 'logos' array."""
        # First try the 'logo' field (already extracted)
        if team_data.get('logo'):
            return team_data['logo']
        
        # Otherwise, try to extract from 'logos' array
        logos = team_data.get('logos', [])
        if isinstance(logos, list) and len(logos) > 0:
            first_logo = logos[0]
            if isinstance(first_logo, dict) and first_logo.get('href'):
                return first_logo['href']
        
        return None
    
    home_team_logo = get_logo_url(home_team_data)
    away_team_logo = get_logo_url(away_team_data)
    
    # Calculate team records
    home_records = calculate_team_records(home_team, season, game_date, league=g.league)
    away_records = calculate_team_records(away_team, season, game_date, league=g.league)
    
    # Store player IDs in games collection document for this game
    # This helps predict() function find players for PER calculations
    if game_id:
        try:
            # Extract player IDs from home and away players
            home_player_ids = [str(p.get('player_id')) for p in home_players if p.get('player_id')]
            away_player_ids = [str(p.get('player_id')) for p in away_players if p.get('player_id')]

            # Update games collection document with player lists
            update_result = db[games_collection].update_one(
                {'game_id': game_id},
                {
                    '$set': {
                        'homeTeam.players': home_player_ids,
                        'awayTeam.players': away_player_ids
                    }
                }
            )

            if update_result.matched_count > 0:
                print(f"DEBUG game_detail: Updated {games_collection} with {len(home_player_ids)} home players and {len(away_player_ids)} away players")
            else:
                print(f"DEBUG game_detail: Game {game_id} not found in {games_collection}, could not update player lists")
        except Exception as e:
            print(f"DEBUG game_detail: Error updating player lists in {games_collection}: {e}")
            import traceback
            traceback.print_exc()

    # Get last prediction and pregame_lines from database
    last_prediction = None
    pregame_lines = None
    print(f"DEBUG game_detail: Looking for last_prediction and pregame_lines for game_id={game_id}")
    game_doc = db[games_collection].find_one({'game_id': game_id}, {'last_prediction': 1, 'pregame_lines': 1})
    if game_doc:
        last_prediction = game_doc.get('last_prediction')
        print(f"DEBUG game_detail: last_prediction found: {last_prediction is not None}")
        if last_prediction:
            # Convert to dict if it's a MongoDB document
            if hasattr(last_prediction, 'to_dict'):
                last_prediction = last_prediction.to_dict()
            # Ensure it's a plain dict (not a SON object or similar)
            elif not isinstance(last_prediction, dict):
                import json
                last_prediction = json.loads(json.dumps(last_prediction, default=str))
        
        pregame_lines = game_doc.get('pregame_lines')
        if pregame_lines:
            # Convert to dict if it's a MongoDB document
            if hasattr(pregame_lines, 'to_dict'):
                pregame_lines = pregame_lines.to_dict()
            elif not isinstance(pregame_lines, dict):
                import json
                pregame_lines = json.loads(json.dumps(pregame_lines, default=str))
    else:
        print(f"DEBUG game_detail: No game document found in database for game_id={game_id}")
    
    print(f"DEBUG game_detail: Passing last_prediction to template: {last_prediction is not None}, type: {type(last_prediction)}")
    print(f"DEBUG game_detail: Passing pregame_lines to template: {pregame_lines is not None}, type: {type(pregame_lines)}")
    
    return render_template(
        'game_detail.html',
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        game_date=game_date.strftime('%Y-%m-%d'),
        season=season,
        get_position_abbreviation=get_position_abbreviation,
        home_players=home_players,
        away_players=away_players,
        home_team_logo=home_team_logo,
        home_team_color=home_team_data.get('color', '667eea'),
        home_team_alternate_color=home_team_data.get('alternateColor', '764ba2'),
        away_team_logo=away_team_logo,
        away_team_color=away_team_data.get('color', '666666'),
        away_team_alternate_color=away_team_data.get('alternateColor', '999999'),
        home_records=home_records,
        away_records=away_records,
        last_prediction=last_prediction,
        pregame_lines=pregame_lines
    )


def calculate_team_records(team: str, season: str, game_date: date, league=None) -> Dict:
    """Calculate team W-L records (overall, home, away, last 10)."""
    before_date = game_date.strftime('%Y-%m-%d')

    # Get league-aware collection and exclude game types
    games_collection = 'stats_nba'
    exclude_game_types = ['preseason', 'allstar']
    if league:
        games_collection = league.collections.get('games', 'stats_nba')
        exclude_game_types = getattr(league, 'exclude_game_types', exclude_game_types)

    # Get all games for this team before the game date
    games = list(db[games_collection].find(
        {
            'season': season,
            'date': {'$lt': before_date},
            '$or': [
                {'homeTeam.name': team},
                {'awayTeam.name': team}
            ],
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': exclude_game_types}
        },
        {
            'homeTeam.name': 1,
            'awayTeam.name': 1,
            'homeWon': 1,
            'date': 1
        }
    ))
    
    if not games:
        return {
            'overall': {'wins': 0, 'losses': 0, 'record': '0-0'},
            'home': {'wins': 0, 'losses': 0, 'record': '0-0'},
            'away': {'wins': 0, 'losses': 0, 'record': '0-0'},
            'last10': {'wins': 0, 'losses': 0, 'record': '0-0'}
        }
    
    # Sort by date descending to get last 10
    games.sort(key=lambda g: g.get('date', ''), reverse=True)
    last_10 = games[:10]
    
    # Calculate records
    overall_wins = 0
    overall_losses = 0
    home_wins = 0
    home_losses = 0
    away_wins = 0
    away_losses = 0
    last10_wins = 0
    last10_losses = 0
    
    for game in games:
        is_home = game.get('homeTeam', {}).get('name') == team
        won = (is_home and game.get('homeWon', False)) or (not is_home and not game.get('homeWon', False))
        
        if won:
            overall_wins += 1
        else:
            overall_losses += 1
        
        if is_home:
            if won:
                home_wins += 1
            else:
                home_losses += 1
        else:
            if won:
                away_wins += 1
            else:
                away_losses += 1
    
    # Last 10 games
    for game in last_10:
        is_home = game.get('homeTeam', {}).get('name') == team
        won = (is_home and game.get('homeWon', False)) or (not is_home and not game.get('homeWon', False))
        if won:
            last10_wins += 1
        else:
            last10_losses += 1
    
    return {
        'overall': {'wins': overall_wins, 'losses': overall_losses, 'record': f'{overall_wins}-{overall_losses}'},
        'home': {'wins': home_wins, 'losses': home_losses, 'record': f'{home_wins}-{home_losses}'},
        'away': {'wins': away_wins, 'losses': away_losses, 'record': f'{away_wins}-{away_losses}'},
        'last10': {'wins': last10_wins, 'losses': last10_losses, 'record': f'{last10_wins}-{last10_losses}'}
    }


def get_position_sort_order(pos_name: str) -> int:
    """
    Get sort order for position.
    Returns: 0 for Guard, 1 for Forward, 2 for Center, 3 for unknown
    """
    if not pos_name:
        return 3
    
    pos_lower = pos_name.lower()
    if 'guard' in pos_lower:
        return 0
    elif 'forward' in pos_lower:
        return 1
    elif 'center' in pos_lower:
        return 2
    else:
        return 3


def get_position_abbreviation(pos_display_name: str) -> str:
    """Get position abbreviation: G, F, or C."""
    if not pos_display_name:
        return ''
    pos_lower = pos_display_name.lower()
    if 'guard' in pos_lower:
        return 'G'
    elif 'forward' in pos_lower:
        return 'F'
    elif 'center' in pos_lower:
        return 'C'
    return ''


def calculate_player_stats(player_id: str, team: str, season: str, before_date: str) -> Dict:
    """Calculate per-game stats for a player from stats_nba_players collection."""
    # Get all games for this player before the game date
    player_games = list(db[g.league.collections.get("player_stats", "nba_player_stats")].find(
        {
            'player_id': player_id,
            'team': team,
            'season': season,
            'date': {'$lt': before_date},
            'stats.min': {'$gt': 0}  # Only games where player played
        },
        {
            'stats.pts': 1,
            'stats.ast': 1,
            'stats.reb': 1,
            'stats.min': 1
        }
    ))
    
    if not player_games:
        return {
            'gp': 0,
            'ppg': 0.0,
            'apg': 0.0,
            'rpg': 0.0,
            'mpg': 0.0,
            'per': 0.0
        }
    
    # Calculate totals
    total_pts = sum(game.get('stats', {}).get('pts', 0) for game in player_games)
    total_ast = sum(game.get('stats', {}).get('ast', 0) for game in player_games)
    total_reb = sum(game.get('stats', {}).get('reb', 0) for game in player_games)
    total_min = sum(game.get('stats', {}).get('min', 0) for game in player_games)
    gp = len(player_games)
    
    # Calculate per-game averages
    ppg = round(total_pts / gp, 1) if gp > 0 else 0.0
    apg = round(total_ast / gp, 1) if gp > 0 else 0.0
    rpg = round(total_reb / gp, 1) if gp > 0 else 0.0
    mpg = round(total_min / gp, 1) if gp > 0 else 0.0
    
    # Calculate PER (simplified approximation)
    # Full PER requires league constants, team stats, etc.
    # Using simplified formula: (PTS + REB + AST) / MPG * scaling factor
    if mpg > 0:
        # Simplified PER approximation scaled to approximate PER range
        per = round((ppg + rpg + apg) / mpg * 15, 1)
    else:
        per = 0.0
    
    return {
        'gp': gp,
        'ppg': ppg,
        'apg': apg,
        'rpg': rpg,
        'mpg': mpg,
        'per': per,
        'min': total_min  # Add total minutes for filtering
    }


def get_team_players_for_game(team: str, season: str, game_date: date) -> List[Dict]:
    """
    Get players for a team from nba_rosters collection with stats.
    Uses nba_rosters as the source of truth for roster and starter status.
    """
    before_date = game_date.strftime('%Y-%m-%d')
    
    # Get roster from nba_rosters collection
    # Each document in nba_rosters is uniquely identified by season + team
    roster_doc = db.nba_rosters.find_one(
        {'season': season, 'team': team}  # Unique key: season + team
    )
    
    if not roster_doc:
        # Fallback: if no roster exists, return empty list
        # (roster should be created by update_rosters.py script)
        return []
    
    roster = roster_doc.get('roster', [])
    if not roster:
        return []
    
    # Get player IDs from roster
    player_ids = [str(p['player_id']) for p in roster]
    
    # Get player details from players_nba collection
    players = list(db[g.league.collections.get("players", "nba_players")].find(
        {'player_id': {'$in': player_ids}},
        {
            'player_id': 1,
            'player_name': 1,
            'headshot': 1,
            'pos_name': 1,
            'pos_display_name': 1,
            'injured': 1,
            'injury_status': 1,
            'injury_date': 1,
            'injury_details': 1
        }
    ))
    
    # Create a map of player_id -> roster entry (for starter status)
    roster_map = {str(p['player_id']): p for p in roster}
    
    # Filter out players without names and calculate stats
    valid_players = []
    for player in players:
        player_name = player.get('player_name')
        if not player_name or not player_name.strip():
            continue
        
        player_id = str(player.get('player_id', ''))
        
        # Get starter and injured status from nba_rosters (source of truth)
        roster_entry = roster_map.get(player_id, {})
        is_starter = roster_entry.get('starter', False)
        is_injured = roster_entry.get('injured', False)  # Use nba_rosters as source of truth
        
        # Calculate player stats
        stats = calculate_player_stats(player_id, team, season, before_date)
        
        # Filter out players with less than 5 MPG or 0 minutes
        mpg = stats.get('mpg', 0.0)
        total_minutes = stats.get('min', 0)
        if mpg < 5.0 or total_minutes == 0:
            continue  # Skip players with < 5 MPG or 0 minutes
        
        # Ensure all required fields exist
        player_dict = {
            'player_id': player_id,
            'player_name': player_name,
            'headshot': player.get('headshot'),
            'pos_name': player.get('pos_name'),
            'pos_display_name': player.get('pos_display_name'),
            'injured': is_injured,  # Use nba_rosters as source of truth, not players_nba
            'injury_status': player.get('injury_status'),  # Keep for display purposes, but injured flag is from nba_rosters
            'was_starter': is_starter,  # Get from nba_rosters (source of truth)
            'stats': stats  # Add stats summary
        }
        
        valid_players.append(player_dict)
    
    # Sort: first by starter status (starters first), then by position (guard, forward, center)
    valid_players.sort(key=lambda p: (
        not p.get('was_starter', False),  # False (starters) come before True (non-starters)
        get_position_sort_order(p.get('pos_name', '')),  # Position order: guard=0, forward=1, center=2
        p.get('player_name', '')  # Tertiary sort by name for consistency
    ))
    
    return valid_players


def get_player_game_status(game_id: str) -> Dict:
    """Get player game status from database."""
    status_docs = list(db.player_game_status.find({'game_id': game_id}))
    status_dict = {}
    for doc in status_docs:
        key = f"{doc['team']}:{doc['player_id']}"
        status_dict[key] = {
            'is_playing': doc.get('is_playing', True),
            'is_starter': doc.get('is_starter', False)
        }
    return status_dict


@app.route('/<league_id>/api/move-player', methods=['POST'])
@app.route('/api/move-player', methods=['POST'])
def move_player(league_id=None):
    """Move a player from one team to another. Thin wrapper around core game_service."""
    from bball_app.core.services.game_service import move_player_to_team

    data = request.json
    player_id = data.get('player_id')
    to_team = data.get('to_team')
    season = data.get('season')

    if not all([player_id, to_team, season]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        result = move_player_to_team(
            db=db,
            player_id=str(player_id),
            to_team=to_team,
            season=season,
            league=g.league
        )
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/teams-list', methods=['GET'])
@app.route('/api/teams-list', methods=['GET'])
def api_teams_list(league_id=None):
    """Return all teams for the league for autocomplete."""
    teams_collection = g.league.collections.get('teams', 'nba_teams')
    teams = list(db[teams_collection].find(
        {},
        {'_id': 0, 'abbreviation': 1, 'displayName': 1, 'name': 1, 'id': 1, 'team_id': 1, 'logo': 1, 'logos': 1}
    ))

    result = []
    for t in teams:
        logo = t.get('logo')
        if not logo:
            logos = t.get('logos', [])
            if isinstance(logos, list) and logos:
                first = logos[0]
                if isinstance(first, dict):
                    logo = first.get('href')

        result.append({
            'team_id': t.get('id') or t.get('team_id') or t.get('abbreviation', ''),
            'abbreviation': t.get('abbreviation', ''),
            'display_name': t.get('displayName') or t.get('name', ''),
            'logo': logo
        })

    return jsonify(result)


@app.route('/<league_id>/api/player-search', methods=['GET'])
@app.route('/api/player-search', methods=['GET'])
def player_search(league_id=None):
    """Search for players by name. Thin wrapper around core game_service."""
    from bball_app.core.services.game_service import search_players_for_roster

    q = request.args.get('q', '').strip()
    season = request.args.get('season', '')

    if len(q) < 2 or not season:
        return jsonify([])

    results = search_players_for_roster(db, q, season, g.league)
    return jsonify(results)


@app.route('/<league_id>/api/add-player-to-roster', methods=['POST'])
@app.route('/api/add-player-to-roster', methods=['POST'])
def add_player_to_roster(league_id=None):
    """Add a player to a team's roster. Thin wrapper around core game_service."""
    from bball_app.core.services.game_service import add_player_to_team_roster

    data = request.json
    player_id = data.get('player_id')
    team = data.get('team')
    season = data.get('season')
    game_date_str = data.get('game_date')

    if not all([player_id, team, season]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        result = add_player_to_team_roster(db, player_id, team, season, game_date_str, g.league)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/drop-player-from-roster', methods=['POST'])
@app.route('/api/drop-player-from-roster', methods=['POST'])
def drop_player_from_roster_endpoint(league_id=None):
    """Drop a player from a team's roster entirely."""
    from bball_app.core.services.lineup_service import drop_player_from_roster

    data = request.json
    player_id = data.get('player_id')
    team = data.get('team')
    season = data.get('season')

    if not all([player_id, team, season]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        result = drop_player_from_roster(db, player_id, team, season, g.league)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/auto-set-lineups', methods=['POST'])
@app.route('/api/auto-set-lineups', methods=['POST'])
def auto_set_lineups_endpoint(league_id=None):
    """Auto-set lineups for all teams playing on a date based on previous game."""
    from bball_app.core.services.lineup_service import auto_set_lineups

    data = request.json or {}
    game_date = data.get('game_date') or datetime.now().strftime('%Y-%m-%d')

    try:
        result = auto_set_lineups(db, g.league, game_date)
        return jsonify({'success': True, **result})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/remove-player', methods=['POST'])
@app.route('/api/remove-player', methods=['POST'])
def remove_player(league_id=None):
    """Remove a player from the game's player list. Does NOT update rosters."""
    data = request.json
    game_id = data.get('game_id')
    player_id = data.get('player_id')
    team = data.get('team')

    if not all([game_id, player_id, team]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    try:
        # Get league-aware games collection
        games_collection = g.league.collections.get('games', 'stats_nba')

        # Get the game document
        game = db[games_collection].find_one({'game_id': game_id})
        if not game:
            return jsonify({'success': False, 'error': f'Game {game_id} not found'}), 404

        # Determine which team field to update
        team_key = 'homeTeam' if team == game.get('homeTeam', {}).get('name') else 'awayTeam'

        # Get current player list
        current_players = game.get(team_key, {}).get('players', [])

        # Convert player_id to the same type as stored (could be string or int)
        player_id_to_remove = str(player_id)
        if current_players and isinstance(current_players[0], int):
            try:
                player_id_to_remove = int(player_id)
            except (ValueError, TypeError):
                pass

        # Remove player from list
        updated_players = [p for p in current_players if str(p) != str(player_id_to_remove)]

        # Update the game document
        result = db[games_collection].update_one(
            {'game_id': game_id},
            {
                '$set': {
                    f'{team_key}.players': updated_players,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        if result.modified_count > 0:
            print(f"DEBUG remove_player: Removed player {player_id} from {team_key} in game {game_id}")
            return jsonify({
                'success': True,
                'message': f'Player {player_id} removed from {team}',
                'remaining_players': len(updated_players)
            })
        else:
            return jsonify({'success': False, 'error': 'Player not found in team list or no changes made'}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/load-players', methods=['POST'])
@app.route('/api/load-players', methods=['POST'])
def load_players(league_id=None):
    """Load players from ESPN API and update players and injuries."""
    data = request.json
    game_id = data.get('game_id')

    if not game_id:
        return jsonify({'success': False, 'error': 'Missing game_id'}), 400

    try:
        # Get league-aware games collection
        games_collection = g.league.collections.get('games', 'stats_nba')

        # Get game from database
        game = db[games_collection].find_one({'game_id': game_id})
        if not game:
            return jsonify({'success': False, 'error': f'Game {game_id} not found'}), 404
        
        home_team = game.get('homeTeam', {}).get('name', '')
        away_team = game.get('awayTeam', {}).get('name', '')
        
        if not home_team or not away_team:
            return jsonify({'success': False, 'error': 'Game missing team information'}), 400
        
        # Get game summary from ESPN API (league-aware)
        espn_client = ESPNClient(league=g.league)
        game_summary = espn_client.get_game_summary(game_id)

        if not game_summary:
            return jsonify({'success': False, 'error': 'Could not fetch game summary from ESPN API'}), 404
        
        # Update teams (use league-specific collection)
        teams_collection = g.league.collections.get('teams', 'nba_teams')
        extract_and_update_teams(game_summary, teams_collection)
        # Update player roster (headshots, names, positions) - updates players_nba only
        extract_and_update_player_roster(game_summary, home_team, away_team)
        # Update injuries - updates players_nba only
        extract_and_update_injuries(game_summary, home_team, away_team)
        
        return jsonify({'success': True, 'message': 'Players loaded and updated successfully'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/update-player', methods=['POST'])
@app.route('/api/update-player', methods=['POST'])
def update_player(league_id=None):
    """Update player status (playing/starter/injured) for a game. Updates rosters collection for starter/injured status."""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No JSON data received'}), 400

    game_id = data.get('game_id')
    player_id = data.get('player_id')
    team = data.get('team')
    is_playing = data.get('is_playing', True)
    is_starter = data.get('is_starter')  # None if not sent
    is_injured = data.get('is_injured')  # None if not sent
    is_disabled = data.get('is_disabled')  # None if not sent

    # Debug logging
    print(f"DEBUG update_player: game_id={game_id}, player_id={player_id}, team={team}, is_starter={is_starter}, is_injured={is_injured}")

    missing = []
    if not game_id:
        missing.append('game_id')
    if not player_id:
        missing.append('player_id')
    if not team:
        missing.append('team')
    if missing:
        print(f"DEBUG update_player: RETURNING 400 - missing fields: {missing}")
        return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing)}'}), 400

    print("DEBUG update_player: Passed validation, getting league config...")
    # Get league-aware collection names
    league = g.league
    games_collection = league.collections.get('games', 'stats_nba')
    players_collection = league.collections.get('players', 'nba_players')
    rosters_collection = league.collections.get('rosters', 'nba_rosters')

    print(f"DEBUG update_player: Looking for game in {games_collection}...")
    # Get season from game - handle both string and int game_id
    game_id_query = [game_id]
    if str(game_id).isdigit():
        game_id_query.append(int(game_id))
        game_id_query.append(str(game_id))
    game = db[games_collection].find_one({'game_id': {'$in': game_id_query}})
    if not game:
        print(f"DEBUG update_player: RETURNING 404 - game not found: {game_id}")
        return jsonify({'success': False, 'error': f'Game not found: {game_id}'}), 404

    print(f"DEBUG update_player: Found game, getting date...")
    game_date_str = game.get('date')
    if game_date_str:
        try:
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
            season = get_season_from_date_core(game_date, league=league)
            print(f"DEBUG update_player: season={season}")
        except Exception as e:
            print(f"DEBUG update_player: RETURNING 400 - invalid date: {e}")
            return jsonify({'success': False, 'error': 'Invalid game date'}), 400
    else:
        print("DEBUG update_player: RETURNING 400 - game missing date")
        return jsonify({'success': False, 'error': 'Game missing date'}), 400

    # Update players collection with injured status
    # Handle both string and int player_ids
    player_id_query = [player_id]
    if str(player_id).isdigit():
        player_id_query.append(int(player_id))
        player_id_query.append(str(player_id))

    if is_injured is not None:
        db[players_collection].update_one(
            {'player_id': {'$in': player_id_query}},
            {
                '$set': {
                    'injured': is_injured,
                    'last_status_update': datetime.utcnow()
                }
            },
            upsert=False
        )

    # Update rosters collection with starter status and injured status
    # Find the roster entry for this player and update it
    roster_doc = db[rosters_collection].find_one(
        {'season': season, 'team': team}
    )

    if not roster_doc:
        print(f"DEBUG update_player: No roster found for season={season}, team={team}")
        # If roster doesn't exist, we can't update it - this is expected if rosters haven't been generated yet
    else:
        roster = roster_doc.get('roster', [])
        # Find and update the player in the roster
        updated = False
        for roster_entry in roster:
            if str(roster_entry.get('player_id')) == str(player_id):
                if is_starter is not None:
                    roster_entry['starter'] = is_starter
                if is_injured is not None:
                    roster_entry['injured'] = is_injured
                if is_disabled is not None:
                    roster_entry['disabled'] = is_disabled
                updated = True
                print(f"DEBUG update_player: Updated player {player_id} in roster - starter={is_starter}, injured={is_injured}, disabled={is_disabled}")
                break

        if updated:
            # Update the roster document
            result = db[rosters_collection].update_one(
                {'season': season, 'team': team},
                {
                    '$set': {
                        'roster': roster,
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            print(f"DEBUG update_player: {rosters_collection} update result - matched={result.matched_count}, modified={result.modified_count}")
        else:
            print(f"DEBUG update_player: Player {player_id} not found in roster for team {team}, season {season}")
            print(f"DEBUG update_player: Roster has {len(roster)} players")

    return jsonify({'success': True})


@app.route('/<league_id>/api/game-detail/<game_id>', methods=['GET'])
@app.route('/api/game-detail/<game_id>', methods=['GET'])
@app.route('/<league_id>/api/game-detail/<game_id>', methods=['GET'])
def api_game_detail(game_id, league_id=None):
    """Get game detail data for modal display. Thin wrapper around core game_service."""
    from bball_app.core.services.game_service import get_game_detail

    date_param = request.args.get('date')
    game_date = None
    if date_param:
        try:
            game_date = datetime.strptime(date_param, '%Y-%m-%d').date()
        except:
            pass

    result = get_game_detail(db, game_id, game_date, g.league)

    if not result.get('success'):
        status_code = 404 if 'not found' in result.get('error', '').lower() else 400
        return jsonify(result), status_code

    return jsonify(result)


@app.route('/<league_id>/api/player-detail', methods=['GET'])
@app.route('/api/player-detail', methods=['GET'])
def api_player_detail(league_id=None):
    """Get player detail stats for the player detail panel."""
    from bball_app.core.services.game_service import get_player_detail

    player_id = request.args.get('player_id')
    team = request.args.get('team')
    game_date_str = request.args.get('date')
    season = request.args.get('season')

    if not all([player_id, team]):
        return jsonify({'success': False, 'error': 'Missing player_id or team'}), 400

    # Parse game date
    game_date = None
    if game_date_str:
        try:
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
        except:
            pass
    if not game_date:
        game_date = date.today()

    # Get season if not provided
    if not season:
        season = get_season_from_date(game_date, league=g.league)

    result = get_player_detail(db, player_id, team, season, game_date, g.league)
    return jsonify(result)


@app.route('/<league_id>/api/player-per', methods=['GET'])
@app.route('/api/player-per', methods=['GET'])
def api_player_per(league_id=None):
    """Get player PER (separate endpoint due to calculation time)."""
    from bball_app.core.services.game_service import get_player_per

    player_id = request.args.get('player_id')
    team = request.args.get('team')
    game_date_str = request.args.get('date')
    season = request.args.get('season')

    if not all([player_id, team]):
        return jsonify({'success': False, 'error': 'Missing player_id or team'}), 400

    # Parse game date
    game_date = None
    if game_date_str:
        try:
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
        except:
            pass
    if not game_date:
        game_date = date.today()

    # Get season if not provided
    if not season:
        season = get_season_from_date(game_date, league=g.league)

    result = get_player_per(db, player_id, team, season, game_date, g.league)
    return jsonify(result)


@app.route('/<league_id>/api/players-per-batch', methods=['POST'])
@app.route('/api/players-per-batch', methods=['POST'])
def api_players_per_batch(league_id=None):
    """Get PER for multiple players at once (more efficient than individual calls)."""
    from bball_app.core.services.game_service import get_players_per_batch

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    players = data.get('players', [])  # List of {player_id, team}
    game_date_str = data.get('date')
    season = data.get('season')

    if not players:
        return jsonify({'success': True, 'per_values': {}})

    # Parse game date
    game_date = None
    if game_date_str:
        try:
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
        except:
            pass
    if not game_date:
        game_date = date.today()

    # Get season if not provided
    if not season:
        season = get_season_from_date(game_date, league=g.league)

    per_values = get_players_per_batch(db, players, season, game_date, g.league)

    return jsonify({
        'success': True,
        'per_values': per_values
    })


@app.route('/<league_id>/api/player-news', methods=['GET'])
@app.route('/api/player-news', methods=['GET'])
def api_player_news(league_id=None):
    """
    Fetch player news/status from Rotowire using WebpageParser.

    Only available for leagues that have a rotowire player_slugs_file configured.
    Returns extracted text from the player's Rotowire page.
    """
    import json
    from bball_app.core.services.webpage_parser import WebpageParser

    player_id = request.args.get('player_id')
    if not player_id:
        return jsonify({'success': False, 'error': 'Missing player_id'}), 400

    # Check if league has rotowire config
    league_config = g.league
    news_sources = league_config.raw.get('news_sources', {})
    rotowire_config = news_sources.get('rotowire', {})
    mappings = rotowire_config.get('mappings', {})
    player_slugs_file = mappings.get('player_slugs_file')

    if not player_slugs_file:
        return jsonify({
            'success': False,
            'error': 'Rotowire player mapping not available for this league'
        }), 404

    # Get the player URL pattern
    patterns = rotowire_config.get('patterns', {})
    player_pattern = patterns.get('player')
    base_url = rotowire_config.get('base_url', 'https://www.rotowire.com')

    if not player_pattern:
        return jsonify({
            'success': False,
            'error': 'Rotowire player URL pattern not configured'
        }), 404

    # Load the player slugs mapping file
    # Path is relative to nba_app/ directory (where leagues/ folder lives)
    import os
    nba_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    slugs_path = os.path.join(nba_app_dir, player_slugs_file)

    if not os.path.exists(slugs_path):
        return jsonify({
            'success': False,
            'error': 'Rotowire player mapping file not found'
        }), 404

    try:
        with open(slugs_path, 'r') as f:
            player_slugs = json.load(f)
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to load player mapping: {str(e)}'
        }), 500

    # Look up the player slug
    player_data = player_slugs.get(str(player_id))
    if not player_data or not player_data.get('rotowire_slug'):
        return jsonify({
            'success': False,
            'error': 'Player not found in Rotowire mapping'
        }), 404

    rotowire_slug = player_data['rotowire_slug']

    # Build the URL
    player_url = base_url + player_pattern.replace('{player_slug}', rotowire_slug)

    try:
        # Fetch and parse the page
        text = WebpageParser.extract_from_url(player_url, timeout=15)

        # Extract only the relevant news section between markers
        start_marker = "Read Past Outlooks"
        end_marker = "NBA Per Game Stats"

        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)

        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            # Both markers exist - extract between them
            text = text[start_idx + len(start_marker):end_idx].strip()
        elif start_idx != -1:
            # Only start marker - extract after it
            text = text[start_idx + len(start_marker):].strip()
        elif end_idx != -1:
            # Only end marker - extract before it
            text = text[:end_idx].strip()
        # else: neither marker - use full text

        # Limit the text to a reasonable length for the UI
        max_length = 2000
        if len(text) > max_length:
            text = text[:max_length] + '...'

        return jsonify({
            'success': True,
            'player_info': text,
            'source_url': player_url,
            'player_name': player_data.get('espn_name', '')
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to fetch player info: {str(e)}'
        }), 500


@app.route('/<league_id>/api/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict(league_id=None):
    """
    Generate prediction for a game.
    Thin wrapper around core PredictionService.
    Rosters are the single source of truth for player lists.
    """
    from bball_app.core.services.prediction import PredictionService

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data received in request'}), 400

    game_id = data.get('game_id')
    game_date_str = data.get('game_date')
    home_team = data.get('home_team')
    away_team = data.get('away_team')

    # Validate required fields
    missing_fields = []
    if not game_id:
        missing_fields.append('game_id')
    if not game_date_str:
        missing_fields.append('game_date')
    if not home_team:
        missing_fields.append('home_team')
    if not away_team:
        missing_fields.append('away_team')

    if missing_fields:
        return jsonify({'success': False, 'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400

    try:
        game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

    try:
        # Use core PredictionService
        league = g.league
        logger.info(f"[predict] league={league.league_id}, collection={league.collections.get('model_config_classifier')}, game_id={game_id}")
        service = PredictionService(db=db, league=league)

        # Make prediction
        result = service.predict_game(
            home_team=home_team,
            away_team=away_team,
            game_date=game_date_str,
            game_id=game_id,
            include_points=True
        )

        # Check for errors
        if result.error:
            logger.warning(f"[predict] Prediction error for {game_id}: {result.error}")
            return jsonify({'success': False, 'error': result.error}), 400

        # Save prediction to database
        service.save_prediction(
            result=result,
            game_id=game_id,
            game_date=game_date,
            home_team=home_team,
            away_team=away_team
        )

        # Build response
        prediction_response = {
            'predicted_winner': result.predicted_winner,
            'home_win_prob': result.home_win_prob,
            'away_win_prob': result.away_win_prob,
            'home_odds': result.home_odds,
            'away_odds': result.away_odds,
            'home_points_pred': result.home_points_pred,
            'away_points_pred': result.away_points_pred,
            'point_diff_pred': result.point_diff_pred
        }

        return jsonify({
            'success': True,
            'prediction': prediction_response
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/game-features', methods=['POST'])
@app.route('/api/game-features', methods=['POST'])
def get_game_features(league_id=None):
    """Get feature values used for a game prediction."""
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data received'}), 400
    game_id = data.get('game_id')
    game_date_str = data.get('game_date')
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    if not all([game_id, game_date_str, home_team, away_team]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    try:
        # Get league-aware collection names
        league = g.league
        games_collection = league.collections.get('games', 'stats_nba')
        predictions_collection = league.collections.get('model_predictions', 'nba_model_predictions')
        classifier_config_collection = league.collections.get('model_config_classifier', 'nba_model_config')

        # Get game document
        game_doc = db[games_collection].find_one({'game_id': game_id})
        if not game_doc:
            return jsonify({'success': False, 'error': f'Game not found: {game_id}'}), 404

        # Get prediction from model_predictions collection (not from game document)
        last_prediction = db[predictions_collection].find_one({'game_id': game_id})
        if not last_prediction:
            return jsonify({'success': False, 'error': 'No prediction found for this game. Please run predictions first.'}), 404
        features_dict = last_prediction.get('features_dict')
        if not features_dict:
            return jsonify({'success': False, 'error': 'No feature values stored for this prediction. Please run predictions again.'}), 404

        # Get player lists for player features
        feature_players = last_prediction.get('feature_players', {})

        # Get feature names from the prediction itself (no need to load model)
        # Filter out internal metadata fields (starting with _)
        feature_names = [k for k in features_dict.keys() if not k.startswith('_')]

        # Organize features by category for display
        # Note: ensemble_outputs removed - base model predictions now shown via ensemble_breakdown
        feature_categories = {
            'outcome_strength': [],
            'shooting_efficiency': [],
            'offensive_engine': [],
            'defensive_engine': [],
            'pace_volatility': [],
            'schedule_fatigue': [],
            'sample_size': [],
            'elo_strength': [],
            'era_normalization': [],
            'player_talent': [],
            'absolute_magnitude': [],
            'injuries': [],
            'point_predictions': []
        }

        # Categorize features
        for feature_name in feature_names:
            value = features_dict.get(feature_name, 0.0)
            feature_lower = feature_name.lower()

            # Skip ensemble meta-features (p_*, conf_*, disagree_*) - shown in ensemble_breakdown section
            if feature_name.startswith('p_') or feature_name.startswith('conf_') or feature_name.startswith('disagree_'):
                continue

            if feature_name.startswith('pred_'):
                feature_categories['point_predictions'].append({'name': feature_name, 'value': value})
            elif feature_name.startswith('inj_'):
                # Check injuries FIRST before checking for 'per' (inj_rotation_per contains 'per')
                feature_categories['injuries'].append({'name': feature_name, 'value': value})
            elif feature_name.startswith('player_') or 'per' in feature_lower:
                feature_categories['player_talent'].append({'name': feature_name, 'value': value})
            elif 'elo' in feature_lower:
                feature_categories['elo_strength'].append({'name': feature_name, 'value': value})
            elif 'rel' in feature_lower:
                feature_categories['era_normalization'].append({'name': feature_name, 'value': value})
            elif 'wins' in feature_lower or 'points' in feature_lower or 'margin' in feature_lower:
                feature_categories['outcome_strength'].append({'name': feature_name, 'value': value})
            elif 'efg' in feature_lower or 'ts' in feature_lower or 'three' in feature_lower:
                feature_categories['shooting_efficiency'].append({'name': feature_name, 'value': value})
            elif 'off_rtg' in feature_lower or 'assists_ratio' in feature_lower:
                feature_categories['offensive_engine'].append({'name': feature_name, 'value': value})
            elif 'def_rtg' in feature_lower or 'blocks' in feature_lower or 'reb_' in feature_lower or 'turnovers' in feature_lower:
                feature_categories['defensive_engine'].append({'name': feature_name, 'value': value})
            elif 'pace' in feature_lower or 'std' in feature_lower or 'volatility' in feature_lower:
                feature_categories['pace_volatility'].append({'name': feature_name, 'value': value})
            elif 'rest' in feature_lower or 'b2b' in feature_lower or 'travel' in feature_lower or 'games_last' in feature_lower:
                feature_categories['schedule_fatigue'].append({'name': feature_name, 'value': value})
            elif 'games_played' in feature_lower or 'sample_size' in feature_lower:
                feature_categories['sample_size'].append({'name': feature_name, 'value': value})
            else:
                feature_categories['absolute_magnitude'].append({'name': feature_name, 'value': value})
            
        # Remove empty categories
        feature_categories = {k: v for k, v in feature_categories.items() if v}
            
        # Get injured player names from last_prediction
        home_injured_players = last_prediction.get('home_injured_players', [])
        away_injured_players = last_prediction.get('away_injured_players', [])

        # Get ensemble breakdown if present (for base model feature values)
        ensemble_breakdown = features_dict.get('_ensemble_breakdown')

        return jsonify({
            'success': True,
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'game_date': game_date_str,
            'feature_categories': feature_categories,
            'home_injured_players': home_injured_players,
            'away_injured_players': away_injured_players,
            'feature_players': feature_players,  # Player lists for each player feature
            'total_features': len(feature_names),
            'ensemble_breakdown': ensemble_breakdown  # Base model feature values for ensemble models
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
        
@app.route('/<league_id>/api/calculation-details', methods=['POST'])
@app.route('/api/calculation-details', methods=['POST'])
def get_calculation_details(league_id=None):
    """Get calculation details for a team including players, features, and stats."""
    data = request.json
    
    if not data:
        return jsonify({'success': False, 'error': 'No data received'}), 400
    
    team = data.get('team')
    opponent_team = data.get('opponent_team')
    game_date_str = data.get('game_date')
    season = data.get('season')
    
    if not all([team, opponent_team, game_date_str, season]):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400
    
    try:
        game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
        year = game_date.year
        month = game_date.month
        day = game_date.day
        
        # Get model and calculators
        model = get_bball_model()
        per_calculator = get_per_calculator()
        stat_handler = model.stat_handler  # StatHandlerV2 instance
        
        # Get players from nba_rosters (excluding injured)
        roster_doc = db.nba_rosters.find_one({'season': season, 'team': team})
        if not roster_doc:
            return jsonify({'success': False, 'error': f'No roster found for {team}'}), 404
        
        roster = roster_doc.get('roster', [])
        # Get all players (including injured) for "with injured" calculation
        all_player_ids = [str(p['player_id']) for p in roster]
        # Get only non-injured players for "without injured" calculation
        player_ids = [str(p['player_id']) for p in roster if not p.get('injured', False)]
        injured_player_ids = [str(p['player_id']) for p in roster if p.get('injured', False)]
        
        # Get player details (all players)
        players_data = []
        for player_id in all_player_ids:
            player_doc = db[g.league.collections.get("players", "nba_players")].find_one({'player_id': player_id})
            if not player_doc:
                continue
            
            roster_entry = next((p for p in roster if str(p['player_id']) == player_id), {})
            
            # Get player stats
            stats = calculate_player_stats(player_id, team, season, game_date_str)
            
            # Get PER for this player
            player_per = per_calculator.get_player_per_before_date(
                player_id, team, season, game_date_str
            ) if per_calculator else 0.0
            
            players_data.append({
                'player_id': player_id,
                'player_name': player_doc.get('player_name', 'Unknown'),
                'position': player_doc.get('pos_display_name', ''),
                'is_starter': roster_entry.get('starter', False),
                'is_injured': roster_entry.get('injured', False),
                'stats': stats,
                'per': round(player_per, 2) if player_per else 0.0
            })
        
        # Build player filters for feature calculation (WITHOUT injured)
        starters = [p['player_id'] for p in players_data if p['is_starter'] and not p['is_injured']]
        player_filters = {
            team: {
                'playing': player_ids,  # Without injured
                'starters': starters
            }
        }
        
        # Calculate PER features WITHOUT injured players
        per_features = per_calculator.compute_team_per_features(
            team, season, game_date_str, top_n=8, player_filters=player_filters.get(team)
        ) if per_calculator else None
        
        # Calculate PER features WITH injured players
        all_starters = [p['player_id'] for p in players_data if p['is_starter']]
        player_filters_with_injured = {
            team: {
                'playing': all_player_ids,  # With injured
                'starters': all_starters
            }
        }
        per_features_with_injured = per_calculator.compute_team_per_features(
            team, season, game_date_str, top_n=8, player_filters=player_filters_with_injured.get(team)
        ) if per_calculator else None
        
        # Calculate injury features
        game_date_obj = game_date
        injured_ids = [str(p['player_id']) for p in roster if p.get('injured', False)]
        
        # Get injury features for this team
        injury_features = stat_handler._calculate_team_injury_features(
            team, season, game_date_str, game_date_obj, injured_ids, 
            per_calculator, recency_decay_k=15.0
        ) if stat_handler else {}
        
        # Calculate opponent injury features for diff calculation
        opponent_roster_doc = db.nba_rosters.find_one({'season': season, 'team': opponent_team})
        opponent_injured_ids = []
        if opponent_roster_doc:
            opponent_roster = opponent_roster_doc.get('roster', [])
            opponent_injured_ids = [str(p['player_id']) for p in opponent_roster if p.get('injured', False)]
        
        opponent_injury_features = stat_handler._calculate_team_injury_features(
            opponent_team, season, game_date_str, game_date_obj, opponent_injured_ids,
            per_calculator, recency_decay_k=15.0
        ) if stat_handler else {}
        
        # Calculate player-level features (WITHOUT injured)
        player_features = {}
        if per_features:
            # PER features (without injured)
            player_features['player_team_per|season|avg'] = per_features.get('per_avg', 0.0)
            player_features['player_team_per|season|weighted_MPG'] = per_features.get('per_weighted', 0.0)
            player_features['player_starters_per|season|avg'] = per_features.get('starters_avg', 0.0)
            player_features['player_per_1|season|top1_avg'] = per_features.get('per1', 0.0)
            player_features['player_per_2|season|top1_avg'] = per_features.get('per2', 0.0)
            player_features['player_per_3|season|top1_avg'] = per_features.get('per3', 0.0)
        
        # Calculate player-level features (WITH injured)
        player_features_with_injured = {}
        if per_features_with_injured:
            # PER features (with injured)
            player_features_with_injured['player_team_per|season|avg'] = per_features_with_injured.get('per_avg', 0.0)
            player_features_with_injured['player_team_per|season|weighted_MPG'] = per_features_with_injured.get('per_weighted', 0.0)
            player_features_with_injured['player_starters_per|season|avg'] = per_features_with_injured.get('starters_avg', 0.0)
            player_features_with_injured['player_per_1|season|top1_avg'] = per_features_with_injured.get('per1', 0.0)
            player_features_with_injured['player_per_2|season|top1_avg'] = per_features_with_injured.get('per2', 0.0)
            player_features_with_injured['player_per_3|season|top1_avg'] = per_features_with_injured.get('per3', 0.0)
        
        # Injury features
        if injury_features:
            # Calculate blended injury feature: 0.45 * injurySeverity + 0.35 * injTop1Per + 0.20 * injRotation
            injury_severity = injury_features.get('inj_severity|none|raw|home', 0.0)
            inj_top1_per = injury_features.get('inj_per|none|top1_avg|home', 0.0)
            inj_rotation = injury_features.get('inj_rotation_per|none|raw|home', 0.0)
            inj_impact_blend = 0.45 * injury_severity + 0.35 * inj_top1_per + 0.20 * inj_rotation
            
            player_features['inj_impact|blend|raw'] = inj_impact_blend
            player_features['inj_per|none|weighted_MIN|home'] = injury_features.get('inj_per|none|weighted_MIN|home', 0.0)
            player_features['inj_per|none|top1_avg|home'] = inj_top1_per
            player_features['inj_rotation_per|none|raw|home'] = inj_rotation
            player_features['inj_severity|none|raw|home'] = injury_severity
        
        # Calculate diffs if we have opponent data (WITHOUT injured)
        opponent_per_features = None
        opponent_per_features_with_injured = None
        if opponent_team:
            opponent_starters = []
            opponent_all_starters = []
            if opponent_roster_doc:
                opponent_roster = opponent_roster_doc.get('roster', [])
                opponent_player_ids = [str(p['player_id']) for p in opponent_roster if not p.get('injured', False)]
                opponent_all_player_ids = [str(p['player_id']) for p in opponent_roster]
                opponent_starters = [str(p['player_id']) for p in opponent_roster if p.get('starter', False) and not p.get('injured', False)]
                opponent_all_starters = [str(p['player_id']) for p in opponent_roster if p.get('starter', False)]
                
                # WITHOUT injured
                opponent_filters = {
                    opponent_team: {
                        'playing': opponent_player_ids,
                        'starters': opponent_starters
                    }
                }
                opponent_per_features = per_calculator.compute_team_per_features(
                    opponent_team, season, game_date_str, top_n=8, 
                    player_filters=opponent_filters.get(opponent_team)
                ) if per_calculator else None
                
                # WITH injured
                opponent_filters_with_injured = {
                    opponent_team: {
                        'playing': opponent_all_player_ids,
                        'starters': opponent_all_starters
                    }
                }
                opponent_per_features_with_injured = per_calculator.compute_team_per_features(
                    opponent_team, season, game_date_str, top_n=8, 
                    player_filters=opponent_filters_with_injured.get(opponent_team)
                ) if per_calculator else None
        
        # Calculate feature diffs (WITHOUT injured)
        feature_diffs = {}
        if per_features and opponent_per_features:
            feature_diffs['player_team_per|season|avg|diff'] = per_features.get('per_avg', 0.0) - opponent_per_features.get('per_avg', 0.0)
            feature_diffs['player_team_per|season|weighted_MPG|diff'] = per_features.get('per_weighted', 0.0) - opponent_per_features.get('per_weighted', 0.0)
            feature_diffs['player_starters_per|season|avg|diff'] = per_features.get('starters_avg', 0.0) - opponent_per_features.get('starters_avg', 0.0)
            feature_diffs['player_per_1|season|top1_avg|diff'] = per_features.get('per1', 0.0) - opponent_per_features.get('per1', 0.0)
            feature_diffs['player_per_2|season|top1_avg|diff'] = per_features.get('per2', 0.0) - opponent_per_features.get('per2', 0.0)
            feature_diffs['player_per_3|season|top1_avg|diff'] = per_features.get('per3', 0.0) - opponent_per_features.get('per3', 0.0)
        
        # Calculate feature diffs (WITH injured)
        feature_diffs_with_injured = {}
        if per_features_with_injured and opponent_per_features_with_injured:
            feature_diffs_with_injured['player_team_per|season|avg|diff'] = per_features_with_injured.get('per_avg', 0.0) - opponent_per_features_with_injured.get('per_avg', 0.0)
            feature_diffs_with_injured['player_team_per|season|weighted_MPG|diff'] = per_features_with_injured.get('per_weighted', 0.0) - opponent_per_features_with_injured.get('per_weighted', 0.0)
            feature_diffs_with_injured['player_starters_per|season|avg|diff'] = per_features_with_injured.get('starters_avg', 0.0) - opponent_per_features_with_injured.get('starters_avg', 0.0)
            feature_diffs_with_injured['player_per_1|season|top1_avg|diff'] = per_features_with_injured.get('per1', 0.0) - opponent_per_features_with_injured.get('per1', 0.0)
            feature_diffs_with_injured['player_per_2|season|top1_avg|diff'] = per_features_with_injured.get('per2', 0.0) - opponent_per_features_with_injured.get('per2', 0.0)
            feature_diffs_with_injured['player_per_3|season|top1_avg|diff'] = per_features_with_injured.get('per3', 0.0) - opponent_per_features_with_injured.get('per3', 0.0)
        
        if injury_features and opponent_injury_features:
            # Calculate blended injury features for both teams (new format)
            team_inj_blend = 0.45 * injury_features.get('inj_severity|none|raw|home', 0.0) + 0.35 * injury_features.get('inj_per|none|top1_avg|home', 0.0) + 0.20 * injury_features.get('inj_rotation_per|none|raw|home', 0.0)
            opponent_inj_blend = 0.45 * opponent_injury_features.get('inj_severity|none|raw|home', 0.0) + 0.35 * opponent_injury_features.get('inj_per|none|top1_avg|home', 0.0) + 0.20 * opponent_injury_features.get('inj_rotation_per|none|raw|home', 0.0)
            
            feature_diffs['inj_impact|blend|raw|diff'] = team_inj_blend - opponent_inj_blend
            feature_diffs['inj_per|none|weighted_MIN|diff'] = injury_features.get('inj_per|none|weighted_MIN|home', 0.0) - opponent_injury_features.get('inj_per|none|weighted_MIN|home', 0.0)
            feature_diffs['inj_per|none|top1_avg|diff'] = injury_features.get('inj_per|none|top1_avg|home', 0.0) - opponent_injury_features.get('inj_per|none|top1_avg|home', 0.0)
            feature_diffs['inj_rotation_per|none|raw|diff'] = injury_features.get('inj_rotation_per|none|raw|home', 0.0) - opponent_injury_features.get('inj_rotation_per|none|raw|home', 0.0)
        
        return jsonify({
            'success': True,
            'team': team,
            'players': players_data,
            'player_features': player_features,
            'player_features_with_injured': player_features_with_injured,
            'feature_diffs': feature_diffs,
            'feature_diffs_with_injured': feature_diffs_with_injured,
            'per_features': per_features,
            'per_features_with_injured': per_features_with_injured,
            'injury_features': injury_features
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/download-master-training', methods=['GET'])
@app.route('/api/download-master-training', methods=['GET'])
def download_master_training(league_id=None):
    """Download the master training data CSV file."""
    try:
        master_training_path = get_master_training_path()
        if os.path.exists(master_training_path):
            return send_file(
                master_training_path,
                as_attachment=True,
                download_name='MASTER_TRAINING.csv',
                mimetype='text/csv'
            )
        else:
            return jsonify({'error': 'Master training data file not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<league_id>/api/download-config-training', methods=['GET'])
@app.route('/api/download-config-training', methods=['GET'])
def download_config_training(league_id=None):
    """Download the training data CSV file for the current model configuration."""
    try:
        # Get training CSV from current selected config
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        selected_config = db[classifier_config_collection].find_one({'selected': True})
        if not selected_config:
            return jsonify({'error': 'No model configuration selected'}), 404
        
        training_csv = selected_config.get('training_csv')
        if not training_csv:
            return jsonify({'error': 'Training data file not found in configuration'}), 404
        
        # If the stored path doesn't exist (old location), try to map to new ../model_outputs
        if not os.path.exists(training_csv):
            try:
                outputs_root = os.path.join(os.path.dirname(parent_dir), 'model_outputs')
            except Exception:
                outputs_root = 'model_outputs'
            # Candidate 1: same basename in consolidated outputs dir
            candidate1 = os.path.join(outputs_root, os.path.basename(training_csv))
            # Candidate 2: replace old nba_app/model_output segment with new consolidated
            old_seg = os.path.join(parent_dir, 'model_output')
            new_seg = outputs_root
            candidate2 = training_csv.replace(old_seg, new_seg) if old_seg in training_csv else None
            new_path = None
            for cand in [candidate1, candidate2]:
                if cand and os.path.exists(cand):
                    new_path = cand
                    break
            if not new_path:
                return jsonify({'error': 'Training data file does not exist'}), 404
            # Update DB with corrected path for future requests
            db[classifier_config_collection].update_one({'_id': selected_config['_id']}, {'$set': {'training_csv': new_path}})
            training_csv = new_path
        
        # Get filename from path
        filename = os.path.basename(training_csv)
        
        return send_file(
            training_csv,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/<league_id>/api/pull-game-data', methods=['POST'])
@app.route('/api/pull-game-data', methods=['POST'])
def pull_game_data(league_id=None):
    """
    Pull game data from ESPN API for a given date.

    Thin wrapper around core sync_pipeline - spins off background job
    and returns job_id for frontend polling.
    """
    from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job
    from bball_app.core.league_config import load_league_config

    data = request.json
    date_str = data.get('date')

    if not date_str:
        return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

    try:
        # Validate date format
        game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format'}), 400

    # Create job and return immediately
    league = g.league
    league_id_str = league.league_id if league else 'nba'
    jobs_collection_name = league.collections.get('jobs', 'jobs_nba') if league else 'jobs_nba'
    logger.info(f"[SYNC] Creating sync job for date={date_str}, league={league_id_str}, jobs_collection={jobs_collection_name}")
    job_id = create_job(
        job_type='sync',
        league=league,
        metadata={'date': date_str, 'league': league_id_str}
    )
    logger.info(f"[SYNC] Created job_id={job_id}")

    def run_sync_background(job_id: str, date_str: str, league_id: str):
        """Background worker for data sync."""
        from bball_app.core.pipeline.sync_pipeline import run_sync_pipeline

        logger.info(f"[SYNC BG] Starting background sync for job {job_id}, date={date_str}, league={league_id}")

        try:
            # Load league config fresh (not from Flask context)
            league_config = load_league_config(league_id)
            from bball_app.core.mongo import Mongo
            bg_db = Mongo().db

            game_date = datetime.strptime(date_str, '%Y-%m-%d').date()

            # Update progress: starting sync
            logger.info(f"[SYNC BG] Job {job_id}: updating progress to 10%")
            update_job_progress(job_id, 10, 'Starting ESPN data sync...', league=league_config)

            # Run sync pipeline (handles games, players, player_stats)
            logger.info(f"[SYNC BG] Job {job_id}: updating progress to 20%, starting pipeline")
            update_job_progress(job_id, 20, 'Pulling games and player stats from ESPN...', league=league_config)

            # Map pipeline progress (0-100) to job progress (20-85)
            def sync_progress(pct, msg):
                job_pct = 20 + int(pct * 0.65)  # 0->20, 100->85
                update_job_progress(job_id, job_pct, msg, league=league_config)

            result = run_sync_pipeline(
                league_config=league_config,
                start_date=game_date,
                end_date=game_date,
                data_types={'games', 'player_stats'},
                dry_run=False,
                verbose=False,
                progress_callback=sync_progress
            )

            logger.info(f"[SYNC BG] Job {job_id}: pipeline complete, counting games")
            update_job_progress(job_id, 90, 'Counting synced games...', league=league_config)

            # Get stats for completion message
            games_collection = league_config.collections.get('games', 'stats_nba')
            games_with_homewon = bg_db[games_collection].count_documents({
                'date': date_str,
                'homeWon': {'$exists': True}
            })
            total_games = bg_db[games_collection].count_documents({
                'date': date_str
            })

            logger.info(f"[SYNC BG] Job {job_id}: calling complete_job with {total_games} games")
            complete_job(
                job_id,
                f'Synced {total_games} games ({games_with_homewon} completed)',
                league=league_config
            )
            logger.info(f"[SYNC BG] Job {job_id}: complete_job finished")

        except Exception as e:
            import traceback
            logger.error(f"[SYNC BG] Job {job_id} failed with exception: {e}")
            traceback.print_exc()
            try:
                league_config = load_league_config(league_id)
                fail_job(job_id, str(e), f'Sync failed: {str(e)}', league=league_config)
            except Exception as e2:
                logger.error(f"[SYNC BG] Job {job_id} failed to call fail_job: {e2}")

    # Start background thread
    thread = threading.Thread(
        target=run_sync_background,
        args=(job_id, date_str, league_id_str),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Sync job started'
    })


@app.route('/<league_id>/api/predict-all', methods=['POST'])
@app.route('/api/predict-all', methods=['POST'])
def predict_all(league_id=None):
    """
    Generate predictions for all games on a given date.

    Thin wrapper around core PredictionService - spins off background job
    and returns job_id for frontend polling.
    """
    from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job
    from bball_app.core.services.prediction import PredictionService
    from bball_app.core.league_config import load_league_config

    data = request.json
    date_str = data.get('date')

    if not date_str:
        return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

    try:
        game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format'}), 400

    # Validate config before starting job
    league = g.league
    classifier_config_collection = league.collections.get('model_config_classifier', 'nba_model_config')
    selected_classifier_config = db[classifier_config_collection].find_one({'selected': True})

    is_valid, error_msg = ModelConfigManager.validate_config_for_prediction(selected_classifier_config)
    if not is_valid:
        return jsonify({'success': False, 'error': error_msg}), 400

    # Create job and return immediately
    league_id_str = league.league_id if league else 'nba'
    job_id = create_job(
        job_type='predict_all',
        league=league,
        metadata={'date': date_str, 'league': league_id_str}
    )

    def run_predictions_background(job_id: str, date_str: str, league_id: str):
        """Background worker for predictions."""
        try:
            # Load league config fresh (not from Flask context)
            league_config = load_league_config(league_id)
            from bball_app.core.mongo import Mongo
            bg_db = Mongo().db

            # Use core PredictionService with job_id for progress tracking
            service = PredictionService(db=bg_db, league=league_config)

            game_date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
            successful = 0
            failed = 0

            # Callback to save each prediction immediately (for real-time UI updates)
            def save_prediction_callback(result, matchup):
                nonlocal successful, failed
                if result.error:
                    failed += 1
                    return
                try:
                    service.save_prediction(
                        result=result,
                        game_id=result.game_id,
                        game_date=game_date_obj,
                        home_team=result.home_team,
                        away_team=result.away_team
                    )
                    successful += 1
                except Exception as e:
                    print(f"Error saving prediction for {result.game_id}: {e}")
                    failed += 1

            # predict_date() updates job progress internally at:
            # 10% - Fetching matchups
            # 15% - Loading models
            # 20% - Starting predictions
            # 20-90% - Per-game progress (predictions saved incrementally via callback)
            results = service.predict_date(
                game_date=date_str,
                include_points=True,
                job_id=job_id,
                on_prediction=save_prediction_callback
            )

            complete_job(
                job_id,
                f'Completed {successful}/{len(results)} predictions',
                league=league_config
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            try:
                league_config = load_league_config(league_id)
                fail_job(job_id, str(e), f'Prediction failed: {str(e)}', league=league_config)
            except:
                pass

    # Start background thread
    thread = threading.Thread(
        target=run_predictions_background,
        args=(job_id, date_str, league_id_str),
        daemon=True
    )
    thread.start()

    return jsonify({
        'success': True,
        'job_id': job_id,
        'message': 'Prediction job started'
    })


@app.route('/<league_id>/api/predictions/<date_str>', methods=['GET'])
@app.route('/api/predictions/<date_str>', methods=['GET'])
def get_predictions_by_date(date_str, league_id=None):
    """Get all predictions for a specific date (used for real-time UI updates during bulk prediction)."""
    try:
        predictions_collection = g.league.collections.get('model_predictions', 'nba_model_predictions')
        predictions = list(db[predictions_collection].find(
            {'game_date': date_str},
            {'_id': 0}  # Exclude MongoDB _id
        ))

        return jsonify({
            'success': True,
            'predictions': predictions
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/betting-report', methods=['POST'])
@app.route('/api/betting-report', methods=['POST'])
def generate_betting_report_endpoint(league_id=None):
    """Generate betting report comparing model vs market odds."""
    try:
        data = request.json
        date_str = data.get('date')
        bankroll = float(data.get('bankroll', 1000))
        edge_threshold = float(data.get('edge_threshold', 0.07))
        force_include_game_ids = data.get('force_include_game_ids')

        if not date_str:
            return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

        # Get Brier score from selected model config
        config_coll = g.league.collections.get('model_config_classifier', 'nba_model_config')
        selected_config = db[config_coll].find_one({'selected': True})
        brier_score = selected_config.get('brier_score', 0.25) if selected_config else 0.25

        from bball_app.core.services.betting_report import generate_betting_report
        recommendations = generate_betting_report(db, date_str, bankroll, brier_score, g.league, edge_threshold, force_include_game_ids)

        return jsonify({
            'success': True,
            'recommendations': [r.to_dict() for r in recommendations],
            'brier_score': brier_score,
            'bankroll': bankroll,
            'edge_threshold': edge_threshold
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model-config')
@app.route('/<league_id>/model-config')
def model_config(league_id=None):
    """Model configuration page."""
    # Load features directly from MASTER_TRAINING.csv
    feature_sets_dict, feature_set_descriptions = load_features_from_master_csv()

    # If CSV loading failed, fallback to feature_sets.py (for backward compatibility)
    if not feature_sets_dict:
        print("Warning: Failed to load features from CSV, falling back to feature_sets.py")
        from bball_app.core.features.sets import FEATURE_SETS, FEATURE_SET_DESCRIPTIONS
        feature_sets_dict = FEATURE_SETS
        feature_set_descriptions = FEATURE_SET_DESCRIPTIONS

    # Get league-aware collection name
    classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

    # Try to load selected config from MongoDB
    default_config = None
    try:
        selected_config = db[classifier_config_collection].find_one({'selected': True})
        if selected_config:
            default_config = {
                'model_types': [selected_config.get('model_type')],
                'features': selected_config.get('features', []),
                'c_values': [selected_config.get('best_c_value')] if selected_config.get('best_c_value') is not None else []
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass  # Fall through to None

    # Build flat set of all available features from the master CSV
    available_features = {f for feats in feature_sets_dict.values() for f in feats}

    return render_template(
        'model_config.html',
        feature_sets=feature_sets_dict,
        feature_set_descriptions=feature_set_descriptions,
        default_config=default_config,
        available_features=available_features
    )

@app.route('/model-config-points')
@app.route('/<league_id>/model-config-points')
def model_config_points(league_id=None):
    """Points regression model configuration page."""
    # Load features directly from MASTER_TRAINING.csv (same as /model-config)
    feature_sets_dict, feature_set_descriptions = load_features_from_master_csv()

    # If CSV loading failed, fallback to hardcoded feature sets (for backward compatibility)
    if not feature_sets_dict:
        print("Warning: Failed to load features from CSV for points config, falling back to hardcoded sets")
        from bball_app.core.features.sets import FEATURE_SETS, FEATURE_SET_DESCRIPTIONS
        feature_sets_dict = dict(FEATURE_SETS)
        feature_set_descriptions = FEATURE_SET_DESCRIPTIONS

    # Get league-aware collection name
    points_config_collection = g.league.collections.get('model_config_points', 'model_config_points_nba')

    # Try to load selected config from MongoDB
    default_config = None
    try:
        selected_config = db[points_config_collection].find_one({'selected': True})
        if selected_config:
            default_config = {
                'model_type': selected_config.get('model_type'),
                'features': selected_config.get('features', []),
            }
    except Exception as e:
        import traceback
        traceback.print_exc()
        pass  # Fall through to None
    
    # Build flat set of all available features from the master CSV
    available_features = {f for feats in feature_sets_dict.values() for f in feats}

    return render_template(
        'model_config_points.html',
        feature_sets=feature_sets_dict,
        feature_set_descriptions=feature_set_descriptions,
        default_config=default_config,
        available_features=available_features
    )


@app.route('/ensemble-config')
@app.route('/<league_id>/ensemble-config')
def ensemble_config(league_id=None):
    """Ensemble model configuration page."""
    feature_sets_dict, feature_set_descriptions = load_features_from_master_csv()

    if not feature_sets_dict:
        from bball_app.core.features.sets import FEATURE_SETS, FEATURE_SET_DESCRIPTIONS
        feature_sets_dict = dict(FEATURE_SETS)
        feature_set_descriptions = FEATURE_SET_DESCRIPTIONS

    return render_template(
        'ensemble_config.html',
        feature_sets=feature_sets_dict,
        feature_set_descriptions=feature_set_descriptions,
    )


@app.route('/elo-manager')
@app.route('/<league_id>/elo-manager')
def elo_manager(league_id=None):
    """Elo ratings manager page."""
    return render_template('elo_manager.html')


@app.route('/injuries-manager')
@app.route('/<league_id>/injuries-manager')
def injuries_manager(league_id=None):
    """Injuries manager page."""
    return render_template('injuries_manager.html')


@app.route('/cached-league-stats')
@app.route('/<league_id>/cached-league-stats')
def cached_league_stats_manager(league_id=None):
    """Cached league stats manager page."""
    return render_template('cached_league_stats.html')


@app.route('/espn-db-audit')
@app.route('/<league_id>/espn-db-audit')
def espn_db_audit_manager(league_id=None):
    """ESPN DB audit page."""
    return render_template('espn_db_audit.html')


@app.route('/market-dashboard')
@app.route('/<league_id>/market-dashboard')
def market_dashboard(league_id=None):
    """Market dashboard page."""
    return render_template('market_dashboard.html')


@app.route('/<league_id>/api/points-model/configs', methods=['GET'])
@app.route('/api/points-model/configs', methods=['GET'])
def get_points_model_configs(league_id=None):
    """Get all points model configurations."""
    try:
        points_config_collection = g.league.collections.get('model_config_points', 'model_config_points_nba')
        configs = list(db[points_config_collection].find({}).sort('trained_at', -1))
        
        # Convert ObjectId to string and sanitize
        for config in configs:
            config['_id'] = str(config['_id'])
            # Remove non-serializable fields if any
            if 'model_path' in config:
                # Keep model_path as string
                pass
        
        return jsonify({
            'success': True,
            'configs': configs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<league_id>/api/points-model/select-config', methods=['POST'])
@app.route('/api/points-model/select-config', methods=['POST'])
def select_points_model_config(league_id=None):
    """Select or deselect a points model configuration."""
    try:
        points_config_collection = g.league.collections.get('model_config_points', 'model_config_points_nba')
        data = request.json
        config_id = data.get('config_id')

        if not config_id:
            return jsonify({
                'success': False,
                'error': 'config_id is required'
            }), 400

        from bson import ObjectId
        # Check if this config is currently selected
        current_config = db[points_config_collection].find_one({'_id': ObjectId(config_id)})
        
        if not current_config:
            return jsonify({
                'success': False,
                'error': 'Config not found'
            }), 404
        
        is_currently_selected = current_config.get('selected', False)
        
        # Unset all selected configs first (ensures only 0 or 1 selected)
        db[points_config_collection].update_many(
            {'selected': True},
            {'$set': {'selected': False}}
        )

        # If it was already selected, we're done (deselected it above)
        # If it wasn't selected, now select it
        if not is_currently_selected:
            result = db[points_config_collection].update_one(
                {'_id': ObjectId(config_id)},
                {'$set': {'selected': True}}
            )
        
        if result.modified_count == 0:
            return jsonify({
                'success': False,
                    'error': 'Failed to select config'
                }), 500
        
        return jsonify({
            'success': True,
            'message': 'Config deselected successfully' if is_currently_selected else 'Config selected successfully',
            'selected': not is_currently_selected
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<league_id>/api/model-config/save', methods=['POST'])
@app.route('/api/model-config/save', methods=['POST'])
def save_model_config(league_id=None):
    """
    Save model configuration from UI.
    Thin wrapper: parses request, delegates to ModelConfigManager.
    """
    try:
        config = request.json
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.info(f"/api/model-config/save payload: {config}")

        # Ensemble configs are saved during training, not via this endpoint
        if config.get('ensemble', False):
            logger.info("Ensemble config detected - skipping save (handled by training endpoint)")
            return jsonify({
                'success': True,
                'message': 'Ensemble config will be saved after training completes',
                'ensemble': True
            })

        parsed = _parse_training_config(config)

        if not parsed['model_types'] or not parsed['features']:
            return jsonify({
                'success': False,
                'error': 'model_types and features (or feature_sets) are required'
            }), 400

        logger.info(f"Requested features: {len(parsed['features'])} features")

        mgr = ModelConfigManager(db, league=g.league)
        saved_configs = []

        # If updating an existing config, preserve its name
        existing_name = None
        existing_config_id = config.get('config_id')
        if existing_config_id:
            classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
            existing_doc = db[classifier_config_collection].find_one({'_id': ObjectId(existing_config_id)})
            if existing_doc:
                existing_name = existing_doc.get('name')

        for idx, model_type in enumerate(parsed['model_types']):
            c_value = None
            if model_type in C_SUPPORTED_MODELS and parsed['c_values']:
                try:
                    c_value = float(parsed['c_values'][0])
                except Exception:
                    c_value = parsed['c_values'][0]

            is_selected = (idx == 0)
            config_id, cfg = mgr.create_classifier_config(
                model_type=model_type,
                features=sorted(parsed['features']),
                c_value=c_value,
                use_time_calibration=parsed['use_time_calibration'],
                calibration_method=parsed['calibration_method'],
                begin_year=parsed['begin_year'],
                calibration_years=parsed['calibration_years'],
                evaluation_year=parsed['evaluation_year'],
                min_games_played=parsed['min_games_played'],
                include_injuries=parsed['include_injuries'],
                recency_decay_k=parsed['recency_decay_k'],
                exclude_seasons=parsed['exclude_seasons'],
                use_master=parsed['use_master'],
                point_model_id=parsed['point_model_id'],
                selected=is_selected,
                name=existing_name,
            )
            logger.info(f"Model {model_type}: config_id={config_id}, hash={cfg.get('config_hash', '')[:8]}")
            saved_configs.append({
                'model_type': model_type,
                'feature_set_hash': cfg.get('feature_set_hash', ''),
                'config_id': config_id
            })

        return jsonify({
            'success': True,
            'message': f"Config saved successfully for {len(saved_configs)} model type(s)",
            'saved_configs': saved_configs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-config/load', methods=['GET'])
@app.route('/api/model-config/load', methods=['GET'])
def load_model_config(league_id=None):
    """Load last saved model configuration (backward compatibility)."""
    try:
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        # Try to get selected config from league-specific collection
        selected_config = db[classifier_config_collection].find_one({'selected': True})
        if selected_config:
            # Convert MongoDB document to config format expected by frontend
            config = {
                'model_types': [selected_config.get('model_type')],
                'features': selected_config.get('features', []),
                'c_values': [selected_config.get('best_c_value')] if selected_config.get('best_c_value') is not None else [],
                # Default to True if missing to honor UX: cached master should be on by default
                'use_master': selected_config.get('use_master', True),
                'use_time_calibration': selected_config.get('use_time_calibration', False),
                'calibration_method': selected_config.get('calibration_method'),
                'begin_year': selected_config.get('begin_year'),
                'calibration_years': selected_config.get('calibration_years') or ([selected_config.get('calibration_year')] if selected_config.get('calibration_year') is not None else None),
                'evaluation_year': selected_config.get('evaluation_year'),
                'include_injuries': selected_config.get('include_injuries', False),
                'recency_decay_k': selected_config.get('recency_decay_k'),
                'min_games_played': selected_config.get('min_games_played', 15),
                'point_model_id': selected_config.get('point_model_id')  # Optional reference to point prediction model
            }
            return jsonify({
                'success': True,
                'config': config
            })
        
        # Fallback: try old model_configs collection
        doc = db.model_configs.find_one({'_id': 'current'})
        if doc and 'config' in doc:
            return jsonify({'success': True, 'config': doc['config']})
        
        return jsonify({'success': False, 'error': 'No config found'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-configs', methods=['GET'])
@app.route('/api/model-configs', methods=['GET'])
def get_all_model_configs(league_id=None):
    """Get all model configurations from MongoDB."""
    try:
        import logging
        logger = logging.getLogger(__name__)

        # Get league-aware collection name
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        # Sort by selected status first (selected=True first), then by trained_at descending
        # Use updated_at as fallback for sorting
        configs = list(db[classifier_config_collection].find({}).sort([
            ('selected', -1),  # True comes before False in descending sort
            ('trained_at', -1),
            ('updated_at', -1)
        ]))

        # Enforce exactly one selected
        try:
            selected_configs = [c for c in configs if c.get('selected') is True]
            if configs:
                if len(selected_configs) == 0:
                    # Select the newest (first in current sort by date)
                    primary = configs[0]
                    db[classifier_config_collection].update_one(
                        {'_id': primary['_id']},
                        {'$set': {'selected': True, 'updated_at': datetime.utcnow()}}
                    )
                    # Reflect change in memory
                    primary['selected'] = True
                    logger.info(f"/api/model-configs: No selected configs found. Auto-selected _id={primary['_id']}")
                elif len(selected_configs) > 1:
                    # Keep the newest selected, unselect the rest
                    # Sort selected by trained_at/updated_at
                    def sort_key(c):
                        return (
                            c.get('trained_at') or datetime.min,
                            c.get('updated_at') or datetime.min
                        )
                    selected_configs.sort(key=sort_key, reverse=True)
                    keep = selected_configs[0]
                    to_unselect_ids = [c['_id'] for c in selected_configs[1:]]
                    if to_unselect_ids:
                        db[classifier_config_collection].update_many(
                            {'_id': {'$in': to_unselect_ids}},
                            {'$set': {'selected': False}}
                        )
                        # Reflect change in memory
                        for c in configs:
                            if c['_id'] in to_unselect_ids:
                                c['selected'] = False
                        logger.info(f"/api/model-configs: Reduced multiple selected to one. Kept _id={keep['_id']}, unselected {len(to_unselect_ids)} others")
        except Exception as e:
            logger.warning(f"/api/model-configs: enforcement error: {e}")

        # Re-sort after enforcement
        configs.sort(key=lambda c: (
            1 if c.get('selected') else 0,
            (c.get('trained_at') or datetime.min),
            (c.get('updated_at') or datetime.min)
        ), reverse=True)

        # Convert ObjectId to string for JSON serialization
        for config in configs:
            config['_id'] = str(config['_id'])
            # Convert datetime to ISO string
            if 'trained_at' in config and config['trained_at']:
                config['trained_at'] = config['trained_at'].isoformat()
            elif 'trained_at' not in config:
                config['trained_at'] = None
            if 'updated_at' in config and config['updated_at']:
                config['updated_at'] = config['updated_at'].isoformat()
            elif 'updated_at' not in config:
                config['updated_at'] = None
        
        return jsonify({
            'success': True,
            'configs': configs
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-configs/selected', methods=['GET'])
@app.route('/api/model-configs/selected', methods=['GET'])
def get_selected_model_config(league_id=None):
    """Get the currently selected model configuration."""
    try:
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        config = db[classifier_config_collection].find_one({'selected': True})

        if not config:
            return jsonify({
                'success': False,
                'error': 'No selected config found'
            }), 404

        # Convert ObjectId to string
        config['_id'] = str(config['_id'])
        # Convert datetime to ISO string
        if 'trained_at' in config and config['trained_at']:
            config['trained_at'] = config['trained_at'].isoformat()
        if 'updated_at' in config and config['updated_at']:
            config['updated_at'] = config['updated_at'].isoformat()

        return jsonify({
            'success': True,
            'config': config
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-configs/<config_id>', methods=['GET'])
@app.route('/api/model-configs/<config_id>', methods=['GET'])
def get_model_config(config_id, league_id=None):
    """Get a single model configuration by ID."""
    try:
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        config = db[classifier_config_collection].find_one({'_id': ObjectId(config_id)})

        if not config:
            return jsonify({'success': False, 'error': 'Config not found'}), 404

        config['_id'] = str(config['_id'])
        if 'trained_at' in config and config['trained_at']:
            config['trained_at'] = config['trained_at'].isoformat()
        if 'updated_at' in config and config['updated_at']:
            config['updated_at'] = config['updated_at'].isoformat()

        return jsonify({'success': True, 'config': config})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-configs/<config_id>', methods=['PUT'])
@app.route('/api/model-configs/<config_id>', methods=['PUT'])
def update_model_config(config_id, league_id=None):
    """Update a model configuration."""
    try:
        from bball_app.core.services.config_manager import ModelConfigManager

        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        data = request.json

        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400

        update_fields = {}

        # Update name if provided
        if 'name' in data:
            update_fields['name'] = data['name']

        # Handle selected flag
        if 'selected' in data:
            if data['selected'] is True:
                # Unset all other configs
                db[classifier_config_collection].update_many(
                    {'selected': True},
                    {'$set': {'selected': False}}
                )
                update_fields['selected'] = True
            else:
                update_fields['selected'] = False

        # Handle model_type
        if 'model_type' in data:
            update_fields['model_type'] = data['model_type']

        # Handle features list
        if 'features' in data:
            features = data['features']
            update_fields['features'] = sorted(features)
            update_fields['feature_count'] = len(features)
            update_fields['feature_set_hash'] = ModelConfigManager.generate_feature_set_hash(features)

        # Handle best_c_value
        if 'best_c_value' in data:
            update_fields['best_c_value'] = data['best_c_value']

        # Handle include_injuries
        if 'include_injuries' in data:
            update_fields['include_injuries'] = bool(data['include_injuries'])

        if not update_fields:
            return jsonify({
                'success': False,
                'error': 'No valid fields to update'
            }), 400

        update_fields['updated_at'] = datetime.utcnow()

        # Recompute config_hash if any hash-relevant fields changed
        hash_relevant = {'model_type', 'features', 'best_c_value', 'include_injuries'}
        if hash_relevant & set(data.keys()):
            current_doc = db[classifier_config_collection].find_one({'_id': ObjectId(config_id)})
            if current_doc:
                merged = {**current_doc, **update_fields}
                fs_hash = merged.get('feature_set_hash') or ModelConfigManager.generate_feature_set_hash(merged.get('features', []))
                update_fields['config_hash'] = ModelConfigManager.generate_config_hash(
                    model_type=merged.get('model_type', 'LogisticRegression'),
                    feature_set_hash=fs_hash,
                    c_value=merged.get('best_c_value'),
                    use_time_calibration=merged.get('use_time_calibration', False),
                    calibration_method=merged.get('calibration_method'),
                    calibration_years=merged.get('calibration_years'),
                    begin_year=merged.get('begin_year'),
                    evaluation_year=merged.get('evaluation_year'),
                    include_injuries=merged.get('include_injuries', False),
                    recency_decay_k=merged.get('recency_decay_k'),
                    use_master=merged.get('use_master', True),
                    min_games_played=merged.get('min_games_played', 15),
                    exclude_seasons=merged.get('exclude_seasons'),
                )

        # Update the config
        result = db[classifier_config_collection].update_one(
            {'_id': ObjectId(config_id)},
            {'$set': update_fields}
        )

        if result.matched_count == 0:
            return jsonify({
                'success': False,
                'error': 'Config not found'
            }), 404

        return jsonify({
            'success': True,
            'message': 'Config updated successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/model-configs/<config_id>', methods=['DELETE'])
@app.route('/api/model-configs/<config_id>', methods=['DELETE'])
def delete_model_config(config_id, league_id=None):
    """Delete a model configuration from the database."""
    try:
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        # Check if config exists and if it's selected
        config = db[classifier_config_collection].find_one({'_id': ObjectId(config_id)})

        if not config:
            return jsonify({
                'success': False,
                'error': 'Config not found'
            }), 404

        # If this config is selected, we should probably not allow deletion
        # or automatically select another one. For now, we'll just warn but allow deletion.
        was_selected = config.get('selected', False)

        # Delete the config
        result = db[classifier_config_collection].delete_one({'_id': ObjectId(config_id)})

        if result.deleted_count == 0:
            return jsonify({
                'success': False,
                'error': 'Failed to delete config'
            }), 500

        # If it was selected, optionally select the most recent config
        if was_selected:
            latest_config = db[classifier_config_collection].find_one(
                {},
                sort=[('trained_at', -1), ('updated_at', -1)]
            )
            if latest_config:
                db[classifier_config_collection].update_one(
                    {'_id': latest_config['_id']},
                    {'$set': {'selected': True, 'updated_at': datetime.utcnow()}}
                )
        
        return jsonify({
            'success': True,
            'message': 'Config deleted successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles', methods=['GET'])
@app.route('/api/ensembles', methods=['GET'])
def get_ensembles(league_id=None):
    """Get all ensemble configurations."""
    try:
        from bball_app.core.services.training_service import TrainingService
        service = TrainingService(league=g.league, db=db)
        ensembles = service.list_ensembles()
        return jsonify({'success': True, 'ensembles': ensembles})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/available-base-models', methods=['GET'])
@app.route('/api/ensembles/available-base-models', methods=['GET'])
def get_available_base_models(league_id=None):
    """Get all trained non-ensemble models available for use in ensembles."""
    try:
        # Get league-aware collection
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        # Find trained non-ensemble models (must have trained_at and no ensemble_models field)
        base_models = list(db[config_collection].find({
            'trained_at': {'$exists': True},
            'ensemble_models': {'$exists': False}
        }))

        # Convert ObjectId to string
        for model in base_models:
            model['_id'] = str(model['_id'])

        return jsonify({
            'success': True,
            'models': base_models
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/<ensemble_id>', methods=['GET'])
@app.route('/api/ensembles/<ensemble_id>', methods=['GET'])
def get_ensemble(ensemble_id, league_id=None):
    """Get a specific ensemble configuration."""
    try:
        from bson import ObjectId
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        ensemble = db[config_collection].find_one({'_id': ObjectId(ensemble_id)})
        if not ensemble:
            return jsonify({'success': False, 'error': 'Ensemble not found'}), 404

        ensemble['_id'] = str(ensemble['_id'])
        if 'ensemble_models' in ensemble:
            ensemble['ensemble_models'] = [str(m) if not isinstance(m, str) else m for m in ensemble['ensemble_models']]

        return jsonify({
            'success': True,
            'ensemble': ensemble
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/<ensemble_id>', methods=['PUT'])
@app.route('/api/ensembles/<ensemble_id>', methods=['PUT'])
def update_ensemble(ensemble_id, league_id=None):
    """Update an ensemble configuration."""
    try:
        from bson import ObjectId
        from datetime import datetime
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Find existing ensemble
        existing = db[config_collection].find_one({'_id': ObjectId(ensemble_id)})
        if not existing:
            return jsonify({'success': False, 'error': 'Ensemble not found'}), 404

        # Build update document
        update_fields = {}
        allowed_fields = ['name', 'ensemble_models', 'ensemble_type', 'ensemble_meta_features',
                          'ensemble_use_disagree', 'ensemble_use_conf', 'stacking_mode', 'selected',
                          'model_type', 'best_c_value', 'meta_model_type', 'meta_c_value']

        for field in allowed_fields:
            if field in data:
                update_fields[field] = data[field]

        if update_fields:
            update_fields['updated_at'] = datetime.utcnow()
            db[config_collection].update_one(
                {'_id': ObjectId(ensemble_id)},
                {'$set': update_fields}
            )

        return jsonify({
            'success': True,
            'message': 'Ensemble updated successfully'
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/validate', methods=['POST'])
@app.route('/api/ensembles/validate', methods=['POST'])
def validate_ensemble(league_id=None):
    """Validate ensemble configuration before saving."""
    try:
        from bson import ObjectId
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        ensemble_models = data.get('ensemble_models', [])
        if len(ensemble_models) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 models are required for ensemble'
            }), 400

        # Validate all models exist and have compatible time configs
        base_models = []
        for model_id in ensemble_models:
            model = db[config_collection].find_one({'_id': ObjectId(model_id)})
            if not model:
                return jsonify({
                    'success': False,
                    'error': f'Model {model_id} not found'
                }), 404
            if model.get('ensemble_models'):
                return jsonify({
                    'success': False,
                    'error': 'Cannot include ensemble model in another ensemble'
                }), 400
            base_models.append(model)

        # Validate time configs match
        if base_models:
            ref = base_models[0]
            for model in base_models[1:]:
                for key in ['begin_year', 'calibration_years', 'evaluation_year']:
                    if model.get(key) != ref.get(key):
                        return jsonify({
                            'success': False,
                            'error': f'Time config mismatch: {key} differs between models'
                        }), 400

        return jsonify({'success': True, 'message': 'Validation passed'})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/<ensemble_id>/retrain-meta', methods=['POST'])
@app.route('/api/ensembles/<ensemble_id>/retrain-meta', methods=['POST'])
def retrain_ensemble_meta(ensemble_id, league_id=None):
    """Retrain the meta-model for an ensemble."""
    try:
        from bson import ObjectId
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        ensemble = db[config_collection].find_one({'_id': ObjectId(ensemble_id)})
        if not ensemble:
            return jsonify({'success': False, 'error': 'Ensemble not found'}), 404

        if not ensemble.get('ensemble_models'):
            return jsonify({'success': False, 'error': 'Not an ensemble configuration'}), 400

        # Get updated meta-feature settings from request
        data = request.json or {}
        ensemble_meta_features = data.get('ensemble_meta_features', ensemble.get('ensemble_meta_features', []))
        ensemble_use_disagree = data.get('ensemble_use_disagree', ensemble.get('ensemble_use_disagree', False))
        ensemble_use_conf = data.get('ensemble_use_conf', ensemble.get('ensemble_use_conf', False))
        meta_model_type = data.get('meta_model_type', ensemble.get('model_type', 'LogisticRegression'))

        # Persist the updated flags to DB
        db[config_collection].update_one(
            {'_id': ObjectId(ensemble_id)},
            {'$set': {
                'ensemble_meta_features': ensemble_meta_features,
                'ensemble_use_disagree': ensemble_use_disagree,
                'ensemble_use_conf': ensemble_use_conf,
                'model_type': meta_model_type,
            }}
        )

        # Build config dict and route to train_ensemble_model()
        train_config = {
            'ensemble': True,
            'ensemble_models': ensemble.get('ensemble_models', []),
            'ensemble_meta_features': ensemble_meta_features,
            'ensemble_use_disagree': ensemble_use_disagree,
            'ensemble_use_conf': ensemble_use_conf,
            'model_types': [meta_model_type],
            'c_values': [ensemble.get('best_c_value', 0.1)],
        }

        return train_ensemble_model(train_config, classifier_config_collection=config_collection, league=g.league)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/ensembles/<ensemble_id>/retrain-base/<base_model_id>', methods=['POST'])
@app.route('/api/ensembles/<ensemble_id>/retrain-base/<base_model_id>', methods=['POST'])
def retrain_ensemble_base(ensemble_id, base_model_id, league_id=None):
    """Retrain a base model within an ensemble, then retrain the ensemble meta-model."""
    from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job

    try:
        from bson import ObjectId
        config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        ensemble = db[config_collection].find_one({'_id': ObjectId(ensemble_id)})
        if not ensemble:
            return jsonify({'success': False, 'error': 'Ensemble not found'}), 404

        ensemble_models = ensemble.get('ensemble_models', [])
        if base_model_id not in [str(m) for m in ensemble_models]:
            return jsonify({'success': False, 'error': 'Base model not part of this ensemble'}), 400

        league = g.league
        league_id_str = league.league_id if league else 'nba'

        job_id = create_job(
            job_type='retrain_base',
            league=league,
            metadata={'ensemble_id': ensemble_id, 'base_model_id': base_model_id, 'league': league_id_str}
        )

        thread = threading.Thread(
            target=_run_retrain_base_job,
            args=(job_id, ensemble_id, base_model_id, league_id_str),
            daemon=True,
        )
        thread.start()

        return jsonify({'success': True, 'job_id': job_id})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


def _run_retrain_base_job(job_id, ensemble_id, base_model_id, league_id):
    """Background job: retrain a base model then retrain the ensemble meta-model."""
    from bball_app.core.league_config import load_league_config
    from bball_app.core.services.config_manager import ModelConfigManager
    from bball_app.core.services.training_service import TrainingService
    from bball_app.core.training.experiment_runner import ExperimentRunner
    from bball_app.core.mongo import Mongo
    from bball_app.core.services.jobs import update_job_progress, complete_job, fail_job
    import traceback as tb
    import uuid

    try:
        league_config = load_league_config(league_id)
        bg_db = Mongo().db
        config_collection = league_config.collections.get('model_config_classifier', 'nba_model_config')

        update_job_progress(job_id, 5, 'Loading base model config...', league=league_config)

        # 1. Load base model config
        base_config = bg_db[config_collection].find_one({'_id': ObjectId(base_model_id)})
        if not base_config:
            fail_job(job_id, 'Base model config not found', league=league_config)
            return

        update_job_progress(job_id, 10, 'Building experiment config...', league=league_config)

        # 2. Build experiment config from the base model's fields
        model_type = base_config.get('model_type', 'LogisticRegression')
        c_value = base_config.get('best_c_value')
        features = base_config.get('features', [])

        experiment_config = {
            'task': 'binary_home_win',
            'model': {
                'type': model_type,
            },
            'features': {
                'features': features,
            },
            'splits': {
                'type': 'year_based_calibration',
                'begin_year': base_config.get('begin_year', 2012),
                'calibration_years': base_config.get('calibration_years', [2023]),
                'evaluation_year': base_config.get('evaluation_year', 2024),
                'min_games_played': base_config.get('min_games_played', 15),
                'exclude_seasons': base_config.get('exclude_seasons'),
            },
            'use_time_calibration': base_config.get('use_time_calibration', True),
            'calibration_method': base_config.get('calibration_method', 'sigmoid'),
        }

        if c_value is not None and model_type in ('LogisticRegression', 'SVM'):
            experiment_config['model']['c_value'] = c_value

        update_job_progress(job_id, 20, 'Running experiment (training base model)...', league=league_config)

        # 3. Run experiment
        runner = ExperimentRunner(db=bg_db, league=league_config)
        session_id = f"retrain-base-{uuid.uuid4().hex[:8]}"
        result = runner.run_experiment(experiment_config, session_id)

        update_job_progress(job_id, 60, 'Linking results to config...', league=league_config)

        # 4. Link results to the same config_id
        config_mgr = ModelConfigManager(db=bg_db, league=league_config)
        config_mgr.link_run_to_config(
            config_id=base_model_id,
            run_id=result['run_id'],
            config_type='classifier',
            metrics=result.get('metrics'),
            artifacts=result.get('artifacts'),
            dataset_id=result.get('dataset_id'),
            f_scores=result.get('diagnostics', {}).get('f_scores'),
            feature_importances=result.get('diagnostics', {}).get('feature_importances'),
            features=features,
            best_c_value=result.get('diagnostics', {}).get('best_c_value', c_value),
            training_stats=result.get('diagnostics', {}).get('training_stats'),
        )

        update_job_progress(job_id, 70, 'Retraining ensemble meta-model...', league=league_config)

        # 5. Retrain ensemble meta-model
        ensemble_doc = bg_db[config_collection].find_one({'_id': ObjectId(ensemble_id)})
        if ensemble_doc:
            service = TrainingService(league=league_config, db=bg_db)
            base_ids = [str(m) for m in ensemble_doc.get('ensemble_models', [])]
            meta_type = ensemble_doc.get('model_type', 'LogisticRegression')
            meta_c = ensemble_doc.get('best_c_value', 0.1)
            stacking_mode = ensemble_doc.get('stacking_mode', 'naive')
            use_disagree = ensemble_doc.get('ensemble_use_disagree', False)
            use_conf = ensemble_doc.get('ensemble_use_conf', False)
            extra_features = ensemble_doc.get('ensemble_meta_features') or None

            service.train_ensemble(
                meta_model_type=meta_type,
                base_model_names_or_ids=base_ids,
                meta_c_value=meta_c if meta_c else 0.1,
                extra_features=extra_features,
                stacking_mode=stacking_mode,
                use_disagree=use_disagree,
                use_conf=use_conf,
            )

        update_job_progress(job_id, 95, 'Finishing up...', league=league_config)
        complete_job(job_id, 'Base model retrained and ensemble meta-model updated', league=league_config)

    except Exception as e:
        tb.print_exc()
        try:
            league_config = load_league_config(league_id)
            fail_job(job_id, str(e), f'Retrain failed: {str(e)}', league=league_config)
        except Exception:
            pass


@app.route('/<league_id>/api/ensembles/<ensemble_id>/update-calibration', methods=['POST'])
@app.route('/api/ensembles/<ensemble_id>/update-calibration', methods=['POST'])
def update_ensemble_calibration(ensemble_id, league_id=None):
    """Recalibrate an ensemble: retrain all base models with new time settings, then create a new ensemble."""
    from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job

    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400

    begin_year = data.get('begin_year')
    calibration_years = data.get('calibration_years')
    evaluation_year = data.get('evaluation_year')
    calibration_method = data.get('calibration_method', 'isotonic')
    exclude_seasons = data.get('exclude_seasons') or None
    min_games_played = data.get('min_games_played')  # None means use base model defaults

    if not begin_year or not calibration_years or not evaluation_year:
        return jsonify({'success': False, 'error': 'begin_year, calibration_years, and evaluation_year are required'}), 400

    league = g.league
    league_id_str = league.league_id if league else 'nba'

    job_id = create_job(
        job_type='recalibrate',
        league=league,
        metadata={'ensemble_id': ensemble_id, 'league': league_id_str}
    )

    def run_recalibrate_background(job_id, ensemble_id, begin_year, calibration_years, evaluation_year, calibration_method, exclude_seasons, min_games_played, league_id):
        from bball_app.core.league_config import load_league_config
        from bball_app.core.services.training_service import TrainingService
        from bball_app.core.mongo import Mongo
        import traceback as tb

        try:
            league_config = load_league_config(league_id)
            bg_db = Mongo().db
            service = TrainingService(league=league_config, db=bg_db)

            result = service.recalibrate_ensemble(
                ensemble_id=ensemble_id,
                begin_year=begin_year,
                calibration_years=calibration_years,
                evaluation_year=evaluation_year,
                calibration_method=calibration_method,
                exclude_seasons=exclude_seasons,
                min_games_played=min_games_played,
                progress_callback=lambda pct, msg: update_job_progress(job_id, pct, msg, league=league_config),
            )

            complete_job(
                job_id,
                f"Recalibration complete — new ensemble {result.get('config_id', '')} ({result.get('time_suffix', '')})",
                league=league_config,
            )
        except Exception as e:
            tb.print_exc()
            try:
                league_config = load_league_config(league_id)
                fail_job(job_id, str(e), f'Recalibration failed: {str(e)}', league=league_config)
            except Exception:
                pass

    thread = threading.Thread(
        target=run_recalibrate_background,
        args=(job_id, ensemble_id, begin_year, calibration_years, evaluation_year, calibration_method, exclude_seasons, min_games_played, league_id_str),
        daemon=True,
    )
    thread.start()

    return jsonify({'success': True, 'job_id': job_id})


@app.route('/<league_id>/api/model-configs/create-ensemble', methods=['POST'])
@app.route('/api/model-configs/create-ensemble', methods=['POST'])
def create_ensemble(league_id=None):
    """Create an ensemble model configuration."""
    try:
        from bson import ObjectId
        import logging
        logger = logging.getLogger(__name__)
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')

        data = request.json
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        ensemble_models = data.get('ensemble_models', [])
        ensemble_type = data.get('ensemble_type', 'stacking')
        ensemble_meta_features = data.get('ensemble_meta_features', [])  # Custom features from user
        ensemble_use_disagree = data.get('ensemble_use_disagree', False)
        ensemble_use_conf = data.get('ensemble_use_conf', False)
        
        if len(ensemble_models) < 2:
            return jsonify({
                'success': False,
                'error': 'At least 2 models are required for ensemble'
            }), 400
        
        # Validate that all ensemble models exist
        base_models = []
        for model_id in ensemble_models:
            model_config = db[classifier_config_collection].find_one({'_id': ObjectId(model_id)})
            if not model_config:
                return jsonify({
                    'success': False,
                    'error': f'Model config {model_id} not found'
                }), 404
            # Skip ensemble configs to avoid nesting
            if model_config.get('ensemble', False):
                return jsonify({
                    'success': False,
                    'error': f'Cannot include ensemble model {model_id} in another ensemble'
                }), 400
            base_models.append(model_config)

        # Rule 1: Validate base model time configs match (comprehensive validation)
        ref_config = base_models[0]
        required_time_keys = ['use_time_calibration', 'begin_year', 'calibration_years', 'evaluation_year']
        
        # Validate reference config has all required time settings
        missing_keys = [key for key in required_time_keys if key not in ref_config]
        if missing_keys:
            return jsonify({
                'success': False,
                'error': f'Reference model missing required time settings: {missing_keys}'
            }), 400
        
        # Validate reference config has time calibration enabled
        if not ref_config.get('use_time_calibration', False):
            return jsonify({
                'success': False,
                'error': 'All base models must have time-based calibration enabled for ensemble creation'
            }), 400
        
        time_config = {
            'use_time_calibration': ref_config['use_time_calibration'],
            'begin_year': ref_config['begin_year'],
            'calibration_years': ref_config['calibration_years'],
            'evaluation_year': ref_config['evaluation_year'],
            'calibration_method': ref_config.get('calibration_method'),
            'exclude_seasons': ref_config.get('exclude_seasons'),
        }
        
        # Validate all base models have identical time configs
        for i, model in enumerate(base_models[1:], 1):
            model_name = model.get('name', f'Model {i+1}')
            
            # Check time calibration is enabled
            if not model.get('use_time_calibration', False):
                return jsonify({
                    'success': False,
                    'error': f'Model "{model_name}" does not have time-based calibration enabled. All base models must have use_time_calibration=True'
                }), 400
            
            # Check each time setting matches
            mismatches = []
            for key in required_time_keys:
                if model.get(key) != time_config[key]:
                    mismatches.append(f'{key}: expected {time_config[key]}, got {model.get(key)}')
            
            if mismatches:
                error_msg = f'Time configuration mismatch in model "{model_name}":\n' + '\n'.join(mismatches)
                return jsonify({
                    'success': False,
                    'error': error_msg + '\n\nAll base models must have identical time-based calibration settings.'
                }), 400
        
        # Rule 2 & 3: Meta config construction with max min_games_played
        # Rule 2: Meta-model time config = base model time config (by default)
        # Rule 3: Meta-model min_games_played = max(min_games_played across all base models)
        min_games_played_values = []
        for model in base_models:
            min_games = model.get('min_games_played', 0)
            if min_games is None:
                min_games = 0
            min_games_played_values.append(min_games)
        
        meta_min_games_played = max(min_games_played_values) if min_games_played_values else 0
        
        # Create ensemble config document
        ensemble_config = {
            'ensemble': True,
            'ensemble_type': ensemble_type,
            'ensemble_models': ensemble_models,
            'ensemble_meta_features': ensemble_meta_features,
            'ensemble_use_disagree': ensemble_use_disagree,
            'ensemble_use_conf': ensemble_use_conf,
            'model_type': data.get('meta_model_type', 'LogisticRegression'),
            'best_c_value': float(data.get('meta_c_value', 0.1)) if data.get('meta_c_value') else None,
            'features': [],  # Ensembles don't have traditional features
            'feature_count': 0,
            'name': f'Ensemble ({len(ensemble_models)} models)',
            'selected': False,  # Don't auto-select ensembles
            'use_master': True,  # Ensembles always use master training data
            # Rule 2: Meta time config = base model time config
            'use_time_calibration': time_config['use_time_calibration'],
            'begin_year': time_config['begin_year'],
            'calibration_years': time_config['calibration_years'],
            'evaluation_year': time_config['evaluation_year'],
            'calibration_method': time_config['calibration_method'],
            'exclude_seasons': time_config['exclude_seasons'],
            # Rule 3: Meta min_games_played = max across base models
            'min_games_played': meta_min_games_played,
            'include_injuries': False,  # Default, can be overridden
            'trained_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'config_hash': f'ensemble_{generate_feature_set_hash(sorted(ensemble_models))}'  # Deterministic hash
        }

        # Check for existing ensemble config first
        # First try to find by config_hash, then fallback to matching ensemble_models
        existing_ensemble = db[classifier_config_collection].find_one({'config_hash': ensemble_config['config_hash']})
        if not existing_ensemble:
            # Fallback: find by ensemble_models (handles old hash format)
            existing_ensemble = db[classifier_config_collection].find_one({
                'ensemble': True,
                'ensemble_models': ensemble_models
            })
            if existing_ensemble:
                # Update the old config's hash to the new format
                db[classifier_config_collection].update_one(
                    {'_id': existing_ensemble['_id']},
                    {'$set': {'config_hash': ensemble_config['config_hash']}}
                )
                logger.info(f"Updated old ensemble config hash to new format: {existing_ensemble['_id']}")
        if existing_ensemble:
            ensemble_id = str(existing_ensemble['_id'])
            # Update existing config with current meta-feature settings
            db[classifier_config_collection].update_one(
                {'_id': existing_ensemble['_id']},
                {'$set': {
                    'updated_at': datetime.utcnow(),
                    'ensemble_meta_features': ensemble_meta_features,
                    'ensemble_use_disagree': ensemble_use_disagree,
                    'ensemble_use_conf': ensemble_use_conf,
                }}
            )
            logger.info(f"Found existing ensemble {ensemble_id} with {len(ensemble_models)} base models")
        else:
            # Insert new ensemble config
            result = db[classifier_config_collection].insert_one(ensemble_config)
            ensemble_id = str(result.inserted_id)
            logger.info(f"Created ensemble {ensemble_id} with {len(ensemble_models)} base models")

        if ensemble_id:
            
            # TODO: Integrate with modeler agent stacking tool
            # For now, ensemble is created but not trained
            # Training will happen when ensemble is selected for prediction
            
            return jsonify({
                'success': True,
                'ensemble_id': ensemble_id,
                'message': f'Ensemble created with {len(ensemble_models)} base models'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to create ensemble'
            }), 500
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =========================================================================
# JOB MANAGEMENT FUNCTIONS
# =========================================================================

@app.route('/<league_id>/api/model-configs/<config_id>/download', methods=['GET'])
@app.route('/api/model-configs/<config_id>/download', methods=['GET'])
def download_model_config_training_csv(config_id, league_id=None):
    """Download the training CSV associated with a model config."""
    try:
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        try:
            oid = ObjectId(config_id)
        except Exception:
            return jsonify({'success': False, 'error': 'Invalid config_id'}), 400
        doc = db[classifier_config_collection].find_one({'_id': oid})
        if not doc:
            return jsonify({'success': False, 'error': 'Config not found'}), 404
        csv_path = doc.get('training_csv')
        if not csv_path:
            return jsonify({'success': False, 'error': 'No training_csv stored for this config'}), 404
        # Normalize to absolute if stored as relative
        if not os.path.isabs(csv_path):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            csv_path = os.path.join(project_root, csv_path)
        # Fallback: if path doesn't exist, try consolidated outputs remap
        if not os.path.exists(csv_path):
            try:
                outputs_root = os.path.join(os.path.dirname(parent_dir), 'model_outputs')
            except Exception:
                outputs_root = 'model_outputs'
            candidate1 = os.path.join(outputs_root, os.path.basename(csv_path))
            old_seg = os.path.join(parent_dir, 'model_output')
            new_seg = outputs_root
            candidate2 = csv_path.replace(old_seg, new_seg) if old_seg in csv_path else None
            new_path = None
            for cand in [candidate1, candidate2]:
                if cand and os.path.exists(cand):
                    new_path = cand
                    break
            if not new_path:
                return jsonify({'success': False, 'error': f'Training CSV not found on disk: {csv_path}'}), 404
            # Persist corrected path
            db[classifier_config_collection].update_one({'_id': doc['_id']}, {'$set': {'training_csv': new_path}})
            csv_path = new_path
        filename = os.path.basename(csv_path)
        return send_file(csv_path, as_attachment=True, download_name=filename)
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500

# Job helper functions — delegate to core/services/jobs.py (league-aware)
from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job


@app.route('/<league_id>/api/jobs/<job_id>', methods=['GET'])
@app.route('/api/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id, league_id=None):
    """Get job status by ID."""
    try:
        # Get league-aware jobs collection
        jobs_collection = g.league.collections.get('jobs', 'jobs_nba')
        job = db[jobs_collection].find_one({'_id': ObjectId(job_id)})

        if not job:
            logger.warning(f"[JOBS API] Job {job_id} not found in collection {jobs_collection}")
            return jsonify({
                'success': False,
                'error': 'Job not found'
            }), 404

        # Log current job status for debugging
        logger.info(f"[JOBS API] get_job_status({job_id}) from {jobs_collection}: status={job.get('status')}, progress={job.get('progress')}%")

        # Convert ObjectId to string
        job['_id'] = str(job['_id'])
        # Convert datetime to ISO string
        if 'created_at' in job and job['created_at']:
            job['created_at'] = job['created_at'].isoformat()
        if 'updated_at' in job and job['updated_at']:
            job['updated_at'] = job['updated_at'].isoformat()

        return jsonify({
            'success': True,
            'job': job
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/jobs/running/<job_type>', methods=['GET'])
@app.route('/api/jobs/running/<job_type>', methods=['GET'])
def get_running_jobs_by_type(job_type, league_id=None):
    """Get the most recent running job of a specific type."""
    try:
        # Get league-aware jobs collection
        jobs_collection = g.league.collections.get('jobs', 'jobs_nba')
        job = db[jobs_collection].find_one(
            {'type': job_type, 'status': 'running'},
            sort=[('created_at', -1)]
        )
        
        if not job:
            return jsonify({
                'success': True,
                'job': None
            })
        
        # Convert ObjectId to string
        job['_id'] = str(job['_id'])
        # Convert datetime to ISO string
        if 'created_at' in job and job['created_at']:
            job['created_at'] = job['created_at'].isoformat()
        if 'updated_at' in job and job['updated_at']:
            job['updated_at'] = job['updated_at'].isoformat()
        
        return jsonify({
            'success': True,
            'job': job
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/<league_id>/api/master-training/resolve-dependencies', methods=['POST'])
@app.route('/api/master-training/resolve-dependencies', methods=['POST'])
def resolve_feature_dependencies(league_id=None):
    """Resolve feature dependencies for regeneration."""
    try:
        data = request.json
        feature_substrings = data.get('feature_substrings', [])
        match_mode = data.get('match_mode', 'OR')  # Default to OR for backward compatibility
        
        # Empty array means regenerate all features
        # Non-empty array means regenerate features matching substrings

        # Load master CSV to get all features
        import pandas as pd
        master_training_path = get_master_training_path()

        if not os.path.exists(master_training_path):
            return jsonify({'success': False, 'error': 'Master training CSV not found'}), 404

        df = pd.read_csv(master_training_path)
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 
                        'home_points', 'away_points', 'pred_home_points', 'pred_away_points', 
                        'pred_margin', 'pred_point_total', 'pred_total']
        all_features = [c for c in df.columns if c not in metadata_cols]
        
        # Find matching features by substring
        from bball_app.core.features.sets import find_features_by_substrings

        if not feature_substrings or len(feature_substrings) == 0:
            # Empty filter = regenerate all features
            requested_features = all_features
        else:
            # Filter by substrings with match mode
            requested_features = find_features_by_substrings(all_features, feature_substrings, match_mode)
            
            if not requested_features:
                return jsonify({
                    'success': False,
                    'error': f'No features found matching substrings: {feature_substrings}'
                }), 404
        
        # Resolve dependencies
        from bball_app.core.features.dependencies import resolve_dependencies, categorize_features
        
        all_to_regenerate, dependency_map = resolve_dependencies(
            requested_features,
            include_transitive=True
        )
        
        # Categorize features
        categorized = categorize_features(requested_features, all_to_regenerate)
        
        # Build response with dependency info
        response_data = {
            'success': True,
            'requested': categorized['requested'],
            'dependencies': categorized['dependencies'],
            'all': categorized['all'],
            'dependency_map': {k: list(v) for k, v in dependency_map.items()},
            'has_extra_dependencies': len(categorized['dependencies']) > 0
        }
        
        return jsonify(response_data)
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/regenerate-features', methods=['POST'])
@app.route('/api/master-training/regenerate-features', methods=['POST'])
def regenerate_master_features(league_id=None):
    """Regenerate specific features in master training CSV."""
    try:
        data = request.json
        feature_substrings = data.get('feature_substrings', [])
        match_mode = data.get('match_mode', 'OR')  # Default to OR for backward compatibility
        confirmed = data.get('confirmed', False)  # User confirmed the dependency list
        
        # feature_substrings can be empty array (means regenerate all)
        
        if not confirmed:
            return jsonify({
                'success': False,
                'error': 'User confirmation required. Call resolve-dependencies first.'
            }), 400

        # Validate CSV exists
        master_training_path = get_master_training_path()
        if not os.path.exists(master_training_path):
            return jsonify({'success': False, 'error': 'Master training CSV not found'}), 404

        # Load CSV to get feature info for job metadata
        import pandas as pd
        df = pd.read_csv(master_training_path)
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 
                        'home_points', 'away_points', 'pred_home_points', 'pred_away_points', 
                        'pred_margin', 'pred_point_total', 'pred_total']
        all_csv_features = [c for c in df.columns if c not in metadata_cols]
        
        # Find matching features
        from bball_app.core.features.sets import find_features_by_substrings

        if not feature_substrings or len(feature_substrings) == 0:
            # Empty filter = regenerate all features
            requested_features = all_csv_features
        else:
            # Filter by substrings with match mode
            requested_features = find_features_by_substrings(all_csv_features, feature_substrings, match_mode)
            
            if not requested_features:
                return jsonify({
                    'success': False,
                    'error': f'No features found matching substrings: {feature_substrings}'
                }), 404
        
        # Resolve dependencies for metadata
        from bball_app.core.features.dependencies import resolve_dependencies, categorize_features
        all_to_regenerate, dependency_map = resolve_dependencies(
            requested_features,
            include_transitive=True
        )
        categorized = categorize_features(requested_features, all_to_regenerate)
        
        # Create job (no config_id for feature regeneration jobs)
        league = g.league
        job_id = create_job('regenerate_features', league=league, metadata={
            'feature_substrings': feature_substrings,
            'requested_features': categorized['requested'],
            'all_features': categorized['all'],
            'dependencies': categorized['dependencies']
        })

        # Spawn subprocess (non-blocking)
        script_path = os.path.join(os.path.dirname(__file__), '..', 'cli', 'populate_master_training_cols.py')
        python_exe = _get_venv_python_executable()

        cmd = [
            python_exe,
            script_path,
            '--feature-substrings', ','.join(feature_substrings),
            '--match-mode', match_mode,
            '--overwrite',
            '--job-id', job_id,
            '--chunk-size', '1000'
        ]

        try:
            _spawn_master_training_job(cmd, job_id=job_id, job_type='regenerate_features', league=league)
        except Exception as e:
            # If subprocess fails to start, mark job as failed
            fail_job(job_id, f"Failed to start regeneration process: {str(e)}", league=league)
            return jsonify({
                'success': False,
                'error': f'Failed to start regeneration process: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Feature regeneration started'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/possible-features', methods=['GET'])
@app.route('/api/master-training/possible-features', methods=['GET'])
def get_possible_features(league_id=None):
    """Get list of all possible features that can be generated."""
    try:
        from bball_app.core.services.training_data import get_all_possible_features
        features = get_all_possible_features(no_player=False)
        return jsonify({
            'success': True,
            'features': features
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/delete-column', methods=['POST'])
@app.route('/api/master-training/delete-column', methods=['POST'])
def delete_master_column(league_id=None):
    """Delete a column from the master training CSV."""
    try:
        data = request.json
        column_name = data.get('column_name')
        
        if not column_name:
            return jsonify({'success': False, 'error': 'column_name required'}), 400

        # Validate CSV exists
        master_training_path = get_master_training_path()
        if not os.path.exists(master_training_path):
            return jsonify({'success': False, 'error': 'Master training CSV not found'}), 404

        # Metadata columns that cannot be deleted
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon']
        if column_name in metadata_cols:
            return jsonify({
                'success': False,
                'error': f'Cannot delete metadata column: {column_name}'
            }), 400

        # Read CSV
        import pandas as pd
        df = pd.read_csv(master_training_path)
        
        # Check if column exists
        if column_name not in df.columns:
            return jsonify({
                'success': False,
                'error': f'Column not found: {column_name}'
            }), 404
        
        # Delete column
        df = df.drop(columns=[column_name])
        
        # Write back to CSV (atomic operation: write to temp, then swap)
        import shutil
        from datetime import datetime
        temp_file_path = f"{master_training_path}.tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        df.to_csv(temp_file_path, index=False)
        shutil.move(temp_file_path, master_training_path)
        
        return jsonify({
            'success': True,
            'message': f'Column {column_name} deleted successfully'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/add-columns', methods=['POST'])
@app.route('/api/master-training/add-columns', methods=['POST'])
def add_master_columns(league_id=None):
    """Add columns to the master training CSV."""
    try:
        data = request.json
        feature_names = data.get('feature_names', [])
        confirmed = data.get('confirmed', False)
        
        if not feature_names:
            return jsonify({'success': False, 'error': 'feature_names required'}), 400
        
        if not confirmed:
            return jsonify({
                'success': False,
                'error': 'User confirmation required'
            }), 400

        # Validate CSV exists
        master_training_path = get_master_training_path()
        if not os.path.exists(master_training_path):
            return jsonify({'success': False, 'error': 'Master training CSV not found'}), 404
        
        # Validate feature names format
        from bball_app.core.features.parser import validate_feature_name
        invalid_features = [f for f in feature_names if not validate_feature_name(f)]
        if invalid_features:
            return jsonify({
                'success': False,
                'error': f'Invalid feature name format: {", ".join(invalid_features[:5])}'
            }), 400
        
        # Get all possible features to validate
        from bball_app.core.services.training_data import get_all_possible_features
        all_possible = set(get_all_possible_features(no_player=False))
        unknown_features = [f for f in feature_names if f not in all_possible]
        if unknown_features:
            return jsonify({
                'success': False,
                'error': f'Unknown features (cannot be generated): {", ".join(unknown_features[:5])}'
            }), 400
        
        # Create job
        league = g.league
        job_id = create_job('add_features', league=league, metadata={
            'feature_names': feature_names
        })

        # Spawn subprocess (non-blocking)
        script_path = os.path.join(os.path.dirname(__file__), '..', 'cli', 'populate_master_training_cols.py')
        python_exe = _get_venv_python_executable()

        cmd = [
            python_exe,
            script_path,
            '--columns', ','.join(feature_names),
            '--overwrite',
            '--job-id', job_id,
            '--chunk-size', '1000'
        ]

        try:
            _spawn_master_training_job(cmd, job_id=job_id, job_type='add_features', league=league)
        except Exception as e:
            fail_job(job_id, f"Failed to start column addition process: {str(e)}", league=league)
            return jsonify({
                'success': False,
                'error': f'Failed to start column addition process: {str(e)}'
            }), 500
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Column addition started'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/available-seasons', methods=['GET'])
@app.route('/api/master-training/available-seasons', methods=['GET'])
def master_training_available_seasons(league_id=None):
    """Get available seasons from MongoDB with game counts and master training status."""
    try:
        from bball_app.core.services.training_data import TrainingDataService

        service = TrainingDataService(league=g.league)
        seasons = service.get_available_seasons()

        return jsonify({
            'success': True,
            'seasons': seasons
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/regenerate-seasons', methods=['POST'])
@app.route('/api/master-training/regenerate-seasons', methods=['POST'])
def master_training_regenerate_seasons(league_id=None):
    """Regenerate specific seasons in master training CSV."""
    try:
        data = request.get_json()
        seasons = data.get('seasons', [])
        no_player = data.get('no_player', False)

        if not seasons:
            return jsonify({
                'success': False,
                'error': 'No seasons provided'
            }), 400

        # Capture league before spawning thread (g.league is request-bound)
        league = g.league

        # Create job for progress tracking
        job_id = create_job('regenerate_seasons', league=league, metadata={
            'seasons': seasons,
            'no_player': no_player
        })

        # Spawn background thread for regeneration
        def run_regeneration():
            try:
                from bball_app.core.services.training_data import TrainingDataService

                def progress_callback(current, total, pct, message):
                    update_job_progress(job_id, pct, message, league=league)

                service = TrainingDataService(league=league)
                games_count, path = service.regenerate_seasons(
                    seasons=seasons,
                    no_player=no_player,
                    progress_callback=progress_callback
                )

                complete_job(job_id, f'Regenerated {games_count} games for {len(seasons)} season(s)', league=league)

            except Exception as e:
                import traceback
                fail_job(job_id, str(e), traceback.format_exc(), league=league)

        import threading
        thread = threading.Thread(target=run_regeneration, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Regenerating {len(seasons)} season(s)'
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@app.route('/<league_id>/api/master-training/regenerate-full', methods=['POST'])
@app.route('/api/master-training/regenerate-full', methods=['POST'])
def master_training_regenerate_full(league_id=None):
    """Regenerate the entire master training CSV from scratch."""
    try:
        data = request.get_json()
        no_player = data.get('no_player', False)

        # Capture league before spawning thread (g.league is request-bound)
        league = g.league

        # Create job for progress tracking
        job_id = create_job('regenerate_full', league=league, metadata={
            'no_player': no_player
        })

        # Spawn background thread for regeneration
        def run_regeneration():
            try:
                from bball_app.core.services.training_data import TrainingDataService

                def progress_callback(current, total, pct, message):
                    update_job_progress(job_id, pct, message, league=league)

                service = TrainingDataService(league=league)
                games_count, path, features = service.regenerate_full(
                    no_player=no_player,
                    progress_callback=progress_callback
                )

                complete_job(job_id, f'Regenerated {games_count} games with {len(features)} features', league=league)

            except Exception as e:
                import traceback
                fail_job(job_id, str(e), traceback.format_exc(), league=league)

        import threading
        thread = threading.Thread(target=run_regeneration, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Starting full regeneration'
        })

    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


def run_training_job(job_id, parsed_config, league):
    """Thin wrapper: delegates training to core TrainingService."""
    from bball_app.core.services.jobs import update_job_progress, complete_job, fail_job

    try:
        update_job_progress(job_id, 1, 'Initializing training...', league=league)

        from bball_app.core.services.training_service import TrainingService

        service = TrainingService(league=league)

        def progress_cb(pct, msg):
            update_job_progress(job_id, pct, msg, league=league)

        result = service.train_model_grid(
            model_types=parsed_config['model_types'],
            c_values=parsed_config['c_values'],
            features=parsed_config['features'],
            use_time_calibration=parsed_config['use_time_calibration'],
            calibration_method=parsed_config['calibration_method'],
            begin_year=parsed_config['begin_year'] or 2012,
            calibration_years=parsed_config['calibration_years'],
            evaluation_year=parsed_config['evaluation_year'],
            min_games_played=parsed_config['min_games_played'],
            include_injuries=parsed_config['include_injuries'],
            exclude_seasons=parsed_config['exclude_seasons'],
            use_master=parsed_config['use_master'],
            progress_callback=progress_cb,
        )

        n_saved = len(result.get('saved_config_ids', []))
        complete_job(job_id, f'Training completed. Saved {n_saved} model configuration(s).', league=league)

    except Exception as e:
        import traceback
        traceback.print_exc()
        from bball_app.core.services.jobs import fail_job
        fail_job(job_id, str(e), f'Training failed: {str(e)}', league=league)


@app.route('/<league_id>/api/model-config/train', methods=['POST'])
@app.route('/api/model-config/train', methods=['POST'])
def train_model_config(league_id=None):
    """
    Start training job asynchronously.
    Thin wrapper: parses request, creates placeholder config via core, spawns job.
    Returns job_id immediately for polling.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Capture league before spawning thread (g is request-scoped)
    league = g.league

    try:
        config = request.json
        logger.info(f"Received training request: {config}")

        # Ensemble training is handled separately
        if config.get('ensemble', False):
            classifier_config_collection = league.collections.get('model_config_classifier', 'nba_model_config')
            return train_ensemble_model(config, classifier_config_collection=classifier_config_collection, league=league)

        parsed = _parse_training_config(config)
        # Override model_types/c_values defaults for training
        if not parsed['model_types']:
            parsed['model_types'] = DEFAULT_MODEL_TYPES
        if not parsed['c_values']:
            parsed['c_values'] = DEFAULT_C_VALUES

        logger.info(f"Model types: {parsed['model_types']}, C-values: {parsed['c_values']}, Features: {len(parsed['features'])}")
        logger.info(f"Time-based calibration: {parsed['use_time_calibration']}, Method: {parsed['calibration_method']}, Begin year: {parsed['begin_year']}, Calibration years: {parsed['calibration_years']}, Eval year: {parsed['evaluation_year']}")

        # Create job
        job_id = create_job('train', league=league)
        logger.info(f"Created job {job_id}")

        # Start training in background thread
        training_thread = threading.Thread(
            target=run_training_job,
            args=(job_id, parsed, league),
            daemon=True
        )
        training_thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Training job started'
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error starting training job: {str(e)}\n{error_trace}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


def train_ensemble_model(config, classifier_config_collection: str = 'nba_model_config', league=None):
    """
    Train ensemble meta-model using stacking trainer.

    Args:
        config: Dictionary containing ensemble training configuration
        classifier_config_collection: League-aware collection name

    Returns:
        JSON response with job_id for async training
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Extract ensemble configuration
        ensemble_models = config.get('ensemble_models', [])
        ensemble_meta_features = config.get('ensemble_meta_features', [])
        ensemble_use_disagree = config.get('ensemble_use_disagree', False)
        ensemble_use_conf = config.get('ensemble_use_conf', False)
        
        # Get meta-model type from selected model types (use first one)
        model_types = config.get('model_types', ['LogisticRegression'])
        meta_model_type = model_types[0] if model_types else 'LogisticRegression'
        
        # Get meta C-value if applicable
        c_values = config.get('c_values', [0.1])
        meta_c_value = c_values[0] if c_values else 0.1
        
        logger.info(f"Training ensemble meta-model: {meta_model_type}")
        logger.info(f"Base models: {ensemble_models}")
        logger.info(f"Meta features: {ensemble_meta_features}")
        logger.info(f"Use disagree: {ensemble_use_disagree}, Use conf: {ensemble_use_conf}")
        
        # Validate ensemble models exist
        from bson import ObjectId
        base_models = []
        for model_id in ensemble_models:
            model_config = db[classifier_config_collection].find_one({'_id': ObjectId(model_id)})
            if not model_config:
                return jsonify({
                    'success': False,
                    'error': f'Model config {model_id} not found'
                }), 404
            base_models.append(model_config)

        # Get time-based calibration config from first base model
        ref_config = base_models[0]

        # Rule 3: min_games_played should be the max across all base models
        # This ensures consistency - if any base model required 20 games, the ensemble should too
        max_min_games = max(
            (m.get('min_games_played', 15) for m in base_models),
            default=15
        )

        time_config = {
            'begin_year': ref_config.get('begin_year'),
            'calibration_years': ref_config.get('calibration_years'),
            'evaluation_year': ref_config.get('evaluation_year'),
            'min_games_played': max_min_games,  # Rule 3: max across base models
            'calibration_method': ref_config.get('calibration_method'),
            'exclude_seasons': ref_config.get('exclude_seasons'),
        }
        
        # Create ensemble config document for training
        ensemble_config = {
            'ensemble': True,
            'ensemble_type': 'stacking',
            'ensemble_models': ensemble_models,
            'ensemble_meta_features': ensemble_meta_features,
            'ensemble_use_disagree': ensemble_use_disagree,
            'ensemble_use_conf': ensemble_use_conf,
            'model_type': meta_model_type,  # Store meta-model type
            'best_c_value': meta_c_value,
            'features': [],  # Ensembles don't have traditional features
            'feature_count': 0,
            'name': f'Ensemble ({len(ensemble_models)} models) - {meta_model_type}',
            'selected': False,
            'use_master': True,
            'use_time_calibration': True,
            'begin_year': time_config['begin_year'],
            'calibration_years': time_config['calibration_years'],
            'evaluation_year': time_config['evaluation_year'],
            'calibration_method': time_config['calibration_method'],
            'exclude_seasons': time_config['exclude_seasons'],
            'min_games_played': time_config['min_games_played'],
            'trained_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
            'config_hash': f'ensemble_{generate_feature_set_hash(sorted(ensemble_models))}_{meta_model_type}'
        }
        
        # Insert or update ensemble config
        # First try to find by config_hash, then fallback to matching ensemble_models + model_type
        existing_ensemble = db[classifier_config_collection].find_one({'config_hash': ensemble_config['config_hash']})
        if not existing_ensemble:
            # Fallback: find by ensemble_models and model_type (handles old hash format)
            existing_ensemble = db[classifier_config_collection].find_one({
                'ensemble': True,
                'ensemble_models': ensemble_models,
                'model_type': meta_model_type
            })
            if existing_ensemble:
                # Update the old config's hash to the new format
                db[classifier_config_collection].update_one(
                    {'_id': existing_ensemble['_id']},
                    {'$set': {'config_hash': ensemble_config['config_hash']}}
                )
                logger.info(f"Updated old ensemble config hash to new format: {existing_ensemble['_id']}")
        if existing_ensemble:
            config_id = str(existing_ensemble['_id'])
            # Update existing config with current meta-feature settings
            db[classifier_config_collection].update_one(
                {'_id': ObjectId(config_id)},
                {'$set': {
                    'updated_at': datetime.utcnow(),
                    'training_in_progress': True,
                    'ensemble_meta_features': ensemble_meta_features,
                    'ensemble_use_disagree': ensemble_use_disagree,
                    'ensemble_use_conf': ensemble_use_conf,
                }}
            )
        else:
            # Insert new config
            result = db[classifier_config_collection].insert_one(ensemble_config)
            config_id = str(result.inserted_id)
        
        # Create job for ensemble training
        job_id = create_job('train', league=league, config_id=config_id)
        logger.info(f"Created ensemble training job {job_id} for config {config_id}")

        # Start ensemble training in background thread
        ensemble_training_thread = threading.Thread(
            target=run_ensemble_training_job,
            args=(
                job_id,
                config_id,
                ensemble_models,
                meta_model_type,
                meta_c_value,
                ensemble_meta_features,
                ensemble_use_disagree,
                ensemble_use_conf,
                time_config,
                classifier_config_collection,
                league
            ),
            daemon=True
        )
        ensemble_training_thread.start()
        
        # Return job_id immediately
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': f'Ensemble training job started for {meta_model_type} meta-model'
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error starting ensemble training job: {str(e)}\n{error_trace}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'traceback': error_trace}), 500


def run_ensemble_training_job(
    job_id: str,
    config_id: str,
    ensemble_models: list,
    meta_model_type: str,
    meta_c_value: float,
    ensemble_meta_features: list,
    ensemble_use_disagree: bool,
    ensemble_use_conf: bool,
    time_config: dict,
    classifier_config_collection: str = 'nba_model_config',
    league=None
):
    """
    Run ensemble training job asynchronously with progress updates.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        update_job_progress(job_id, 1, 'Initializing ensemble training...', league=league)

        # Import required modules
        from bball_app.core.training.stacking_trainer import StackingTrainer
        import uuid

        update_job_progress(job_id, 5, 'Loading base models...', league=league)

        # Create stacking trainer directly (no ModelerAgent needed)
        session_id = str(uuid.uuid4())
        stacking_trainer = StackingTrainer(db=db, league=league)

        # Prepare dataset specification with validated time config (Rules 2 & 3)
        dataset_spec = {
            'begin_year': time_config['begin_year'],
            'calibration_years': time_config['calibration_years'],
            'evaluation_year': time_config['evaluation_year'],
            'min_games_played': time_config.get('min_games_played', 0),  # Rule 3: max across base models
            'use_master': True
        }

        update_job_progress(job_id, 10, 'Training ensemble meta-model...', league=league)

        # Determine stacking mode based on meta features
        use_any_meta = ensemble_use_disagree or ensemble_use_conf or len(ensemble_meta_features) > 0
        stacking_mode = 'informed' if use_any_meta else 'naive'

        logger.info(f"Training ensemble with {len(ensemble_models)} base models")
        logger.info(f"Meta-model type: {meta_model_type}, C-value: {meta_c_value}")
        logger.info(f"Stacking mode: {stacking_mode}")

        # Train stacked model
        result = stacking_trainer.train_stacked_model(
            base_config_ids=ensemble_models,  # Use MongoDB config IDs
            dataset_spec=dataset_spec,
            session_id=session_id,
            meta_c_value=meta_c_value,
            stacking_mode=stacking_mode,
            meta_features=ensemble_meta_features,
            use_disagree=ensemble_use_disagree,
            use_conf=ensemble_use_conf
        )
        
        update_job_progress(job_id, 90, 'Finalizing ensemble model...', league=league)
        
        if result and 'run_id' in result:
            # Update ensemble config with training results
            ensemble_update = {
                'ensemble_run_id': result['run_id'],
                'training_in_progress': False,
                'updated_at': datetime.utcnow()
            }

            diagnostics = result.get('diagnostics') if isinstance(result, dict) else None
            if isinstance(diagnostics, dict):
                meta_feature_importances = diagnostics.get('meta_feature_importances')
                if isinstance(meta_feature_importances, dict) and meta_feature_importances:
                    features_ranked = []
                    for rank, (name, score) in enumerate(meta_feature_importances.items(), 1):
                        features_ranked.append({
                            'rank': rank,
                            'name': name,
                            'score': sanitize_nan(score)
                        })
                    ensemble_update['features_ranked'] = features_ranked
                    ensemble_update['feature_count'] = len(features_ranked)
                    ensemble_update['features'] = [f['name'] for f in features_ranked]

                base_models_summary = diagnostics.get('base_models_summary')
                if isinstance(base_models_summary, list) and base_models_summary:
                    ensemble_update['ensemble_base_models_summary'] = base_models_summary

            # Add metrics if available
            if 'metrics' in result:
                metrics = result['metrics']
                # Stacking trainer uses *_mean/*_std naming convention
                ensemble_update.update({
                    'accuracy': sanitize_nan(metrics.get('accuracy_mean') or metrics.get('accuracy')),
                    'log_loss': sanitize_nan(metrics.get('log_loss_mean') or metrics.get('log_loss')),
                    'brier_score': sanitize_nan(metrics.get('brier_mean') or metrics.get('brier_score')),
                    'std_dev': sanitize_nan(metrics.get('accuracy_std') or metrics.get('std_dev'))
                })

            # Add artifact paths and meta-model config if available
            if 'artifacts' in result:
                artifacts = result['artifacts']
                if artifacts.get('meta_model_path'):
                    ensemble_update['meta_model_path'] = artifacts['meta_model_path']
                if artifacts.get('meta_scaler_path'):
                    ensemble_update['meta_scaler_path'] = artifacts['meta_scaler_path']
                if artifacts.get('ensemble_config_path'):
                    ensemble_update['ensemble_config_path'] = artifacts['ensemble_config_path']
                if artifacts.get('meta_model_type'):
                    ensemble_update['meta_model_type'] = artifacts['meta_model_type']
                if artifacts.get('meta_c_value') is not None:
                    ensemble_update['meta_c_value'] = artifacts['meta_c_value']
                    # Also set best_c_value for UI compatibility
                    ensemble_update['best_c_value'] = artifacts['meta_c_value']

            db[classifier_config_collection].update_one(
                {'_id': ObjectId(config_id)},
                {'$set': ensemble_update}
            )

            complete_job(job_id, f'Ensemble training completed successfully', league=league)
            logger.info(f"Ensemble training completed: {result['run_id']}")
            
        else:
            error_msg = f"Ensemble training failed: {result}"
            fail_job(job_id, error_msg, league=league)
            logger.error(error_msg)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = f'Ensemble training failed: {str(e)}'
        fail_job(job_id, error_msg, league=league)
        logger.error(f"{error_msg}\n{error_trace}")
        traceback.print_exc()


@app.route('/<league_id>/api/points-model/train', methods=['POST'])
@app.route('/api/points-model/train', methods=['POST'])
def train_points_model(league_id=None):
    """Train a points regression model."""
    try:
        from bball_app.core.models import PointsRegressionTrainer
        from bball_app.core.features.sets import get_features_by_sets
        points_config_collection = g.league.collections.get('model_config_points', 'model_config_points_nba')
        
        config = request.json
        model_type = config.get('model_type', 'Ridge')
        # Handle alpha/alphas - support both single value and list
        if 'alpha' in config and config['alpha'] is not None:
            # Single alpha value provided (e.g., from agent workflow)
            alphas = [float(config['alpha'])]
        elif 'alphas' in config and config['alphas']:
            # List of alphas provided (e.g., from UI form)
            alphas = [float(a) for a in config['alphas'] if a is not None]
        else:
            # Default alphas if neither provided
            alphas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        
        # Ensure alphas is not empty
        if not alphas:
            alphas = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        
        # Get calibration settings
        print(f"[train_points_model] Raw config values from request:")
        print(f"  config.get('use_time_calibration'): {config.get('use_time_calibration')}")
        print(f"  config.get('calibration_years'): {config.get('calibration_years')}")
        print(f"  config.get('evaluation_year'): {config.get('evaluation_year')}")
        print(f"  config.get('begin_year'): {config.get('begin_year')}")
        
        use_time_calibration_raw = config.get('use_time_calibration', False)
        if isinstance(use_time_calibration_raw, str):
            use_time_calibration = use_time_calibration_raw.lower() in ('true', '1', 'yes')
        else:
            use_time_calibration = bool(use_time_calibration_raw)
        print(f"[train_points_model] Parsed use_time_calibration: {use_time_calibration} (from raw: {use_time_calibration_raw})")
        
        # Extract year fields and normalize them first
        calibration_years_raw = config.get('calibration_years')
        evaluation_year_raw = config.get('evaluation_year')
        begin_year_raw = config.get('begin_year')
        
        # Normalize year fields (do this before auto-enable check)
        begin_year = None
        calibration_years = None
        evaluation_year = None
        
        # Normalize begin_year
        if begin_year_raw is not None:
            try:
                begin_year = int(begin_year_raw) if isinstance(begin_year_raw, str) else begin_year_raw
            except (ValueError, TypeError):
                begin_year = None
                print(f"[train_points_model] WARNING: Failed to parse begin_year: {begin_year_raw}")
        
        # Normalize calibration_years
        if calibration_years_raw is not None:
            if isinstance(calibration_years_raw, list):
                try:
                    calibration_years = [int(y) for y in calibration_years_raw if y is not None]
                except (ValueError, TypeError) as e:
                    calibration_years = None
                    print(f"[train_points_model] WARNING: Failed to parse calibration_years list: {calibration_years_raw}, error: {e}")
            elif isinstance(calibration_years_raw, str):
                # Handle comma-separated string
                try:
                    calibration_years = [int(y.strip()) for y in calibration_years_raw.split(',') if y.strip()]
                except (ValueError, TypeError) as e:
                    calibration_years = None
                    print(f"[train_points_model] WARNING: Failed to parse calibration_years string: {calibration_years_raw}, error: {e}")
            else:
                # Single integer
                try:
                    calibration_years = [int(calibration_years_raw)]
                except (ValueError, TypeError):
                    calibration_years = None
                    print(f"[train_points_model] WARNING: Failed to parse calibration_years as integer: {calibration_years_raw}")
        
        # Normalize evaluation_year
        if evaluation_year_raw is not None:
            try:
                evaluation_year = int(evaluation_year_raw) if isinstance(evaluation_year_raw, str) else evaluation_year_raw
            except (ValueError, TypeError) as e:
                evaluation_year = None
                print(f"[train_points_model] WARNING: Failed to parse evaluation_year: {evaluation_year_raw}, error: {e}")
        
        # Safety check: If normalized calibration_years and evaluation_year are provided, auto-enable time-based calibration
        # This handles cases where checkbox might not be checked but values are present
        if not use_time_calibration and (calibration_years is not None and evaluation_year is not None):
            print(f"[train_points_model] WARNING: Time-based calibration checkbox was unchecked but valid year fields are present. Auto-enabling time-based calibration.")
            use_time_calibration = True
        
        calibration_method = config.get('calibration_method', 'isotonic') if use_time_calibration else None
        
        print(f"[train_points_model] After normalization and auto-enable check:")
        print(f"  use_time_calibration: {use_time_calibration}")
        print(f"  begin_year: {begin_year}")
        print(f"  calibration_years: {calibration_years}")
        print(f"  evaluation_year: {evaluation_year}")
        
        # Only use normalized values if calibration is enabled
        if not use_time_calibration:
            print(f"[train_points_model] Time-based calibration disabled, clearing year fields")
            begin_year = None
            calibration_years = None
            evaluation_year = None
        
        # Validation: Ensure required fields are present if calibration is enabled
        if use_time_calibration and (calibration_years is None or evaluation_year is None):
            print(f"[train_points_model] ERROR: Time-based calibration is enabled but required fields are missing: calibration_years={calibration_years}, evaluation_year={evaluation_year}")
            print(f"[train_points_model] ERROR: Disabling time-based calibration and falling back to TimeSeriesSplit CV")
            use_time_calibration = False
            begin_year = None
            calibration_years = None
            evaluation_year = None
            calibration_method = None
        
        # Get selected features
        features = config.get('features', [])
        feature_sets = config.get('feature_sets', [])
        
        # If feature sets are specified, get features from sets
        if feature_sets and not features:
            features = get_features_by_sets(feature_sets)
        # If both are specified, combine (features take precedence)
        elif feature_sets and features:
            set_features = get_features_by_sets(feature_sets)
            # Merge, keeping individual features
            all_features = list(set(set_features + features))
            features = all_features
        
        # Use master training data if available (same master CSV as classifiers)
        from bball_app.core.services.training_data import (
            extract_features_from_master_for_points,
            check_master_needs_regeneration,
            get_master_training_metadata,
            generate_master_training_data,
        )
        import logging
        logger = logging.getLogger(__name__)
        master_training_path = get_master_training_path()

        training_csv = None
        use_master = True  # Always try to use master if available

        if use_master and os.path.exists(master_training_path):
            # Check if master needs regeneration
            league_config = getattr(g, 'league', None)
            needs_regeneration, missing_features = check_master_needs_regeneration(db, features if features else [], league=league_config)

            if needs_regeneration and missing_features:
                # Master missing features - error out instead of regenerating
                raise ValueError(
                    f"Master training CSV is missing required features: {missing_features[:10]}"
                    f"{'... and ' + str(len(missing_features) - 10) + ' more' if len(missing_features) > 10 else ''}. "
                    f"Add them via generate_training.sh --add --features or the master-training UI."
                )

            # Extract requested features from master (for points regression, includes home_points/away_points)
            import tempfile
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_csv = os.path.join(tempfile.gettempdir(), f'extracted_points_training_{timestamp}.csv')

            # Use default min_games_played=15 to match agent workflow
            min_games_played = 15
            training_csv = extract_features_from_master_for_points(
                master_training_path,
                features if features else None,
                output_path=temp_csv,
                begin_year=begin_year,
                min_games_played=min_games_played
            )
            logger.info(f"Extracted {len(features) if features else 'all'} features from master training data for points regression (begin_year={begin_year}, min_games_played={min_games_played})")
        elif use_master:
            # Master doesn't exist - generate it
            logger.info("Master training data does not exist. Generating...")
            master_path, master_features, master_game_count = generate_master_training_data()
            logger.info(f"Generated master training data: {master_game_count} games, {len(master_features)} features")

            # Extract features
            import tempfile
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            temp_csv = os.path.join(tempfile.gettempdir(), f'extracted_points_training_{timestamp}.csv')

            # Use default min_games_played=15 to match agent workflow (same as dataset_builder default)
            min_games_played = 15
            training_csv = extract_features_from_master_for_points(
                master_training_path,
                features if features else None,
                output_path=temp_csv,
                begin_year=begin_year,
                min_games_played=min_games_played
            )
        
        # Create trainer (injury features will be auto-detected from selected features)
        trainer = PointsRegressionTrainer(db=db)
        
        # Prepare model kwargs
        model_kwargs = {}
        if model_type == 'ElasticNet':
            model_kwargs['alpha'] = config.get('elasticnet_alpha', 1.0)
            model_kwargs['l1_ratio'] = config.get('elasticnet_l1_ratio', 0.5)
        elif model_type == 'RandomForest':
            model_kwargs['n_estimators'] = config.get('rf_n_estimators', 100)
            max_depth = config.get('rf_max_depth')
            if max_depth:
                model_kwargs['max_depth'] = int(max_depth)
        elif model_type == 'XGBoost':
            model_kwargs['n_estimators'] = config.get('xgb_n_estimators', 100)
            model_kwargs['max_depth'] = config.get('xgb_max_depth', 6)
            model_kwargs['learning_rate'] = config.get('xgb_learning_rate', 0.1)
        
        # Log calibration settings (using print for visibility)
        print(f"[train_points_model] Training configuration:")
        print(f"  use_time_calibration: {use_time_calibration} (type: {type(use_time_calibration)})")
        print(f"  calibration_method: {calibration_method}")
        print(f"  begin_year: {begin_year} (type: {type(begin_year)})")
        print(f"  calibration_years: {calibration_years} (type: {type(calibration_years)})")
        print(f"  evaluation_year: {evaluation_year} (type: {type(evaluation_year)})")
        print(f"  model_type: {model_type}, alphas: {alphas}")
        
        if use_time_calibration:
            print(f"[train_points_model] Time-based calibration settings: method={calibration_method}, begin_year={begin_year}, calibration_years={calibration_years}, evaluation_year={evaluation_year}")
            print(f"[train_points_model] Calibration splits: Train (SeasonStartYear < {min(calibration_years) if calibration_years else 'N/A'}), Calibrate (SeasonStartYear in {calibration_years}), Evaluate (SeasonStartYear == {evaluation_year})")
        else:
            print(f"[train_points_model] WARNING: Time-based calibration is DISABLED. Will use TimeSeriesSplit CV and evaluate on full dataset.")
        
        # Train model with selected features (using master CSV if available)
        # Pass calibration parameters to trainer.train()
        if model_type == 'Ridge':
            training_results = trainer.train(
                model_type=model_type, 
                alphas=alphas, 
                selected_features=features if features else None,
                training_csv=training_csv,
                use_time_calibration=use_time_calibration,
                calibration_years=calibration_years,
                evaluation_year=evaluation_year,
                begin_year=begin_year
            )
        else:
            training_results = trainer.train(
                model_type=model_type, 
                selected_features=features if features else None,
                training_csv=training_csv,
                use_time_calibration=use_time_calibration,
                calibration_years=calibration_years,
                evaluation_year=evaluation_year,
                begin_year=begin_year,
                **model_kwargs
            )
        
        # Save model
        model_path = trainer.save_model()
        
        # Generate diagnostics
        report_path = trainer.generate_diagnostics(training_results)
        
        # Save config to MongoDB
        # generate_feature_set_hash is aliased from ModelConfigManager (line ~659)
        import hashlib
        
        # Use all features if none selected
        if not features:
            # Get features from training results (set during create_training_data)
            features = training_results.get('feature_names', [])
            if not features and hasattr(trainer, 'feature_names') and trainer.feature_names:
                features = trainer.feature_names
        
        # Generate feature set hash
        feature_set_hash = generate_feature_set_hash(features) if features else ''
        
        # Get best metrics (for Ridge, use best alpha; for others, use CV results)
        final_metrics = training_results['final_metrics']
        best_alpha = None
        alphas_tested = None
        if model_type == 'Ridge' and training_results.get('cv_results'):
            # Find best alpha from CV results
            best_cv = min(training_results['cv_results'], 
                         key=lambda x: (x['home_mae'] + x['away_mae']) / 2)
            best_alpha = best_cv.get('alpha')
            # Extract all alphas tested
            alphas_tested = [r.get('alpha') for r in training_results['cv_results'] if r.get('alpha') is not None]
        elif model_type == 'Ridge' and training_results.get('selected_alpha'):
            best_alpha = training_results.get('selected_alpha')
            alphas_tested = training_results.get('alphas_tested')
        
        # Generate config hash (include model type + features + model parameters + calibration settings)
        hash_parts = [model_type, feature_set_hash]
        
        if model_type == 'Ridge' and best_alpha is not None:
            hash_parts.append(f"alpha={best_alpha}")
        elif model_kwargs:
            params_str = '|'.join([f"{k}={v}" for k, v in sorted(model_kwargs.items())])
            hash_parts.append(params_str)
        
        # Include calibration settings in hash
        if use_time_calibration:
            hash_parts.append(f"cal=1")
            if calibration_method:
                hash_parts.append(f"cal_method={calibration_method}")
            if begin_year is not None:
                hash_parts.append(f"begin_year={begin_year}")
            if calibration_years:
                cal_years_str = ','.join(sorted([str(y) for y in calibration_years]))
                hash_parts.append(f"cal_years={cal_years_str}")
            if evaluation_year is not None:
                hash_parts.append(f"eval_year={evaluation_year}")
        else:
            hash_parts.append(f"cal=0")
        
        config_hash_str = '|'.join(hash_parts)
        config_hash = hashlib.md5(config_hash_str.encode()).hexdigest()
        
        # Prepare config document
        config_doc = {
            'config_hash': config_hash,
            'model_type': model_type,
            'feature_set_hash': feature_set_hash,
            'features': sorted(features) if features else [],
            'feature_count': len(features) if features else training_results['n_features'],
            'model_path': model_path,
            'report_path': report_path,
            'metrics': {
                'home_mae': sanitize_nan(final_metrics['home_mae']),
                'home_rmse': sanitize_nan(final_metrics['home_rmse']),
                'home_r2': sanitize_nan(final_metrics['home_r2']),
                'home_mape': sanitize_nan(final_metrics['home_mape']),
                'away_mae': sanitize_nan(final_metrics['away_mae']),
                'away_rmse': sanitize_nan(final_metrics['away_rmse']),
                'away_r2': sanitize_nan(final_metrics['away_r2']),
                'away_mape': sanitize_nan(final_metrics['away_mape']),
                'total_mae': sanitize_nan(final_metrics['total_mae']),
                'total_rmse': sanitize_nan(final_metrics['total_rmse']),
                'total_r2': sanitize_nan(final_metrics['total_r2']),
                'diff_mae': sanitize_nan(final_metrics['diff_mae']),
                'diff_rmse': sanitize_nan(final_metrics['diff_rmse']),
                'diff_r2': sanitize_nan(final_metrics['diff_r2']),
            },
            'training_stats': {
                'n_samples': training_results['n_samples'],
                'n_features': training_results['n_features'],
            },
            'model_params': model_kwargs,
            'use_time_calibration': use_time_calibration,
            'calibration_method': calibration_method if use_time_calibration else None,
            'begin_year': begin_year if use_time_calibration else None,
            'calibration_years': calibration_years if use_time_calibration else None,
            'evaluation_year': evaluation_year if use_time_calibration else None,
            'updated_at': datetime.utcnow(),
        }
        
        if best_alpha is not None:
            config_doc['best_alpha'] = best_alpha
        if alphas_tested is not None:
            config_doc['alphas_tested'] = alphas_tested
        
        # Check if config exists
        existing = db[points_config_collection].find_one({'config_hash': config_hash})
        
        if existing and existing.get('name'):
            # Preserve custom name
            existing_name = existing['name']
            auto_name_prefix = f"{model_type} - {feature_set_hash[:8] if feature_set_hash else 'default'}"
            if not existing_name.startswith(auto_name_prefix):
                config_doc['name'] = existing_name
            else:
                config_doc['name'] = f"{model_type} - {feature_set_hash[:8] if feature_set_hash else 'default'}"
        else:
            config_doc['name'] = f"{model_type} - {feature_set_hash[:8] if feature_set_hash else 'default'}"
        
        # Set selected flag (first config becomes selected)
        if not existing:
            existing_selected = db[points_config_collection].find_one({'selected': True})
            config_doc['selected'] = (existing_selected is None)
            config_doc['trained_at'] = datetime.utcnow()
        else:
            config_doc['selected'] = existing.get('selected', False)
        
        # Upsert config
        result = db[points_config_collection].update_one(
            {'config_hash': config_hash},
            {'$set': config_doc},
            upsert=True
        )

        # Get document ID
        if result.upserted_id:
            config_id = str(result.upserted_id)
        else:
            doc = db[points_config_collection].find_one({'config_hash': config_hash})
            config_id = str(doc['_id'])
        
        # Sanitize training_results to remove non-serializable model objects
        sanitized_results = sanitize_training_results(training_results)
        
        return jsonify({
            'success': True,
            'results': sanitized_results,
            'model_path': model_path,
            'report_path': report_path,
            'config_id': config_id
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/<league_id>/api/points-model/predict', methods=['POST'])
@app.route('/api/points-model/predict', methods=['POST'])
def predict_points(league_id=None):
    """Predict points for a game using the trained points regression model."""
    try:
        from bball_app.core.models import PointsRegressionTrainer
        import glob
        
        data = request.json
        game_id = data.get('game_id')
        game_date_str = data.get('game_date')
        model_name = data.get('model_name')  # Optional: specify model to use
        
        if not game_id or not game_date_str:
            return jsonify({'error': 'game_id and game_date are required'}), 400

        # Get league-aware games collection
        games_collection = g.league.collections.get('games', 'stats_nba')

        # Get game from database
        game = db[games_collection].find_one({'game_id': game_id})
        if not game:
            return jsonify({'error': 'Game not found'}), 404
        
        # Parse date
        game_date = datetime.strptime(game_date_str, '%Y-%m-%d').date()
        before_date = game_date_str
        
        # Load trainer and model (injury features will be auto-detected from model's feature list)
        trainer = PointsRegressionTrainer(db=db)
        
        # Load the latest model or specified model
        if model_name:
            trainer.load_model(model_name)
        else:
            # Find latest model
            artifacts_dir = trainer.artifacts_dir
            model_files = glob.glob(os.path.join(artifacts_dir, 'points_regression_*.pkl'))
            if not model_files:
                return jsonify({'error': 'No trained model found. Please train a model first.'}), 404
            
            # Get latest by modification time
            latest_model = max(model_files, key=os.path.getmtime)
            model_name = os.path.basename(latest_model).replace('.pkl', '')
            trainer.load_model(model_name)
        
        # Make prediction
        prediction = trainer.predict(game, before_date)
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'model_name': model_name
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/<league_id>/api/model-config/predict', methods=['POST'])
@app.route('/api/model-config/predict', methods=['POST'])
def predict_model_config(league_id=None):
    """Make predictions with configuration."""
    try:
        config = request.json
        # This would call the prediction logic
        # For now, return a placeholder response
        # TODO: Integrate with actual prediction logic
        return jsonify({
            'success': True,
            'results': [{
                'model_type': config.get('model_types', [])[0] if config.get('model_types') else 'Unknown',
                'c_value': config.get('c_values', [0.1])[0] if config.get('c_values') else None,
                'date': date.today().strftime('%Y-%m-%d'),
                'predictions': 'Placeholder predictions'
            }]
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# Matchup chat agent session storage (in-memory, per-process)
# In production, use Redis or database-backed sessions
_matchup_agent_sessions = {}
_session_lock = threading.Lock()

# Track system message file modification time to detect changes
_matchup_system_message_mtime = None


# =============================================================================
# MATCHUP CHAT API ENDPOINTS
# =============================================================================

@app.route('/<league_id>/api/matchup-chat/sessions', methods=['POST'])
@app.route('/api/matchup-chat/sessions', methods=['POST'])
def create_matchup_chat_session(league_id=None):
    """Create or get existing matchup chat session for a game"""
    try:
        data = request.json
        game_id = data.get('game_id')
        home_team = data.get('home_team')
        away_team = data.get('away_team')
        game_date = data.get('game_date')
        
        if not game_id:
            return jsonify({'error': 'game_id is required'}), 400

        # Get league-aware collection names
        games_collection = g.league.collections.get('games', 'stats_nba')
        sessions_collection = g.league.collections.get('matchup_sessions', 'nba_matchup_sessions')

        # Check if a session already exists for this game_id
        existing_session = db[sessions_collection].find_one(
            {'game_id': game_id},
            sort=[('updated_at', -1)]  # Get the most recently updated session
        )

        if existing_session:
            # Return existing session
            session_id = str(existing_session['_id'])
            session_name = existing_session.get('name', f"Matchup Chat - {game_id}")

            return jsonify({
                'success': True,
                'session_id': session_id,
                'name': session_name,
                'game_id': game_id,
                'existing': True
            })

        # No existing session found, create a new one
        # Get game info
        game = db[games_collection].find_one({'game_id': game_id})
        if not game:
            return jsonify({'error': f'Game {game_id} not found'}), 404
        
        # Extract team info
        if not home_team:
            home_team = game.get('homeTeam', {}).get('name', '')
        if not away_team:
            away_team = game.get('awayTeam', {}).get('name', '')
        
        if not game_date:
            game_date = game.get('date', '')
        
        if not home_team or not away_team:
            return jsonify({'error': 'Could not determine teams for game'}), 400
        
        # Create session name
        session_name = f"{away_team} @ {home_team}"
        if game_date:
            try:
                game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
                session_name += f" - {game_date}"
            except:
                pass
        
        # Get selected model config ID
        classifier_config_collection = g.league.collections.get('model_config_classifier', 'nba_model_config')
        selected_model_config = db[classifier_config_collection].find_one({'selected': True})
        model_config_id = str(selected_model_config['_id']) if selected_model_config else None
        
        # Get season
        if game_date:
            try:
                game_date_obj = datetime.strptime(game_date, '%Y-%m-%d').date()
                season = get_season_from_date(game_date_obj)
            except:
                season = None
        else:
            season = None
        
        new_session = {
            'game_id': game_id,
            'name': session_name,
            'messages': [],
            'context': {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date,
                'season': season,
                'model_config_id': model_config_id
            },
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow()
        }
        
        result = db.nba_matchup_sessions.insert_one(new_session)
        session_id = str(result.inserted_id)
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'name': session_name,
            'game_id': game_id,
            'existing': False
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat', methods=['POST'])
@app.route('/api/matchup-chat', methods=['POST'])
def matchup_chat_api(league_id=None):
    """Matchup chat API endpoint"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        session_id = data.get('session_id')
        memory = data.get('memory')  # Can be None (all messages) or integer
        
        logger.info(f"[MATCHUP_CHAT_API] Received message for session {session_id}: {message[:100]}...")
        
        if not message:
            logger.warning("[MATCHUP_CHAT_API] Empty message received")
            return jsonify({'error': 'Message is required'}), 400
        
        if not session_id:
            logger.warning("[MATCHUP_CHAT_API] Missing session_id")
            return jsonify({'error': 'session_id is required'}), 400
        
        # Verify session exists
        try:
            session_obj_id = ObjectId(session_id)
        except:
            logger.error(f"[MATCHUP_CHAT_API] Invalid session_id format: {session_id}")
            return jsonify({'error': 'Invalid session_id'}), 400

        # Get league-aware sessions collection
        league_config = getattr(g, 'league', None)
        sessions_collection = league_config.collections.get('matchup_sessions', 'nba_matchup_sessions') if league_config else 'nba_matchup_sessions'

        session = db[sessions_collection].find_one({'_id': session_obj_id})
        if not session:
            logger.error(f"[MATCHUP_CHAT_API] Session not found: {session_id}")
            return jsonify({'error': 'Session not found'}), 404

        game_id = session.get('game_id')
        if not game_id:
            logger.error(f"[MATCHUP_CHAT_API] Session {session_id} missing game_id")
            return jsonify({'error': 'Session missing game_id'}), 400

        logger.info(f"[MATCHUP_CHAT_API] Processing for game_id: {game_id}")

        # Save user message to MongoDB
        db[sessions_collection].update_one(
            {'_id': session_obj_id},
            {
                '$push': {
                    'messages': {
                        'role': 'user',
                        'content': message,
                        'timestamp': datetime.utcnow()
                    }
                },
                '$set': {'updated_at': datetime.utcnow()}
            }
        )

        # Get league config (already retrieved earlier)
        league_id = league_config.league_id if league_config else 'nba'

        # Get show_agent_actions from request
        show_agent_actions = data.get('show_agent_actions', False)

        # Load conversation history from MongoDB (as simple dicts - Controller handles format)
        session_doc = db[sessions_collection].find_one({'_id': session_obj_id})
        conversation_history = []
        if session_doc and 'messages' in session_doc:
            # Filter out tool outputs
            for msg in session_doc['messages']:
                if msg.get('role') == 'tool':
                    continue
                if msg['role'] in ['user', 'assistant']:
                    conversation_history.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

            logger.debug(f"[MATCHUP_CHAT_API] Loaded {len(conversation_history)} messages from history")

            # Apply memory window if specified
            if memory is not None:
                try:
                    memory_int = int(memory)
                    if memory_int > 0:
                        conversation_history = conversation_history[-memory_int:] if len(conversation_history) > memory_int else conversation_history
                        logger.debug(f"[MATCHUP_CHAT_API] Applied memory window: {memory_int} messages")
                except (ValueError, TypeError):
                    pass  # Invalid memory value, use all messages

        logger.info(f"[MATCHUP_CHAT_API] Using multi-agent Controller with {len(conversation_history)} history messages")

        # Create Controller and process message
        try:
            controller = MatchupChatController(db=db, league=league_config, league_id=league_id)
            options = ControllerOptions(show_agent_actions=show_agent_actions)

            result = controller.handle_user_message(
                game_id=game_id,
                user_message=message,
                conversation_history=conversation_history,
                options=options,
            )

            response_text = result.get('response', '')
            agent_actions = result.get('agent_actions', [])
            turn_plan = result.get('turn_plan', {})

            logger.info(f"[MATCHUP_CHAT_API] Controller returned response ({len(response_text)} chars) with {len(agent_actions)} agent action(s)")

            # Save assistant response to MongoDB
            db[sessions_collection].update_one(
                {'_id': session_obj_id},
                {
                    '$push': {
                        'messages': {
                            'role': 'assistant',
                            'content': response_text,
                            'timestamp': datetime.utcnow(),
                            'agent_actions': agent_actions,
                            'turn_plan': turn_plan
                        }
                    },
                    '$set': {'updated_at': datetime.utcnow()}
                }
            )

            return jsonify({
                'success': True,
                'response': response_text,
                'agent_actions': agent_actions,
                'turn_plan': turn_plan,
                'session_id': session_id
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"[MATCHUP_CHAT_API] Controller error: {str(e)}", exc_info=True)
            print(f"[MATCHUP CHAT API ERROR] Controller error: {str(e)}")
            print(f"[MATCHUP CHAT API ERROR] Traceback:\n{error_trace}")
            return jsonify({
                'error': f'Agent error: {str(e)}',
                'session_id': session_id,
                'traceback': error_trace if app.debug else None
            }), 500
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat/sessions', methods=['GET'])
@app.route('/api/matchup-chat/sessions', methods=['GET'])
def get_matchup_chat_sessions(league_id=None):
    """Get list of all matchup chat sessions"""
    try:
        game_id = request.args.get('game_id')  # Optional filter
        
        query = {}
        if game_id:
            query['game_id'] = game_id
        
        sessions = list(db.nba_matchup_sessions.find(
            query,
            {'name': 1, 'game_id': 1, 'created_at': 1, 'updated_at': 1, 'messages': {'$slice': 1}},
            sort=[('updated_at', -1)]
        ))
        
        # Convert ObjectId to string and format dates
        for session in sessions:
            session['_id'] = str(session['_id'])
            session['created_at'] = session['created_at'].isoformat() if session.get('created_at') else None
            session['updated_at'] = session['updated_at'].isoformat() if session.get('updated_at') else None
            # Get message count
            full_session = db.nba_matchup_sessions.find_one({'_id': ObjectId(session['_id'])})
            session['message_count'] = len(full_session.get('messages', [])) if full_session else 0
        
        return jsonify({'success': True, 'sessions': sessions})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat/sessions/<session_id>', methods=['GET'])
@app.route('/api/matchup-chat/sessions/<session_id>', methods=['GET'])
def get_matchup_chat_session(session_id, league_id=None):
    """Get a specific matchup chat session with all messages"""
    try:
        try:
            session_obj_id = ObjectId(session_id)
        except:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        session = db.nba_matchup_sessions.find_one({'_id': session_obj_id})
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Convert ObjectId to string and format dates
        session['_id'] = str(session['_id'])
        session['created_at'] = session['created_at'].isoformat() if session.get('created_at') else None
        session['updated_at'] = session['updated_at'].isoformat() if session.get('updated_at') else None
        
        # Format message timestamps
        for msg in session.get('messages', []):
            if 'timestamp' in msg:
                msg['timestamp'] = msg['timestamp'].isoformat()
        
        return jsonify({'success': True, 'session': session})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat/sessions/<session_id>/messages', methods=['DELETE'])
@app.route('/api/matchup-chat/sessions/<session_id>/messages', methods=['DELETE'])
def delete_matchup_chat_message(session_id, league_id=None):
    """Delete a message from a matchup chat session"""
    try:
        try:
            session_obj_id = ObjectId(session_id)
        except:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        data = request.json
        message_index = data.get('message_index')
        message_content = data.get('message_content')  # For matching when index not available
        message_role = data.get('message_role')
        
        # Get session
        session = db.nba_matchup_sessions.find_one({'_id': session_obj_id})
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        messages = session.get('messages', [])
        
        deleted = False
        
        if message_index is not None and 0 <= message_index < len(messages):
            # Delete by index (preferred method)
            messages.pop(message_index)
            deleted = True
        elif message_content:
            # Try to find and delete by matching content (for newly sent messages without index)
            # Match by exact content and role
            for i, msg in enumerate(messages):
                if msg.get('content') == message_content:
                    if message_role is None or msg.get('role') == message_role:
                        messages.pop(i)
                        deleted = True
                        break
        
        if not deleted:
            return jsonify({'error': 'Message not found. Provide message_index or message_content.'}), 400
        
        # Update session
        db.nba_matchup_sessions.update_one(
            {'_id': session_obj_id},
            {
                '$set': {
                    'messages': messages,
                    'updated_at': datetime.utcnow()
                }
            }
        )
        
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat/sessions/<session_id>', methods=['DELETE'])
@app.route('/api/matchup-chat/sessions/<session_id>', methods=['DELETE'])
def delete_matchup_chat_session(session_id, league_id=None):
    """Delete a matchup chat session"""
    try:
        try:
            session_obj_id = ObjectId(session_id)
        except:
            return jsonify({'error': 'Invalid session_id'}), 400
        
        # Delete session from MongoDB
        result = db.nba_matchup_sessions.delete_one({'_id': session_obj_id})
        
        if result.deleted_count == 0:
            return jsonify({'error': 'Session not found'}), 404
        
        # Also remove from in-memory agent sessions if it exists
        with _session_lock:
            if session_id in _matchup_agent_sessions:
                del _matchup_agent_sessions[session_id]
        
        return jsonify({'success': True})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/matchup-chat/sessions/by-game/<game_id>', methods=['DELETE'])
@app.route('/api/matchup-chat/sessions/by-game/<game_id>', methods=['DELETE'])
def delete_matchup_chat_sessions_by_game(game_id, league_id=None):
    """Delete all matchup chat sessions for a specific game_id and clear cached agents"""
    try:
        # Find all sessions with this game_id
        sessions = list(db.nba_matchup_sessions.find({'game_id': game_id}, {'_id': 1}))
        
        if not sessions:
            # Still clear cache even if no DB sessions found
            with _session_lock:
                # Find and remove any cached agents for this game_id
                # We need to check each agent's game_id
                sessions_to_remove = []
                for session_id, agent in _matchup_agent_sessions.items():
                    if hasattr(agent, 'game_id') and agent.game_id == game_id:
                        sessions_to_remove.append(session_id)
                
                for session_id in sessions_to_remove:
                    del _matchup_agent_sessions[session_id]
            
            return jsonify({
                'success': True,
                'message': f'No sessions found for game_id: {game_id}. Cleared {len(sessions_to_remove)} cached agent(s)',
                'deleted_count': 0,
                'cleared_cache_count': len(sessions_to_remove) if 'sessions_to_remove' in locals() else 0
            })
        
        session_ids = [str(s['_id']) for s in sessions]
        
        # Delete from MongoDB
        result = db.nba_matchup_sessions.delete_many({'game_id': game_id})
        
        # Remove from in-memory agent sessions
        with _session_lock:
            cleared_count = 0
            for session_id in session_ids:
                if session_id in _matchup_agent_sessions:
                    del _matchup_agent_sessions[session_id]
                    cleared_count += 1
            
            # Also check for any agents with matching game_id (in case session_id doesn't match)
            sessions_to_remove = []
            for session_id, agent in _matchup_agent_sessions.items():
                if hasattr(agent, 'game_id') and agent.game_id == game_id:
                    if session_id not in session_ids:  # Don't double-delete
                        sessions_to_remove.append(session_id)
            
            for session_id in sessions_to_remove:
                del _matchup_agent_sessions[session_id]
                cleared_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Deleted {result.deleted_count} session(s) and cleared {cleared_count} cached agent(s)',
            'deleted_count': result.deleted_count,
            'cleared_cache_count': cleared_count
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/master-training')
@app.route('/<league_id>/master-training')
def master_training(league_id=None):
    """Master training data browser page."""
    return render_template('master_training.html')


@app.route('/<league_id>/api/master-training/columns', methods=['GET'])
@app.route('/api/master-training/columns', methods=['GET'])
def master_training_columns(league_id=None):
    """Get column names and total row count from master training CSV."""
    try:
        master_training_path = get_master_training_path()

        if not os.path.exists(master_training_path):
            return jsonify({'error': 'Master training CSV file not found'}), 404

        # Read just the header to get columns
        try:
            df = pd.read_csv(master_training_path, nrows=0, on_bad_lines='skip', engine='python')
        except TypeError:
            df = pd.read_csv(master_training_path, nrows=0, error_bad_lines=False, warn_bad_lines=True, engine='python')

        columns = list(df.columns)

        # Count total rows (efficiently, without loading all data)
        total_rows = 0
        try:
            # Use a chunked approach to count rows
            chunk_size = 10000
            for chunk in pd.read_csv(master_training_path, chunksize=chunk_size, on_bad_lines='skip', engine='python'):
                total_rows += len(chunk)
        except TypeError:
            for chunk in pd.read_csv(master_training_path, chunksize=chunk_size, error_bad_lines=False, warn_bad_lines=True, engine='python'):
                total_rows += len(chunk)

        return jsonify({
            'columns': columns,
            'total_rows': total_rows
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/master-training/rows', methods=['GET'])
@app.route('/api/master-training/rows', methods=['GET'])
def master_training_rows(league_id=None):
    """Get paginated rows from master training CSV with sorting and filtering."""
    try:
        master_training_path = get_master_training_path()

        if not os.path.exists(master_training_path):
            return jsonify({'error': 'Master training CSV file not found'}), 404

        offset = int(request.args.get('offset', 0))
        limit = int(request.args.get('limit', 100))  # Default to 100 for initial load
        columns_param = request.args.get('columns', '')
        sort_column = request.args.get('sort_column', '')
        sort_direction = request.args.get('sort_direction', 'asc')

        # Parse date filter parameters
        year_min = request.args.get('year_min')
        year_max = request.args.get('year_max')
        month_min = request.args.get('month_min')
        month_max = request.args.get('month_max')
        day_min = request.args.get('day_min')
        day_max = request.args.get('day_max')
        date_start = request.args.get('date_start')  # YYYY-MM-DD format
        date_end = request.args.get('date_end')  # YYYY-MM-DD format

        # Parse requested columns
        requested_columns = [col.strip() for col in columns_param.split(',') if col.strip()] if columns_param else []

        # Read CSV in chunks to handle large files efficiently
        try:
            df = pd.read_csv(master_training_path, on_bad_lines='skip', engine='python')
        except TypeError:
            df = pd.read_csv(master_training_path, error_bad_lines=False, warn_bad_lines=True, engine='python')
        
        # Apply date filters before column filtering (need Year, Month, Day columns for filtering)
        if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
            # Filter by year range
            if year_min:
                year_min_val = int(year_min)
                df = df[df['Year'].astype(int) >= year_min_val]
            if year_max:
                year_max_val = int(year_max)
                df = df[df['Year'].astype(int) <= year_max_val]
            
            # Filter by month range
            if month_min:
                month_min_val = int(month_min)
                df = df[df['Month'].astype(int) >= month_min_val]
            if month_max:
                month_max_val = int(month_max)
                df = df[df['Month'].astype(int) <= month_max_val]
            
            # Filter by day range
            if day_min:
                day_min_val = int(day_min)
                df = df[df['Day'].astype(int) >= day_min_val]
            if day_max:
                day_max_val = int(day_max)
                df = df[df['Day'].astype(int) <= day_max_val]
            
            # Filter by date range (YYYY-MM-DD)
            if date_start or date_end:
                # Create a date column for comparison
                df = df.copy()
                try:
                    # Convert Year, Month, Day to datetime for comparison
                    df['_filter_date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
                    if date_start:
                        start_date = pd.to_datetime(date_start)
                        df = df[df['_filter_date'] >= start_date]
                    if date_end:
                        end_date = pd.to_datetime(date_end)
                        df = df[df['_filter_date'] <= end_date]
                    # Drop the helper column after filtering
                    df = df.drop('_filter_date', axis=1)
                except Exception as e:
                    # If date conversion fails, skip date range filtering
                    import logging
                    logging.warning(f"Error applying date range filter: {e}")
                    if '_filter_date' in df.columns:
                        df = df.drop('_filter_date', axis=1)
        
        # Filter to requested columns if specified
        if requested_columns:
            # Only include columns that exist in the CSV
            available_columns = [col for col in requested_columns if col in df.columns]
            if available_columns:
                df = df[available_columns]
            else:
                # If none of the requested columns exist, return all columns
                requested_columns = list(df.columns)
                df = df[requested_columns]
        else:
            requested_columns = list(df.columns)
        
        # Apply sorting if specified
        if sort_column and sort_column in df.columns:
            ascending = sort_direction == 'asc'
            df = df.sort_values(by=sort_column, ascending=ascending, na_position='last')
        
        # Get total count before pagination
        total_count = len(df)
        
        # Apply pagination
        end_idx = offset + limit
        paginated_df = df.iloc[offset:end_idx]
        
        # Convert to list of dictionaries
        rows = paginated_df.to_dict('records')
        
        # Replace NaN with None for JSON serialization
        for row in rows:
            for key, value in row.items():
                if pd.isna(value):
                    row[key] = None
        
        has_more = end_idx < total_count
        
        return jsonify({
            'rows': rows,
            'has_more': has_more,
            'total_count': total_count,
            'offset': offset,
            'limit': limit
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/<league_id>/api/market-prices', methods=['GET'])
@app.route('/api/market-prices', methods=['GET'])
def get_market_prices(league_id=None):
    """
    Get Kalshi market prices for all games on a specific date.

    Uses unauthenticated Kalshi public API to fetch live market prices.
    Thin wrapper around core/market/kalshi.get_game_market_data().
    """
    from bball_app.core.market.kalshi import get_game_market_data

    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

    try:
        game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format'}), 400

    # Get league config
    league = g.league
    league_id_str = league.league_id if league else 'nba'
    games_collection = league.collections.get('games', 'stats_nba')

    # Get games for this date
    games = list(db[games_collection].find(
        {'date': date_str},
        {'game_id': 1, 'homeTeam.name': 1, 'awayTeam.name': 1}
    ))

    if not games:
        return jsonify({
            'success': True,
            'markets': {},
            'message': 'No games found for this date'
        })

    # Fetch market data for each game
    markets = {}
    for game in games:
        game_id = game.get('game_id')
        home_team = game.get('homeTeam', {}).get('name', '')
        away_team = game.get('awayTeam', {}).get('name', '')

        if not home_team or not away_team:
            continue

        try:
            market_data = get_game_market_data(
                game_date=game_date,
                away_team=away_team,
                home_team=home_team,
                league_id=league_id_str,
                use_cache=True,
                cache_ttl=30  # 30 second cache for live data
            )

            if market_data:
                markets[game_id] = market_data.to_dict()
        except Exception as e:
            logger.warning(f"Failed to fetch market data for game {game_id}: {e}")
            continue

    return jsonify({
        'success': True,
        'markets': markets
    })


@app.route('/<league_id>/api/market/dashboard', methods=['GET'])
@app.route('/api/market/dashboard', methods=['GET'])
def get_market_dashboard(league_id=None):
    """
    Get market dashboard data including balance, returns, fills, and settlements.

    Uses authenticated Kalshi API to fetch user's trading statistics.
    """
    import os
    from datetime import timedelta
    from bball_app.core.market.connector import MarketConnector

    # Check for API credentials
    api_key = os.environ.get('KALSHI_API_KEY')
    private_key_dir = os.environ.get('KALSHI_PRIVATE_KEY_DIR')

    if not api_key or not private_key_dir:
        return jsonify({
            'success': False,
            'error': 'Kalshi API credentials not configured'
        }), 400

    try:
        connector = MarketConnector({
            'KALSHI_API_KEY': api_key,
            'KALSHI_PRIVATE_KEY_DIR': private_key_dir
        })
    except Exception as e:
        logger.error(f"Failed to initialize MarketConnector: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to connect to Kalshi API: {str(e)}'
        }), 500

    try:
        # Get balance
        balance_resp = connector.get_balance()
        balance_cents = balance_resp.get('balance', 0)

        # Fetch first page of settlements for display (keep cursor for lazy-loading)
        first_resp = connector.get_settlements(limit=20)
        settlements = list(first_resp.get('settlements', []))
        settlements_cursor = first_resp.get('cursor')

        # Continue fetching remaining settlements for stats/chart calculation
        stats_cursor = settlements_cursor
        while stats_cursor:
            settlements_resp = connector.get_settlements(limit=100, cursor=stats_cursor)
            page = settlements_resp.get('settlements', [])
            if not page:
                break
            settlements.extend(page)
            stats_cursor = settlements_resp.get('cursor')

        # Get fills
        fills_resp = connector.get_fills(limit=100)
        fills = fills_resp.get('fills', [])
        fills_cursor = fills_resp.get('cursor')

        # Calculate time-based returns from settlements
        now = datetime.now()
        ts_24h = int((now - timedelta(hours=24)).timestamp() * 1000)
        ts_7d = int((now - timedelta(days=7)).timestamp() * 1000)
        ts_30d = int((now - timedelta(days=30)).timestamp() * 1000)

        total_return_cents = 0
        return_24h_cents = 0
        return_7d_cents = 0
        return_30d_cents = 0
        winning_trades = 0
        losing_trades = 0

        for s in settlements:
            revenue = s.get('revenue', 0)
            cost_basis = s.get('yes_total_cost', 0) + s.get('no_total_cost', 0)
            pnl = revenue - cost_basis
            ts_raw = s.get('settled_time', 0)

            # Parse ISO timestamp string to epoch ms for comparison
            if isinstance(ts_raw, str):
                try:
                    ts = int(datetime.fromisoformat(ts_raw.replace('Z', '+00:00')).timestamp() * 1000)
                except (ValueError, TypeError):
                    ts = 0
            else:
                ts = ts_raw

            total_return_cents += pnl
            if ts >= ts_24h:
                return_24h_cents += pnl
            if ts >= ts_7d:
                return_7d_cents += pnl
            if ts >= ts_30d:
                return_30d_cents += pnl

            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1

        # Calculate total invested from fills
        total_invested_cents = 0
        for f in fills:
            # Each fill has a cost (price * count in cents)
            if f.get('side') == 'yes':
                total_invested_cents += f.get('yes_price', 0) * f.get('count', 0)
            else:
                total_invested_cents += f.get('no_price', 0) * f.get('count', 0)

        # Calculate ROI
        total_trades = winning_trades + losing_trades
        roi = (total_return_cents / total_invested_cents * 100) if total_invested_cents > 0 else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        # Format recent fills for display
        recent_fills = []
        for f in fills[:20]:
            side = f.get('side', '')
            price_cents = f.get('yes_price', 0) if side == 'yes' else f.get('no_price', 0)
            count = f.get('count', 0)
            recent_fills.append({
                'created_time': f.get('created_time'),
                'ticker': f.get('ticker', ''),
                'action': f.get('action', 'buy'),
                'side': side,
                'count': count,
                'price': price_cents,
                'cost': count * price_cents,
            })

        # Format recent settlements for display
        recent_settlements = []
        for s in settlements[:20]:
            revenue = s.get('revenue', 0)
            cost_basis = s.get('yes_total_cost', 0) + s.get('no_total_cost', 0)
            pnl = revenue - cost_basis
            recent_settlements.append({
                'settled_time': s.get('settled_time'),
                'ticker': s.get('ticker', ''),
                'market_result': s.get('market_result', ''),
                'count': s.get('count', 0),
                'revenue': revenue,
                'pnl': pnl,
            })

        # Build cumulative returns over time from settlements (sorted by time)
        sorted_settlements = sorted(settlements, key=lambda s: s.get('settled_time', ''))
        cumulative = 0
        returns_over_time = []
        for s in sorted_settlements:
            revenue = s.get('revenue', 0)
            cost_basis = s.get('yes_total_cost', 0) + s.get('no_total_cost', 0)
            pnl = revenue - cost_basis
            cumulative += pnl
            ts_raw = s.get('settled_time', '')
            if isinstance(ts_raw, str) and ts_raw:
                try:
                    dt = datetime.fromisoformat(ts_raw.replace('Z', '+00:00'))
                    date_label = dt.strftime('%m/%d')
                except (ValueError, TypeError):
                    date_label = ts_raw[:10] if len(ts_raw) >= 10 else ts_raw
            else:
                date_label = str(ts_raw)
            returns_over_time.append({
                'date': date_label,
                'cumulative_return': cumulative,
            })

        return jsonify({
            'success': True,
            'stats': {
                'balance_cents': balance_cents,
                'total_return_cents': total_return_cents,
                'return_24h_cents': return_24h_cents,
                'return_7d_cents': return_7d_cents,
                'return_30d_cents': return_30d_cents,
                'total_invested_cents': total_invested_cents,
                'roi': roi,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
            },
            'recent_fills': recent_fills,
            'recent_settlements': recent_settlements,
            'fills_cursor': fills_cursor,
            'settlements_cursor': settlements_cursor,
            'returns_over_time': returns_over_time
        })

    except Exception as e:
        import traceback
        logger.error(f"Market dashboard error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<league_id>/api/market/fills', methods=['GET'])
@app.route('/api/market/fills', methods=['GET'])
def get_market_fills(league_id=None):
    """Get paginated fills from Kalshi API."""
    import os
    from bball_app.core.market.connector import MarketConnector

    cursor = request.args.get('cursor')

    # Check for API credentials
    api_key = os.environ.get('KALSHI_API_KEY')
    private_key_dir = os.environ.get('KALSHI_PRIVATE_KEY_DIR')

    if not api_key or not private_key_dir:
        return jsonify({
            'success': False,
            'error': 'Kalshi API credentials not configured'
        }), 400

    try:
        connector = MarketConnector({
            'KALSHI_API_KEY': api_key,
            'KALSHI_PRIVATE_KEY_DIR': private_key_dir
        })

        fills_resp = connector.get_fills(limit=100, cursor=cursor)
        fills = fills_resp.get('fills', [])
        next_cursor = fills_resp.get('cursor')

        formatted_fills = []
        for f in fills:
            side = f.get('side', '')
            price_cents = f.get('yes_price', 0) if side == 'yes' else f.get('no_price', 0)
            count = f.get('count', 0)
            formatted_fills.append({
                'created_time': f.get('created_time'),
                'ticker': f.get('ticker', ''),
                'action': f.get('action', 'buy'),
                'side': side,
                'count': count,
                'price': price_cents,
                'cost': count * price_cents,
            })

        return jsonify({
            'success': True,
            'fills': formatted_fills,
            'cursor': next_cursor
        })

    except Exception as e:
        logger.error(f"Market fills error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<league_id>/api/market/settlements', methods=['GET'])
@app.route('/api/market/settlements', methods=['GET'])
def get_market_settlements(league_id=None):
    """Get paginated settlements from Kalshi API."""
    import os
    from bball_app.core.market.connector import MarketConnector

    cursor = request.args.get('cursor')

    # Check for API credentials
    api_key = os.environ.get('KALSHI_API_KEY')
    private_key_dir = os.environ.get('KALSHI_PRIVATE_KEY_DIR')

    if not api_key or not private_key_dir:
        return jsonify({
            'success': False,
            'error': 'Kalshi API credentials not configured'
        }), 400

    try:
        connector = MarketConnector({
            'KALSHI_API_KEY': api_key,
            'KALSHI_PRIVATE_KEY_DIR': private_key_dir
        })

        settlements_resp = connector.get_settlements(limit=100, cursor=cursor)
        settlements = settlements_resp.get('settlements', [])
        next_cursor = settlements_resp.get('cursor')

        formatted_settlements = []
        for s in settlements:
            revenue = s.get('revenue', 0)
            cost_basis = s.get('yes_total_cost', 0) + s.get('no_total_cost', 0)
            pnl = revenue - cost_basis
            formatted_settlements.append({
                'settled_time': s.get('settled_time'),
                'ticker': s.get('ticker', ''),
                'market_result': s.get('market_result', ''),
                'count': s.get('count', 0),
                'revenue': revenue,
                'pnl': pnl,
            })

        return jsonify({
            'success': True,
            'settlements': formatted_settlements,
            'cursor': next_cursor
        })

    except Exception as e:
        logger.error(f"Market settlements error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/<league_id>/api/live-games', methods=['GET'])
@app.route('/api/live-games', methods=['GET'])
def get_live_games(league_id=None):
    """
    Get live game data (scores, period, clock, status) from ESPN API.

    Used for live polling to update game cards and modal with real-time data.
    """
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

    try:
        game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format'}), 400

    # Get league config
    league = g.league
    league_id_str = league.league_id if league else 'nba'

    # Fetch from ESPN API using league config template
    date_yyyymmdd = date_str.replace('-', '')
    try:
        espn_url = league.espn_endpoint('scoreboard_site_template').format(YYYYMMDD=date_yyyymmdd)
    except Exception:
        # Fallback for NBA
        espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_yyyymmdd}"

    live_games = {}
    try:
        import requests
        resp = requests.get(espn_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        for event in data.get('events', []):
            game_id = event.get('id')
            if not game_id:
                continue

            # Extract status - can be string or dict
            status_obj = event.get('status', {})
            game_status = 'pre'
            game_completed = False
            period = None
            clock = None
            status_detail = None

            if isinstance(status_obj, str):
                if status_obj.lower() in ('final', 'completed', 'post'):
                    game_status = 'post'
                    game_completed = True
                elif status_obj.lower() in ('active', 'in', 'in progress', 'live'):
                    game_status = 'in'
                else:
                    game_status = 'pre'
            elif isinstance(status_obj, dict):
                status_type = status_obj.get('type', {})
                if isinstance(status_type, dict):
                    raw_status = status_type.get('name', '').lower()
                    game_completed = status_type.get('completed', False)
                    # Get human-readable status detail (e.g., "Halftime", "End of 3rd")
                    status_detail = status_type.get('shortDetail') or status_type.get('detail')
                    # Normalize ESPN status names to 'pre', 'in', or 'post'
                    if game_completed or 'final' in raw_status or 'post' in raw_status:
                        game_status = 'post'
                    elif 'progress' in raw_status or 'halftime' in raw_status or raw_status == 'in':
                        game_status = 'in'
                    else:
                        game_status = 'pre'
                period = status_obj.get('period')
                clock = status_obj.get('displayClock')

            # Extract scores from competitors
            home_score = None
            away_score = None
            competitions = event.get('competitions', [])
            if competitions:
                competitors = competitions[0].get('competitors', [])
                for comp in competitors:
                    score_val = comp.get('score')
                    # Only parse if score exists and is not empty
                    if score_val is not None and score_val != '':
                        try:
                            score = int(score_val)
                            if comp.get('homeAway') == 'home':
                                home_score = score
                            else:
                                away_score = score
                        except (ValueError, TypeError):
                            pass

            live_games[game_id] = {
                'status': game_status,
                'completed': game_completed,
                'period': period,
                'clock': clock,
                'status_detail': status_detail,
                'home_score': home_score,
                'away_score': away_score
            }

    except Exception as e:
        logger.warning(f"Failed to fetch live games from ESPN: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

    return jsonify({
        'success': True,
        'games': live_games
    })


@app.route('/<league_id>/api/portfolio/game-positions', methods=['GET'])
@app.route('/api/portfolio/game-positions', methods=['GET'])
def get_portfolio_game_positions(league_id=None):
    """
    Get portfolio positions, orders, and fills for games on a specific date.

    Uses authenticated Kalshi API to fetch user's trading activity.
    Thin wrapper around core/market/kalshi.match_portfolio_to_games().
    """
    import os
    from bball_app.core.market.connector import MarketConnector
    from bball_app.core.market.kalshi import match_portfolio_to_games
    from bball_app.core.league_config import load_league_config

    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'success': False, 'error': 'Missing date parameter'}), 400

    try:
        game_date = datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        return jsonify({'success': False, 'error': 'Invalid date format'}), 400

    # Check for API credentials
    api_key = os.environ.get('KALSHI_API_KEY')
    private_key_dir = os.environ.get('KALSHI_PRIVATE_KEY_DIR')

    if not api_key or not private_key_dir:
        return jsonify({
            'success': True,
            'available': False,
            'message': 'Kalshi API credentials not configured'
        })

    try:
        connector = MarketConnector({
            'KALSHI_API_KEY': api_key,
            'KALSHI_PRIVATE_KEY_DIR': private_key_dir
        })
    except Exception as e:
        logger.error(f"Failed to initialize MarketConnector: {e}")
        return jsonify({
            'success': True,
            'available': False,
            'message': 'Failed to connect to Kalshi API'
        })

    # Get league from request context
    league_id = getattr(g, "league_id", "nba")
    league_config = getattr(g, 'league', None)
    games_collection = league_config.collections.get('games', 'stats_nba') if league_config else 'stats_nba'

    # Get games for this date
    games = list(db[games_collection].find(
        {'date': date_str},
        {'game_id': 1, 'homeTeam.name': 1, 'awayTeam.name': 1}
    ))

    if not games:
        return jsonify({
            'success': True,
            'available': True,
            'positions': {},
            'message': 'No games found for this date'
        })

    # Get series tickers for filtering Kalshi data by league
    try:
        league_cfg = load_league_config(league_id)
        series_ticker = league_cfg.raw.get("market", {}).get("series_ticker", "KXNBAGAME")
        spread_series_ticker = league_cfg.raw.get("market", {}).get("spread_series_ticker", "")
    except Exception:
        series_ticker = "KXNBAGAME"
        spread_series_ticker = "KXNBASPREAD"

    # Fetch raw portfolio data from Kalshi API
    try:
        positions_resp = connector.get_positions(limit=200)
        all_positions = positions_resp.get('market_positions', [])
    except Exception as e:
        logger.error(f"Failed to fetch positions: {e}")
        all_positions = []

    # Fetch recent fills (last 24 hours)
    try:
        min_ts = int((datetime.now(utc).timestamp() - 86400) * 1000)
        fills_resp = connector.get_fills(min_ts=min_ts, limit=200)
        all_fills = fills_resp.get('fills', [])

        # DEBUG: Log ALL unique tickers and event_tickers from raw Kalshi response
        all_raw_tickers = set()
        all_raw_event_tickers = set()
        for f in all_fills:
            if f.get('ticker'):
                all_raw_tickers.add(f.get('ticker'))
            if f.get('event_ticker'):
                all_raw_event_tickers.add(f.get('event_ticker'))

        logger.info(f"[RAW KALSHI DEBUG] Total fills from API: {len(all_fills)}")
        logger.info(f"[RAW KALSHI DEBUG] Unique tickers: {sorted(all_raw_tickers)}")
        logger.info(f"[RAW KALSHI DEBUG] Unique event_tickers: {sorted(all_raw_event_tickers)}")

        # Check specifically for SPREAD
        spread_tickers = [t for t in all_raw_tickers if 'SPREAD' in t.upper()]
        spread_event_tickers = [t for t in all_raw_event_tickers if 'SPREAD' in t.upper()]
        logger.info(f"[RAW KALSHI DEBUG] Spread tickers found: {spread_tickers}")
        logger.info(f"[RAW KALSHI DEBUG] Spread event_tickers found: {spread_event_tickers}")
    except Exception as e:
        logger.error(f"Failed to fetch fills: {e}")
        all_fills = []

    # Fetch open orders
    try:
        orders_resp = connector.get_orders(status='resting', limit=200)
        all_orders = orders_resp.get('orders', [])
    except Exception as e:
        logger.error(f"Failed to fetch orders: {e}")
        all_orders = []

    # Fetch settlements for this date range to determine won/lost
    try:
        # Get settlements from yesterday through tomorrow to catch all relevant games
        min_ts = int((datetime.combine(game_date, datetime.min.time()).timestamp() - 86400) * 1000)
        settlements_resp = connector.get_settlements(min_ts=min_ts, limit=200)
        all_settlements = settlements_resp.get('settlements', [])
    except Exception as e:
        logger.error(f"Failed to fetch settlements: {e}")
        all_settlements = []

    # Filter all Kalshi data to only include items for the current league's series_ticker
    # Also include spread series ticker to show spread fills alongside W/L fills
    # Also include MULTIGAME combo fills (they contain legs for our league's games)
    def matches_league(item):
        ticker = item.get('ticker', '') or item.get('event_ticker', '')
        if ticker.startswith(series_ticker):
            return True
        if spread_series_ticker and ticker.startswith(spread_series_ticker):
            return True
        # Include MULTIGAME combo fills - we'll filter by legs later
        if 'MULTIGAME' in ticker.upper():
            return True
        return False

    # Debug: log ALL fills from Kalshi API (before any filtering)
    logger.info(f"[SPREAD DEBUG] Raw fills from Kalshi: {len(all_fills)} total")
    spread_fills_before = [f for f in all_fills if 'SPREAD' in f.get('ticker', '').upper()]
    logger.info(f"[SPREAD DEBUG] Spread fills in raw response: {len(spread_fills_before)}")
    for sf in spread_fills_before[:5]:
        logger.info(f"[SPREAD DEBUG]   RAW ticker: {sf.get('ticker')}")

    all_positions = [p for p in all_positions if matches_league(p)]
    all_fills = [f for f in all_fills if matches_league(f)]
    all_orders = [o for o in all_orders if matches_league(o)]
    all_settlements = [s for s in all_settlements if matches_league(s)]

    # Debug: log spread fills after filtering
    spread_fills_after = [f for f in all_fills if f.get('ticker', '').startswith('KXNBASPREAD')]
    logger.info(f"[SPREAD DEBUG] After filter: {len(spread_fills_after)} spread fills, series_ticker={series_ticker}, spread_series_ticker={spread_series_ticker}")

    # Use core layer to match portfolio items to games
    result = match_portfolio_to_games(
        games=games,
        positions=all_positions,
        fills=all_fills,
        orders=all_orders,
        settlements=all_settlements,
        game_date=game_date,
        league_id=league_id
    )

    # Include debug info for spread fills
    spread_fills_count = len([f for f in all_fills if f.get('ticker', '').startswith('KXNBASPREAD')])
    total_matched_fills = sum(len(gd.get('fills', [])) for gd in result.game_data.values())
    unmatched_spread_tickers = [f.get('ticker') for f in result.unmatched_fills if f.get('ticker', '').startswith('KXNBASPREAD')]

    # Get all unique fill tickers that passed filtering
    all_fill_tickers = list(set(f.get('ticker', '') for f in all_fills))
    spread_fill_tickers = [t for t in all_fill_tickers if 'SPREAD' in t.upper()]

    return jsonify({
        'success': True,
        'available': True,
        'date': date_str,
        'positions': result.game_data,
        'fetched_at': datetime.now(utc).isoformat(),
        '_debug': {
            'spread_fills_passed_to_core': spread_fills_count,
            'spread_fill_tickers': spread_fill_tickers[:10],
            'total_matched_fills': total_matched_fills,
            'unmatched_fills_count': len(result.unmatched_fills),
            'unmatched_spread_tickers': unmatched_spread_tickers[:5],
            'all_unmatched_tickers': [f.get('ticker') for f in result.unmatched_fills][:10],
            'games_count': len(games),
            'spread_series_ticker': spread_series_ticker,
            'core_debug': result.debug_info
        }
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

