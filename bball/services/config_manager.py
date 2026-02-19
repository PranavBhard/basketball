"""
Unified Model Configuration Management.
Centralizes all model configuration operations using MongoDB as single source of truth.

This is the SINGLE POINT of config DB interaction for:
- Web UI (model_config endpoints)
- Modeler Agent (config creation/management tools)
- Experiment Runner (linking runs to configs)

Uses data layer repositories for all database operations.
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
from bson import ObjectId

from bball.data import ClassifierConfigRepository, PointsConfigRepository

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


class ModelConfigManager:
    """
    Centralized model configuration management using MongoDB.

    Supports both classifier (model_config_nba) and points regression
    (model_config_points_nba) configurations.

    Uses ClassifierConfigRepository and PointsConfigRepository for data access.
    """

    # Collection names (kept for backward compatibility)
    CLASSIFIER_COLLECTION = 'nba_model_config'
    POINTS_COLLECTION = 'model_config_points_nba'

    def __init__(self, db, league: Optional["LeagueConfig"] = None):
        self.db = db
        self.league = league
        # Initialize repositories
        self._classifier_repo = ClassifierConfigRepository(db, league=league)
        self._points_repo = PointsConfigRepository(db, league=league)

    # =========================================================================
    # HASH GENERATION (Single source of truth)
    # =========================================================================

    @staticmethod
    def generate_feature_set_hash(features: List[str]) -> str:
        """
        Generate deterministic hash from feature list.

        Args:
            features: List of feature names

        Returns:
            MD5 hash of sorted, joined feature names
        """
        if not features:
            return hashlib.md5(''.encode()).hexdigest()
        sorted_features = sorted(features)
        return hashlib.md5(','.join(sorted_features).encode()).hexdigest()

    @staticmethod
    def generate_config_hash(
        model_type: str,
        feature_set_hash: str,
        c_value: float = None,
        alpha: float = None,
        use_time_calibration: bool = False,
        calibration_method: str = None,
        calibration_years: List[int] = None,
        begin_year: int = None,
        evaluation_year: int = None,
        include_injuries: bool = False,
        recency_decay_k: float = None,
        use_master: bool = True,
        min_games_played: int = 15,
        target: str = None,  # For points regression
        exclude_seasons: List[int] = None
    ) -> str:
        """
        Generate unique config hash from parameters.

        This is the SINGLE hash generation method - used by web UI and agents.

        Returns:
            MD5 hash string
        """
        # Build hash string with all relevant fields
        parts = [
            f"model_type:{model_type}",
            f"feature_set_hash:{feature_set_hash}",
        ]

        # Add optional fields only if set
        if c_value is not None:
            parts.append(f"c_value:{c_value}")
        if alpha is not None:
            parts.append(f"alpha:{alpha}")
        if use_time_calibration:
            parts.append(f"use_time_calibration:{use_time_calibration}")
        if calibration_method:
            parts.append(f"calibration_method:{calibration_method}")
        if calibration_years:
            parts.append(f"calibration_years:{','.join(map(str, sorted(calibration_years)))}")
        if begin_year is not None:
            parts.append(f"begin_year:{begin_year}")
        if evaluation_year is not None:
            parts.append(f"evaluation_year:{evaluation_year}")
        if include_injuries:
            parts.append(f"include_injuries:{include_injuries}")
        if recency_decay_k is not None:
            parts.append(f"recency_decay_k:{recency_decay_k}")
        parts.append(f"use_master:{use_master}")
        parts.append(f"min_games_played:{min_games_played}")
        if exclude_seasons:
            parts.append(f"exclude_seasons:{','.join(map(str, sorted(exclude_seasons)))}")
        if target:
            parts.append(f"target:{target}")

        hash_str = '|'.join(parts)
        return hashlib.md5(hash_str.encode()).hexdigest()

    # =========================================================================
    # CONFIG CREATION (From experiment specs)
    # =========================================================================

    def create_classifier_config(
        self,
        model_type: str,
        features: List[str],
        c_value: float = 0.1,
        use_time_calibration: bool = True,
        calibration_method: str = 'sigmoid',
        begin_year: int = 2012,
        calibration_years: List[int] = None,
        evaluation_year: int = 2024,
        min_games_played: int = 15,
        include_injuries: bool = False,
        recency_decay_k: float = None,
        exclude_seasons: List[int] = None,
        use_master: bool = True,
        name: str = None,
        # Dataset spec fields (for reproducibility)
        dataset_spec: Dict = None,
        diff_mode: str = 'home_minus_away',
        feature_blocks: List[str] = None,
        include_per: bool = True,
        point_model_id: str = None,
        # Don't auto-select by default
        selected: bool = False,
        force_insert: bool = False
    ) -> Tuple[str, Dict]:
        """
        Create a classifier model config.

        This method creates a config document and upserts it by hash.
        Same config params = same hash = same document (no duplicates).

        Args:
            model_type: Model type (LogisticRegression, RandomForest, etc.)
            features: List of feature names
            c_value: Regularization parameter
            use_time_calibration: Whether to use time-based calibration
            calibration_method: 'sigmoid' or 'isotonic'
            begin_year: Training data start year
            calibration_years: Years for calibration set
            evaluation_year: Year for evaluation set
            min_games_played: Minimum games filter
            include_injuries: Include injury features
            recency_decay_k: Injury recency decay parameter
            exclude_seasons: Seasons to exclude from training
            use_master: Use master training CSV
            name: Optional custom name
            dataset_spec: Full dataset spec for reproducibility
            diff_mode: Feature differencing mode
            feature_blocks: Feature blocks used (for reference)
            include_per: Whether PER features included
            point_model_id: Points model reference
            selected: Whether to mark as selected
            force_insert: If True, always create a new document (skip upsert by hash).
                         Used by recalibrate_ensemble to avoid overwriting base models
                         shared with other ensembles.

        Returns:
            Tuple of (config_id, config_dict)
        """
        if calibration_years is None:
            calibration_years = [2023]

        # Generate hashes
        feature_set_hash = self.generate_feature_set_hash(features)
        config_hash = self.generate_config_hash(
            model_type=model_type,
            feature_set_hash=feature_set_hash,
            c_value=c_value,
            use_time_calibration=use_time_calibration,
            calibration_method=calibration_method,
            calibration_years=calibration_years,
            begin_year=begin_year,
            evaluation_year=evaluation_year,
            include_injuries=include_injuries,
            recency_decay_k=recency_decay_k,
            use_master=use_master,
            min_games_played=min_games_played,
            exclude_seasons=exclude_seasons,
        )

        # Auto-generate name if not provided
        if not name:
            name = f"{model_type} - {config_hash[:8]}"

        # Build config document
        config = {
            'config_hash': config_hash,
            'model_type': model_type,
            'features': features,
            'feature_count': len(features),
            'feature_set_hash': feature_set_hash,
            'name': name,
            'best_c_value': c_value,
            'use_time_calibration': use_time_calibration,
            'calibration_method': calibration_method,
            'begin_year': begin_year,
            'calibration_years': calibration_years,
            'evaluation_year': evaluation_year,
            'min_games_played': min_games_played,
            'include_injuries': include_injuries,
            'recency_decay_k': recency_decay_k,
            'exclude_seasons': exclude_seasons,
            'use_master': use_master,
            'ensemble': False,
            # Dataset reproducibility fields
            'diff_mode': diff_mode,
            'feature_blocks': feature_blocks or [],
            'include_per': include_per,
            'point_model_id': point_model_id,
            'dataset_spec': dataset_spec,
            # Timestamps
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }

        if force_insert:
            # Always create a new document (used by recalibrate_ensemble to isolate
            # base model configs from existing ensembles that share the same hash)
            import uuid
            config['config_hash'] = f"{config_hash}_{uuid.uuid4().hex[:8]}"
            result = self._classifier_repo.insert_one(config)
            config_id = str(result.inserted_id)
        else:
            # Upsert by config_hash using repository
            self._classifier_repo.upsert_config(config_hash, config)
            doc = self._classifier_repo.find_by_hash(config_hash)
            config_id = str(doc['_id'])

        # Handle selection safely (after upsert)
        if selected:
            self._safe_select(self.CLASSIFIER_COLLECTION, config_id)

        config['_id'] = config_id
        return config_id, config

    def create_points_config(
        self,
        model_type: str,
        features: List[str],
        target: str = 'home_away',
        alpha: float = 1.0,
        l1_ratio: float = None,
        begin_year: int = 2012,
        calibration_years: List[int] = None,
        evaluation_year: int = 2024,
        min_games_played: int = 15,
        use_master: bool = True,
        name: str = None,
        # Dataset spec fields
        dataset_spec: Dict = None,
        diff_mode: str = 'home_minus_away',
        feature_blocks: List[str] = None,
        include_per: bool = True,
        selected: bool = False
    ) -> Tuple[str, Dict]:
        """
        Create a points regression model config.

        Args:
            model_type: Model type (Ridge, ElasticNet, RandomForest, XGBoost)
            features: List of feature names
            target: 'home_away' (separate models) or 'margin' (single model)
            alpha: Regularization parameter
            l1_ratio: L1 ratio for ElasticNet
            begin_year: Training data start year
            calibration_years: Years for calibration
            evaluation_year: Year for evaluation
            min_games_played: Minimum games filter
            use_master: Use master training CSV
            name: Optional custom name
            dataset_spec: Full dataset spec for reproducibility
            diff_mode: Feature differencing mode
            feature_blocks: Feature blocks used
            include_per: Whether PER features included
            selected: Whether to mark as selected

        Returns:
            Tuple of (config_id, config_dict)
        """
        if calibration_years is None:
            calibration_years = [2023]

        # Generate hashes
        feature_set_hash = self.generate_feature_set_hash(features)
        config_hash = self.generate_config_hash(
            model_type=model_type,
            feature_set_hash=feature_set_hash,
            alpha=alpha,
            begin_year=begin_year,
            evaluation_year=evaluation_year,
            min_games_played=min_games_played,
            use_master=use_master,
            target=target
        )

        # Auto-generate name if not provided
        if not name:
            name = f"{model_type} ({target}) - {config_hash[:8]}"

        # Build config document
        config = {
            'config_hash': config_hash,
            'model_type': model_type,
            'target': target,
            'features': features,
            'feature_count': len(features),
            'feature_set_hash': feature_set_hash,
            'name': name,
            'best_alpha': alpha,
            'l1_ratio': l1_ratio,
            'begin_year': begin_year,
            'calibration_years': calibration_years,
            'evaluation_year': evaluation_year,
            'min_games_played': min_games_played,
            'use_master': use_master,
            # Dataset reproducibility fields
            'diff_mode': diff_mode,
            'feature_blocks': feature_blocks or [],
            'include_per': include_per,
            'dataset_spec': dataset_spec,
            # Timestamps
            'created_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }

        # Upsert by config_hash using repository
        self._points_repo.upsert_config(config_hash, config)

        # Get config ID
        doc = self._points_repo.find_by_hash(config_hash)
        config_id = str(doc['_id'])

        # Handle selection safely
        if selected:
            self._safe_select(self.POINTS_COLLECTION, config_id)

        config['_id'] = config_id
        return config_id, config

    def link_run_to_config(
        self,
        config_id: str,
        run_id: str,
        config_type: str = 'classifier',
        metrics: Dict = None,
        artifacts: Dict = None,
        dataset_id: str = None,
        training_csv: str = None,
        f_scores: Dict = None,
        feature_importances: Dict = None,
        features: List[str] = None,
        c_values_grid: Dict = None,
        best_c_value: float = None,
        best_c_accuracy: float = None,
        training_stats: Dict = None,
        selected: bool = None
    ) -> bool:
        """
        Link experiment run results to a config.

        Updates the config with training metrics, artifacts, dataset reference,
        and feature rankings (both F-scores and model importances). This is the
        single source of truth for linking training results to configs - used by
        both web UI and CLI.

        Args:
            config_id: Config MongoDB ID
            run_id: Experiment run ID
            config_type: 'classifier' or 'points'
            metrics: Training metrics (accuracy, log_loss, etc.)
            artifacts: Model artifact paths
            dataset_id: Dataset ID used for training
            training_csv: Path to training CSV
            f_scores: Dict of {feature_name: f_score} from ANOVA F-test
            feature_importances: Dict of {feature_name: importance_score} from model
            features: List of feature names used in training
            c_values_grid: Dict of {c_value_str: accuracy} from grid search
            best_c_value: Best C-value from grid search
            best_c_accuracy: Accuracy at best C-value
            training_stats: Dict with training metadata (total_games, flags, etc.)
            selected: Whether to mark this config as selected after linking

        Returns:
            True if successful
        """
        repo = self._classifier_repo if config_type == 'classifier' else self._points_repo
        collection_name = self.CLASSIFIER_COLLECTION if config_type == 'classifier' else self.POINTS_COLLECTION

        update_doc = {
            'run_id': run_id,
            'trained_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }

        if metrics:
            update_doc['accuracy'] = metrics.get('accuracy_mean')
            update_doc['std_dev'] = metrics.get('accuracy_std')
            update_doc['log_loss'] = metrics.get('log_loss_mean')
            update_doc['brier_score'] = metrics.get('brier_mean')
            update_doc['auc'] = metrics.get('auc')
            # Points-specific metrics
            if 'margin_mae' in metrics:
                update_doc['margin_mae'] = metrics.get('margin_mae')
                update_doc['margin_rmse'] = metrics.get('margin_rmse')
            if 'total_mae' in metrics:
                update_doc['total_mae'] = metrics.get('total_mae')
                update_doc['total_rmse'] = metrics.get('total_rmse')

        # Store features
        if features:
            update_doc['features'] = sorted(features)
            update_doc['feature_count'] = len(features)

        # Store F-scores as features_ranked (ANOVA F-test scores)
        if f_scores:
            sorted_f_scores = sorted(f_scores.items(), key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
            features_ranked = []
            for rank, (name, score) in enumerate(sorted_f_scores, 1):
                # Sanitize NaN/Inf values
                if score is None or (isinstance(score, float) and (score != score or abs(score) == float('inf'))):
                    score = 0.0
                features_ranked.append({
                    'rank': rank,
                    'name': name,
                    'score': float(score)
                })
            update_doc['features_ranked'] = features_ranked

        # Store model importances as features_ranked_by_importance
        if feature_importances:
            sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1] if x[1] is not None else float('-inf'), reverse=True)
            features_ranked_by_importance = []
            for rank, (name, score) in enumerate(sorted_importances, 1):
                # Sanitize NaN/Inf values
                if score is None or (isinstance(score, float) and (score != score or abs(score) == float('inf'))):
                    score = 0.0
                features_ranked_by_importance.append({
                    'rank': rank,
                    'name': name,
                    'score': float(score)
                })
            update_doc['features_ranked_by_importance'] = features_ranked_by_importance

        if artifacts:
            update_doc['model_artifact_path'] = artifacts.get('model_path')
            update_doc['scaler_artifact_path'] = artifacts.get('scaler_path')
            # ExperimentRunner uses 'feature_names_path', ArtifactLoader expects 'features_path'
            update_doc['features_path'] = artifacts.get('features_path') or artifacts.get('feature_names_path')
            update_doc['artifacts_saved_at'] = datetime.utcnow()

        if dataset_id:
            update_doc['dataset_id'] = dataset_id

        if training_csv:
            update_doc['training_csv'] = training_csv

        # C-value grid search results
        if c_values_grid is not None:
            update_doc['c_values'] = c_values_grid
        if best_c_value is not None:
            update_doc['best_c_value'] = best_c_value
        if best_c_accuracy is not None:
            update_doc['best_c_accuracy'] = best_c_accuracy

        # Training stats metadata
        if training_stats is not None:
            update_doc['training_stats'] = training_stats

        try:
            result = repo.update_one(
                {'_id': ObjectId(config_id)},
                {'$set': update_doc}
            )

            # Handle selection after update
            if selected:
                self._safe_select(collection_name, config_id)

            return result.modified_count > 0
        except Exception as e:
            print(f"Error linking run to config: {e}")
            return False

    def _safe_select(self, collection_name: str, config_id: str):
        """
        Safely select a config (unselect others AFTER insert).

        This prevents race conditions where no config is selected.
        """
        repo = self._classifier_repo if collection_name == self.CLASSIFIER_COLLECTION else self._points_repo

        # First, unselect all EXCEPT this one
        repo.update_many(
            {'_id': {'$ne': ObjectId(config_id)}, 'selected': True},
            {'$set': {'selected': False}}
        )

        # Then select this one
        repo.update_one(
            {'_id': ObjectId(config_id)},
            {'$set': {'selected': True}}
        )

    # =========================================================================
    # POINTS CONFIG METHODS
    # =========================================================================

    def get_points_config(self, config_id: str = None, config_hash: str = None, selected: bool = False) -> Optional[Dict]:
        """Get points regression config by ID, hash, or selected flag."""
        try:
            if config_id:
                return self._points_repo.find_by_id(config_id)
            elif config_hash:
                return self._points_repo.find_by_hash(config_hash)
            elif selected:
                return self._points_repo.find_selected()
            else:
                raise ValueError("Must specify config_id, config_hash, or selected=True")
        except Exception as e:
            print(f"Error getting points config: {e}")
            return None

    def list_points_configs(self, model_type: str = None, trained_only: bool = False) -> List[Dict]:
        """List points regression configs with optional filtering."""
        try:
            if model_type:
                configs = self._points_repo.find_by_model_type(model_type)
            else:
                configs = self._points_repo.find_all(trained_only=trained_only)

            # Filter by trained_only if model_type was also specified
            if model_type and trained_only:
                configs = [c for c in configs if c.get('model_artifact_path')]

            for config in configs:
                if '_id' in config:
                    config['_id'] = str(config['_id'])

            return configs
        except Exception as e:
            print(f"Error listing points configs: {e}")
            return []

    def set_selected_points_config(self, config_id: str) -> bool:
        """Set a points config as selected."""
        self._safe_select(self.POINTS_COLLECTION, config_id)
        return True
    
    def get_config(self, config_id: str = None, config_hash: str = None, selected: bool = False) -> Optional[Dict]:
        """
        Get model configuration by ID, hash, or selected flag.

        Args:
            config_id: MongoDB _id as string
            config_hash: Config hash for unique identification
            selected: Get the currently selected config

        Returns:
            Configuration dict or None if not found
        """
        try:
            if config_id:
                return self._classifier_repo.find_by_id(config_id)
            elif config_hash:
                return self._classifier_repo.find_by_hash(config_hash)
            elif selected:
                return self._classifier_repo.find_selected()
            else:
                raise ValueError("Must specify config_id, config_hash, or selected=True")
        except Exception as e:
            print(f"Error getting config: {e}")
            return None
    
    def save_config(self, config: Dict) -> str:
        """
        Save or update model configuration.

        Args:
            config: Configuration dictionary

        Returns:
            MongoDB document _id as string
        """
        try:
            # Generate config hash if not provided
            if 'config_hash' not in config:
                config['config_hash'] = self._generate_config_hash(config)

            # Add timestamps
            config['updated_at'] = datetime.utcnow()
            if 'created_at' not in config:
                config['created_at'] = datetime.utcnow()

            # Upsert using config_hash as unique identifier
            self._classifier_repo.upsert_config(config['config_hash'], config)

            # Get document ID
            doc = self._classifier_repo.find_by_hash(config['config_hash'])
            config_id = str(doc['_id'])

            print(f"✅ Saved config {config_id[:8]} with hash {config['config_hash'][:8]}")
            return config_id

        except Exception as e:
            print(f"Error saving config: {e}")
            raise
    
    def set_selected_config(self, config_id: str) -> bool:
        """
        Set a configuration as selected (unselect all others).

        Args:
            config_id: MongoDB _id as string

        Returns:
            True if successful
        """
        try:
            success = self._classifier_repo.set_selected_by_id(config_id)

            if success:
                print(f"✅ Selected config {config_id[:8]}")
            else:
                print(f"❌ Config {config_id[:8]} not found")

            return success

        except Exception as e:
            print(f"Error selecting config: {e}")
            return False
    
    def get_selected_config(self) -> Optional[Dict]:
        """Get the currently selected configuration."""
        return self.get_config(selected=True)

    @staticmethod
    def validate_config_for_prediction(config: Dict, check_file_exists: bool = True) -> tuple:
        """
        Validate that a model config is trained and ready for predictions.

        This is the single source of truth for validation logic used by:
        - Web app prediction endpoints
        - Agent prediction tools
        - Any other prediction interface

        Args:
            config: Model configuration dictionary
            check_file_exists: Whether to verify training_csv file exists on disk

        Returns:
            Tuple of (is_valid: bool, error_message: Optional[str])
            If valid, error_message is None.
        """
        if not config:
            return False, "No model config provided."

        config_name = config.get('name', 'Unnamed')
        is_ensemble = bool(config.get('ensemble', False))

        if is_ensemble:
            # Ensemble models are considered trained when they have an ensemble_run_id
            ensemble_run_id = config.get('ensemble_run_id')
            if not ensemble_run_id:
                return False, (
                    f'The selected ensemble model config "{config_name}" has not been trained yet. '
                    f'Please train the ensemble meta-model first.'
                )
        else:
            # Regular models are considered trained when they have a training_csv path
            training_csv = config.get('training_csv')
            if not training_csv:
                return False, (
                    f'The selected model config "{config_name}" has not been trained yet. '
                    f'Please train the model first.'
                )
            if check_file_exists and not os.path.exists(training_csv):
                return False, (
                    f'The selected model config "{config_name}" training data file not found at: {training_csv}. '
                    f'The file may have been deleted or moved. Please retrain the model.'
                )

        return True, None
    
    def list_configs(self, model_type: str = None, ensemble: bool = None) -> List[Dict]:
        """
        List configurations with optional filtering.

        Args:
            model_type: Filter by model type
            ensemble: Filter by ensemble flag

        Returns:
            List of configuration dictionaries
        """
        try:
            if model_type:
                configs = self._classifier_repo.find_by_model_type(model_type)
            elif ensemble is not None:
                configs = self._classifier_repo.find_ensembles() if ensemble else self._classifier_repo.find_all()
                if not ensemble:
                    configs = [c for c in configs if not c.get('ensemble')]
            else:
                configs = self._classifier_repo.find_all()

            # Convert ObjectIds to strings for JSON serialization
            for config in configs:
                if '_id' in config:
                    config['_id'] = str(config['_id'])

            return configs

        except Exception as e:
            print(f"Error listing configs: {e}")
            return []
    
    def delete_config(self, config_id: str) -> bool:
        """
        Delete a configuration.

        Args:
            config_id: MongoDB _id as string

        Returns:
            True if successful
        """
        try:
            # Get config hash first for deletion
            config = self._classifier_repo.find_by_id(config_id)
            if not config:
                print(f"❌ Config {config_id[:8]} not found")
                return False

            config_hash = config.get('config_hash')
            if config_hash:
                success = self._classifier_repo.delete_config(config_hash)
            else:
                # Fallback to direct delete
                result = self._classifier_repo.delete_one({'_id': ObjectId(config_id)})
                success = result.deleted_count > 0

            if success:
                print(f"✅ Deleted config {config_id[:8]}")
            else:
                print(f"❌ Config {config_id[:8]} not found")

            return success

        except Exception as e:
            print(f"Error deleting config: {e}")
            return False
    
    def _generate_config_hash(self, config: Dict) -> str:
        """
        Generate unique hash for configuration identification.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            MD5 hash string
        """
        # Use key fields for hash generation
        hash_fields = {
            'model_type': config.get('model_type'),
            'feature_set_hash': config.get('feature_set_hash'),
            'best_c_value': config.get('best_c_value'),
            'use_time_calibration': config.get('use_time_calibration'),
            'calibration_method': config.get('calibration_method'),
            'calibration_years': config.get('calibration_years'),
            'begin_year': config.get('begin_year'),
            'evaluation_year': config.get('evaluation_year'),
            'include_injuries': config.get('include_injuries'),
            'recency_decay_k': config.get('recency_decay_k'),
            'use_master': config.get('use_master'),
            'min_games_played': config.get('min_games_played')
        }
        
        # Create deterministic string representation
        hash_str = '|'.join(f"{k}:{v}" for k, v in sorted(hash_fields.items()) if v is not None)
        
        # Generate MD5 hash
        return hashlib.md5(hash_str.encode()).hexdigest()
    
    @staticmethod
    def create_from_request(request_data: Dict) -> Dict:
        """
        Create configuration dictionary from web request data.
        
        Args:
            request_data: JSON data from web request
            
        Returns:
            Configuration dictionary
        """
        # Extract and validate fields from request
        config = {
            'model_type': request_data.get('model_type'),
            'features': request_data.get('features', []),
            'use_time_calibration': request_data.get('use_time_calibration', False),
            'calibration_method': request_data.get('calibration_method'),
            'begin_year': request_data.get('begin_year'),
            'calibration_years': request_data.get('calibration_years', []),
            'evaluation_year': request_data.get('evaluation_year'),
            'include_injuries': request_data.get('include_injuries', False),
            'recency_decay_k': request_data.get('recency_decay_k'),
            'use_master': request_data.get('use_master', True),
            'min_games_played': request_data.get('min_games_played', 15),
            'ensemble': request_data.get('ensemble', False),
            'ensemble_models': request_data.get('ensemble_models', []),
            'ensemble_type': request_data.get('ensemble_type'),
            'ensemble_meta_features': request_data.get('ensemble_meta_features', []),
            'ensemble_use_disagree': request_data.get('ensemble_use_disagree', False),
            'ensemble_use_conf': request_data.get('ensemble_use_conf', False)
        }
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
