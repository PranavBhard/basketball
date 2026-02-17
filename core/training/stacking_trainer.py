"""
Stacking Tool - Trains meta-models that combine predictions from multiple base models
"""

import sys
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss, roc_auc_score
from bson import ObjectId

# Supported meta-model types
META_MODEL_TYPES = ['LogisticRegression', 'SVM', 'GradientBoosting']
C_SUPPORTED_META_MODELS = ['LogisticRegression', 'SVM']

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bball_app.core.training.experiment_runner import ExperimentRunner
from bball_app.core.training.dataset_builder import DatasetBuilder
from bball_app.core.training.run_tracker import RunTracker
from bball_app.core.data import ClassifierConfigRepository
from bball_app.core.training.cache_utils import read_csv_safe
from bball_app.core.training.schemas import ExperimentConfig


class StackingTrainer:
    """Trains stacked models that combine multiple base model predictions"""

    def __init__(self, db=None, league=None):
        """
        Initialize StackingTrainer.

        Args:
            db: MongoDB database instance (optional)
            league: LeagueConfig instance for league-specific collections
        """
        if db is None:
            from bball_app.core.mongo import Mongo
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league

        # Initialize repository with league awareness
        self._classifier_repo = ClassifierConfigRepository(self.db, league=league)

        self.experiment_runner = ExperimentRunner(db=self.db, league=league)
        self.dataset_builder = DatasetBuilder(db=self.db, league=league)
        self.run_tracker = RunTracker(db=self.db, league=league)
    
    def train_stacked_model(
        self,
        dataset_spec: Dict,
        session_id: str,
        base_run_ids: List[str] = None,
        base_config_ids: List[str] = None,
        meta_model_type: str = 'LogisticRegression',
        meta_c_value: float = 0.1,
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False
    ) -> Dict:
        """
        Train a stacked model that combines multiple base model predictions.
        
        Args:
            base_run_ids: List of run_ids for base models (legacy support)
            base_config_ids: List of MongoDB config _id strings for base models (preferred)
            dataset_spec: Dataset specification (must match base models' configs)
            session_id: Chat session ID
            meta_model_type: Type of meta-model to train
            meta_c_value: C-value for meta-model (if applicable)
            stacking_mode: 'naive' (default) or 'informed'. Informed allows adding derived and/or user features.
            meta_features: Optional list of feature names from dataset to include in meta-model (only used when stacking_mode='informed')
            use_disagree: If True, include pairwise disagreement features (only used when stacking_mode='informed')
            use_conf: If True, include confidence features for each model (only used when stacking_mode='informed')

        Returns:
            Dict with:
                - run_id: Stacking run identifier
                - metrics: Evaluation metrics
                - diagnostics: Model diagnostics
                - artifacts: Paths to saved artifacts
        """
        # Use base_config_ids if provided, otherwise fall back to base_run_ids
        if base_config_ids is not None:
            base_ids = base_config_ids
        else:
            base_ids = base_run_ids
            
        if len(base_ids) < 2:
            raise ValueError(f"Stacking requires at least 2 base models, got {len(base_ids)}")

        # Validate meta_model_type
        if meta_model_type not in META_MODEL_TYPES:
            raise ValueError(f"Invalid meta_model_type: {meta_model_type}. Must be one of {META_MODEL_TYPES}")

        # Validate stacking_mode
        if stacking_mode not in ['naive', 'informed']:
            raise ValueError(f"Invalid stacking_mode: {stacking_mode}. Must be 'naive' or 'informed'")
        
        # Load and validate base models
        base_models_info = self._load_base_models(base_ids)
        
        # Validate all base models have compatible configs
        self._validate_base_models_compatible(base_models_info)
        
        # Get reference config from first base model
        ref_config = base_models_info[0]['config']
        ref_splits = ref_config.get('splits', {})
        
        # Extract time-based calibration parameters
        calibration_years = ref_splits.get('calibration_years', [2023])
        evaluation_year = ref_splits.get('evaluation_year', 2024)
        begin_year = ref_splits.get('begin_year', 2012)
        
        # Ensure calibration_years is a list
        if not isinstance(calibration_years, list):
            calibration_years = [calibration_years]
        
        # Create stacking run
        stacking_config = {
            'task': 'stacking',
            'base_run_ids': base_run_ids,
            'meta_model_type': meta_model_type,
            'meta_c_value': meta_c_value if meta_model_type in C_SUPPORTED_META_MODELS else None,
            'stacking_mode': stacking_mode,
            'meta_features': meta_features,
            'use_disagree': use_disagree,
            'use_conf': use_conf,
            'splits': ref_splits,
            'features': ref_config.get('features', {})
        }
        
        run_id = self.run_tracker.create_run(
            config=stacking_config,
            dataset_id=None,  # Stacking doesn't use a single dataset_id
            model_type='Stacked',
            session_id=session_id,
            baseline=False
        )
        
        self.run_tracker.update_run(run_id, status='running')
        
        try:
            # Rule 4: Validate OOF-only dataset construction
            # Ensure we're not training meta-model on in-sample base predictions
            print(f"[STACKING] Rule 4: Validating OOF-only dataset construction")
            print(f"[STACKING] Using dataset_spec: {dataset_spec}")
            
            # Extract time configuration for validation
            begin_year = dataset_spec.get('begin_year')
            calibration_years = dataset_spec.get('calibration_years', [])
            evaluation_year = dataset_spec.get('evaluation_year')
            min_games_played = dataset_spec.get('min_games_played', 0)
            
            if not all([begin_year, calibration_years, evaluation_year]):
                raise ValueError("Rule 4 violation: Missing required time configuration (begin_year, calibration_years, evaluation_year)")
            
            print(f"[STACKING] Time config - Begin: {begin_year}, Calibration: {calibration_years}, Evaluation: {evaluation_year}")
            print(f"[STACKING] Min games played: {min_games_played}")
            
            # Rule 4: Ensure meta-model will be trained on OOF predictions only
            # We'll validate this in _generate_stacking_data method
            print(f"[STACKING] Rule 4: Meta-model will use only OOF base predictions")

            # Collect ALL unique features needed by ALL base models
            # This ensures the dataset includes features for all base models (including injury features, etc.)
            all_base_features = set()
            for model_info in base_models_info:
                model_features = model_info.get('feature_names', [])
                all_base_features.update(model_features)

            # Also include meta_features if provided (for informed stacking)
            if meta_features:
                all_base_features.update(meta_features)

            print(f"[STACKING] Collected {len(all_base_features)} unique features from {len(base_models_info)} base models")

            # Build dataset for calibration and evaluation periods
            # We need data from calibration_years and evaluation_year
            # Remove fields that aren't part of DatasetSpec (calibration_years, evaluation_year, use_master, etc.)
            # These are used for filtering after the dataset is built
            # Ensure begin_year is included from base model config
            excluded_keys = ['calibration_years', 'evaluation_year', 'use_master', 'training_csv',
                            'model_type', 'best_c_value', 'config_hash', 'feature_set_hash']
            dataset_spec_clean = {k: v for k, v in dataset_spec.items()
                                 if k not in excluded_keys}

            # Add all base model features to the dataset request
            dataset_spec_clean['individual_features'] = list(all_base_features)
            
            # Ensure begin_year is set (use from base model config if not in dataset_spec)
            if 'begin_year' not in dataset_spec_clean:
                dataset_spec_clean['begin_year'] = begin_year
            print(f"[STACKING] Building dataset with begin_year={dataset_spec_clean.get('begin_year')}, calibration_years={calibration_years}, evaluation_year={evaluation_year}")
            dataset_result = self.dataset_builder.build_dataset(dataset_spec_clean)
            csv_path = dataset_result['csv_path']
            
            # Load dataset
            df = read_csv_safe(csv_path)
            if df.empty:
                raise ValueError(f"Dataset is empty: {csv_path}")
            
            # Calculate SeasonStartYear for filtering
            df['SeasonStartYear'] = np.where(df['Month'] >= 10, df['Year'], df['Year'] - 1)
            
            # Rule 5: Proper temporal split for meta-model training/evaluation
            # Train period: begin_year ... (min(calibration_years)-1)
            # Calibration period: calibration_years
            # Evaluation period: evaluation_year
            # Meta-model should be trained on TRAIN + CALIBRATION years using OOF predictions
            # Meta-model should be evaluated only on evaluation_year
            
            # Filter to calibration and evaluation periods
            cal_mask = df['SeasonStartYear'].isin(calibration_years)
            eval_mask = df['SeasonStartYear'] == evaluation_year
            
            # Rule 5: Train period = data before calibration period
            train_mask = df['SeasonStartYear'] < min(calibration_years) if calibration_years else df['SeasonStartYear'] < evaluation_year
            
            df_cal = df[cal_mask].copy()
            df_eval = df[eval_mask].copy()
            df_train = df[train_mask].copy()

            if len(df_cal) == 0:
                raise ValueError(f"No data found for calibration years {calibration_years}")
            if len(df_eval) == 0:
                raise ValueError(f"No data found for evaluation year {evaluation_year}")

            print(f"[STACKING] Temporal split:")
            print(f"[STACKING]   Base model train period: {begin_year} to {min(calibration_years)-1 if calibration_years else evaluation_year-1} ({len(df_train)} games)")
            print(f"[STACKING]   Meta-model training (calibration only): {calibration_years} ({len(df_cal)} games)")
            print(f"[STACKING]   Evaluation period: {evaluation_year} ({len(df_eval)} games)")

            # Meta-model trains ONLY on calibration years (base models' OOF predictions)
            # Do NOT reuse base model training rows - those would be in-sample predictions
            stacking_df = self._generate_stacking_data(
                base_models_info=base_models_info,
                df=df_cal,  # Use ONLY calibration years for meta-model training
                calibration_years=calibration_years,
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf
            )
            
            # Train meta-model on calibration predictions
            meta_model, meta_scaler = self._train_meta_model(
                stacking_df=stacking_df,
                meta_model_type=meta_model_type,
                meta_c_value=meta_c_value
            )

            # Evaluate stacked model on evaluation period
            metrics, diagnostics = self._evaluate_stacked_model(
                meta_model=meta_model,
                meta_scaler=meta_scaler,
                base_models_info=base_models_info,
                df=df_eval,
                evaluation_year=evaluation_year,
                calibration_years=calibration_years,
                begin_year=begin_year,
                n_train_samples=len(df_train),  # Base models' training period (for diagnostics)
                n_cal_samples=len(df_cal),      # Meta-model training period (calibration years only)
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf
            )
            
            # Save ensemble artifacts to disk for later loading
            meta_feature_cols = [c for c in stacking_df.columns if c != 'HomeWon']
            artifact_paths = self._save_ensemble_artifacts(
                run_id=run_id,
                meta_model=meta_model,
                meta_scaler=meta_scaler,
                base_model_ids=base_ids,
                meta_feature_cols=meta_feature_cols,
                meta_model_type=meta_model_type,
                meta_c_value=meta_c_value,
                stacking_mode=stacking_mode,
                meta_features=meta_features,
                use_disagree=use_disagree,
                use_conf=use_conf
            )

            # Prepare artifacts with file paths
            artifacts = {
                'dataset_path': csv_path,
                'base_ids': base_ids,  # Use the actual IDs used (config_ids or run_ids)
                'meta_model_type': meta_model_type,
                **artifact_paths  # Include saved artifact paths
            }

            # Update run with results
            self.run_tracker.update_run(
                run_id=run_id,
                metrics=metrics,
                diagnostics=diagnostics,
                artifacts=artifacts,
                status='completed'
            )
            
            return {
                'run_id': run_id,
                'metrics': metrics,
                'diagnostics': diagnostics,
                'artifacts': artifacts
            }
        
        except Exception as e:
            self.run_tracker.update_run(
                run_id=run_id,
                status='failed',
                diagnostics={'error': str(e)}
            )
            raise
    
    def _load_base_models(self, base_ids: List[str]) -> List[Dict]:
        """
        Load all base models and their metadata.
        Supports both run_ids (legacy) and MongoDB config _ids (preferred).
        
        Args:
            base_ids: List of identifiers (run_ids or MongoDB config _ids)
            
        Returns:
            List of dicts, each containing:
                - run_id: Identifier (run_id or config_id)
                - model: Trained sklearn model
                - scaler: Fitted StandardScaler (or None)
                - feature_names: List of feature names
                - config: Run configuration
        """
        base_models_info = []
        
        for base_id in base_ids:
            # Try to load as MongoDB config first (preferred)
            try:
                print(f"[STACKING] Loading base model {base_id} from MongoDB config...")

                # Check if base_id is a valid ObjectId
                try:
                    obj_id = ObjectId(base_id)
                except Exception as e:
                    print(f"[STACKING] Invalid ObjectId format for {base_id}: {e}")
                    raise ValueError(f"Base model {base_id} is not a valid MongoDB ObjectId")

                config = self._classifier_repo.find_one({'_id': obj_id})
                if config:
                    print(f"[STACKING] ✅ Found MongoDB config for {base_id}")
                    # Load model from MongoDB config
                    model, scaler, feature_names = self._load_model_from_config(config)
                    base_models_info.append({
                        'run_id': base_id,  # Store config_id for consistency
                        'model': model,
                        'scaler': scaler,
                        'feature_names': feature_names,
                        'config': config
                    })
                    continue
                else:
                    print(f"[STACKING] ❌ No MongoDB config found for {base_id}")
            except Exception as e:
                print(f"[STACKING] Error loading MongoDB config for {base_id}: {e}")
                pass  # Fall back to run_id loading
            
            # Fall back to run_id loading (legacy support)
            run = self.run_tracker.get_run(base_id)
            if not run:
                raise ValueError(f"Base model {base_id} not found as run_id or config_id")
            
            # Load model artifacts
            try:
                model, scaler, feature_names = self.experiment_runner._load_classification_model(base_id)
            except FileNotFoundError as e:
                raise ValueError(
                    f"Base model {base_id} does not have saved model artifacts. "
                    f"Only models trained after adding model persistence support can be used for stacking. "
                    f"Error: {e}"
                )
            
            base_models_info.append({
                'run_id': base_id,
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'config': run.get('config', {})
            })
        
        return base_models_info
    
    def _load_model_from_config(self, config: dict):
        """
        Load model from MongoDB config document.
        Prioritizes saved artifacts for fast loading, falls back to retraining if needed.
        
        Args:
            config: MongoDB config document
            
        Returns:
            Tuple of (model, scaler, feature_names)
        """
        import os
        import pickle
        import json
        from sklearn.preprocessing import StandardScaler
        
        # Priority 1: Try to load saved artifacts (fast path)
        model_artifact_path = config.get('model_artifact_path')
        scaler_artifact_path = config.get('scaler_artifact_path')
        features_path = config.get('features_path')
        
        if model_artifact_path and scaler_artifact_path and features_path:
            if os.path.exists(model_artifact_path) and os.path.exists(scaler_artifact_path) and os.path.exists(features_path):
                try:
                    print(f"[STACKING] Loading saved artifacts for model...")
                    
                    # Load model artifact
                    with open(model_artifact_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Load scaler artifact
                    with open(scaler_artifact_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    # Load feature names
                    with open(features_path, 'r') as f:
                        feature_names = json.load(f)
                    
                    print(f"[STACKING] ✅ Successfully loaded saved artifacts")
                    return model, scaler, feature_names
                    
                except Exception as e:
                    print(f"[STACKING] ❌ Error loading saved artifacts: {e}")
                    print(f"[STACKING] Will fall back to retraining from data...")
            else:
                print(f"[STACKING] ⚠️  Expected artifacts not found:")
                print(f"[STACKING]   Model: {model_artifact_path} {'✅' if os.path.exists(model_artifact_path) else '❌'}")
                print(f"[STACKING]   Scaler: {scaler_artifact_path} {'✅' if os.path.exists(scaler_artifact_path) else '❌'}")
                print(f"[STACKING]   Features: {features_path} {'✅' if os.path.exists(features_path) else '❌'}")
                print(f"[STACKING] Will fall back to retraining from data...")
        
        # Priority 2: Fallback to retraining from training data
        print(f"[STACKING] Retraining model from training data...")
        training_csv = config.get('training_csv')
        if not training_csv or not os.path.exists(training_csv):
            raise FileNotFoundError(
                f"Cannot load model: No saved artifacts found and training CSV not found: {training_csv}\n"
                f"Base models must be trained with model persistence support to be used in ensembles.\n"
                f"Please retrain the base models with the current system version."
            )
        
        # Load and prepare data
        import pandas as pd
        df = pd.read_csv(training_csv)
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id', 'home_points', 'away_points']
        feature_cols = [c for c in df.columns if c not in meta_cols]

        # Coerce features to numeric (object dtype will break np.isnan)
        X_df = df[feature_cols].apply(pd.to_numeric, errors='coerce')

        # Drop feature columns that are entirely non-numeric
        all_nan_cols = [c for c in X_df.columns if X_df[c].isna().all()]
        if all_nan_cols:
            print(f"[STACKING] Dropping non-numeric columns: {all_nan_cols}")
            X_df = X_df.drop(columns=all_nan_cols)
            feature_cols = [c for c in feature_cols if c not in all_nan_cols]

        # Coerce y to numeric as well
        y_series = pd.to_numeric(df['HomeWon'], errors='coerce')

        X = X_df.to_numpy(dtype=float)
        y = y_series.to_numpy()

        # Handle NaN values
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(y)
        if nan_mask.sum() > 0:
            print(f"[STACKING] Dropping {nan_mask.sum()} rows with NaN values")
            # Also filter df so temporal mask stays aligned
            df = df[~nan_mask].reset_index(drop=True)
            X = X[~nan_mask]
            y = y[~nan_mask]

        # Apply temporal split: train only on pre-calibration data to prevent
        # data leakage (calibration rows must be out-of-sample for stacking).
        calibration_years = config.get('calibration_years', [])
        if not calibration_years:
            # Try nested splits config
            splits_cfg = config.get('splits', {})
            calibration_years = splits_cfg.get('calibration_years', [])

        if calibration_years:
            df_tmp = df.copy()
            df_tmp['SeasonStartYear'] = np.where(
                df_tmp['Month'] >= 10, df_tmp['Year'], df_tmp['Year'] - 1
            )
            temporal_train_mask = (df_tmp['SeasonStartYear'] < min(calibration_years)).values
            n_before = len(X)
            X = X[temporal_train_mask]
            y = y[temporal_train_mask]
            print(f"[STACKING] Temporal split for retrain: {n_before} -> {len(X)} rows (pre-calibration only)")

        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Create model
        model_type = config.get('model_type', 'LogisticRegression')
        if model_type == 'LogisticRegression':
            from sklearn.linear_model import LogisticRegression
            c_value = config.get('best_c_value', 0.1)
            model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
        elif model_type == 'SVM':
            from sklearn.svm import SVC
            c_value = config.get('best_c_value', 0.1)
            model = SVC(C=c_value, probability=True, random_state=42)
        elif model_type == 'GradientBoosting':
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(random_state=42)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        model.fit(X_scaled, y)
        
        return model, scaler, feature_cols

    def _save_ensemble_artifacts(
        self,
        run_id: str,
        meta_model,
        meta_scaler,
        base_model_ids: List[str],
        meta_feature_cols: List[str],
        meta_model_type: str,
        meta_c_value: float,
        stacking_mode: str,
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False
    ) -> Dict[str, str]:
        """
        Save ensemble model artifacts to disk for later loading.

        Args:
            run_id: Unique run identifier
            meta_model: Trained meta-model
            base_model_ids: List of base model config IDs
            meta_feature_cols: Feature columns for meta-model input
            meta_model_type: Type of meta-model
            meta_c_value: C-value used for meta-model
            stacking_mode: Stacking mode ('naive' or 'informed')
            meta_features: Additional meta-features used
            use_disagree: Whether disagreement feature was used
            use_conf: Whether confidence features were used

        Returns:
            Dict with artifact file paths
        """
        import pickle
        import json

        # Create ensemble models directory
        ensemble_dir = 'cli/models/ensembles'
        os.makedirs(ensemble_dir, exist_ok=True)

        # Generate file paths
        model_path = os.path.join(ensemble_dir, f'{run_id}_meta_model.pkl')
        scaler_path = os.path.join(ensemble_dir, f'{run_id}_meta_scaler.pkl')
        config_path = os.path.join(ensemble_dir, f'{run_id}_ensemble_config.json')

        try:
            # Save meta-model
            with open(model_path, 'wb') as f:
                pickle.dump(meta_model, f)
            print(f"[STACKING] ✅ Saved meta-model: {model_path}")

            # Save meta-scaler
            if meta_scaler is not None:
                with open(scaler_path, 'wb') as f:
                    pickle.dump(meta_scaler, f)
                print(f"[STACKING] ✅ Saved meta-scaler: {scaler_path}")

            # Save ensemble configuration (everything needed to reload and use the ensemble)
            ensemble_config = {
                'run_id': run_id,
                'base_model_ids': base_model_ids,
                'meta_feature_cols': meta_feature_cols,
                'meta_model_type': meta_model_type,
                'meta_c_value': meta_c_value,
                'stacking_mode': stacking_mode,
                'meta_features': meta_features or [],
                'use_disagree': use_disagree,
                'use_conf': use_conf
            }
            with open(config_path, 'w') as f:
                json.dump(ensemble_config, f, indent=2)
            print(f"[STACKING] ✅ Saved ensemble config: {config_path}")

            return {
                'meta_model_path': model_path,
                'meta_scaler_path': scaler_path,
                'ensemble_config_path': config_path,
                'meta_model_type': meta_model_type,
                'meta_c_value': meta_c_value
            }

        except Exception as e:
            print(f"[STACKING] ❌ Error saving ensemble artifacts: {e}")
            return {}

    def _validate_base_models_compatible(self, base_models_info: List[Dict]):
        """
        Validate that all base models have compatible configurations.
        
        Checks:
        - Same time-based calibration config (begin_year, calibration_years, evaluation_year)
        
        Note: Feature sets can differ between models - each model will use its own feature set
        when generating predictions. This allows stacking models trained on different features.
        
        Args:
            base_models_info: List of base model info dicts
            
        Raises:
            ValueError if models are incompatible
        """
        if len(base_models_info) < 2:
            return
        
        # Get reference config from first model
        ref_config = base_models_info[0]['config']
        ref_splits = ref_config.get('splits', {})
        ref_features = ref_config.get('features', {})
        ref_feature_names = set(base_models_info[0]['feature_names'])
        
        # Check each subsequent model
        for i, model_info in enumerate(base_models_info[1:], 1):
            config = model_info['config']
            splits = config.get('splits', {})
            features = config.get('features', {})
            feature_names = set(model_info['feature_names'])
            
            # Check time-based calibration config
            if splits.get('begin_year') != ref_splits.get('begin_year'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible begin_year. "
                    f"Expected {ref_splits.get('begin_year')}, got {splits.get('begin_year')}"
                )
            
            if splits.get('calibration_years') != ref_splits.get('calibration_years'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible calibration_years. "
                    f"Expected {ref_splits.get('calibration_years')}, got {splits.get('calibration_years')}"
                )
            
            if splits.get('evaluation_year') != ref_splits.get('evaluation_year'):
                raise ValueError(
                    f"Base model {model_info['run_id']} has incompatible evaluation_year. "
                    f"Expected {ref_splits.get('evaluation_year')}, got {splits.get('evaluation_year')}"
                )
            
            # Note: Feature sets can differ between models - this is allowed for stacking
            # Each model will use its own feature set when generating predictions
    
    def _generate_stacking_data(
        self,
        base_models_info: List[Dict],
        df: pd.DataFrame,
        calibration_years: List[int],
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False
    ) -> pd.DataFrame:
        """
        Generate stacking training data using base model predictions on calibration period.

        For each game in df:
        - Extract features for each base model (using each model's own feature set)
        - Get predictions from each base model (probability of home win)
        - For naive stacking: Create stacking training row: [p_model1, p_model2, ..., p_modelN, true_label]
        - For informed stacking: Add derived features (disagree, conf) and/or user features as requested

        Args:
            base_models_info: List of base model info dicts
            df: DataFrame with calibration period games
            calibration_years: List of calibration years (for logging)
            stacking_mode: 'naive' or 'informed'
            meta_features: Optional list of feature names from dataset to include (only used when stacking_mode='informed')
            use_disagree: If True, add pairwise disagreement features (only when stacking_mode='informed')
            use_conf: If True, add confidence features (only when stacking_mode='informed')

        Returns:
            DataFrame with columns:
            - Naive: [p_model1, p_model2, ..., p_modelN, HomeWon]
            - Informed: [p_model1, p_model2, ..., p_modelN, <disagree_* if requested>, <conf_* if requested>, <user_feats if provided>, HomeWon]
        """
        # Extract metadata and target columns
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'SeasonStartYear']
        target_cols = ['HomeWon']
        excluded_cols = meta_cols + target_cols
        # Exclude prediction columns EXCEPT those explicitly requested as meta-features
        pred_cols = ['pred_home_points', 'pred_away_points', 'pred_point_total']
        # Only exclude pred_margin if NOT explicitly requested as meta-feature
        if not meta_features or 'pred_margin' not in meta_features:
            pred_cols.append('pred_margin')
        excluded_cols.extend([c for c in pred_cols if c in df.columns])
        
        # Get all available features in dataset
        available_features = set([c for c in df.columns if c not in excluded_cols])
        
        # Get predictions from each base model (each uses its own feature set)
        stacking_data = {}
        used_model_names = set()  # Track used names to avoid collisions

        for i, model_info in enumerate(base_models_info):
            model = model_info['model']
            scaler = model_info['scaler']
            model_feature_names = model_info['feature_names']
            
            # Check which features are available and which are missing
            missing_features = [f for f in model_feature_names if f not in available_features]
            available_model_features = [f for f in model_feature_names if f in available_features]
            
            if len(available_model_features) == 0:
                raise ValueError(
                    f"Base model {model_info['run_id']} requires features that are not in the dataset. "
                    f"Missing all {len(model_feature_names)} features. "
                    f"Model was trained with: {model_feature_names[:5]}... "
                )
            
            if len(missing_features) > 0:
                print(
                    f"Warning: Base model {model_info['run_id']} is missing {len(missing_features)}/{len(model_feature_names)} features "
                    f"in dataset: {sorted(missing_features)[:10]}... "
                    f"Missing features will be set to 0.0 for prediction."
                )
            
            # Build feature matrix with ALL features the model was trained on, in the same order
            # Fill missing features with 0.0 (safe default for most models)
            X = np.zeros((len(df), len(model_feature_names)))
            for idx, feature_name in enumerate(model_feature_names):
                if feature_name in available_features:
                    X[:, idx] = df[feature_name].values
                else:
                    # Missing feature - fill with 0.0
                    X[:, idx] = 0.0

            # Handle NaN and Inf values - replace with 0.0 (safe default)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Scale features if scaler exists
            if scaler is not None:
                # Now X has the correct shape (all features in training order)
                try:
                    X_scaled = scaler.transform(X)
                except Exception as e:
                    raise ValueError(
                        f"Cannot scale features for model {model_info['run_id']}. "
                        f"Scaler expects {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'unknown'} features "
                        f"but got {X.shape[1]}. "
                        f"Error: {e}"
                    )
            else:
                X_scaled = X
            
            # Get predictions (probability of home win)
            y_proba = model.predict_proba(X_scaled)
            p_home_win = y_proba[:, 1]  # Probability of home win
            
            # Store predictions with model identifier
            # Prefer config 'name' field if available, otherwise use shortened run_id
            config = model_info.get('config', {})
            model_name = config.get('name')
            if model_name:
                # Sanitize name for use as column name (replace spaces/special chars with underscore)
                model_id_short = re.sub(r'[^a-zA-Z0-9_]', '_', model_name)
            else:
                # Fallback to first 8 chars of run_id (e.g., 'p_550e8400')
                model_id = model_info['run_id']
                model_id_short = model_id[:8] if len(model_id) > 8 else model_id

            # Handle name collisions by appending a number
            base_name = model_id_short
            counter = 1
            while model_id_short in used_model_names:
                model_id_short = f"{base_name}_{counter}"
                counter += 1
            used_model_names.add(model_id_short)

            stacking_data[f'p_{model_id_short}'] = p_home_win
        
        # For informed stacking, add derived features if requested
        print(f"[STACKING] _generate_stacking_data: stacking_mode={stacking_mode}, meta_features={meta_features}, use_disagree={use_disagree}, use_conf={use_conf}")
        if stacking_mode == 'informed':
            # Get prediction column names (all columns starting with 'p_')
            pred_cols = [col for col in stacking_data.keys() if col.startswith('p_')]
            n_models = len(pred_cols)

            # Calculate pairwise disagreements for all unique pairs (i < j) - only if requested
            if use_disagree:
                for i in range(n_models):
                    for j in range(i + 1, n_models):
                        col_i = pred_cols[i]
                        col_j = pred_cols[j]
                        # Extract model IDs from column names (e.g., 'p_550e8400' -> '550e8400')
                        id_i = col_i.replace('p_', '')
                        id_j = col_j.replace('p_', '')
                        disagree_name = f'disagree_{id_i}_{id_j}'
                        stacking_data[disagree_name] = np.abs(stacking_data[col_i] - stacking_data[col_j])

            # Calculate confidence for each model (distance from 0.5) - only if requested
            if use_conf:
                for col in pred_cols:
                    model_id = col.replace('p_', '')
                    conf_name = f'conf_{model_id}'
                    stacking_data[conf_name] = np.abs(stacking_data[col] - 0.5)
            
            # Add user-provided features if specified
            if meta_features:
                print(f"[STACKING] Adding user meta-features: {meta_features}")
                available_features = set(df.columns)
                meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'SeasonStartYear']
                target_cols = ['HomeWon']
                # Exclude prediction columns EXCEPT those explicitly requested as meta-features
                pred_cols_exclude = ['pred_home_points', 'pred_away_points', 'pred_point_total']
                # Only exclude pred_margin if NOT explicitly requested as meta-feature
                if 'pred_margin' not in meta_features:
                    pred_cols_exclude.append('pred_margin')
                excluded_cols = set(meta_cols + target_cols + [c for c in pred_cols_exclude if c in df.columns])
                print(f"[STACKING] pred_margin in dataset columns: {'pred_margin' in available_features}")
                print(f"[STACKING] pred_margin excluded: {'pred_margin' in excluded_cols}")

                for feat_name in meta_features:
                    if feat_name in excluded_cols:
                        print(f"Warning: Meta-feature '{feat_name}' is in excluded columns (metadata/target/predictions). Skipping.")
                        continue
                    if feat_name in available_features:
                        stacking_data[feat_name] = df[feat_name].values
                    else:
                        print(f"Warning: Meta-feature '{feat_name}' not found in dataset. Skipping.")
        
        # Add true labels
        stacking_data['HomeWon'] = df['HomeWon'].values

        # Create DataFrame
        stacking_df = pd.DataFrame(stacking_data)

        # Handle NaN values in stacking DataFrame
        # Replace NaN with 0.0 for all columns except HomeWon
        feature_cols = [c for c in stacking_df.columns if c != 'HomeWon']
        stacking_df[feature_cols] = stacking_df[feature_cols].fillna(0.0)

        # Drop rows where HomeWon is NaN (can't train without labels)
        stacking_df = stacking_df.dropna(subset=['HomeWon'])

        return stacking_df
    
    def _train_meta_model(
        self,
        stacking_df: pd.DataFrame,
        meta_model_type: str = 'LogisticRegression',
        meta_c_value: float = 0.1
    ):
        """
        Train meta-model on stacking training data.

        Applies StandardScaler so regularization treats all features equally
        regardless of scale (e.g., travel distance in miles vs probabilities in [0,1]).

        Args:
            stacking_df: DataFrame with base model predictions and true labels
            meta_model_type: Type of model ('LogisticRegression', 'SVM', 'GradientBoosting')
            meta_c_value: C-value (only used for LogisticRegression and SVM)

        Returns:
            Tuple of (trained meta-model, fitted StandardScaler)
        """
        # Extract feature columns (all columns except HomeWon)
        feature_cols = [c for c in stacking_df.columns if c != 'HomeWon']
        X_meta = stacking_df[feature_cols].values
        y_meta = stacking_df['HomeWon'].values

        # Handle any remaining NaN/Inf values
        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features so regularization is fair across different scales
        meta_scaler = StandardScaler()
        X_meta = meta_scaler.fit_transform(X_meta)

        # Create meta-model based on type
        if meta_model_type == 'LogisticRegression':
            meta_model = LogisticRegression(C=meta_c_value, max_iter=10000, random_state=42)
        elif meta_model_type == 'SVM':
            meta_model = SVC(C=meta_c_value, probability=True, random_state=42)
        elif meta_model_type == 'GradientBoosting':
            meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unsupported meta_model_type: {meta_model_type}")

        meta_model.fit(X_meta, y_meta)

        return meta_model, meta_scaler
    
    def _evaluate_stacked_model(
        self,
        meta_model,  # Can be LogisticRegression, SVC, or GradientBoostingClassifier
        meta_scaler,  # StandardScaler fitted during training
        base_models_info: List[Dict],
        df: pd.DataFrame,
        evaluation_year: int,
        calibration_years: List[int],
        begin_year: int,
        n_train_samples: int = 0,
        n_cal_samples: int = 0,
        stacking_mode: str = 'naive',
        meta_features: Optional[List[str]] = None,
        use_disagree: bool = False,
        use_conf: bool = False
    ) -> Tuple[Dict, Dict]:
        """
        Evaluate stacked model on evaluation period.

        Args:
            meta_model: Trained meta-model
            base_models_info: List of base model info dicts
            df: DataFrame with evaluation period games
            evaluation_year: Evaluation year (for logging)
            calibration_years: Calibration years (for logging)
            begin_year: Begin year (for logging)
            n_train_samples: Number of training samples
            n_cal_samples: Number of calibration samples
            stacking_mode: 'naive' or 'informed'
            meta_features: Optional list of user-provided features
            use_disagree: If True, include disagreement features
            use_conf: If True, include confidence features

        Returns:
            Tuple of (metrics_dict, diagnostics_dict)
        """
        # Generate stacking data for evaluation period
        stacking_df = self._generate_stacking_data(
            base_models_info=base_models_info,
            df=df,
            calibration_years=[],  # Not used for evaluation, just for consistency
            stacking_mode=stacking_mode,
            meta_features=meta_features,
            use_disagree=use_disagree,
            use_conf=use_conf
        )
        
        # Extract features and labels
        feature_cols = [c for c in stacking_df.columns if c != 'HomeWon']
        X_meta = stacking_df[feature_cols].values
        y_true = stacking_df['HomeWon'].values

        # Handle any remaining NaN/Inf values
        X_meta = np.nan_to_num(X_meta, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply same scaler used during training
        if meta_scaler is not None:
            X_meta = meta_scaler.transform(X_meta)

        # Get meta-model predictions
        y_proba_meta = meta_model.predict_proba(X_meta)
        y_pred_meta = (y_proba_meta[:, 1] >= 0.5).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_meta) * 100
        log_loss_val = log_loss(y_true, y_proba_meta)
        brier = brier_score_loss(y_true, y_proba_meta[:, 1])
        
        try:
            auc = roc_auc_score(y_true, y_proba_meta[:, 1])
        except ValueError:
            auc = 0.0  # Can't calculate AUC if only one class present
        
        metrics = {
            'accuracy_mean': float(accuracy),
            'accuracy_std': 0.0,
            'log_loss_mean': float(log_loss_val),
            'log_loss_std': 0.0,
            'brier_mean': float(brier),
            'brier_std': 0.0,
            'auc_mean': float(auc),
            'auc_std': 0.0,
            'n_folds': 1,
            'split_type': 'time_based_calibration',
            'evaluation_year': evaluation_year
        }
        
        # Calculate meta-model feature importances (coefficients)
        meta_feature_importances = {}
        if hasattr(meta_model, 'coef_'):
            coef = meta_model.coef_[0] if len(meta_model.coef_.shape) > 1 else meta_model.coef_
            meta_feature_importances = dict(zip(feature_cols, np.abs(coef).tolist()))
            meta_feature_importances = dict(sorted(
                meta_feature_importances.items(),
                key=lambda x: x[1],
                reverse=True
            ))
        
        # Re-evaluate base models on the same evaluation set for fair comparison
        # This ensures we're comparing apples-to-apples
        base_models_summary = []
        for model_info in base_models_info:
            base_run_id = model_info['run_id']
            base_run = self.run_tracker.get_run(base_run_id)
            model = model_info['model']
            scaler = model_info['scaler']
            model_feature_names = model_info['feature_names']
            
            # Re-evaluate base model on evaluation set
            base_model_metrics = {}
            try:
                # Extract features for this base model from evaluation set
                meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'SeasonStartYear']
                target_cols = ['HomeWon']
                excluded_cols = meta_cols + target_cols
                pred_cols = ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total']
                excluded_cols.extend([c for c in pred_cols if c in df.columns])
                
                # Build feature vector for this model (handle missing features)
                X_base = np.zeros((len(df), len(model_feature_names)))
                for i, feat_name in enumerate(model_feature_names):
                    if feat_name in df.columns:
                        X_base[:, i] = df[feat_name].values
                    else:
                        # Missing feature - fill with 0
                        X_base[:, i] = 0.0
                
                # Handle NaN/Inf
                X_base = np.nan_to_num(X_base, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Scale if scaler exists
                if scaler:
                    X_base_scaled = scaler.transform(X_base)
                else:
                    X_base_scaled = X_base
                
                # Get predictions
                y_proba_base = model.predict_proba(X_base_scaled)
                y_pred_base = (y_proba_base[:, 1] >= 0.5).astype(int)
                y_true_base = df['HomeWon'].values
                
                # Calculate metrics
                base_accuracy = accuracy_score(y_true_base, y_pred_base) * 100
                base_log_loss = log_loss(y_true_base, y_proba_base)
                base_brier = brier_score_loss(y_true_base, y_proba_base[:, 1])
                try:
                    base_auc = roc_auc_score(y_true_base, y_proba_base[:, 1])
                except ValueError:
                    base_auc = 0.0
                
                base_model_metrics = {
                    'accuracy_mean': float(base_accuracy),
                    'log_loss_mean': float(base_log_loss),
                    'brier_mean': float(base_brier),
                    'auc_mean': float(base_auc),
                    'n_samples_evaluation': len(y_true_base)
                }
            except Exception as e:
                print(f"Warning: Could not re-evaluate base model {base_run_id}: {e}")
                # Fall back to stored metrics
                if base_run:
                    base_model_metrics = base_run.get('metrics', {})
            
            # Get config and other info
            if base_run:
                base_config = base_run.get('config', {})
                base_splits = base_config.get('splits', {})
                base_models_summary.append({
                    'run_id': base_run_id,
                    'metrics': base_model_metrics,  # Use re-evaluated metrics
                    'config': base_config,
                    'model_type': base_run.get('model_type', 'Unknown'),
                    'feature_names': model_info.get('feature_names', []),
                    'begin_year': base_splits.get('begin_year'),
                    'calibration_years': base_splits.get('calibration_years', []),
                    'evaluation_year': base_splits.get('evaluation_year'),
                    'n_samples_train': base_run.get('metrics', {}).get('train_set_size'),
                    'n_samples_calibration': base_run.get('metrics', {}).get('calibrate_set_size'),
                    'n_samples_evaluation': base_model_metrics.get('n_samples_evaluation')
                })
        
        # Extract derived features and user features from stacking_df for diagnostics
        derived_features_used = []
        meta_features_used = []
        if stacking_mode == 'informed':
            # Get all column names except predictions and target
            all_cols = set(stacking_df.columns)
            pred_cols = {col for col in all_cols if col.startswith('p_')}
            target_cols = {'HomeWon'}
            
            # Derived features are disagreements and confidences
            derived_features_used = [col for col in all_cols if col.startswith('disagree_') or col.startswith('conf_')]
            
            # User-provided features are the remaining columns (excluding predictions, target, and derived)
            remaining_cols = all_cols - pred_cols - target_cols - set(derived_features_used)
            if meta_features:
                # Only include features that were actually requested and found
                meta_features_used = [col for col in remaining_cols if col in meta_features]
        
        diagnostics = {
            'meta_model_type': 'LogisticRegression',
            'meta_feature_importances': meta_feature_importances,
            'n_base_models': len(base_models_info),
            'base_run_ids': [info['run_id'] for info in base_models_info],
            'base_models_summary': base_models_summary,  # NEW: Include base model details
            'n_samples_train': n_train_samples,
            'n_samples_calibration': n_cal_samples,
            'n_samples_evaluation': len(y_true),
            'evaluation_year': evaluation_year,
            'calibration_years': calibration_years,
            'begin_year': begin_year,
            'split_type': 'time_based_calibration',
            'stacking_mode': stacking_mode,
            'use_disagree': use_disagree,
            'use_conf': use_conf,
            'meta_features_used': meta_features_used,
            'derived_features_used': derived_features_used
        }
        
        return metrics, diagnostics
