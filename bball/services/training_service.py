"""
Training Service - Thin orchestration layer for CLI training scripts.

Coordinates existing components:
- ModelConfigManager: Config creation and management
- ExperimentRunner: Training execution
- StackingTrainer: Ensemble model training
- ClassifierConfigRepository: Config data access
"""

import uuid
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from bson import ObjectId

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


# Model type aliases for CLI convenience
MODEL_TYPE_ALIASES = {
    'LR': 'LogisticRegression',
    'GB': 'GradientBoosting',
    'SVM': 'SVM',
}


class TrainingService:
    """
    Thin orchestration layer for model training operations.

    Coordinates existing core components for CLI training workflows.
    This is NOT a replacement for existing logic - it simply wires together
    existing functionality for CLI use.
    """

    def __init__(self, league: Optional["LeagueConfig"] = None, db=None):
        """
        Initialize TrainingService.

        Args:
            league: LeagueConfig instance for league-specific operations
            db: MongoDB database instance (optional, creates if not provided)
        """
        # Lazy imports to avoid circular dependencies
        from bball.mongo import Mongo
        from bball.data import ClassifierConfigRepository, ExperimentRunsRepository
        from bball.services.config_manager import ModelConfigManager
        from bball.training.experiment_runner import ExperimentRunner
        from bball.training.stacking_trainer import StackingTrainer
        from bball.training.run_tracker import RunTracker

        if db is None:
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league

        # Initialize components with league awareness
        self._classifier_repo = ClassifierConfigRepository(self.db, league=league)
        self._runs_repo = ExperimentRunsRepository(self.db, league=league)
        self._config_manager = ModelConfigManager(self.db, league=league)
        self._experiment_runner = ExperimentRunner(db=self.db, league=league)
        self._stacking_trainer = StackingTrainer(db=self.db, league=league)
        self._run_tracker = RunTracker(db=self.db, league=league)

    def resolve_model(self, name_or_id: str) -> Optional[Dict]:
        """
        Resolve a model by name OR MongoDB _id.

        Args:
            name_or_id: Model name (string) or MongoDB ObjectId (24-char hex)

        Returns:
            Model config dict or None if not found
        """
        # 1. Try as ObjectId (24-char hex string)
        if len(name_or_id) == 24:
            try:
                config = self._classifier_repo.find_one({'_id': ObjectId(name_or_id)})
                if config:
                    config['_id'] = str(config['_id'])
                    return config
            except Exception:
                pass  # Not a valid ObjectId, try as name

        # 2. Try as name
        config = self._classifier_repo.find_one({'name': name_or_id})
        if config:
            config['_id'] = str(config['_id'])
            return config

        return None

    def resolve_model_type(self, model_type: str) -> str:
        """
        Resolve model type alias to full name.

        Args:
            model_type: Model type or alias (e.g., 'LR', 'GB')

        Returns:
            Full model type name (e.g., 'LogisticRegression')
        """
        return MODEL_TYPE_ALIASES.get(model_type.upper(), model_type)

    def train_base_model(
        self,
        model_type: str,
        features: List[str],
        train_seasons: List[int],
        calibration_seasons: Optional[List[int]] = None,
        evaluation_season: Optional[int] = None,
        c_value: float = 0.1,
        calibration_method: str = 'sigmoid',
        name: Optional[str] = None,
        min_games_played: int = 20,
        include_injuries: bool = False,
        use_master: bool = True,
        exclude_seasons: Optional[List[int]] = None,
        use_time_calibration: bool = True,
        force_rebuild: bool = False,
    ) -> Dict:
        """
        Train a base classifier model.

        Args:
            model_type: Model type ('LogisticRegression', 'GradientBoosting', etc.)
            features: List of feature names
            train_seasons: List of training season years (begin_year values)
            calibration_seasons: List of calibration season years (required when use_time_calibration=True)
            evaluation_season: Evaluation season year (required when use_time_calibration=True)
            c_value: Regularization parameter (for LR, SVM)
            calibration_method: 'sigmoid' or 'isotonic'
            name: Optional model name
            min_games_played: Minimum games filter
            include_injuries: Include injury features
            use_master: Use master training CSV
            exclude_seasons: Optional list of seasons to exclude from training
            use_time_calibration: Use year-based calibration splits (True) or rolling CV (False)

        Returns:
            Dict with config_id, run_id, metrics, feature_importances
        """
        # Resolve model type alias
        model_type = self.resolve_model_type(model_type)

        # Determine begin_year from train_seasons
        begin_year = min(train_seasons) if train_seasons else 2012

        # Create or get config via ModelConfigManager
        config_id, config = self._config_manager.create_classifier_config(
            model_type=model_type,
            features=features,
            c_value=c_value,
            use_time_calibration=use_time_calibration,
            calibration_method=calibration_method,
            begin_year=begin_year,
            calibration_years=calibration_seasons if use_time_calibration else None,
            evaluation_year=evaluation_season if use_time_calibration else None,
            min_games_played=min_games_played,
            include_injuries=include_injuries,
            exclude_seasons=exclude_seasons,
            use_master=use_master,
            name=name,
        )

        # Build model config - only include c_value for models that support it
        model_config = {'type': model_type}
        if model_type in ('LogisticRegression', 'SVM'):
            model_config['c_value'] = c_value

        # Build splits config based on calibration mode
        if use_time_calibration:
            splits_config = {
                'type': 'year_based_calibration',
                'begin_year': begin_year,
                'calibration_years': calibration_seasons,
                'evaluation_year': evaluation_season,
                'min_games_played': min_games_played,
                'exclude_seasons': exclude_seasons or [],
            }
        else:
            splits_config = {
                'type': 'rolling_cv',
                'begin_year': begin_year,
                'min_games_played': min_games_played,
                'exclude_seasons': exclude_seasons or [],
            }

        # Build experiment config for ExperimentRunner
        experiment_config = {
            'task': 'binary_home_win',
            'model': model_config,
            'features': {
                'blocks': [],
                'features': features,
                'include_per': True,
                'diff_mode': 'home_minus_away',
                'point_model_id': None,
            },
            'splits': splits_config,
            'preprocessing': {
                'scaler': 'standard',
            },
            'calibration_method': calibration_method,
        }

        # Generate a CLI session ID
        session_id = f"cli-{uuid.uuid4().hex[:8]}"

        # Run the experiment
        result = self._experiment_runner.run_experiment(experiment_config, session_id, force_rebuild=force_rebuild)

        # Extract F-scores and feature importances from diagnostics
        diagnostics = result.get('diagnostics', {})
        f_scores = diagnostics.get('f_scores', {})
        feature_importances = diagnostics.get('feature_importances', {})

        # Use actual features from experiment (dataset builder may drop missing features)
        actual_features = diagnostics.get('all_feature_names') or features

        # Build training_stats with split sizes from metrics
        metrics = result.get('metrics', {})
        training_stats = {
            'total_games': diagnostics.get('n_samples', 0),
            'train_games': metrics.get('train_set_size'),
            'calibration_games': metrics.get('calibrate_set_size'),
            'eval_games': metrics.get('eval_set_size'),
        }

        # Link run to config (including features, F-scores, and model importances)
        self._config_manager.link_run_to_config(
            config_id=config_id,
            run_id=result['run_id'],
            config_type='classifier',
            metrics=result.get('metrics'),
            artifacts=result.get('artifacts'),
            dataset_id=result.get('dataset_id'),
            training_csv=result.get('artifacts', {}).get('dataset_path'),
            f_scores=f_scores,
            feature_importances=feature_importances,
            features=actual_features,
            training_stats=training_stats,
        )

        return {
            'config_id': config_id,
            'run_id': result['run_id'],
            'metrics': result.get('metrics', {}),
            'f_scores': f_scores,
            'feature_importances': feature_importances,
            'n_features': diagnostics.get('n_features', 0),
            'n_samples': diagnostics.get('n_samples', 0),
        }

    def train_model_grid(
        self,
        model_types: List[str],
        c_values: List[float],
        features: List[str],
        use_time_calibration: bool = True,
        calibration_method: str = 'sigmoid',
        begin_year: int = 2012,
        calibration_years: Optional[List[int]] = None,
        evaluation_year: Optional[int] = None,
        min_games_played: int = 15,
        include_injuries: bool = False,
        exclude_seasons: Optional[List[int]] = None,
        use_master: bool = True,
        name_prefix: Optional[str] = None,
        progress_callback: Optional[Any] = None,
        force_rebuild: bool = False,
    ) -> Dict:
        """
        Train models across a grid of model_types x c_values.

        For each model_type, trains all C-value variants (if C-supported),
        picks the best by accuracy, and saves the config with c_values_grid metadata.

        Args:
            model_types: List of model types to train
            c_values: List of C-values for regularized models
            features: List of feature names
            use_time_calibration: Use year-based calibration splits
            calibration_method: 'sigmoid' or 'isotonic'
            begin_year: Start year for training data
            calibration_years: Years for calibration (required when use_time_calibration=True)
            evaluation_year: Year for evaluation (required when use_time_calibration=True)
            min_games_played: Minimum games filter
            include_injuries: Include injury features
            exclude_seasons: Optional seasons to exclude from training
            use_master: Use master training CSV
            name_prefix: Optional name prefix for saved configs
            progress_callback: Optional callable(pct, msg) for progress updates

        Returns:
            Dict with model_type_results and saved_config_ids
        """
        from bball.training.constants import C_SUPPORTED_MODELS

        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        # Compute train_seasons from begin_year up to calibration boundary
        if use_time_calibration and calibration_years:
            cal_start = min(calibration_years)
        else:
            # When no calibration, use all seasons from begin_year
            cal_start = 2100  # Effectively no upper bound
        train_seasons = [
            y for y in range(begin_year, cal_start)
            if y not in (exclude_seasons or [])
        ]

        # Count total combos for progress
        total_combos = 0
        for mt in model_types:
            resolved_mt = self.resolve_model_type(mt)
            if resolved_mt in C_SUPPORTED_MODELS and c_values:
                total_combos += len(c_values)
            else:
                total_combos += 1

        combo_done = 0
        model_type_results = {}
        saved_config_ids = []

        for mt in model_types:
            resolved_mt = self.resolve_model_type(mt)
            mt_results = []
            best_result = None
            best_accuracy = -1.0

            if resolved_mt in C_SUPPORTED_MODELS and c_values:
                c_list = c_values
            else:
                c_list = [None]

            for c_val in c_list:
                combo_done += 1
                pct = 5 + int(90 * combo_done / total_combos)
                c_label = f' (C={c_val})' if c_val is not None else ''
                _progress(pct, f'Training {resolved_mt}{c_label}... [{combo_done}/{total_combos}]')

                try:
                    result = self.train_base_model(
                        model_type=resolved_mt,
                        features=features,
                        train_seasons=train_seasons,
                        calibration_seasons=calibration_years,
                        evaluation_season=evaluation_year,
                        c_value=c_val if c_val is not None else 0.1,
                        calibration_method=calibration_method,
                        name=name_prefix,
                        min_games_played=min_games_played,
                        include_injuries=include_injuries,
                        use_master=use_master,
                        exclude_seasons=exclude_seasons,
                        use_time_calibration=use_time_calibration,
                        force_rebuild=force_rebuild,
                    )

                    acc = result.get('metrics', {}).get('accuracy_mean', 0.0) or 0.0
                    entry = {
                        'config_id': result['config_id'],
                        'run_id': result['run_id'],
                        'c_value': c_val,
                        'accuracy': acc,
                        'metrics': result.get('metrics', {}),
                    }
                    mt_results.append(entry)

                    if acc > best_accuracy:
                        best_accuracy = acc
                        best_result = entry

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    mt_results.append({
                        'c_value': c_val,
                        'error': str(e),
                    })

            # After all combos for this model_type, update the best config with c_values_grid
            if best_result and len(c_list) > 1:
                c_values_grid = {}
                best_c_value = None
                best_c_accuracy = None
                for entry in mt_results:
                    if 'error' not in entry and entry.get('c_value') is not None:
                        c_values_grid[str(entry['c_value'])] = entry['accuracy']
                        if entry['accuracy'] >= (best_c_accuracy or -1):
                            best_c_accuracy = entry['accuracy']
                            best_c_value = entry['c_value']

                if c_values_grid:
                    self._config_manager.link_run_to_config(
                        config_id=best_result['config_id'],
                        run_id=best_result['run_id'],
                        c_values_grid=c_values_grid,
                        best_c_value=best_c_value,
                        best_c_accuracy=best_c_accuracy,
                    )

            if best_result:
                saved_config_ids.append(best_result['config_id'])

            model_type_results[resolved_mt] = {
                'best': best_result,
                'all': mt_results,
            }

        _progress(100, 'Training complete')

        return {
            'model_type_results': model_type_results,
            'saved_config_ids': saved_config_ids,
        }

    # =========================================================================
    # Ensemble validation & config helpers
    # =========================================================================

    def validate_ensemble_models(self, base_model_ids: List[str]) -> Dict:
        """
        Validate base models for ensemble creation/training.

        Checks:
        - At least 2 models
        - All exist and are non-ensemble
        - All have use_time_calibration enabled
        - All time configs match (begin_year, calibration_years, evaluation_year)

        Returns:
            Dict with 'base_models', 'base_config_ids', 'time_config', 'meta_min_games_played'
        """
        if len(base_model_ids) < 2:
            raise ValueError('At least 2 models are required for ensemble')

        base_models = []
        base_config_ids = []
        ref_time = None

        for i, model_id in enumerate(base_model_ids):
            config = self.resolve_model(model_id)
            if not config:
                raise ValueError(f'Model not found: {model_id}')
            if config.get('ensemble_models') or config.get('ensemble'):
                raise ValueError(f'Cannot include ensemble model {model_id} in another ensemble')
            if not config.get('use_time_calibration', False):
                name = config.get('name', model_id)
                raise ValueError(
                    f'Model "{name}" does not have time-based calibration enabled. '
                    f'All base models must have use_time_calibration=True'
                )

            base_models.append(config)
            base_config_ids.append(config['_id'])

            time = {
                'begin_year': config.get('begin_year'),
                'calibration_years': config.get('calibration_years', []),
                'evaluation_year': config.get('evaluation_year'),
            }

            if i == 0:
                ref_time = time
            else:
                for key in ('begin_year', 'calibration_years', 'evaluation_year'):
                    if time[key] != ref_time[key]:
                        name = config.get('name', model_id)
                        raise ValueError(
                            f"Time config mismatch: model \"{name}\" has {key}={time[key]}, "
                            f"but first model has {key}={ref_time[key]}. "
                            f"All base models must have identical temporal configurations."
                        )

        meta_min_games_played = max(
            (m.get('min_games_played', 0) or 0 for m in base_models),
            default=0,
        )

        return {
            'base_models': base_models,
            'base_config_ids': base_config_ids,
            'time_config': {
                **ref_time,
                'calibration_method': base_models[0].get('calibration_method', 'sigmoid'),
                'exclude_seasons': base_models[0].get('exclude_seasons'),
            },
            'meta_min_games_played': meta_min_games_played,
        }

    def create_ensemble_config(
        self,
        base_model_ids: List[str],
        meta_model_type: str = 'LogisticRegression',
        meta_c_value: Optional[float] = 0.1,
        stacking_mode: str = 'naive',
        ensemble_meta_features: Optional[List[str]] = None,
        ensemble_use_disagree: bool = False,
        ensemble_use_conf: bool = False,
        ensemble_use_logit: bool = False,
        name: Optional[str] = None,
        ensemble_id: Optional[str] = None,
        meta_calibration_method: Optional[str] = None,
        meta_train_years: Optional[List[int]] = None,
        meta_calibration_years: Optional[List[int]] = None,
        meta_evaluation_year: Optional[int] = None,
    ) -> Dict:
        """
        Create or update an ensemble config doc WITHOUT training.

        If ensemble_id is provided, updates that document.
        Otherwise, looks up by ensemble_models + model_type, or creates new.

        Returns:
            Dict with 'ensemble_id', 'is_new'
        """
        from datetime import datetime

        meta_model_type = self.resolve_model_type(meta_model_type)
        validated = self.validate_ensemble_models(base_model_ids)
        base_config_ids = validated['base_config_ids']
        time_config = validated['time_config']

        ensemble_doc = {
            'ensemble': True,
            'ensemble_type': 'stacking',
            'ensemble_models': base_config_ids,
            'ensemble_meta_features': ensemble_meta_features or [],
            'ensemble_use_disagree': ensemble_use_disagree,
            'ensemble_use_conf': ensemble_use_conf,
            'ensemble_use_logit': ensemble_use_logit,
            'model_type': meta_model_type,
            'best_c_value': meta_c_value if meta_model_type in ('LogisticRegression', 'SVM') else None,
            'features': [],
            'feature_count': 0,
            'name': name or f'Ensemble ({len(base_config_ids)} models) - {meta_model_type}',
            'selected': False,
            'use_master': True,
            'use_time_calibration': True,
            'stacking_mode': stacking_mode,
            'calibration_method': time_config['calibration_method'],
            'begin_year': time_config['begin_year'],
            'calibration_years': time_config['calibration_years'],
            'evaluation_year': time_config['evaluation_year'],
            'exclude_seasons': time_config.get('exclude_seasons'),
            'min_games_played': validated['meta_min_games_played'],
            'include_injuries': False,
            'meta_calibration_method': meta_calibration_method,
            'meta_train_years': meta_train_years,
            'meta_calibration_years': meta_calibration_years,
            'meta_evaluation_year': meta_evaluation_year,
            'updated_at': datetime.utcnow(),
        }

        # Find existing by explicit ID only — never match by content to avoid
        # updating the wrong ensemble when multiple share the same base models.
        existing = None
        if ensemble_id:
            existing = self._classifier_repo.find_one({'_id': ObjectId(ensemble_id)})

        if existing:
            # Preserve selected status and user-set name
            ensemble_doc['selected'] = existing.get('selected', False)
            if not name and existing.get('name'):
                ensemble_doc['name'] = existing['name']
            self._classifier_repo.update_one(
                {'_id': existing['_id']},
                {'$set': ensemble_doc}
            )
            return {
                'ensemble_id': str(existing['_id']),
                'is_new': False,
            }
        else:
            ensemble_doc['trained_at'] = datetime.utcnow()
            result = self._classifier_repo.insert_one(ensemble_doc)
            return {
                'ensemble_id': str(result.inserted_id),
                'is_new': True,
            }

    def list_available_base_models(self) -> List[Dict]:
        """List all trained non-ensemble models available for ensembles."""
        models = self._classifier_repo.find_trained_non_ensemble()
        for m in models:
            m['_id'] = str(m['_id'])
        return models

    def update_ensemble_settings(self, ensemble_id: str, updates: Dict) -> Dict:
        """
        Update allowed fields on an existing ensemble config.

        Returns:
            Dict with 'success', 'message'
        """
        from datetime import datetime

        existing = self._classifier_repo.find_one({'_id': ObjectId(ensemble_id)})
        if not existing:
            raise ValueError('Ensemble not found')

        allowed_fields = {
            'name', 'ensemble_models', 'ensemble_type', 'ensemble_meta_features',
            'ensemble_use_disagree', 'ensemble_use_conf', 'ensemble_use_logit',
            'stacking_mode', 'selected', 'model_type', 'best_c_value',
            'meta_model_type', 'meta_c_value',
            'meta_calibration_method', 'meta_train_years',
            'meta_calibration_years', 'meta_evaluation_year',
        }
        update_fields = {k: v for k, v in updates.items() if k in allowed_fields}

        if update_fields:
            update_fields['updated_at'] = datetime.utcnow()

            self._classifier_repo.update_one(
                {'_id': ObjectId(ensemble_id)},
                {'$set': update_fields}
            )

        return {'success': True, 'message': 'Ensemble updated successfully'}

    def retrain_base_model(
        self,
        ensemble_id: str,
        base_model_id: str,
        retrain_meta: bool = True,
        progress_callback: Optional[Any] = None,
    ) -> Dict:
        """
        Retrain a base model within an ensemble, then optionally retrain the meta-model.

        Args:
            ensemble_id: MongoDB _id of the ensemble
            base_model_id: MongoDB _id of the base model to retrain
            retrain_meta: If True, retrain ensemble meta-model after base model
            progress_callback: Optional callable(pct, msg)

        Returns:
            Dict with base_result and optional ensemble_result
        """
        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        # Validate ensemble and membership
        ensemble = self.resolve_model(ensemble_id)
        if not ensemble:
            raise ValueError('Ensemble not found')

        ensemble_models = ensemble.get('ensemble_models', [])
        if base_model_id not in [str(m) for m in ensemble_models]:
            raise ValueError('Base model not part of this ensemble')

        _progress(5, 'Loading base model config...')

        base_config = self.resolve_model(base_model_id)
        if not base_config:
            raise ValueError('Base model config not found')

        _progress(10, 'Building experiment config...')

        # Build experiment config from the base model's fields
        model_type = base_config.get('model_type', 'LogisticRegression')
        c_value = base_config.get('best_c_value')
        features = base_config.get('features', [])

        experiment_config = {
            'task': 'binary_home_win',
            'model': {'type': model_type},
            'features': {'features': features},
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

        _progress(20, 'Running experiment (training base model)...')

        session_id = f"retrain-base-{uuid.uuid4().hex[:8]}"
        result = self._experiment_runner.run_experiment(experiment_config, session_id)

        _progress(60, 'Linking results to config...')

        diagnostics = result.get('diagnostics', {})
        metrics = result.get('metrics', {})
        training_stats = {
            'total_games': diagnostics.get('n_samples', 0),
            'train_games': metrics.get('train_set_size'),
            'calibration_games': metrics.get('calibrate_set_size'),
            'eval_games': metrics.get('eval_set_size'),
        }

        self._config_manager.link_run_to_config(
            config_id=base_model_id,
            run_id=result['run_id'],
            config_type='classifier',
            metrics=result.get('metrics'),
            artifacts=result.get('artifacts'),
            dataset_id=result.get('dataset_id'),
            f_scores=diagnostics.get('f_scores'),
            feature_importances=diagnostics.get('feature_importances'),
            features=features,
            best_c_value=diagnostics.get('best_c_value', c_value),
            training_stats=training_stats,
        )

        output = {'base_result': result}

        # Retrain ensemble meta-model
        if retrain_meta:
            _progress(70, 'Retraining ensemble meta-model...')
            base_ids = [str(m) for m in ensemble.get('ensemble_models', [])]
            ensemble_result = self.train_ensemble(
                meta_model_type=ensemble.get('model_type', 'LogisticRegression'),
                base_model_names_or_ids=base_ids,
                meta_c_value=ensemble.get('best_c_value', 0.1) or 0.1,
                extra_features=ensemble.get('ensemble_meta_features') or None,
                stacking_mode=ensemble.get('stacking_mode', 'naive'),
                use_disagree=ensemble.get('ensemble_use_disagree', False),
                use_conf=ensemble.get('ensemble_use_conf', False),
                use_logit=ensemble.get('ensemble_use_logit', False),
            )
            output['ensemble_result'] = ensemble_result

        _progress(95, 'Finishing up...')
        return output

    # =========================================================================
    # Ensemble training
    # =========================================================================

    def train_ensemble(
        self,
        meta_model_type: str,
        base_model_names_or_ids: List[str],
        meta_c_value: float = 0.1,
        extra_features: Optional[List[str]] = None,
        stacking_mode: str = 'informed',
        use_disagree: bool = False,
        use_conf: bool = False,
        use_logit: bool = False,
        name: Optional[str] = None,
        ensemble_id: Optional[str] = None,
        meta_calibration_method: Optional[str] = None,
        meta_train_years: Optional[List[int]] = None,
        meta_calibration_years: Optional[List[int]] = None,
        meta_evaluation_year: Optional[int] = None,
    ) -> Dict:
        """
        Train an ensemble (stacking) model.

        IMPORTANT: Temporal parameters (begin_year, calibration_years, evaluation_year)
        are derived from the base models to ensure consistency. All base models must
        have identical temporal configurations.

        Args:
            meta_model_type: Meta-model type ('LogisticRegression', 'GradientBoosting')
            base_model_names_or_ids: List of base model names or MongoDB _ids
            meta_c_value: C-value for meta-model (if applicable)
            extra_features: Additional features to include in meta-model
            stacking_mode: 'naive' or 'informed'
            use_disagree: Include disagreement features
            use_conf: Include confidence features
            use_logit: If True, feed logit(p_*) to meta-learner (eps handled by sportscore)
            name: Optional ensemble name
            ensemble_id: Optional existing ensemble _id to update (instead of creating new)

        Returns:
            Dict with run_id, metrics, base_models summary
        """
        meta_model_type = self.resolve_model_type(meta_model_type)

        # Validate and extract time config from base models
        validated = self.validate_ensemble_models(base_model_names_or_ids)
        base_config_ids = validated['base_config_ids']
        time_config = validated['time_config']

        base_models_info = [
            {'name': m.get('name'), 'id': m['_id'], 'model_type': m.get('model_type')}
            for m in validated['base_models']
        ]

        # Build dataset spec DERIVED FROM base models
        dataset_spec = {
            'begin_year': time_config['begin_year'],
            'calibration_years': time_config['calibration_years'],
            'evaluation_year': time_config['evaluation_year'],
            'min_games_played': validated['meta_min_games_played'],
        }

        # If meta calibration uses different years, ensure the dataset covers them
        if meta_calibration_method and meta_evaluation_year:
            all_meta_years = list(meta_train_years or []) + list(meta_calibration_years or []) + [meta_evaluation_year]
            max_meta_year = max(all_meta_years)
            if max_meta_year > dataset_spec['evaluation_year']:
                dataset_spec['evaluation_year'] = max_meta_year
            # Ensure calibration_years covers all meta years for dataset building
            existing_cal = set(dataset_spec['calibration_years'])
            for y in all_meta_years:
                if y not in existing_cal and y >= min(dataset_spec['calibration_years']):
                    existing_cal.add(y)
            dataset_spec['calibration_years'] = sorted(existing_cal)

        session_id = f"cli-ensemble-{uuid.uuid4().hex[:8]}"

        # Train stacked model
        result = self._stacking_trainer.train_stacked_model(
            dataset_spec=dataset_spec,
            session_id=session_id,
            base_config_ids=base_config_ids,
            meta_model_type=meta_model_type,
            meta_c_value=meta_c_value,
            stacking_mode=stacking_mode,
            meta_features=extra_features,
            use_disagree=use_disagree,
            use_conf=use_conf,
            use_logit=use_logit,
            meta_calibration_method=meta_calibration_method,
            meta_train_years=meta_train_years,
            meta_calibration_years=meta_calibration_years,
            meta_evaluation_year=meta_evaluation_year,
        )

        # Create/update ensemble config doc
        config_result = self.create_ensemble_config(
            base_model_ids=base_model_names_or_ids,
            meta_model_type=meta_model_type,
            meta_c_value=meta_c_value,
            stacking_mode=stacking_mode,
            ensemble_meta_features=extra_features,
            ensemble_use_disagree=use_disagree,
            ensemble_use_conf=use_conf,
            ensemble_use_logit=use_logit,
            name=name,
            ensemble_id=ensemble_id,
            meta_calibration_method=meta_calibration_method,
            meta_train_years=meta_train_years,
            meta_calibration_years=meta_calibration_years,
            meta_evaluation_year=meta_evaluation_year,
        )
        ensemble_id = config_result['ensemble_id']

        # Update with training results
        from datetime import datetime

        training_update = {
            'run_id': result['run_id'],
            'ensemble_run_id': result['run_id'],
            'trained_at': datetime.utcnow(),
            'updated_at': datetime.utcnow(),
        }

        # Metrics
        metrics = result.get('metrics', {})
        if metrics:
            training_update['accuracy'] = metrics.get('accuracy_mean')
            training_update['std_dev'] = metrics.get('accuracy_std')
            training_update['log_loss'] = metrics.get('log_loss_mean')
            training_update['brier_score'] = metrics.get('brier_mean')

        # Artifacts
        artifacts = result.get('artifacts', {})
        if artifacts:
            training_update['meta_model_path'] = artifacts.get('meta_model_path')
            training_update['meta_scaler_path'] = artifacts.get('meta_scaler_path')
            training_update['ensemble_config_path'] = artifacts.get('ensemble_config_path')
            if artifacts.get('meta_calibrator_path'):
                training_update['meta_calibrator_path'] = artifacts['meta_calibrator_path']

        # Meta-feature importances -> features_ranked
        diagnostics = result.get('diagnostics', {})
        if isinstance(diagnostics, dict):
            meta_fi = diagnostics.get('meta_feature_importances')
            if isinstance(meta_fi, dict) and meta_fi:
                features_ranked = []
                for rank, (fname, score) in enumerate(meta_fi.items(), 1):
                    safe_score = 0.0 if (score is None or (isinstance(score, float) and (score != score or abs(score) == float('inf')))) else float(score)
                    features_ranked.append({'rank': rank, 'name': fname, 'score': safe_score})
                training_update['features_ranked'] = features_ranked
                training_update['feature_count'] = len(features_ranked)
                training_update['features'] = [f['name'] for f in features_ranked]

            base_models_summary = diagnostics.get('base_models_summary')
            if isinstance(base_models_summary, list) and base_models_summary:
                training_update['ensemble_base_models_summary'] = base_models_summary

        self._classifier_repo.update_one(
            {'_id': ObjectId(ensemble_id)},
            {'$set': training_update}
        )

        return {
            'config_id': ensemble_id,
            'run_id': result['run_id'],
            'metrics': result.get('metrics', {}),
            'base_models': base_models_info,
            'diagnostics': result.get('diagnostics', {}),
        }

    def recalibrate_ensemble(
        self,
        ensemble_id: str,
        begin_year: int,
        calibration_years: List[int],
        evaluation_year: int,
        calibration_method: str = 'isotonic',
        exclude_seasons: Optional[List[int]] = None,
        min_games_played: Optional[int] = None,
        progress_callback: Optional[Any] = None,
    ) -> Dict:
        """
        Recalibrate an ensemble: retrain all base models with new time settings,
        then create a new ensemble with the retrained models.

        Args:
            ensemble_id: MongoDB _id of the source ensemble
            begin_year: New begin year for training
            calibration_years: New calibration years
            evaluation_year: New evaluation year
            calibration_method: 'isotonic' or 'sigmoid'
            exclude_seasons: Optional seasons to exclude from training
            min_games_played: Optional override for min games filter; None uses each base model's value
            progress_callback: Optional callable(pct: int, msg: str) for progress updates

        Returns:
            Dict with config_id, run_id, metrics, time_suffix
        """
        import re
        from bball.utils import build_time_suffix

        def _progress(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        # Step 1: Load ensemble and base model configs
        _progress(1, 'Loading ensemble configuration...')

        ensemble = self.resolve_model(ensemble_id)
        if not ensemble:
            raise ValueError(f"Ensemble not found: {ensemble_id}")

        base_model_ids = ensemble.get('ensemble_models', [])
        if not base_model_ids:
            raise ValueError("Ensemble has no base models")

        base_configs = []
        for mid in base_model_ids:
            cfg = self.resolve_model(mid)
            if not cfg:
                raise ValueError(f"Base model not found: {mid}")
            base_configs.append(cfg)

        # Extract meta-model settings from ensemble
        meta_model_type = ensemble.get('model_type', 'LogisticRegression')
        meta_c_value = ensemble.get('best_c_value', 0.1)
        meta_features = ensemble.get('ensemble_meta_features', [])
        use_disagree = ensemble.get('ensemble_use_disagree', False)
        use_conf = ensemble.get('ensemble_use_conf', False)
        use_logit = ensemble.get('ensemble_use_logit', False)
        stacking_mode = ensemble.get('stacking_mode', 'naive')
        # Infer stacking mode from settings if not stored
        if not ensemble.get('stacking_mode'):
            if use_disagree or use_conf or meta_features:
                stacking_mode = 'informed'

        # Step 2: Build time suffix and compute train_seasons
        time_suffix = build_time_suffix(begin_year, calibration_years, evaluation_year)
        train_seasons = [
            y for y in range(begin_year, min(calibration_years))
            if y not in (exclude_seasons or [])
        ]

        # Step 3: Retrain each base model
        suffix_regex = re.compile(r'\s*T\d{2,}C\d{2,}E\d{2}\s*$')
        n_base = len(base_configs)
        new_config_ids = []

        for i, base in enumerate(base_configs):
            base_name = base.get('name', base.get('model_type', 'Model'))
            pct = 2 + int(75 * i / n_base)
            _progress(pct, f'Training base model {i+1}/{n_base}: {base_name}...')

            # Strip old time suffix from name
            stripped_name = suffix_regex.sub('', base_name).strip()
            new_name = f"{stripped_name} {time_suffix}"

            result = self.train_base_model(
                model_type=base.get('model_type', 'LogisticRegression'),
                features=base.get('features', []),
                train_seasons=train_seasons,
                calibration_seasons=calibration_years,
                evaluation_season=evaluation_year,
                c_value=base.get('best_c_value', 0.1),
                calibration_method=calibration_method,
                name=new_name,
                min_games_played=min_games_played if min_games_played is not None else base.get('min_games_played', 15),
                include_injuries=base.get('include_injuries', False),
                exclude_seasons=exclude_seasons,
                use_master=True,
            )
            new_config_ids.append(result['config_id'])

        # Step 4: Create and train new ensemble
        _progress(80, 'Creating & training new ensemble...')

        ensemble_result = self.train_ensemble(
            meta_model_type=meta_model_type,
            base_model_names_or_ids=new_config_ids,
            meta_c_value=meta_c_value if meta_c_value else 0.1,
            extra_features=meta_features if meta_features else None,
            stacking_mode=stacking_mode,
            use_disagree=use_disagree,
            use_conf=use_conf,
            use_logit=use_logit,
        )

        # Step 5: Update the new ensemble doc name to include time_suffix
        _progress(95, 'Finalizing...')
        new_ensemble_id = ensemble_result['config_id']

        original_name = ensemble.get('name', 'Ensemble')
        stripped_ensemble_name = suffix_regex.sub('', original_name).strip()
        new_ensemble_name = f"{stripped_ensemble_name} {time_suffix}"

        update_fields = {
            'name': new_ensemble_name,
            'stacking_mode': stacking_mode,
        }
        if exclude_seasons:
            update_fields['exclude_seasons'] = exclude_seasons
        if min_games_played is not None:
            update_fields['min_games_played'] = min_games_played

        self._classifier_repo.update_one(
            {'_id': ObjectId(new_ensemble_id)},
            {'$set': update_fields}
        )

        _progress(100, 'Recalibration complete — new ensemble created')

        return {
            'config_id': new_ensemble_id,
            'run_id': ensemble_result.get('run_id'),
            'metrics': ensemble_result.get('metrics', {}),
            'time_suffix': time_suffix,
        }

    def get_model_results(self, name_or_id: str) -> Optional[Dict]:
        """
        Get detailed model results by name or ID.

        Args:
            name_or_id: Model name or MongoDB _id

        Returns:
            Dict with config details, metrics, feature importances
        """
        config = self.resolve_model(name_or_id)
        if not config:
            return None

        is_ensemble = config.get('ensemble', False)

        result = {
            'config_id': config.get('_id'),
            'name': config.get('name'),
            'model_type': config.get('model_type'),
            'is_ensemble': is_ensemble,
            'features': config.get('features', []),
            'feature_count': config.get('feature_count', len(config.get('features', []))),
            'begin_year': config.get('begin_year'),
            'calibration_years': config.get('calibration_years', []),
            'evaluation_year': config.get('evaluation_year'),
            'c_value': config.get('best_c_value'),
            'calibration_method': config.get('calibration_method'),
        }

        # Get metrics from config (they're stored there after training)
        result['metrics'] = {
            'accuracy': config.get('accuracy'),
            'log_loss': config.get('log_loss'),
            'brier_score': config.get('brier_score'),
            'auc': config.get('auc'),
        }

        # For feature importances, look at the linked run
        run_id = config.get('run_id')
        if run_id:
            run = self._run_tracker.get_run(run_id)
            if run:
                diagnostics = run.get('diagnostics', {})
                result['feature_importances'] = diagnostics.get('feature_importances', {})
                # Update metrics from run if available
                run_metrics = run.get('metrics', {})
                if run_metrics:
                    result['metrics'] = {
                        'accuracy': run_metrics.get('accuracy_mean'),
                        'log_loss': run_metrics.get('log_loss_mean'),
                        'brier_score': run_metrics.get('brier_mean'),
                        'auc': run_metrics.get('auc_mean') or run_metrics.get('auc'),
                    }

        # For ensemble models, get base model details
        if is_ensemble:
            ensemble_models = config.get('ensemble_models', [])
            base_models = []
            for base_id in ensemble_models:
                base_config = self.resolve_model(base_id)
                if base_config:
                    base_run_id = base_config.get('run_id')
                    base_feature_importances = {}
                    if base_run_id:
                        base_run = self._run_tracker.get_run(base_run_id)
                        if base_run:
                            base_feature_importances = base_run.get('diagnostics', {}).get('feature_importances', {})

                    base_models.append({
                        'id': base_config.get('_id'),
                        'name': base_config.get('name'),
                        'model_type': base_config.get('model_type'),
                        'features': base_config.get('features', []),
                        'feature_importances': base_feature_importances,
                    })
            result['base_models'] = base_models

        return result

    def list_ensembles(self) -> List[Dict]:
        """
        List all ensemble configurations with enriched base model details.

        Returns a JSON-serializable list of ensemble dicts, each with a
        ``base_models_details`` array containing the relevant base model info.
        """
        ensembles = self._classifier_repo.find_ensembles()

        # Serialize ObjectIds to strings
        for ens in ensembles:
            ens['_id'] = str(ens['_id'])
            if 'ensemble_models' in ens:
                ens['ensemble_models'] = [
                    str(m) if not isinstance(m, str) else m
                    for m in ens['ensemble_models']
                ]

            # Back-fill fields from first base model when missing
            if not ens.get('calibration_method') and ens.get('ensemble_models'):
                first_base = self._classifier_repo.find_by_ids(
                    [ens['ensemble_models'][0]],
                    projection={'calibration_method': 1, 'exclude_seasons': 1, 'min_games_played': 1},
                )
                if first_base:
                    fb = first_base[0]
                    ens.setdefault('calibration_method', fb.get('calibration_method'))
                    ens.setdefault('exclude_seasons', fb.get('exclude_seasons'))
                    ens.setdefault('min_games_played', fb.get('min_games_played'))

        # Batch-fetch base model details
        all_base_ids = {
            mid
            for ens in ensembles
            for mid in ens.get('ensemble_models', [])
        }

        base_models_map: Dict[str, Dict] = {}
        if all_base_ids:
            for bm in self._classifier_repo.find_by_ids(
                list(all_base_ids),
                projection={
                    'name': 1, 'model_type': 1, 'best_c_value': 1,
                    'accuracy': 1, 'log_loss': 1, 'brier_score': 1,
                    'min_games_played': 1, 'training_stats': 1,
                    'features_ranked': 1, 'features_ranked_by_importance': 1,
                    'feature_count': 1,
                },
            ):
                bm_id = str(bm['_id'])
                bm['_id'] = bm_id
                bm['c_value'] = bm.pop('best_c_value', None)
                base_models_map[bm_id] = bm

        for ens in ensembles:
            ens['base_models_details'] = [
                base_models_map[mid]
                for mid in ens.get('ensemble_models', [])
                if mid in base_models_map
            ]

        return ensembles

    def list_models(
        self,
        model_type: Optional[str] = None,
        ensemble_only: bool = False,
        trained_only: bool = False,
        limit: int = 50,
    ) -> List[Dict]:
        """
        List available models with optional filters.

        Args:
            model_type: Filter by model type
            ensemble_only: Only return ensemble models
            trained_only: Only return trained models
            limit: Maximum number of results

        Returns:
            List of model config summaries
        """
        if model_type:
            model_type = self.resolve_model_type(model_type)
            configs = self._classifier_repo.find_by_model_type(model_type)
        elif ensemble_only:
            configs = self._classifier_repo.find_ensembles()
        else:
            configs = self._classifier_repo.find_all(trained_only=trained_only, limit=limit)

        results = []
        for config in configs[:limit]:
            results.append({
                'id': str(config.get('_id')),
                'name': config.get('name'),
                'model_type': config.get('model_type'),
                'is_ensemble': config.get('ensemble', False),
                'accuracy': config.get('accuracy'),
                'trained': bool(config.get('model_artifact_path') or config.get('run_id')),
                'created_at': config.get('created_at'),
            })

        return results
