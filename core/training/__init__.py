"""
Core Training Module

This module provides the foundational training infrastructure for the NBA prediction system.
All model training, evaluation, and related utilities live here.

Consumer layers (web/, cli/, agents/) should import from this module rather than
implementing training logic themselves.
"""

from bball_app.core.training.constants import (
    DEFAULT_MODEL_TYPES,
    DEFAULT_C_VALUES,
    C_SUPPORTED_MODELS,
    MODEL_CACHE_FILE,
    MODEL_CACHE_FILE_NO_PER,
    OUTPUTS_DIR,
)

from bball_app.core.training.model_factory import (
    create_model_with_c,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
)

from bball_app.core.training.model_evaluation import (
    evaluate_model_combo,
    evaluate_model_combo_with_calibration,
    compute_feature_importance,
)

from bball_app.core.training.cache_utils import (
    load_model_cache,
    save_model_cache,
    get_best_config,
    get_latest_training_csv,
    read_csv_safe,
)

# Training workflow tools (moved from agents/tools/)
from bball_app.core.training.experiment_runner import ExperimentRunner
from bball_app.core.training.stacking_trainer import StackingTrainer
from bball_app.core.training.run_tracker import RunTracker
from bball_app.core.training.dataset_builder import DatasetBuilder

# Schemas (moved from agents/schemas/)
from bball_app.core.training.schemas import (
    ModelConfig,
    PointsRegressionModelConfig,
    FeatureConfig,
    SplitConfig,
    PreprocessingConfig,
    ConstraintsConfig,
    StackingConfig,
    ExperimentConfig,
    DatasetSpec,
)

__all__ = [
    # Constants
    'DEFAULT_MODEL_TYPES',
    'DEFAULT_C_VALUES',
    'C_SUPPORTED_MODELS',
    'MODEL_CACHE_FILE',
    'MODEL_CACHE_FILE_NO_PER',
    'OUTPUTS_DIR',
    # Model factory
    'create_model_with_c',
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE',
    # Model evaluation
    'evaluate_model_combo',
    'evaluate_model_combo_with_calibration',
    'compute_feature_importance',
    # Cache utilities
    'load_model_cache',
    'save_model_cache',
    'get_best_config',
    'get_latest_training_csv',
    'read_csv_safe',
    # Training workflow tools
    'ExperimentRunner',
    'StackingTrainer',
    'RunTracker',
    'DatasetBuilder',
    # Schemas
    'ModelConfig',
    'PointsRegressionModelConfig',
    'FeatureConfig',
    'SplitConfig',
    'PreprocessingConfig',
    'ConstraintsConfig',
    'StackingConfig',
    'ExperimentConfig',
    'DatasetSpec',
]
