"""
DEPRECATED: This module is a backward-compatibility shim.
Please import from bball_app.core.training.schemas instead.

Experiment Configuration Schemas - Pydantic models for typed config validation
"""

# Re-export all schemas from core.training.schemas
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

