"""
Experiment Configuration Schemas - Pydantic models for typed config validation

Basketball-specific schemas that extend sportscore's base schemas.
Adds basketball-specific fields (include_per, point_model_id, basketball feature blocks)
while inheriting all shared configuration from sportscore.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator

from sportscore.training.schemas import (
    ModelConfig,
    RegressionModelConfig as PointsRegressionModelConfig,
    SplitConfig,
    PreprocessingConfig,
    ConstraintsConfig,
    StackingConfig,
    ExperimentConfig as _BaseExperimentConfig,
    DatasetSpec as _BaseDatasetSpec,
)


class FeatureConfig(BaseModel):
    """Feature selection configuration with basketball-specific blocks."""
    # Feature blocks (from feature_sets.py)
    blocks: List[str] = Field(
        default_factory=list,
        description="Feature set names from FEATURE_SETS (e.g., 'outcome_strength', 'shooting_efficiency')"
    )

    # Alternative: specify individual features
    features: Optional[List[str]] = Field(
        None,
        description="Specific feature names (overrides blocks if provided)"
    )

    # Diffing strategy
    diff_mode: Literal['home_minus_away', 'away_minus_home', 'absolute', 'mixed', 'all'] = 'home_minus_away'

    # Perspective flip augmentation
    flip_augment: bool = Field(False, description="Include perspective-flipped training examples")

    # Basketball-specific: PER features
    include_per: bool = Field(True, description="Include PER (Player Efficiency Rating) features")

    # Basketball-specific: Point prediction features
    point_model_id: Optional[str] = Field(
        None,
        description="Model ID for point predictions to merge. Only pred_margin is included as a feature by default. Other prediction columns (pred_home_points, pred_away_points, pred_point_total) are merged into the dataframe for reference but excluded from the feature set."
    )

    @validator('blocks')
    def validate_blocks(cls, v):
        """Validate feature block names against FeatureGroups SSoT."""
        from bball.features.groups import FeatureGroups
        valid_blocks = set(FeatureGroups.get_all_groups())
        # Also accept catch-all groups returned by get_group_for_feature()
        valid_blocks.update(['other', 'era_normalization'])
        for block in v:
            if block not in valid_blocks:
                raise ValueError(
                    f"Invalid feature block: {block}. "
                    f"Must be one of {sorted(valid_blocks)}"
                )
        return v


class ExperimentConfig(_BaseExperimentConfig):
    """Complete experiment configuration with basketball-specific task types."""
    task: Literal['binary_home_win', 'points_regression', 'stacking'] = 'binary_home_win'

    # Override to use basketball's FeatureConfig (with include_per, point_model_id)
    features: FeatureConfig

    # Basketball uses 'points_model' field name (sportscore uses 'regression_model')
    points_model: Optional[PointsRegressionModelConfig] = Field(None, description="Model config for regression (required if task='points_regression')")

    @validator('points_model')
    def validate_points_model_for_regression(cls, v, values):
        """Ensure points_model is provided for regression task"""
        task = values.get('task', 'binary_home_win')
        if task == 'points_regression' and v is None:
            raise ValueError("points_model is required when task='points_regression'")
        return v

    class Config:
        extra = 'forbid'


class DatasetSpec(_BaseDatasetSpec):
    """Dataset specification with basketball-specific fields."""
    # Basketball-specific: PER features
    include_per: bool = True

    # Basketball-specific: Point prediction features
    point_model_id: Optional[str] = Field(
        None,
        description="Model ID for point predictions to merge. Only pred_margin is included as a feature by default. Other prediction columns (pred_home_points, pred_away_points, pred_point_total) are merged into the dataframe for reference but excluded from the feature set."
    )

    class Config:
        extra = 'forbid'


# Backward compatibility aliases
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
