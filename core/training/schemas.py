"""
Experiment Configuration Schemas - Pydantic models for typed config validation

This module contains all training-related schemas including:
- ModelConfig: Classification model configuration
- PointsRegressionModelConfig: Regression model configuration
- FeatureConfig: Feature selection configuration
- SplitConfig: Data splitting strategy
- PreprocessingConfig: Preprocessing options
- ConstraintsConfig: Resource constraints
- StackingConfig: Stacking ensemble configuration
- ExperimentConfig: Complete experiment configuration
- DatasetSpec: Dataset specification for build_dataset()
"""

from typing import List, Optional, Dict, Literal, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class ModelConfig(BaseModel):
    """Model type and hyperparameters for classification"""
    type: Literal[
        'LogisticRegression',
        'RandomForest',
        'GradientBoosting',
        'SVM',
        'NaiveBayes',
        'NeuralNetwork',
        'XGBoost',
        'LightGBM',
        'CatBoost'
    ]

    # Hyperparameters (bounded ranges)
    c_value: Optional[float] = Field(None, ge=0.001, le=100.0, description="Regularization parameter for LogisticRegression/SVM")
    n_estimators: Optional[int] = Field(None, ge=10, le=1000, description="Number of trees for ensemble methods")
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="Max depth for tree models")
    learning_rate: Optional[float] = Field(None, ge=0.001, le=1.0, description="Learning rate for boosting methods")

    @validator('c_value')
    def validate_c_value(cls, v, values):
        """C-value only applies to LogisticRegression and SVM"""
        if v is not None and values.get('type') not in ['LogisticRegression', 'SVM']:
            raise ValueError(f"c_value only applies to LogisticRegression and SVM, not {values.get('type')}")
        return v


class PointsRegressionModelConfig(BaseModel):
    """Model type and hyperparameters for points regression"""
    type: Literal[
        'Ridge',
        'ElasticNet',
        'RandomForest',
        'XGBoost'
    ]
    target: Literal['home_away', 'margin'] = Field(
        'home_away',
        description="Target variable: 'home_away' trains separate models for home/away points (default), 'margin' trains single model on margin (home - away)"
    )

    # Hyperparameters (bounded ranges)
    alpha: Optional[float] = Field(None, ge=0.01, le=1000.0, description="Regularization parameter for Ridge/ElasticNet")
    l1_ratio: Optional[float] = Field(None, ge=0.0, le=1.0, description="L1 ratio for ElasticNet (0=Ridge, 1=Lasso)")
    n_estimators: Optional[int] = Field(None, ge=10, le=1000, description="Number of trees for ensemble methods")
    max_depth: Optional[int] = Field(None, ge=1, le=20, description="Max depth for tree models")

    @validator('alpha')
    def validate_alpha(cls, v, values):
        """Alpha only applies to Ridge and ElasticNet"""
        if v is not None and values.get('type') not in ['Ridge', 'ElasticNet']:
            raise ValueError(f"alpha only applies to Ridge and ElasticNet, not {values.get('type')}")
        return v

    @validator('l1_ratio')
    def validate_l1_ratio(cls, v, values):
        """L1 ratio only applies to ElasticNet"""
        if v is not None and values.get('type') != 'ElasticNet':
            raise ValueError(f"l1_ratio only applies to ElasticNet, not {values.get('type')}")
        return v


class FeatureConfig(BaseModel):
    """Feature selection configuration"""
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
    # 'home_minus_away': Use only diff features (home - away)
    # 'away_minus_home': Use only diff features (away - home, calculation reversed)
    # 'absolute': Use only home and away features (per-team absolute values)
    # 'mixed' or 'all': Use all feature types (diff, home, away) together
    diff_mode: Literal['home_minus_away', 'away_minus_home', 'absolute', 'mixed', 'all'] = 'home_minus_away'

    # Perspective flip augmentation
    flip_augment: bool = Field(False, description="Include perspective-flipped training examples")

    # PER features
    include_per: bool = Field(True, description="Include PER (Player Efficiency Rating) features")

    # Point prediction features (optional)
    point_model_id: Optional[str] = Field(
        None,
        description="Model ID for point predictions to merge. Only pred_margin is included as a feature by default. Other prediction columns (pred_home_points, pred_away_points, pred_point_total) are merged into the dataframe for reference but excluded from the feature set."
    )

    @validator('blocks')
    def validate_blocks(cls, v):
        """Validate feature block names"""
        valid_blocks = [
            'outcome_strength',
            'shooting_efficiency',
            'offensive_engine',
            'defensive_engine',
            'pace_volatility',
            'schedule_fatigue',
            'sample_size',
            'elo_strength',
            'era_normalization',
            'player_talent',
            'injuries'
        ]
        for block in v:
            if block not in valid_blocks:
                raise ValueError(f"Invalid feature block: {block}. Must be one of {valid_blocks}")
        return v


class SplitConfig(BaseModel):
    """Data splitting strategy"""
    type: Literal['time_split', 'rolling_cv', 'year_based_calibration'] = 'year_based_calibration'

    # Time split parameters
    test_size: Optional[float] = Field(None, ge=0.1, le=0.5, description="Test set size for time_split")

    # Rolling CV parameters
    n_splits: Optional[int] = Field(None, ge=3, le=10, description="Number of CV folds for rolling_cv")

    # Year-based calibration parameters
    train_end_year: Optional[int] = Field(default=2022, ge=2000, le=2100, description="Last year for training set (2012-2022 = 2012-2013 through 2021-2022 seasons)")
    calibration_years: Optional[List[int]] = Field(default=[2023], description="Years to use for calibration (2023 = 2023-2024 season)")
    evaluation_year: Optional[int] = Field(default=2024, ge=2000, le=2100, description="Year to use for evaluation (2024 = 2024-2025 season)")

    # Time window constraints
    begin_year: Optional[int] = Field(2012, ge=2000, le=2100, description="Earliest year to include (2012 = 2012-2013 season)")
    min_games_played: Optional[int] = Field(15, ge=0, le=100, description="Minimum games per team before including (default: 15, matches model-config UI)")
    exclude_seasons: Optional[List[int]] = Field(None, description="Season start years to exclude from training (e.g. [2019] excludes 2019-2020 season)")

    @validator('type')
    def validate_split_type(cls, v, values):
        """Validate split-specific parameters"""
        if v == 'time_split' and values.get('test_size') is None:
            values['test_size'] = 0.2  # Default
        elif v == 'rolling_cv' and values.get('n_splits') is None:
            values['n_splits'] = 5  # Default
        elif v == 'year_based_calibration':
            # Set defaults if not provided
            if values.get('train_end_year') is None:
                values['train_end_year'] = 2022
            if values.get('calibration_years') is None:
                values['calibration_years'] = [2023]
            if values.get('evaluation_year') is None:
                values['evaluation_year'] = 2024
        return v


class PreprocessingConfig(BaseModel):
    """Data preprocessing configuration"""
    scaler: Literal['standard', 'none'] = 'standard'
    impute: Literal['median', 'mean', 'zero', 'none'] = 'median'
    clip_outliers: bool = Field(False, description="Clip outliers to 3 standard deviations")
    clip_range: Optional[tuple] = Field(None, description="Custom clip range (min, max)")


class ConstraintsConfig(BaseModel):
    """Resource and data constraints"""
    max_features: Optional[int] = Field(None, ge=1, le=1000, description="Maximum number of features")
    max_train_rows: Optional[int] = Field(None, ge=100, le=1000000, description="Maximum training rows")
    max_time_window_days: Optional[int] = Field(None, ge=1, le=3650, description="Maximum time window in days")


class StackingConfig(BaseModel):
    """Configuration for stacking ensemble models"""
    base_run_ids: List[str] = Field(
        ...,
        min_items=2,
        description="List of run_ids for base models to stack (must have at least 2 models)"
    )
    meta_model_type: Literal['LogisticRegression'] = Field(
        'LogisticRegression',
        description="Meta-model type (always LogisticRegression for stacking best practices)"
    )
    meta_c_value: Optional[float] = Field(
        0.1,
        ge=0.001,
        le=100.0,
        description="C-value for meta-model Logistic Regression (default: 0.1)"
    )
    stacking_mode: Literal['naive', 'informed'] = Field(
        'naive',
        description="Stacking mode: 'naive' uses only base model predictions, 'informed' adds derived features (disagreements, confidences) and optional user features"
    )
    meta_features: Optional[List[str]] = Field(
        None,
        description="Optional list of feature names from dataset to include in meta-model (only used when stacking_mode='informed')"
    )


class ExperimentConfig(BaseModel):
    """Complete experiment configuration"""
    task: Literal['binary_home_win', 'points_regression', 'stacking'] = 'binary_home_win'

    # Model config - different types for classification vs regression
    model: Optional[ModelConfig] = Field(None, description="Model config for classification (required if task='binary_home_win')")
    points_model: Optional[PointsRegressionModelConfig] = Field(None, description="Model config for regression (required if task='points_regression')")
    stacking: Optional[StackingConfig] = Field(None, description="Stacking config (required if task='stacking')")

    features: FeatureConfig
    splits: SplitConfig
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    constraints: Optional[ConstraintsConfig] = Field(None)

    # Calibration (only for classification)
    use_time_calibration: bool = Field(True, description="Use time-based probability calibration")
    calibration_method: Literal['isotonic', 'sigmoid'] = 'sigmoid'

    # Metadata
    description: Optional[str] = Field(None, description="Human-readable description of experiment")
    tags: Optional[List[str]] = Field(None, description="Tags for organizing experiments")

    @validator('model')
    def validate_model_for_classification(cls, v, values):
        """Ensure model is provided for classification task"""
        task = values.get('task', 'binary_home_win')
        if task == 'binary_home_win' and v is None:
            raise ValueError("model is required when task='binary_home_win'")
        return v

    @validator('points_model')
    def validate_points_model_for_regression(cls, v, values):
        """Ensure points_model is provided for regression task"""
        task = values.get('task', 'binary_home_win')
        if task == 'points_regression' and v is None:
            raise ValueError("points_model is required when task='points_regression'")
        return v

    @validator('stacking')
    def validate_stacking_for_task(cls, v, values):
        """Ensure stacking is provided for stacking task"""
        task = values.get('task', 'binary_home_win')
        if task == 'stacking' and v is None:
            raise ValueError("stacking is required when task='stacking'")
        return v

    class Config:
        extra = 'forbid'  # Don't allow extra fields


class DatasetSpec(BaseModel):
    """Dataset specification for build_dataset()"""
    label: Literal['home_win'] = 'home_win'
    unit: Literal['game'] = 'game'

    # Feature selection
    feature_blocks: List[str] = Field(default_factory=list)
    individual_features: Optional[List[str]] = None

    # Time window (begin_year=2012 means 2012-2013 season)
    begin_year: Optional[int] = Field(default=2012, ge=2000, le=2100, description="Earliest year to include (2012 = 2012-2013 season)")
    end_year: Optional[int] = None
    begin_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None  # YYYY-MM-DD

    # Filtering
    min_games_played: Optional[int] = None
    exclude_preseason: bool = True
    exclude_seasons: Optional[List[int]] = None

    # Diffing
    # 'home_minus_away': Use only diff features (home - away)
    # 'away_minus_home': Use only diff features (away - home, calculation reversed)
    # 'absolute': Use only home and away features (per-team absolute values)
    # 'mixed' or 'all': Use all feature types (diff, home, away) together
    diff_mode: Literal['home_minus_away', 'away_minus_home', 'absolute', 'mixed', 'all'] = 'home_minus_away'

    # PER features
    include_per: bool = True

    # Point prediction features (optional)
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
