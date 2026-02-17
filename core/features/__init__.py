"""
Feature System Module.

This module contains all feature-related functionality:
- FeatureRegistry: Central registry of all available features
- FeatureGroups: Feature groupings by layer and category
- Feature parsing, dependencies, and generation

Note: generator and manager are not imported here to avoid circular imports.
Import them directly: from bball_app.core.features.generator import SharedFeatureGenerator
"""

# Registry - central feature definitions
from bball_app.core.features.registry import FeatureRegistry, FeatureGroups

# Feature sets
from bball_app.core.features.sets import FEATURE_SETS

# Parser utilities
from bball_app.core.features.parser import parse_feature_name

# Dependencies
from bball_app.core.features.dependencies import resolve_dependencies, categorize_features

# Prediction feature mapping
from bball_app.core.features.prediction_mapping import (
    is_pred_feature,
    parse_pred_feature,
    validate_pred_feature_model_type,
    get_pred_internal_columns,
    get_all_pred_features,
    PRED_FEATURE_MAP,
    LEGACY_PRED_COLUMNS,
)

# Note: generator and manager omitted to avoid circular imports with stats module
# Use direct imports: from bball_app.core.features.generator import SharedFeatureGenerator
# Use direct imports: from bball_app.core.features.manager import FeatureManager

__all__ = [
    # Registry
    'FeatureRegistry',
    'FeatureGroups',
    # Sets
    'FEATURE_SETS',
    # Parser
    'parse_feature_name',
    # Dependencies
    'resolve_dependencies',
    'categorize_features',
    # Prediction mapping
    'is_pred_feature',
    'parse_pred_feature',
    'validate_pred_feature_model_type',
    'get_pred_internal_columns',
    'get_all_pred_features',
    'PRED_FEATURE_MAP',
    'LEGACY_PRED_COLUMNS',
]
