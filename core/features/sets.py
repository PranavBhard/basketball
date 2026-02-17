"""
Feature Sets Module - Derives from FeatureRegistry SSoT

This module provides feature sets derived from FeatureGroups in feature_registry.py.
All feature enumeration is done dynamically from the registry - no manual feature lists.

The FeatureRegistry (feature_registry.py) is the Single Source of Truth for:
- All valid stat definitions
- All valid feature combinations per group
- Group descriptions and layer organization

This module provides convenience functions for accessing features by group/layer,
with optional filtering by model type or availability in master training CSV.
"""

from typing import List, Dict, Set, Optional
from bball_app.core.features.registry import FeatureRegistry, FeatureGroups


# =============================================================================
# FEATURE SETS - Derived from FeatureGroups (SSoT)
# =============================================================================

def _build_feature_sets() -> Dict[str, List[str]]:
    """Build FEATURE_SETS from FeatureGroups (SSoT)."""
    return FeatureGroups.get_all_features(include_side=True)


# FEATURE_SETS is derived from FeatureGroups - the SSoT
# This is computed once at import time
FEATURE_SETS: Dict[str, List[str]] = _build_feature_sets()


# =============================================================================
# FEATURE SET DESCRIPTIONS - Derived from FeatureGroups
# =============================================================================

FEATURE_SET_DESCRIPTIONS: Dict[str, str] = {
    group_name: FeatureGroups.get_group_description(group_name)
    for group_name in FeatureGroups.get_all_groups()
}


# =============================================================================
# LAYER DEFINITIONS - Derived from FeatureGroups
# =============================================================================

FEATURE_LAYERS: Dict[str, List[str]] = {
    f"layer_{layer}": FeatureGroups.get_groups_by_layer(layer)
    for layer in [1, 2, 3, 4]
    if FeatureGroups.get_groups_by_layer(layer)  # Only include non-empty layers
}

LAYER_DESCRIPTIONS: Dict[str, str] = {
    "layer_1": "Outcome & ratings - core strength comparison",
    "layer_2": "Tempo, schedule, sample size - situational context",
    "layer_3": "Meta priors & normalization - global calibration",
    "layer_4": "Player talent & absolute magnitudes - detailed structure",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_diff_feature(feature_name: str) -> bool:
    """
    Determine if a feature is a differential (diff) feature.

    Features with |diff as the 4th component (e.g., 'points|season|avg|diff')
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if parsed:
        return parsed['perspective'] == 'diff'
    return False


def is_per_team_feature(feature_name: str) -> bool:
    """
    Determine if a feature is a per-team feature (home/away separate).

    Features with |home or |away as the 4th component (e.g., 'points|season|avg|home')
    Diff features are NOT per-team.
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if parsed:
        return parsed['perspective'] in ('home', 'away')
    return False


def is_matchup_scalar_feature(feature_name: str) -> bool:
    """
    Matchup-scalar feature: the perspective is 'none' (applies to the whole matchup).
    Example: 'pace_interaction|season|harmonic_mean|none'
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if parsed:
        return parsed['perspective'] == 'none'
    return False


def filter_features_by_model_type(features: List[str], model_type: Optional[str] = None) -> List[str]:
    """
    Filter features based on model type requirements.

    Args:
        features: List of feature names to filter
        model_type: One of 'LogisticRegression', 'Tree', 'NeuralNetwork', or None
                   If None, returns all features (no filtering)

    Returns:
        Filtered list of features based on model type:
        - LogisticRegression: Only diff features
        - Tree/NeuralNetwork: Only per-team features
        - None: All features (no filtering)
    """
    if model_type is None:
        return features

    model_type_lower = model_type.lower()

    # LogisticRegression: only diffs
    if 'logistic' in model_type_lower or 'linear' in model_type_lower:
        # Include matchup-scalar features (|none) alongside diffs
        return [f for f in features if is_diff_feature(f) or is_matchup_scalar_feature(f)]

    # Tree models and Neural Networks: only per-team
    if 'tree' in model_type_lower or 'neural' in model_type_lower or 'xgboost' in model_type_lower or \
       'lightgbm' in model_type_lower or 'catboost' in model_type_lower or 'gradient' in model_type_lower or \
       'randomforest' in model_type_lower or 'svm' in model_type_lower:
        # Include matchup-scalar features (|none) alongside per-team
        return [f for f in features if is_per_team_feature(f) or is_matchup_scalar_feature(f)]

    # Default: return all (for unknown model types)
    return features


def get_all_features(model_type: Optional[str] = None, league=None) -> List[str]:
    """
    Get all features from all sets, optionally filtered by model type.

    Args:
        model_type: Optional model type to filter features ('LogisticRegression', 'Tree', 'NeuralNetwork')
        league: Optional LeagueConfig to include league-specific features

    Returns:
        List of feature names, filtered if model_type is provided
    """
    all_features = FeatureGroups.get_all_features_flat(include_side=True, league=league)
    return filter_features_by_model_type(all_features, model_type)


def get_features_by_sets(set_names: List[str], model_type: Optional[str] = None) -> List[str]:
    """
    Get features for specified sets, optionally filtered by model type.

    Args:
        set_names: List of feature set names to include
        model_type: Optional model type to filter features ('LogisticRegression', 'Tree', 'NeuralNetwork')

    Returns:
        List of feature names, filtered if model_type is provided
    """
    features = []
    for set_name in set_names:
        features.extend(FeatureGroups.get_features_for_group(set_name, include_side=True))

    return filter_features_by_model_type(features, model_type)


def get_features_excluding_sets(excluded_sets: List[str], model_type: Optional[str] = None) -> List[str]:
    """
    Get all features except those in excluded sets, optionally filtered by model type.

    Args:
        excluded_sets: List of feature set names to exclude
        model_type: Optional model type to filter features ('LogisticRegression', 'Tree', 'NeuralNetwork')

    Returns:
        List of feature names, filtered if model_type is provided
    """
    all_features = get_all_features(model_type=model_type)
    excluded_features = []
    for set_name in excluded_sets:
        excluded_features.extend(FeatureGroups.get_features_for_group(set_name, include_side=True))

    # Filter excluded features by model type too
    excluded_features = filter_features_by_model_type(excluded_features, model_type)

    return [f for f in all_features if f not in excluded_features]


def get_set_name_for_feature(feature_name: str) -> Optional[str]:
    """
    Get the feature set name for a given feature.

    Uses FeatureGroups.get_group_for_feature() - the SSoT for feature categorization.

    Args:
        feature_name: Name of the feature

    Returns:
        Feature set name, or 'other' if not matched to a specific group
    """
    return FeatureGroups.get_group_for_feature(feature_name)


def get_feature_set_info() -> Dict[str, int]:
    """
    Get information about all feature sets.

    Returns:
        Dict mapping set names to their feature counts
    """
    return {
        set_name: len(FeatureGroups.get_features_for_group(set_name, include_side=True))
        for set_name in FeatureGroups.get_all_groups()
    }


def get_features_by_layers(layer_names: List[str], model_type: Optional[str] = None) -> List[str]:
    """
    Get features for specified layers, optionally filtered by model type.

    Args:
        layer_names: List of layer names (e.g., ['layer_1', 'layer_2'])
        model_type: Optional model type to filter features ('LogisticRegression', 'Tree', 'NeuralNetwork')

    Returns:
        List of feature names, filtered if model_type is provided
    """
    feature_sets = []
    for layer_name in layer_names:
        if layer_name in FEATURE_LAYERS:
            feature_sets.extend(FEATURE_LAYERS[layer_name])

    return get_features_by_sets(feature_sets, model_type=model_type)


def get_layer_info() -> Dict[str, dict]:
    """
    Get information about all layers.

    Returns:
        Dict mapping layer names to their feature counts and set names
    """
    info = {}
    for layer_name, set_names in FEATURE_LAYERS.items():
        total_features = sum(
            len(FeatureGroups.get_features_for_group(set_name, include_side=True))
            for set_name in set_names
        )
        info[layer_name] = {
            'sets': set_names,
            'feature_count': total_features,
            'description': LAYER_DESCRIPTIONS.get(layer_name, '')
        }
    return info


def get_common_layer_configs() -> Dict[str, List[str]]:
    """
    Get common layer configuration presets for testing.

    Returns:
        Dict mapping config names to layer lists
    """
    return {
        'layer_1_only': ['layer_1'],
        'layer_1_2': ['layer_1', 'layer_2'],
        'layer_1_2_3': ['layer_1', 'layer_2', 'layer_3'],
        'layer_1_2_4': ['layer_1', 'layer_2', 'layer_4'],
        'all_layers': ['layer_1', 'layer_2', 'layer_3', 'layer_4'],
        'core_only': ['layer_1', 'layer_3'],  # Core strength + meta priors
        'context_heavy': ['layer_1', 'layer_2', 'layer_4'],  # Skip era normalization
    }


def find_features_by_substrings(features: List[str], substrings: List[str], match_mode: str = 'OR') -> List[str]:
    """
    Find features that match any or all of the given substrings.

    Args:
        features: List of all feature names to search
        substrings: List of substrings to match against (case-insensitive)
        match_mode: 'OR' to match features containing ANY substring, 'AND' to match features containing ALL substrings

    Returns:
        List of matching feature names
    """
    if not substrings:
        return []

    # Normalize substrings to lowercase for case-insensitive matching
    substrings_lower = [s.lower().strip() for s in substrings if s.strip()]

    if not substrings_lower:
        return []

    matching_features = []

    for feature in features:
        feature_lower = feature.lower()

        if match_mode.upper() == 'AND':
            # AND mode: feature must contain ALL substrings
            if all(substring in feature_lower for substring in substrings_lower):
                matching_features.append(feature)
        else:
            # OR mode (default): feature must contain ANY substring
            if any(substring in feature_lower for substring in substrings_lower):
                matching_features.append(feature)

    return matching_features
