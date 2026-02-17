#!/usr/bin/env python3
"""
Feature Name Parser - Core Infrastructure

Parse and validate feature names in the pipe-delimited format:
    <stat_name>|<time_period>|<calc_weight>|<perspective>[|side]

This module provides functions to:
1. Parse feature names into components
2. Convert parsed components back to feature names
3. Validate feature name format (using feature_registry as SSoT)
"""

from typing import Dict, Optional, Tuple, List
import re

from bball_app.core.features.registry import FeatureRegistry


class FeatureNameComponents:
    """Container for parsed feature name components."""

    def __init__(
        self,
        stat_name: str,
        time_period: str,
        calc_weight: str,
        home_away_diff: str,
        is_side: bool = False
    ):
        self.stat_name = stat_name
        self.time_period = time_period
        self.calc_weight = calc_weight
        self.home_away_diff = home_away_diff  # 'home', 'away', 'diff', or 'none'
        self.is_side = is_side

    def to_feature_name(self) -> str:
        """Convert components back to feature name."""
        parts = [
            self.stat_name,
            self.time_period,
            self.calc_weight,
            self.home_away_diff
        ]

        if self.is_side:
            parts.append('side')

        return '|'.join(parts)

    def __repr__(self):
        return f"FeatureNameComponents({self.to_feature_name()})"


def parse_feature_name(feature_name: str) -> Optional[FeatureNameComponents]:
    """
    Parse a feature name in the pipe-delimited format.

    Format: <stat_name>|<time_period>|<calc_weight>|<perspective>[|side]

    Args:
        feature_name: Feature name to parse

    Returns:
        FeatureNameComponents object, or None if parsing fails
    """
    if not feature_name or '|' not in feature_name:
        return None

    parts = feature_name.split('|')

    # Minimum 4 parts required (stat_name, time_period, calc_weight, perspective)
    if len(parts) < 4:
        return None

    stat_name = parts[0]
    time_period = parts[1]
    calc_weight = parts[2]
    home_away_diff = parts[3]

    # Check for optional side flag (5th part)
    is_side = len(parts) > 4 and parts[4] == 'side'

    # Validate perspective
    if home_away_diff not in ['home', 'away', 'diff', 'none']:
        return None

    return FeatureNameComponents(
        stat_name=stat_name,
        time_period=time_period,
        calc_weight=calc_weight,
        home_away_diff=home_away_diff,
        is_side=is_side
    )


def validate_feature_name(feature_name: str, strict: bool = False) -> bool:
    """
    Validate that a feature name follows the pipe-delimited format.

    Args:
        feature_name: Feature name to validate
        strict: If True, also validate against FeatureRegistry SSoT.
                If False, only validate format structure.

    Returns:
        True if valid, False otherwise
    """
    components = parse_feature_name(feature_name)
    if components is None:
        return False

    if strict:
        # Full validation against FeatureRegistry SSoT
        is_valid, _ = FeatureRegistry.validate_feature(feature_name)
        return is_valid

    return True


def validate_feature_name_strict(feature_name: str) -> Tuple[bool, Optional[str]]:
    """
    Strictly validate a feature name against the FeatureRegistry.

    This validates:
    - Format structure (4+ pipe-delimited parts)
    - Stat name exists in registry
    - Time period is valid for stat
    - Calc weight is valid for stat
    - Perspective is valid for stat

    Args:
        feature_name: Feature name to validate

    Returns:
        (is_valid, error_message) tuple
    """
    return FeatureRegistry.validate_feature(feature_name)


def get_feature_type(feature_name: str) -> Optional[str]:
    """
    Determine the type of feature based on FeatureRegistry categorization.

    Returns:
        Feature type string: 'basic', 'rate', 'net', 'derived', 'special', or None
    """
    components = parse_feature_name(feature_name)
    if not components:
        return None

    stat_name = components.stat_name

    # Use FeatureRegistry as SSoT for categorization
    stat_def = FeatureRegistry.get_stat_definition(stat_name)
    if stat_def:
        return stat_def.category.value

    # Check for _net variants
    if stat_name.endswith('_net'):
        base_stat = stat_name[:-4]
        if base_stat in FeatureRegistry.get_net_stats():
            return 'net'

    # Fallback categorization for stats not in registry
    if stat_name in FeatureRegistry.get_basic_stats():
        return 'basic'
    if stat_name in FeatureRegistry.get_rate_stats():
        return 'rate'
    if stat_name in FeatureRegistry.get_derived_stats():
        return 'derived'
    if stat_name in FeatureRegistry.get_special_stats():
        return 'special'

    return 'unknown'


def filter_features_by_type(features: List[str], feature_type: str) -> List[str]:
    """Filter features by type."""
    return [f for f in features if get_feature_type(f) == feature_type]


def is_diff_feature(feature_name: str) -> bool:
    """Check if feature is a differential feature (home - away)."""
    components = parse_feature_name(feature_name)
    return components is not None and components.home_away_diff == 'diff'


def is_per_team_feature(feature_name: str) -> bool:
    """Check if feature is a per-team feature (home/away separate)."""
    components = parse_feature_name(feature_name)
    return components is not None and components.home_away_diff in ['home', 'away']


def is_side_feature(feature_name: str) -> bool:
    """Check if feature is a side-split feature."""
    components = parse_feature_name(feature_name)
    return components is not None and components.is_side
