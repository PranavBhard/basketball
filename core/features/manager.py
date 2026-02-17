"""
Feature Management - Simplified interface for FeatureRegistry.

The FeatureRegistry is the SSoT for all feature definitions.
This module provides a simplified interface for common operations.
"""

from typing import Dict, List, Optional
import hashlib

from bball_app.core.features.registry import FeatureRegistry, FeatureGroups


class FeatureManager:
    """
    Feature management wrapper around FeatureRegistry.

    For full feature definition access, use FeatureRegistry directly:
        from bball_app.core.feature_registry import FeatureRegistry
    """

    @staticmethod
    def validate_features(features: List[str]) -> Dict[str, any]:
        """
        Validate feature names against the FeatureRegistry.

        Args:
            features: List of feature names to validate

        Returns:
            Dictionary with validation results including 'valid', 'errors', 'invalid_features'
        """
        return FeatureRegistry.validate_feature_list(features)

    @staticmethod
    def validate_feature(feature_name: str) -> tuple:
        """
        Validate a single feature name.

        Args:
            feature_name: Feature name to validate

        Returns:
            (is_valid, error_message) tuple
        """
        return FeatureRegistry.validate_feature(feature_name)

    @staticmethod
    def get_stat_definition(stat_name: str):
        """Get the definition for a stat from the registry."""
        return FeatureRegistry.get_stat_definition(stat_name)

    @staticmethod
    def get_all_stat_names() -> List[str]:
        """Get all valid stat names from the registry."""
        return FeatureRegistry.get_all_stat_names()

    @staticmethod
    def get_rate_stats() -> set:
        """Get stats that require rate calculation."""
        return FeatureRegistry.get_rate_stats()

    @staticmethod
    def get_side_splittable_stats() -> set:
        """Get stats that support home/away side splitting."""
        return FeatureRegistry.get_side_splittable_stats()

    @staticmethod
    def get_net_stats() -> set:
        """Get stats that have opponent-adjusted versions."""
        return FeatureRegistry.get_net_stats()

    @staticmethod
    def get_db_field(stat_name: str) -> str:
        """Get the MongoDB field name for a stat."""
        return FeatureRegistry.get_db_field(stat_name)

    @staticmethod
    def build_feature_name(
        stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        side_split: bool = False
    ) -> str:
        """Build a valid feature name from components."""
        return FeatureRegistry.build_feature_name(
            stat_name, time_period, calc_weight, perspective, side_split
        )

    @staticmethod
    def parse_feature_name(feature_name: str) -> Optional[Dict[str, str]]:
        """Parse a feature name into its components."""
        return FeatureRegistry.parse_feature_name(feature_name)

    @staticmethod
    def generate_feature_set_hash(features: List[str]) -> str:
        """
        Generate hash for feature set identification.

        Args:
            features: List of feature names

        Returns:
            MD5 hash string
        """
        sorted_features = sorted(features)
        feature_str = '|'.join(sorted_features)
        return hashlib.md5(feature_str.encode()).hexdigest()

    @staticmethod
    def get_feature_groups() -> List[str]:
        """Get all feature group names."""
        return FeatureGroups.get_all_groups()

    @staticmethod
    def get_group_stats(group_name: str) -> List[str]:
        """Get stats in a feature group."""
        return FeatureGroups.get_group_stats(group_name)

    @staticmethod
    def get_valid_time_periods() -> List[str]:
        """Get all valid time periods."""
        return FeatureRegistry.get_all_time_periods()

    @staticmethod
    def get_valid_calc_weights() -> List[str]:
        """Get all valid calc weights."""
        return FeatureRegistry.get_all_calc_weights()

    @staticmethod
    def get_valid_perspectives() -> List[str]:
        """Get all valid perspectives."""
        return FeatureRegistry.get_all_perspectives()
