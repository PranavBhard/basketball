"""
Feature Registry - Single Source of Truth for NBA Feature Definitions

This module provides:
1. Complete catalog of all valid stat names with metadata (loaded from stats.yaml)
2. Valid time periods, calc weights, and perspectives
3. Stat categorization (rate stats, side-splittable stats, etc.)
4. DB field mappings for stat computation
5. Feature name generation and validation

Feature groupings (FeatureGroups) have been moved to bball.features.groups
but are re-exported here for backward compatibility.

All other modules (StatHandlerV2, BballModel, feature_sets, feature_name_parser)
should import from this registry rather than defining their own stat lists.
"""

import os
from typing import Dict, List, Set, Optional, Tuple

from sportscore.features.base_registry import (
    StatCategory,
    CalcWeight,
    Perspective,
    StatDefinition,
    BaseFeatureRegistry,
)
from sportscore.features.stat_loader import load_stat_definitions

# Backward compatibility: FeatureGroups moved to bball.features.groups
from bball.features.groups import FeatureGroups  # noqa: F401


def _stats_yaml_path():
    """Get the path to stats.yaml relative to this module."""
    return os.path.join(os.path.dirname(__file__), "stats.yaml")


class FeatureRegistry(BaseFeatureRegistry):
    """
    Single Source of Truth for all feature definitions.

    Stat definitions are loaded declaratively from stats.yaml.
    Time period validation, calc weight validation, and feature name
    parsing remain in Python (blend/delta parsing, parameterized patterns).

    Usage:
        from bball.feature_registry import FeatureRegistry

        # Get all valid stat names
        stats = FeatureRegistry.get_all_stat_names()

        # Validate a feature name
        valid, error = FeatureRegistry.validate_feature("points|season|avg|diff")

        # Get stat definition
        stat_def = FeatureRegistry.get_stat_definition("efg")

        # Get DB field mapping
        db_field = FeatureRegistry.get_db_field("efg")  # Returns "effective_fg_perc"
    """

    # ==========================================================================
    # CORE STAT DEFINITIONS (loaded from stats.yaml)
    # ==========================================================================

    STAT_DEFINITIONS: Dict[str, StatDefinition] = load_stat_definitions(_stats_yaml_path())

    # ==========================================================================
    # VALID TIME PERIODS
    # ==========================================================================

    # Base time periods (non-parameterized)
    BASE_TIME_PERIODS: Set[str] = {"season", "none"}

    # Parameterized time period prefixes (accept any integer N)
    # e.g., games_10, games_20, days_5, months_1, games_close5, last_3 (for H2H)
    TIME_PERIOD_PREFIXES: Set[str] = {"games_", "days_", "months_", "games_close", "last_"}

    @classmethod
    def is_valid_time_period(cls, tp: str) -> bool:
        """
        Check if a time period is valid.

        Valid formats:
        - "season", "none" (base periods)
        - "games_N", "days_N", "months_N" where N is any positive integer
        """
        if tp in cls.BASE_TIME_PERIODS:
            return True
        for prefix in cls.TIME_PERIOD_PREFIXES:
            if tp.startswith(prefix):
                suffix = tp[len(prefix):]
                if suffix.isdigit() and int(suffix) > 0:
                    return True
        # Composite time periods: blend and delta
        if tp.startswith('blend:'):
            parsed = cls.parse_blend_time_period(tp)
            if parsed is None:
                return False
            # Validate each component is a valid base time period and weights sum to ~1.0
            total_weight = 0.0
            for comp_tp, weight in parsed['components']:
                if not cls.is_valid_time_period(comp_tp):
                    return False
                total_weight += weight
            if abs(total_weight - 1.0) > 0.01:
                return False
            return True

        if tp.startswith('delta:'):
            parsed = cls.parse_delta_time_period(tp)
            if parsed is None:
                return False
            # Validate all leaf time periods
            for leaf_tp in cls._extract_leaf_time_periods(tp):
                if not cls.is_valid_time_period(leaf_tp):
                    return False
            return True

        return False

    @classmethod
    def parse_blend_time_period(cls, time_period: str) -> dict:
        """
        Parse a blend time_period string.

        Input:  "blend:games_5:0.70/games_10:0.30"
        Output: {"type": "blend", "components": [("games_5", 0.70), ("games_10", 0.30)]}

        Returns None if malformed.
        """
        if not time_period.startswith('blend:'):
            return None

        blend_str = time_period[6:]  # Remove 'blend:' prefix
        parts = blend_str.split('/')

        components = []
        for part in parts:
            if ':' not in part:
                return None
            # Split on last ':' to get (tp, weight) — handles tp names with colons
            idx = part.rfind(':')
            tp = part[:idx]
            weight_str = part[idx + 1:]
            try:
                weight = float(weight_str)
            except ValueError:
                return None
            components.append((tp, weight))

        if not components:
            return None

        return {"type": "blend", "components": components}

    @classmethod
    def parse_delta_time_period(cls, time_period: str) -> dict:
        """
        Parse a delta time_period string.

        Input:  "delta:games_5-season"
        Output: {"type": "delta", "recent": "games_5", "baseline": "season"}

        Input:  "delta:blend:games_5:0.70/games_10:0.30-season"
        Output: {"type": "blend_delta",
                 "recent": {"type": "blend", "components": [("games_5", 0.70), ("games_10", 0.30)]},
                 "baseline": "season"}

        Returns None if malformed.
        """
        if not time_period.startswith('delta:'):
            return None

        delta_str = time_period[6:]  # Remove 'delta:' prefix

        # Use rfind('-') to split recent from baseline (baseline is always a simple tp)
        sep_idx = delta_str.rfind('-')
        if sep_idx <= 0 or sep_idx == len(delta_str) - 1:
            return None

        recent_str = delta_str[:sep_idx]
        baseline = delta_str[sep_idx + 1:]

        # Check if recent is itself a blend
        if recent_str.startswith('blend:'):
            blend_parsed = cls.parse_blend_time_period(recent_str)
            if blend_parsed is None:
                return None
            return {
                "type": "blend_delta",
                "recent": blend_parsed,
                "baseline": baseline,
            }

        return {
            "type": "delta",
            "recent": recent_str,
            "baseline": baseline,
        }

    @classmethod
    def _extract_leaf_time_periods(cls, time_period: str) -> list:
        """
        Extract all base (leaf) time periods from any composite time period.

        "season"                                        -> ["season"]
        "blend:games_5:0.70/games_10:0.30"              -> ["games_5", "games_10"]
        "delta:games_5-season"                           -> ["games_5", "season"]
        "delta:blend:games_5:0.70/games_10:0.30-season"  -> ["games_5", "games_10", "season"]
        """
        if time_period.startswith('blend:'):
            parsed = cls.parse_blend_time_period(time_period)
            if parsed is None:
                return [time_period]
            return [tp for tp, _ in parsed['components']]

        if time_period.startswith('delta:'):
            parsed = cls.parse_delta_time_period(time_period)
            if parsed is None:
                return [time_period]

            leaves = []
            if parsed['type'] == 'blend_delta':
                # Recent is a blend
                leaves.extend(tp for tp, _ in parsed['recent']['components'])
            else:
                leaves.append(parsed['recent'])
            leaves.append(parsed['baseline'])
            return leaves

        return [time_period]

    # Legacy set for backwards compatibility (common time periods)
    VALID_TIME_PERIODS: Set[str] = {
        "season", "games_2", "games_3", "games_5", "games_10", "games_12", "games_20",
        "months_1", "days_2", "days_3", "days_5", "days_12", "none",
        "games_close5",
        "last_3", "last_5",  # H2H cross-season lookups
    }

    # ==========================================================================
    # VALID CALC WEIGHTS
    # ==========================================================================

    VALID_CALC_WEIGHTS: Set[str] = {
        "raw",
        "avg",
        "std",
        "weighted_MPG",
        "weighted_MIN",
        "weighted_MIN_REC",
        "harmonic_mean",
        "derived",
        "top1", "top1_avg", "top2_avg", "top3_avg",
        "top1_weighted_MPG", "top2_weighted_MPG", "top3_weighted_MPG",
        "top3_sum",
        "top1_share", "top3_share",  # Star share (non-recency)
        # Model types for prediction features (pred_margin|none|ridge|none)
        "ridge", "elasticnet", "randomforest", "xgboost",
        # H2H shrinkage/reliability calc weights
        "beta",  # Beta-prior-smoothed win probability
        "eb",    # Empirical Bayes shrinkage for margin
        "logw",  # Log-weighted margin (scales by sample size)
        # Binary features
        "binary",  # Binary indicator (0 or 1)
        # Aggregation types
        "sum",  # Sum/total (e.g., total travel distance)
    }

    # Parameterized calc weight patterns (accept integer k values)
    # e.g., weighted_MIN_REC(k=20), derived(k=50), top1_MIN_REC(k=25)
    CALC_WEIGHT_PATTERNS: Set[str] = {
        "weighted_MIN_REC",      # weighted_MIN_REC(k=N)
        "derived",               # derived(k=N)
        "top1_MIN_REC",          # top1_MIN_REC(k=N)
        "top3_MIN_REC",          # top3_MIN_REC(k=N)
        "top1_share_MIN_REC",    # top1_share_MIN_REC(k=N)
        "top3_share_MIN_REC",    # top3_share_MIN_REC(k=N)
    }

    @classmethod
    def is_parameterized_calc_weight(cls, calc_weight: str) -> bool:
        """
        Check if calc_weight is a valid parameterized format.

        Valid formats: pattern(k=N) where pattern is in CALC_WEIGHT_PATTERNS
        and N is a positive integer.
        """
        import re
        match = re.match(r'^(.+)\(k=(\d+)\)$', calc_weight)
        if not match:
            return False
        pattern, k_val = match.groups()
        return pattern in cls.CALC_WEIGHT_PATTERNS and int(k_val) > 0

    # Note: blend:* is validated separately via is_blend_format()

    # ==========================================================================
    # VALID PERSPECTIVES
    # ==========================================================================

    VALID_PERSPECTIVES: Set[str] = {"diff", "home", "away", "none"}

    # ==========================================================================
    # STAT CATEGORIES FOR COMPUTATION
    # ==========================================================================

    @classmethod
    def get_rate_stats(cls) -> Set[str]:
        """Get stats that require aggregation before computing rate."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.requires_aggregation or defn.category == StatCategory.RATE
        }

    @classmethod
    def get_side_splittable_stats(cls) -> Set[str]:
        """Get stats that can be split by home/away side."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.supports_side_split
        }

    @classmethod
    def get_net_stats(cls) -> Set[str]:
        """Get stats that have opponent-adjusted (*_net) versions."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.supports_net
        }

    @classmethod
    def get_basic_stats(cls) -> Set[str]:
        """Get basic counting stats."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.BASIC
        }

    @classmethod
    def get_derived_stats(cls) -> Set[str]:
        """Get derived/computed stats."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.DERIVED
        }

    @classmethod
    def get_special_stats(cls) -> Set[str]:
        """Get stats requiring special handling."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.SPECIAL
        }

    # ==========================================================================
    # DB FIELD MAPPING
    # ==========================================================================

    @classmethod
    def get_db_field(cls, stat_name: str) -> str:
        """
        Get the MongoDB field name for a stat.

        Args:
            stat_name: Canonical stat name (e.g., "efg")

        Returns:
            DB field name (e.g., "effective_fg_perc")
        """
        defn = cls.STAT_DEFINITIONS.get(stat_name)
        if defn and defn.db_field:
            return defn.db_field
        return stat_name  # Default: use stat name as-is

    @classmethod
    def get_stat_name_map(cls) -> Dict[str, str]:
        """
        Get the complete stat name to DB field mapping.

        Used by StatHandlerV2 for computation.

        Returns:
            Dict mapping stat names to DB field names
        """
        return {
            name: defn.db_field if defn.db_field else name
            for name, defn in cls.STAT_DEFINITIONS.items()
        }

    # ==========================================================================
    # FEATURE NAME GENERATION
    # ==========================================================================

    @classmethod
    def build_feature_name(
        cls,
        stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        side_split: bool = False
    ) -> str:
        """
        Build a valid feature name from components.

        Args:
            stat_name: Base stat name (e.g., "points")
            time_period: Time period (e.g., "season", "games_10")
            calc_weight: Calculation weight (e.g., "raw", "avg")
            perspective: Perspective (e.g., "diff", "home")
            side_split: Whether to add |side suffix

        Returns:
            Formatted feature name (e.g., "points|season|avg|diff|side")
        """
        parts = [stat_name, time_period, calc_weight, perspective]
        if side_split:
            parts.append("side")
        return "|".join(parts)

    @classmethod
    def parse_feature_name(cls, feature_name: str) -> Optional[Dict[str, str]]:
        """
        Parse a feature name into its components.

        Args:
            feature_name: Full feature name (e.g., "points|season|avg|diff|side")

        Returns:
            Dict with keys: stat_name, time_period, calc_weight, perspective, has_side
            Or None if parsing fails
        """
        parts = feature_name.split("|")

        if len(parts) < 4:
            return None

        result = {
            "stat_name": parts[0],
            "time_period": parts[1],
            "calc_weight": parts[2],
            "perspective": parts[3],
            "has_side": len(parts) >= 5 and parts[4] == "side"
        }

        return result

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    @classmethod
    def is_blend_format(cls, calc_weight: str) -> bool:
        """Check if calc_weight is a blend format (e.g., 'blend:season:0.8/games_10:0.2')."""
        return calc_weight.startswith("blend:")

    @classmethod
    def validate_blend_format(cls, calc_weight: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a blend format calc_weight.

        Args:
            calc_weight: e.g., "blend:season:0.80/games_20:0.10/games_12:0.10"

        Returns:
            (is_valid, error_message)
        """
        if not calc_weight.startswith("blend:"):
            return False, "Blend format must start with 'blend:'"

        blend_spec = calc_weight[6:]  # Remove "blend:" prefix

        # Handle special injury blend format
        if "/" in blend_spec and ":" in blend_spec:
            components = blend_spec.split("/")
            total_weight = 0.0

            for component in components:
                if ":" not in component:
                    return False, f"Invalid blend component: {component}"

                parts = component.split(":")
                if len(parts) != 2:
                    return False, f"Invalid blend component format: {component}"

                period_or_key, weight_str = parts
                try:
                    weight = float(weight_str)
                    total_weight += weight
                except ValueError:
                    return False, f"Invalid weight value: {weight_str}"

            # Weights should sum to approximately 1.0
            if abs(total_weight - 1.0) > 0.01:
                return False, f"Blend weights sum to {total_weight}, expected 1.0"

        return True, None

    @classmethod
    def validate_stat_name(cls, stat_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a stat name.

        Args:
            stat_name: The stat name to validate

        Returns:
            (is_valid, error_message)
        """
        if stat_name in cls.STAT_DEFINITIONS:
            return True, None

        # Check if it's a valid _net variant
        if stat_name.endswith("_net"):
            base_stat = stat_name[:-4]
            if base_stat in cls.STAT_DEFINITIONS:
                defn = cls.STAT_DEFINITIONS[base_stat]
                if defn.supports_net:
                    return True, None
                return False, f"Stat '{base_stat}' does not support _net variant"

        return False, f"Unknown stat name: '{stat_name}'"

    @classmethod
    def validate_time_period(cls, time_period: str) -> Tuple[bool, Optional[str]]:
        """Validate a time period (including composite blend/delta formats)."""
        if cls.is_valid_time_period(time_period):
            return True, None

        # Provide specific error messages for malformed composite formats
        if time_period.startswith('blend:'):
            parsed = cls.parse_blend_time_period(time_period)
            if parsed is None:
                return False, f"Malformed blend time period: '{time_period}'. Expected format: 'blend:tp1:weight1/tp2:weight2'"
            total_weight = 0.0
            for comp_tp, weight in parsed['components']:
                if not cls.is_valid_time_period(comp_tp):
                    return False, f"Invalid component time period '{comp_tp}' in blend: '{time_period}'"
                total_weight += weight
            if abs(total_weight - 1.0) > 0.01:
                return False, f"Blend weights sum to {total_weight:.2f}, expected 1.0 in: '{time_period}'"
            return True, None

        if time_period.startswith('delta:'):
            parsed = cls.parse_delta_time_period(time_period)
            if parsed is None:
                return False, f"Malformed delta time period: '{time_period}'. Expected format: 'delta:recent_tp-baseline_tp'"
            for leaf_tp in cls._extract_leaf_time_periods(time_period):
                if not cls.is_valid_time_period(leaf_tp):
                    return False, f"Invalid leaf time period '{leaf_tp}' in delta: '{time_period}'"
            return True, None

        return False, f"Invalid time period: '{time_period}'. Valid formats: 'season', 'none', 'games_N', 'days_N', 'months_N', 'blend:...', 'delta:...' (N = positive integer)"

    @classmethod
    def validate_calc_weight(cls, calc_weight: str) -> Tuple[bool, Optional[str]]:
        """Validate a calc weight."""
        if calc_weight in cls.VALID_CALC_WEIGHTS:
            return True, None

        if cls.is_blend_format(calc_weight):
            return cls.validate_blend_format(calc_weight)

        if cls.is_parameterized_calc_weight(calc_weight):
            return True, None

        return False, f"Invalid calc weight: '{calc_weight}'. Valid: {sorted(cls.VALID_CALC_WEIGHTS)}, pattern(k=N), or blend:*"

    @classmethod
    def validate_perspective(cls, perspective: str) -> Tuple[bool, Optional[str]]:
        """Validate a perspective."""
        if perspective in cls.VALID_PERSPECTIVES:
            return True, None
        return False, f"Invalid perspective: '{perspective}'. Valid: {sorted(cls.VALID_PERSPECTIVES)}"

    @classmethod
    def validate_feature(cls, feature_name: str) -> Tuple[bool, Optional[str]]:
        """
        Fully validate a feature name.

        Args:
            feature_name: Full feature name (e.g., "points|season|avg|diff")

        Returns:
            (is_valid, error_message)
        """
        parsed = cls.parse_feature_name(feature_name)

        if not parsed:
            return False, f"Invalid feature format: '{feature_name}'. Expected: stat|period|weight|perspective[|side]"

        # Validate stat name
        valid, error = cls.validate_stat_name(parsed["stat_name"])
        if not valid:
            return False, error

        # Validate time period format
        valid, error = cls.validate_time_period(parsed["time_period"])
        if not valid:
            return False, error

        # Validate time period is allowed for this specific stat
        stat_name = parsed["stat_name"]
        base_stat = stat_name[:-4] if stat_name.endswith("_net") else stat_name
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_time_periods:  # If restrictions exist, enforce them
                # For composite time periods, validate each leaf tp
                leaf_tps = cls._extract_leaf_time_periods(parsed["time_period"])
                for tp in leaf_tps:
                    is_allowed = False
                    for allowed_tp in stat_def.valid_time_periods:
                        if tp == allowed_tp:
                            is_allowed = True
                            break
                        # Check if it matches a pattern like "games_10" matching "games_N"
                        if '_' in allowed_tp and '_' in tp:
                            prefix = allowed_tp.rsplit('_', 1)[0]
                            if tp.startswith(prefix + '_'):
                                is_allowed = True
                                break
                    if not is_allowed:
                        return False, f"Time period '{tp}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_time_periods)}"

        # Validate calc weight format
        valid, error = cls.validate_calc_weight(parsed["calc_weight"])
        if not valid:
            return False, error

        # Validate calc weight is allowed for this specific stat
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_calc_weights:  # If restrictions exist, enforce them
                calc_weight = parsed["calc_weight"]
                # Check for exact match first
                if calc_weight not in stat_def.valid_calc_weights:
                    # For blend stats, also check if blend format is allowed
                    if "blend" in stat_def.valid_calc_weights and cls.is_blend_format(calc_weight):
                        pass  # Blend format is allowed
                    else:
                        return False, f"Calc weight '{calc_weight}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_calc_weights)}"

        # Validate perspective format
        valid, error = cls.validate_perspective(parsed["perspective"])
        if not valid:
            return False, error

        # Validate perspective is allowed for this specific stat
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_perspectives:  # If restrictions exist, enforce them
                if parsed["perspective"] not in stat_def.valid_perspectives:
                    return False, f"Perspective '{parsed['perspective']}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_perspectives)}"

        # Validate side split is allowed for this stat
        if parsed["has_side"]:
            stat_name = parsed["stat_name"]
            base_stat = stat_name[:-4] if stat_name.endswith("_net") else stat_name
            if base_stat in cls.STAT_DEFINITIONS:
                if not cls.STAT_DEFINITIONS[base_stat].supports_side_split:
                    return False, f"Stat '{stat_name}' does not support side split"

        return True, None

    @classmethod
    def validate_feature_list(cls, features: List[str]) -> Dict[str, any]:
        """
        Validate a list of features.

        Args:
            features: List of feature names

        Returns:
            Dict with 'valid', 'errors', 'warnings', 'invalid_features'
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "invalid_features": [],
            "valid_count": 0
        }

        for feature in features:
            valid, error = cls.validate_feature(feature)
            if valid:
                result["valid_count"] += 1
            else:
                result["valid"] = False
                result["invalid_features"].append(feature)
                result["errors"].append(f"{feature}: {error}")

        return result

    # ==========================================================================
    # CONVENIENCE METHODS
    # ==========================================================================

    @classmethod
    def get_all_stat_names(cls) -> List[str]:
        """Get all valid stat names."""
        return list(cls.STAT_DEFINITIONS.keys())

    @classmethod
    def get_stat_definition(cls, stat_name: str) -> Optional[StatDefinition]:
        """Get the definition for a stat."""
        return cls.STAT_DEFINITIONS.get(stat_name)

    @classmethod
    def get_all_time_periods(cls) -> List[str]:
        """Get all valid time periods."""
        return sorted(cls.VALID_TIME_PERIODS)

    @classmethod
    def get_all_calc_weights(cls) -> List[str]:
        """Get all valid calc weights (excluding blend)."""
        return sorted(cls.VALID_CALC_WEIGHTS)

    @classmethod
    def get_all_perspectives(cls) -> List[str]:
        """Get all valid perspectives."""
        return sorted(cls.VALID_PERSPECTIVES)

    @classmethod
    def generate_feature_variants(
        cls,
        stat_name: str,
        time_periods: Optional[List[str]] = None,
        calc_weights: Optional[List[str]] = None,
        perspectives: Optional[List[str]] = None,
        include_side: bool = False
    ) -> List[str]:
        """
        Generate all feature variants for a stat.

        Args:
            stat_name: Base stat name
            time_periods: List of time periods (default: all)
            calc_weights: List of calc weights (default: raw, avg)
            perspectives: List of perspectives (default: all)
            include_side: Whether to include side-split variants

        Returns:
            List of feature names
        """
        if time_periods is None:
            time_periods = list(cls.VALID_TIME_PERIODS - {"none"})
        if calc_weights is None:
            calc_weights = ["raw", "avg"]
        if perspectives is None:
            perspectives = ["diff", "home", "away"]

        features = []
        defn = cls.STAT_DEFINITIONS.get(stat_name)

        for period in time_periods:
            for weight in calc_weights:
                for perspective in perspectives:
                    feature = cls.build_feature_name(stat_name, period, weight, perspective)
                    features.append(feature)

                    # Add side split variant if applicable
                    if include_side and defn and defn.supports_side_split:
                        features.append(cls.build_feature_name(
                            stat_name, period, weight, perspective, side_split=True
                        ))

        return features
