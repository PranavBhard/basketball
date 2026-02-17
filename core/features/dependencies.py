"""
Feature Dependency Management - Part of the Feature Registry SSoT Core Layer

This module provides infrastructure for tracking which features depend on other features,
and for resolving transitive dependencies when regenerating features.

Key dependency types:
1. Diff features depend on their home/away counterparts
2. Derived features depend on their component features (e.g., exp_points depends on est_possessions)
3. Blend features depend on their time-period components
4. Net stats depend on the base stat and opponent stats
"""

from typing import Dict, List, Set, Tuple, Optional
import re
from bball_app.core.features.registry import FeatureRegistry


# =============================================================================
# EXPLICIT DEPENDENCY MAPPINGS
# =============================================================================

# Direct dependency mappings for features that can't be auto-derived
# Format: {dependent_feature: [list of features it depends on]}
EXPLICIT_DEPENDENCIES: Dict[str, List[str]] = {
    # Injury impact blend features
    'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|home': [
        'inj_severity|none|raw|home',
        'inj_per|none|top1_avg|home',
        'inj_rotation_per|none|raw|home'
    ],
    'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|away': [
        'inj_severity|none|raw|away',
        'inj_per|none|top1_avg|away',
        'inj_rotation_per|none|raw|away'
    ],
    'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|diff': [
        'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|home',
        'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|away',
    ],
}


# =============================================================================
# DEPENDENCY RULE ENGINE
# =============================================================================

def get_diff_dependencies(feature_name: str) -> List[str]:
    """
    Get dependencies for a diff feature (diff = home - away).

    Args:
        feature_name: Feature name ending in |diff

    Returns:
        List of home and away feature names this diff depends on
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if not parsed or parsed['perspective'] != 'diff':
        return []

    # Build home and away versions
    base = f"{parsed['stat_name']}|{parsed['time_period']}|{parsed['calc_weight']}"
    home_feature = f"{base}|home"
    away_feature = f"{base}|away"

    # Add side suffix if present
    if parsed.get('has_side'):
        home_feature += '|side'
        away_feature += '|side'

    return [home_feature, away_feature]


def get_derived_stat_dependencies(feature_name: str) -> List[str]:
    """
    Get dependencies for derived stats (e.g., exp_points depends on est_possessions).

    Args:
        feature_name: Feature name

    Returns:
        List of features this derived stat depends on
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if not parsed:
        return []

    stat_name = parsed['stat_name']
    time_period = parsed['time_period']
    perspective = parsed['perspective']

    dependencies = []

    # exp_points_off depends on est_possessions and off_rtg
    if stat_name == 'exp_points_off':
        dependencies.append(f"est_possessions|{time_period}|derived|none")
        dependencies.append(f"off_rtg|{time_period}|avg|{perspective}")

    # exp_points_def depends on est_possessions and def_rtg
    elif stat_name == 'exp_points_def':
        dependencies.append(f"est_possessions|{time_period}|derived|none")
        dependencies.append(f"def_rtg|{time_period}|avg|{perspective}")

    # exp_points_matchup depends on exp_points_off and exp_points_def
    elif stat_name == 'exp_points_matchup':
        dependencies.append(f"exp_points_off|{time_period}|derived|{perspective}")
        dependencies.append(f"exp_points_def|{time_period}|derived|{perspective}")

    # est_possessions depends on pace_interaction
    elif stat_name == 'est_possessions':
        dependencies.append(f"pace_interaction|{time_period}|harmonic_mean|none")

    # pace_interaction depends on pace from both teams
    elif stat_name == 'pace_interaction':
        dependencies.append(f"pace|{time_period}|avg|home")
        dependencies.append(f"pace|{time_period}|avg|away")

    return dependencies


def get_blend_dependencies(feature_name: str) -> List[str]:
    """
    Get dependencies for blend features (new composite time_period format).

    New format: stat|blend:tp1:w1/tp2:w2|calc_weight|perspective
    Dependencies: [stat|tp1|calc_weight|perspective, stat|tp2|calc_weight|perspective]

    Args:
        feature_name: Feature name with blend time_period

    Returns:
        List of component feature names this blend depends on
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if not parsed:
        return []

    time_period = parsed['time_period']
    if not time_period.startswith('blend:'):
        return []

    blend_parsed = FeatureRegistry.parse_blend_time_period(time_period)
    if blend_parsed is None:
        return []

    stat_name = parsed['stat_name']
    calc_weight = parsed['calc_weight']
    perspective = parsed['perspective']
    side_suffix = '|side' if parsed.get('has_side') else ''

    dependencies = []
    for tp, _ in blend_parsed['components']:
        dep = f"{stat_name}|{tp}|{calc_weight}|{perspective}{side_suffix}"
        dependencies.append(dep)

    return dependencies


def get_delta_dependencies(feature_name: str) -> List[str]:
    """
    Get dependencies for delta features (new composite time_period format).

    Delta format: stat|delta:recent_tp-baseline_tp|calc_weight|perspective
    Dependencies: [stat|recent_tp|..., stat|baseline_tp|...]

    Blend-delta format: stat|delta:blend:tp1:w1/tp2:w2-baseline_tp|calc_weight|perspective
    Dependencies: [stat|blend:tp1:w1/tp2:w2|..., stat|baseline_tp|...]

    Args:
        feature_name: Feature name with delta time_period

    Returns:
        List of component feature names this delta depends on
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if not parsed:
        return []

    time_period = parsed['time_period']
    if not time_period.startswith('delta:'):
        return []

    delta_parsed = FeatureRegistry.parse_delta_time_period(time_period)
    if delta_parsed is None:
        return []

    stat_name = parsed['stat_name']
    calc_weight = parsed['calc_weight']
    perspective = parsed['perspective']
    side_suffix = '|side' if parsed.get('has_side') else ''

    dependencies = []

    # Recent dependency
    if delta_parsed['type'] == 'blend_delta':
        # Reconstruct the blend time_period string
        recent_tp = 'blend:' + '/'.join(
            f"{tp}:{w}" for tp, w in delta_parsed['recent']['components']
        )
    else:
        recent_tp = delta_parsed['recent']

    dependencies.append(f"{stat_name}|{recent_tp}|{calc_weight}|{perspective}{side_suffix}")

    # Baseline dependency
    baseline_tp = delta_parsed['baseline']
    dependencies.append(f"{stat_name}|{baseline_tp}|{calc_weight}|{perspective}{side_suffix}")

    return dependencies


def get_net_stat_dependencies(feature_name: str) -> List[str]:
    """
    Get dependencies for net stats (stat_net = stat - opp_stat).

    Args:
        feature_name: Net stat feature name

    Returns:
        List of base and opponent stat features this net stat depends on
    """
    parsed = FeatureRegistry.parse_feature_name(feature_name)
    if not parsed:
        return []

    stat_name = parsed['stat_name']
    if not stat_name.endswith('_net'):
        return []

    base_stat = stat_name[:-4]  # Remove '_net'
    time_period = parsed['time_period']
    calc_weight = parsed['calc_weight']
    perspective = parsed['perspective']

    # Net stat depends on the base stat
    base_feature = f"{base_stat}|{time_period}|{calc_weight}|{perspective}"

    # Also depends on opponent version (for opponent-adjusted calculation)
    # Note: This is a simplification - actual dependency might be more complex
    return [base_feature]


# =============================================================================
# MAIN DEPENDENCY RESOLUTION API
# =============================================================================

def get_direct_dependencies(feature_name: str) -> List[str]:
    """
    Get all direct dependencies for a feature.

    Checks in order:
    1. Explicit mappings
    2. Diff dependencies (diff depends on home + away)
    3. Derived stat dependencies
    4. Blend feature dependencies
    5. Net stat dependencies

    Args:
        feature_name: Feature name

    Returns:
        List of feature names this feature directly depends on
    """
    # 1. Check explicit mappings first
    if feature_name in EXPLICIT_DEPENDENCIES:
        return EXPLICIT_DEPENDENCIES[feature_name].copy()

    dependencies = []

    # 2. Check diff dependencies
    diff_deps = get_diff_dependencies(feature_name)
    if diff_deps:
        dependencies.extend(diff_deps)

    # 3. Check derived stat dependencies
    derived_deps = get_derived_stat_dependencies(feature_name)
    if derived_deps:
        dependencies.extend(derived_deps)

    # 4. Check blend dependencies
    blend_deps = get_blend_dependencies(feature_name)
    if blend_deps:
        dependencies.extend(blend_deps)

    # 5. Check delta dependencies
    delta_deps = get_delta_dependencies(feature_name)
    if delta_deps:
        dependencies.extend(delta_deps)

    # 6. Check net stat dependencies (if not already handled)
    if not dependencies:
        net_deps = get_net_stat_dependencies(feature_name)
        dependencies.extend(net_deps)

    return dependencies


def resolve_dependencies(
    feature_names: List[str],
    include_transitive: bool = True
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    """
    Resolve all dependencies for a list of features (transitive closure).

    Args:
        feature_names: List of feature names to resolve
        include_transitive: If True, include transitive dependencies (A -> B -> C includes C)

    Returns:
        Tuple of:
        - Set of all features to regenerate (original + all dependencies)
        - Dict mapping each original feature to its direct dependencies
    """
    all_features = set(feature_names)
    dependency_map = {f: set() for f in feature_names}

    if not include_transitive:
        # Just get direct dependencies
        for feature in feature_names:
            deps = get_direct_dependencies(feature)
            dependency_map[feature] = set(deps)
            all_features.update(deps)
        return all_features, dependency_map

    # Transitive closure: keep resolving until no new dependencies found
    to_process = set(feature_names)
    processed = set()

    while to_process:
        current = to_process.pop()
        if current in processed:
            continue

        processed.add(current)
        deps = get_direct_dependencies(current)

        # Track direct dependencies for original features
        if current in feature_names:
            dependency_map[current].update(deps)

        # Add new dependencies to process
        for dep in deps:
            if dep not in processed and dep not in to_process:
                to_process.add(dep)
                all_features.add(dep)

    return all_features, dependency_map


def get_dependents(feature_name: str, candidate_features: List[str]) -> List[str]:
    """
    Get all features that depend on the given feature (reverse lookup).

    This is useful for knowing which features need to be regenerated when
    a base feature changes.

    Args:
        feature_name: Feature name to check
        candidate_features: List of features to check for dependencies

    Returns:
        List of features from candidate_features that depend on feature_name
    """
    dependents = []
    for feature in candidate_features:
        deps = get_direct_dependencies(feature)
        if feature_name in deps:
            dependents.append(feature)
    return dependents


def categorize_features(
    requested_features: List[str],
    all_features: Set[str]
) -> Dict[str, List[str]]:
    """
    Categorize features into requested vs. dependencies.

    Args:
        requested_features: Features explicitly requested for regeneration
        all_features: All features that will be regenerated (including dependencies)

    Returns:
        Dict with keys:
        - 'requested': List of requested features
        - 'dependencies': List of dependency features (not in requested)
        - 'all': List of all features (requested + dependencies)
    """
    requested_set = set(requested_features)
    all_set = set(all_features)

    dependencies = sorted(list(all_set - requested_set))

    return {
        'requested': sorted(list(requested_set)),
        'dependencies': dependencies,
        'all': sorted(list(all_set))
    }


def get_regeneration_order(features: Set[str]) -> List[str]:
    """
    Get the correct order to regenerate features (dependencies first).

    Uses topological sort to ensure dependencies are regenerated before
    the features that depend on them.

    Args:
        features: Set of features to regenerate

    Returns:
        List of features in correct regeneration order
    """
    # Build dependency graph
    in_degree = {f: 0 for f in features}
    dependents = {f: [] for f in features}

    for feature in features:
        deps = get_direct_dependencies(feature)
        for dep in deps:
            if dep in features:
                in_degree[feature] += 1
                dependents[dep].append(feature)

    # Topological sort using Kahn's algorithm
    queue = [f for f in features if in_degree[f] == 0]
    result = []

    while queue:
        current = queue.pop(0)
        result.append(current)

        for dependent in dependents[current]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    # If we couldn't process all features, there's a cycle (shouldn't happen)
    if len(result) != len(features):
        # Fall back to sorted order
        remaining = features - set(result)
        result.extend(sorted(remaining))

    return result
