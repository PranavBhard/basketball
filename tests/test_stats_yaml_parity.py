"""
Test that stats.yaml exactly matches the Python STAT_DEFINITIONS.

This is a parity test: it proves the YAML file is a faithful representation
of the existing Python definitions, so we can safely switch to loading from YAML.
"""

import os
import pytest
from sportscore.features import load_stat_definitions
from bball.features.registry import FeatureRegistry


YAML_PATH = os.path.join(os.path.dirname(__file__), '..', 'bball', 'features', 'stats.yaml')


@pytest.fixture
def yaml_defs():
    return load_stat_definitions(YAML_PATH)


@pytest.fixture
def python_defs():
    return FeatureRegistry.STAT_DEFINITIONS


def test_yaml_has_all_python_stats(yaml_defs, python_defs):
    """Every stat in Python definitions must exist in YAML."""
    missing = set(python_defs.keys()) - set(yaml_defs.keys())
    assert not missing, f"Stats missing from YAML: {sorted(missing)}"


def test_yaml_has_no_extra_stats(yaml_defs, python_defs):
    """YAML should not define stats that don't exist in Python."""
    extra = set(yaml_defs.keys()) - set(python_defs.keys())
    assert not extra, f"Extra stats in YAML not in Python: {sorted(extra)}"


def test_stat_count_matches(yaml_defs, python_defs):
    """YAML and Python must define the same number of stats."""
    assert len(yaml_defs) == len(python_defs), (
        f"Count mismatch: YAML={len(yaml_defs)}, Python={len(python_defs)}"
    )


def test_each_stat_fields_match(yaml_defs, python_defs):
    """Every field of every stat must match between YAML and Python."""
    errors = []

    for name, py_def in python_defs.items():
        if name not in yaml_defs:
            errors.append(f"{name}: missing from YAML")
            continue

        y = yaml_defs[name]

        if y.category != py_def.category:
            errors.append(f"{name}.category: YAML={y.category} vs Python={py_def.category}")

        if y.db_field != py_def.db_field:
            errors.append(f"{name}.db_field: YAML={y.db_field} vs Python={py_def.db_field}")

        if y.supports_side_split != py_def.supports_side_split:
            errors.append(f"{name}.supports_side_split: YAML={y.supports_side_split} vs Python={py_def.supports_side_split}")

        if y.supports_net != py_def.supports_net:
            errors.append(f"{name}.supports_net: YAML={y.supports_net} vs Python={py_def.supports_net}")

        if y.requires_aggregation != py_def.requires_aggregation:
            errors.append(f"{name}.requires_aggregation: YAML={y.requires_aggregation} vs Python={py_def.requires_aggregation}")

        if y.valid_calc_weights != py_def.valid_calc_weights:
            errors.append(
                f"{name}.valid_calc_weights: YAML={sorted(y.valid_calc_weights)} vs Python={sorted(py_def.valid_calc_weights)}"
            )

        if y.valid_time_periods != py_def.valid_time_periods:
            errors.append(
                f"{name}.valid_time_periods: YAML={sorted(y.valid_time_periods)} vs Python={sorted(py_def.valid_time_periods)}"
            )

        if y.valid_perspectives != py_def.valid_perspectives:
            errors.append(
                f"{name}.valid_perspectives: YAML={sorted(y.valid_perspectives)} vs Python={sorted(py_def.valid_perspectives)}"
            )

    assert not errors, "Field mismatches:\n" + "\n".join(errors)


def test_description_present(yaml_defs):
    """Every YAML stat should have a non-empty description."""
    missing_desc = [name for name, d in yaml_defs.items() if not d.description]
    assert not missing_desc, f"Stats without description: {sorted(missing_desc)}"
