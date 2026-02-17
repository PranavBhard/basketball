#!/usr/bin/env python3
"""
Integration tests for the Training Workflow.

Tests that the training pipeline works correctly:
1. BballModel initialization and feature calculation
2. Training data generation
3. Model training and evaluation
4. Config management

Usage:
    source venv/bin/activate
    PYTHONPATH=/Users/pranav/Documents/NBA python tests/test_training_workflow.py
"""

import sys
import os
import tempfile

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_nba_model_initialization():
    """Test that BballModel initializes correctly with repositories."""
    print("\n" + "=" * 60)
    print("TEST: BballModel Initialization")
    print("=" * 60)

    from bball_app.core.models.bball_model import BballModel

    results = []

    # Test 1: Initialize with minimal features
    print("\n  1. Testing minimal initialization...")
    try:
        model = BballModel(
            classifier_features=['wins|season|avg|diff'],
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=False
        )
        is_valid = model is not None and hasattr(model, 'db')
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: BballModel initialized with minimal features")
        results.append(('nba_model_init_minimal', is_valid))
    except Exception as e:
        print(f"     FAIL: BballModel initialization failed: {e}")
        results.append(('nba_model_init_minimal', False))

    # Test 2: Verify repositories are initialized
    print("\n  2. Testing repository initialization...")
    try:
        has_repos = (
            hasattr(model, '_games_repo') and
            hasattr(model, '_players_repo') and
            hasattr(model, '_rosters_repo')
        )
        is_valid = has_repos
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Repositories initialized on BballModel")
        results.append(('nba_model_repos', is_valid))
    except Exception as e:
        print(f"     FAIL: Repository check failed: {e}")
        results.append(('nba_model_repos', False))

    # Test 3: Initialize with Elo features
    print("\n  3. Testing initialization with Elo features...")
    try:
        model_with_elo = BballModel(
            classifier_features=['elo|none|raw|diff'],  # 'none' is valid for elo
            include_elo=True,
            include_per_features=False,
            include_injuries=False,
            preload_data=False
        )
        is_valid = model_with_elo is not None
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: BballModel initialized with Elo features")
        results.append(('nba_model_init_elo', is_valid))
    except Exception as e:
        print(f"     FAIL: BballModel with Elo failed: {e}")
        results.append(('nba_model_init_elo', False))

    return results


def test_feature_calculation():
    """Test that feature calculation works via StatHandler."""
    print("\n" + "=" * 60)
    print("TEST: Feature Calculation")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.stats.handler import StatHandlerV2

    results = []
    db = Mongo().db

    # Test 1: StatHandler initialization
    print("\n  1. Testing StatHandler initialization...")
    try:
        sh = StatHandlerV2(db=db, statistics=[])
        is_valid = sh is not None
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: StatHandler initialized")
        results.append(('stat_handler_init', is_valid))
    except Exception as e:
        print(f"     FAIL: StatHandler initialization failed: {e}")
        results.append(('stat_handler_init', False))
        return results

    # Test 2: Calculate basic feature
    print("\n  2. Testing basic feature calculation...")
    try:
        feature_value = sh.calculate_feature(
            'wins|season|avg|diff',
            'LAL', 'BOS', '2023-2024', 2024, 1, 15
        )
        is_valid = isinstance(feature_value, (int, float)) and not (feature_value != feature_value)  # not NaN
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: wins|season|avg|diff = {feature_value:.4f}")
        results.append(('feature_calc_basic', is_valid))
    except Exception as e:
        print(f"     FAIL: Feature calculation failed: {e}")
        results.append(('feature_calc_basic', False))

    # Test 3: Calculate net feature
    print("\n  3. Testing net feature calculation...")
    try:
        feature_value = sh.calculate_feature(
            'points_net|season|avg|diff',
            'LAL', 'BOS', '2023-2024', 2024, 1, 15
        )
        is_valid = isinstance(feature_value, (int, float))
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: points_net|season|avg|diff = {feature_value:.4f}")
        results.append(('feature_calc_net', is_valid))
    except Exception as e:
        print(f"     FAIL: Net feature calculation failed: {e}")
        results.append(('feature_calc_net', False))

    # Test 4: Calculate blend feature (composite time_period format)
    print("\n  4. Testing blend feature calculation...")
    try:
        feature_value = sh.calculate_feature(
            'wins|blend:season:0.80/games_12:0.20|avg|diff',
            'LAL', 'BOS', '2023-2024', 2024, 1, 15
        )
        is_valid = isinstance(feature_value, (int, float))
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: wins|blend:season:0.80/games_12:0.20|avg|diff = {feature_value:.4f}")
        results.append(('feature_calc_blend', is_valid))
    except Exception as e:
        print(f"     FAIL: Blend feature calculation failed: {e}")
        results.append(('feature_calc_blend', False))

    return results


def test_build_features_dict():
    """Test BballModel._build_features_dict() method."""
    print("\n" + "=" * 60)
    print("TEST: Build Features Dict")
    print("=" * 60)

    from bball_app.core.models.bball_model import BballModel

    results = []

    # Test with a small set of features
    test_features = [
        'wins|season|avg|diff',
        'points|games_10|avg|home',
        'points|games_10|avg|away',
    ]

    print("\n  1. Testing _build_features_dict()...")
    try:
        model = BballModel(
            classifier_features=test_features,
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=False
        )
        model.feature_names = test_features

        features_dict = model._build_features_dict(
            'LAL', 'BOS', '2023-2024', 2024, 1, 15
        )

        # Verify all features are present
        missing = [f for f in test_features if f not in features_dict]
        is_valid = len(missing) == 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Built features dict with {len(features_dict)} features")
        if missing:
            print(f"     Missing: {missing}")
        results.append(('build_features_dict', is_valid))

        # Verify values are numeric
        print("\n  2. Verifying feature values are numeric...")
        non_numeric = [k for k, v in features_dict.items()
                       if not isinstance(v, (int, float))]
        is_valid = len(non_numeric) == 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: All {len(features_dict)} feature values are numeric")
        results.append(('features_numeric', is_valid))

    except Exception as e:
        print(f"     FAIL: _build_features_dict failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('build_features_dict', False))
        results.append(('features_numeric', False))

    return results


def test_create_training_data():
    """Test BballModel.create_training_data() method."""
    print("\n" + "=" * 60)
    print("TEST: Create Training Data")
    print("=" * 60)

    from bball_app.core.models.bball_model import BballModel
    import pandas as pd

    results = []

    # Test with minimal features and limited date range
    test_features = [
        'wins|season|avg|diff',
        'points|games_10|avg|diff',
    ]

    print("\n  1. Testing create_training_data() with limited data...")
    try:
        model = BballModel(
            classifier_features=test_features,
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=True  # Need preloaded data for training
        )

        # Create training data for just a few games using query format
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name

        # Use query dict format (correct signature)
        query = {
            'season': '2023-2024',
            'year': 2024,
            'month': {'$in': [1]},  # Just January
        }

        result = model.create_training_data(
            query=query,
            classifier_csv=csv_path,
            min_games_filter=5
        )
        # Handle return: (rows_written, classifier_csv_path, points_csv_path)
        if isinstance(result, tuple):
            classifier_csv = result[1]  # CSV path is at index 1, not 0
        else:
            classifier_csv = result

        # Verify CSV was created and has data
        if classifier_csv and os.path.exists(classifier_csv):
            df = pd.read_csv(classifier_csv)
            is_valid = len(df) > 0 and 'HomeWon' in df.columns
            status = "PASS" if is_valid else "FAIL"
            print(f"     {status}: Created training data with {len(df)} rows")
            print(f"     Columns: {list(df.columns)[:5]}...")
            # Cleanup
            os.unlink(classifier_csv)
        else:
            is_valid = False
            print(f"     FAIL: No CSV created")

        results.append(('create_training_data', is_valid))

    except Exception as e:
        print(f"     FAIL: create_training_data failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('create_training_data', False))

    return results


def test_model_factory():
    """Test ArtifactLoader for model creation."""
    print("\n" + "=" * 60)
    print("TEST: ArtifactLoader")
    print("=" * 60)

    from bball_app.core.models.artifact_loader import ArtifactLoader

    results = []

    # Test 1: Create LogisticRegression model
    print("\n  1. Testing create_sklearn_model() for LogisticRegression...")
    try:
        model = ArtifactLoader.create_sklearn_model('LogisticRegression', c_value=0.1)
        is_valid = model is not None and hasattr(model, 'fit')
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Created LogisticRegression model")
        results.append(('model_factory_lr', is_valid))
    except Exception as e:
        print(f"     FAIL: ArtifactLoader failed: {e}")
        results.append(('model_factory_lr', False))

    # Test 2: Create GradientBoosting model
    print("\n  2. Testing create_sklearn_model() for GradientBoosting...")
    try:
        model = ArtifactLoader.create_sklearn_model('GradientBoosting')
        is_valid = model is not None and hasattr(model, 'fit')
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Created GradientBoosting model")
        results.append(('model_factory_gb', is_valid))
    except Exception as e:
        print(f"     FAIL: ArtifactLoader failed: {e}")
        results.append(('model_factory_gb', False))

    return results


def test_config_manager():
    """Test ModelConfigManager operations."""
    print("\n" + "=" * 60)
    print("TEST: ModelConfigManager")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.services.config_manager import ModelConfigManager

    results = []
    db = Mongo().db
    mgr = ModelConfigManager(db)

    # Test 1: Get selected classifier config
    print("\n  1. Testing get_selected_config()...")
    try:
        config = mgr.get_selected_config()  # Correct method name
        is_valid = config is None or isinstance(config, dict)
        status = "PASS" if is_valid else "FAIL"
        if config:
            print(f"     {status}: Found selected config: {config.get('model_type', 'unknown')}")
        else:
            print(f"     {status}: No selected config (valid state)")
        results.append(('config_mgr_get_selected', is_valid))
    except Exception as e:
        print(f"     FAIL: get_selected_config failed: {e}")
        results.append(('config_mgr_get_selected', False))

    # Test 2: List all configs
    print("\n  2. Testing list_configs()...")
    try:
        configs = mgr.list_configs()  # Correct method name
        is_valid = isinstance(configs, list)
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Found {len(configs)} classifier configs")
        results.append(('config_mgr_list', is_valid))
    except Exception as e:
        print(f"     FAIL: list_configs failed: {e}")
        results.append(('config_mgr_list', False))

    return results


def main():
    """Run all training workflow tests."""
    print("=" * 60)
    print("TRAINING WORKFLOW INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\nThis suite tests the training pipeline after the")
    print("layered architecture migration.")

    all_results = []

    # Run all tests
    all_results.extend(test_nba_model_initialization())
    all_results.extend(test_feature_calculation())
    all_results.extend(test_build_features_dict())
    all_results.extend(test_create_training_data())
    all_results.extend(test_model_factory())
    all_results.extend(test_config_manager())

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in all_results if result)
    failed = sum(1 for _, result in all_results if not result)

    print(f"\nTotal tests: {len(all_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed tests:")
        for name, result in all_results:
            if not result:
                print(f"  X {name}")

    print("\n" + "=" * 60)
    if failed == 0:
        print("ALL TRAINING WORKFLOW TESTS PASSED")
    else:
        print(f"TESTS FAILED: {failed} failures")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
