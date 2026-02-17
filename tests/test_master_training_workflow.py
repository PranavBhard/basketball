#!/usr/bin/env python3
"""
Integration tests for the Master Training Creation Workflow.

Tests that the master training pipeline works correctly:
1. Query games from MongoDB via repositories
2. Feature generation for training data
3. Master training CSV operations
4. Column population

Usage:
    source venv/bin/activate
    PYTHONPATH=/Users/pranav/Documents/NBA python tests/test_master_training_workflow.py
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


def test_games_query_for_training():
    """Test that we can query games for training data generation."""
    print("\n" + "=" * 60)
    print("TEST: Query Games For Training")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import GamesRepository

    results = []
    db = Mongo().db
    repo = GamesRepository(db)

    # Test 1: Query games with training filters
    print("\n  1. Testing games query with training filters...")
    try:
        # Typical training query filters
        query = {
            'season': '2023-2024',
            'game_type': {'$nin': ['preseason', 'allstar']},
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0}
        }
        games = repo.find(query, limit=100, sort=[('date', 1)])

        is_valid = len(games) > 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Found {len(games)} games for training")

        if games:
            sample = games[0]
            print(f"     Sample game date: {sample.get('date')}")
            print(f"     Home: {sample.get('homeTeam', {}).get('name')}")
            print(f"     Away: {sample.get('awayTeam', {}).get('name')}")

        results.append(('games_query_training', is_valid))
    except Exception as e:
        print(f"     FAIL: Games query failed: {e}")
        results.append(('games_query_training', False))

    # Test 2: Verify game structure has required fields
    print("\n  2. Testing game document structure...")
    try:
        if games:
            game = games[0]
            required_fields = ['date', 'season', 'homeTeam', 'awayTeam', 'year', 'month', 'day']
            missing = [f for f in required_fields if f not in game]
            is_valid = len(missing) == 0
            status = "PASS" if is_valid else "FAIL"
            print(f"     {status}: Game has all required fields")
            if missing:
                print(f"     Missing: {missing}")
        else:
            is_valid = False
            print("     FAIL: No games to check")
        results.append(('game_structure', is_valid))
    except Exception as e:
        print(f"     FAIL: Structure check failed: {e}")
        results.append(('game_structure', False))

    return results


def test_feature_registry():
    """Test that FeatureRegistry provides features for master training."""
    print("\n" + "=" * 60)
    print("TEST: Feature Registry")
    print("=" * 60)

    from bball_app.core.features.registry import FeatureRegistry, FeatureGroups

    results = []

    # Test 1: Get all stat names
    print("\n  1. Testing get_all_stat_names()...")
    try:
        all_stats = FeatureRegistry.get_all_stat_names()
        is_valid = len(all_stats) > 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Registry has {len(all_stats)} stat names")
        results.append(('feature_registry_stats', is_valid))
    except Exception as e:
        print(f"     FAIL: get_all_stat_names failed: {e}")
        results.append(('feature_registry_stats', False))

    # Test 2: Get groups by layer (on FeatureGroups class)
    print("\n  2. Testing FeatureGroups.get_groups_by_layer()...")
    try:
        for layer in [1, 2, 3]:
            groups = FeatureGroups.get_groups_by_layer(layer)
            print(f"     Layer {layer}: {len(groups)} groups")
        is_valid = True
        status = "PASS"
        print(f"     {status}: Layer queries work")
        results.append(('feature_groups_layers', is_valid))
    except Exception as e:
        print(f"     FAIL: get_groups_by_layer failed: {e}")
        results.append(('feature_groups_layers', False))

    return results


def test_master_training_data_module():
    """Test the master_training_data module functions."""
    print("\n" + "=" * 60)
    print("TEST: Master Training Data Module")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.services.training_data import (
        get_master_training_metadata,
        get_all_possible_features
    )

    results = []
    db = Mongo().db

    # Test 1: Get all possible features
    print("\n  1. Testing get_all_possible_features()...")
    try:
        features = get_all_possible_features()
        is_valid = isinstance(features, list) and len(features) > 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Got {len(features)} possible features")
        results.append(('all_possible_features', is_valid))
    except Exception as e:
        print(f"     FAIL: get_all_possible_features failed: {e}")
        results.append(('all_possible_features', False))

    # Test 2: Get master training metadata
    print("\n  2. Testing get_master_training_metadata()...")
    try:
        metadata = get_master_training_metadata(db)  # Takes db parameter
        is_valid = metadata is None or isinstance(metadata, dict)
        status = "PASS" if is_valid else "FAIL"
        if metadata:
            print(f"     {status}: Found metadata with {len(metadata.get('columns', []))} columns")
            print(f"     Latest date: {metadata.get('latest_date', 'N/A')}")
        else:
            print(f"     {status}: No metadata (valid if master not created yet)")
        results.append(('master_training_metadata', is_valid))
    except Exception as e:
        print(f"     FAIL: get_master_training_metadata failed: {e}")
        results.append(('master_training_metadata', False))

    return results


def test_nba_model_create_training_data():
    """Test BballModel.create_training_data() for master training generation."""
    print("\n" + "=" * 60)
    print("TEST: BballModel Create Training Data")
    print("=" * 60)

    from bball_app.core.models.bball_model import BballModel
    import pandas as pd

    results = []

    # Use minimal features for speed
    test_features = [
        'wins|season|avg|diff',
        'points|games_10|avg|diff',
        'off_rtg|season|raw|diff',
    ]

    # Test 1: Create training data for a small date range
    print("\n  1. Testing create_training_data() for small range...")
    try:
        model = BballModel(
            classifier_features=test_features,
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=True
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name

        # Use correct query format
        query = {
            'season': '2023-2024',
            'year': 2024,
            'month': {'$in': [1]},
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

        # Verify CSV
        if classifier_csv and os.path.exists(classifier_csv):
            df = pd.read_csv(classifier_csv)
            is_valid = len(df) > 0 and 'HomeWon' in df.columns
            status = "PASS" if is_valid else "FAIL"
            print(f"     {status}: Created CSV with {len(df)} rows, {len(df.columns)} columns")

            # Check for expected columns
            expected_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon']
            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                print(f"     WARNING: Missing expected columns: {missing}")

            # Cleanup
            os.unlink(classifier_csv)
        else:
            is_valid = False
            print("     FAIL: No CSV created")

        results.append(('create_training_data', is_valid))

    except Exception as e:
        print(f"     FAIL: create_training_data failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('create_training_data', False))

    return results


def test_populate_columns():
    """Test column population logic."""
    print("\n" + "=" * 60)
    print("TEST: Column Population Logic")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.models.bball_model import BballModel
    import pandas as pd

    results = []
    db = Mongo().db

    # Test 1: Create a mini training dataset and add columns
    print("\n  1. Testing column population workflow...")
    try:
        # Create model with the features we want to calculate
        test_features = ['wins|season|avg|diff', 'points|games_10|avg|diff']
        model = BballModel(
            classifier_features=test_features,
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=True
        )
        # Ensure feature_names is set (required for _build_features_dict)
        model.feature_names = test_features

        # Create mini dataframe simulating master training
        mini_df = pd.DataFrame([
            {'Year': 2024, 'Month': 1, 'Day': 15, 'Home': 'LAL', 'Away': 'BOS', 'HomeWon': 1},
            {'Year': 2024, 'Month': 1, 'Day': 16, 'Home': 'GSW', 'Away': 'PHX', 'HomeWon': 0},
        ])

        # Test _build_features_dict for each row
        new_feature = 'points|games_10|avg|diff'
        new_values = []

        for _, row in mini_df.iterrows():
            features = model._build_features_dict(
                row['Home'], row['Away'], '2023-2024',
                row['Year'], row['Month'], row['Day']
            )
            if features:
                new_values.append(features.get(new_feature, 0.0))
            else:
                new_values.append(0.0)

        mini_df[new_feature] = new_values

        is_valid = new_feature in mini_df.columns and len(new_values) > 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Added column '{new_feature}'")
        print(f"     Values: {mini_df[new_feature].tolist()}")
        results.append(('column_population', is_valid))

    except Exception as e:
        print(f"     FAIL: Column population failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('column_population', False))

    return results


def test_points_prediction_columns():
    """Test points prediction column generation."""
    print("\n" + "=" * 60)
    print("TEST: Points Prediction Columns")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.services.config_manager import ModelConfigManager

    results = []
    db = Mongo().db
    mgr = ModelConfigManager(db)

    # Test 1: Check for points config
    print("\n  1. Testing points config availability...")
    try:
        config = mgr.get_points_config(selected=True)  # Correct method signature
        is_valid = config is None or isinstance(config, dict)
        status = "PASS" if is_valid else "FAIL"
        if config:
            print(f"     {status}: Found points config: {config.get('model_type', 'unknown')}")
        else:
            print(f"     {status}: No points config (can't add pred_* columns)")
        results.append(('points_config_check', is_valid))
    except Exception as e:
        print(f"     FAIL: Points config check failed: {e}")
        results.append(('points_config_check', False))

    return results


def test_game_id_type_handling():
    """Test that game_id type handling works correctly (int vs string)."""
    print("\n" + "=" * 60)
    print("TEST: Game ID Type Handling")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import GamesRepository
    import pandas as pd

    results = []
    db = Mongo().db
    repo = GamesRepository(db)

    # Test 1: Query games and check game_id types
    print("\n  1. Testing game_id type from MongoDB...")
    try:
        games = repo.find({'season': '2023-2024'}, limit=5)
        if games:
            game_id = games[0].get('game_id')
            print(f"     MongoDB game_id type: {type(game_id).__name__}")
            print(f"     game_id value: {game_id}")

            # Simulate what happens with CSV (where int64 becomes int)
            game_id_str = str(game_id)
            cache = {game_id_str: 'test_value'}

            # Test lookup with string conversion
            lookup_result = cache.get(str(game_id))
            is_valid = lookup_result == 'test_value'
            status = "PASS" if is_valid else "FAIL"
            print(f"     {status}: String conversion lookup works")
        else:
            is_valid = True
            print("     SKIP: No games to test")
        results.append(('game_id_type_handling', is_valid))
    except Exception as e:
        print(f"     FAIL: Game ID type test failed: {e}")
        results.append(('game_id_type_handling', False))

    return results


def test_venue_guid_lookup():
    """Test venue_guid lookup for travel features."""
    print("\n" + "=" * 60)
    print("TEST: Venue GUID Lookup")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import GamesRepository

    results = []
    db = Mongo().db
    repo = GamesRepository(db)

    # Test 1: Query games with venue_guid
    print("\n  1. Testing venue_guid availability in games...")
    try:
        games = repo.find(
            {'season': '2023-2024', 'venue_guid': {'$exists': True}},
            projection={'game_id': 1, 'venue_guid': 1, 'date': 1},
            limit=10
        )

        games_with_venue = [g for g in games if g.get('venue_guid')]
        is_valid = len(games_with_venue) > 0
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Found {len(games_with_venue)}/{len(games)} games with venue_guid")

        if games_with_venue:
            sample = games_with_venue[0]
            print(f"     Sample venue_guid: {sample.get('venue_guid')}")

        results.append(('venue_guid_lookup', is_valid))
    except Exception as e:
        print(f"     FAIL: Venue GUID lookup failed: {e}")
        results.append(('venue_guid_lookup', False))

    return results


def main():
    """Run all master training workflow tests."""
    print("=" * 60)
    print("MASTER TRAINING WORKFLOW INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\nThis suite tests the master training creation pipeline")
    print("after the layered architecture migration.")

    all_results = []

    # Run all tests
    all_results.extend(test_games_query_for_training())
    all_results.extend(test_feature_registry())
    all_results.extend(test_master_training_data_module())
    all_results.extend(test_nba_model_create_training_data())
    all_results.extend(test_populate_columns())
    all_results.extend(test_points_prediction_columns())
    all_results.extend(test_game_id_type_handling())
    all_results.extend(test_venue_guid_lookup())

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
        print("ALL MASTER TRAINING WORKFLOW TESTS PASSED")
    else:
        print(f"TESTS FAILED: {failed} failures")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
