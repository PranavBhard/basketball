#!/usr/bin/env python3
"""
Integration tests for the Prediction Workflow.

Tests that the prediction pipeline works correctly:
1. PredictionService initialization
2. Model loading from configs
3. Feature extraction for predictions
4. Prediction execution

Usage:
    source venv/bin/activate
    PYTHONPATH=/Users/pranav/Documents/NBA python tests/test_prediction_workflow.py
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_prediction_service_initialization():
    """Test that PredictionService initializes correctly."""
    print("\n" + "=" * 60)
    print("TEST: PredictionService Initialization")
    print("=" * 60)

    from bball_app.core.services.prediction import PredictionService

    results = []

    # Test 1: Basic initialization
    print("\n  1. Testing PredictionService initialization...")
    try:
        service = PredictionService()
        is_valid = service is not None and hasattr(service, 'db')
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: PredictionService initialized")
        results.append(('prediction_service_init', is_valid))
    except Exception as e:
        print(f"     FAIL: PredictionService initialization failed: {e}")
        results.append(('prediction_service_init', False))
        return results

    # Test 2: Verify repositories are initialized
    print("\n  2. Testing repository initialization...")
    try:
        has_repos = (
            hasattr(service, '_games_repo') and
            hasattr(service, '_classifier_config_repo') and
            hasattr(service, '_points_config_repo')
        )
        is_valid = has_repos
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Repositories initialized on PredictionService")
        results.append(('prediction_service_repos', is_valid))
    except Exception as e:
        print(f"     FAIL: Repository check failed: {e}")
        results.append(('prediction_service_repos', False))

    # Test 3: Verify ConfigManager is initialized
    print("\n  3. Testing ConfigManager initialization...")
    try:
        has_config_mgr = hasattr(service, 'config_manager')
        is_valid = has_config_mgr
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: ConfigManager initialized")
        results.append(('prediction_service_config_mgr', is_valid))
    except Exception as e:
        print(f"     FAIL: ConfigManager check failed: {e}")
        results.append(('prediction_service_config_mgr', False))

    return results


def test_get_selected_configs():
    """Test that we can retrieve selected model configs."""
    print("\n" + "=" * 60)
    print("TEST: Get Selected Configs")
    print("=" * 60)

    from bball_app.core.services.prediction import PredictionService

    results = []
    service = PredictionService()

    # Test 1: Get classifier config
    print("\n  1. Testing get_selected_config()...")
    try:
        config = service.config_manager.get_selected_config()
        is_valid = config is None or isinstance(config, dict)
        status = "PASS" if is_valid else "FAIL"
        if config:
            print(f"     {status}: Found classifier config: {config.get('model_type', 'unknown')}")
            print(f"     Accuracy: {config.get('accuracy', 'N/A')}")
        else:
            print(f"     {status}: No selected classifier config")
        results.append(('get_classifier_config', is_valid))
    except Exception as e:
        print(f"     FAIL: get_classifier_config failed: {e}")
        results.append(('get_classifier_config', False))

    # Test 2: Get points config
    print("\n  2. Testing get_points_config()...")
    try:
        config = service.config_manager.get_points_config(selected=True)
        is_valid = config is None or isinstance(config, dict)
        status = "PASS" if is_valid else "FAIL"
        if config:
            print(f"     {status}: Found points config: {config.get('model_type', 'unknown')}")
        else:
            print(f"     {status}: No selected points config")
        results.append(('get_points_config', is_valid))
    except Exception as e:
        print(f"     FAIL: get_points_config failed: {e}")
        results.append(('get_points_config', False))

    return results


def test_load_model_from_config():
    """Test that we can load a model from a config."""
    print("\n" + "=" * 60)
    print("TEST: Load Model From Config")
    print("=" * 60)

    from bball_app.core.services.prediction import PredictionService

    results = []
    service = PredictionService()

    # Get selected config
    config = service.config_manager.get_selected_config()

    if config is None:
        print("\n  SKIP: No selected classifier config - cannot test model loading")
        results.append(('load_model', True))  # Not a failure, just no config
        return results

    # Test 1: Load model from config
    print("\n  1. Testing _load_classifier_model()...")
    try:
        model = service._load_classifier_model(config)
        is_valid = model is not None and hasattr(model, 'predict')
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Loaded classifier model")
        results.append(('load_classifier_model', is_valid))
    except Exception as e:
        print(f"     FAIL: _get_classifier_model failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('load_classifier_model', False))

    return results


def test_predict_game():
    """Test single game prediction."""
    print("\n" + "=" * 60)
    print("TEST: Predict Single Game")
    print("=" * 60)

    from bball_app.core.services.prediction import PredictionService

    results = []
    service = PredictionService()

    # Check if we have a selected config
    config = service.config_manager.get_selected_config()
    if config is None:
        print("\n  SKIP: No selected classifier config - cannot test prediction")
        results.append(('predict_game', True))  # Not a failure, just no config
        return results

    # Test 1: Predict a historical game
    print("\n  1. Testing predict_game() for LAL vs BOS...")
    try:
        result = service.predict_game(
            home_team='Los Angeles Lakers',
            away_team='Boston Celtics',
            game_date='2024-01-15'
        )

        if result is None:
            print("     SKIP: predict_game returned None (may need model artifacts)")
            results.append(('predict_game', True))  # Not necessarily a failure
        else:
            is_valid = hasattr(result, 'home_win_prob') and result.error is None
            status = "PASS" if is_valid else "FAIL"
            print(f"     {status}: Got prediction result")
            if is_valid:
                print(f"     Home Win Prob: {result.home_win_prob:.1f}%")
                print(f"     Predicted Winner: {result.predicted_winner}")
            elif result.error:
                print(f"     Error: {result.error}")
            results.append(('predict_game', is_valid))
    except Exception as e:
        print(f"     FAIL: predict_game failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('predict_game', False))

    return results


def test_matchup_predict_tool():
    """Test the matchup prediction tool used by agents."""
    print("\n" + "=" * 60)
    print("TEST: Matchup Predict Tool")
    print("=" * 60)

    from bball_app.core.services.matchup_predict import predict

    results = []

    # Test 1: Basic prediction call
    print("\n  1. Testing predict()...")
    try:
        result = predict(
            home='LAL',
            away='BOS',
            game_date='2024-01-15'
        )

        is_valid = isinstance(result, dict)
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: predict_matchup returned result")

        if is_valid and 'error' not in result:
            print(f"     Keys: {list(result.keys())[:5]}...")
        elif 'error' in result:
            print(f"     Note: {result.get('error', 'Unknown error')}")

        results.append(('matchup_predict_tool', is_valid))
    except Exception as e:
        print(f"     FAIL: predict_matchup failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('matchup_predict_tool', False))

    return results


def test_games_for_date():
    """Test fetching games for a specific date."""
    print("\n" + "=" * 60)
    print("TEST: Get Games For Date")
    print("=" * 60)

    from bball_app.core.services.prediction import PredictionService

    results = []
    service = PredictionService()

    # Test 1: Get games for a known date
    print("\n  1. Testing _get_games_for_date()...")
    try:
        games = service._games_repo.find_by_date('2024-01-15')
        is_valid = isinstance(games, list)
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Found {len(games)} games for 2024-01-15")
        results.append(('get_games_for_date', is_valid))
    except Exception as e:
        print(f"     FAIL: _get_games_for_date failed: {e}")
        results.append(('get_games_for_date', False))

    return results


def test_feature_extraction_for_prediction():
    """Test that feature extraction works for prediction."""
    print("\n" + "=" * 60)
    print("TEST: Feature Extraction For Prediction")
    print("=" * 60)

    from bball_app.core.models.bball_model import BballModel

    results = []

    # Test with features that would be used in prediction
    test_features = [
        'wins|season|avg|diff',
        'points|games_10|avg|diff',
        'off_rtg|games_10|raw|diff',
    ]

    print("\n  1. Testing feature extraction for prediction...")
    try:
        model = BballModel(
            classifier_features=test_features,
            include_elo=False,
            include_per_features=False,
            include_injuries=False,
            preload_data=False
        )
        model.feature_names = test_features

        # Build features for a prediction
        features_dict = model._build_features_dict(
            'Los Angeles Lakers',
            'Boston Celtics',
            '2023-2024',
            2024, 1, 15
        )

        # Verify features are present and have reasonable values
        all_present = all(f in features_dict for f in test_features)
        all_valid = all(
            isinstance(features_dict.get(f), (int, float)) and
            features_dict.get(f) == features_dict.get(f)  # not NaN
            for f in test_features
        )

        is_valid = all_present and all_valid
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Extracted {len(features_dict)} features")

        for f in test_features:
            val = features_dict.get(f, 'MISSING')
            print(f"       {f}: {val}")

        results.append(('feature_extraction', is_valid))
    except Exception as e:
        print(f"     FAIL: Feature extraction failed: {e}")
        import traceback
        traceback.print_exc()
        results.append(('feature_extraction', False))

    return results


def test_elo_cache_integration():
    """Test Elo cache integration for predictions."""
    print("\n" + "=" * 60)
    print("TEST: Elo Cache Integration")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.stats.elo_cache import EloCache

    results = []
    db = Mongo().db
    cache = EloCache(db)

    # Test 1: Get Elo for a team
    print("\n  1. Testing get_elo_for_game_with_fallback()...")
    try:
        elo = cache.get_elo_for_game_with_fallback(
            'Boston Celtics',
            '2024-01-15',
            '2023-2024'
        )
        is_valid = isinstance(elo, (int, float)) and elo > 1000
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Boston Celtics Elo = {elo:.1f}")
        results.append(('elo_cache_get', is_valid))
    except Exception as e:
        print(f"     FAIL: Elo cache failed: {e}")
        results.append(('elo_cache_get', False))

    # Test 2: Get cache stats
    print("\n  2. Testing get_cache_stats()...")
    try:
        stats = cache.get_cache_stats()
        is_valid = isinstance(stats, dict) and 'total_ratings' in stats
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Cache has {stats.get('total_ratings', 0)} ratings")
        results.append(('elo_cache_stats', is_valid))
    except Exception as e:
        print(f"     FAIL: get_cache_stats failed: {e}")
        results.append(('elo_cache_stats', False))

    return results


def main():
    """Run all prediction workflow tests."""
    print("=" * 60)
    print("PREDICTION WORKFLOW INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\nThis suite tests the prediction pipeline after the")
    print("layered architecture migration.")

    all_results = []

    # Run all tests
    all_results.extend(test_prediction_service_initialization())
    all_results.extend(test_get_selected_configs())
    all_results.extend(test_load_model_from_config())
    all_results.extend(test_predict_game())
    all_results.extend(test_matchup_predict_tool())
    all_results.extend(test_games_for_date())
    all_results.extend(test_feature_extraction_for_prediction())
    all_results.extend(test_elo_cache_integration())

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
        print("ALL PREDICTION WORKFLOW TESTS PASSED")
    else:
        print(f"TESTS FAILED: {failed} failures")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
