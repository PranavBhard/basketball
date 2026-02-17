#!/usr/bin/env python3
"""
Integration tests for the Data Layer (Repository Pattern).

Tests that all repositories can:
1. Connect to MongoDB correctly
2. Execute basic CRUD operations
3. Return properly typed results

Usage:
    source venv/bin/activate
    PYTHONPATH=/Users/pranav/Documents/NBA python tests/test_data_layer_integration.py
"""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_mongo_connection():
    """Test that Mongo connection works."""
    print("\n" + "=" * 60)
    print("TEST: MongoDB Connection")
    print("=" * 60)

    from bball_app.core.mongo import Mongo

    try:
        mongo = Mongo()
        db = mongo.db

        # Verify db is not None and has expected collections
        collections = db.list_collection_names()
        print(f"  Connected to database: {db.name}")
        print(f"  Available collections: {len(collections)}")

        expected_collections = ['stats_nba', 'stats_nba_players', 'model_config_nba']
        found = [c for c in expected_collections if c in collections]
        print(f"  Expected collections found: {found}")

        is_valid = len(found) == len(expected_collections)
        status = "PASS" if is_valid else "FAIL"
        print(f"  {status}: MongoDB connection and collections")
        return [('mongo_connection', is_valid)]
    except Exception as e:
        print(f"  FAIL: MongoDB connection failed: {e}")
        return [('mongo_connection', False)]


def test_games_repository():
    """Test GamesRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: GamesRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import GamesRepository

    results = []
    db = Mongo().db
    repo = GamesRepository(db)

    # Test 1: find() with query
    print("\n  1. Testing find() with season filter...")
    games = repo.find({'season': '2023-2024'}, limit=5)
    is_valid = len(games) > 0 and all(g.get('season') == '2023-2024' for g in games)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: Found {len(games)} games for 2023-2024 season")
    results.append(('games_find_with_filter', is_valid))

    # Test 2: find() with sort
    print("\n  2. Testing find() with sort...")
    games = repo.find({'season': '2023-2024'}, sort=[('date', -1)], limit=5)
    if len(games) >= 2:
        dates_descending = all(games[i].get('date', '') >= games[i+1].get('date', '')
                               for i in range(len(games)-1))
        is_valid = dates_descending
    else:
        is_valid = len(games) > 0
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: Sort order correct")
    results.append(('games_find_with_sort', is_valid))

    # Test 3: find_one()
    print("\n  3. Testing find_one()...")
    game = repo.find_one({'season': '2023-2024'})
    is_valid = game is not None and 'homeTeam' in game and 'awayTeam' in game
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: find_one returned valid game document")
    results.append(('games_find_one', is_valid))

    # Test 4: count()
    print("\n  4. Testing count()...")
    count = repo.count({'season': '2023-2024'})
    is_valid = count > 0
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: count returned {count} games")
    results.append(('games_count', is_valid))

    # Test 5: find_by_date() - custom method
    print("\n  5. Testing find_by_date()...")
    games = repo.find_by_date('2024-01-15')
    is_valid = isinstance(games, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: find_by_date returned {len(games)} games")
    results.append(('games_find_by_date', is_valid))

    return results


def test_player_stats_repository():
    """Test PlayerStatsRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: PlayerStatsRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import PlayerStatsRepository

    results = []
    db = Mongo().db
    repo = PlayerStatsRepository(db)

    # Test 1: find() with query
    print("\n  1. Testing find() with team filter...")
    stats = repo.find({'team': 'BOS', 'season': '2023-2024'}, limit=10)
    is_valid = len(stats) > 0
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: Found {len(stats)} player stats for BOS")
    results.append(('player_stats_find', is_valid))

    # Test 2: Verify stats structure
    print("\n  2. Testing stats document structure...")
    if stats:
        stat = stats[0]
        has_required = all(k in stat for k in ['player_id', 'team', 'season', 'stats'])
        is_valid = has_required
        status = "PASS" if is_valid else "FAIL"
        print(f"     {status}: Stats document has required fields")
    else:
        is_valid = False
        print(f"     FAIL: No stats to verify")
    results.append(('player_stats_structure', is_valid))

    return results


def test_classifier_config_repository():
    """Test ClassifierConfigRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: ClassifierConfigRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import ClassifierConfigRepository

    results = []
    db = Mongo().db
    repo = ClassifierConfigRepository(db)

    # Test 1: find_selected()
    print("\n  1. Testing find_selected()...")
    config = repo.find_selected()
    is_valid = config is None or (isinstance(config, dict) and 'selected' in config)
    status = "PASS" if is_valid else "FAIL"
    if config:
        print(f"     {status}: Found selected config: {config.get('model_type', 'unknown')}")
    else:
        print(f"     {status}: No selected config (valid state)")
    results.append(('classifier_config_find_selected', is_valid))

    # Test 2: find_all()
    print("\n  2. Testing find_all()...")
    configs = repo.find_all(limit=5)
    is_valid = isinstance(configs, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: find_all returned {len(configs)} configs")
    results.append(('classifier_config_find_all', is_valid))

    # Test 3: has_selected()
    print("\n  3. Testing has_selected()...")
    has_sel = repo.has_selected()
    is_valid = isinstance(has_sel, bool)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: has_selected = {has_sel}")
    results.append(('classifier_config_has_selected', is_valid))

    return results


def test_points_config_repository():
    """Test PointsConfigRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: PointsConfigRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import PointsConfigRepository

    results = []
    db = Mongo().db
    repo = PointsConfigRepository(db)

    # Test 1: find_selected()
    print("\n  1. Testing find_selected()...")
    config = repo.find_selected()
    is_valid = config is None or isinstance(config, dict)
    status = "PASS" if is_valid else "FAIL"
    if config:
        print(f"     {status}: Found selected points config")
    else:
        print(f"     {status}: No selected points config (valid state)")
    results.append(('points_config_find_selected', is_valid))

    # Test 2: find_all()
    print("\n  2. Testing find_all()...")
    configs = repo.find_all(limit=5)
    is_valid = isinstance(configs, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: find_all returned {len(configs)} configs")
    results.append(('points_config_find_all', is_valid))

    return results


def test_experiment_runs_repository():
    """Test ExperimentRunsRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: ExperimentRunsRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import ExperimentRunsRepository

    results = []
    db = Mongo().db
    repo = ExperimentRunsRepository(db)

    # Test 1: find_by_date_range()
    print("\n  1. Testing find_by_date_range()...")
    runs = repo.find_by_date_range(limit=5)
    is_valid = isinstance(runs, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: find_by_date_range returned {len(runs)} runs")
    results.append(('experiment_runs_find', is_valid))

    # Test 2: count_by_session() with non-existent session
    print("\n  2. Testing count_by_session()...")
    count = repo.count_by_session('non-existent-session-id')
    is_valid = count == 0
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: count_by_session returned {count} for non-existent session")
    results.append(('experiment_runs_count', is_valid))

    return results


def test_rosters_repository():
    """Test RostersRepository operations."""
    print("\n" + "=" * 60)
    print("TEST: RostersRepository")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import RostersRepository

    results = []
    db = Mongo().db
    repo = RostersRepository(db)

    # Test 1: find() with team filter
    print("\n  1. Testing find() with team filter...")
    rosters = repo.find({'team': 'BOS'}, limit=5)
    is_valid = isinstance(rosters, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: Found {len(rosters)} roster entries for BOS")
    results.append(('rosters_find', is_valid))

    return results


def test_league_stats_cache():
    """Test LeagueStatsCache operations."""
    print("\n" + "=" * 60)
    print("TEST: LeagueStatsCache")
    print("=" * 60)

    from bball_app.core.mongo import Mongo
    from bball_app.core.data import LeagueStatsCache

    results = []
    db = Mongo().db
    cache = LeagueStatsCache(db)

    # Test 1: find() for cached stats
    print("\n  1. Testing find() for cached league stats...")
    stats = cache.find({'season': '2023-2024'}, limit=1)
    is_valid = isinstance(stats, list)
    status = "PASS" if is_valid else "FAIL"
    print(f"     {status}: Found {len(stats)} cached league stats entries")
    results.append(('league_stats_cache_find', is_valid))

    return results


def main():
    """Run all data layer integration tests."""
    print("=" * 60)
    print("DATA LAYER INTEGRATION TEST SUITE")
    print("=" * 60)
    print("\nThis suite tests that all repositories work correctly")
    print("after the layered architecture migration.")

    all_results = []

    # Run all tests
    all_results.extend(test_mongo_connection())
    all_results.extend(test_games_repository())
    all_results.extend(test_player_stats_repository())
    all_results.extend(test_classifier_config_repository())
    all_results.extend(test_points_config_repository())
    all_results.extend(test_experiment_runs_repository())
    all_results.extend(test_rosters_repository())
    all_results.extend(test_league_stats_cache())

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
        print("ALL DATA LAYER TESTS PASSED")
    else:
        print(f"TESTS FAILED: {failed} failures")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
