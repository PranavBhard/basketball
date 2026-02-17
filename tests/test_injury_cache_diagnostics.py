#!/usr/bin/env python3
"""
Diagnose injury preload cache state used by master-training Add Columns flow.

This mirrors the SharedFeatureContext preloading logic to verify:
- _injury_cache_loaded is True
- how many (team, season) keys are cached
- how many cached keys have zero player records
- which (team, season) combos from stats_nba are missing in the cache

Usage:
  source venv/bin/activate
  PYTHONPATH=/Users/pranav/Documents/NBA \
    python tests/test_injury_cache_diagnostics.py
"""

import os
import sys

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def main() -> int:
    from bball_app.core.mongo import Mongo
    from bball_app.core.stats.handler import StatHandlerV2
    from bball_app.core.utils.collection import import_collection

    try:
        mongo = Mongo()
        db = mongo.db
    except Exception as e:
        print(f"SKIP: Mongo not configured or not reachable: {e}")
        return 0

    print("=" * 80)
    print("INJURY CACHE DIAGNOSTICS")
    print("=" * 80)

    print("Loading games from stats_nba...")
    all_games = import_collection('stats_nba')
    games_home, games_away = all_games

    # Build game list as populate_master_training_cols does
    games_list = []
    for season_data in games_home.values():
        for date_data in season_data.values():
            for game in date_data.values():
                games_list.append(game)

    print(f"Loaded {len(games_list)} games")

    # Build expected (team, season) keys from games
    team_seasons = set()
    for game in games_list:
        season = game.get('season')
        home_team = (game.get('homeTeam') or {}).get('name')
        away_team = (game.get('awayTeam') or {}).get('name')
        if home_team and season:
            team_seasons.add((home_team, season))
        if away_team and season:
            team_seasons.add((away_team, season))

    print(f"Expected team-seasons from games: {len(team_seasons)}")

    print("\nInitializing StatHandlerV2 (preloaded games)...")
    stat_handler = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=all_games,
        db=db,
        lazy_load=False
    )

    print("Preloading injury cache...")
    stat_handler.preload_injury_cache(games_list)

    cache_loaded = getattr(stat_handler, "_injury_cache_loaded", False)
    cache = getattr(stat_handler, "_injury_preloaded_players", {})
    cache_keys = set(cache.keys())

    print(f"\n_cache_loaded: {cache_loaded}")
    print(f"Cached team-seasons: {len(cache_keys)}")

    empty_keys = [k for k, v in cache.items() if not v]
    print(f"Cached keys with zero records: {len(empty_keys)}")

    missing_keys = sorted(team_seasons - cache_keys)
    print(f"Team-seasons missing from cache: {len(missing_keys)}")

    # Show a small sample for debugging
    def _sample(lst, n=15):
        return lst[:n]

    if empty_keys:
        print("\nSample empty cached keys:")
        for k in _sample(empty_keys):
            print(f"  - {k}")

    if missing_keys:
        print("\nSample missing keys (not cached):")
        for k in _sample(missing_keys):
            print(f"  - {k}")

    # Check for full-name team keys (possible mismatch)
    full_name_cached = [k for k in cache_keys if " " in k[0]]
    full_name_expected = [k for k in team_seasons if " " in k[0]]
    print(f"\nCached keys with full team names: {len(full_name_cached)}")
    print(f"Expected keys with full team names: {len(full_name_expected)}")

    # Sample counts from stats_nba_players for missing keys (first 5)
    if missing_keys:
        print("\nSample stats_nba_players counts for missing keys:")
        for team, season in _sample(missing_keys, n=5):
            count = db.stats_nba_players.count_documents(
                {'team': team, 'season': season, 'stats.min': {'$gt': 0}}
            )
            print(f"  - {team} {season}: {count} player records")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
