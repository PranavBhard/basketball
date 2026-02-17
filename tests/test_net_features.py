#!/usr/bin/env python3
"""
Test script to investigate _net feature calculations.

Tests:
1. Whether team_against_agg is being populated correctly
2. Whether opp_effective_fg_perc, opp_true_shooting_perc, etc. return different values
3. Whether there's any caching or reuse of values that shouldn't be shared
"""

import sys
import os

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
# Script is in nba_app/tests/, so go up two levels to NBA/ (parent of nba_app/)
nba_app_dir = os.path.dirname(script_dir)  # nba_app/
nba_dir = os.path.dirname(nba_app_dir)  # NBA/
if nba_dir not in sys.path:
    sys.path.insert(0, nba_dir)

from bball_app.core.stats.handler import StatHandlerV2
from bball_app.core.mongo import Mongo
from datetime import date
from collections import defaultdict


def test_team_against_agg_population():
    """Test if team_against_agg is being populated correctly."""
    print("=" * 80)
    print("TEST 1: team_against_agg Population")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    stat_handler = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=None,
        db=db,
        lazy_load=False
    )
    
    # Get a sample game to test with (exclude October and November, use later in season)
    # Find a game that has at least 20 games before it for both teams
    # Only use games from 2008-2009 season onwards
    sample_game = None
    potential_games = list(db.stats_nba.find({
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': ['preseason', 'allstar']},
        'month': {'$gte': 1, '$lte': 6},  # Use January through June
        'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
    }).sort('date', 1).limit(100))
    
    for game in potential_games:
        home_team = game['homeTeam']['name']
        away_team = game['awayTeam']['name']
        season = game.get('season', '2023-2024')
        game_date = game['date']
        year, month, day = map(int, game_date.split('-'))
        
        # Check if both teams have enough games before this date
        home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
        away_games = stat_handler.get_team_games_before_date(away_team, year, month, day, season)
        
        if len(home_games) >= 20 and len(away_games) >= 20:
            sample_game = game
            break
    
    if not sample_game:
        # Fallback: just get any game from later in season
        sample_game = db.stats_nba.find_one({
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': ['preseason', 'allstar']},
            'month': {'$gte': 1, '$lte': 6},  # January through June
            'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
        }, sort=[('date', 1)])
    
    if not sample_game:
        print("ERROR: No sample game found in database")
        return False
    
    home_team = sample_game['homeTeam']['name']
    away_team = sample_game['awayTeam']['name']
    season = sample_game.get('season', '2023-2024')
    game_date = sample_game['date']
    year, month, day = map(int, game_date.split('-'))
    
    print(f"\nTesting with game: {away_team} @ {home_team} on {game_date}")
    
    # Get games for home team
    home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
    print(f"\nHome team ({home_team}) has {len(home_games)} games before {game_date}")
    
    # Manually aggregate to check team_against_agg
    team_agg = defaultdict(float)
    team_against_agg = defaultdict(float)
    
    for game in home_games[:5]:  # Check first 5 games
        if game['homeTeam']['name'] == home_team:
            team_data = game['homeTeam']
            opp_data = game['awayTeam']
        elif game['awayTeam']['name'] == home_team:
            team_data = game['awayTeam']
            opp_data = game['homeTeam']
        else:
            continue
        
        # Aggregate team stats
        for key, value in team_data.items():
            if isinstance(value, (int, float)):
                team_agg[key] += value
        
        # Aggregate opponent stats
        for key, value in opp_data.items():
            if isinstance(value, (int, float)):
                team_against_agg[key] += value
    
    print(f"\nTeam aggregated stats (sample):")
    print(f"  FG_made: {team_agg.get('FG_made', 0):.1f}")
    print(f"  FG_att: {team_agg.get('FG_att', 0):.1f}")
    print(f"  three_made: {team_agg.get('three_made', 0):.1f}")
    print(f"  points: {team_agg.get('points', 0):.1f}")
    
    print(f"\nOpponent aggregated stats (team_against_agg):")
    print(f"  FG_made: {team_against_agg.get('FG_made', 0):.1f}")
    print(f"  FG_att: {team_against_agg.get('FG_att', 0):.1f}")
    print(f"  three_made: {team_against_agg.get('three_made', 0):.1f}")
    print(f"  points: {team_against_agg.get('points', 0):.1f}")
    
    # Check if team_against_agg is different from team_agg
    if team_agg.get('FG_made', 0) == team_against_agg.get('FG_made', 0):
        print("\n⚠️  WARNING: team_agg and team_against_agg have same FG_made values!")
        print("   This suggests team_against_agg is not being populated correctly.")
        return False
    else:
        print("\n✓ team_against_agg is populated with different values from team_agg")
        return True


def test_opponent_stats_different_values():
    """Test if different opponent stats return different values."""
    print("\n" + "=" * 80)
    print("TEST 2: Opponent Stats Return Different Values")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    stat_handler = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=None,
        db=db,
        lazy_load=False
    )
    
    # Get a sample game (exclude October and November, use later in season)
    # Find a game that has at least 20 games before it for both teams
    # Only use games from 2008-2009 season onwards
    sample_game = None
    potential_games = list(db.stats_nba.find({
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': ['preseason', 'allstar']},
        'month': {'$gte': 1, '$lte': 6},  # January through June
        'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
    }).sort('date', 1).limit(100))
    
    for game in potential_games:
        home_team = game['homeTeam']['name']
        away_team = game['awayTeam']['name']
        season = game.get('season', '2023-2024')
        game_date = game['date']
        year, month, day = map(int, game_date.split('-'))
        
        # Check if both teams have enough games before this date
        home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
        away_games = stat_handler.get_team_games_before_date(away_team, year, month, day, season)
        
        if len(home_games) >= 20 and len(away_games) >= 20:
            sample_game = game
            break
    
    if not sample_game:
        # Fallback
        sample_game = db.stats_nba.find_one({
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': ['preseason', 'allstar']},
            'month': {'$gte': 1, '$lte': 6},
            'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
        }, sort=[('date', 1)])
    
    if not sample_game:
        print("ERROR: No sample game found")
        return False
    
    home_team = sample_game['homeTeam']['name']
    away_team = sample_game['awayTeam']['name']
    season = sample_game.get('season', '2023-2024')
    game_date = sample_game['date']
    year, month, day = map(int, game_date.split('-'))
    
    print(f"\nTesting with game: {away_team} @ {home_team} on {game_date}")
    
    # Get games
    home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
    away_games = stat_handler.get_team_games_before_date(away_team, year, month, day, season)
    
    if len(home_games) == 0 or len(away_games) == 0:
        print(f"ERROR: Not enough games (home: {len(home_games)}, away: {len(away_games)})")
        return False
    
    reference_date = date(year, month, day)
    
    # Test different opponent stats
    opponent_stats = [
        'opp_effective_fg_perc',
        'opp_true_shooting_perc',
        'opp_three_perc',
        'opp_assists_ratio',
        'opp_points'
    ]
    
    print(f"\nCalculating opponent stats for {home_team} (from {len(home_games)} games):")
    home_opp_values = {}
    for stat_name in opponent_stats:
        value = stat_handler._calculate_team_stat(
            stat_name, home_team, home_games, 'raw', reference_date
        )
        home_opp_values[stat_name] = value
        print(f"  {stat_name}: {value:.4f}" if value is not None else f"  {stat_name}: None")
    
    print(f"\nCalculating opponent stats for {away_team} (from {len(away_games)} games):")
    away_opp_values = {}
    for stat_name in opponent_stats:
        value = stat_handler._calculate_team_stat(
            stat_name, away_team, away_games, 'raw', reference_date
        )
        away_opp_values[stat_name] = value
        print(f"  {stat_name}: {value:.4f}" if value is not None else f"  {stat_name}: None")
    
    # Check if all values are the same
    home_values = [v for v in home_opp_values.values() if v is not None]
    away_values = [v for v in away_opp_values.values() if v is not None]
    
    if len(set(home_values)) == 1 and len(home_values) > 1:
        print("\n⚠️  WARNING: All home opponent stats have the same value!")
        return False
    elif len(set(away_values)) == 1 and len(away_values) > 1:
        print("\n⚠️  WARNING: All away opponent stats have the same value!")
        return False
    else:
        print("\n✓ Different opponent stats return different values")
        return True


def test_net_vs_regular_features():
    """Test if net features are calculated differently from regular features."""
    print("\n" + "=" * 80)
    print("TEST 3: Net vs Regular Feature Calculations")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    stat_handler = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=None,
        db=db,
        lazy_load=False
    )
    
    # Get multiple sample games (exclude October and November, use later in season)
    # Filter to games that have sufficient historical data
    # Only use games from 2008-2009 season onwards
    stat_handler_temp = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=None,
        db=db,
        lazy_load=False
    )
    
    potential_games = list(db.stats_nba.find({
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': ['preseason', 'allstar']},
        'month': {'$gte': 1, '$lte': 6},  # January through June
        'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
    }).sort('date', 1).limit(100))
    
    sample_games = []
    for game in potential_games:
        home_team = game['homeTeam']['name']
        away_team = game['awayTeam']['name']
        season = game.get('season', '2023-2024')
        game_date = game['date']
        year, month, day = map(int, game_date.split('-'))
        
        # Check if both teams have enough games before this date
        home_games = stat_handler_temp.get_team_games_before_date(home_team, year, month, day, season)
        away_games = stat_handler_temp.get_team_games_before_date(away_team, year, month, day, season)
        
        if len(home_games) >= 20 and len(away_games) >= 20:
            sample_games.append(game)
            if len(sample_games) >= 10:
                break
    
    if not sample_games:
        print("ERROR: No sample games found")
        return False
    
    print(f"\nTesting with {len(sample_games)} games:")
    
    net_features = [
        'efg_net|season|raw|diff',
        'ts_net|season|raw|diff',
        'three_pct_net|season|raw|diff',
        'off_rtg_net|season|raw|diff',
        'assists_ratio_net|season|raw|diff',
    ]
    
    regular_features = [
        'efg|season|raw|diff',
        'ts|season|raw|diff',
        'three_pct|season|raw|diff',
        'off_rtg|season|raw|diff',
        'assists_ratio|season|raw|diff',
    ]
    
    results = []
    
    for game in sample_games:
        home_team = game['homeTeam']['name']
        away_team = game['awayTeam']['name']
        season = game.get('season', '2023-2024')
        game_date = game['date']
        year, month, day = map(int, game_date.split('-'))
        
        game_results = {
            'game': f"{away_team} @ {home_team}",
            'date': game_date,
            'net_features': {},
            'regular_features': {}
        }
        
        # Calculate net features
        for feature_name in net_features:
            try:
                value = stat_handler.calculate_feature(
                    feature_name, home_team, away_team, season, year, month, day
                )
                game_results['net_features'][feature_name] = value
            except Exception as e:
                game_results['net_features'][feature_name] = f"ERROR: {e}"
        
        # Calculate regular features
        for feature_name in regular_features:
            try:
                value = stat_handler.calculate_feature(
                    feature_name, home_team, away_team, season, year, month, day
                )
                game_results['regular_features'][feature_name] = value
            except Exception as e:
                game_results['regular_features'][feature_name] = f"ERROR: {e}"
        
        results.append(game_results)
    
    # Analyze results
    print("\n" + "-" * 80)
    print("Results Summary:")
    print("-" * 80)
    
    for i, result in enumerate(results[:3], 1):  # Show first 3 games
        print(f"\nGame {i}: {result['game']} ({result['date']})")
        print("\n  Net Features:")
        for feature, value in result['net_features'].items():
            print(f"    {feature}: {value:.6f}" if isinstance(value, (int, float)) else f"    {feature}: {value}")
        print("\n  Regular Features:")
        for feature, value in result['regular_features'].items():
            print(f"    {feature}: {value:.6f}" if isinstance(value, (int, float)) else f"    {feature}: {value}")
    
    # Check if all net features have the same value
    print("\n" + "-" * 80)
    print("Checking for identical values:")
    print("-" * 80)
    
    all_net_values = []
    for result in results:
        net_vals = [v for v in result['net_features'].values() if isinstance(v, (int, float))]
        all_net_values.extend(net_vals)
    
    if len(set(all_net_values)) == 1 and len(all_net_values) > 1:
        print("⚠️  WARNING: All net feature values are identical!")
        print(f"   Value: {all_net_values[0]}")
        return False
    else:
        unique_values = len(set(all_net_values))
        print(f"✓ Net features have {unique_values} unique values out of {len(all_net_values)} total")
        
        # Check if net and regular features are different
        print("\nComparing net vs regular features:")
        for result in results[:3]:
            print(f"\n  {result['game']}:")
            for base_stat in ['efg', 'ts', 'three_pct', 'off_rtg', 'assists_ratio']:
                net_feat = f"{base_stat}_net|season|raw|diff"
                reg_feat = f"{base_stat}|season|raw|diff"
                net_val = result['net_features'].get(net_feat)
                reg_val = result['regular_features'].get(reg_feat)
                
                if isinstance(net_val, (int, float)) and isinstance(reg_val, (int, float)):
                    if abs(net_val - reg_val) < 0.0001:
                        print(f"    ⚠️  {base_stat}: net={net_val:.6f}, regular={reg_val:.6f} (SAME!)")
                    else:
                        print(f"    ✓ {base_stat}: net={net_val:.6f}, regular={reg_val:.6f} (different)")
        
        return True


def test_caching_issues():
    """Test for caching or reuse of values."""
    print("\n" + "=" * 80)
    print("TEST 4: Caching/Reuse Issues")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    stat_handler = StatHandlerV2(
        statistics=[],
        use_exponential_weighting=False,
        preloaded_games=None,
        db=db,
        lazy_load=False
    )
    
    # Get a sample game (exclude October and November, use later in season)
    # Find a game that has at least 20 games before it for both teams
    # Only use games from 2008-2009 season onwards
    sample_game = None
    potential_games = list(db.stats_nba.find({
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': ['preseason', 'allstar']},
        'month': {'$gte': 1, '$lte': 6},  # January through June
        'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
    }).sort('date', 1).limit(100))
    
    for game in potential_games:
        home_team = game['homeTeam']['name']
        away_team = game['awayTeam']['name']
        season = game.get('season', '2023-2024')
        game_date = game['date']
        year, month, day = map(int, game_date.split('-'))
        
        # Check if both teams have enough games before this date
        home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
        away_games = stat_handler.get_team_games_before_date(away_team, year, month, day, season)
        
        if len(home_games) >= 20 and len(away_games) >= 20:
            sample_game = game
            break
    
    if not sample_game:
        # Fallback
        sample_game = db.stats_nba.find_one({
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': ['preseason', 'allstar']},
            'month': {'$gte': 1, '$lte': 6},
            'season': {'$gte': '2008-2009'}  # Only 2008-2009 season and onwards
        }, sort=[('date', 1)])
    
    if not sample_game:
        print("ERROR: No sample game found")
        return False
    
    home_team = sample_game['homeTeam']['name']
    away_team = sample_game['awayTeam']['name']
    season = sample_game.get('season', '2023-2024')
    game_date = sample_game['date']
    year, month, day = map(int, game_date.split('-'))
    
    print(f"\nTesting with game: {away_team} @ {home_team} on {game_date}")
    
    # Verify we have games
    home_games = stat_handler.get_team_games_before_date(home_team, year, month, day, season)
    away_games = stat_handler.get_team_games_before_date(away_team, year, month, day, season)
    if len(home_games) == 0 or len(away_games) == 0:
        print(f"ERROR: Not enough games (home: {len(home_games)}, away: {len(away_games)})")
        return False
    
    # Calculate same feature multiple times
    feature_name = 'efg_net|season|raw|diff'
    values = []
    
    print(f"\nCalculating '{feature_name}' 5 times:")
    for i in range(5):
        value = stat_handler.calculate_feature(
            feature_name, home_team, away_team, season, year, month, day
        )
        values.append(value)
        print(f"  Attempt {i+1}: {value:.6f}")
    
    if len(set(values)) == 1:
        print("\n✓ Feature calculation is consistent (same value each time)")
    else:
        print("\n⚠️  WARNING: Feature calculation returns different values!")
        print(f"   Values: {values}")
        return False
    
    # Test different net features for same game
    net_features = [
        'efg_net|season|raw|diff',
        'ts_net|season|raw|diff',
        'three_pct_net|season|raw|diff',
        'off_rtg_net|season|raw|diff',
        'assists_ratio_net|season|raw|diff',
    ]
    
    print(f"\nCalculating different net features for same game:")
    net_values = {}
    for feature_name in net_features:
        value = stat_handler.calculate_feature(
            feature_name, home_team, away_team, season, year, month, day
        )
        net_values[feature_name] = value
        print(f"  {feature_name}: {value:.6f}")
    
    unique_values = len(set(net_values.values()))
    if unique_values == 1:
        print(f"\n⚠️  WARNING: All {len(net_features)} net features have the same value!")
        print(f"   This suggests a caching or calculation bug.")
        return False
    else:
        print(f"\n✓ Different net features return different values ({unique_values} unique)")
        return True


if __name__ == '__main__':
    print("=" * 80)
    print("NET FEATURES INVESTIGATION TEST SUITE")
    print("=" * 80)
    
    results = {}
    
    try:
        results['test1'] = test_team_against_agg_population()
    except Exception as e:
        print(f"\nERROR in test 1: {e}")
        import traceback
        traceback.print_exc()
        results['test1'] = False
    
    try:
        results['test2'] = test_opponent_stats_different_values()
    except Exception as e:
        print(f"\nERROR in test 2: {e}")
        import traceback
        traceback.print_exc()
        results['test2'] = False
    
    try:
        results['test3'] = test_net_vs_regular_features()
    except Exception as e:
        print(f"\nERROR in test 3: {e}")
        import traceback
        traceback.print_exc()
        results['test3'] = False
    
    try:
        results['test4'] = test_caching_issues()
    except Exception as e:
        print(f"\nERROR in test 4: {e}")
        import traceback
        traceback.print_exc()
        results['test4'] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")

