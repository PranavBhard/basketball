#!/usr/bin/env python3
"""
Debug script to trace the exact prediction flow for MIL@DEN on 2026-01-11
to find why Giannis (3032977) and 4397475 are not found in available players.
"""

import sys
import os
from datetime import datetime

# Add parent directory to path (the directory containing nba_app folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)  # This is the nba_app folder
parent_of_nba_app = os.path.dirname(nba_app_dir)  # Directory containing nba_app
if parent_of_nba_app not in sys.path:
    sys.path.insert(0, parent_of_nba_app)

from bball_app.core.mongo import Mongo
from bball_app.core.stats.per_calculator import PERCalculator
from bball_app.core.models.bball_model import BballModel

def debug_mil_den_prediction():
    """Debug the exact prediction flow for MIL@DEN 2026-01-11."""
    
    # Exact parameters from the user's report
    home_team = 'DEN'
    away_team = 'MIL'
    game_date_str = '2026-01-11'
    season = '2025-2026'
    before_date = game_date_str  # PER calculator uses < before_date
    
    giannis_id = '3032977'
    other_id = '4397475'
    
    print("=" * 80)
    print("DEBUGGING MIL@DEN PREDICTION - 2026-01-11")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    
    # Step 1: Check what player_filters would be created
    print("\n1. Building player_filters (as matchup_predict.py does)...")
    home_roster = db.nba_rosters.find_one({'season': season, 'team': home_team})
    away_roster = db.nba_rosters.find_one({'season': season, 'team': away_team})
    
    if not away_roster:
        print(f"   ERROR: No roster found for {away_team} in {season}")
        return
    
    away_playing = set()
    for player in away_roster.get('roster', []):
        player_id = str(player.get('player_id', ''))
        if player_id:
            away_playing.add(player_id)
    
    player_filters = {
        away_team: {
            'playing': list(away_playing),
            'starters': [str(p.get('player_id', '')) for p in away_roster.get('roster', []) if p.get('starter', False)]
        }
    }
    
    print(f"   {away_team} roster: {len(away_roster.get('roster', []))} players")
    print(f"   {away_team} playing list: {len(player_filters[away_team]['playing'])} players")
    print(f"   Giannis in playing list: {giannis_id in player_filters[away_team]['playing']}")
    print(f"   {other_id} in playing list: {other_id in player_filters[away_team]['playing']}")
    print(f"   Playing list: {sorted(player_filters[away_team]['playing'])}")
    
    # Step 2: Check what PER calculator finds (with preload=False to match prediction flow)
    print("\n2. Checking PER calculator (preload=False, as in prediction flow)...")
    per_calc_no_preload = PERCalculator(db=db, preload=False)
    players_available = per_calc_no_preload._get_team_players_before_date_cached(
        away_team, season, before_date, min_games=1
    )
    
    print(f"   Available players found: {len(players_available)}")
    if players_available:
        available_ids = {str(p.get('_id', '')) for p in players_available}
        print(f"   Available IDs: {sorted(available_ids)}")
        print(f"   Giannis in available: {giannis_id in available_ids}")
        print(f"   {other_id} in available: {other_id in available_ids}")
        
        # Check Giannis specifically
        giannis_in_results = [p for p in players_available if str(p.get('_id', '')) == giannis_id]
        if giannis_in_results:
            print(f"   Giannis entry: {giannis_in_results[0]}")
        else:
            print(f"   ERROR: Giannis not found in available players!")
            # Check if it's a type issue
            giannis_int = [p for p in players_available if p.get('_id') == int(giannis_id)]
            if giannis_int:
                print(f"   BUT found as integer: {giannis_int[0].get('_id')}")
    
    # Step 3: Check what PER calculator finds (with preload=True for comparison)
    print("\n3. Checking PER calculator (preload=True, for comparison)...")
    per_calc_preload = PERCalculator(db=db, preload=True)
    players_available_preload = per_calc_preload._get_team_players_before_date_cached(
        away_team, season, before_date, min_games=1
    )
    
    print(f"   Available players (preload=True): {len(players_available_preload)}")
    if players_available_preload:
        available_ids_preload = {str(p.get('_id', '')) for p in players_available_preload}
        print(f"   Giannis in available (preload=True): {giannis_id in available_ids_preload}")
        print(f"   {other_id} in available (preload=True): {other_id in available_ids_preload}")
    
    # Step 4: Check stats_nba_players directly for these players before the date
    print("\n4. Checking stats_nba_players directly...")
    query = {
        'team': away_team,
        'season': season,
        'date': {'$lt': before_date},
        'stats.min': {'$gt': 0},
        'player_id': {'$in': [giannis_id, other_id]}
    }
    direct_stats = list(db.stats_nba_players.find(query))
    print(f"   Direct query found {len(direct_stats)} records for Giannis and {other_id}")
    
    for player_id in [giannis_id, other_id]:
        player_stats = [s for s in direct_stats if s.get('player_id') == player_id]
        print(f"   {player_id}: {len(player_stats)} games before {before_date}")
        if player_stats:
            print(f"     Sample dates: {[s.get('date') for s in player_stats[:3]]}")
            print(f"     Latest date: {max(s.get('date') for s in player_stats)}")
    
    # Step 5: Simulate the exact mismatch check from PER calculator
    print("\n5. Simulating exact PER calculator mismatch check...")
    if players_available:
        playing_list = player_filters[away_team]['playing']
        playing_set = set(playing_list)
        available_ids = {p['_id'] for p in players_available}
        
        print(f"   playing_set count: {len(playing_set)}")
        print(f"   available_ids count: {len(available_ids)}")
        print(f"   playing_set types: {[type(pid).__name__ for pid in list(playing_set)[:3]]}")
        print(f"   available_ids types: {[type(aid).__name__ for aid in list(available_ids)[:3]]}")
        
        # Original check (without normalization)
        unmatched_original = playing_set - available_ids
        print(f"   Unmatched (original, no normalization): {len(unmatched_original)}")
        if unmatched_original:
            print(f"     Unmatched IDs: {list(unmatched_original)}")
        
        # With normalization (my fix)
        available_ids_str = {str(p['_id']) for p in players_available}
        playing_set_str = {str(pid) for pid in playing_list}
        unmatched_normalized = playing_set_str - available_ids_str
        print(f"   Unmatched (with normalization): {len(unmatched_normalized)}")
        if unmatched_normalized:
            print(f"     Unmatched IDs: {list(unmatched_normalized)}")
        
        # Check if Giannis is in each set
        print(f"\n   Detailed check for Giannis ({giannis_id}):")
        print(f"     In playing_set: {giannis_id in playing_set}")
        print(f"     In available_ids: {giannis_id in available_ids}")
        print(f"     In playing_set_str: {giannis_id in playing_set_str}")
        print(f"     In available_ids_str: {giannis_id in available_ids_str}")
        
        # Check exact matches
        for aid in available_ids:
            if str(aid) == giannis_id or aid == giannis_id:
                print(f"     Found Giannis in available_ids as: {aid} (type: {type(aid).__name__})")
        for pid in playing_set:
            if str(pid) == giannis_id or pid == giannis_id:
                print(f"     Found Giannis in playing_set as: {pid} (type: {type(pid).__name__})")
    
    # Step 6: Check the DB aggregation directly (what _get_team_players_before_date_db returns)
    print("\n6. Checking DB aggregation directly...")
    pipeline = [
        {
            '$match': {
                'team': away_team,
                'season': season,
                'date': {'$lt': before_date},
                'stats.min': {'$gt': 0}
            }
        },
        {
            '$group': {
                '_id': '$player_id',
                'games': {'$sum': 1},
                'total_min': {'$sum': '$stats.min'}
            }
        },
        {'$match': {'games': {'$gte': 1}, 'total_min': {'$gt': 0}}},
        {'$sort': {'total_min': -1}}
    ]
    agg_results = list(db.stats_nba_players.aggregate(pipeline))
    print(f"   DB aggregation found: {len(agg_results)} players")
    
    if agg_results:
        agg_ids = {str(doc.get('_id', '')) for doc in agg_results}
        print(f"   Giannis in aggregation: {giannis_id in agg_ids}")
        print(f"   {other_id} in aggregation: {other_id in agg_ids}")
        
        giannis_agg = [doc for doc in agg_results if str(doc.get('_id', '')) == giannis_id]
        if giannis_agg:
            print(f"   Giannis aggregation result: {giannis_agg[0]}")
        else:
            print(f"   ERROR: Giannis not in aggregation results!")
            # Check all _id values
            all_agg_ids = [str(doc.get('_id', '')) for doc in agg_results]
            print(f"   All aggregation _id values: {sorted(all_agg_ids)}")
    
    # Step 7: Check if there's a date issue - maybe the date comparison is wrong
    print("\n7. Checking date comparison...")
    # Get all Giannis games for this season/team
    all_giannis_games = list(db.stats_nba_players.find({
        'player_id': giannis_id,
        'team': away_team,
        'season': season,
        'stats.min': {'$gt': 0}
    }, {'date': 1}).sort('date', -1))
    
    print(f"   Total Giannis games in season: {len(all_giannis_games)}")
    if all_giannis_games:
        print(f"   Latest game date: {all_giannis_games[0].get('date')}")
        print(f"   Earliest game date: {all_giannis_games[-1].get('date')}")
        
        # Check which games are before the target date
        games_before = [g for g in all_giannis_games if g.get('date', '') < before_date]
        print(f"   Games before {before_date}: {len(games_before)}")
        if games_before:
            print(f"     Latest before date: {games_before[0].get('date')}")
        else:
            print(f"     ERROR: No games before {before_date}!")
    
    print("\n" + "=" * 80)
    print("DEBUG COMPLETE")

if __name__ == '__main__':
    debug_mil_den_prediction()
