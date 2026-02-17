#!/usr/bin/env python3
"""
Debug script to trace the exact filtering flow in PER calculator
to understand why players are being filtered out.
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

def trace_per_calculator_filtering():
    """Trace the exact filtering flow step by step."""
    
    # Exact parameters
    team = 'MIL'
    season = '2025-2026'
    before_date = '2026-01-11'
    giannis_id = '3032977'
    other_id = '4397475'
    
    print("=" * 80)
    print("TRACING PER CALCULATOR FILTERING FLOW")
    print("=" * 80)
    
    mongo = Mongo()
    db = mongo.db
    
    # Build player_filters exactly as matchup_predict.py does
    roster_doc = db.nba_rosters.find_one({'season': season, 'team': team})
    if not roster_doc:
        print(f"ERROR: No roster found for {team} in {season}")
        return
    
    roster = roster_doc.get('roster', [])
    playing_list = [str(p.get('player_id', '')) for p in roster if p.get('player_id')]
    player_filters = {
        team: {
            'playing': playing_list,
            'starters': [str(p.get('player_id', '')) for p in roster if p.get('starter', False)]
        }
    }
    
    print(f"\n1. Initial state:")
    print(f"   Roster players: {len(roster)}")
    print(f"   Playing list: {len(playing_list)}")
    print(f"   Giannis in playing_list: {giannis_id in playing_list}")
    print(f"   {other_id} in playing_list: {other_id in playing_list}")
    
    # Initialize PER calculator (preload=False to match prediction flow)
    per_calc = PERCalculator(db=db, preload=False)
    
    # Step 2: Get initial players (before any filtering)
    print(f"\n2. Getting initial players from _get_team_players_before_date_cached...")
    initial_players = per_calc._get_team_players_before_date_cached(team, season, before_date, min_games=1)
    print(f"   Initial players found: {len(initial_players)}")
    
    if initial_players:
        initial_ids = {str(p.get('_id', '')) for p in initial_players}
        print(f"   Giannis in initial: {giannis_id in initial_ids}")
        print(f"   {other_id} in initial: {other_id in initial_ids}")
        
        giannis_initial = [p for p in initial_players if str(p.get('_id', '')) == giannis_id]
        if giannis_initial:
            print(f"   Giannis initial entry: games={giannis_initial[0].get('games')}, total_min={giannis_initial[0].get('total_min')}")
    
    # Step 3: Simulate the filtering that happens in compute_team_per_features
    print(f"\n3. Simulating filtering in compute_team_per_features...")
    
    # This is what happens in compute_team_per_features when player_filters is provided
    players = initial_players.copy() if initial_players else []
    print(f"   Starting with {len(players)} players")
    
    # Phase 2.1: Filter by roster (line 1112)
    if player_filters and players:
        player_ids = [p['_id'] for p in players]
        roster_player_ids = set()
        
        if player_ids:
            roster_doc_check = db.nba_rosters.find_one({'season': season, 'team': team})
            if roster_doc_check:
                roster_check = roster_doc_check.get('roster', [])
                for roster_entry in roster_check:
                    roster_player_id = str(roster_entry.get('player_id', ''))
                    roster_player_ids.add(roster_player_id)
        
        print(f"   Roster player_ids: {len(roster_player_ids)}")
        print(f"   Giannis in roster_player_ids: {giannis_id in roster_player_ids}")
        print(f"   {other_id} in roster_player_ids: {other_id in roster_player_ids}")
        
        # Filter to only players in roster
        players_before_roster_filter = len(players)
        players = [p for p in players if str(p['_id']) in roster_player_ids]
        players_after_roster_filter = len(players)
        print(f"   After roster filter: {players_before_roster_filter} -> {players_after_roster_filter}")
        
        if players_after_roster_filter < players_before_roster_filter:
            removed = players_before_roster_filter - players_after_roster_filter
            print(f"   WARNING: {removed} players removed by roster filter!")
            # Check which players were removed
            initial_ids_set = {str(p.get('_id', '')) for p in initial_players}
            removed_ids = initial_ids_set - {str(p.get('_id', '')) for p in players}
            if removed_ids:
                print(f"   Removed player IDs: {list(removed_ids)}")
                if giannis_id in removed_ids:
                    print(f"   ERROR: Giannis was removed by roster filter!")
                if other_id in removed_ids:
                    print(f"   ERROR: {other_id} was removed by roster filter!")
    
    # Step 4: Check the mismatch (line 1140-1143)
    print(f"\n4. Checking mismatch (as in line 1140-1143)...")
    if players and player_filters:
        playing_list_check = player_filters[team].get('playing', [])
        if playing_list_check:
            available_ids = {str(p['_id']) for p in players}
            playing_set_str = {str(pid) for pid in playing_list_check}
            unmatched = playing_set_str - available_ids
            
            print(f"   available_ids count: {len(available_ids)}")
            print(f"   playing_set_str count: {len(playing_set_str)}")
            print(f"   unmatched count: {len(unmatched)}")
            
            if unmatched:
                print(f"   ERROR: Unmatched IDs: {list(unmatched)}")
                if giannis_id in unmatched:
                    print(f"   ERROR: Giannis is unmatched!")
                    print(f"     Giannis in available_ids: {giannis_id in available_ids}")
                    print(f"     Giannis in playing_set_str: {giannis_id in playing_set_str}")
                if other_id in unmatched:
                    print(f"   ERROR: {other_id} is unmatched!")
            else:
                print(f"   ✓ No unmatched players")
    
    # Step 5: Check if there's an issue with the roster filter logic
    print(f"\n5. Checking roster filter logic...")
    if initial_players:
        # Check if any initial players are NOT in the roster
        initial_ids_set = {str(p.get('_id', '')) for p in initial_players}
        roster_ids_set = {str(p.get('player_id', '')) for p in roster}
        
        not_in_roster = initial_ids_set - roster_ids_set
        not_in_initial = roster_ids_set - initial_ids_set
        
        print(f"   Players in initial but NOT in roster: {len(not_in_roster)}")
        if not_in_roster:
            print(f"     IDs: {list(not_in_roster)}")
        
        print(f"   Players in roster but NOT in initial: {len(not_in_initial)}")
        if not_in_initial:
            print(f"     IDs: {list(not_in_initial)}")
            if giannis_id in not_in_initial:
                print(f"     ERROR: Giannis is in roster but NOT in initial players!")
            if other_id in not_in_initial:
                print(f"     ERROR: {other_id} is in roster but NOT in initial players!")
    
    print("\n" + "=" * 80)

if __name__ == '__main__':
    trace_per_calculator_filtering()
