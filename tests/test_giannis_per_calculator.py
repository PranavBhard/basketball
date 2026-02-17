#!/usr/bin/env python3
"""
Test script to debug why Giannis (3032977) and another player (4397475) are not found in PER calculator's available players.
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

def test_giannis_per_calculator():
    """Test why Giannis and another player are not found in PER calculator."""
    
    # Test parameters
    team = 'MIL'
    season = '2025-2026'
    before_date = '2026-01-11'
    giannis_id = '3032977'
    other_id = '4397475'  # The other unmatched ID from the warning
    
    print(f"Testing PER calculator for MIL players")
    print(f"Team: {team}, Season: {season}, Before Date: {before_date}")
    print(f"Unmatched IDs from warning: {giannis_id} (Giannis), {other_id}")
    print("=" * 80)
    
    # Initialize MongoDB and PER calculator
    mongo = Mongo()
    db = mongo.db
    per_calc = PERCalculator(db=db, preload=True)
    
    # Step 1: Check rosters
    print("\n1. Checking nba_rosters...")
    roster_doc = db.nba_rosters.find_one({'season': season, 'team': team})
    if roster_doc:
        roster = roster_doc.get('roster', [])
        print(f"   Roster has {len(roster)} players")
        
        # Check both players
        for player_id in [giannis_id, other_id]:
            player_in_roster = [p for p in roster if str(p.get('player_id', '')) == player_id]
            if player_in_roster:
                print(f"   {player_id} in roster: True (starter: {player_in_roster[0].get('starter', False)})")
            else:
                print(f"   {player_id} in roster: False")
    else:
        print(f"   ERROR: No roster found for {team} in {season}")
        return
    
    # Step 2: Check stats_nba_players
    print("\n2. Checking stats_nba_players...")
    for player_id in [giannis_id, other_id]:
        stats_query = {
            'player_id': player_id,
            'team': team,
            'season': season,
            'date': {'$lt': before_date},
            'stats.min': {'$gt': 0}
        }
        stats = list(db.stats_nba_players.find(stats_query).limit(1))
        print(f"   {player_id} stats before {before_date}: {len(stats)} games")
        if stats:
            print(f"     Sample: date={stats[0].get('date')}, player_id type={type(stats[0].get('player_id'))}")
    
    # Step 3: Check PER calculator results
    print("\n3. Checking PER calculator's _get_team_players_before_date_cached...")
    players = per_calc._get_team_players_before_date_cached(team, season, before_date, min_games=1)
    print(f"   PER calculator found {len(players)} players")
    
    # Check both players
    for player_id in [giannis_id, other_id]:
        player_in_results = [p for p in players if str(p.get('_id', '')) == player_id]
        if player_in_results:
            print(f"   {player_id} in results: True")
            print(f"     _id: {player_in_results[0].get('_id')} (type: {type(player_in_results[0].get('_id'))})")
        else:
            print(f"   {player_id} in results: False")
            # Check if it's a type mismatch
            int_id = int(player_id)
            player_in_results_int = [p for p in players if p.get('_id') == int_id]
            if player_in_results_int:
                print(f"     BUT found as integer {int_id}!")
    
    # Step 4: Build player_filters exactly as matchup_predict.py does
    print("\n4. Building player_filters (as in matchup_predict.py)...")
    if roster_doc:
        roster = roster_doc.get('roster', [])
        playing_list = []
        for player in roster:
            player_id = str(player.get('player_id', ''))
            if player_id:
                playing_list.append(player_id)
        
        player_filters = {
            team: {
                'playing': playing_list,
                'starters': [str(p.get('player_id', '')) for p in roster if p.get('starter', False)]
            }
        }
        
        print(f"   player_filters['playing'] count: {len(player_filters[team]['playing'])}")
        print(f"   {giannis_id} in playing: {giannis_id in player_filters[team]['playing']}")
        print(f"   {other_id} in playing: {other_id in player_filters[team]['playing']}")
    
    # Step 5: Simulate the exact mismatch check from PER calculator
    print("\n5. Simulating PER calculator's mismatch check (line 1140-1141)...")
    if players:
        available_ids = {p['_id'] for p in players}
        if roster_doc:
            playing_set = set(player_filters[team]['playing'])
            unmatched = playing_set - available_ids
            
            print(f"   available_ids count: {len(available_ids)}")
            print(f"   playing_set count: {len(playing_set)}")
            print(f"   unmatched count: {len(unmatched)}")
            if unmatched:
                print(f"   unmatched IDs: {list(unmatched)}")
                for unmatched_id in unmatched:
                    print(f"     Checking {unmatched_id}:")
                    # Check if it's a type issue
                    if unmatched_id in available_ids:
                        print(f"       Found in available_ids as string!")
                    elif str(unmatched_id) in available_ids:
                        print(f"       Found in available_ids when converted to string!")
                    elif int(unmatched_id) in available_ids:
                        print(f"       Found in available_ids when converted to int!")
                    else:
                        # Check if player has stats
                        stats_check = list(db.stats_nba_players.find({
                            'player_id': unmatched_id,
                            'team': team,
                            'season': season,
                            'date': {'$lt': before_date},
                            'stats.min': {'$gt': 0}
                        }).limit(1))
                        print(f"       Has stats before date: {len(stats_check) > 0}")
                        if not stats_check:
                            # Check if they have ANY stats
                            any_stats = list(db.stats_nba_players.find({
                                'player_id': unmatched_id,
                                'team': team,
                                'season': season
                            }).limit(1))
                            print(f"       Has ANY stats in season: {len(any_stats) > 0}")
                            if any_stats:
                                print(f"       Sample date: {any_stats[0].get('date')}")
    
    # Step 6: Check what player 4397475 is
    print("\n6. Checking player 4397475 details...")
    player_info = db.players_nba.find_one({'player_id': other_id})
    if player_info:
        print(f"   Player name: {player_info.get('player_name', 'Unknown')}")
        print(f"   Player ID: {player_info.get('player_id')} (type: {type(player_info.get('player_id'))})")
    else:
        print(f"   Player not found in players_nba")
    
    # Check if they're in the roster
    if roster_doc:
        roster = roster_doc.get('roster', [])
        other_in_roster = [p for p in roster if str(p.get('player_id', '')) == other_id]
        if other_in_roster:
            print(f"   In roster: True")
        else:
            print(f"   In roster: False")
    
    print("\n" + "=" * 80)
    print("Test complete!")

if __name__ == '__main__':
    test_giannis_per_calculator()
