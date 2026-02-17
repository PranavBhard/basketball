#!/usr/bin/env python3
"""
Test script to verify how player_id is stored (int vs string) in relevant MongoDB collections.
"""

import sys
import os
from collections import Counter

# Add parent directory to path (the directory containing nba_app folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)  # This is the nba_app folder
parent_of_nba_app = os.path.dirname(nba_app_dir)  # Directory containing nba_app
if parent_of_nba_app not in sys.path:
    sys.path.insert(0, parent_of_nba_app)

from bball_app.core.mongo import Mongo

def check_player_id_types():
    """Check how player_id is stored in relevant collections."""
    
    mongo = Mongo()
    db = mongo.db
    
    print("Checking player_id types in relevant MongoDB collections...")
    print("=" * 80)
    
    # Collection 1: stats_nba_players (used by PER calculator)
    print("\n1. stats_nba_players collection:")
    print("   (Used by PER calculator to find available players)")
    sample_stats = list(db.stats_nba_players.find({}, {'player_id': 1}).limit(100))
    if sample_stats:
        player_id_types = Counter(type(doc.get('player_id')).__name__ for doc in sample_stats if 'player_id' in doc)
        print(f"   Sample size: {len(sample_stats)} documents")
        print(f"   player_id types: {dict(player_id_types)}")
        
        # Show examples
        str_examples = [doc.get('player_id') for doc in sample_stats if isinstance(doc.get('player_id'), str)][:3]
        int_examples = [doc.get('player_id') for doc in sample_stats if isinstance(doc.get('player_id'), int)][:3]
        if str_examples:
            print(f"   String examples: {str_examples}")
        if int_examples:
            print(f"   Integer examples: {int_examples}")
        
        # Check specific player (Giannis)
        giannis_stats = list(db.stats_nba_players.find(
            {'player_id': {'$in': ['3032977', 3032977]}},
            {'player_id': 1}
        ).limit(5))
        if giannis_stats:
            print(f"   Giannis (3032977) found: {len(giannis_stats)} records")
            print(f"   Giannis player_id type: {type(giannis_stats[0].get('player_id')).__name__}")
            print(f"   Giannis player_id value: {giannis_stats[0].get('player_id')}")
    else:
        print("   No documents found")
    
    # Collection 2: nba_rosters (used to build player_filters)
    print("\n2. nba_rosters collection:")
    print("   (Used to build player_filters in matchup_predict.py)")
    sample_rosters = list(db.nba_rosters.find({}, {'roster.player_id': 1}).limit(10))
    if sample_rosters:
        player_id_types = Counter()
        for roster_doc in sample_rosters:
            roster = roster_doc.get('roster', [])
            for player in roster:
                if 'player_id' in player:
                    player_id_types[type(player['player_id']).__name__] += 1
        
        print(f"   Sample size: {len(sample_rosters)} roster documents")
        print(f"   player_id types in roster entries: {dict(player_id_types)}")
        
        # Show examples
        for roster_doc in sample_rosters[:3]:
            roster = roster_doc.get('roster', [])
            if roster:
                sample_player = roster[0]
                if 'player_id' in sample_player:
                    print(f"   Example roster entry player_id: {sample_player['player_id']} (type: {type(sample_player['player_id']).__name__})")
                    break
        
        # Check specific team/season (MIL 2025-2026)
        mil_roster = db.nba_rosters.find_one({'season': '2025-2026', 'team': 'MIL'})
        if mil_roster:
            roster = mil_roster.get('roster', [])
            print(f"   MIL 2025-2026 roster: {len(roster)} players")
            if roster:
                giannis_in_roster = [p for p in roster if str(p.get('player_id', '')) == '3032977']
                if giannis_in_roster:
                    print(f"   Giannis in roster: player_id={giannis_in_roster[0].get('player_id')} (type: {type(giannis_in_roster[0].get('player_id')).__name__})")
    else:
        print("   No roster documents found")
    
    # Collection 3: players_nba (player metadata)
    print("\n3. players_nba collection:")
    print("   (Player metadata)")
    sample_players = list(db.players_nba.find({}, {'player_id': 1}).limit(100))
    if sample_players:
        player_id_types = Counter(type(doc.get('player_id')).__name__ for doc in sample_players if 'player_id' in doc)
        print(f"   Sample size: {len(sample_players)} documents")
        print(f"   player_id types: {dict(player_id_types)}")
        
        # Check specific player (Giannis)
        giannis_player = db.players_nba.find_one({'player_id': {'$in': ['3032977', 3032977]}})
        if giannis_player:
            print(f"   Giannis found: player_id={giannis_player.get('player_id')} (type: {type(giannis_player.get('player_id')).__name__})")
    else:
        print("   No documents found")
    
    # Check aggregation result format
    print("\n4. MongoDB aggregation result format:")
    print("   (What _id looks like after $group by player_id)")
    pipeline = [
        {
            '$match': {
                'team': 'MIL',
                'season': '2025-2026',
                'stats.min': {'$gt': 0}
            }
        },
        {
            '$group': {
                '_id': '$player_id',
                'count': {'$sum': 1}
            }
        },
        {'$limit': 5}
    ]
    agg_results = list(db.stats_nba_players.aggregate(pipeline))
    if agg_results:
        print(f"   Sample size: {len(agg_results)} aggregated results")
        _id_types = Counter(type(doc.get('_id')).__name__ for doc in agg_results)
        print(f"   _id types (from $group by player_id): {dict(_id_types)}")
        print(f"   Example _id values: {[doc.get('_id') for doc in agg_results[:3]]}")
        print(f"   Example _id types: {[type(doc.get('_id')).__name__ for doc in agg_results[:3]]}")
        
        # Check if Giannis is in results
        giannis_agg = [doc for doc in agg_results if str(doc.get('_id')) == '3032977' or doc.get('_id') == 3032977]
        if giannis_agg:
            print(f"   Giannis in aggregation: _id={giannis_agg[0].get('_id')} (type: {type(giannis_agg[0].get('_id')).__name__})")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("  - If all player_id types are 'str', then the fix should work correctly")
    print("  - If there are mixed types (str and int), the normalization fix is necessary")
    print("  - If all are 'int', we may need to adjust the fix")

if __name__ == '__main__':
    check_player_id_types()
