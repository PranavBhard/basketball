#!/usr/bin/env python3
"""
Test script to check if MongoDB aggregation converts string player_ids to integers.
"""

import sys
import os

# Add parent directory to path (the directory containing nba_app folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)  # This is the nba_app folder
parent_of_nba_app = os.path.dirname(nba_app_dir)  # Directory containing nba_app
if parent_of_nba_app not in sys.path:
    sys.path.insert(0, parent_of_nba_app)

from bball_app.core.mongo import Mongo

def test_mongodb_aggregation_types():
    """Test if MongoDB aggregation preserves string types for player_id."""
    
    mongo = Mongo()
    db = mongo.db
    
    print("Testing MongoDB aggregation type preservation...")
    print("=" * 80)
    
    # Test the exact aggregation pipeline used in PER calculator
    team = 'MIL'
    season = '2025-2026'
    before_date = '2026-01-11'
    
    pipeline = [
        {
            '$match': {
                'team': team,
                'season': season,
                'date': {'$lt': before_date},
                'stats.min': {'$gt': 0}
            }
        },
        {
            '$group': {
                '_id': '$player_id',
                'player_name': {'$first': '$player_name'},
                'games': {'$sum': 1}
            }
        },
        {'$limit': 10}
    ]
    
    print(f"\nRunning aggregation pipeline for {team} {season} before {before_date}...")
    results = list(db.stats_nba_players.aggregate(pipeline))
    
    print(f"   Found {len(results)} players")
    
    if results:
        print(f"\n   Checking _id types from aggregation:")
        _id_types = {}
        for doc in results:
            _id = doc.get('_id')
            _id_type = type(_id).__name__
            if _id_type not in _id_types:
                _id_types[_id_type] = []
            _id_types[_id_type].append(_id)
        
        for _id_type, examples in _id_types.items():
            print(f"     {_id_type}: {len(examples)} examples")
            print(f"       Sample values: {examples[:3]}")
        
        # Check specific player (Giannis)
        giannis_result = [doc for doc in results if str(doc.get('_id')) == '3032977' or doc.get('_id') == 3032977]
        if giannis_result:
            print(f"\n   Giannis (3032977) in results:")
            print(f"     _id: {giannis_result[0].get('_id')}")
            print(f"     _id type: {type(giannis_result[0].get('_id')).__name__}")
    
    # Also check what the original player_id types are in the source documents
    print(f"\n   Checking source document player_id types:")
    source_docs = list(db.stats_nba_players.find(
        {'team': team, 'season': season, 'date': {'$lt': before_date}, 'stats.min': {'$gt': 0}},
        {'player_id': 1}
    ).limit(10))
    
    if source_docs:
        source_types = Counter(type(doc.get('player_id')).__name__ for doc in source_docs)
        print(f"     Source player_id types: {dict(source_types)}")
        print(f"     Sample source player_ids: {[doc.get('player_id') for doc in source_docs[:3]]}")
    
    print("\n" + "=" * 80)
    print("Conclusion:")
    print("  - If aggregation _id types match source player_id types, MongoDB preserves types")
    print("  - If aggregation converts strings to ints, that would explain the mismatch")

if __name__ == '__main__':
    from collections import Counter
    test_mongodb_aggregation_types()
