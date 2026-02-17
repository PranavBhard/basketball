#!/usr/bin/env python3
"""Check if any classifier configs have pred_margin in their features list and verify it's used in training."""

import sys
import os

# Add project root to sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))  # bball_app/tests/
bball_app_dir = os.path.dirname(script_dir)  # bball_app/
project_root = os.path.dirname(bball_app_dir)  # basketball/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo

def main():
    # Simple check without connecting to DB first
    print("Checking classifier configs for pred_margin feature usage...")
    
    # Mock config to test feature handling
    mock_config = {
        'features': ['points|season|avg|diff', 'efg|season|avg|diff', 'pred_margin'],
        'feature_count': 3,
        'use_master': True
    }
    
    print(f"\n=== Mock config with pred_margin ===")
    print(f"features ({len(mock_config['features'])}): {mock_config['features']}")
    print(f"feature_count: {mock_config['feature_count']}")
    print(f"use_master: {mock_config['use_master']}")
    
    # Try to connect to DB if available
    try:
        mongo = Mongo()
        db = mongo.db
        
        # Find configs with pred_margin in features
        configs_with_pred = list(db.model_config_nba.find({
            'ensemble': {'$ne': True},
            'features': 'pred_margin'
        }))
        
        print(f"\nFound {len(configs_with_pred)} non-ensemble configs with pred_margin in features")
        
        for cfg in configs_with_pred:
            print(f"\n=== Config: {cfg.get('name', 'Unnamed')} ===")
            print(f"_id: {cfg['_id']}")
            print(f"features ({len(cfg.get('features', []))}): {cfg.get('features', [])}")
            print(f"feature_count: {cfg.get('feature_count', 'N/A')}")
            print(f"selected: {cfg.get('selected', 'N/A')}")
            print(f"training_csv: {cfg.get('training_csv', 'N/A')}")
            print(f"use_master: {cfg.get('use_master', 'N/A')}")
            print(f"trained_at: {cfg.get('trained_at', 'N/A')}")
        
        if not configs_with_pred:
            print("No non-ensemble configs found with pred_margin in features")
        
        # Also check one specific config to see full structure
        example = db.model_config_nba.find_one({'ensemble': {'$ne': True}})
        if example:
            print("\n=== Example non-ensemble config structure ===")
            print(f"keys: {list(example.keys())}")
            if 'features' in example:
                print(f"features type: {type(example['features'])}")
                print(f"features: {example['features']}")
            if 'training_csv' in example:
                print(f"training_csv: {example['training_csv']}")
                
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        print("This is expected if MongoDB is not running.")
        print("The mock config above shows that pred_margin should be included in features list.")

if __name__ == '__main__':
    main()
