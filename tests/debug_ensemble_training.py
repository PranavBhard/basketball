#!/usr/bin/env python3
"""Script to debug ensemble training and see what features are actually passed."""

import sys
import os
sys.path.append(os.getcwd())

from bball_app.core.mongo import Mongo

def main():
    mongo = Mongo()
    db = mongo.db
    
    # Get selected ensemble config
    selected_config = db.model_config_nba.find_one({'selected': True})
    
    if not selected_config or not selected_config.get('ensemble', False):
        print("❌ No selected ensemble config found")
        return
    
    print("=== ENSEMBLE CONFIG DEBUG ===")
    print(f"Ensemble Models: {selected_config.get('ensemble_models', [])}")
    print(f"Ensemble Meta Features: {selected_config.get('ensemble_meta_features', [])}")
    
    # Check what the dataset builder would extract for meta_features
    meta_features = selected_config.get('ensemble_meta_features', [])
    
    # Simulate dataset builder logic for meta_target_cols
    meta_target_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 'home_points', 'away_points']
    
    print(f"\n=== DATASET BUILDER LOGIC ===")
    print(f"Meta target cols: {meta_target_cols}")
    print(f"Requested meta features: {meta_features}")
    
    # Check master training data
    import pandas as pd
    master_path = "../master_training/MASTER_TRAINING.csv"
    if os.path.exists(master_path):
        master_df = pd.read_csv(master_path, nrows=0)
        master_features = [c for c in master_df.columns if c not in meta_target_cols]
        master_features_set = set(master_features)
        
        print(f"\nMaster CSV has {len(master_df.columns)} total columns")
        print(f"Available features after excluding meta cols: {len(master_features)}")
        
        # Check if pred_margin is in available features
        if 'pred_margin' in master_features_set:
            print(f"✅ pred_margin is in available features")
        else:
            print(f"❌ pred_margin is NOT in available features")
        
        # Check what happens with meta_features
        available_meta = [f for f in meta_features if f in master_features_set]
        missing_meta = [f for f in meta_features if f not in master_features_set]
        
        print(f"Available meta features: {available_meta}")
        print(f"Missing meta features: {missing_meta}")
        
        # Simulate stacking tool exclusion
        pred_cols_exclude = ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total']
        final_meta = [f for f in available_meta if f not in pred_cols_exclude]
        
        print(f"After stacking tool exclusion: {final_meta}")
        print(f"❌ This is why pred_margin disappears!")

if __name__ == "__main__":
    main()
