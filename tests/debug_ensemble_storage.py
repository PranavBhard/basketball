#!/usr/bin/env python3
"""Debug what's actually stored in ensemble base_models_summary."""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
bball_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(bball_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo

def main():
    mongo = Mongo()
    
    # Get selected ensemble config
    ensemble_config = mongo.db.model_config_nba.find_one({'selected': True})
    
    if not ensemble_config or not ensemble_config.get('ensemble', False):
        print("❌ No selected ensemble config found")
        return
    
    print("=== CURRENT ENSEMBLE BASE_MODELS_SUMMARY ===")
    base_models_summary = ensemble_config.get('ensemble_base_models_summary', [])
    
    print(f"Number of base models: {len(base_models_summary)}")
    
    for i, bm in enumerate(base_models_summary):
        print(f"\n--- Base Model {i+1} ---")
        print(f"Run ID: {bm.get('run_id', 'Unknown')}")
        print(f"Model Type: {bm.get('model_type', 'Unknown')}")
        
        metrics = bm.get('metrics', {})
        print(f"Metrics available: {bool(metrics)}")
        
        if metrics:
            print(f"Accuracy: {metrics.get('accuracy_mean', 'N/A')}")
            print(f"Log Loss: {metrics.get('log_loss_mean', 'N/A')}")
            print(f"Brier: {metrics.get('brier_mean', 'N/A')}")
            print(f"AUC: {metrics.get('auc_mean', 'N/A')}")
        else:
            print("❌ No metrics found!")
        
        config = bm.get('config', {})
        print(f"Config available: {bool(config)}")
        
        if config:
            feature_names = config.get('feature_names', [])
            print(f"Feature names count: {len(feature_names)}")
            print(f"First 3 features: {feature_names[:3]}")

if __name__ == "__main__":
    main()
