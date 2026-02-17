#!/usr/bin/env python3
"""Check if run tracker has base model runs."""

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
    
    # Get current ensemble config
    ensemble_config = mongo.db.model_config_nba.find_one({'selected': True})
    
    if not ensemble_config or not ensemble_config.get('ensemble', False):
        print("❌ No selected ensemble config found")
        return
    
    ensemble_models = ensemble_config.get('ensemble_models', [])
    print(f"=== CHECKING BASE MODEL RUNS ===")
    print(f"Ensemble models: {ensemble_models}")
    
    # Check if run tracker has these runs
    from agents.tools.run_tracker import RunTracker
    tracker = RunTracker(db=mongo.db)
    
    for i, model_id in enumerate(ensemble_models):
        run = tracker.get_run(model_id)
        print(f"Base model {i+1} ({model_id}):")
        print(f"  Run exists: {run is not None}")
        if run:
            print(f"  Run ID: {run.get('run_id', 'Unknown')}")
            print(f"  Has metrics: {'metrics' in run and bool(run['metrics'])}")
            print(f"  Has config: {'config' in run and bool(run['config'])}")
        else:
            print("  ❌ Run not found in tracker!")

if __name__ == "__main__":
    main()
