#!/usr/bin/env python3
"""Script to retrain ensemble and verify pred_margin is included."""

import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
bball_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(bball_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import requests
import json

from bball_app.core.mongo import Mongo

def main():
    print("=== RETRAINING ENSEMBLE TO TEST PRED_MARGIN FIX ===")

    mongo = Mongo()
    selected_config = mongo.db.model_config_nba.find_one({'selected': True})
    
    if not selected_config or not selected_config.get('ensemble', False):
        print("❌ No selected ensemble config found")
        return
    
    print(f"Current ensemble: {selected_config.get('name')}")
    print(f"Meta features: {selected_config.get('ensemble_meta_features', [])}")
    
    # Prepare training request (same as current config)
    training_config = {
        'ensemble': True,
        'ensemble_models': selected_config.get('ensemble_models', []),
        'ensemble_meta_features': selected_config.get('ensemble_meta_features', []),
        'ensemble_use_disagree': selected_config.get('ensemble_use_disagree', False),
        'ensemble_use_conf': selected_config.get('ensemble_use_conf', False),
        'model_types': [selected_config.get('model_type', 'LogisticRegression')],
        'c_values': [0.1],
        'use_master': True,
        'use_time_calibration': True,
        'begin_year': selected_config.get('begin_year'),
        'calibration_years': selected_config.get('calibration_years'),
        'evaluation_year': selected_config.get('evaluation_year'),
        'min_games_played': selected_config.get('min_games_played', 15)
    }
    
    print(f"\nTraining request config:")
    print(f"- Ensemble models: {len(training_config['ensemble_models'])}")
    print(f"- Meta features: {training_config['ensemble_meta_features']}")
    print(f"- Use disagree: {training_config['ensemble_use_disagree']}")
    print(f"- Use conf: {training_config['ensemble_use_conf']}")
    
    # Make training request
    try:
        response = requests.post(
            'http://localhost:5000/api/model-config/train',
            json=training_config,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            print(f"\n✅ Ensemble training started successfully!")
            print(f"Job ID: {job_id}")
            print(f"\nTo check progress, visit: http://localhost:5000/model-config")
            print(f"After training completes, check if pred_margin appears in features_ranked")
        else:
            print(f"❌ Training request failed: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to web app. Make sure it's running on localhost:5000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
