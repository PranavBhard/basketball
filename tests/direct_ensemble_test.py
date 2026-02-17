#!/usr/bin/env python3
"""Direct test of ensemble training with pred_margin fix."""

import sys
import os
sys.path.append(os.getcwd())

from bball_app.core.mongo import Mongo

def main():
    print("=== DIRECT ENSEMBLE TRAINING TEST ===")
    
    mongo = Mongo()
    selected_config = mongo.db.model_config_nba.find_one({'selected': True})
    
    if not selected_config or not selected_config.get('ensemble', False):
        print("❌ No selected ensemble config found")
        return
    
    print(f"Current ensemble: {selected_config.get('name')}")
    print(f"Meta features: {selected_config.get('ensemble_meta_features', [])}")
    
    # Import the train_ensemble_model function directly
    from web.app import train_ensemble_model
    
    # Prepare training config
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
    
    print(f"\nCalling train_ensemble_model directly...")
    print(f"Meta features requested: {training_config['ensemble_meta_features']}")
    
    try:
        # This will train the ensemble and update the database
        result = train_ensemble_model(training_config)
        
        if result and result.status_code == 200:
            response_data = result.get_json()
            job_id = response_data.get('job_id')
            print(f"✅ Ensemble training started! Job ID: {job_id}")
        else:
            print(f"❌ Training failed: {result}")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
