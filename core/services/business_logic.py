"""
Unified Business Logic for model training, prediction, and validation.
Centralizes all core algorithms and workflows.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime


class ModelBusinessLogic:
    """Centralized business logic for model operations."""
    
    @staticmethod
    def train_model(config: Dict, db) -> Dict:
        """
        Unified training pipeline for all model types.
        
        Args:
            config: Model configuration dictionary
            db: MongoDB database instance
            
        Returns:
            Training result dictionary
        """
        try:
            print(f"🚀 Starting training for {config.get('model_type', 'Unknown')}")
            
            # Validate configuration
            validation = ModelBusinessLogic.validate_config(config)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': f"Invalid configuration: {'; '.join(validation['errors'])}"
                }
            
            # Create model using unified factory
            from ..models.artifact_loader import ArtifactLoader
            model, scaler, feature_names = ArtifactLoader.create_model(config, use_artifacts=False)
            
            # Save artifacts for future fast loading
            run_id = config.get('run_id')
            if run_id:
                ArtifactLoader.save_model_artifacts(model, scaler, feature_names, config, run_id)
            
            # Calculate training metrics
            training_metrics = ModelBusinessLogic._calculate_training_metrics(model, scaler, feature_names)
            
            result = {
                'success': True,
                'model': model,
                'scaler': scaler,
                'feature_names': feature_names,
                'metrics': training_metrics,
                'config': config,
                'trained_at': datetime.utcnow()
            }
            
            print(f"✅ Training completed for {config.get('model_type')}")
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Training failed: {str(e)}"
            }
    
    @staticmethod
    def predict(model: Any, scaler: Any, feature_names: List[str], 
               data: pd.DataFrame, config: Dict = None) -> Dict:
        """
        Unified prediction pipeline.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_names: List of feature names
            data: Input data for prediction
            config: Optional configuration for prediction settings
            
        Returns:
            Prediction result dictionary
        """
        try:
            print(f"🔮 Making predictions for {len(data)} samples")
            
            # Validate inputs
            if model is None:
                return {'success': False, 'error': 'No model provided'}
            
            if data.empty:
                return {'success': False, 'error': 'No data provided for prediction'}
            
            # Prepare features
            meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'home_points', 'away_points']
            available_features = [col for col in data.columns if col not in meta_cols]
            
            # Check for missing features
            missing_features = set(feature_names) - set(available_features)
            if missing_features:
                return {
                    'success': False,
                    'error': f"Missing features: {', '.join(missing_features)}"
                }
            
            # Extract and prepare features
            X = data[feature_names].values
            
            # Handle NaN values
            nan_mask = np.isnan(X).any(axis=1)
            if nan_mask.sum() > 0:
                print(f"⚠️  Found {nan_mask.sum()} rows with NaN values, excluding from prediction")
                X = X[~nan_mask]
                prediction_indices = np.where(~nan_mask)[0]
            else:
                X = X
                prediction_indices = np.arange(len(data))
            
            # Scale features
            X_scaled = scaler.transform(X)
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                predictions = model.predict(X_scaled)
                
                result = {
                    'success': True,
                    'predictions': predictions.tolist(),
                    'probabilities': probabilities.tolist(),
                    'prediction_indices': prediction_indices.tolist(),
                    'feature_names': feature_names,
                    'n_samples': len(X),
                    'has_probabilities': True
                }
            else:
                predictions = model.predict(X_scaled)
                result = {
                    'success': True,
                    'predictions': predictions.tolist(),
                    'prediction_indices': prediction_indices.tolist(),
                    'feature_names': feature_names,
                    'n_samples': len(X),
                    'has_probabilities': False
                }
            
            # Add metadata
            result['predicted_at'] = datetime.utcnow()
            result['config'] = config or {}
            
            print(f"✅ Generated {len(result['predictions'])} predictions")
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Prediction failed: {str(e)}"
            }
    
    @staticmethod
    def validate_config(config: Dict) -> Dict:
        """
        Validate model configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Validation result dictionary
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Required fields
        required_fields = ['model_type']
        for field in required_fields:
            if field not in config or config[field] is None:
                validation['errors'].append(f"Missing required field: {field}")
                validation['valid'] = False
        
        # Validate model type
        model_type = config.get('model_type')
        if model_type:
            valid_types = ['LogisticRegression', 'SVM', 'GradientBoosting']
            if model_type not in valid_types:
                validation['errors'].append(f"Invalid model type: {model_type}")
                validation['valid'] = False
        
        # Validate features
        features = config.get('features', [])
        if features:
            from .feature_manager import FeatureManager
            feature_validation = FeatureManager.validate_features(features)
            if not feature_validation['valid']:
                validation['errors'].extend(feature_validation['errors'])
                validation['valid'] = False
            validation['warnings'].extend(feature_validation['warnings'])
        
        # Validate time calibration
        if config.get('use_time_calibration', False):
            time_fields = ['begin_year', 'calibration_years', 'evaluation_year']
            for field in time_fields:
                if field not in config or config[field] is None:
                    validation['errors'].append(f"Time calibration enabled but missing {field}")
                    validation['valid'] = False
        
        # Validate ensemble configuration
        if config.get('ensemble', False):
            ensemble_fields = ['ensemble_models', 'ensemble_type']
            for field in ensemble_fields:
                if field not in config or not config[field]:
                    validation['errors'].append(f"Ensemble enabled but missing {field}")
                    validation['valid'] = False
        
        return validation
    
    @staticmethod
    def _calculate_training_metrics(model: Any, scaler: Any, feature_names: List[str]) -> Dict:
        """
        Calculate training metrics for a trained model.
        
        Args:
            model: Trained model
            scaler: Fitted scaler
            feature_names: List of feature names
            
        Returns:
            Metrics dictionary
        """
        metrics = {
            'model_type': type(model).__name__,
            'n_features': len(feature_names),
            'feature_names': feature_names,
            'trained_at': datetime.utcnow()
        }
        
        # Add model-specific metrics
        if hasattr(model, 'C'):
            metrics['c_value'] = model.C
        if hasattr(model, 'n_estimators'):
            metrics['n_estimators'] = model.n_estimators
        if hasattr(model, 'feature_importances_'):
            metrics['has_feature_importance'] = True
            # Get top features by importance
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            metrics['top_features'] = feature_importance[:10]
        
        # Scaler metrics
        if hasattr(scaler, 'mean_'):
            metrics['scaler_mean'] = scaler.mean_.tolist()
            metrics['scaler_scale'] = scaler.scale_.tolist()
        
        return metrics
    
    @staticmethod
    def create_ensemble_predictions(base_predictions: List[Dict], meta_features: List[str] = None,
                               use_disagree: bool = False, use_conf: bool = False) -> pd.DataFrame:
        """
        Create ensemble predictions from base model predictions.
        
        Args:
            base_predictions: List of prediction dictionaries from base models
            meta_features: List of meta-features to generate
            use_disagree: Whether to include disagreement features
            use_conf: Whether to include confidence features
            
        Returns:
            DataFrame with ensemble features
        """
        if not base_predictions:
            raise ValueError("No base predictions provided for ensemble")
        
        # Combine base predictions
        ensemble_df = pd.DataFrame()
        
        # Add base model predictions
        for i, pred_dict in enumerate(base_predictions):
            if pred_dict['success']:
                pred_col = f'base_pred_{i}'
                ensemble_df[pred_col] = pred_dict['predictions']
                
                # Add probabilities if available
                if pred_dict.get('has_probabilities'):
                    prob_col = f'base_prob_{i}'
                    ensemble_df[prob_col] = [p[1] for p in pred_dict['probabilities']]
        
        # Add disagreement features
        if use_disagree and len(base_predictions) >= 2:
            pred_cols = [f'base_pred_{i}' for i in range(len(base_predictions))]
            ensemble_df['pred_disagree'] = ensemble_df[pred_cols].std(axis=1)
            ensemble_df['pred_range'] = ensemble_df[pred_cols].max(axis=1) - ensemble_df[pred_cols].min(axis=1)
        
        # Add confidence features
        if use_conf and len(base_predictions) >= 2:
            prob_cols = [f'base_prob_{i}' for i in range(len(base_predictions)) 
                       if f'base_prob_{i}' in ensemble_df.columns]
            if prob_cols:
                ensemble_df['prob_mean'] = ensemble_df[prob_cols].mean(axis=1)
                ensemble_df['prob_std'] = ensemble_df[prob_cols].std(axis=1)
                ensemble_df['prob_max'] = ensemble_df[prob_cols].max(axis=1)
        
        # Add additional meta-features
        if meta_features:
            for feature in meta_features:
                if feature in ensemble_df.columns:
                    ensemble_df[f'meta_{feature}'] = ensemble_df[feature]
        
        return ensemble_df
    
    @staticmethod
    def evaluate_predictions(predictions: np.ndarray, actual: np.ndarray, 
                        probabilities: np.ndarray = None) -> Dict:
        """
        Evaluate prediction quality.
        
        Args:
            predictions: Predicted values
            actual: Actual values
            probabilities: Prediction probabilities (optional)
            
        Returns:
            Evaluation metrics dictionary
        """
        from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
        
        metrics = {
            'n_samples': len(predictions),
            'evaluated_at': datetime.utcnow()
        }
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(actual, predictions)
        
        # Probability-based metrics
        if probabilities is not None:
            try:
                metrics['log_loss'] = log_loss(actual, probabilities)
                metrics['brier_score'] = brier_score_loss(actual, probabilities)
            except Exception as e:
                print(f"⚠️  Could not calculate probability metrics: {e}")
        
        # Additional metrics
        metrics['error_rate'] = 1.0 - metrics['accuracy']
        
        return metrics
