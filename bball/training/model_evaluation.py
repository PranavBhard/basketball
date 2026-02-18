"""Model evaluation — re-exports from sportscore."""
from sportscore.training.model_evaluation import (
    evaluate_model_combo,
    evaluate_model_combo_with_calibration,
    compute_feature_importance,
)

__all__ = ['evaluate_model_combo', 'evaluate_model_combo_with_calibration', 'compute_feature_importance']
