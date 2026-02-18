"""
ML Models Module.

This module contains machine learning model implementations:
- BballModel: Main classifier model for game predictions (league-aware)
- PointsRegressionTrainer: Points prediction model
- ArtifactLoader: Factory for creating sklearn models
- EnsemblePredictor: Ensemble model handling

Import directly from submodules to avoid circular imports:
    from bball.models.bball_model import BballModel
    from bball.models.ensemble import EnsemblePredictor
"""


def __getattr__(name):
    """Lazy imports to avoid circular dependency with bball.services."""
    if name in ("BballModel", "NBAModel"):
        from bball.models.bball_model import BballModel
        if name == "NBAModel":
            return BballModel
        return BballModel
    if name == "PointsRegressionTrainer":
        from bball.models.points_regression import PointsRegressionTrainer
        return PointsRegressionTrainer
    if name in ("ArtifactLoader", "ModelFactory"):
        from bball.models.artifact_loader import ArtifactLoader
        if name == "ModelFactory":
            return ArtifactLoader
        return ArtifactLoader
    if name == "EnsemblePredictor":
        from bball.models.ensemble import EnsemblePredictor
        return EnsemblePredictor
    if name == "create_ensemble_predictor":
        from bball.models.ensemble import create_ensemble_predictor
        return create_ensemble_predictor
    raise AttributeError(f"module 'bball.models' has no attribute {name!r}")


__all__ = [
    'BballModel',
    'NBAModel',
    'PointsRegressionTrainer',
    'ArtifactLoader',
    'ModelFactory',
    'EnsemblePredictor',
    'create_ensemble_predictor',
]
