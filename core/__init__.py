"""
Core infrastructure for unified model management across web app, CLI, and agent tooling.
"""

from .services.config_manager import ModelConfigManager
from .models.artifact_loader import ArtifactLoader
from .features.manager import FeatureManager
from .services.business_logic import ModelBusinessLogic
from .services.artifacts import ArtifactManager
from .features.registry import FeatureRegistry, FeatureGroups, StatCategory
from .features.generator import SharedFeatureGenerator, collect_unique_features
from .models.ensemble import EnsemblePredictor, create_ensemble_predictor
from .services.prediction import PredictionService, PredictionResult, MatchupInfo

__all__ = [
    'ModelConfigManager',
    'ArtifactLoader',
    'FeatureManager',
    'ModelBusinessLogic',
    'ArtifactManager',
    'FeatureRegistry',
    'FeatureGroups',
    'StatCategory',
    'SharedFeatureGenerator',
    'collect_unique_features',
    'EnsemblePredictor',
    'create_ensemble_predictor',
    'PredictionService',
    'PredictionResult',
    'MatchupInfo',
]
