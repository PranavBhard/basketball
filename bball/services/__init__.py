"""
Business Services Module.

This module contains high-level service orchestration:
- PredictionService: Single source of truth for predictions
- ModelConfigManager: Model configuration management
- ModelBusinessLogic: Business logic utilities
- ArtifactManager: Model artifact management
- TrainingDataService: Master training data generation and management
- WebpageParser: Extract clean text from HTML/URLs
- LineupService: Live game lineup data from ESPN
- NewsService: News/content fetcher from configured sources
- RosterService: Build team rosters from player game stats

Import directly from submodules to avoid circular imports:
    from bball.services.prediction import PredictionService
    from bball.services.config_manager import ModelConfigManager
"""


def __getattr__(name):
    """Lazy imports to avoid circular dependency with bball.models."""
    _prediction = {"PredictionService", "PredictionResult", "MatchupInfo"}
    if name in _prediction:
        from bball.services import prediction
        return getattr(prediction, name)

    _config_manager = {"ModelConfigManager"}
    if name in _config_manager:
        from bball.services.config_manager import ModelConfigManager
        return ModelConfigManager

    _business_logic = {"ModelBusinessLogic"}
    if name in _business_logic:
        from bball.services.business_logic import ModelBusinessLogic
        return ModelBusinessLogic

    _artifacts = {"ArtifactManager"}
    if name in _artifacts:
        from bball.services.artifacts import ArtifactManager
        return ArtifactManager

    _training_data = {
        "TrainingDataService", "MASTER_TRAINING_PATH", "MASTER_COLLECTION",
        "get_master_training_path", "get_master_collection_name",
        "get_all_possible_features", "get_available_seasons",
        "extract_features_from_master", "extract_features_from_master_for_points",
        "check_master_needs_regeneration", "register_existing_master_csv",
    }
    if name in _training_data:
        from bball.services import training_data
        return getattr(training_data, name)

    if name == "TrainingService":
        from bball.services.training_service import TrainingService
        return TrainingService

    if name == "WebpageParser":
        from bball.services.webpage_parser import WebpageParser
        return WebpageParser

    if name == "get_lineups":
        from bball.services.lineup_service import get_lineups
        return get_lineups

    _news = {"NewsService", "FetchResult", "NewsResults"}
    if name in _news:
        from bball.services import news_service
        return getattr(news_service, name)

    _game = {"get_game_detail", "get_team_players", "get_team_info"}
    if name in _game:
        from bball.services import game_service
        return getattr(game_service, name)

    if name == "build_rosters":
        from bball.services.roster_service import build_rosters
        return build_rosters

    _jobs = {"create_job", "update_job_progress", "complete_job", "fail_job", "get_job"}
    if name in _jobs:
        from bball.services import jobs
        return getattr(jobs, name)

    raise AttributeError(f"module 'bball.services' has no attribute {name!r}")


__all__ = [
    'PredictionService', 'PredictionResult', 'MatchupInfo',
    'ModelConfigManager',
    'ModelBusinessLogic',
    'ArtifactManager',
    'TrainingDataService', 'MASTER_TRAINING_PATH', 'MASTER_COLLECTION',
    'get_master_training_path', 'get_master_collection_name',
    'get_all_possible_features', 'get_available_seasons',
    'extract_features_from_master', 'extract_features_from_master_for_points',
    'check_master_needs_regeneration', 'register_existing_master_csv',
    'TrainingService',
    'WebpageParser',
    'get_lineups',
    'NewsService', 'FetchResult', 'NewsResults',
    'get_game_detail', 'get_team_players', 'get_team_info',
    'build_rosters',
    'create_job', 'update_job_progress', 'complete_job', 'fail_job', 'get_job',
]
