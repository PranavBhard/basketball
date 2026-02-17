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
"""

# Prediction service
from bball_app.core.services.prediction import PredictionService, PredictionResult, MatchupInfo

# Config manager
from bball_app.core.services.config_manager import ModelConfigManager

# Business logic
from bball_app.core.services.business_logic import ModelBusinessLogic

# Artifacts
from bball_app.core.services.artifacts import ArtifactManager

# Training data
from bball_app.core.services.training_data import (
    TrainingDataService,
    # Module-level constants
    MASTER_TRAINING_PATH,
    MASTER_COLLECTION,
    # Convenience functions (backward compatibility)
    get_master_training_path,
    get_master_collection_name,
    get_all_possible_features,
    get_available_seasons,
    extract_features_from_master,
    extract_features_from_master_for_points,
    check_master_needs_regeneration,
    register_existing_master_csv,
)

# Training service (CLI orchestration)
from bball_app.core.services.training_service import TrainingService

# Webpage parser
from bball_app.core.services.webpage_parser import WebpageParser

# Lineup service
from bball_app.core.services.lineup_service import get_lineups

# News service
from bball_app.core.services.news_service import NewsService, FetchResult, NewsResults

# Game service
from bball_app.core.services.game_service import (
    get_game_detail,
    get_team_players,
    get_team_info,
)

# Roster service
from bball_app.core.services.roster_service import build_rosters

# Jobs infrastructure (for background task tracking)
from bball_app.core.services.jobs import (
    create_job,
    update_job_progress,
    complete_job,
    fail_job,
    get_job,
)

__all__ = [
    # Prediction
    'PredictionService',
    'PredictionResult',
    'MatchupInfo',
    # Config
    'ModelConfigManager',
    # Business logic
    'ModelBusinessLogic',
    # Artifacts
    'ArtifactManager',
    # Training data
    'TrainingDataService',
    'MASTER_TRAINING_PATH',
    'MASTER_COLLECTION',
    'get_master_training_path',
    'get_master_collection_name',
    'get_all_possible_features',
    'get_available_seasons',
    'extract_features_from_master',
    'extract_features_from_master_for_points',
    'check_master_needs_regeneration',
    'register_existing_master_csv',
    # Training service
    'TrainingService',
    # Webpage parser
    'WebpageParser',
    # Lineup service
    'get_lineups',
    # News service
    'NewsService',
    'FetchResult',
    'NewsResults',
    # Game service
    'get_game_detail',
    'get_team_players',
    'get_team_info',
    # Roster service
    'build_rosters',
    # Jobs infrastructure
    'create_job',
    'update_job_progress',
    'complete_job',
    'fail_job',
    'get_job',
]
