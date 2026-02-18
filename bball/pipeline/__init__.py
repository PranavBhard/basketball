"""
Pipeline module for league-specific data processing.

This module provides:
- PipelineConfig: Configuration loaded from league YAML pipelines section
- TrainingConfig: Training-specific configuration
- SyncConfig: Data sync configuration
- SharedFeatureContext: Pre-loads data ONCE for efficient parallel feature calculation
- training_pipeline: Chunked async master training generation
- full_pipeline: Full data refresh orchestrator (ESPN + training)
- sync_pipeline: Data sync from ESPN to MongoDB
- injuries_pipeline: Injury computation and updates
"""

from bball.pipeline.config import PipelineConfig, TrainingConfig, SyncConfig
from bball.pipeline.shared_context import SharedFeatureContext
from bball.pipeline.full_pipeline import BasketballFullPipeline

__all__ = [
    "PipelineConfig",
    "TrainingConfig",
    "SyncConfig",
    "SharedFeatureContext",
    "BasketballFullPipeline",
]
