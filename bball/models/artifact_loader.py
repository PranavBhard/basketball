"""
Basketball Artifact Loader.

Thin subclass of sportscore's BaseArtifactLoader.
Default column names (HomeWon, Home, Away) are correct for basketball.
"""

from sportscore.models.base_artifact_loader import BaseArtifactLoader


class ArtifactLoader(BaseArtifactLoader):
    """Basketball artifact loader. Inherits all base functionality."""
    pass
