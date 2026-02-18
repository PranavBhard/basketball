"""
League configuration loader.

The app supports multiple basketball leagues (NBA, CBB, ...). All league-specific
variables live in YAML files under `leagues/<league_id>.yaml`.

This module provides BasketballLeagueConfig (extending sportscore's BaseLeagueConfig)
with basketball-specific properties, and a convenience loader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sportscore.league_config import (
    BaseLeagueConfig,
    LeagueConfigError,
    load_league_config as _base_load,
    get_available_leagues as _base_get_available,
    _normalize_path,
)
from sportscore.sport_config import get_builtin_sport_config


def _leagues_dir() -> str:
    """Return the leagues directory path."""
    # bball/league_config.py -> bball -> basketball (repo root) -> leagues/
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "leagues")


@dataclass(frozen=True)
class BasketballLeagueConfig(BaseLeagueConfig):
    """Basketball-specific league configuration."""

    @property
    def _repo_root(self) -> str:
        # basketball/leagues/nba.yaml -> up 2 levels
        leagues_dir = os.path.dirname(self.config_path)
        return os.path.dirname(leagues_dir)

    @property
    def _required_collections(self) -> List[str]:
        return [
            "games",
            "player_stats",
            "players",
            "teams",
            "venues",
            "rosters",
            "model_config_classifier",
            "model_config_points",
            "master_training_metadata",
            "cached_league_stats",
            "elo_cache",
            "experiment_runs",
            "jobs",
        ]

    @property
    def extra_feature_stats(self) -> List[str]:
        """League-specific extra stat names from YAML config.

        Basketball uses nested ``extra_features.stats`` in the YAML,
        while the base class reads a flat ``extra_feature_stats`` key.
        """
        ef = self.raw.get("extra_features") or {}
        stats = ef.get("stats", [])
        return stats if isinstance(stats, list) else []


# Backward compatibility alias — 48 files import `LeagueConfig`
LeagueConfig = BasketballLeagueConfig


# -----------------------
# Module-level loaders
# -----------------------

_CACHE: Dict[str, BasketballLeagueConfig] = {}


def get_available_leagues() -> List[str]:
    """Return list of available league IDs from YAML files in leagues/ dir."""
    return _base_get_available(_leagues_dir())


def get_league_config_path(league_id: str) -> str:
    """Return the file path for a league's YAML config."""
    return os.path.join(_leagues_dir(), f"{league_id}.yaml")


def load_league_config(league_id: str = "nba", *, use_cache: bool = True) -> BasketballLeagueConfig:
    """
    Load and validate a basketball league configuration.

    Args:
        league_id: League identifier (e.g., 'nba', 'cbb'). Defaults to 'nba'.
        use_cache: Whether to use cached config. Set False for fresh load.

    Returns:
        BasketballLeagueConfig instance.

    Raises:
        LeagueConfigError: If config is missing or invalid.
    """
    league_id = (league_id or "nba").strip().lower()

    if use_cache and league_id in _CACHE:
        return _CACHE[league_id]

    sport_cfg = get_builtin_sport_config("basketball")

    cfg = _base_load(
        league_id=league_id,
        leagues_dir=_leagues_dir(),
        config_class=BasketballLeagueConfig,
        use_cache=False,  # We handle caching ourselves
        required_endpoints=[
            "scoreboard_header_template",
            "scoreboard_site_template",
            "game_summary_template",
        ],
        sport_config=sport_cfg,
    )

    if use_cache:
        _CACHE[league_id] = cfg

    return cfg


def clear_config_cache():
    """Clear the league config cache (useful for testing)."""
    _CACHE.clear()
