"""
Pipeline configuration loaded from league YAML.

The league YAML files can include a `pipelines` section that configures:
- Full pipeline steps (ESPN pull, post-processing, training, registration)
- Training-specific settings (workers, chunk size, preload options)
- Sync settings (data types, workers, post-processing options)
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set

from bball.league_config import LeagueConfig


# Valid data types for sync pipeline
SYNC_DATA_TYPES = {'games', 'players', 'player_stats', 'venues', 'rosters', 'teams'}


@dataclass
class TrainingConfig:
    """Training pipeline configuration."""
    workers: int = 32
    chunk_size: int = 500
    include_player_features: bool = True
    preload_games: bool = True
    preload_venues: bool = True
    preload_per_cache: bool = True
    preload_injury_cache: bool = True


@dataclass
class SyncConfig:
    """
    Data sync pipeline configuration.

    Controls which data types to sync and post-processing options.
    """
    # Data types to sync (default: all)
    data_types: Set[str] = field(default_factory=lambda: SYNC_DATA_TYPES.copy())

    # Worker settings
    workers: int = 4

    # Post-processing options
    with_injuries: bool = False
    with_elo: bool = False

    # Refresh settings
    refresh_venues: bool = True
    refresh_players: bool = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'SyncConfig':
        """Load SyncConfig from a dictionary."""
        data_types = config.get('data_types')
        if data_types:
            if isinstance(data_types, str):
                data_types = {t.strip() for t in data_types.split(',')}
            else:
                data_types = set(data_types)
        else:
            data_types = SYNC_DATA_TYPES.copy()

        return cls(
            data_types=data_types,
            workers=config.get('workers', 4),
            with_injuries=config.get('with_injuries', False),
            with_elo=config.get('with_elo', False),
            refresh_venues=config.get('refresh_venues', True),
            refresh_players=config.get('refresh_players', True),
        )


@dataclass
class OddsBackfillConfig:
    """Odds backfill step configuration from YAML."""
    max_workers: int = 4
    min_season: Optional[str] = None   # e.g. "2024-2025"
    source: str = "espn"               # future: "csv", "odds_api"


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    league: LeagueConfig

    # ESPN pull settings
    espn_parallel: bool = True
    espn_max_workers: int = 4

    # Post-processing settings
    refresh_venues: bool = True
    refresh_players: bool = True
    refresh_teams: bool = True
    compute_injuries: bool = True
    build_rosters: bool = True

    # Training settings
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Odds backfill settings
    odds_backfill: OddsBackfillConfig = field(default_factory=OddsBackfillConfig)

    # Flags
    cache_elo: bool = True
    register_csv: bool = True

    @classmethod
    def from_league(cls, league: LeagueConfig) -> 'PipelineConfig':
        """
        Load pipeline config from league YAML.

        The league YAML can have a `pipelines` section like:
            pipelines:
              full:
                steps:
                  - espn_pull:
                      parallel: true
                      max_workers: 4
                  - post_processing:
                      refresh_venues: true
                      refresh_players: true
                      compute_injuries: true
                  - cache_elo: true
                  - generate_training:
                      workers: 32
                      chunk_size: 500
                      include_player_features: true
                  - register_csv: true
              training:
                workers: 32
                chunk_size: 500
                include_player_features: true
                preload:
                  games: true
                  venues: true
                  per_cache: true
                  injury_cache: true
        """
        raw = league.pipelines
        full = raw.get('full', {})
        training_raw = raw.get('training', {})

        # Parse step configs from full pipeline
        steps = full.get('steps', [])

        espn_config: Dict[str, Any] = {}
        post_config: Dict[str, Any] = {}
        train_config: Dict[str, Any] = {}
        odds_config: Dict[str, Any] = {}
        cache_elo_val = True
        register_csv_val = True

        for step in steps:
            if isinstance(step, dict):
                if 'espn_pull' in step:
                    espn_config = step['espn_pull'] if isinstance(step['espn_pull'], dict) else {}
                elif 'post_processing' in step:
                    post_config = step['post_processing'] if isinstance(step['post_processing'], dict) else {}
                elif 'generate_training' in step:
                    train_config = step['generate_training'] if isinstance(step['generate_training'], dict) else {}
                elif 'odds_backfill' in step:
                    odds_config = step['odds_backfill'] if isinstance(step['odds_backfill'], dict) else {}
                elif 'cache_elo' in step:
                    cache_elo_val = step['cache_elo'] if isinstance(step['cache_elo'], bool) else True
                elif 'register_csv' in step:
                    register_csv_val = step['register_csv'] if isinstance(step['register_csv'], bool) else True

        # Build preload config
        preload = training_raw.get('preload', {})

        training = TrainingConfig(
            workers=training_raw.get('workers', train_config.get('workers', 32)),
            chunk_size=training_raw.get('chunk_size', train_config.get('chunk_size', 500)),
            include_player_features=training_raw.get(
                'include_player_features',
                train_config.get('include_player_features', True)
            ),
            preload_games=preload.get('games', True),
            preload_venues=preload.get('venues', True),
            preload_per_cache=preload.get('per_cache', True),
            preload_injury_cache=preload.get('injury_cache', True),
        )

        odds_backfill = OddsBackfillConfig(
            max_workers=odds_config.get('max_workers', 4),
            min_season=odds_config.get('min_season'),
            source=odds_config.get('source', 'espn'),
        )

        return cls(
            league=league,
            espn_parallel=espn_config.get('parallel', True),
            espn_max_workers=espn_config.get('max_workers', 4),
            refresh_venues=post_config.get('refresh_venues', True),
            refresh_players=post_config.get('refresh_players', True),
            refresh_teams=post_config.get('refresh_teams', True),
            compute_injuries=post_config.get('compute_injuries', True),
            build_rosters=post_config.get('build_rosters', True),
            training=training,
            odds_backfill=odds_backfill,
            cache_elo=cache_elo_val,
            register_csv=register_csv_val,
        )


def get_default_training_config(league: LeagueConfig) -> TrainingConfig:
    """
    Get training config with league-specific defaults.

    Some leagues (like CBB) may want different defaults:
    - CBB: More games, so may want different chunk sizes
    - CBB: Player features less reliable, may want to skip
    """
    pipeline_config = PipelineConfig.from_league(league)
    return pipeline_config.training
