"""
League Database Proxy - Maps legacy collection attribute names to league-configured names.

Provides transparent access to MongoDB collections using either legacy NBA-specific
attribute names (e.g., db.stats_nba) or normalized names (e.g., db.games), resolving
them via the active league's YAML configuration.

Usage:
    from bball.data.league_db_proxy import LeagueDbProxy
    from bball.mongo import Mongo
    from bball.league_config import load_league_config

    league = load_league_config('nba')
    db = LeagueDbProxy(Mongo().db, league)

    # Both work — attribute access resolves through league config:
    db.stats_nba          # -> db['nba_games'] (from league.collections['games'])
    db.player_stats       # -> db['nba_player_stats'] (from league.collections['player_stats'])
"""

from typing import TYPE_CHECKING
from sportscore.db.league_db_proxy import LeagueDbProxy as _BaseLeagueDbProxy

if TYPE_CHECKING:
    from bball.league_config import BasketballLeagueConfig


class LeagueDbProxy(_BaseLeagueDbProxy):
    """
    Database proxy mapping NBA-coded collection attribute access (db.stats_nba, ...)
    to the active league's configured collections.
    """

    _ATTR_TO_KEY = {
        # Legacy NBA-specific attribute names -> config keys
        "stats_nba": "games",
        "stats_nba_players": "player_stats",
        "players_nba": "players",
        "teams_nba": "teams",
        "nba_venues": "venues",
        "nba_rosters": "rosters",
        "model_config_nba": "model_config_classifier",
        "model_config_points_nba": "model_config_points",
        "master_training_data_nba": "master_training_metadata",
        "cached_league_stats": "cached_league_stats",
        "nba_cached_elo_ratings": "elo_cache",
        "experiment_runs": "experiment_runs",
        "jobs_nba": "jobs",
        # Normalized names (preferred for new code)
        "games": "games",
        "player_stats": "player_stats",
        "players": "players",
        "teams": "teams",
        "venues": "venues",
        "rosters": "rosters",
        "model_config_classifier": "model_config_classifier",
        "model_config_points": "model_config_points",
        "master_training_metadata": "master_training_metadata",
        "elo_cache": "elo_cache",
        "jobs": "jobs",
    }
