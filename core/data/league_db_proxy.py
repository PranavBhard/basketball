"""
League Database Proxy - Maps legacy collection attribute names to league-configured names.

Provides transparent access to MongoDB collections using either legacy NBA-specific
attribute names (e.g., db.stats_nba) or normalized names (e.g., db.games), resolving
them via the active league's YAML configuration.

Usage:
    from bball_app.core.data.league_db_proxy import LeagueDbProxy
    from bball_app.core.mongo import Mongo
    from bball_app.core.league_config import load_league_config

    league = load_league_config('nba')
    db = LeagueDbProxy(Mongo().db, league)

    # Both work — attribute access resolves through league config:
    db.stats_nba          # -> db['stats_nba'] (from league.collections['games'])
    db.player_stats       # -> db['nba_player_stats'] (from league.collections['player_stats'])
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


class LeagueDbProxy:
    """
    Database proxy mapping NBA-coded collection attribute access (db.stats_nba, ...)
    to the active league's configured collections.

    This keeps consumer code largely unchanged while enabling
    ``--league cbb`` (or future leagues).
    """

    _NBA_ATTR_TO_KEY = {
        # Backwards-compat: old NBA-specific attribute names
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

    def __init__(self, db, league: "LeagueConfig"):
        self._db = db
        self._league = league

    def __getitem__(self, name: str):
        return self._db[name]

    def __getattr__(self, name: str):
        key = self._NBA_ATTR_TO_KEY.get(name)
        if key is not None:
            coll_name = self._league.collections.get(key)
            if coll_name:
                return self._db[coll_name]
        return getattr(self._db, name)
