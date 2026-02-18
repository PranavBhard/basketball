"""
Elo Rating Cache — thin wrapper around sportscore's EloCache.

Auto-injects basketball's GamesRepository so callers don't need to
pass games_repo explicitly.  All Elo logic lives in sportscore.
"""

from typing import Optional
from pymongo.database import Database

from sportscore.stats.elo_cache import EloCache as _BaseEloCache
from bball.data import GamesRepository


class EloCache(_BaseEloCache):
    """Basketball EloCache — auto-injects GamesRepository."""

    def __init__(self, db: Database, league=None, **kwargs):
        if "games_repo" not in kwargs:
            kwargs["games_repo"] = GamesRepository(db, league=league)
        super().__init__(db, league=league, **kwargs)


def get_elo_cache(db: Database = None) -> EloCache:
    """
    Convenience factory for an EloCache instance.

    Args:
        db: Optional MongoDB database. If None, creates new connection.
    """
    if db is None:
        from bball.mongo import Mongo
        db = Mongo().db
    return EloCache(db)
