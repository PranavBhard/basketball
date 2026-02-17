"""
Collection to Dict - Core Database Utilities

Utility functions for importing MongoDB collections into dictionaries
for fast in-memory lookups during batch processing.
"""

from pprint import pprint
from bball_app.core.mongo import Mongo

# Lazy initialization - connection created on first use
_mongo = None


def _get_mongo():
    global _mongo
    if _mongo is None:
        _mongo = Mongo()
    return _mongo


def import_collection(coll, query={'season': {'$exists': True}}):
    """
    Import a MongoDB collection into nested dictionaries for fast lookup.

    Args:
        coll: Collection name (e.g., 'stats_nba')
        query: MongoDB query filter

    Returns:
        Tuple of (home_games_dict, away_games_dict) where each dict is
        nested as: {season: {date: {team_name: game_doc}}}
    """
    mongo = _get_mongo()
    stats = mongo.db[coll].find(query, {
        'homeTeam': 1, 'awayTeam': 1, 'homeWon': 1,
        'game_type': 1, 'season': 1, 'date': 1,
        'venue_guid': 1, 'year': 1, 'month': 1, 'day': 1,
        'vegas': 1, 'pregame_lines': 1,  # Vegas betting lines
    }, batch_size=2**8)

    home_games = {}
    away_games = {}
    for game in stats:
        game['_id'] = str(game['_id'])
        home = game['homeTeam']['name']
        away = game['awayTeam']['name']

        key1 = game['season']
        key2 = game['date']
        home_games.setdefault(key1, {}).setdefault(key2, {})[home] = game
        away_games.setdefault(key1, {}).setdefault(key2, {})[away] = game

    return [home_games, away_games]
