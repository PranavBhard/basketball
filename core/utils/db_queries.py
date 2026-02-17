"""
Database Query Functions - Core Database Utilities

Helper functions for querying game data from in-memory collections.
Used primarily for feature calculations that need to look up historical games.
"""

from bball_app.core.utils.collection import import_collection
from bball_app.core.mongo import Mongo
from datetime import datetime, timedelta, date
from dateutil import relativedelta


def avg(ls):
    """Calculate average of a list."""
    return sum(ls) / float(len(ls))


def getDatesFromDate(theDate, beginDate):
    """Get list of date strings between beginDate and theDate."""
    dates = [beginDate + timedelta(days=x) for x in range((theDate - beginDate).days)]
    dates = [d.strftime('%Y-%m-%d') for d in dates]
    return dates


def getTeamLastNMonthsSeasonGames(team, year, month, day, season, Nmonths, stats_collection, exclude_game_types=None):
    """Get games for a team in the last N months of a season."""
    if exclude_game_types is None:
        exclude_game_types = ['preseason', 'allstar']
    home_map = stats_collection[0][season]
    theDate = datetime(year, month, day)
    beginDate = theDate - relativedelta.relativedelta(months=Nmonths)
    dates = getDatesFromDate(theDate, beginDate)
    games = []
    for d in dates:
        if d in home_map:
            home = home_map[d]
            for homeTeam, game in home.items():
                # Use .get() to handle missing game_type field (defaults to 'regseason')
                game_type = game.get('game_type', 'regseason')
                if game_type not in exclude_game_types and game.get('season') == season:
                    if team == game['homeTeam']['name']:
                        games.append(game)
                    elif team == game['awayTeam']['name']:
                        games.append(game)

    return games


def getTeamSeasonGamesFromDate(team, year, month, day, season, stats_collection, retHome=False, retAway=False, exclude_game_types=None):
    """Get all season games for a team up to a given date."""
    if exclude_game_types is None:
        exclude_game_types = ['preseason', 'allstar']
    home_map = stats_collection[0][season]
    home_games = []
    away_games = []
    for d, home in home_map.items():
        for homeTeam, game in home.items():
            # Use .get() to handle missing game_type field (defaults to 'regseason')
            game_type = game.get('game_type', 'regseason')
            if game_type not in exclude_game_types and game.get('season') == season:
                if team == game.get('homeTeam', {}).get('name'):
                    home_games.append(game)
                elif team == game.get('awayTeam', {}).get('name'):
                    away_games.append(game)

    if retHome:
        return home_games
    elif retAway:
        return away_games
    else:
        return home_games + away_games


def getTeamLastNDaysSeasonGames(team, year, month, day, season, Ndays, stats_collection, exclude_game_types=None):
    """Get games for a team in the last N days of a season."""
    if exclude_game_types is None:
        exclude_game_types = ['preseason', 'allstar']
    home_map = stats_collection[0][season]
    theDate = datetime(year, month, day)
    beginDate = theDate - relativedelta.relativedelta(days=Ndays)
    dates = getDatesFromDate(theDate, beginDate)
    games = []
    for d in dates:
        if d in home_map:
            home = home_map[d]
            for homeTeam, game in home.items():
                if game.get('game_type', 'regseason') not in exclude_game_types and game['season'] == season:
                    if team == game['homeTeam']['name']:
                        games.append(game)
                    elif team == game['awayTeam']['name']:
                        games.append(game)

    return games


# These 2 are DEPRECATED (only used in "normalized" stats)
def getTeamSeasonWins(team, year, month, day, stats_collection):
    """DEPRECATED: Get season wins for a team."""
    games = getTeamSeasonGamesFromDate(team, year, month, day, stats_collection)
    wins = 0
    for game in games:
        isHome = game['homeTeam']['name'] == team
        if (isHome and game['homeWon']) or (not isHome and not game['homeWon']):
            wins += 1
    return wins


def getTeamSeasonAverageForStat(stat, team, year, month, day, stats_collection):
    """DEPRECATED: Get season average for a stat against opponents."""
    games = getTeamSeasonGamesFromDate(team, year, month, day, stats_collection)
    if len(games) == 0:
        return 'SOME BS'

    opp_stat_against = 0
    for game in games:
        against = 'awayTeam' if game['homeTeam']['name'] == team else 'homeTeam'
        opp_stat_against += game[against][stat]

    retVal = opp_stat_against

    return retVal
