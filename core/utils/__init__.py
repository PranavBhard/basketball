"""
Utility Functions Module.

This module contains shared utility functions used across the NBA app.

Submodules:
- collection: MongoDB collection import utilities
- db_queries: Database query helpers for feature calculations
- players: Player utility functions for predictions
- espn_audit: ESPN database audit tools
"""

from datetime import date, datetime
from typing import Optional, Union

from bball_app.core.league_config import LeagueConfig, load_league_config

# Collection utilities
from bball_app.core.utils.collection import import_collection

# Database query functions
from bball_app.core.utils.db_queries import (
    getTeamSeasonGamesFromDate,
    getTeamLastNMonthsSeasonGames,
    getTeamLastNDaysSeasonGames,
)

# Player utilities
from bball_app.core.utils.players import build_player_lists_for_prediction

# ESPN audit tools
from bball_app.core.utils.espn_audit import (
    run_espn_vs_db_audit,
    fetch_espn_games_for_date_range,
    filter_games_for_team,
)


def get_season_from_date(
    game_date: Union[date, datetime],
    league: Optional[LeagueConfig] = None,
    league_id: Optional[str] = None
) -> str:
    """
    Get season string from date using league config's season_cutover_month.

    Args:
        game_date: Date to get season for (date or datetime)
        league: Optional LeagueConfig instance
        league_id: Optional league ID (e.g., 'nba', 'cbb') to load config

    Returns:
        Season string (e.g., '2024-2025')

    The season cutover month determines when a new season starts. For basketball,
    games after the cutover month (typically August) are in the next season.
    For example, with cutover_month=8:
    - October 2024 -> '2024-2025' season
    - January 2025 -> '2024-2025' season
    """
    # Get cutover month from league config, default to 8 (August)
    cutover_month = 8
    if league:
        cutover_month = league.season_rules.get('season_cutover_month', 8)
    elif league_id:
        try:
            loaded_league = load_league_config(league_id)
            cutover_month = loaded_league.season_rules.get('season_cutover_month', 8)
        except Exception:
            pass  # Use default if loading fails

    # Convert datetime to date if needed
    if isinstance(game_date, datetime):
        game_date = game_date.date()

    if game_date.month > cutover_month:
        return f"{game_date.year}-{game_date.year + 1}"
    else:
        return f"{game_date.year - 1}-{game_date.year}"


def build_time_suffix(begin_year: int, calibration_years: list, evaluation_year: int) -> str:
    """Build time config suffix like 'T10C212223E24'."""
    begin_yy = str(begin_year % 100).zfill(2)
    cal_yy = ''.join(str(y % 100).zfill(2) for y in sorted(calibration_years))
    eval_yy = str(evaluation_year % 100).zfill(2)
    return f"T{begin_yy}C{cal_yy}E{eval_yy}"


__all__ = [
    # Collection
    'import_collection',
    # DB Queries
    'getTeamSeasonGamesFromDate',
    'getTeamLastNMonthsSeasonGames',
    'getTeamLastNDaysSeasonGames',
    # Players
    'build_player_lists_for_prediction',
    # ESPN Audit
    'run_espn_vs_db_audit',
    'fetch_espn_games_for_date_range',
    'filter_games_for_team',
    # Season calculation
    'get_season_from_date',
    # Time suffix
    'build_time_suffix',
]
