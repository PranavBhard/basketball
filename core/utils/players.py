"""
Player Filtering Utilities - Core Infrastructure

Shared logic for building player lists for predictions.

CRITICAL: PREDICTION-ONLY MODULE

This module is specifically designed for PREDICTION workflows, NOT training data generation.

## Data Source Distinction:

### TRAINING (DO NOT USE THIS MODULE):
- Uses: stats_nba.homeTeam.injured_players and stats_nba.awayTeam.injured_players
- These are HISTORICAL snapshots captured when games were actually played
- Represents ground truth for "who was actually injured when this game happened"
- Handled by: StatHandlerV2.getInjuryFeatures() and per_calculator with game_doc.injured_players

### PREDICTION (USE THIS MODULE):
- Uses: nba_rosters.roster[].injured, nba_rosters.roster[].starter, nba_rosters.roster[].disabled flags
- These are USER-EDITABLE via the game_details UI toggles
- Gets updated in real-time for upcoming game predictions
- Used by: web/app.py predict() endpoint and agents/matchup_predict.py

## When to use this module:
- Web app prediction endpoints (api/predict)
- Agent prediction tools (matchup_predict.py)
- NOT for training data generation (BballModel.create_training_data)
- NOT for feature calculation for historical games (StatHandlerV2, per_calculator)

Uses data layer repositories for all database operations.
"""

from typing import Dict, List, Optional

from bball_app.core.data import RostersRepository


def build_player_lists_for_prediction(
    home_team: str,
    away_team: str,
    season: str,
    db=None,
    league=None
) -> Dict[str, Dict[str, List[str]]]:
    """
    Build player_filters dict for PREDICTION with injury/starter information.

    PREDICTION-ONLY: This function uses nba_rosters for injury/starter data.
    DO NOT use for training data generation (which requires stats_nba.injured_players).

    Rosters are the single source of truth. Player status is determined by
    three flags on each roster entry:
    - disabled=True  -> skip entirely (excluded from predictions)
    - injured=True   -> add to 'injured' list (not in 'playing')
    - starter=True   -> add to 'starters' list (if not injured/disabled)

    Args:
        home_team: Home team abbreviation (e.g., 'LAL')
        away_team: Away team abbreviation (e.g., 'BOS')
        season: Season string (e.g., '2023-2024')
        db: MongoDB database instance

    Returns:
        Dict with structure:
        {
            'LAL': {
                'playing': [player_id1, player_id2, ...],
                'starters': [starter_id1, starter_id2, ...],
                'injured': [injured_id1, injured_id2, ...]
            },
            'BOS': {
                'playing': [player_id1, player_id2, ...],
                'starters': [starter_id1, starter_id2, ...],
                'injured': [injured_id1, injured_id2, ...]
            }
        }
    """
    if db is None:
        from bball_app.core.mongo import Mongo
        db = Mongo().db

    rosters_repo = RostersRepository(db, league=league)

    player_filters = {}

    for team in [home_team, away_team]:
        roster_doc = rosters_repo.find_roster(team, season)

        playing = []
        starters = []
        injured = []

        if roster_doc:
            for entry in roster_doc.get('roster', []):
                player_id = str(entry.get('player_id', ''))
                if not player_id:
                    continue

                is_disabled = entry.get('disabled', False)
                if is_disabled:
                    continue

                is_injured = entry.get('injured', False)
                if is_injured:
                    injured.append(player_id)
                    continue

                playing.append(player_id)

                is_starter = entry.get('starter', False)
                if is_starter:
                    starters.append(player_id)

        player_filters[team] = {
            'playing': playing,
            'starters': starters,
            'injured': injured
        }

    return player_filters
