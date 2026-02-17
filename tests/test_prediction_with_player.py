#!/usr/bin/env python3
"""
Test script to predict game outcome with and without a specific player.

Tests game "401810208" with and without player "3032977" to verify
that player changes affect predictions correctly.
"""

import sys
import os
from datetime import datetime, date

# Add parent of nba_app to path for imports
# This allows imports like "from bball_app.core.mongo import Mongo"
# Script is in nba_app/tests/, so we need to go up to the parent of nba_app/
script_dir = os.path.dirname(os.path.abspath(__file__))  # nba_app/tests/
nba_app_dir = os.path.dirname(script_dir)  # nba_app/
project_root = os.path.dirname(nba_app_dir)  # Parent of nba_app/
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo
from bball_app.core.models.bball_model import BballModel
from bball_app.core.training.model_factory import create_model_with_c
from bball_app.core.services.training_data import get_all_possible_features
from bball_app.web.app import load_model_from_mongo_config
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def get_season_from_date(game_date: date) -> str:
    """Get NBA season string from date."""
    if game_date.month > 8:  # Oct-Dec
        return f"{game_date.year}-{game_date.year + 1}"
    else:  # Jan-Jun
        return f"{game_date.year - 1}-{game_date.year}"


def get_game_data(db, game_id: str):
    """Get game data from stats_nba collection."""
    game = db.stats_nba.find_one({'game_id': game_id})
    if not game:
        raise ValueError(f"Game {game_id} not found in database")
    
    return game


def get_players_from_game(db, game_id: str, team: str):
    """
    Get player IDs for a team from a game.
    Priority:
    1. stats_nba.homeTeam.players / awayTeam.players
    2. stats_nba_players query by game_id
    """
    game = db.stats_nba.find_one({'game_id': game_id})
    
    # Priority 1: Check stats_nba for stored player lists
    if game:
        team_key = 'homeTeam' if team == game.get('homeTeam', {}).get('name') else 'awayTeam'
        players_list = game.get(team_key, {}).get('players')
        if players_list:
            # Convert to strings for consistency
            players_list_str = [str(pid) for pid in players_list if pid]
            
            # Get starter information from stats_nba_players
            # Need to convert back to int for query, then back to str for return
            players_list_int = [int(pid) if isinstance(pid, str) else pid for pid in players_list if pid]
            starter_docs = list(db.stats_nba_players.find(
                {'game_id': game_id, 'team': team, 'player_id': {'$in': players_list_int}, 'starter': True},
                {'player_id': 1}
            ))
            starters = [str(p['player_id']) for p in starter_docs]
            
            return {
                'playing': players_list_str,
                'starters': starters
            }
    
    # Priority 2: Query stats_nba_players by game_id
    players_in_game = list(db.stats_nba_players.find(
        {'game_id': game_id, 'team': team, 'stats.min': {'$gt': 0}},
        {'player_id': 1, 'starter': 1}
    ))
    
    if players_in_game:
        playing = [str(p['player_id']) for p in players_in_game]
        starters = [str(p['player_id']) for p in players_in_game if p.get('starter', False)]
        return {
            'playing': playing,
            'starters': starters
        }
    
    return {'playing': [], 'starters': []}


def load_model_with_selected_config(db):
    """Load BballModel with the selected config from MongoDB."""
    # Get selected config
    selected_config = db.model_config_nba.find_one({'selected': True})
    if not selected_config:
        raise ValueError("No selected model config found in MongoDB. Please select a config in the web UI.")
    
    print(f"Loading model with config: {selected_config.get('name', 'Unknown')}")
    print(f"  Model Type: {selected_config.get('model_type')}")
    print(f"  Features: {selected_config.get('feature_count', 0)}")
    print(f"  Training CSV: {selected_config.get('training_csv')}")
    
    # Initialize model - features loaded from config below
    model = BballModel(
        classifier_features=[],  # Set from config
        points_features=[],  # Set from config
        include_elo=True,
        use_exponential_weighting=True,
        include_era_normalization=False,
        include_per_features=True
    )
    
    # Load model from config
    if not load_model_from_mongo_config(model, selected_config):
        raise ValueError("Failed to load model from selected config")
    
    print("Model loaded successfully!")
    return model


def make_prediction(model, game, player_filters):
    """Make a prediction for the game with given player filters."""
    home_team = game['homeTeam']['name']
    away_team = game['awayTeam']['name']
    
    # Get game date
    year = game['year']
    month = game['month']
    day = game['day']
    game_date = datetime(year, month, day).date()
    season = get_season_from_date(game_date)
    game_date_str = f"{year}-{month:02d}-{day:02d}"
    
    # Make prediction
    prediction = model.predict_with_player_config(
        home_team, away_team, season, game_date_str, player_filters
    )
    
    return prediction


def main():
    """Main test function."""
    game_id = "401810208"
    test_player_id = "3032977"
    
    print("=" * 80)
    print("PREDICTION TEST: Game with and without specific player")
    print("=" * 80)
    print(f"Game ID: {game_id}")
    print(f"Test Player ID: {test_player_id}")
    print()
    
    # Connect to MongoDB
    print("Connecting to MongoDB...")
    mongo = Mongo()
    db = mongo.db
    print("Connected!")
    print()
    
    # Load model with selected config
    print("Loading model with selected config...")
    model = load_model_with_selected_config(db)
    print()
    
    # Get game data
    print(f"Fetching game data for {game_id}...")
    game = get_game_data(db, game_id)
    home_team = game['homeTeam']['name']
    away_team = game['awayTeam']['name']
    
    print(f"Game: {away_team} @ {home_team}")
    print(f"Date: {game['year']}-{game['month']:02d}-{game['day']:02d}")
    print()
    
    # Get player lists for both teams
    print("Fetching player lists...")
    home_players = get_players_from_game(db, game_id, home_team)
    away_players = get_players_from_game(db, game_id, away_team)
    
    print(f"Home team ({home_team}): {len(home_players['playing'])} players, {len(home_players['starters'])} starters")
    print(f"Away team ({away_team}): {len(away_players['playing'])} players, {len(away_players['starters'])} starters")
    print()
    
    # Determine which team the test player is on
    test_player_team = None
    if test_player_id in home_players['playing']:
        test_player_team = home_team
        print(f"Test player {test_player_id} is on HOME team ({home_team})")
    elif test_player_id in away_players['playing']:
        test_player_team = away_team
        print(f"Test player {test_player_id} is on AWAY team ({away_team})")
    else:
        print(f"WARNING: Test player {test_player_id} not found in either team's roster!")
        print("Will proceed with full roster prediction only.")
        test_player_team = None
    print()
    
    # Create player filters with all players
    player_filters_all = {
        home_team: {
            'playing': home_players['playing'],
            'starters': home_players['starters']
        },
        away_team: {
            'playing': away_players['playing'],
            'starters': away_players['starters']
        }
    }
    
    # Make prediction with all players
    print("=" * 80)
    print("PREDICTION 1: With ALL players")
    print("=" * 80)
    prediction_all = make_prediction(model, game, player_filters_all)
    
    print(f"Predicted Winner: {prediction_all['predicted_winner']}")
    print(f"Home Win Probability: {prediction_all['home_win_prob']:.1f}%")
    print(f"Away Win Probability: {100 - prediction_all['home_win_prob']:.1f}%")
    print(f"American Odds: {prediction_all['odds']:+d}")
    if prediction_all.get('home_pts') and prediction_all.get('away_pts'):
        print(f"Predicted Score: {home_team} {prediction_all['home_pts']} - {away_team} {prediction_all['away_pts']}")
    print()
    
    # Make prediction without test player
    if test_player_team:
        print("=" * 80)
        print(f"PREDICTION 2: WITHOUT player {test_player_id} ({test_player_team})")
        print("=" * 80)
        
        # Create player filters without test player
        player_filters_without = {
            home_team: {
                'playing': [p for p in home_players['playing'] if p != test_player_id],
                'starters': [p for p in home_players['starters'] if p != test_player_id]
            },
            away_team: {
                'playing': [p for p in away_players['playing'] if p != test_player_id],
                'starters': [p for p in away_players['starters'] if p != test_player_id]
            }
        }
        
        print(f"Removed player {test_player_id} from {test_player_team}")
        print(f"  {test_player_team} now has {len(player_filters_without[test_player_team]['playing'])} players")
        print()
        
        prediction_without = make_prediction(model, game, player_filters_without)
        
        print(f"Predicted Winner: {prediction_without['predicted_winner']}")
        print(f"Home Win Probability: {prediction_without['home_win_prob']:.1f}%")
        print(f"Away Win Probability: {100 - prediction_without['home_win_prob']:.1f}%")
        print(f"American Odds: {prediction_without['odds']:+d}")
        if prediction_without.get('home_pts') and prediction_without.get('away_pts'):
            print(f"Predicted Score: {home_team} {prediction_without['home_pts']} - {away_team} {prediction_without['away_pts']}")
        print()
        
        # Compare results
        print("=" * 80)
        print("COMPARISON")
        print("=" * 80)
        home_prob_diff = prediction_without['home_win_prob'] - prediction_all['home_win_prob']
        away_prob_diff = (100 - prediction_without['home_win_prob']) - (100 - prediction_all['home_win_prob'])
        
        print(f"Home Win Probability Change: {home_prob_diff:+.1f}%")
        print(f"Away Win Probability Change: {away_prob_diff:+.1f}%")
        print()
        
        if prediction_all['predicted_winner'] != prediction_without['predicted_winner']:
            print(f"⚠️  WINNER CHANGED: {prediction_all['predicted_winner']} → {prediction_without['predicted_winner']}")
        else:
            print(f"Winner unchanged: {prediction_all['predicted_winner']}")
        
        if abs(home_prob_diff) < 0.1:
            print("⚠️  WARNING: Probability change is very small (< 0.1%).")
            print("   This might indicate:")
            print("   - Player filters are not being applied correctly")
            print("   - PER features are not being recalculated")
            print("   - Model is not sensitive to PER feature changes")
        else:
            print(f"✓ Probability changed by {abs(home_prob_diff):.1f}%, indicating player filters are working.")
        
        print()
    
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\nERROR: {e}")
        traceback.print_exc()
        sys.exit(1)

