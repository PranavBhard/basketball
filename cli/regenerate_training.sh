#!/bin/bash
#
# Master Training Data Regeneration for NBA/CBB
#
# Usage:
#   ./cli/regenerate_training.sh <league> [options]
#
# Examples:
#   ./cli/regenerate_training.sh nba
#   ./cli/regenerate_training.sh cbb --no-player
#   ./cli/regenerate_training.sh nba --season 2023-2024
#   ./cli/regenerate_training.sh cbb --min-season 2015-2016
#   ./cli/regenerate_training.sh nba --limit 100
#
# Output CSV Structure:
#   The generated master training CSV follows this column structure:
#
#   Metadata columns (first):
#     Year, Month, Day    - Date components (integers) for filtering
#     Home, Away          - Team abbreviations
#     game_id             - Unique game identifier
#
#   Feature columns (middle):
#     ~7,400+ feature columns (e.g., assists_ratio_net|days_12|avg|home)
#
#   Target columns (last):
#     HomeWon             - Binary classification target (1=home win, 0=away win)
#     home_points         - Home team final score
#     away_points         - Away team final score
#

set -e

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment
source ./setup.sh

# Set Python path
export PYTHONPATH=/Users/pranav/Documents/basketball

# Run the regeneration script
python cli/scripts/regenerate_training.py "$@"
