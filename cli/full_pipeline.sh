#!/bin/bash
#
# Full Data Pipeline for NBA/CBB
#
# Runs the complete data pipeline: ESPN pull -> Master Training generation
#
# Usage:
#   ./cli/full_pipeline.sh <league> [options]
#
# Examples:
#   ./cli/full_pipeline.sh nba
#   ./cli/full_pipeline.sh cbb --max-workers 8
#   ./cli/full_pipeline.sh nba --dry-run
#   ./cli/full_pipeline.sh cbb --seasons 2023-2024,2024-2025
#
# Options:
#   --max-workers N     Max parallel workers for ESPN pull
#   --seasons LIST      Comma-separated seasons to process
#   --skip-espn         Skip ESPN data pull
#   --skip-training     Skip master training generation
#   --skip-post         Skip post-processing
#   --skip-rosters      Skip roster build
#   --dry-run           Show what would be done
#   -v, --verbose       Show detailed output
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

LEAGUE=${1:-nba}
shift 2>/dev/null || true  # Remove first arg, pass rest to Python

# Change to project directory
cd "$(dirname "$0")/.."

# Load environment
source ./setup.sh

# Set Python path
export PYTHONPATH=/Users/pranav/Documents/basketball

# Run the full pipeline
python -m bball_app.core.pipeline.full_pipeline "$LEAGUE" "$@"
