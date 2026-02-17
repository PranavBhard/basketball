#!/bin/bash
#
# Generate Master Training Data for NBA/CBB
#
# Uses SharedFeatureContext + 32 workers + 500-row chunks for efficient
# parallel processing.
#
# Usage:
#   ./cli/generate_training.sh <league> [options]
#
# Examples:
#   ./cli/generate_training.sh nba
#   ./cli/generate_training.sh cbb --workers 16
#   ./cli/generate_training.sh nba --season 2023-2024
#   ./cli/generate_training.sh cbb --no-player
#   ./cli/generate_training.sh nba --min-season 2020-2021
#   ./cli/generate_training.sh nba --features "elo_rating|none|raw|home,elo_rating|none|raw|away"
#   ./cli/generate_training.sh nba --add --features "vegas_*"  # Pattern: all vegas features
#   ./cli/generate_training.sh nba --add --features "player_bench_per|season|weighted_MIN_REC(k=35)|home"
#   ./cli/generate_training.sh nba --add --season 2024-2025
#   ./cli/generate_training.sh nba --add --seasons "2020-2021,2024-2025"  # range: 2020-2021 through 2024-2025
#   ./cli/generate_training.sh cbb --seasons "2015-2016,2024-2025"        # generate only these seasons
#
# Options:
#   --workers N         Number of parallel workers (default: 32)
#   --chunk-size N      Rows per chunk (default: 500)
#   --season SEASON     Specific season to generate (e.g., "2024-2025")
#   --seasons START,END Season range to generate (e.g., "2015-2016,2024-2025" = all seasons from 2015-2016 through 2024-2025)
#   --min-season SEASON Minimum season to include
#   --no-player         Skip player-level features (PER, injuries)
#   --limit N           Limit number of games (for testing)
#   --output PATH       Output CSV path
#   --dry-run           Show what would be done
#   --features LIST     Comma-separated list of feature names or patterns to generate
#                       Supports wildcards: "vegas_*" matches all vegas features
#                       (default: all features)
#   --exclude-features  Comma-separated list of feature names or patterns to EXCLUDE
#                       Supports wildcards: "player_*,inj_*" excludes all player/injury features
#   --add               Add/update to existing CSV:
#                       - With --features/--exclude-features: updates specified columns only
#                       - With --season/--seasons: replaces rows for those seasons
#                       - Combine all three to update specific columns for specific seasons
#
# Prediction Columns:
#   If a points model is selected in model_config_points_nba, prediction columns
#   are automatically generated using the already-calculated features (no extra DB calls):
#     pred_home_points, pred_away_points, pred_margin, pred_point_total
#
# Output CSV Structure:
#   The generated CSV follows this column structure (required by web UI):
#
#   Metadata columns (first):
#     Year, Month, Day    - Date components (integers) for filtering
#     Home, Away          - Team abbreviations
#     game_id             - Unique game identifier
#
#   Feature columns (middle):
#     ~7,400+ feature columns (e.g., assists_ratio_net|days_12|avg|home)
#
#   Prediction columns (if points model selected):
#     pred_home_points    - Predicted home team score
#     pred_away_points    - Predicted away team score
#     pred_margin         - Predicted point differential (home - away)
#     pred_point_total    - Predicted total points (home + away)
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

# Run the training pipeline
python -m bball_app.core.pipeline.training_pipeline "$LEAGUE" "$@"
