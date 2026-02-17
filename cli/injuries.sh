#!/bin/bash
#
# Injury Computation Pipeline
#
# Computes and updates injured_players for games based on roster vs played analysis.
# Displays live progress with per-season status tracking.
#
# Usage:
#   ./cli/injuries.sh <league> [options]
#
# Examples:
#   ./cli/injuries.sh nba
#   ./cli/injuries.sh cbb --workers 4
#   ./cli/injuries.sh cbb --season 2024-2025
#   ./cli/injuries.sh nba --date 2024-01-15
#   ./cli/injuries.sh nba --date-range 2024-01-01,2024-01-31
#   ./cli/injuries.sh nba --game-ids 401234567,401234568
#   ./cli/injuries.sh nba --dry-run
#
# Options:
#   Positional:
#     league                    nba, cbb
#
#   Parallel Processing:
#     -w, --workers N           Number of parallel workers for season processing (default: 1)
#
#   Date Filtering:
#     --date YYYY-MM-DD         Single date
#     --date-range START,END    Date range (e.g., 2024-01-01,2024-01-31)
#     --season YYYY-YYYY        Season filter
#     --game-ids ID1,ID2,...    Specific game IDs
#
#   Common:
#     --dry-run                 Preview without database changes
#     -v, --verbose             Detailed output
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

# Run the injuries pipeline
python -m bball_app.core.pipeline.injuries_pipeline "$LEAGUE" "$@"
