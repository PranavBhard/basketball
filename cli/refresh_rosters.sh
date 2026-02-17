#!/bin/bash
#
# Roster Build Pipeline
#
# Builds team rosters from player game stats for each season.
# Determines roster members (>= 5 MPG) and likely starters by position.
#
# Usage:
#   ./cli/refresh_rosters.sh <league> [options]
#
# Examples:
#   ./cli/refresh_rosters.sh nba
#   ./cli/refresh_rosters.sh wcbb
#   ./cli/refresh_rosters.sh nba --season 2024-2025
#   ./cli/refresh_rosters.sh wcbb --dry-run
#   ./cli/refresh_rosters.sh cbb --verbose
#
# Options:
#   Positional:
#     league                    nba, cbb, wcbb
#
#   Filtering:
#     --season YYYY-YYYY        Build rosters for a single season only
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

# Run the rosters pipeline
python -m bball_app.core.pipeline.rosters_pipeline "$LEAGUE" "$@"
