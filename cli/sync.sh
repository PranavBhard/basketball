#!/bin/bash
#
# Data Sync Pipeline
#
# Pulls/upserts ESPN data into MongoDB (games, players, venues, rosters, teams).
#
# Usage:
#   ./cli/sync.sh <league> [options]
#
# Examples:
#   ./cli/sync.sh nba --date 2024-01-15
#   ./cli/sync.sh cbb --date-range 2024-01-01,2024-01-31
#   ./cli/sync.sh nba --season 2024-2025
#   ./cli/sync.sh nba --season 2024-2025 --with-injuries
#   ./cli/sync.sh nba --dry-run --date 2024-01-15
#
# Options:
#   Positional:
#     league                    nba, cbb
#
#   Date Filtering (one required):
#     --date YYYY-MM-DD         Single date
#     --date-range START,END    Date range (e.g., 2024-01-01,2024-01-31)
#     --season YYYY-YYYY        Entire season
#
#   Data Selection:
#     --only games,players      Only sync specific data types
#     --skip venues,rosters     Skip specific data types
#
#   Post-Processing:
#     --with-injuries           Compute injuries after sync
#     --with-elo                Update ELO cache after sync
#     --with-geocoding          Geocode venues without lat/lon (rate-limited)
#
#   Common:
#     --workers N               Parallel workers (default: 4)
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

# Run the sync pipeline
python -m bball_app.core.pipeline.sync_pipeline "$LEAGUE" "$@"
