#!/bin/bash
#
# Data Ingestion Pipeline
#
# Runs the full data ingestion portion of the pipeline: ESPN pull, post-processing,
# injuries, ELO cache, and roster build. Populates all MongoDB collections without
# generating master training CSVs.
#
# Collections populated:
#   - games (stats_nba)           ESPN game results and box scores
#   - player_stats                Per-game player statistics
#   - players                     Player metadata (name, headshot, position)
#   - teams                       Team metadata (name, abbreviation, conference)
#   - venues                      Venue info with geocoding
#   - rosters                     Team rosters per season (starters, bench)
#   - elo_cache                   Precomputed ELO ratings per team per game
#   - games.injured_players       Injury fields computed on game documents
#
# Usage:
#   ./cli/ingest.sh <league> [options]
#
# Examples:
#   ./cli/ingest.sh nba
#   ./cli/ingest.sh cbb --max-workers 8
#   ./cli/ingest.sh nba --seasons 2023-2024,2024-2025
#   ./cli/ingest.sh nba --skip-injuries
#   ./cli/ingest.sh nba --skip-rosters
#   ./cli/ingest.sh cbb --dry-run
#
# Options:
#   --max-workers N     Max parallel workers for ESPN pull
#   --seasons LIST      Comma-separated seasons to process
#   --skip-espn         Skip ESPN data pull
#   --skip-post         Skip post-processing (venues, teams, players)
#   --skip-injuries     Skip injury computation
#   --skip-rosters      Skip roster build
#   --dry-run           Show what would be done
#   -v, --verbose       Show detailed output
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

# Run the full pipeline with training generation skipped
python -m bball_app.core.pipeline.full_pipeline "$LEAGUE" --skip-training "$@"
