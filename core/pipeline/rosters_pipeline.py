"""
Rosters Pipeline - Build team rosters from player game stats.

Standalone entry point for roster generation. Can be run for all seasons
or a specific season.

Usage:
    python -m bball_app.core.pipeline.rosters_pipeline <league> [options]
"""

import argparse
import logging
import sys
import time
from datetime import datetime

from bball_app.core.league_config import load_league_config, get_available_leagues
from bball_app.core.mongo import Mongo
from bball_app.core.services.roster_service import build_rosters

logger = logging.getLogger(__name__)


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(description="Build team rosters from player game data")
    parser.add_argument("league", choices=available_leagues,
                       help="League to build rosters for")
    parser.add_argument("--season", type=str, default=None,
                       help="Specific season (e.g. 2024-2025). Default: all seasons.")
    parser.add_argument("--dry-run", action="store_true",
                       help="Preview without writing to database")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Show detailed output")
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    league_config = load_league_config(args.league)
    mongo = Mongo()
    db = mongo.db

    print(f"\n{'=' * 60}")
    print(f"  Roster Build — {league_config.display_name}")
    print(f"{'=' * 60}")

    start = time.time()

    if args.season:
        # Single season
        seasons = [args.season]
    else:
        # All seasons
        current_year = datetime.now().year
        current_month = datetime.now().month
        end_year = current_year + 1 if current_month > 8 else current_year
        seasons = [f"{y}-{y + 1}" for y in range(league_config.start_year, end_year)]

    print(f"  Seasons: {len(seasons)}")
    if args.dry_run:
        print(f"  Mode:    DRY RUN")
    print()

    total_teams = 0
    for season in seasons:
        count = build_rosters(db, season, league=league_config, dry_run=args.dry_run)
        if count > 0:
            print(f"  {season}: {count} teams")
        total_teams += count

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    action = "Would build" if args.dry_run else "Built"
    print(f"  {action} {total_teams} team-season rosters in {elapsed:.1f}s")
    print(f"{'=' * 60}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
