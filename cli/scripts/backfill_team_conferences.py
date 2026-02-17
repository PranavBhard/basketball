#!/usr/bin/env python3
"""
Backfill conference data for teams from ESPN groups API.

Thin CLI wrapper around core.services.espn_sync.refresh_team_conferences().

Usage:
    python -m bball_app.cli.scripts.backfill_team_conferences <league>
    python -m bball_app.cli.scripts.backfill_team_conferences wcbb
    python -m bball_app.cli.scripts.backfill_team_conferences cbb --dry-run
"""

import argparse
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo
from bball_app.core.league_config import load_league_config
from bball_app.core.services.espn_sync import refresh_team_conferences


def backfill_team_conferences(league_id: str, dry_run: bool = False):
    """Backfill conference field on team documents from ESPN groups API."""
    league = load_league_config(league_id)
    db = Mongo().db

    teams_collection = league.collections.get('teams', f'{league_id}_teams')
    before = db[teams_collection].count_documents({'conference': {'$exists': True}})
    print(f"Teams with conference before: {before}")
    print()

    result = refresh_team_conferences(db, league, dry_run=dry_run)

    print()
    after = db[teams_collection].count_documents({'conference': {'$exists': True}})
    print(f"Teams with conference after:  {after}")
    print(f"Updated: {result.get('updated', 0)}")
    print(f"Conferences found: {result.get('conferences', 0)}")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill conference data for teams from ESPN groups API'
    )
    parser.add_argument('league', help='League identifier (nba, cbb, wcbb, etc.)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')

    args = parser.parse_args()

    backfill_team_conferences(args.league, args.dry_run)


if __name__ == '__main__':
    main()
