#!/usr/bin/env python3
"""
Backfill player metadata (headshot, position) from ESPN API to players collection.

This script:
1. Gets player_ids from current-season rosters
2. Finds which of those players are missing headshot or pos_name/pos_display_name
3. Fetches metadata from ESPN API for those players
4. Updates the players collection

Usage:
    python -m bball_app.cli.scripts.backfill_player_metadata <league>
    python -m bball_app.cli.scripts.backfill_player_metadata nba
    python -m bball_app.cli.scripts.backfill_player_metadata cbb
    python -m bball_app.cli.scripts.backfill_player_metadata cbb --dry-run
    python -m bball_app.cli.scripts.backfill_player_metadata cbb --limit 100
    python -m bball_app.cli.scripts.backfill_player_metadata nba --season 2024-2025
"""

import argparse
import sys
import os
import requests
import time

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo
from bball_app.core.league_config import load_league_config


def fetch_player_from_espn(player_id: str, sport_path: str) -> dict:
    """
    Fetch player metadata from ESPN API.

    Args:
        player_id: ESPN player ID
        sport_path: ESPN sport path (e.g., 'nba', 'mens-college-basketball')

    Returns:
        Dict with headshot, pos_name, pos_display_name (or empty dict on failure)
    """
    url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/{sport_path}/athletes/{player_id}"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}

        data = resp.json()
        athlete = data.get('athlete', data)  # Handle both response formats

        result = {}

        # Extract headshot
        headshot = athlete.get('headshot', {})
        if headshot and headshot.get('href'):
            result['headshot'] = headshot['href']

        # Extract position
        position = athlete.get('position', {})
        if position:
            if position.get('name'):
                result['pos_name'] = position['name']
            if position.get('displayName'):
                result['pos_display_name'] = position['displayName']

        # Also get player name if available
        if athlete.get('displayName'):
            result['player_name'] = athlete['displayName']

        return result

    except Exception as e:
        print(f"    Error fetching player {player_id}: {e}")
        return {}


def backfill_player_metadata(league_id: str, dry_run: bool = False, limit: int = None,
                             season: str = None):
    """
    Backfill player metadata from ESPN API for players on current-season rosters.

    Args:
        league_id: League identifier (nba, cbb, etc.)
        dry_run: If True, only show what would be done
        limit: Max players to process (for testing)
        season: Season to pull rosters from (default: current season)
    """
    print(f"Backfilling player metadata for league: {league_id}")
    print(f"Dry run: {dry_run}")
    if limit:
        print(f"Limit: {limit} players")
    print()

    # Load league config
    league = load_league_config(league_id)
    db = Mongo().db

    # Get collection names and ESPN config from league config
    players_collection = league.collections.get('players', 'nba_players')
    rosters_collection = league.collections.get('rosters', 'nba_rosters')
    sport_path = league.espn.get('sport_path', 'nba')

    # Determine season
    if not season:
        from bball_app.core.utils import get_season_from_date
        from datetime import date
        season = get_season_from_date(date.today(), league=league)

    print(f"Players collection: {players_collection}")
    print(f"Rosters collection: {rosters_collection}")
    print(f"Season: {season}")
    print(f"ESPN sport path: {sport_path}")
    print()

    # Step 1: Get all player_ids from current-season rosters
    print("Step 1: Collecting player IDs from current-season rosters...")

    roster_docs = list(db[rosters_collection].find(
        {'season': season},
        {'roster': 1, 'team': 1}
    ))
    print(f"  Found {len(roster_docs)} team rosters for season {season}")

    roster_player_ids = set()
    for doc in roster_docs:
        for entry in doc.get('roster', []):
            pid = entry.get('player_id')
            if pid:
                roster_player_ids.add(str(pid))

    print(f"  Total unique players on rosters: {len(roster_player_ids)}")

    # Step 2: Find which of those players are missing headshot or position
    print("\nStep 2: Finding rostered players missing headshot or position...")

    # Query players collection for these IDs, expanding to handle int/str
    expanded_ids = []
    for pid in roster_player_ids:
        expanded_ids.append(pid)
        if pid.isdigit():
            expanded_ids.append(int(pid))

    missing_query = {
        'player_id': {'$in': expanded_ids},
        '$or': [
            {'headshot': {'$exists': False}},
            {'headshot': None},
            {'headshot': ''},
            {'pos_name': {'$exists': False}},
            {'pos_name': None},
            {'pos_name': ''},
            {'pos_display_name': {'$exists': False}},
            {'pos_display_name': None},
            {'pos_display_name': ''},
        ]
    }

    players_to_update = list(db[players_collection].find(
        missing_query,
        {'player_id': 1, 'player_name': 1, 'headshot': 1, 'pos_name': 1, 'pos_display_name': 1}
    ))

    # Also find rostered players with no entry in players collection at all
    existing_ids = set()
    for doc in db[players_collection].find(
        {'player_id': {'$in': expanded_ids}},
        {'player_id': 1}
    ):
        existing_ids.add(str(doc.get('player_id', '')))

    missing_entirely = roster_player_ids - existing_ids
    for pid in missing_entirely:
        players_to_update.append({
            'player_id': pid,
            'player_name': 'Unknown',
            '_new': True
        })

    print(f"  Rostered players missing metadata: {len(players_to_update)}")
    if missing_entirely:
        print(f"    ({len(missing_entirely)} not in players collection at all)")

    if limit:
        players_to_update = players_to_update[:limit]
        print(f"  Limited to {len(players_to_update)} players")

    if not players_to_update:
        print("\nNo players need updating.")
        return

    # Step 2: Fetch from ESPN API and update
    print(f"\nStep 2: Fetching metadata from ESPN API for {len(players_to_update)} players...")

    updated_count = 0
    failed_count = 0
    skipped_count = 0

    for i, player in enumerate(players_to_update):
        player_id = player.get('player_id')
        player_name = player.get('player_name', 'Unknown')

        if not player_id:
            skipped_count += 1
            continue

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(players_to_update)} ({updated_count} updated, {failed_count} failed)")

        if dry_run:
            missing = []
            if not player.get('headshot'):
                missing.append('headshot')
            if not player.get('pos_name'):
                missing.append('pos_name')
            if not player.get('pos_display_name'):
                missing.append('pos_display_name')
            print(f"  [DRY RUN] Would fetch for {player_name} ({player_id}) - missing: {', '.join(missing)}")
            continue

        # Fetch from ESPN
        metadata = fetch_player_from_espn(str(player_id), sport_path)

        if not metadata:
            failed_count += 1
            continue

        # Build update with only the fields we got back
        update_data = {}
        if metadata.get('player_name'):
            update_data['player_name'] = metadata['player_name']
        if metadata.get('headshot'):
            update_data['headshot'] = metadata['headshot']
        if metadata.get('pos_name'):
            update_data['pos_name'] = metadata['pos_name']
        if metadata.get('pos_display_name'):
            update_data['pos_display_name'] = metadata['pos_display_name']

        if not update_data:
            skipped_count += 1
            continue

        db[players_collection].update_one(
            {'player_id': player_id},
            {'$set': update_data}
        )
        updated_count += 1

        # Rate limit to avoid hitting ESPN too hard
        time.sleep(0.1)

    print(f"\nDone!")
    print(f"  Updated: {updated_count}")
    print(f"  Failed (API error): {failed_count}")
    print(f"  Skipped (no useful data from API): {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description='Backfill player metadata (headshot, position) from ESPN API'
    )
    parser.add_argument('league', help='League identifier (nba, cbb, etc.)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--limit', type=int, default=None,
                       help='Maximum number of players to process')
    parser.add_argument('--season', type=str, default=None,
                       help='Season to pull rosters from (default: current season)')

    args = parser.parse_args()

    backfill_player_metadata(args.league, args.dry_run, args.limit, args.season)


if __name__ == '__main__':
    main()
