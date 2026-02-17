#!/usr/bin/env python3
"""
Backfill team_id in player_stats collections.

Adds team_id and opponent_id fields to player_stats records based on
team abbreviation mappings from the teams collection.

Usage:
    python -m bball_app.core.pipeline.backfill_team_ids cbb
    python -m bball_app.core.pipeline.backfill_team_ids cbb --dry-run
    python -m bball_app.core.pipeline.backfill_team_ids cbb --batch-size 10000
"""

import argparse
import sys
import os
import time
from typing import Dict, Optional

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.league_config import load_league_config, get_available_leagues
from bball_app.core.mongo import Mongo


def build_team_abbrev_to_id_map(db, league_config) -> Dict[str, str]:
    """
    Build mapping from team abbreviation to team_id.

    Returns:
        Dict mapping abbreviation (uppercase) -> team_id (string)
    """
    teams_coll = league_config.collections.get('teams')
    if not teams_coll:
        raise ValueError(f"No teams collection configured for {league_config.league_id}")

    mapping = {}
    for team in db[teams_coll].find({}, {'id': 1, 'abbreviation': 1}):
        abbrev = team.get('abbreviation', '').upper()
        team_id = team.get('id')
        if abbrev and team_id:
            mapping[abbrev] = str(team_id)

    return mapping


def backfill_player_stats_team_ids(
    db,
    league_config,
    batch_size: int = 5000,
    dry_run: bool = False
) -> dict:
    """
    Backfill team_id and opponent_id in player_stats collection.

    Args:
        db: MongoDB database
        league_config: League configuration
        batch_size: Number of records to update per batch
        dry_run: If True, don't actually update

    Returns:
        Stats dict with counts
    """
    player_stats_coll = league_config.collections.get('player_stats')
    if not player_stats_coll:
        raise ValueError(f"No player_stats collection configured for {league_config.league_id}")

    # Build abbreviation -> team_id mapping
    print("Building team abbreviation to ID mapping...")
    abbrev_to_id = build_team_abbrev_to_id_map(db, league_config)
    print(f"  Loaded {len(abbrev_to_id)} team mappings")

    if not abbrev_to_id:
        print("  ERROR: No team mappings found. Aborting.")
        return {'error': 'No team mappings found'}

    # Get collection
    coll = db[player_stats_coll]

    # Count records needing update
    total_missing = coll.count_documents({'team_id': {'$exists': False}})
    print(f"\nRecords missing team_id: {total_missing:,}")

    if total_missing == 0:
        print("Nothing to backfill!")
        return {'updated': 0, 'skipped': 0, 'not_found': 0}

    # Process in batches using aggregation pipeline for efficiency
    updated = 0
    skipped = 0
    not_found = 0
    not_found_teams = set()

    start_time = time.time()
    last_print = start_time

    # Use cursor to iterate through records missing team_id
    cursor = coll.find(
        {'team_id': {'$exists': False}},
        {'_id': 1, 'team': 1, 'opponent': 1}
    ).batch_size(batch_size)

    bulk_ops = []

    for doc in cursor:
        team_abbrev = (doc.get('team') or '').upper()
        opponent_abbrev = (doc.get('opponent') or '').upper()

        team_id = abbrev_to_id.get(team_abbrev)
        opponent_id = abbrev_to_id.get(opponent_abbrev)

        if not team_id:
            not_found += 1
            not_found_teams.add(team_abbrev)
            continue

        update_fields = {'team_id': team_id}
        if opponent_id:
            update_fields['opponent_id'] = opponent_id

        if not dry_run:
            bulk_ops.append({
                'update_one': {
                    'filter': {'_id': doc['_id']},
                    'update': {'$set': update_fields}
                }
            })

            # Execute bulk write when batch is full
            if len(bulk_ops) >= batch_size:
                result = coll.bulk_write([
                    __import__('pymongo').UpdateOne(op['update_one']['filter'], op['update_one']['update'])
                    for op in bulk_ops
                ])
                updated += result.modified_count
                bulk_ops = []
        else:
            updated += 1

        # Progress update
        processed = updated + not_found + skipped
        now = time.time()
        if now - last_print >= 2.0:  # Print every 2 seconds
            elapsed = now - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            remaining = (total_missing - processed) / rate if rate > 0 else 0
            print(f"  Progress: {processed:,}/{total_missing:,} ({100*processed/total_missing:.1f}%) "
                  f"- {rate:.0f}/s - ETA: {remaining/60:.1f}m", end='\r')
            last_print = now

    # Execute remaining bulk ops
    if bulk_ops and not dry_run:
        from pymongo import UpdateOne
        result = coll.bulk_write([
            UpdateOne(op['update_one']['filter'], op['update_one']['update'])
            for op in bulk_ops
        ])
        updated += result.modified_count

    elapsed = time.time() - start_time
    print(f"\n\nCompleted in {elapsed:.1f}s")
    print(f"  Updated: {updated:,}")
    print(f"  Not found (no team mapping): {not_found:,}")

    if not_found_teams:
        print(f"  Teams without mapping: {sorted(not_found_teams)[:20]}")
        if len(not_found_teams) > 20:
            print(f"    ... and {len(not_found_teams) - 20} more")

    if dry_run:
        print("\n  [DRY RUN - no changes made]")

    return {
        'updated': updated,
        'skipped': skipped,
        'not_found': not_found,
        'not_found_teams': list(not_found_teams)
    }


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Backfill team_id in player_stats collections",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "league",
        choices=available_leagues,
        help=f"League to backfill ({', '.join(available_leagues)})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Batch size for bulk updates (default: 5000)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making changes"
    )

    args = parser.parse_args()

    # Load config and connect
    league_config = load_league_config(args.league)
    mongo = Mongo()

    print(f"=== Backfill team_id for {league_config.display_name} ===\n")

    try:
        stats = backfill_player_stats_team_ids(
            db=mongo.db,
            league_config=league_config,
            batch_size=args.batch_size,
            dry_run=args.dry_run
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
