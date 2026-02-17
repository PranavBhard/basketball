#!/usr/bin/env python3
"""
Migrate positions from player_stats collection to players collection.

This script:
1. Finds all positions stored in player_stats collection
2. Copies them to the players collection
3. Removes the position fields from player_stats collection

Usage:
    python -m bball_app.cli.scripts.migrate_positions_to_players <league>
    python -m bball_app.cli.scripts.migrate_positions_to_players nba
    python -m bball_app.cli.scripts.migrate_positions_to_players cbb
    python -m bball_app.cli.scripts.migrate_positions_to_players nba --dry-run
"""

import argparse
import sys
import os

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if nba_app_dir not in sys.path:
    sys.path.insert(0, nba_app_dir)

from bball_app.core.mongo import Mongo
from bball_app.core.league_config import load_league_config


def migrate_positions(league_id: str, dry_run: bool = False):
    """
    Migrate positions from player_stats to players collection.

    Args:
        league_id: League identifier (nba, cbb, etc.)
        dry_run: If True, only show what would be done
    """
    print(f"Migrating positions for league: {league_id}")
    print(f"Dry run: {dry_run}")
    print()

    # Load league config
    league = load_league_config(league_id)
    db = Mongo().db

    # Get collection names from league config
    player_stats_collection = league.collections.get('player_stats', 'nba_player_stats')
    players_collection = league.collections.get('players', 'nba_players')

    print(f"Player stats collection: {player_stats_collection}")
    print(f"Players collection: {players_collection}")
    print()

    # Step 1: Find all unique players with positions in player_stats
    print("Step 1: Finding positions in player_stats collection...")

    pipeline = [
        {'$match': {
            '$or': [
                {'pos_name': {'$exists': True, '$ne': None}},
                {'pos_display_name': {'$exists': True, '$ne': None}}
            ]
        }},
        {'$group': {
            '_id': '$player_id',
            'pos_name': {'$first': '$pos_name'},
            'pos_display_name': {'$first': '$pos_display_name'},
            'player_name': {'$first': '$player_name'}
        }}
    ]

    players_with_positions = list(db[player_stats_collection].aggregate(pipeline))
    print(f"  Found {len(players_with_positions)} players with positions in {player_stats_collection}")

    if not players_with_positions:
        print("  No positions to migrate.")
        return

    # Step 2: Update players collection with positions
    print("\nStep 2: Updating players collection with positions...")

    updated_count = 0
    created_count = 0
    skipped_count = 0

    for player in players_with_positions:
        player_id = player['_id']
        pos_name = player.get('pos_name')
        pos_display_name = player.get('pos_display_name')
        player_name = player.get('player_name', 'Unknown')

        if dry_run:
            print(f"  [DRY RUN] Would update player {player_id} ({player_name}) with position: {pos_name}")
            updated_count += 1
        else:
            # Check if player exists in players collection
            existing = db[players_collection].find_one({'player_id': player_id})

            update_data = {}
            if pos_name:
                update_data['pos_name'] = pos_name
            if pos_display_name:
                update_data['pos_display_name'] = pos_display_name

            if not update_data:
                skipped_count += 1
                continue

            if existing:
                # Update existing player with position
                result = db[players_collection].update_one(
                    {'player_id': player_id},
                    {'$set': update_data}
                )
                if result.modified_count > 0:
                    updated_count += 1
                else:
                    skipped_count += 1  # Already had the same position
            else:
                # Player doesn't exist - create minimal entry
                db[players_collection].insert_one({
                    'player_id': player_id,
                    'player_name': player_name,
                    **update_data
                })
                created_count += 1

    print(f"  Updated: {updated_count}, Created: {created_count}, Skipped: {skipped_count}")

    # Step 3: Remove position fields from player_stats collection
    print("\nStep 3: Removing position fields from player_stats collection...")

    if dry_run:
        count = db[player_stats_collection].count_documents({
            '$or': [
                {'pos_name': {'$exists': True}},
                {'pos_display_name': {'$exists': True}}
            ]
        })
        print(f"  [DRY RUN] Would remove position fields from {count} documents")
    else:
        result = db[player_stats_collection].update_many(
            {'$or': [
                {'pos_name': {'$exists': True}},
                {'pos_display_name': {'$exists': True}}
            ]},
            {'$unset': {'pos_name': '', 'pos_display_name': ''}}
        )
        print(f"  Removed position fields from {result.modified_count} documents")

    print("\nMigration complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Migrate positions from player_stats to players collection'
    )
    parser.add_argument(
        'league',
        help='League ID (nba, cbb, etc.)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )

    args = parser.parse_args()

    migrate_positions(args.league, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
