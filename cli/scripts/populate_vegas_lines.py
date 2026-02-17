#!/usr/bin/env python3
"""
Populate stats_nba collection with Vegas betting lines from CSV.

Reads nba_2008-2025.csv and adds a 'vegas' object to each game document with:
- home_ML: Home team moneyline
- away_ML: Away team moneyline
- home_spread: Spread from home perspective (negative = home favored)
- away_spread: Spread from away perspective (negative = away favored)
- OU: Over/under total

Uses bulk updates per season for performance.

Usage:
    ./cli/scripts/populate_vegas.sh
    # Or directly:
    source venv/bin/activate
    PYTHONPATH=/Users/pranav/Documents/NBA python cli/scripts/populate_vegas_lines.py
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import sys

from pymongo import UpdateOne
from bball_app.core.mongo import Mongo


# Team abbreviation mapping: CSV (lowercase) -> DB (uppercase)
# Most just need uppercasing, but list explicitly for clarity
TEAM_MAP = {
    'atl': 'ATL',
    'bkn': 'BKN',
    'bos': 'BOS',
    'cha': 'CHA',
    'chi': 'CHI',
    'cle': 'CLE',
    'dal': 'DAL',
    'den': 'DEN',
    'det': 'DET',
    'gs': 'GS',
    'hou': 'HOU',
    'ind': 'IND',
    'lac': 'LAC',
    'lal': 'LAL',
    'mem': 'MEM',
    'mia': 'MIA',
    'mil': 'MIL',
    'min': 'MIN',
    'no': 'NO',
    'ny': 'NY',
    'okc': 'OKC',
    'orl': 'ORL',
    'phi': 'PHI',
    'phx': 'PHX',
    'por': 'POR',
    'sa': 'SA',
    'sac': 'SAC',
    'tor': 'TOR',
    'utah': 'UTAH',
    'wsh': 'WSH',
}

# Historical team relocations/rebrands
# CSV uses current names, but DB has historical names before these dates
HISTORICAL_TEAMS = {
    # BKN was NJ (New Jersey Nets) before 2012-10-13
    'bkn': ('NJ', '2012-10-13'),
    # OKC was SEA (Seattle SuperSonics) before 2008-10-08
    'okc': ('SEA', '2008-10-08'),
}


def get_db_team(csv_team: str, game_date: str) -> str:
    """
    Map CSV team abbreviation to DB abbreviation, handling historical relocations.
    """
    csv_team = csv_team.lower()

    # Check if this team has a historical name
    if csv_team in HISTORICAL_TEAMS:
        historical_name, cutoff_date = HISTORICAL_TEAMS[csv_team]
        if game_date < cutoff_date:
            return historical_name

    # Use standard mapping
    return TEAM_MAP.get(csv_team, csv_team.upper())


def build_vegas_object(row: pd.Series) -> Optional[Dict]:
    """
    Build vegas object from CSV row.

    Returns None if essential data is missing.
    """
    # Get spread - convert to home perspective
    spread = row.get('spread')
    whos_favored = row.get('whos_favored')

    if pd.isna(spread):
        home_spread = None
        away_spread = None
    else:
        # If home is favored, home_spread is negative (they give points)
        # If away is favored, home_spread is positive (they get points)
        if whos_favored == 'home':
            home_spread = -float(spread)
        else:  # away favored
            home_spread = float(spread)
        away_spread = -home_spread

    # Get moneylines
    home_ml = row.get('moneyline_home')
    away_ml = row.get('moneyline_away')

    if pd.isna(home_ml):
        home_ml = None
    else:
        home_ml = int(home_ml)

    if pd.isna(away_ml):
        away_ml = None
    else:
        away_ml = int(away_ml)

    # Get over/under
    ou = row.get('total')
    if pd.isna(ou):
        ou = None
    else:
        ou = float(ou)

    # Only return if we have at least some data
    if home_spread is None and home_ml is None and ou is None:
        return None

    return {
        'home_ML': home_ml,
        'away_ML': away_ml,
        'home_spread': home_spread,
        'away_spread': away_spread,
        'OU': ou,
    }


def build_bulk_operations(df_season: pd.DataFrame, season: int) -> Tuple[List[UpdateOne], int]:
    """
    Build bulk update operations for a season's worth of games.
    No DB lookups - just queue UpdateOne ops and let bulk_write handle matching.

    Returns: (operations, skipped_no_data)
    """
    operations = []
    skipped_no_data = 0
    total_rows = len(df_season)

    for _, row in df_season.iterrows():
        # Get date first (needed for historical team mapping)
        game_date = row['date']  # Already in YYYY-MM-DD format

        # Map team abbreviations (handles historical relocations)
        home_csv = row['home']
        away_csv = row['away']

        home_db = get_db_team(home_csv, game_date)
        away_db = get_db_team(away_csv, game_date)

        if not home_db or not away_db:
            continue

        # Build vegas object
        vegas = build_vegas_object(row)
        if vegas is None:
            skipped_no_data += 1
            continue

        # Queue update - MongoDB will match during bulk_write
        operations.append(
            UpdateOne(
                {
                    'date': game_date,
                    'homeTeam.name': home_db,
                    'awayTeam.name': away_db,
                },
                {'$set': {'vegas': vegas}}
            )
        )

    return operations, skipped_no_data


def main():
    print("=" * 60)
    print("Populating Vegas Lines from CSV")
    print("=" * 60)

    # Load CSV
    csv_path = '/Users/pranav/Documents/NBA/nba_2008-2025.csv'
    print(f"\n[LOAD] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[LOAD] Loaded {len(df):,} rows")

    # Connect to MongoDB
    print(f"\n[DB] Connecting to MongoDB...")
    mongo = Mongo()
    db = mongo.db
    stats_nba = db.stats_nba

    # Check current state
    existing_vegas = stats_nba.count_documents({'vegas': {'$exists': True}})
    total_games = stats_nba.count_documents({})
    print(f"[DB] Connected. Collection has {total_games:,} games, {existing_vegas:,} already have vegas data")

    # Stats
    total_matched = 0
    total_not_found = 0
    total_updated = 0
    total_skipped = 0
    total_errors = 0

    # Get unique seasons and process each
    seasons = sorted(df['season'].unique())
    num_seasons = len(seasons)
    print(f"\n[PLAN] Will process {num_seasons} seasons: {seasons[0]} -> {seasons[-1]}")
    print(f"[PLAN] Strategy: Build UpdateOne ops per season, then bulk_write(ordered=False)")
    print("=" * 60)

    for season_idx, season in enumerate(seasons, 1):
        df_season = df[df['season'] == season]

        try:
            # Build bulk operations (no DB lookups, just queuing)
            operations, skipped = build_bulk_operations(df_season, season)
            total_skipped += skipped

            if operations:
                # Execute bulk write - MongoDB handles matching
                print(f"[{season_idx:>2}/{num_seasons}] {season}: {len(operations):,} ops -> ", end="", flush=True)
                result = stats_nba.bulk_write(operations, ordered=False)
                matched = result.matched_count
                modified = result.modified_count
                not_found = len(operations) - matched

                total_matched += matched
                total_updated += modified
                total_not_found += not_found

                print(f"{matched:,} matched, {modified:,} modified, {not_found:,} not found")
            else:
                print(f"[{season_idx:>2}/{num_seasons}] {season}: No ops (all skipped)")

        except Exception as e:
            total_errors += 1
            print(f"[{season_idx:>2}/{num_seasons}] {season}: ERROR - {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"CSV rows processed:    {len(df):,}")
    print(f"Games found in DB:     {total_matched:,}")
    print(f"Games updated:         {total_updated:,}")
    print(f"Games not in DB:       {total_not_found:,}")
    print(f"Skipped (no vegas):    {total_skipped:,}")
    print(f"Errors:                {total_errors:,}")

    # Verify final state
    print("\n[VERIFY] Checking final state...")
    final_vegas_count = stats_nba.count_documents({'vegas': {'$exists': True}})
    print(f"[VERIFY] Games with 'vegas' field: {final_vegas_count:,} (was {existing_vegas:,}, added {final_vegas_count - existing_vegas:,})")

    # Show sample
    sample = stats_nba.find_one({'vegas': {'$exists': True}}, sort=[('date', -1)])
    if sample:
        print(f"\n[SAMPLE] Most recent game with vegas data:")
        print(f"         {sample.get('date')} - {sample.get('awayTeam', {}).get('name')} @ {sample.get('homeTeam', {}).get('name')}")
        print(f"         Vegas: {sample.get('vegas')}")


if __name__ == '__main__':
    main()
