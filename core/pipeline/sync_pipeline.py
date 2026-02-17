#!/usr/bin/env python3
"""
Data sync pipeline for pulling ESPN data into MongoDB.

Pulls/upserts ESPN data (games, players, venues, rosters, teams) into MongoDB
with flexible date filtering and data type selection.

Usage:
    python -m bball_app.core.pipeline.sync_pipeline nba --date 2024-01-15
    python -m bball_app.core.pipeline.sync_pipeline cbb --date-range 2024-01-01,2024-01-31
    python -m bball_app.core.pipeline.sync_pipeline nba --season 2024-2025
    python -m bball_app.core.pipeline.sync_pipeline nba --season 2024-2025 --with-injuries
"""

import argparse
import subprocess
import sys
import os
import time
from datetime import datetime, date, timedelta
from typing import List, Optional, Set

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.league_config import load_league_config, get_available_leagues, LeagueConfig
from bball_app.core.mongo import Mongo


# Valid data types for sync
VALID_DATA_TYPES = {'games', 'players', 'player_stats', 'venues', 'rosters', 'teams'}


def parse_date(date_str: str) -> date:
    """Parse date string in YYYY-MM-DD format."""
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}. Use YYYY-MM-DD")


def parse_date_range(date_str: str) -> tuple:
    """Parse date range string like 'YYYY-MM-DD,YYYY-MM-DD' and return (start, end)."""
    parts = date_str.split(',')
    if len(parts) != 2:
        raise ValueError(f"Invalid date range format: {date_str}. Use YYYY-MM-DD,YYYY-MM-DD")

    start = parse_date(parts[0].strip())
    end = parse_date(parts[1].strip())

    if start > end:
        raise ValueError(f"Start date must be before end date: {start} > {end}")

    return start, end


def parse_season(season_str: str, league_config: LeagueConfig) -> tuple:
    """
    Parse season string like 'YYYY-YYYY' and return (start_date, end_date).

    Uses league config for season start/end month/day.
    """
    parts = season_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid season format: {season_str}. Use YYYY-YYYY (e.g., 2024-2025)")

    try:
        start_year = int(parts[0])
        end_year = int(parts[1])
    except ValueError:
        raise ValueError(f"Invalid season years: {season_str}. Use YYYY-YYYY")

    if end_year != start_year + 1:
        raise ValueError(f"Season must span consecutive years: {season_str}")

    # Build dates from league config
    start_date = date(
        start_year,
        league_config.season_start_month,
        league_config.season_start_day
    )
    end_date = date(
        end_year,
        league_config.season_end_month,
        league_config.season_end_day
    )

    # For current season, cap at today if before season end
    today = date.today()
    if end_date > today:
        end_date = today

    return start_date, end_date


def parse_data_types(types_str: str) -> Set[str]:
    """Parse comma-separated data types."""
    types = set()
    for t in types_str.split(','):
        t = t.strip().lower()
        if t not in VALID_DATA_TYPES:
            raise ValueError(f"Invalid data type: {t}. Valid types: {', '.join(sorted(VALID_DATA_TYPES))}")
        types.add(t)
    return types


def run_espn_pull(
    league_config: LeagueConfig,
    start_date: date,
    end_date: date,
    dry_run: bool = False,
    verbose: bool = False,
    progress_callback=None
) -> dict:
    """
    Run ESPN data pull for a date range.

    Args:
        progress_callback: Optional callable(percent: int, message: str) for progress updates.

    Returns dict with statistics.
    """
    from bball_app.core.services.espn_sync import fetch_and_save_games

    mongo = Mongo()
    try:
        stats = fetch_and_save_games(
            mongo.db, league_config,
            start_date, end_date,
            dry_run=dry_run,
            verbose=verbose,
            progress_callback=progress_callback
        )
    except Exception as e:
        stats = {
            'games_processed': 0,
            'players_processed': 0,
            'success': False,
            'error': str(e)
        }

    return stats


def run_post_processing(
    league_config: LeagueConfig,
    data_types: Set[str],
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Run post-processing steps (venues, teams, players).

    Returns dict with step results.
    """
    from bball_app.core.services.espn_sync import refresh_venues, refresh_players, refresh_teams

    mongo = Mongo()
    results = {}

    if 'venues' in data_types:
        print("  Refreshing venues...")
        try:
            result = refresh_venues(mongo.db, league_config, dry_run=dry_run)
            results['venues'] = {'success': result.get('success', True)}
        except Exception as e:
            results['venues'] = {'success': False, 'error': str(e)[:200]}

    if 'teams' in data_types:
        print("  Refreshing teams metadata...")
        try:
            result = refresh_teams(mongo.db, league_config, dry_run=dry_run)
            results['teams'] = {'success': result.get('success', True)}
        except Exception as e:
            results['teams'] = {'success': False, 'error': str(e)[:200]}

    if 'players' in data_types:
        print("  Refreshing players metadata...")
        try:
            result = refresh_players(mongo.db, league_config, dry_run=dry_run)
            results['players'] = {'success': result.get('success', True)}
        except Exception as e:
            results['players'] = {'success': False, 'error': str(e)[:200]}

    return results


def run_injuries(
    league_config: LeagueConfig,
    start_date: date = None,
    end_date: date = None,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Run injury computation.

    Returns dict with statistics.
    """
    print("  Computing injuries...")

    cmd = [
        sys.executable, "-m", "bball_app.core.pipeline.injuries_pipeline",
        league_config.league_id
    ]

    if start_date and end_date:
        cmd.extend(["--date-range", f"{start_date.strftime('%Y-%m-%d')},{end_date.strftime('%Y-%m-%d')}"])

    if dry_run:
        cmd.append("--dry-run")

    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(
            cmd,
            cwd=nba_app_dir,
            env={**os.environ, "PYTHONPATH": project_root},
            capture_output=True,
            text=True,
            check=True
        )
        if verbose:
            for line in result.stdout.split('\n')[:20]:
                print(f"    {line}")
        return {'success': True}
    except subprocess.CalledProcessError as e:
        return {'success': False, 'error': e.stderr[:200] if e.stderr else str(e)}


def run_elo_cache(
    league_config: LeagueConfig,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Run ELO cache refresh.

    Returns dict with result.
    """
    from bball_app.core.stats.elo_cache import EloCache

    print("  Caching ELO ratings...")

    if dry_run:
        print(f"    [DRY RUN] Would compute and cache ELO ratings")
        return {'success': True, 'dry_run': True}

    try:
        mongo = Mongo()
        elo_cache = EloCache(mongo.db, league=league_config)
        stats = elo_cache.compute_and_cache_all()
        if verbose:
            print(f"    Done - {stats.get('ratings_cached', 0)} ratings cached")
        return {'success': True, **stats}
    except Exception as e:
        return {'success': False, 'error': str(e)[:200]}


def run_venue_geocoding(
    league_config: LeagueConfig,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Geocode venues that don't have location data.

    Delegates to geocode_missing_venues() in core/services/espn_sync.py.

    Returns dict with statistics.
    """
    from bball_app.core.services.espn_sync import geocode_missing_venues

    print("  Geocoding venues without location...")

    mongo = Mongo()
    result = geocode_missing_venues(mongo.db, league_config, dry_run=dry_run)

    geocoded = result.get('geocoded', 0)
    failed = result.get('failed', 0)
    already = result.get('already_have_location', 0)

    if verbose:
        print(f"    Already had location: {already}")
        print(f"    Geocoded: {geocoded}, Failed: {failed}")

    return {
        'success': True,
        **result,
    }


def run_sync_pipeline(
    league_config: LeagueConfig,
    start_date: date,
    end_date: date,
    data_types: Set[str] = None,
    skip_types: Set[str] = None,
    with_injuries: bool = False,
    with_elo: bool = False,
    with_geocoding: bool = False,
    workers: int = 4,
    dry_run: bool = False,
    verbose: bool = False,
    progress_callback=None
) -> dict:
    """
    Run the full sync pipeline.

    Args:
        league_config: League configuration
        start_date: Start date for sync
        end_date: End date for sync
        data_types: Set of data types to sync (default: all)
        skip_types: Set of data types to skip
        with_injuries: Compute injuries after sync
        with_elo: Update ELO cache after sync
        with_geocoding: Geocode venues without location data
        workers: Number of parallel workers
        dry_run: Preview without database changes
        verbose: Detailed output
        progress_callback: Optional callable(percent: int, message: str) for progress updates

    Returns:
        Dict with pipeline statistics
    """
    pipeline_start = time.time()

    # Determine active data types
    active_types = data_types or VALID_DATA_TYPES.copy()
    if skip_types:
        active_types = active_types - skip_types

    # Calculate date range info
    days = (end_date - start_date).days + 1

    print(f"\n{'='*60}")
    print(f"  {league_config.display_name} Data Sync")
    print(f"{'='*60}")
    print(f"  Date range:      {start_date} to {end_date} ({days} days)")
    print(f"  Data types:      {', '.join(sorted(active_types))}")
    print(f"  With injuries:   {with_injuries}")
    print(f"  With ELO:        {with_elo}")
    print(f"  With geocoding:  {with_geocoding}")
    print(f"  Dry run:         {dry_run}")
    print(f"{'='*60}\n")

    results = {
        'steps': {},
        'success': True,
        'total_time': 0
    }

    # Helper to report progress
    def report_progress(percent, message):
        if progress_callback:
            progress_callback(percent, message)

    # Determine active steps and assign progress ranges
    steps_active = []
    if 'games' in active_types or 'player_stats' in active_types:
        steps_active.append('espn_pull')
    if active_types & {'venues', 'players'}:
        steps_active.append('post_processing')
    if with_geocoding and 'venues' in active_types:
        steps_active.append('geocoding')
    if with_injuries:
        steps_active.append('injuries')
    if with_elo:
        steps_active.append('elo')

    # Weight ESPN pull heavier since it's typically the longest step
    step_weights = {
        'espn_pull': 60,
        'post_processing': 20,
        'geocoding': 10,
        'injuries': 10,
        'elo': 10,
    }
    total_weight = sum(step_weights.get(s, 10) for s in steps_active) or 1
    cumulative = 0
    step_ranges = {}
    for s in steps_active:
        w = step_weights.get(s, 10)
        start_pct = int(cumulative * 100 / total_weight)
        cumulative += w
        end_pct = int(cumulative * 100 / total_weight)
        step_ranges[s] = (start_pct, end_pct)

    # Step 1: ESPN data pull (games, player_stats)
    if 'games' in active_types or 'player_stats' in active_types:
        step_name = "ESPN Data Pull"
        print(f"[1] {step_name}...")
        espn_range = step_ranges.get('espn_pull', (0, 70))
        report_progress(espn_range[0], 'Pulling games and player stats from ESPN...')

        # Create sub-callback that maps ESPN pull progress (0-100%) to this step's range
        def espn_progress_callback(pct, msg):
            # Map 0-100 from ESPN pull to espn_range[0]-espn_range[1]
            mapped_pct = espn_range[0] + int(pct * (espn_range[1] - espn_range[0]) / 100)
            report_progress(mapped_pct, msg)

        step_start = time.time()
        espn_stats = run_espn_pull(
            league_config,
            start_date,
            end_date,
            dry_run=dry_run,
            verbose=verbose,
            progress_callback=espn_progress_callback if progress_callback else None
        )
        step_time = time.time() - step_start

        results['steps']['espn_pull'] = {
            **espn_stats,
            'time': step_time
        }

        report_progress(step_ranges.get('espn_pull', (0, 70))[1], f"ESPN pull done: {espn_stats['games_processed']} games, {espn_stats['players_processed']} players")

        print(f"    Games: {espn_stats['games_processed']:,}")
        print(f"    Players: {espn_stats['players_processed']:,}")
        print(f"    Time: {step_time:.1f}s")

        if not espn_stats['success']:
            print(f"    Warning: {espn_stats.get('error', 'Unknown error')}")
            results['success'] = False

    # Step 2: Post-processing (venues, players metadata)
    post_types = active_types & {'venues', 'players'}
    if post_types:
        step_name = "Post-Processing"
        print(f"\n[2] {step_name}...")
        report_progress(step_ranges.get('post_processing', (70, 90))[0], 'Post-processing: refreshing venues and player metadata...')

        step_start = time.time()
        post_stats = run_post_processing(
            league_config,
            post_types,
            dry_run=dry_run,
            verbose=verbose
        )
        step_time = time.time() - step_start

        results['steps']['post_processing'] = {
            **post_stats,
            'time': step_time
        }

        report_progress(step_ranges.get('post_processing', (70, 90))[1], 'Post-processing complete')

        for dtype, status in post_stats.items():
            status_str = "OK" if status.get('success') else f"FAILED: {status.get('error', 'Unknown')}"
            print(f"    {dtype}: {status_str}")
        print(f"    Time: {step_time:.1f}s")

    # Step 3: Venue Geocoding (optional)
    if with_geocoding and 'venues' in active_types:
        step_name = "Venue Geocoding"
        print(f"\n[3] {step_name}...")
        report_progress(step_ranges.get('geocoding', (0, 0))[0], 'Geocoding venues...')

        step_start = time.time()
        geocoding_stats = run_venue_geocoding(
            league_config,
            dry_run=dry_run,
            verbose=verbose
        )
        step_time = time.time() - step_start

        results['steps']['venue_geocoding'] = {
            **geocoding_stats,
            'time': step_time
        }

        report_progress(step_ranges.get('geocoding', (0, 0))[1], 'Geocoding complete')
        print(f"    Time: {step_time:.1f}s")

    # Step 4: Injuries (optional)
    if with_injuries:
        step_name = "Injury Computation"
        print(f"\n[4] {step_name}...")
        report_progress(step_ranges.get('injuries', (0, 0))[0], 'Computing injuries...')

        step_start = time.time()
        injury_stats = run_injuries(
            league_config,
            start_date=start_date,
            end_date=end_date,
            dry_run=dry_run,
            verbose=verbose
        )
        step_time = time.time() - step_start

        results['steps']['injuries'] = {
            **injury_stats,
            'time': step_time
        }

        report_progress(step_ranges.get('injuries', (0, 0))[1], 'Injuries complete')
        status_str = "OK" if injury_stats.get('success') else f"FAILED: {injury_stats.get('error', 'Unknown')}"
        print(f"    Status: {status_str}")
        print(f"    Time: {step_time:.1f}s")

    # Step 5: ELO (optional)
    if with_elo:
        step_name = "ELO Cache"
        print(f"\n[5] {step_name}...")
        report_progress(step_ranges.get('elo', (0, 0))[0], 'Caching ELO ratings...')

        step_start = time.time()
        elo_stats = run_elo_cache(
            league_config,
            dry_run=dry_run,
            verbose=verbose
        )
        step_time = time.time() - step_start

        results['steps']['elo_cache'] = {
            **elo_stats,
            'time': step_time
        }

        report_progress(step_ranges.get('elo', (0, 0))[1], 'ELO cache complete')
        status_str = "OK" if elo_stats.get('success') else f"FAILED: {elo_stats.get('error', 'Unknown')}"
        print(f"    Status: {status_str}")
        print(f"    Time: {step_time:.1f}s")

    # Summary
    total_time = time.time() - pipeline_start
    results['total_time'] = total_time

    print(f"\n{'='*60}")
    print(f"  Sync Complete!")
    print(f"{'='*60}")
    print(f"  Total time: {total_time:.1f}s")
    if dry_run:
        print(f"  Mode: DRY RUN (no changes made)")
    print(f"{'='*60}\n")

    report_progress(100, 'Sync pipeline complete')

    return results


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Pull/upsert ESPN data into MongoDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m bball_app.core.pipeline.sync_pipeline nba --date 2024-01-15
    python -m bball_app.core.pipeline.sync_pipeline cbb --date-range 2024-01-01,2024-01-31
    python -m bball_app.core.pipeline.sync_pipeline nba --season 2024-2025
    python -m bball_app.core.pipeline.sync_pipeline nba --season 2024-2025 --with-injuries
    python -m bball_app.core.pipeline.sync_pipeline nba --only games,player_stats --dry-run

Data Types:
    games           Game box scores from ESPN scoreboard
    players         Player metadata
    player_stats    Individual player game stats
    venues          Arena/venue information
    rosters         Team rosters
    teams           Team metadata
        """
    )

    # Positional argument
    parser.add_argument(
        "league",
        choices=available_leagues,
        help=f"League to sync ({', '.join(available_leagues)})"
    )

    # Date filtering (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument(
        "--date",
        type=str,
        help="Single date (YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--date-range",
        type=str,
        help="Date range (YYYY-MM-DD,YYYY-MM-DD)"
    )
    date_group.add_argument(
        "--season",
        type=str,
        help="Entire season (YYYY-YYYY)"
    )

    # Data selection
    parser.add_argument(
        "--only",
        type=str,
        help="Only sync specific data types (comma-separated)"
    )
    parser.add_argument(
        "--skip",
        type=str,
        help="Skip specific data types (comma-separated)"
    )

    # Post-processing
    parser.add_argument(
        "--with-injuries",
        action="store_true",
        help="Compute injuries after sync"
    )
    parser.add_argument(
        "--with-elo",
        action="store_true",
        help="Update ELO cache after sync"
    )
    parser.add_argument(
        "--with-geocoding",
        action="store_true",
        help="Geocode venues without location data (uses Nominatim, rate-limited)"
    )

    # Common options
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel workers (default: 4)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without database changes"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Detailed output"
    )

    args = parser.parse_args()

    # Load league config
    league_config = load_league_config(args.league)

    # Parse date arguments
    if args.date:
        start_date = parse_date(args.date)
        end_date = start_date
    elif args.date_range:
        start_date, end_date = parse_date_range(args.date_range)
    elif args.season:
        start_date, end_date = parse_season(args.season, league_config)

    # Parse data types
    data_types = None
    skip_types = None

    if args.only:
        data_types = parse_data_types(args.only)

    if args.skip:
        skip_types = parse_data_types(args.skip)

    # Run pipeline
    try:
        results = run_sync_pipeline(
            league_config=league_config,
            start_date=start_date,
            end_date=end_date,
            data_types=data_types,
            skip_types=skip_types,
            with_injuries=args.with_injuries,
            with_elo=args.with_elo,
            with_geocoding=args.with_geocoding,
            workers=args.workers,
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        return 0 if results['success'] else 1

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
