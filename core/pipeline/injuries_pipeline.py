#!/usr/bin/env python3
"""
Injury computation pipeline with parallel season processing.

Computes and updates injured_players for games based on roster vs played analysis.

Usage:
    python -m bball_app.core.pipeline.injuries_pipeline nba
    python -m bball_app.core.pipeline.injuries_pipeline cbb --workers 4
    python -m bball_app.core.pipeline.injuries_pipeline cbb --season 2024-2025
    python -m bball_app.core.pipeline.injuries_pipeline nba --date-range 2024-01-01,2024-01-31
"""

import argparse
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.league_config import load_league_config, get_available_leagues, LeagueConfig
from bball_app.core.mongo import Mongo
from bball_app.core.injury_manager import InjuryManager


class Status(Enum):
    PENDING = "⏳"
    RUNNING = "🔄"
    SUCCESS = "✅"
    FAILED = "❌"
    SKIPPED = "⏭️"


@dataclass
class SeasonProgress:
    season: str
    status: Status = Status.PENDING
    phase: str = "waiting"
    games_total: int = 0
    games_processed: int = 0
    games_updated: int = 0
    injured_count: int = 0
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def elapsed(self) -> str:
        if self.start_time is None:
            return "--:--"
        end = self.end_time or time.time()
        elapsed = int(end - self.start_time)
        mins, secs = divmod(elapsed, 60)
        return f"{mins:02d}:{secs:02d}"

    @property
    def progress_pct(self) -> str:
        if self.games_total == 0:
            return "0%"
        return f"{100 * self.games_processed // self.games_total}%"


@dataclass
class PipelineState:
    seasons: Dict[str, SeasonProgress] = field(default_factory=dict)
    overall_phase: str = "Initializing"
    league: str = "nba"
    league_display: str = "NBA"
    lock: threading.Lock = field(default_factory=threading.Lock)
    build_phase_done: bool = False
    build_phase_records: int = 0

    def update_season(self, season: str, **kwargs):
        with self.lock:
            if season in self.seasons:
                for key, value in kwargs.items():
                    setattr(self.seasons[season], key, value)


def clear_screen():
    """Clear terminal screen."""
    print("\033[2J\033[H", end="")


def render_progress(state: PipelineState):
    """Render progress display."""
    clear_screen()

    print("=" * 90)
    print(f"  {state.league_display} Injury Pipeline - {state.overall_phase}")
    print("=" * 90)

    if state.build_phase_done:
        print(f"  Player maps built from {state.build_phase_records:,} records")
        print("-" * 90)

    print(f"{'Season':<14} {'Status':<8} {'Phase':<18} {'Progress':<12} {'Updated':<10} {'Injured':<10} {'Time':<8}")
    print("-" * 90)

    with state.lock:
        for season in sorted(state.seasons.keys(), reverse=True):
            prog = state.seasons[season]
            status_icon = prog.status.value

            if prog.games_total > 0:
                progress_str = f"{prog.games_processed}/{prog.games_total} ({prog.progress_pct})"
            else:
                progress_str = "-"

            updated_str = str(prog.games_updated) if prog.games_updated > 0 else "-"
            injured_str = str(prog.injured_count) if prog.injured_count > 0 else "-"

            print(f"{prog.season:<14} {status_icon:<8} {prog.phase:<18} {progress_str:<12} {updated_str:<10} {injured_str:<10} {prog.elapsed:<8}")

    # Show errors at the bottom
    errors = [(s, p.error) for s, p in state.seasons.items() if p.error]
    if errors:
        print("-" * 90)
        print("  Errors:")
        for season, error in sorted(errors, reverse=True)[:5]:  # Show up to 5 errors
            error_short = error[:70] + "..." if len(error) > 70 else error
            print(f"    {season}: {error_short}")

    print("=" * 90)


def progress_monitor(state: PipelineState, stop_event: threading.Event):
    """Background thread to update progress display."""
    while not stop_event.is_set():
        render_progress(state)
        time.sleep(0.5)
    # Final render
    render_progress(state)


def parse_date(date_str: str) -> str:
    """Parse date string in YYYY-MM-DD format and validate."""
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
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


def get_seasons_from_db(db, league_config: LeagueConfig) -> List[str]:
    """Get all seasons from the games collection."""
    seasons = db.games.distinct('season')
    return sorted([s for s in seasons if s], reverse=True)


def process_season(
    season: str,
    state: PipelineState,
    league_config: LeagueConfig,
    injury_manager: InjuryManager,
    db,
    dry_run: bool = False
) -> dict:
    """Process injuries for a single season."""
    state.update_season(season, status=Status.RUNNING, phase="fetching games", start_time=time.time())

    try:
        # Get games for this season
        query = {
            'game_id': {'$exists': True, '$ne': None},
            'homeWon': {'$exists': True},
            'homeTeam.points': {'$exists': True},
            'season': season
        }

        games = list(db.games.find(
            query,
            {'game_id': 1, 'date': 1, 'homeTeam.name': 1, 'awayTeam.name': 1, 'season': 1}
        ))

        state.update_season(season, games_total=len(games), phase="computing")

        if not games:
            state.update_season(
                season,
                status=Status.SKIPPED,
                phase="no games",
                end_time=time.time()
            )
            return {'games_processed': 0, 'games_updated': 0, 'injured': 0}

        # Process games
        games_updated = 0
        total_injured = 0

        for i, game in enumerate(games):
            game_id = game['game_id']

            # Compute injuries using the correct method
            home_injured, away_injured = injury_manager.compute_injuries_for_game(game)

            injured_count = len(home_injured) + len(away_injured)
            total_injured += injured_count

            if not dry_run:
                # Update the game document (use correct field paths)
                db.games.update_one(
                    {'game_id': game_id},
                    {'$set': {
                        'homeTeam.injured_players': home_injured,
                        'awayTeam.injured_players': away_injured
                    }}
                )
                games_updated += 1

            state.update_season(
                season,
                games_processed=i + 1,
                games_updated=games_updated,
                injured_count=total_injured
            )

        state.update_season(
            season,
            status=Status.SUCCESS,
            phase="done",
            end_time=time.time()
        )

        return {
            'games_processed': len(games),
            'games_updated': games_updated,
            'injured': total_injured
        }

    except Exception as e:
        state.update_season(
            season,
            status=Status.FAILED,
            phase="error",
            error=str(e),
            end_time=time.time()
        )
        return {'games_processed': 0, 'games_updated': 0, 'injured': 0, 'error': str(e)}


def run_injuries_pipeline(
    league_config: LeagueConfig,
    workers: int = 1,
    date: str = None,
    date_range: tuple = None,
    season: str = None,
    game_ids: List[str] = None,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    Run the injuries computation pipeline with parallel season processing.

    Args:
        league_config: League configuration
        workers: Number of parallel workers for season processing
        date: Single date (YYYY-MM-DD)
        date_range: Date range tuple (start, end)
        season: Season string (YYYY-YYYY)
        game_ids: List of specific game IDs
        dry_run: If True, don't update database
        verbose: Show detailed output

    Returns:
        Dict with statistics
    """
    start_time = time.time()

    # Connect to database
    mongo = Mongo()

    # Get league-specific database proxy
    from bball_app.core.data.league_db_proxy import LeagueDbProxy
    db = LeagueDbProxy(mongo.db, league_config)

    # Create injury manager
    injury_manager = InjuryManager(db)

    # Initialize state
    state = PipelineState()
    state.league = league_config.league_id
    state.league_display = league_config.display_name
    state.overall_phase = "Building player maps..."

    # Determine seasons to process
    if game_ids:
        # For specific game IDs, process without parallel (single batch)
        seasons_to_process = ['selected']
    elif season:
        seasons_to_process = [season]
    else:
        # Get all seasons
        seasons_to_process = get_seasons_from_db(db, league_config)

    # Initialize season progress
    for s in seasons_to_process:
        state.seasons[s] = SeasonProgress(season=s)

    # Start progress monitor
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(state, stop_event), daemon=True)
    monitor_thread.start()

    try:
        # Phase 1: Build player maps (shared across all seasons)
        state.overall_phase = "Building player maps..."

        def build_progress(current, total, message):
            state.overall_phase = f"Building maps: {message}"

        total_records = injury_manager._build_precomputed_maps(build_progress)
        state.build_phase_done = True
        state.build_phase_records = total_records

        # Phase 2: Process seasons in parallel
        state.overall_phase = f"Processing {len(seasons_to_process)} seasons ({workers} workers)"

        total_stats = {
            'games_processed': 0,
            'games_updated': 0,
            'games_skipped': 0,
            'total_home_injured': 0,
            'total_away_injured': 0,
            'errors': 0
        }

        if game_ids:
            # Process specific games (no parallel)
            state.update_season('selected', status=Status.RUNNING, phase="processing", start_time=time.time())
            stats = injury_manager.update_all_games(
                game_ids=game_ids,
                dry_run=dry_run
            )
            total_stats = stats
            state.update_season('selected', status=Status.SUCCESS, phase="done",
                              games_processed=stats['games_processed'],
                              games_updated=stats['games_updated'],
                              end_time=time.time())
        else:
            # Process seasons in parallel
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        process_season, s, state, league_config, injury_manager, db, dry_run
                    ): s for s in seasons_to_process
                }

                for future in as_completed(futures):
                    season_name = futures[future]
                    try:
                        result = future.result()
                        total_stats['games_processed'] += result.get('games_processed', 0)
                        total_stats['games_updated'] += result.get('games_updated', 0)
                        total_stats['total_home_injured'] += result.get('injured', 0) // 2
                        total_stats['total_away_injured'] += result.get('injured', 0) // 2
                        if result.get('error'):
                            total_stats['errors'] += 1
                    except Exception as e:
                        total_stats['errors'] += 1
                        state.update_season(season_name, status=Status.FAILED, error=str(e))

        state.overall_phase = "Complete! ✅"

    except KeyboardInterrupt:
        state.overall_phase = "Interrupted ⚠️"
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)

    # Final render
    render_progress(state)

    # Summary
    total_time = time.time() - start_time
    print(f"\n  Total games processed: {total_stats['games_processed']:,}")
    print(f"  Total games updated:   {total_stats['games_updated']:,}")
    print(f"  Total injured players: {total_stats.get('total_home_injured', 0) + total_stats.get('total_away_injured', 0):,}")
    if total_stats.get('errors', 0) > 0:
        print(f"  Errors: {total_stats['errors']}")
    print(f"  Total time: {total_time:.1f}s")
    if dry_run:
        print(f"  Mode: DRY RUN (no changes made)")
    print()

    return total_stats


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Compute and update injured players for games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m bball_app.core.pipeline.injuries_pipeline nba
    python -m bball_app.core.pipeline.injuries_pipeline cbb --workers 4
    python -m bball_app.core.pipeline.injuries_pipeline cbb --season 2024-2025
    python -m bball_app.core.pipeline.injuries_pipeline nba --date 2024-01-15
    python -m bball_app.core.pipeline.injuries_pipeline nba --date-range 2024-01-01,2024-01-31
        """
    )

    # Positional argument
    parser.add_argument(
        "league",
        choices=available_leagues,
        help=f"League to process ({', '.join(available_leagues)})"
    )

    # Parallel processing
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers for season processing (default: 1)"
    )

    # Date filtering (mutually exclusive group for date/date-range)
    date_group = parser.add_mutually_exclusive_group()
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

    # Additional filters
    parser.add_argument(
        "--season",
        type=str,
        help="Season filter (YYYY-YYYY)"
    )
    parser.add_argument(
        "--game-ids",
        type=str,
        help="Comma-separated list of specific game IDs"
    )

    # Common options
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
    date = None
    date_range = None
    game_ids = None

    if args.date:
        date = parse_date(args.date)
    elif args.date_range:
        date_range = parse_date_range(args.date_range)

    if args.game_ids:
        game_ids = [gid.strip() for gid in args.game_ids.split(',') if gid.strip()]

    # Run pipeline
    try:
        run_injuries_pipeline(
            league_config=league_config,
            workers=args.workers,
            date=date,
            date_range=date_range,
            season=args.season,
            game_ids=game_ids,
            dry_run=args.dry_run,
            verbose=args.verbose
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
