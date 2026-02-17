#!/usr/bin/env python3
"""
Full data pipeline for a league.

Steps:
1. ESPN data pull (parallel across seasons)
2. Post-processing (venues, players)
3. Injury computation
4. ELO cache refresh
5. Master training generation (chunked async)
6. CSV registration

Usage:
    python -m bball_app.core.pipeline.full_pipeline nba
    python -m bball_app.core.pipeline.full_pipeline cbb --max-workers 8
    python -m bball_app.core.pipeline.full_pipeline nba --seasons 2023-2024,2024-2025
    python -m bball_app.core.pipeline.full_pipeline nba --skip-injuries
"""

import argparse
import subprocess
import sys
import os
import time
import threading
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.league_config import load_league_config, get_available_leagues, LeagueConfig
from bball_app.core.pipeline.config import PipelineConfig


class Status(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SeasonProgress:
    season: str
    start_date: str
    end_date: str
    status: Status = Status.PENDING
    phase: str = "waiting"
    games_found: int = 0
    players_found: int = 0
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


@dataclass
class PipelineState:
    seasons: Dict[str, SeasonProgress] = field(default_factory=dict)
    overall_phase: str = "Initializing"
    league: str = "nba"
    league_display: str = "NBA"
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update_season(self, season: str, **kwargs):
        with self.lock:
            if season in self.seasons:
                for key, value in kwargs.items():
                    setattr(self.seasons[season], key, value)


def get_season_date_ranges(config: LeagueConfig) -> List[tuple]:
    """Generate date ranges for each season based on league configuration."""
    seasons = []
    current_year = datetime.now().year
    current_month = datetime.now().month

    # Determine the latest season
    if current_month > 8:
        end_year = current_year + 1
    else:
        end_year = current_year

    # Generate seasons from league start year to present
    for start_year in range(config.start_year, end_year):
        season_name = f"{start_year}-{start_year + 1}"
        start_date = f"{start_year}{config.season_start_month:02d}{config.season_start_day:02d}"
        end_date = f"{start_year + 1}{config.season_end_month:02d}{config.season_end_day:02d}"

        # For current season, use today's date if before season end
        if start_year == end_year - 1:
            today = datetime.now().strftime("%Y%m%d")
            if today < end_date:
                end_date = today

        seasons.append((season_name, start_date, end_date))

    return seasons


def run_espn_pull(
    season: str,
    start_date: str,
    end_date: str,
    state: PipelineState,
    league: str,
    dry_run: bool = False,
    verbose: bool = False
) -> bool:
    """Pull ESPN data for a single season."""
    from datetime import datetime as _dt
    from bball_app.core.services.espn_sync import fetch_and_save_games
    from bball_app.core.league_config import load_league_config as _load_lc
    from bball_app.core.mongo import Mongo

    state.update_season(season, status=Status.RUNNING, phase="pulling games",
                       start_time=time.time())

    try:
        league_config = _load_lc(league)
        mongo = Mongo()

        sd = _dt.strptime(start_date, '%Y%m%d').date()
        ed = _dt.strptime(end_date, '%Y%m%d').date()

        state.update_season(season, phase="fetching games")

        # Use quiet mode with a progress callback that updates state incrementally
        def on_progress(pct, msg):
            # Parse game count from the callback message
            # Format: "Pulled N games (D/T days)"
            import re as _re
            m = _re.search(r'Pulled (\d+) games', msg)
            if m:
                state.update_season(season, games_found=int(m.group(1)),
                                   phase=f"{pct}% ({m.group(1)} games)")

        stats = fetch_and_save_games(
            mongo.db, league_config,
            sd, ed,
            dry_run=dry_run,
            verbose=verbose,
            quiet=(not verbose),
            progress_callback=on_progress
        )

        games_count = stats.get('games_processed', 0)
        players_count = stats.get('players_processed', 0)

        state.update_season(season, games_found=games_count, players_found=players_count)

        if stats.get('success', False):
            state.update_season(season, status=Status.SUCCESS, phase="complete",
                              end_time=time.time())
            return True
        else:
            state.update_season(season, status=Status.FAILED, phase="failed",
                              error=stats.get('error', 'Unknown error'),
                              end_time=time.time())
            return False

    except Exception as e:
        state.update_season(season, status=Status.FAILED, phase="error",
                          error=str(e), end_time=time.time())
        return False


def run_post_processing(config: PipelineConfig, dry_run: bool = False):
    """Run post-processing steps (venues, teams, players)."""
    from bball_app.core.services.espn_sync import refresh_venues, refresh_players, refresh_teams
    from bball_app.core.mongo import Mongo

    mongo = Mongo()

    if config.refresh_venues:
        print("  Refreshing venues...")
        try:
            refresh_venues(mongo.db, config.league, dry_run=dry_run)
            print("    Done.")
        except Exception as e:
            print(f"    Warning: Refreshing venues failed: {str(e)[:200]}")

        print("  Geocoding venues missing coordinates...")
        try:
            from bball_app.core.services.espn_sync import geocode_missing_venues
            result = geocode_missing_venues(mongo.db, config.league, dry_run=dry_run)
            geocoded = result.get('geocoded', 0)
            if geocoded:
                print(f"    Geocoded {geocoded} venues.")
            else:
                print(f"    All venues already have coordinates.")
        except Exception as e:
            print(f"    Warning: Venue geocoding failed: {str(e)[:200]}")

    if config.refresh_teams:
        print("  Refreshing teams...")
        try:
            refresh_teams(mongo.db, config.league, dry_run=dry_run)
            print("    Done.")
        except Exception as e:
            print(f"    Warning: Refreshing teams failed: {str(e)[:200]}")

    # Backfill conference data if league uses conference features
    if config.league.extra_feature_stats and any(
        s.startswith('conf_') or s == 'same_conf'
        for s in config.league.extra_feature_stats
    ):
        print("  Backfilling team conferences from ESPN groups...")
        try:
            from bball_app.core.services.espn_sync import refresh_team_conferences
            refresh_team_conferences(mongo.db, config.league, dry_run=dry_run)
            print("    Done.")
        except Exception as e:
            print(f"    Warning: Conference backfill failed: {str(e)[:200]}")

    if config.refresh_players:
        print("  Refreshing players...")
        try:
            refresh_players(mongo.db, config.league, dry_run=dry_run)
            print("    Done.")
        except Exception as e:
            print(f"    Warning: Refreshing players failed: {str(e)[:200]}")


def run_rosters(config: PipelineConfig, dry_run: bool = False):
    """Build team rosters for all seasons from player game data."""
    if not config.build_rosters:
        print("  Skipping roster build (disabled in config)")
        return

    from bball_app.core.services.roster_service import build_rosters
    from bball_app.core.mongo import Mongo

    print("  Building rosters...")

    if dry_run:
        print("    [DRY RUN] Would build rosters for all seasons")
        return

    try:
        mongo = Mongo()
        league = config.league

        # Determine seasons to process (same range as ESPN pull)
        current_year = datetime.now().year
        current_month = datetime.now().month
        end_year = current_year + 1 if current_month > 8 else current_year

        total_teams = 0
        for start_year in range(league.start_year, end_year):
            season = f"{start_year}-{start_year + 1}"
            count = build_rosters(mongo.db, season, league=league, dry_run=False)
            total_teams += count

        print(f"    Done - {total_teams} team-season rosters built.")

    except Exception as e:
        print(f"    Warning: Roster build failed: {str(e)[:200]}")


def run_injuries(config: PipelineConfig, dry_run: bool = False):
    """Run injury computation using the injuries pipeline."""
    if not config.compute_injuries:
        print("  Skipping injury computation (disabled in config)")
        return

    print("  Computing injuries...")
    cmd = [
        sys.executable, "-m", "bball_app.core.pipeline.injuries_pipeline",
        config.league.league_id
    ]

    if dry_run:
        cmd.append("--dry-run")
        print(f"    [DRY RUN] Would run: {' '.join(cmd)}")
        return

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=nba_app_dir,
            env={**os.environ, "PYTHONPATH": project_root, "PYTHONUNBUFFERED": "1"}
        )

        for line in process.stdout:
            line = line.strip()
            if line and not line.startswith('='):
                print(f"    {line}")

        process.wait()

        if process.returncode != 0:
            print(f"    Warning: Injury computation had issues (exit code: {process.returncode})")
        else:
            print("    Done.")

    except Exception as e:
        print(f"    Warning: Injury computation failed: {e}")


def run_elo_cache(config: PipelineConfig, dry_run: bool = False):
    """Refresh ELO cache."""
    from bball_app.core.stats.elo_cache import EloCache
    from bball_app.core.mongo import Mongo

    if not config.cache_elo:
        print("  Skipping ELO cache (disabled in config)")
        return

    print("  Caching ELO ratings...")

    if dry_run:
        print(f"    [DRY RUN] Would compute and cache ELO ratings")
        return

    try:
        mongo = Mongo()
        elo_cache = EloCache(mongo.db, league=config.league)
        stats = elo_cache.compute_and_cache_all()
        print(f"    Done - {stats.get('ratings_cached', 0)} ratings cached.")
    except Exception as e:
        print(f"    Warning: ELO cache failed: {str(e)[:200]}")


def run_training_generation(config: PipelineConfig, dry_run: bool = False, **kwargs):
    """Run master training generation with chunked async."""
    print(f"  Generating training data ({config.training.workers} workers, {config.training.chunk_size} chunk size)...")

    cmd = [
        sys.executable, "-m", "bball_app.core.pipeline.training_pipeline",
        config.league.league_id,
        "--workers", str(config.training.workers),
        "--chunk-size", str(config.training.chunk_size),
    ]

    if not config.training.include_player_features:
        cmd.append("--no-player")

    min_season = kwargs.get('min_season', config.league.min_season)
    if min_season:
        cmd.extend(["--min-season", min_season])

    if dry_run:
        cmd.append("--dry-run")
        print(f"    [DRY RUN] Would run: {' '.join(cmd)}")
        return True

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=nba_app_dir,
            env={**os.environ, "PYTHONPATH": project_root, "PYTHONUNBUFFERED": "1"}
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                print(f"    {line}")

        process.wait()
        return process.returncode == 0

    except Exception as e:
        print(f"    Error in training generation: {e}")
        return False


def run_csv_registration(config: PipelineConfig, dry_run: bool = False):
    """Register master CSV in MongoDB."""
    from bball_app.core.services.training_data import register_existing_master_csv
    from bball_app.core.mongo import Mongo

    if not config.register_csv:
        print("  Skipping CSV registration (disabled in config)")
        return True

    print("  Registering master CSV...")

    if dry_run:
        print(f"    [DRY RUN] Would register master CSV")
        return True

    try:
        mongo = Mongo()
        register_existing_master_csv(mongo.db, league=config.league)
        print("    Done.")
        return True
    except Exception as e:
        print(f"    Warning: CSV registration failed: {str(e)[:200]}")
        return False


def ensure_indexes(league_config: LeagueConfig):
    """Ensure MongoDB indexes exist for the league's collections before ingestion."""
    from bball_app.core.mongo import Mongo

    db = Mongo().db
    colls = league_config.collections

    index_specs = [
        (colls["games"], [("game_id", 1)], {"unique": True}),
        (colls["player_stats"], [("game_id", 1), ("player_id", 1)], {"unique": True}),
        (colls["player_stats"], [("player_id", 1), ("season", 1)], {}),
        (colls["players"], [("player_id", 1)], {"unique": True}),
        (colls["venues"], [("venue_guid", 1)], {"unique": True}),
        (colls["teams"], [("abbreviation", 1)], {}),
        (colls["rosters"], [("team", 1), ("season", 1)], {}),
    ]

    created = 0
    for coll_name, keys, kwargs in index_specs:
        existing = db[coll_name].index_information()
        # Check if an equivalent index already exists
        key_list = [(k, int(d)) for k, d in keys]
        already_exists = any(
            idx.get("key") == key_list for idx in existing.values()
        )
        if not already_exists:
            db[coll_name].create_index(keys, **kwargs)
            created += 1

    if created:
        print(f"  Created {created} MongoDB index(es)")
    else:
        print(f"  All indexes already exist")


def render_progress(state: PipelineState, final: bool = False):
    """Render a live progress bar for ESPN data pull."""
    with state.lock:
        completed = sum(1 for p in state.seasons.values() if p.status == Status.SUCCESS)
        failed = sum(1 for p in state.seasons.values() if p.status == Status.FAILED)
        running = sum(1 for p in state.seasons.values() if p.status == Status.RUNNING)
        total_games = sum(p.games_found for p in state.seasons.values())
        total = len(state.seasons)
        done = completed + failed

        start_times = [p.start_time for p in state.seasons.values() if p.start_time]
        elapsed = (time.time() - min(start_times)) if start_times else 0
        running_seasons = [s for s, p in state.seasons.items() if p.status == Status.RUNNING]

    # Format elapsed
    mins, secs = divmod(int(elapsed), 60)
    hours, mins = divmod(mins, 60)
    elapsed_str = f"{hours}h {mins:02d}m {secs:02d}s" if hours > 0 else f"{mins:02d}m {secs:02d}s"

    pct = (done / total * 100) if total > 0 else 0
    bar_width = 30
    filled = int(bar_width * pct / 100)
    bar = "\u2588" * filled + "\u2591" * (bar_width - filled)

    rate = total_games / elapsed if elapsed > 0 else 0

    line = (
        f"\r  [{bar}] {pct:5.1f}% | "
        f"{done}/{total} seasons | "
        f"{total_games:,} games | "
        f"{rate:,.1f} games/s | "
        f"{elapsed_str}"
    )
    if running_seasons and not final:
        line += f" | active: {', '.join(running_seasons[:4])}"

    if final:
        print(line)
        if failed > 0:
            print(f"  Failed seasons: {failed}")
    else:
        print(line, end="", flush=True)


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Full data pipeline for a league",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m bball_app.core.pipeline.full_pipeline nba
    python -m bball_app.core.pipeline.full_pipeline cbb --max-workers 8
    python -m bball_app.core.pipeline.full_pipeline nba --seasons 2023-2024,2024-2025
    python -m bball_app.core.pipeline.full_pipeline cbb --dry-run
        """
    )
    parser.add_argument("league", choices=available_leagues,
                       help=f"League to process ({', '.join(available_leagues)})")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Max parallel workers for ESPN pull")
    parser.add_argument("--seasons", type=str, default=None,
                       help="Comma-separated seasons to process (e.g., '2022-2023,2023-2024')")
    parser.add_argument("--skip-espn", action="store_true",
                       help="Skip ESPN data pull")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip master training generation")
    parser.add_argument("--skip-post", action="store_true",
                       help="Skip post-processing (venues, players)")
    parser.add_argument("--skip-injuries", action="store_true",
                       help="Skip injury computation")
    parser.add_argument("--skip-rosters", action="store_true",
                       help="Skip roster build")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without modifying data")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output")
    args = parser.parse_args()

    # Load configuration
    league_config = load_league_config(args.league)
    config = PipelineConfig.from_league(league_config)

    if args.max_workers:
        config.espn_max_workers = args.max_workers

    print("\n" + "=" * 70)
    print(f"  {league_config.display_name} Full Pipeline")
    print("=" * 70)
    print(f"  League:          {args.league.upper()}")
    print(f"  ESPN workers:    {config.espn_max_workers}")
    print(f"  Training workers:{config.training.workers}")
    print(f"  Player features: {'Yes' if config.training.include_player_features else 'No'}")
    print(f"  Dry run:         {args.dry_run}")
    print("=" * 70 + "\n")

    pipeline_start = time.time()

    # Step 0: Ensure indexes
    print("[0/7] Ensuring MongoDB indexes...")
    ensure_indexes(league_config)

    # Step 1: ESPN data pull
    if not args.skip_espn:
        print("[1/7] ESPN data pull...")

        # Get season date ranges
        all_seasons = get_season_date_ranges(league_config)

        # Filter seasons if specified
        if args.seasons:
            filter_seasons = set(args.seasons.split(","))
            all_seasons = [(s, sd, ed) for s, sd, ed in all_seasons if s in filter_seasons]

        print(f"  Seasons to process: {len(all_seasons)}")

        # Initialize state
        state = PipelineState()
        state.league = args.league
        state.league_display = league_config.display_name

        for season, start_date, end_date in all_seasons:
            state.seasons[season] = SeasonProgress(
                season=season,
                start_date=start_date,
                end_date=end_date
            )

        # Background progress rendering
        stop_progress = threading.Event()

        def progress_loop():
            while not stop_progress.is_set():
                render_progress(state)
                stop_progress.wait(0.5)

        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()

        # Run parallel ESPN pulls
        with ThreadPoolExecutor(max_workers=config.espn_max_workers) as executor:
            futures = {}
            for season, start_date, end_date in all_seasons:
                future = executor.submit(
                    run_espn_pull, season, start_date, end_date, state,
                    args.league, args.dry_run, args.verbose
                )
                futures[future] = season

            for future in as_completed(futures):
                season = futures[future]
                try:
                    future.result()
                except Exception as e:
                    state.update_season(season, status=Status.FAILED, error=str(e))

        # Stop progress thread and render final summary
        stop_progress.set()
        progress_thread.join(timeout=1)
        render_progress(state, final=True)
    else:
        print("[1/7] ESPN data pull... SKIPPED")

    # Step 2: Post-processing
    if not args.skip_post:
        print("\n[2/7] Post-processing...")
        run_post_processing(config, args.dry_run)
    else:
        print("\n[2/7] Post-processing... SKIPPED")

    # Step 3: Injury computation
    if not args.skip_injuries:
        print("\n[3/7] Injury computation...")
        run_injuries(config, args.dry_run)
    else:
        print("\n[3/7] Injury computation... SKIPPED")

    # Step 4: ELO cache
    print("\n[4/7] ELO cache...")
    run_elo_cache(config, args.dry_run)

    # Step 5: Training generation
    if not args.skip_training:
        print("\n[5/7] Master training generation...")
        success = run_training_generation(config, args.dry_run)
        if not success:
            print("  Warning: Training generation had issues")
    else:
        print("\n[5/7] Master training generation... SKIPPED")

    # Step 6: Roster build (for prediction use — NOT used by training generation,
    # which gets player data directly from player_stats game records)
    if not args.skip_rosters:
        print("\n[6/7] Roster build...")
        run_rosters(config, args.dry_run)
    else:
        print("\n[6/7] Roster build... SKIPPED")

    # Step 7: CSV registration
    print("\n[7/7] CSV registration...")
    run_csv_registration(config, args.dry_run)

    # Final summary
    elapsed = int(time.time() - pipeline_start)
    mins, secs = divmod(elapsed, 60)
    hours, mins = divmod(mins, 60)

    print("\n" + "=" * 70)
    print("  Pipeline complete!")
    if hours > 0:
        print(f"  Total time: {hours}h {mins:02d}m {secs:02d}s")
    else:
        print(f"  Total time: {mins}m {secs:02d}s")
    print(f"  Output: {league_config.master_training_csv}")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
