#!/usr/bin/env python3
"""
Full data pipeline for a league.

Steps:
1. Ensure MongoDB indexes
2. ESPN data pull (parallel across seasons)
3. Post-processing (venues, players)
4. Parallel enrichment (injuries + ELO + rosters)
5. Master training generation (chunked async)
6. CSV registration

Usage:
    python -m bball.pipeline.full_pipeline nba
    python -m bball.pipeline.full_pipeline cbb --max-workers 8
    python -m bball.pipeline.full_pipeline nba --seasons 2023-2024,2024-2025
    python -m bball.pipeline.full_pipeline nba --skip-injuries
"""

import argparse
import subprocess
import sys
import os
import time
import threading
import re
from sportscore.pipeline.parallel import ParallelItemProcessor
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball.league_config import load_league_config, get_available_leagues, LeagueConfig
from bball.pipeline.config import PipelineConfig
from sportscore.pipeline import BasePipeline, StepDefinition, PipelineContext
from sportscore.pipeline.parallel import ParallelStepGroup


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
    from bball.services.espn_sync import fetch_and_save_games
    from bball.league_config import load_league_config as _load_lc
    from bball.mongo import Mongo

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
    from bball.services.espn_sync import refresh_venues, refresh_players, refresh_teams
    from bball.mongo import Mongo

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
            from bball.services.espn_sync import geocode_missing_venues
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
            from bball.services.espn_sync import refresh_team_conferences
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

    from bball.services.roster_service import build_rosters
    from bball.mongo import Mongo

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
    """Run injury computation using the injuries pipeline (subprocess)."""
    if not config.compute_injuries:
        print("  Skipping injury computation (disabled in config)")
        return

    print("  Computing injuries...")
    cmd = [
        sys.executable, "-m", "bball.pipeline.injuries_pipeline",
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
            cwd=project_root,
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
    from bball.stats.elo_cache import EloCache
    from bball.mongo import Mongo

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
        sys.executable, "-m", "bball.pipeline.training_pipeline",
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
            cwd=project_root,
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
    from bball.services.training_data import register_existing_master_csv
    from bball.mongo import Mongo

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
    from bball.mongo import Mongo

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


# ---------------------------------------------------------------------------
# BasketballFullPipeline — structured pipeline using sportscore BasePipeline
# ---------------------------------------------------------------------------

class BasketballFullPipeline(BasePipeline):
    """
    Full data pipeline for basketball, extending sportscore's BasePipeline.

    Orchestrates: indexes -> ESPN pull -> post-processing -> parallel enrichment
    (injuries + ELO + rosters) -> training generation -> CSV registration.
    """

    def __init__(self, league_id: str, config: PipelineConfig = None,
                 skip_espn: bool = False, skip_post: bool = False,
                 skip_injuries: bool = False, skip_rosters: bool = False,
                 skip_training: bool = False,
                 seasons: List[str] = None, max_workers: int = None,
                 dry_run: bool = False, verbose: bool = False):
        super().__init__(league_id)
        self.league_config = load_league_config(league_id)
        self.config = config or PipelineConfig.from_league(self.league_config)
        self.skip_espn = skip_espn
        self.skip_post = skip_post
        self.skip_injuries = skip_injuries
        self.skip_rosters = skip_rosters
        self.skip_training = skip_training
        self.seasons = seasons
        self.dry_run = dry_run
        self.verbose = verbose

        if max_workers:
            self.config.espn_max_workers = max_workers

    def define_steps(self):
        return [
            StepDefinition("ensure_indexes", self._step_ensure_indexes,
                          description="Ensuring MongoDB indexes"),
            StepDefinition("espn_pull", self._step_espn_pull,
                          skip_condition=lambda ctx: self.skip_espn,
                          description="ESPN data pull"),
            StepDefinition("post_processing", self._step_post_processing,
                          skip_condition=lambda ctx: self.skip_post,
                          description="Post-processing"),
            StepDefinition("parallel_enrichment", self._step_parallel_enrichment,
                          continue_on_failure=True,
                          description="Injuries + ELO + Rosters in parallel"),
            StepDefinition("training_generation", self._step_training_generation,
                          skip_condition=lambda ctx: self.skip_training,
                          description="Master training generation"),
            StepDefinition("csv_registration", self._step_csv_registration,
                          description="CSV registration"),
        ]

    # -- Hooks for formatted output --

    def on_step_start(self, step, index, total):
        print(f"\n[{index + 1}/{total}] {step.description or step.name}...")

    def on_step_complete(self, step, result, index, total):
        if not result.success:
            print(f"  FAILED: {result.error}")

    def on_step_skip(self, step, index, total):
        print(f"\n[{index + 1}/{total}] {step.description or step.name}... SKIPPED")

    # -- Step implementations --

    def _step_ensure_indexes(self, context: PipelineContext):
        ensure_indexes(self.league_config)

    def _step_espn_pull(self, context: PipelineContext):
        all_seasons = get_season_date_ranges(self.league_config)

        if self.seasons:
            filter_set = set(self.seasons)
            all_seasons = [(s, sd, ed) for s, sd, ed in all_seasons if s in filter_set]

        print(f"  Seasons to process: {len(all_seasons)}")

        state = PipelineState()
        state.league = self.league_id
        state.league_display = self.league_config.display_name

        for season, start_date, end_date in all_seasons:
            state.seasons[season] = SeasonProgress(
                season=season, start_date=start_date, end_date=end_date
            )

        # Background progress rendering
        stop_progress = threading.Event()

        def progress_loop():
            while not stop_progress.is_set():
                render_progress(state)
                stop_progress.wait(0.5)

        progress_thread = threading.Thread(target=progress_loop, daemon=True)
        progress_thread.start()

        def process_fn(season_tuple):
            season, start_date, end_date = season_tuple
            return run_espn_pull(season, start_date, end_date, state,
                                self.league_id, self.dry_run, self.verbose)

        processor = ParallelItemProcessor(
            items=all_seasons,
            process_fn=process_fn,
            max_workers=self.config.espn_max_workers,
        )
        results = processor.execute()

        for (season, _, _), result, error in results:
            if error:
                state.update_season(season, status=Status.FAILED, error=str(error))

        stop_progress.set()
        progress_thread.join(timeout=1)
        render_progress(state, final=True)

    def _step_post_processing(self, context: PipelineContext):
        run_post_processing(self.config, self.dry_run)

    def _step_parallel_enrichment(self, context: PipelineContext):
        tasks = []

        if self.config.compute_injuries and not self.skip_injuries:
            tasks.append(("injuries", self._run_injuries_direct, {}))
        if self.config.cache_elo:
            tasks.append(("elo_cache", lambda: run_elo_cache(self.config, self.dry_run), {}))
        if self.config.build_rosters and not self.skip_rosters:
            tasks.append(("rosters", lambda: run_rosters(self.config, self.dry_run), {}))

        if not tasks:
            print("  All enrichment steps skipped")
            return

        def _on_complete(name, task_result):
            status = "OK" if task_result.success else f"FAILED: {task_result.error}"
            duration = f"{task_result.duration_seconds:.1f}s" if task_result.duration_seconds else "?"
            print(f"  {name}: {status} ({duration})")

        def _on_error(name, task_result):
            print(f"  {name}: FAILED: {task_result.error}")

        group = ParallelStepGroup(
            tasks=tasks,
            max_workers=len(tasks),
            on_complete=_on_complete,
            on_error=_on_error,
        )
        results = group.execute()

        # Store sub-task results in the step stats
        context.step_results["parallel_enrichment"].stats = {
            name: {"success": r.success, "error": r.error,
                   "duration": r.duration_seconds}
            for name, r in results.items()
        }

        # If any sub-task failed, note it but don't raise (continue_on_failure=True)
        failures = [name for name, r in results.items() if not r.success]
        if failures:
            print(f"  Warning: Failed sub-tasks: {', '.join(failures)}")

    def _run_injuries_direct(self):
        """Run injuries pipeline as direct in-process call instead of subprocess."""
        from bball.pipeline.injuries_pipeline import run_injuries_pipeline

        if self.dry_run:
            print("  [DRY RUN] Would compute injuries")
            return

        return run_injuries_pipeline(
            league_config=self.league_config,
            dry_run=self.dry_run,
        )

    def _step_training_generation(self, context: PipelineContext):
        success = run_training_generation(self.config, self.dry_run)
        if not success:
            raise RuntimeError("Training generation failed")

    def _step_csv_registration(self, context: PipelineContext):
        run_csv_registration(self.config, self.dry_run)

    def run(self, **kwargs) -> 'PipelineResult':
        """Run pipeline with banner output."""
        from sportscore.pipeline import PipelineResult

        print("\n" + "=" * 70)
        print(f"  {self.league_config.display_name} Full Pipeline")
        print("=" * 70)
        print(f"  League:          {self.league_id.upper()}")
        print(f"  ESPN workers:    {self.config.espn_max_workers}")
        print(f"  Training workers:{self.config.training.workers}")
        print(f"  Player features: {'Yes' if self.config.training.include_player_features else 'No'}")
        print(f"  Dry run:         {self.dry_run}")
        print("=" * 70 + "\n")

        pipeline_start = time.time()

        result = super().run(dry_run=self.dry_run, verbose=self.verbose)

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
        print(f"  Steps: {len(result.steps_completed)} completed, "
              f"{len(result.steps_failed)} failed, "
              f"{len(result.steps_skipped)} skipped")
        print(f"  Output: {self.league_config.master_training_csv}")
        print("=" * 70 + "\n")

        return result


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Full data pipeline for a league",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m bball.pipeline.full_pipeline nba
    python -m bball.pipeline.full_pipeline cbb --max-workers 8
    python -m bball.pipeline.full_pipeline nba --seasons 2023-2024,2024-2025
    python -m bball.pipeline.full_pipeline cbb --dry-run
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

    pipeline = BasketballFullPipeline(
        league_id=args.league,
        skip_espn=args.skip_espn,
        skip_post=args.skip_post,
        skip_injuries=args.skip_injuries,
        skip_rosters=args.skip_rosters,
        skip_training=args.skip_training,
        seasons=args.seasons.split(",") if args.seasons else None,
        max_workers=args.max_workers,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )

    result = pipeline.run()
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())
