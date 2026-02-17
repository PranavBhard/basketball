#!/usr/bin/env python3
"""
Parallel Data Pipeline for NBA/CBB

Pulls game/player/venue data from ESPN API for multiple seasons in parallel,
then generates master training data.

Usage:
    python scripts/parallel_pipeline.py <league> [--max-workers N] [--dry-run] [--skip-training]

Examples:
    python scripts/parallel_pipeline.py nba
    python scripts/parallel_pipeline.py cbb --max-workers 6
    python scripts/parallel_pipeline.py nba --seasons 2023-2024,2024-2025
"""

import argparse
import subprocess
import sys
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bball_app.core.league_config import load_league_config, get_available_leagues, LeagueConfig


class Status(Enum):
    PENDING = "⏳"
    RUNNING = "🔄"
    SUCCESS = "✅"
    FAILED = "❌"
    SKIPPED = "⏭️"


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

    # Determine the latest season (season cutover is August)
    if current_month > 8:  # After August = new season started
        end_year = current_year + 1
    else:
        end_year = current_year

    # Generate seasons from league start year to present
    for start_year in range(config.start_year, end_year):
        season_name = f"{start_year}-{start_year + 1}"

        # Season start date
        start_date = f"{start_year}{config.season_start_month:02d}{config.season_start_day:02d}"

        # Season end date
        end_date = f"{start_year + 1}{config.season_end_month:02d}{config.season_end_day:02d}"

        # For current season, use today's date if before season end
        if start_year == end_year - 1:
            today = datetime.now().strftime("%Y%m%d")
            if today < end_date:
                end_date = today

        seasons.append((season_name, start_date, end_date))

    return seasons


def run_espn_pull(season: str, start_date: str, end_date: str,
                  state: PipelineState, league: str, dry_run: bool = False,
                  verbose: bool = False) -> bool:
    """Pull ESPN data for a single season."""
    state.update_season(season, status=Status.RUNNING, phase="pulling games",
                       start_time=time.time())

    cmd = [
        sys.executable, "cli/espn_api.py",
        "--league", league,
        "--dates", f"{start_date},{end_date}"
    ]

    if dry_run:
        cmd.append("--dry-run")

    try:
        # Run the command and capture output
        # Use PYTHONUNBUFFERED to get real-time output
        env = {**os.environ, "PYTHONPATH": "/Users/pranav/Documents/NBA", "PYTHONUNBUFFERED": "1"}
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env
        )

        games_count = 0
        players_count = 0

        # Parse output for progress
        for line in process.stdout:
            line = line.strip()
            line_lower = line.lower()

            if verbose:
                print(f"[{season}] {line}")

            # Detect phase changes
            if "fetching games for" in line_lower:
                state.update_season(season, phase="fetching games")
            elif "found" in line_lower and "events" in line_lower:
                state.update_season(season, phase="processing games")
            elif "stored" in line_lower and "player" in line_lower:
                state.update_season(season, phase="processing players")

            # Count games: "✓ Stored game:" or "[DRY RUN] Would store game:"
            if "stored game" in line_lower or "would store game" in line_lower:
                games_count += 1
                state.update_season(season, games_found=games_count)
                if verbose:
                    print(f"  -> Matched game, count now: {games_count}")

            # Count players: "✓ Stored X player stats" or "[DRY RUN] Would store X player stats"
            if "player stats" in line_lower:
                match = re.search(r'(\d+)\s*player\s*stats', line_lower)
                if match:
                    players_count += int(match.group(1))
                    state.update_season(season, players_found=players_count)
                    if verbose:
                        print(f"  -> Matched players: +{match.group(1)}, total now: {players_count}")

        process.wait()

        if process.returncode == 0:
            state.update_season(season, status=Status.SUCCESS, phase="complete",
                              end_time=time.time())
            return True
        else:
            state.update_season(season, status=Status.FAILED, phase="failed",
                              error=f"Exit code: {process.returncode}",
                              end_time=time.time())
            return False

    except Exception as e:
        state.update_season(season, status=Status.FAILED, phase="error",
                          error=str(e), end_time=time.time())
        return False


def render_progress_line(state: PipelineState):
    """Render a compact single-line progress update."""
    with state.lock:
        completed = sum(1 for p in state.seasons.values() if p.status == Status.SUCCESS)
        failed = sum(1 for p in state.seasons.values() if p.status == Status.FAILED)
        running = sum(1 for p in state.seasons.values() if p.status == Status.RUNNING)
        total_games = sum(p.games_found for p in state.seasons.values())
        total_players = sum(p.players_found for p in state.seasons.values())

        # Get current running seasons
        running_seasons = [s for s, p in state.seasons.items() if p.status == Status.RUNNING]
        running_str = ", ".join(running_seasons[:3])
        if len(running_seasons) > 3:
            running_str += f" +{len(running_seasons) - 3}"

    total = len(state.seasons)
    line = f"[{completed}/{total}] Games: {total_games:,} | Players: {total_players:,} | Running: {running}"
    if running_str:
        line += f" ({running_str})"

    # \r moves to start of line, \033[K clears to end of line
    sys.stdout.write(f"\r{line}\033[K")
    sys.stdout.flush()


def render_full_table(state: PipelineState, header: str = ""):
    """Render the full progress table (for initial/final display)."""
    if header:
        print(header)

    print("=" * 85)
    print(f"  {state.league_display} Parallel Pipeline - {state.overall_phase}")
    print("=" * 85)
    print(f"{'Season':<12} {'Status':<8} {'Phase':<22} {'Games':<10} {'Players':<12} {'Time':<8}")
    print("-" * 85)

    with state.lock:
        for season in sorted(state.seasons.keys()):
            prog = state.seasons[season]
            status_icon = prog.status.value
            games_str = str(prog.games_found) if prog.games_found > 0 else "-"
            players_str = str(prog.players_found) if prog.players_found > 0 else "-"
            print(f"{prog.season:<12} {status_icon:<8} {prog.phase:<22} {games_str:<10} {players_str:<12} {prog.elapsed:<8}")

    print("-" * 85)

    # Summary stats
    completed = sum(1 for p in state.seasons.values() if p.status == Status.SUCCESS)
    failed = sum(1 for p in state.seasons.values() if p.status == Status.FAILED)
    running = sum(1 for p in state.seasons.values() if p.status == Status.RUNNING)
    pending = sum(1 for p in state.seasons.values() if p.status == Status.PENDING)
    total_games = sum(p.games_found for p in state.seasons.values())
    total_players = sum(p.players_found for p in state.seasons.values())

    print(f"  ✅ {completed} complete | 🔄 {running} running | ⏳ {pending} pending | ❌ {failed} failed")
    print(f"  Total: {total_games} games, {total_players} player stats")
    print("=" * 85)


def progress_monitor(state: PipelineState, stop_event: threading.Event):
    """Background thread to update the progress display."""
    # Show initial table
    render_full_table(state)

    last_full_render = time.time()

    while not stop_event.is_set():
        # Show compact progress line (overwrites itself on same line)
        render_progress_line(state)

        # Show full table every 30 seconds
        if time.time() - last_full_render > 30:
            # Clear the progress line and move to new line for table
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()
            render_full_table(state)
            last_full_render = time.time()

        time.sleep(0.5)

    # Final render - clear progress line and show full table
    sys.stdout.write("\r\033[K\n")
    sys.stdout.flush()
    render_full_table(state, header="--- Final Status ---")


def run_post_processing(state: PipelineState, league: str, dry_run: bool = False):
    """Run post-processing steps (venues, players, injuries)."""
    state.overall_phase = "Post-processing: Refreshing venues"

    steps = [
        ("Refreshing venues", ["--refresh-venues"]),
        ("Refreshing players", ["--refresh-players"]),
        ("Computing injuries", ["--player-injuries"]),
    ]

    for desc, args in steps:
        state.overall_phase = f"Post-processing: {desc}"
        cmd = [
            sys.executable, "cli/espn_api.py",
            "--league", league,
        ] + args

        if dry_run:
            cmd.append("--dry-run")

        try:
            subprocess.run(
                cmd,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                env={**os.environ, "PYTHONPATH": "/Users/pranav/Documents/NBA"},
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"\nWarning: {desc} failed: {e.stderr[:200] if e.stderr else 'Unknown error'}")


def run_master_training(state: PipelineState, league: str, config: LeagueConfig, dry_run: bool = False):
    """Generate master training data."""
    state.overall_phase = "Generating master training data (all features)..."

    cmd = [
        sys.executable, "cli/generate_master_training.py",
        "--league", league,
        "--min-season", config.min_season
    ]

    if dry_run:
        print("\n[DRY RUN] Would run master training generation")
        return True

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**os.environ, "PYTHONPATH": "/Users/pranav/Documents/NBA"}
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                # Update phase based on output
                if "feature" in line.lower():
                    state.overall_phase = f"Master training: {line[:50]}..."
                elif "%" in line or "progress" in line.lower():
                    state.overall_phase = f"Master training: {line[:50]}..."

        process.wait()
        return process.returncode == 0

    except Exception as e:
        print(f"\nError in master training: {e}")
        return False


def register_master_csv(league: str, dry_run: bool = False):
    """Register master CSV in MongoDB."""
    if dry_run:
        print("\n[DRY RUN] Would register master CSV")
        return True

    cmd = [
        sys.executable, "cli/register_master_csv.py",
        "--league", league
    ]

    try:
        subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**os.environ, "PYTHONPATH": "/Users/pranav/Documents/NBA"},
            capture_output=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError:
        return False


def main():
    # Get available leagues dynamically from YAML configs
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Parallel Data Pipeline for NBA/CBB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/parallel_pipeline.py nba
    python scripts/parallel_pipeline.py cbb --max-workers 6
    python scripts/parallel_pipeline.py nba --seasons 2023-2024,2024-2025
    python scripts/parallel_pipeline.py cbb --dry-run
        """
    )
    parser.add_argument("league", choices=available_leagues,
                       help=f"League to process ({', '.join(available_leagues)})")
    parser.add_argument("--max-workers", "--workers", type=int, default=4,
                       help="Maximum parallel season pulls (default: 4)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Don't modify database, just show what would be done")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip master training generation")
    parser.add_argument("--seasons", type=str, default=None,
                       help="Comma-separated seasons to process (e.g., '2022-2023,2023-2024')")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show detailed output for debugging")
    args = parser.parse_args()

    league = args.league
    league_config = load_league_config(league)

    print("\n" + "=" * 80)
    print(f"  {league_config.display_name} Parallel Data Pipeline")
    print("=" * 80)

    # Get season date ranges
    all_seasons = get_season_date_ranges(league_config)

    # Filter seasons if specified
    if args.seasons:
        filter_seasons = set(args.seasons.split(","))
        all_seasons = [(s, sd, ed) for s, sd, ed in all_seasons if s in filter_seasons]

    print(f"  League: {league.upper()}")
    print(f"  Seasons to process: {len(all_seasons)}")
    print(f"  Max parallel workers: {args.max_workers}")
    print(f"  Dry run: {args.dry_run}")
    print("=" * 80 + "\n")

    # Initialize state
    state = PipelineState()
    state.league = league
    state.league_display = league_config.display_name
    state.overall_phase = "Pulling ESPN data"

    for season, start_date, end_date in all_seasons:
        state.seasons[season] = SeasonProgress(
            season=season,
            start_date=start_date,
            end_date=end_date
        )

    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(state, stop_event))
    monitor_thread.start()

    try:
        # Phase 1: Parallel ESPN pulls
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {}
            for season, start_date, end_date in all_seasons:
                future = executor.submit(
                    run_espn_pull, season, start_date, end_date, state, league,
                    args.dry_run, args.verbose
                )
                futures[future] = season

            # Wait for all to complete
            for future in as_completed(futures):
                season = futures[future]
                try:
                    future.result()
                except Exception as e:
                    state.update_season(season, status=Status.FAILED, error=str(e))

        # Phase 2: Post-processing (sequential)
        state.overall_phase = "Post-processing"
        run_post_processing(state, league, args.dry_run)

        # Phase 3: Master training generation
        if not args.skip_training:
            state.overall_phase = "Generating master training data"
            success = run_master_training(state, league, league_config, args.dry_run)

            if success:
                state.overall_phase = "Registering master CSV"
                register_master_csv(league, args.dry_run)

        state.overall_phase = "Complete! ✅"

    except KeyboardInterrupt:
        state.overall_phase = "Interrupted ⚠️"
        print("\n\nInterrupted by user")
    finally:
        stop_event.set()
        monitor_thread.join()

    # Final summary
    print("\n")
    completed = sum(1 for p in state.seasons.values() if p.status == Status.SUCCESS)
    failed = sum(1 for p in state.seasons.values() if p.status == Status.FAILED)

    if failed > 0:
        print("Failed seasons:")
        for season, prog in state.seasons.items():
            if prog.status == Status.FAILED:
                print(f"  - {season}: {prog.error}")

    print(f"\nTotal: {completed}/{len(state.seasons)} seasons completed successfully")

    if not args.skip_training and not args.dry_run:
        csv_path = league_config.raw.get("paths", {}).get("master_training_csv", "master_training/MASTER_TRAINING.csv")
        print(f"Master CSV: {csv_path}")


if __name__ == "__main__":
    main()
