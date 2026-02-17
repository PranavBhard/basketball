#!/usr/bin/env python3
"""
Master Training Data Regeneration Script

Regenerates master training data with real-time progress logging.

Usage:
    python scripts/regenerate_training.py <league> [options]

Examples:
    python scripts/regenerate_training.py nba
    python scripts/regenerate_training.py cbb --no-player
    python scripts/regenerate_training.py nba --season 2023-2024
    python scripts/regenerate_training.py cbb --min-season 2015-2016
"""

import argparse
import subprocess
import sys
import os
import time
import threading
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from enum import Enum

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bball_app.core.league_config import load_league_config, get_available_leagues, LeagueConfig


class Phase(Enum):
    INITIALIZING = "Initializing"
    LOADING_GAMES = "Loading games"
    GENERATING_FEATURES = "Generating features"
    COMPUTING_PER = "Computing PER"
    PROCESSING_INJURIES = "Processing injuries"
    WRITING_CSV = "Writing CSV"
    REGISTERING = "Registering in MongoDB"
    COMPLETE = "Complete"
    FAILED = "Failed"


@dataclass
class ProgressState:
    league: str = "nba"
    league_display: str = "NBA"
    phase: Phase = Phase.INITIALIZING
    total_games: int = 0
    processed_games: int = 0
    current_season: str = ""
    current_game: str = ""
    features_generated: int = 0
    start_time: float = field(default_factory=time.time)
    error: Optional[str] = None
    warnings: list = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def elapsed(self) -> str:
        elapsed = int(time.time() - self.start_time)
        mins, secs = divmod(elapsed, 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h {mins:02d}m {secs:02d}s"
        return f"{mins:02d}m {secs:02d}s"

    @property
    def progress_pct(self) -> float:
        if self.total_games == 0:
            return 0.0
        return (self.processed_games / self.total_games) * 100

    @property
    def eta(self) -> str:
        if self.processed_games == 0 or self.total_games == 0:
            return "--:--"
        elapsed = time.time() - self.start_time
        rate = self.processed_games / elapsed
        remaining = self.total_games - self.processed_games
        if rate > 0:
            eta_secs = int(remaining / rate)
            mins, secs = divmod(eta_secs, 60)
            hours, mins = divmod(mins, 60)
            if hours > 0:
                return f"{hours}h {mins:02d}m"
            return f"{mins:02d}m {secs:02d}s"
        return "--:--"


def render_progress(state: ProgressState, clear: bool = True):
    """Render the progress dashboard."""
    if clear:
        # Move cursor up and clear lines (ANSI escape codes)
        print("\033[15A\033[J", end="")

    # Phase icons
    phase_icons = {
        Phase.INITIALIZING: "⏳",
        Phase.LOADING_GAMES: "📊",
        Phase.GENERATING_FEATURES: "⚙️",
        Phase.COMPUTING_PER: "🏀",
        Phase.PROCESSING_INJURIES: "🏥",
        Phase.WRITING_CSV: "💾",
        Phase.REGISTERING: "📝",
        Phase.COMPLETE: "✅",
        Phase.FAILED: "❌",
    }

    icon = phase_icons.get(state.phase, "🔄")

    print("=" * 70)
    print(f"  {state.league_display} Master Training Regeneration")
    print("=" * 70)
    print(f"  Phase: {icon} {state.phase.value}")
    print("-" * 70)

    # Progress bar
    bar_width = 50
    filled = int(bar_width * state.progress_pct / 100)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  Progress: [{bar}] {state.progress_pct:5.1f}%")
    print(f"  Games:    {state.processed_games:,} / {state.total_games:,}")
    print(f"  Elapsed:  {state.elapsed}")
    print(f"  ETA:      {state.eta}")
    print("-" * 70)

    # Current activity
    if state.current_season:
        print(f"  Season:   {state.current_season}")
    if state.current_game:
        print(f"  Game:     {state.current_game[:55]}")

    print("-" * 70)

    # Warnings (show last 2)
    if state.warnings:
        print(f"  Warnings: {len(state.warnings)} total")
        for w in state.warnings[-2:]:
            print(f"    ⚠️  {w[:60]}")

    print("=" * 70)


def progress_monitor(state: ProgressState, stop_event: threading.Event):
    """Background thread to update the progress display."""
    # Initial render - make space
    print("\n" * 15)

    while not stop_event.is_set():
        render_progress(state)
        time.sleep(0.3)

    # Final render
    render_progress(state)


def parse_output_line(line: str, state: ProgressState):
    """Parse output line and update state."""
    line_lower = line.lower()

    with state.lock:
        # Detect phase changes based on specific output patterns
        if "connecting to mongodb" in line_lower:
            state.phase = Phase.INITIALIZING
        elif "loading game data" in line_lower:
            state.phase = Phase.LOADING_GAMES
        elif "preloading data caches" in line_lower:
            state.phase = Phase.INITIALIZING
        elif "initializing per calculator" in line_lower or "preloading player stats" in line_lower or "[3/4] per player cache" in line_lower:
            state.phase = Phase.COMPUTING_PER
        elif "preloading injury" in line_lower or "[2/4] injury" in line_lower:
            state.phase = Phase.PROCESSING_INJURIES
        elif "[4/4] elo" in line_lower:
            state.phase = Phase.INITIALIZING  # Elo is part of initialization
        elif "parallel processing" in line_lower or ("processing" in line_lower and "games in" in line_lower and "chunks" in line_lower):
            state.phase = Phase.GENERATING_FEATURES
        elif "combining" in line_lower and "chunks" in line_lower:
            state.phase = Phase.WRITING_CSV
        elif "generation complete" in line_lower or "written to:" in line_lower or "rows written:" in line_lower:
            state.phase = Phase.WRITING_CSV

        # Extract total games from "Found X games" or "Processing X games"
        found_match = re.search(r'(?:found|processing)\s+(\d+)\s+games', line_lower)
        if found_match:
            total = int(found_match.group(1))
            if total > 0 and state.total_games == 0:
                state.total_games = total

        # Extract progress from "Progress: X/Y (Z%)" pattern (legacy format)
        progress_match = re.search(r'progress:\s*(\d+)/(\d+)\s*\((\d+(?:\.\d+)?)%\)', line_lower)
        if progress_match:
            current = int(progress_match.group(1))
            total = int(progress_match.group(2))
            if total > 0:
                state.processed_games = current
                state.total_games = total

        # Extract progress from new bar format: "[████░░░] 15.0% | 3,000/20,000 games"
        bar_progress_match = re.search(r'\]\s*(\d+(?:\.\d+)?)\s*%\s*\|\s*([\d,]+)/([\d,]+)\s*games', line)
        if bar_progress_match:
            current = int(bar_progress_match.group(2).replace(',', ''))
            total = int(bar_progress_match.group(3).replace(',', ''))
            if total > 0:
                state.processed_games = current
                state.total_games = total

        # Extract total games from "Total games: X" in the new header format
        total_games_match = re.search(r'total games:\s*([\d,]+)', line_lower)
        if total_games_match:
            total = int(total_games_match.group(1).replace(',', ''))
            if total > 0 and state.total_games == 0:
                state.total_games = total

        # Extract from "[Chunk N] Processed game X/Y: TEAM @ TEAM (DATE)" pattern
        chunk_match = re.search(r'\[chunk\s*\d+\]\s*processed\s+game\s+(\d+)/(\d+):\s*(.+)', line, re.I)
        if chunk_match:
            current = int(chunk_match.group(1))
            total = int(chunk_match.group(2))
            game_info = chunk_match.group(3).strip()
            if total > 0:
                state.processed_games = current
                if state.total_games == 0:
                    state.total_games = total
                state.current_game = game_info

        # Extract season info from game processing lines
        season_match = re.search(r'\((\d{4}-\d{2}-\d{2})\)', line)
        if season_match:
            date_str = season_match.group(1)
            year = int(date_str[:4])
            month = int(date_str[5:7])
            # Determine season from date
            if month > 8:
                state.current_season = f"{year}-{year+1}"
            else:
                state.current_season = f"{year-1}-{year}"

        # Extract feature count from final output "Features: 7,460" (with optional commas)
        feat_count_match = re.search(r'features:\s*([\d,]+)', line_lower)
        if feat_count_match:
            state.features_generated = int(feat_count_match.group(1).replace(',', ''))

        # Detect warnings (but not the [PER] warnings which are expected)
        if ("warning" in line_lower or "warn" in line_lower) and "[per]" not in line_lower:
            state.warnings.append(line.strip()[:100])

        # Detect errors
        if "error" in line_lower or "exception" in line_lower or "traceback" in line_lower:
            state.error = line.strip()


def run_master_training(state: ProgressState, args, config: LeagueConfig) -> bool:
    """Run the master training generation."""
    cmd = [
        sys.executable, "cli/generate_master_training.py",
        "--league", args.league,
    ]

    # Add optional arguments
    if args.min_season:
        cmd.extend(["--min-season", args.min_season])
    elif not args.season:
        cmd.extend(["--min-season", config.min_season])

    if args.season:
        cmd.extend(["--season", args.season])

    if args.no_player:
        cmd.append("--no-player")

    if args.limit:
        cmd.extend(["--limit", str(args.limit)])

    if args.games:
        cmd.extend(["--games", args.games])

    if args.months:
        cmd.extend(["--months", args.months])

    if args.days:
        cmd.extend(["--days", args.days])

    if args.skip_months:
        cmd.extend(["--skip-months", str(args.skip_months)])

    if args.date_range:
        cmd.extend(["--date-range", args.date_range])

    if args.workers:
        cmd.extend(["--workers", str(args.workers)])

    # Always skip confirmation when running as subprocess
    cmd.append("--yes")

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env={**os.environ, "PYTHONPATH": "/Users/pranav/Documents/NBA", "PYTHONUNBUFFERED": "1"}
        )

        for line in process.stdout:
            line = line.strip()
            if line:
                parse_output_line(line, state)

        process.wait()

        if process.returncode == 0:
            with state.lock:
                state.phase = Phase.COMPLETE
                state.processed_games = state.total_games
            return True
        else:
            with state.lock:
                state.phase = Phase.FAILED
                state.error = f"Process exited with code {process.returncode}"
            return False

    except Exception as e:
        with state.lock:
            state.phase = Phase.FAILED
            state.error = str(e)
        return False


def register_master_csv(state: ProgressState, league: str) -> bool:
    """Register master CSV in MongoDB."""
    with state.lock:
        state.phase = Phase.REGISTERING

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
    except subprocess.CalledProcessError as e:
        with state.lock:
            state.warnings.append(f"Registration failed: {e.stderr[:100] if e.stderr else 'Unknown'}")
        return False


def main():
    # Get available leagues dynamically from YAML configs
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Master Training Data Regeneration with Progress Logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/regenerate_training.py nba
    python scripts/regenerate_training.py cbb --no-player
    python scripts/regenerate_training.py nba --season 2023-2024
    python scripts/regenerate_training.py cbb --min-season 2015-2016
    python scripts/regenerate_training.py nba --limit 100
        """
    )
    parser.add_argument("league", choices=available_leagues,
                       help=f"League to process ({', '.join(available_leagues)})")
    parser.add_argument("--season", type=str, default=None,
                       help="Specific season to regenerate (e.g., '2023-2024')")
    parser.add_argument("--min-season", type=str, default=None,
                       help="Minimum season to include (e.g., '2015-2016')")
    parser.add_argument("--no-player", action="store_true",
                       help="Skip player-level features (PER, injuries) for faster generation")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of games (for testing)")
    parser.add_argument("--games", type=str, default=None,
                       help="Comma-separated game window sizes (e.g., '5,10,15')")
    parser.add_argument("--months", type=str, default=None,
                       help="Comma-separated month window sizes (e.g., '1,2,3')")
    parser.add_argument("--days", type=str, default=None,
                       help="Comma-separated day window sizes (e.g., '7,14,30')")
    parser.add_argument("--skip-months", type=int, default=None,
                       help="Skip first N months of games in season")
    parser.add_argument("--date-range", type=str, default=None,
                       help="Date range to regenerate (YYYY-MM-DD,YYYY-MM-DD)")
    parser.add_argument("--skip-register", action="store_true",
                       help="Skip registering CSV in MongoDB after generation")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: CPU count, recommend 32 for fast machines)")
    args = parser.parse_args()

    league_config = load_league_config(args.league)

    # Print header
    print("\n" + "=" * 70)
    print(f"  {league_config.display_name} Master Training Regeneration")
    print("=" * 70)
    print(f"  League:      {args.league.upper()}")
    if args.season:
        print(f"  Season:      {args.season}")
    elif args.min_season:
        print(f"  Min Season:  {args.min_season}")
    else:
        print(f"  Min Season:  {league_config.min_season} (default)")
    print(f"  Player feat: {'No (faster)' if args.no_player else 'Yes'}")
    print(f"  Workers:     {args.workers or 'CPU count (default)'}")
    if args.limit:
        print(f"  Limit:       {args.limit} games")
    if args.date_range:
        print(f"  Date Range:  {args.date_range}")
    print(f"  Output:      {league_config.master_training_csv}")
    print("=" * 70 + "\n")

    # Initialize state
    state = ProgressState()
    state.league = args.league
    state.league_display = league_config.display_name

    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(target=progress_monitor, args=(state, stop_event))
    monitor_thread.start()

    success = False
    try:
        # Run master training generation
        success = run_master_training(state, args, league_config)

        # Register in MongoDB if successful and not skipped
        if success and not args.skip_register and not args.limit:
            register_master_csv(state, args.league)
            with state.lock:
                state.phase = Phase.COMPLETE

    except KeyboardInterrupt:
        with state.lock:
            state.phase = Phase.FAILED
            state.error = "Interrupted by user"
        print("\n\nInterrupted by user")
    finally:
        stop_event.set()
        monitor_thread.join()

    # Final summary
    print("\n")
    if success:
        print("✅ Master training regeneration complete!")
        print(f"   Output: {league_config.master_training_csv}")
        print(f"   Games processed: {state.processed_games:,}")
        if state.features_generated > 0:
            print(f"   Features: {state.features_generated:,}")
        print(f"   Time elapsed: {state.elapsed}")
    else:
        print("❌ Master training regeneration failed!")
        if state.error:
            print(f"   Error: {state.error}")

    if state.warnings:
        print(f"\n⚠️  {len(state.warnings)} warnings during generation")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
