#!/usr/bin/env python3
"""
Master training data generation with chunked async processing.

Uses SharedFeatureContext for efficient parallel processing:
- Pre-loads all data ONCE in main thread
- Processes in 500-row chunks with 32 workers (configurable)
- Thread-safe progress tracking

Usage:
    python -m bball_app.core.pipeline.training_pipeline nba
    python -m bball_app.core.pipeline.training_pipeline cbb --workers 16
    python -m bball_app.core.pipeline.training_pipeline nba --season 2024-2025
"""

import argparse
import fnmatch
import math
import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd

# Add project root to path for direct execution
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(os.path.dirname(script_dir))
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.league_config import load_league_config, get_available_leagues
from bball_app.core.pipeline.config import PipelineConfig, TrainingConfig
from bball_app.core.pipeline.shared_context import SharedFeatureContext
from bball_app.core.features.registry import FeatureRegistry, FeatureGroups


class PointsModelPredictor:
    """
    Handles loading and running a selected points model for prediction column generation.

    Loads the model once at startup (single DB query), then runs vectorized predictions
    using already-calculated features from the training data generation.
    """

    def __init__(self, db, league_config):
        """
        Initialize and load the selected points model if one exists.

        Args:
            db: MongoDB database connection
            league_config: League configuration
        """
        self.db = db
        self.league_config = league_config
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.trainer = None
        self.model_config = None
        self.target_type = None
        self.use_perspective_split = False

        self._load_selected_model()

    def _load_selected_model(self):
        """Load the selected points model from MongoDB (single query at startup)."""
        import os
        import pickle
        import json

        from bball_app.core.data.models import PointsConfigRepository

        # Get selected points config (single DB query)
        repo = PointsConfigRepository(self.db, league=self.league_config)
        self.model_config = repo.find_selected()

        if not self.model_config:
            print("  No selected points model found - skipping prediction columns")
            return

        # Get model artifact path
        model_path = self.model_config.get('model_artifact_path') or self.model_config.get('model_path')
        if not model_path or not os.path.exists(model_path):
            print(f"  Points model artifact not found at {model_path} - skipping prediction columns")
            return

        model_dir = os.path.dirname(model_path)
        model_stem = os.path.splitext(os.path.basename(model_path))[0]

        # Derive paths for scaler and features
        scaler_path = self.model_config.get('scaler_artifact_path') or os.path.join(model_dir, f"{model_stem}_scaler.pkl")
        features_path = self.model_config.get('features_path') or os.path.join(model_dir, f"{model_stem}_features.json")

        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            print(f"  Points model scaler or features not found - skipping prediction columns")
            return

        try:
            # Load model artifacts
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            with open(features_path, 'r') as f:
                self.feature_names = json.load(f)

            # Determine target type
            if isinstance(self.model, dict) and 'home' in self.model and 'away' in self.model:
                self.target_type = 'home_away'
            else:
                self.target_type = 'margin'

            # Check for perspective-split scalers (newer models)
            home_scaler_path = os.path.join(model_dir, f"{model_stem}_home_scaler.pkl")
            away_scaler_path = os.path.join(model_dir, f"{model_stem}_away_scaler.pkl")
            home_features_path = os.path.join(model_dir, f"{model_stem}_home_features.json")
            away_features_path = os.path.join(model_dir, f"{model_stem}_away_features.json")

            if (self.target_type == 'home_away' and
                os.path.exists(home_scaler_path) and os.path.exists(away_scaler_path) and
                os.path.exists(home_features_path) and os.path.exists(away_features_path)):

                with open(home_scaler_path, 'rb') as f:
                    self.home_scaler = pickle.load(f)
                with open(away_scaler_path, 'rb') as f:
                    self.away_scaler = pickle.load(f)
                with open(home_features_path, 'r') as f:
                    self.home_feature_names = json.load(f)
                with open(away_features_path, 'r') as f:
                    self.away_feature_names = json.load(f)
                self.use_perspective_split = True

            model_name = self.model_config.get('name', 'Unnamed')
            print(f"  Loaded points model: {model_name} ({self.target_type})")
            print(f"    Features: {len(self.feature_names)}, Perspective split: {self.use_perspective_split}")

        except Exception as e:
            print(f"  Error loading points model: {e}")
            self.model = None
            self.model_config = None

    def is_loaded(self) -> bool:
        """Check if a points model is loaded and ready for predictions."""
        return self.model is not None and self.model_config is not None

    def generate_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate prediction columns using already-calculated features.

        Uses vectorized operations for efficiency - no per-row DB calls.

        Args:
            df: DataFrame with feature columns already calculated

        Returns:
            DataFrame with added pred_home_points, pred_away_points, pred_margin, pred_point_total columns
        """
        import numpy as np

        if not self.is_loaded():
            return df

        print(f"\nGenerating prediction columns using points model...")

        def _build_feature_matrix(feature_names: List[str]) -> np.ndarray:
            """Build dense matrix in the model's expected feature order."""
            matrix = np.zeros((len(df), len(feature_names)))
            missing_count = 0
            for idx, feature_name in enumerate(feature_names):
                if feature_name in df.columns:
                    matrix[:, idx] = df[feature_name].fillna(0.0).values
                else:
                    missing_count += 1

            if missing_count > 0:
                print(f"  Warning: {missing_count}/{len(feature_names)} features not found in training data (using 0.0)")

            # Handle NaN/Inf
            if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
                matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

            return matrix

        try:
            # Build and scale feature matrices
            if self.use_perspective_split:
                home_matrix = _build_feature_matrix(self.home_feature_names)
                away_matrix = _build_feature_matrix(self.away_feature_names)
                home_matrix_scaled = self.home_scaler.transform(home_matrix)
                away_matrix_scaled = self.away_scaler.transform(away_matrix)
                feature_matrix_scaled = None
            else:
                feature_matrix = _build_feature_matrix(self.feature_names)
                feature_matrix_scaled = self.scaler.transform(feature_matrix)
                home_matrix_scaled = None
                away_matrix_scaled = None

            # Generate predictions
            if self.target_type == 'margin':
                pred_margin = self.model.predict(feature_matrix_scaled)
                pred_margin = np.clip(pred_margin, -60, 60)

                df['pred_margin'] = pred_margin
                df['pred_home_points'] = np.nan
                df['pred_away_points'] = np.nan
                df['pred_point_total'] = np.nan

                print(f"  Generated margin predictions: range [{pred_margin.min():.1f}, {pred_margin.max():.1f}]")
            else:
                # Home/away models
                if self.use_perspective_split:
                    pred_home = self.model['home'].predict(home_matrix_scaled)
                    pred_away = self.model['away'].predict(away_matrix_scaled)
                else:
                    pred_home = self.model['home'].predict(feature_matrix_scaled)
                    pred_away = self.model['away'].predict(feature_matrix_scaled)

                pred_home = np.clip(pred_home, 0, 200)
                pred_away = np.clip(pred_away, 0, 200)

                df['pred_home_points'] = pred_home
                df['pred_away_points'] = pred_away
                df['pred_margin'] = pred_home - pred_away
                df['pred_point_total'] = pred_home + pred_away

                print(f"  Generated home/away predictions:")
                print(f"    Home: [{pred_home.min():.1f}, {pred_home.max():.1f}]")
                print(f"    Away: [{pred_away.min():.1f}, {pred_away.max():.1f}]")
                print(f"    Margin: [{df['pred_margin'].min():.1f}, {df['pred_margin'].max():.1f}]")

            print(f"  Added prediction columns for {len(df)} games")

        except Exception as e:
            print(f"  Error generating predictions: {e}")
            # Add empty columns on error
            df['pred_home_points'] = np.nan
            df['pred_away_points'] = np.nan
            df['pred_margin'] = np.nan
            df['pred_point_total'] = np.nan

        return df


def expand_feature_patterns(feature_specs: List[str], league=None) -> List[str]:
    """
    Expand feature patterns (wildcards) to actual feature names.

    Supports patterns like:
    - "vegas_*" -> all features starting with "vegas_" (vegas_ML|..., vegas_spread|..., vegas_ou|...)
    - "elo*" -> all features starting with "elo"
    - "pts|season|*|home" -> all pts|season features with home perspective

    Regular feature names (no wildcards) are passed through unchanged.

    Args:
        feature_specs: List of feature names or patterns (may contain * wildcards)
        league: Optional LeagueConfig to include league-specific features

    Returns:
        List of expanded feature names (duplicates removed, order preserved)
    """
    # Get all valid features from registry (FeatureGroups has the full enumeration)
    all_features = FeatureGroups.get_all_features_flat(include_side=True, league=league)

    expanded = []
    seen = set()

    for spec in feature_specs:
        spec = spec.strip()
        if not spec:
            continue

        if '*' in spec:
            # Pattern - expand using fnmatch
            # Convert to fnmatch pattern (already uses * for wildcards)
            matched = [f for f in all_features if fnmatch.fnmatch(f, spec)]
            if matched:
                print(f"  Pattern '{spec}' matched {len(matched)} features")
                for f in matched:
                    if f not in seen:
                        expanded.append(f)
                        seen.add(f)
            else:
                print(f"  Warning: Pattern '{spec}' matched no features")
        else:
            # Exact feature name
            if spec not in seen:
                expanded.append(spec)
                seen.add(spec)

    return expanded


def process_chunk(
    chunk_df: pd.DataFrame,
    chunk_idx: int,
    feature_names: List[str],
    shared_context: SharedFeatureContext,
    progress_callback: Callable[[int], None] = None,
) -> pd.DataFrame:
    """
    Process one chunk of rows using shared context.

    Args:
        chunk_df: DataFrame chunk to process
        chunk_idx: Index of this chunk (for logging)
        feature_names: Features to calculate
        shared_context: Pre-loaded shared context
        progress_callback: Optional callback for progress updates

    Returns:
        DataFrame with features filled in
    """
    for row_idx, (idx, row) in enumerate(chunk_df.iterrows()):
        # Extract row data
        home_team = row.get('Home', row.get('homeTeam', ''))
        away_team = row.get('Away', row.get('awayTeam', ''))

        # Parse date - handle multiple formats
        date_str = row.get('Date', row.get('date', ''))
        if date_str:
            parts = str(date_str).split('-')
            if len(parts) == 3:
                year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                year, month, day = 0, 0, 0
        else:
            year = int(row.get('Year', 0))
            month = int(row.get('Month', 0))
            day = int(row.get('Day', 0))

        # Get season
        season = row.get('Season', row.get('season', ''))
        if not season and year and month:
            # Infer season from date
            if month >= 10:
                season = f"{year}-{year + 1}"
            else:
                season = f"{year - 1}-{year}"

        game_id = str(row.get('game_id')) if row.get('game_id') else None

        # Existing row data for share feature calculations
        existing_row_data = row.to_dict()

        # Use shared context to calculate features
        features_dict = shared_context.calculate_features_for_row(
            home_team=home_team,
            away_team=away_team,
            season=season,
            year=year,
            month=month,
            day=day,
            game_id=game_id,
            existing_row_data=existing_row_data,
        )

        # Update chunk DataFrame
        for feature_name in feature_names:
            if feature_name in features_dict:
                chunk_df.at[idx, feature_name] = features_dict[feature_name]

        if progress_callback:
            progress_callback(1)

    return chunk_df


def generate_training_chunked(
    df: pd.DataFrame,
    feature_names: List[str],
    config: PipelineConfig,
    progress_callback: Callable[[int, int, float], None] = None,
    target_seasons_override: List[str] = None,
) -> pd.DataFrame:
    """
    Generate training features using chunked parallel processing.

    Pattern from populate_master_training_cols.py:
    1. Create SharedFeatureContext ONCE (loads all data)
    2. Split DataFrame into chunks (default: 500 rows)
    3. Process chunks in parallel (default: 32 workers)
    4. Merge results with pd.concat (3-5x faster than iterative)

    Args:
        df: DataFrame with game rows
        feature_names: List of feature names to calculate
        config: Pipeline configuration
        progress_callback: Optional callback(processed, total, pct)
        target_seasons_override: If provided, only preload data for these seasons
            (plus lookback). Used in --add --season mode where the DataFrame
            contains all seasons but we only need to regenerate specific ones.

    Returns:
        DataFrame with features populated
    """
    chunk_size = config.training.chunk_size
    max_workers = config.training.workers

    # Ensure all feature columns exist - add all at once to avoid fragmentation
    missing_cols = [fname for fname in feature_names if fname not in df.columns]
    if missing_cols:
        # Create all missing columns at once using concat (avoids PerformanceWarning)
        new_cols = pd.DataFrame(0.0, index=df.index, columns=missing_cols)
        df = pd.concat([df, new_cols], axis=1)

    # OPTIMIZATION: In --add --season mode, only process target season rows
    # Other rows keep their existing feature values unchanged
    df_to_process = df
    df_unchanged = None
    unchanged_indices = None

    if target_seasons_override and 'Season' in df.columns:
        # Split DataFrame: rows to process vs rows to keep unchanged
        target_mask = df['Season'].isin(target_seasons_override)
        df_to_process = df[target_mask].copy()
        df_unchanged = df[~target_mask]
        unchanged_indices = df_unchanged.index

        print(f"--add --season mode: Processing {len(df_to_process):,} rows for target seasons {target_seasons_override}")
        print(f"  Keeping {len(df_unchanged):,} rows unchanged (other seasons)")

        if len(df_to_process) == 0:
            print("Warning: No rows match target seasons. Nothing to process.")
            return df

    total_rows = len(df_to_process)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    print(f"Generating {len(feature_names)} features for {total_rows:,} rows")
    print(f"  Chunks: {total_chunks} x {chunk_size} rows")
    print(f"  Workers: {max_workers}")

    # Determine which seasons to preload for efficient data access
    # Include previous season for lookback calculations (e.g., "last 10 games" at start of season)
    preload_seasons = None

    # Use override if provided (for --add --season mode where we only regenerate specific seasons)
    if target_seasons_override:
        target_seasons_for_preload = target_seasons_override
        print(f"Target seasons (from override): {sorted(target_seasons_for_preload)}")
    elif 'Season' in df_to_process.columns:
        target_seasons_for_preload = df_to_process['Season'].dropna().unique().tolist()
        print(f"Target seasons (from DataFrame): {sorted(target_seasons_for_preload)}")
    else:
        target_seasons_for_preload = None

    if target_seasons_for_preload:
        # Add previous season for each target (needed for lookback features)
        preload_seasons_set = set()
        for season in target_seasons_for_preload:
            preload_seasons_set.add(season)
            # Add previous season for lookback
            try:
                start_year = int(season.split('-')[0])
                prev_season = f"{start_year - 1}-{start_year}"
                preload_seasons_set.add(prev_season)
            except (ValueError, IndexError):
                pass

        preload_seasons = sorted(preload_seasons_set)
        print(f"Seasons to preload (including lookback): {preload_seasons}")

    # Step 1: Create shared context ONCE
    print("\n[1/4] Creating shared feature context (loads data ONCE)...")
    shared_context = SharedFeatureContext(
        feature_names=feature_names,
        league_config=config.league,
        preload_games=config.training.preload_games,
        preload_venues=config.training.preload_venues,
        preload_per_cache=config.training.preload_per_cache,
        preload_injury_cache=config.training.preload_injury_cache,
        preload_seasons=preload_seasons,
    )

    # Initialize points model predictor (single DB query at startup)
    print("Checking for selected points model...")
    points_predictor = PointsModelPredictor(shared_context.db, config.league)

    # Step 2: Pre-load venue GUIDs (for all games - lookback features may need other seasons)
    if 'game_id' in df.columns:
        game_ids = [str(gid) for gid in df['game_id'].dropna().unique()]
        print(f"\n[2/4] Pre-loading venue GUIDs for {len(game_ids):,} games...")
        shared_context.preload_venue_guids(game_ids)
    else:
        print("\n[2/4] No game_id column - skipping venue GUID preload")

    # Step 3: Split into chunks (only process df_to_process, not unchanged rows)
    print(f"\n[3/4] Processing {total_chunks} chunks with {max_workers} workers...")
    chunks: List[Tuple[int, pd.DataFrame]] = []
    for i in range(0, total_rows, chunk_size):
        chunk_df = df_to_process.iloc[i:i + chunk_size].copy()
        chunks.append((len(chunks), chunk_df))

    # Thread-safe progress tracking with detailed stats
    import time as time_module
    progress_lock = threading.Lock()
    stats = {
        'processed': 0,
        'active_workers': 0,
        'chunks_done': 0,
        'start_time': time_module.time(),
        'last_update': time_module.time(),
    }

    def format_time(seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS."""
        if seconds < 0 or seconds > 86400:  # More than 24 hours
            return "--:--"
        hours, remainder = divmod(int(seconds), 3600)
        mins, secs = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {mins:02d}m {secs:02d}s"
        return f"{mins:02d}m {secs:02d}s"

    def update_progress(n):
        with progress_lock:
            stats['processed'] += n
            now = time_module.time()

            # Throttle updates to avoid excessive output (every 100ms)
            if now - stats['last_update'] < 0.1 and stats['processed'] < total_rows:
                return
            stats['last_update'] = now

            pct = (stats['processed'] / total_rows) * 100
            elapsed = now - stats['start_time']

            # Calculate rate and ETA
            if elapsed > 0:
                rate = stats['processed'] / elapsed
                remaining = total_rows - stats['processed']
                eta = remaining / rate if rate > 0 else 0
            else:
                rate = 0
                eta = 0

            if progress_callback:
                progress_callback(stats['processed'], total_rows, pct)

            # Build progress line with detailed stats
            bar_width = 30
            filled = int(bar_width * pct / 100)
            bar = "█" * filled + "░" * (bar_width - filled)

            # Format: [████░░░░] 15.2% | 3,040/20,000 games | 152.3/s | ETA: 01m 52s | 32 workers
            progress_line = (
                f"\r  [{bar}] {pct:5.1f}% | "
                f"{stats['processed']:,}/{total_rows:,} games | "
                f"{rate:,.1f}/s | "
                f"ETA: {format_time(eta)} | "
                f"{stats['active_workers']}/{max_workers} workers"
            )
            print(progress_line, end="", flush=True)

    def on_chunk_start():
        with progress_lock:
            stats['active_workers'] += 1

    def on_chunk_done():
        with progress_lock:
            stats['active_workers'] -= 1
            stats['chunks_done'] += 1

    def process_chunk_with_tracking(chunk_df, chunk_idx, feature_names, shared_context, progress_cb):
        """Wrapper to track active workers."""
        on_chunk_start()
        try:
            return process_chunk(chunk_df, chunk_idx, feature_names, shared_context, progress_cb)
        finally:
            on_chunk_done()

    # Step 4: Process in parallel
    results: List[Tuple[int, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_chunk_with_tracking, chunk_df, chunk_idx, feature_names,
                shared_context, update_progress
            ): chunk_idx
            for chunk_idx, chunk_df in chunks
        }

        for future in as_completed(futures):
            chunk_idx = futures[future]
            try:
                result_df = future.result()
                results.append((chunk_idx, result_df))
            except Exception as e:
                print(f"\n  Warning: Chunk {chunk_idx} failed: {e}")

    # Final progress update
    elapsed = time_module.time() - stats['start_time']
    rate = total_rows / elapsed if elapsed > 0 else 0
    print(f"\n  Completed: {total_rows:,} games in {format_time(elapsed)} ({rate:,.1f} games/sec)")

    # Sort by chunk index and merge
    results.sort(key=lambda x: x[0])
    print(f"\n[4/4] Merging {len(results)} chunks...")

    # Efficient merge: concat feature columns only, then update original df
    # Deduplicate feature_names to avoid "non-unique columns" error
    unique_features = list(dict.fromkeys(feature_names))
    feature_chunks = [r[1][unique_features] for r in results]
    combined = pd.concat(feature_chunks, axis=0)

    # Remove any duplicate columns from combined DataFrame
    combined = combined.loc[:, ~combined.columns.duplicated()]

    # Also ensure df doesn't have duplicate columns for features we're setting
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    # Update only the processed rows (df_to_process indices) in the original df
    # Unchanged rows (other seasons in --add mode) keep their existing values
    df.loc[combined.index, unique_features] = combined

    if df_unchanged is not None:
        print(f"  Updated {len(combined):,} rows (target seasons)")
        print(f"  Preserved {len(df_unchanged):,} rows (other seasons, unchanged)")

    # Print normalization stats
    shared_context.print_normalization_stats()

    # Generate prediction columns if points model is loaded
    if points_predictor.is_loaded():
        df = points_predictor.generate_predictions(df)

    return df


def load_games_for_training(
    config: PipelineConfig,
    season: Optional[str] = None,
    min_season: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load games from MongoDB into a DataFrame for training generation.

    Args:
        config: Pipeline configuration
        season: Specific season to load (e.g., "2023-2024")
        min_season: Minimum season to include
        limit: Max games to load (for testing)

    Returns:
        DataFrame with game rows ready for feature generation
    """
    from bball_app.core.mongo import Mongo

    mongo = Mongo()
    db = mongo.db
    games_collection = config.league.collections.get('games', 'stats_nba')

    query = {}
    if season:
        query['season'] = season
    elif min_season:
        # Filter by min_season - need to get seasons >= min_season
        min_year = int(min_season.split('-')[0])
        query['$expr'] = {
            '$gte': [
                {'$toInt': {'$substr': ['$season', 0, 4]}},
                min_year
            ]
        }

    # Exclude certain game types if configured
    exclude_types = config.league.exclude_game_types
    if exclude_types:
        query['game_type'] = {'$nin': exclude_types}

    projection = {
        'game_id': 1,
        'season': 1,
        'date': 1,
        'homeTeam.name': 1,
        'awayTeam.name': 1,
        'homeTeam.points': 1,
        'awayTeam.points': 1,
        'homeWon': 1,
        '_id': 0
    }

    # When limit is set, skip early-season (Oct/Nov) games — too little
    # historical data for meaningful features, wastes the limited budget
    if limit:
        query['month'] = {'$nin': [10, 11]}

    print(f"Loading games from {games_collection}...")
    cursor = db[games_collection].find(query, projection).sort('date', 1)

    if limit:
        cursor = cursor.limit(limit)

    games = list(cursor)
    print(f"Loaded {len(games):,} games")

    if not games:
        return pd.DataFrame()

    # Convert to DataFrame format expected by feature generation
    rows = []
    for g in games:
        home = g.get('homeTeam', {})
        away = g.get('awayTeam', {})
        rows.append({
            'game_id': g.get('game_id'),
            'Season': g.get('season'),
            'Date': g.get('date'),
            'Home': home.get('name'),
            'Away': away.get('name'),
            'home_points': home.get('points'),
            'away_points': away.get('points'),
            'homeWon': g.get('homeWon'),
        })

    return pd.DataFrame(rows)


def get_default_features(config: PipelineConfig) -> List[str]:
    """
    Get default feature list for training generation.

    Returns features based on league configuration:
    - Core stat features (points, rebounds, etc.)
    - Player features (PER) if enabled
    - Injury features if enabled
    """
    from bball_app.core.services.training_data import TrainingDataService

    # Get base features from training data service (league-aware)
    service = TrainingDataService(league=config.league)
    features = service.get_all_possible_features()

    # Filter based on config
    if not config.training.include_player_features:
        features = [f for f in features
                   if not f.startswith('player_')
                   and not f.startswith('per_available')
                   and not f.startswith('inj_')]

    return features


def main():
    available_leagues = get_available_leagues()

    parser = argparse.ArgumentParser(
        description="Generate master training data with chunked async processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m bball_app.core.pipeline.training_pipeline nba
    python -m bball_app.core.pipeline.training_pipeline cbb --workers 16
    python -m bball_app.core.pipeline.training_pipeline nba --season 2024-2025
    python -m bball_app.core.pipeline.training_pipeline cbb --no-player
    python -m bball_app.core.pipeline.training_pipeline nba --features "elo_rating|none|raw|home,elo_rating|none|raw|away"
    python -m bball_app.core.pipeline.training_pipeline nba --add --features "col1,col2"  # Update columns only
    python -m bball_app.core.pipeline.training_pipeline nba --add --season 2024-2025     # Replace season rows
    python -m bball_app.core.pipeline.training_pipeline nba --add --seasons "2023-2024,2024-2025"
        """
    )
    parser.add_argument("league", choices=available_leagues,
                       help=f"League to process ({', '.join(available_leagues)})")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: from config or 32)")
    parser.add_argument("--chunk-size", type=int, default=None,
                       help="Rows per chunk (default: from config or 500)")
    parser.add_argument("--season", type=str, default=None,
                       help="Specific season to generate (e.g., '2023-2024')")
    parser.add_argument("--min-season", type=str, default=None,
                       help="Minimum season to include")
    parser.add_argument("--no-player", action="store_true",
                       help="Skip player-level features (PER, injuries)")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of games (for testing)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV path (default: from league config)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without generating")
    parser.add_argument("--features", type=str, default=None,
                       help="Comma-separated list of feature names to generate (default: all features)")
    parser.add_argument("--seasons", type=str, default=None,
                       help="Comma-separated list of seasons (e.g., '2023-2024,2024-2025')")
    parser.add_argument("--exclude-features", type=str, default=None,
                       help="Comma-separated list of feature names or patterns to EXCLUDE (e.g., 'player_*,inj_*')")
    parser.add_argument("--add", action="store_true",
                       help="Add/update to existing CSV: with --features updates columns, with --season/--seasons replaces season rows")
    args = parser.parse_args()

    # Validate --add usage
    if args.add:
        has_features = args.features is not None or args.exclude_features is not None
        has_seasons = args.season is not None or args.seasons is not None
        if not has_features and not has_seasons:
            parser.error("--add requires --features/--exclude-features (to update columns) or --season/--seasons (to replace season rows)")

    # Load league config
    league_config = load_league_config(args.league)
    config = PipelineConfig.from_league(league_config)

    # Override from CLI args
    if args.workers:
        config.training.workers = args.workers
    if args.chunk_size:
        config.training.chunk_size = args.chunk_size
    if args.no_player:
        config.training.include_player_features = False
        config.training.preload_per_cache = False
        config.training.preload_injury_cache = False

    # Get output path
    # If --limit is used without explicit --output, append _limitN to avoid overwriting main file
    if args.output:
        output_path = args.output
    elif args.limit:
        base_path = league_config.master_training_csv
        # Insert _limitN before .csv extension
        if base_path.endswith('.csv'):
            output_path = base_path[:-4] + f'_limit{args.limit}.csv'
        else:
            output_path = base_path + f'_limit{args.limit}'
    else:
        output_path = league_config.master_training_csv

    print("\n" + "=" * 70)
    print(f"  {league_config.display_name} Master Training Generation")
    print("=" * 70)
    print(f"  League:      {args.league.upper()}")
    print(f"  Workers:     {config.training.workers}")
    print(f"  Chunk size:  {config.training.chunk_size}")
    print(f"  Player feat: {'No' if not config.training.include_player_features else 'Yes'}")
    if args.add:
        has_feature_filter = args.features or args.exclude_features
        if has_feature_filter and (args.season or args.seasons):
            print(f"  Mode:        ADD/UPDATE columns for specific seasons only")
        elif has_feature_filter:
            print(f"  Mode:        ADD/UPDATE columns in existing CSV")
        elif args.season or args.seasons:
            print(f"  Mode:        ADD/REPLACE season rows in existing CSV")
    if args.seasons:
        print(f"  Seasons:     {args.seasons}")
    elif args.season:
        print(f"  Season:      {args.season}")
    elif args.min_season:
        print(f"  Min season:  {args.min_season}")
    else:
        print(f"  Min season:  {league_config.min_season} (default)")
    if args.limit:
        print(f"  Limit:       {args.limit} games")
    if args.features:
        feature_list = [f.strip() for f in args.features.split(',') if f.strip()]
        has_wildcards = any('*' in f for f in feature_list)
        if has_wildcards:
            print(f"  Features:    {len(feature_list)} pattern(s) (will expand wildcards)")
        else:
            print(f"  Features:    {len(feature_list)} specified")
    else:
        print(f"  Features:    all (default)")
    if args.exclude_features:
        print(f"  Excluding:   {args.exclude_features}")
    print(f"  Output:      {output_path}")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("[DRY RUN] Would generate training data with above settings")
        return 0

    # Helper to infer Season from Year/Month
    def infer_season_from_row(row):
        year, month = row['Year'], row['Month']
        if pd.isna(year) or pd.isna(month):
            return None
        year, month = int(year), int(month)
        if month >= 10:
            return f"{year}-{year + 1}"
        else:
            return f"{year - 1}-{year}"

    # Parse --seasons as a range (START,END) into list of all seasons in between
    target_seasons = None
    if args.seasons:
        parts = [s.strip() for s in args.seasons.split(',') if s.strip()]
        if len(parts) == 2:
            # Range format: "2015-2016,2024-2025" means all seasons from 2015-2016 to 2024-2025
            start_season, end_season = parts
            try:
                start_year = int(start_season.split('-')[0])
                end_year = int(end_season.split('-')[0])
                target_seasons = [f"{y}-{y+1}" for y in range(start_year, end_year + 1)]
            except (ValueError, IndexError):
                print(f"Error: Invalid --seasons format '{args.seasons}'. Expected 'START,END' like '2015-2016,2024-2025'")
                return 1
        else:
            print(f"Error: --seasons requires exactly 2 values (start,end). Got {len(parts)}: {parts}")
            return 1
    elif args.season:
        target_seasons = [args.season]

    # Track feature columns from existing CSV (for --add --seasons without --features)
    existing_feature_cols = None

    # Load data - different modes
    if args.add and (args.features or args.exclude_features):
        # --add --features mode: Load existing CSV and update columns only
        # Works with or without --season/--seasons:
        #   without seasons: updates specified features for ALL rows
        #   with seasons: updates specified features for ONLY target season rows
        if not os.path.exists(output_path):
            print(f"Error: --add requires existing CSV at {output_path}")
            return 1
        print(f"Loading existing CSV: {output_path}")
        df = pd.read_csv(output_path)
        print(f"Loaded {len(df):,} rows from existing CSV")

        # Infer Season column from Year/Month if not present (needed for preloading)
        if 'Season' not in df.columns and 'Year' in df.columns and 'Month' in df.columns:
            df['Season'] = df.apply(infer_season_from_row, axis=1)
            print(f"Inferred Season column from Year/Month")

        if target_seasons:
            season_mask = df['Season'].isin(target_seasons)
            print(f"Will update features for {season_mask.sum():,} rows in seasons: {target_seasons}")
            print(f"  Preserving all other rows and columns unchanged")

    elif args.add and target_seasons:
        # --add --season/--seasons mode: Replace season rows in existing CSV
        if not os.path.exists(output_path):
            print(f"Error: --add requires existing CSV at {output_path}")
            return 1
        print(f"Loading existing CSV: {output_path}")
        existing_df = pd.read_csv(output_path)
        print(f"Loaded {len(existing_df):,} rows from existing CSV")

        # Extract feature columns from existing CSV (for use when --features not specified)
        # Exclude metadata and target columns
        metadata_cols = {'Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'Season', 'Date'}
        target_cols = {'HomeWon', 'home_points', 'away_points'}
        pred_cols = {'pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total'}
        exclude_cols = metadata_cols | target_cols | pred_cols
        existing_feature_cols = [c for c in existing_df.columns if c not in exclude_cols]
        print(f"Detected {len(existing_feature_cols)} feature columns in existing CSV")

        # Infer Season column if not present
        if 'Season' not in existing_df.columns and 'Year' in existing_df.columns and 'Month' in existing_df.columns:
            existing_df['Season'] = existing_df.apply(infer_season_from_row, axis=1)
            print(f"Inferred Season column from Year/Month")

        # Remove rows for target seasons
        rows_before = len(existing_df)
        existing_df = existing_df[~existing_df['Season'].isin(target_seasons)]
        rows_removed = rows_before - len(existing_df)
        print(f"Removed {rows_removed:,} existing rows for seasons: {target_seasons}")

        # Load new games from DB for target seasons
        print(f"Loading new games from DB for seasons: {target_seasons}")
        new_dfs = []
        for season in target_seasons:
            season_df = load_games_for_training(config, season=season, limit=None)
            if not season_df.empty:
                new_dfs.append(season_df)
                print(f"  {season}: {len(season_df):,} games")

        if not new_dfs:
            print("No new games found for specified seasons")
            # Keep existing data without the removed seasons
            df = existing_df
        else:
            new_df = pd.concat(new_dfs, ignore_index=True)
            print(f"Total new games: {len(new_df):,}")

            # Convert Date to Year/Month/Day in new data to match existing CSV format
            if 'Date' in new_df.columns:
                new_df['Date'] = pd.to_datetime(new_df['Date'], errors='coerce')
                new_df['Year'] = new_df['Date'].dt.year.astype('Int64')
                new_df['Month'] = new_df['Date'].dt.month.astype('Int64')
                new_df['Day'] = new_df['Date'].dt.day.astype('Int64')
                new_df = new_df.drop(columns=['Date'])

            # Convert homeWon to HomeWon to match existing CSV format
            if 'homeWon' in new_df.columns:
                new_df['HomeWon'] = new_df['homeWon'].astype('Int64')
                new_df = new_df.drop(columns=['homeWon'])

            # Drop Season column from new_df (existing CSV doesn't have it, we add it back later)
            if 'Season' in new_df.columns and 'Season' not in existing_df.columns:
                new_df = new_df.drop(columns=['Season'])

            # Combine existing (minus removed seasons) with new data
            df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Combined total: {len(df):,} rows")

            # Sort chronologically
            if 'Year' in df.columns and 'Month' in df.columns and 'Day' in df.columns:
                df = df.sort_values(['Year', 'Month', 'Day']).reset_index(drop=True)
                print(f"Sorted chronologically")

    elif target_seasons:
        # Normal mode with specific seasons (--season or --seasons without --add)
        print(f"Loading games for specific seasons: {target_seasons}")
        dfs = []
        for season in target_seasons:
            season_df = load_games_for_training(config, season=season, limit=None)
            if not season_df.empty:
                dfs.append(season_df)
                print(f"  {season}: {len(season_df):,} games")
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            print(f"Total: {len(df):,} games")
        else:
            df = pd.DataFrame()

    else:
        # Normal mode: Load games from DB using min_season
        # Debug: this path is taken when neither --season nor --seasons is specified
        effective_min_season = args.min_season or league_config.min_season
        print(f"Loading games with min_season={effective_min_season} (no specific season filter)")
        df = load_games_for_training(
            config,
            season=args.season,
            min_season=effective_min_season,
            limit=args.limit,
        )

    # When --limit is used, skip early-season games (Oct/Nov) which have
    # too little historical data for meaningful feature calculation.
    # Filter BEFORE applying limit so the limit budget isn't wasted on Oct/Nov.
    if args.limit and not df.empty:
        # Determine month column: 'Month' for existing CSVs, parsed from 'Date' for fresh loads
        if 'Month' in df.columns:
            month_series = df['Month'].astype(int)
        elif 'Date' in df.columns:
            month_series = pd.to_datetime(df['Date'], errors='coerce').dt.month
        else:
            month_series = None

        if month_series is not None:
            before = len(df)
            df = df[~month_series.isin([10, 11])].reset_index(drop=True)
            skipped = before - len(df)
            if skipped > 0:
                print(f"Skipped {skipped:,} early-season (Oct/Nov) games (--limit mode)")

        df = df.head(args.limit)
        print(f"Limited to {len(df):,} rows")

    if df.empty:
        print("No games found matching criteria")
        return 1

    # Adjust chunk size to utilize all workers if dataset is small
    total_rows = len(df)
    rows_per_worker = total_rows // config.training.workers
    if rows_per_worker < config.training.chunk_size and rows_per_worker > 0:
        # Decrease chunk size so all workers get utilized
        # Use ceiling division to ensure we have at least `workers` chunks
        new_chunk_size = math.ceil(total_rows / config.training.workers)
        print(f"Adjusting chunk size: {config.training.chunk_size} -> {new_chunk_size} (to utilize {config.training.workers} workers for {total_rows} rows)")
        config.training.chunk_size = new_chunk_size

    # Get features
    if args.features:
        # Parse comma-separated feature specs (may include patterns like "vegas_*")
        feature_specs = [f.strip() for f in args.features.split(',') if f.strip()]

        # Check if any specs contain wildcards
        has_patterns = any('*' in spec for spec in feature_specs)

        if has_patterns:
            print(f"Expanding {len(feature_specs)} feature patterns...")
            features = expand_feature_patterns(feature_specs, league=league_config)
        else:
            # No patterns - use as-is (deduplicate while preserving order)
            seen = set()
            features = []
            for f in feature_specs:
                if f not in seen:
                    features.append(f)
                    seen.add(f)

        print(f"Using {len(features)} user-specified features")
    elif existing_feature_cols is not None:
        # --add --seasons mode without --features: use features from existing CSV
        features = existing_feature_cols
        print(f"Using {len(features)} features from existing CSV")
    else:
        features = get_default_features(config)

    # Apply --exclude-features filter
    if args.exclude_features:
        exclude_specs = [s.strip() for s in args.exclude_features.split(',') if s.strip()]
        before_count = len(features)
        features = [f for f in features
                    if not any(fnmatch.fnmatch(f, pat) for pat in exclude_specs)]
        excluded_count = before_count - len(features)
        print(f"Excluded {excluded_count} features matching: {', '.join(exclude_specs)}")

    print(f"Features to generate: {len(features)}")

    # Determine if this is a "features only" add (preserve structure) or full regeneration
    is_features_only_add = args.add and (args.features or args.exclude_features)

    # Track which columns are new vs updated (for --add --features mode reporting)
    if is_features_only_add:
        existing_cols = set(df.columns)
        new_cols = [f for f in features if f not in existing_cols]
        updated_cols = [f for f in features if f in existing_cols]

    # Generate features
    # Pass target_seasons to optimize preloading in --add --season mode
    # (avoids loading all seasons when only regenerating specific ones)
    df = generate_training_chunked(
        df, features, config,
        target_seasons_override=target_seasons if (args.add and target_seasons) else None
    )

    if is_features_only_add:
        # --add --features mode: Just save with updated columns (preserve existing structure)
        print(f"\nColumns added: {len(new_cols)}, updated: {len(updated_cols)}")
        if target_seasons:
            print(f"Seasons updated: {', '.join(target_seasons)}")
    else:
        # Normal mode or --add --season/--seasons: Add metadata columns and reorder

        # Add Year, Month, Day columns from Date (web UI expects these for filtering)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df['Year'] = df['Date'].dt.year.astype('Int64')
            df['Month'] = df['Date'].dt.month.astype('Int64')
            df['Day'] = df['Date'].dt.day.astype('Int64')
            df = df.drop(columns=['Date'])

        # Add HomeWon column (web UI expects this for classification models)
        # Use homeWon from DB if available (more reliable), otherwise compute from points
        if 'homeWon' in df.columns:
            # Use Int64 to handle potential NaN values
            df['HomeWon'] = df['homeWon'].astype('Int64')
            df = df.drop(columns=['homeWon'])
        elif 'home_points' in df.columns and 'away_points' in df.columns:
            df['HomeWon'] = (df['home_points'] > df['away_points']).astype('Int64')

        # Reorder columns: metadata first, then features, then predictions, then targets
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id']
        pred_cols = ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total']
        target_cols = ['HomeWon', 'home_points', 'away_points']
        existing_meta = [c for c in meta_cols if c in df.columns]
        existing_pred = [c for c in pred_cols if c in df.columns]
        existing_targets = [c for c in target_cols if c in df.columns]
        excluded_cols = meta_cols + pred_cols + target_cols + ['Season']
        feature_cols = [c for c in df.columns if c not in excluded_cols]
        df = df[existing_meta + feature_cols + existing_pred + existing_targets]

        if args.add and target_seasons:
            print(f"\nReplaced {len(target_seasons)} season(s): {', '.join(target_seasons)}")

    # Filter out unplayed games (NaN targets) before saving
    # These are future games or games without final scores
    rows_before = len(df)
    if 'HomeWon' in df.columns:
        df = df.dropna(subset=['HomeWon'])
    if 'home_points' in df.columns and 'away_points' in df.columns:
        df = df.dropna(subset=['home_points', 'away_points'])
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(f"\nDropped {rows_dropped} unplayed/future games (NaN targets)")

    # Save to CSV
    # Count prediction columns
    pred_cols_present = [c for c in ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total'] if c in df.columns]

    print(f"\nSaving to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    summary_parts = [f"{len(df):,} rows", f"{len(features)} features"]
    if pred_cols_present:
        summary_parts.append(f"{len(pred_cols_present)} pred columns")
    print(f"Saved {', '.join(summary_parts)}")

    print("\n" + "=" * 70)
    print("  Training generation complete!")
    print("=" * 70 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
