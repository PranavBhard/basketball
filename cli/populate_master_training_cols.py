#!/usr/bin/env python3
"""
Script to populate additional columns in the master training CSV.

Supports:
- Feature names (e.g., 'points|season|avg|diff') - calculated on-demand
- Metadata columns: 'game_id', 'home_points', 'away_points' - extracted from MongoDB
- Prediction columns: 'pred_home_points', 'pred_away_points', 'pred_margin', 'pred_total' - from selected point model

Usage:
    python cli/populate_master_training_cols.py \
        --columns "game_id,home_points,away_points,points|season|avg|diff" \
        [--overwrite] \
        [--backup] \
        [--master-csv PATH]
    
    python cli/populate_master_training_cols.py \
        --columns "pred_home_points,pred_away_points,pred_margin,pred_total" \
        [--overwrite] \
        [--backup]
"""

import sys
import os
import argparse
import pandas as pd
import shutil
import threading
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from bson import ObjectId
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball_app.core.mongo import Mongo
from bball_app.core.models.bball_model import BballModel
from bball_app.core.services.training_data import MASTER_TRAINING_PATH, get_all_possible_features
from bball_app.core.features.dependencies import resolve_dependencies, categorize_features
from bball_app.core.features.registry import FeatureRegistry
from bball_app.core.features.prediction_mapping import (
    is_pred_feature,
    parse_pred_feature,
    validate_pred_feature_model_type,
    get_pred_internal_columns,
    LEGACY_PRED_COLUMNS,
)
import numpy as np


# =============================================================================
# SHARED FEATURE CONTEXT - Pre-loads all data ONCE for efficient parallel processing
# =============================================================================

class SharedFeatureContext:
    """
    Pre-loads all necessary data for feature calculation ONCE, then shares
    across multiple worker threads. This avoids each thread creating its own
    MongoDB connections and loading the same 500K+ records.

    Usage:
        # Main thread - create once
        context = SharedFeatureContext(feature_names)

        # Worker threads - use shared context (read-only)
        features = context.calculate_features_for_row(row_data)
    """

    def __init__(self, feature_names: List[str], preload_data: bool = True):
        """
        Initialize shared feature context by pre-loading all necessary data.

        Args:
            feature_names: List of feature names that will be calculated
            preload_data: If True, preload games and player stats into memory
        """
        from bball_app.core.mongo import Mongo
        from bball_app.core.stats.handler import StatHandlerV2
        from bball_app.core.stats.per_calculator import PERCalculator
        from bball_app.core.data import GamesRepository, RostersRepository
        from bball_app.core.utils.collection import import_collection
        from bball_app.core.features.parser import parse_feature_name

        self.feature_names = feature_names

        # Infer what components are needed
        self._needs_per = any(
            f.startswith("player_") or f.startswith("per_available") or
            f.split("|", 1)[0].lower().endswith("_per")
            for f in feature_names
        )
        self._needs_injuries = any(f.startswith("inj_") for f in feature_names)
        self._needs_elo = any(f.split("|", 1)[0].lower().startswith("elo") for f in feature_names)

        print("=" * 60)
        print("INITIALIZING SHARED FEATURE CONTEXT (loads data ONCE)")
        print("=" * 60)

        # Single MongoDB connection
        print("Connecting to MongoDB...")
        self.mongo = Mongo()
        self.db = self.mongo.db
        print("Connected to MongoDB.")

        # Build team name normalization map (displayName -> abbreviation)
        # This allows handling both full names ("Milwaukee Bucks") and abbreviations ("MIL")
        self._team_name_map = {}
        self._normalization_count = 0  # Track how many normalizations happen
        self._normalization_examples = set()  # Track unique normalized team names
        try:
            for team in self.db['nba_teams'].find({}, {'displayName': 1, 'abbreviation': 1}):
                display_name = team.get('displayName', '')
                abbr = team.get('abbreviation', '')
                if display_name and abbr:
                    self._team_name_map[display_name] = abbr
                    self._team_name_map[display_name.lower()] = abbr  # case-insensitive
            print(f"Loaded {len(self._team_name_map) // 2} team name mappings.")
        except Exception as e:
            print(f"Warning: Could not load team name mappings: {e}")

        # Initialize repositories
        self._games_repo = GamesRepository(self.db)
        self._rosters_repo = RostersRepository(self.db)

        # Load game data once
        self.all_games = None
        if preload_data:
            print("Loading game data from database (this may take a moment)...")
            self.all_games = import_collection('stats_nba')
            print("Loaded game data.")

        # Initialize stat handler with preloaded games
        print("Initializing shared stat handler...")
        self.stat_handler = StatHandlerV2(
            statistics=[],
            use_exponential_weighting=False,
            preloaded_games=self.all_games,
            db=self.db,
            lazy_load=(not preload_data)
        )

        # Preload venue cache
        print("Preloading venue cache...")
        try:
            self.stat_handler.preload_venue_cache()
        except Exception:
            pass

        # Initialize PER calculator if needed
        self.per_calculator = None
        if self._needs_per or self._needs_injuries:
            print("Initializing shared PER calculator (preloading player stats)...")
            self.per_calculator = PERCalculator(self.db, preload=preload_data)

        # Preload injury cache if needed (for season injury severity calculations)
        if self._needs_injuries and preload_data and self.all_games:
            print("Preloading injury cache (player stats by team-season)...")
            # Convert all_games tuple to flat list for preload_injury_cache
            games_list = []
            games_home, games_away = self.all_games
            for season_data in games_home.values():
                for date_data in season_data.values():
                    for game in date_data.values():
                        games_list.append(game)
            self.stat_handler.preload_injury_cache(games_list)

        # Season severity uses lazy caching in stat_handler._get_season_injury_severity()
        # The cache is shared across all threads, so first call computes, subsequent calls hit cache
        # This is more efficient than pre-computing 40K+ combinations upfront
        self._precomputed_season_severity = {}

        # Pre-load venue_guids for travel features (will be populated per batch)
        self.venue_guid_cache = {}

        print("=" * 60)
        print("SHARED CONTEXT READY - Workers will use cached data")
        print("=" * 60)

    def normalize_team_name(self, team_name: str) -> str:
        """
        Normalize team name to abbreviation format.

        Handles:
        - Full names: "Milwaukee Bucks" -> "MIL"
        - Already abbreviated: "MIL" -> "MIL" (no change)
        - Case insensitive: "milwaukee bucks" -> "MIL"

        Returns original if no mapping found.
        """
        if not team_name:
            return team_name

        # Check if it's a full name that needs mapping
        if team_name in self._team_name_map:
            abbr = self._team_name_map[team_name]
            self._normalization_count += 1
            if len(self._normalization_examples) < 5:
                self._normalization_examples.add(f"{team_name} -> {abbr}")
            return abbr

        # Try case-insensitive
        if team_name.lower() in self._team_name_map:
            abbr = self._team_name_map[team_name.lower()]
            self._normalization_count += 1
            if len(self._normalization_examples) < 5:
                self._normalization_examples.add(f"{team_name} -> {abbr}")
            return abbr

        # Already an abbreviation or unknown - return as-is
        return team_name

    def print_normalization_stats(self):
        """Print statistics about team name normalizations."""
        if self._normalization_count > 0:
            print(f"\n[DIAGNOSTIC] Team name normalizations: {self._normalization_count}")
            print(f"  Examples: {list(self._normalization_examples)}")
            print("  This confirms team names in CSV needed conversion from full names to abbreviations.")
        else:
            print("\n[DIAGNOSTIC] No team name normalizations needed (all names already abbreviated).")

    def preload_venue_guids(self, game_ids: List[str]):
        """
        Pre-load venue GUIDs for a batch of games.
        Called once per batch, not per row.
        """
        if not game_ids:
            return

        games_with_venue = self.db.stats_nba.find(
            {'game_id': {'$in': game_ids}},
            {'game_id': 1, 'venue_guid': 1}
        )
        for g in games_with_venue:
            if g.get('venue_guid'):
                self.venue_guid_cache[str(g['game_id'])] = g['venue_guid']

    def calculate_features_for_row(
        self,
        home_team: str,
        away_team: str,
        season: str,
        year: int,
        month: int,
        day: int,
        game_id: str = None,
        venue_guid: str = None,
        existing_row_data: Dict[str, float] = None
    ) -> Dict[str, float]:
        """
        Calculate features for a single row using the shared pre-loaded data.
        This method is thread-safe for read operations.

        Args:
            existing_row_data: Optional dict of existing column values from the row.
                Used to calculate share features from existing raw injury values.

        Returns:
            Dict mapping feature names to their values
        """
        from bball_app.core.features.parser import parse_feature_name

        features_dict = {}
        game_date_str = f"{year}-{month:02d}-{day:02d}"

        # Normalize team names to abbreviations (handles both "Milwaukee Bucks" and "MIL")
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)

        # Use cached venue_guid if available
        if venue_guid is None and game_id:
            venue_guid = self.venue_guid_cache.get(str(game_id))

        # Calculate each feature
        for feature_name in self.feature_names:
            # Handle special non-pipe features
            if '|' not in feature_name:
                if feature_name == 'SeasonStartYear':
                    features_dict[feature_name] = int(year) if int(month) >= 10 else int(year) - 1
                elif feature_name == 'Year':
                    features_dict[feature_name] = int(year)
                elif feature_name == 'Month':
                    features_dict[feature_name] = int(month)
                elif feature_name == 'Day':
                    features_dict[feature_name] = int(day)
                elif feature_name == 'pred_margin':
                    features_dict[feature_name] = 0.0
                else:
                    features_dict[feature_name] = 0.0
                continue

            # Skip PER and injury features - handled separately
            if feature_name.startswith('player_') or feature_name.startswith('per_available'):
                continue
            if feature_name.startswith('inj_'):
                continue

            # Regular stat feature - use shared stat handler
            try:
                value = self.stat_handler.calculate_feature(
                    feature_name, home_team, away_team, season,
                    year, month, day, self.per_calculator, venue_guid
                )
                features_dict[feature_name] = value if value is not None else 0.0
            except Exception:
                features_dict[feature_name] = 0.0

        # Add PER features if needed
        has_per_features = any(
            fname.startswith('player_') or fname.startswith('per_available')
            for fname in self.feature_names
        )
        if has_per_features and self.per_calculator:
            # Get injured players from game doc (for training data)
            injured_players_dict = None
            try:
                game_doc = self._games_repo.find_one({
                    'homeTeam.name': home_team,
                    'awayTeam.name': away_team,
                    'season': season,
                    'date': game_date_str
                })
                if game_doc:
                    home_injured = game_doc.get('homeTeam', {}).get('injured_players', [])
                    away_injured = game_doc.get('awayTeam', {}).get('injured_players', [])
                    if home_injured or away_injured:
                        injured_players_dict = {
                            home_team: [str(pid) for pid in home_injured] if home_injured else [],
                            away_team: [str(pid) for pid in away_injured] if away_injured else []
                        }
            except Exception:
                pass

            # Calculate PER features
            try:
                per_features = self.per_calculator.get_game_per_features(
                    home_team=home_team,
                    away_team=away_team,
                    season=season,
                    game_date=game_date_str,
                    player_filters=None,
                    injured_players=injured_players_dict
                )
                if per_features:
                    for fname in self.feature_names:
                        if fname.startswith('player_') or fname.startswith('per_available'):
                            if fname in per_features:
                                features_dict[fname] = per_features[fname]
                            else:
                                features_dict[fname] = 0.0
            except Exception:
                # Default PER features to 0 on error
                for fname in self.feature_names:
                    if fname.startswith('player_') or fname.startswith('per_available'):
                        features_dict[fname] = 0.0

        # Add injury features if needed
        has_injury_features = any(fname.startswith('inj_') for fname in self.feature_names)
        if has_injury_features:
            # ONE-TIME VERIFICATION: Print cache state on first call
            if not hasattr(self, '_cache_verified'):
                self._cache_verified = True
                sh = self.stat_handler
                print(f"\n[CACHE VERIFICATION]")
                print(f"  stat_handler.games_home is set: {sh.games_home is not None}")
                print(f"  stat_handler.games_away is set: {sh.games_away is not None}")
                print(f"  stat_handler._injury_cache_loaded: {getattr(sh, '_injury_cache_loaded', 'NOT SET')}")
                inj_cache = getattr(sh, '_injury_preloaded_players', {})
                print(f"  stat_handler._injury_preloaded_players count: {len(inj_cache)}")
                if self.per_calculator:
                    pc = self.per_calculator
                    print(f"  per_calculator._preloaded: {getattr(pc, '_preloaded', 'NOT SET')}")
                    print(f"  per_calculator._player_stats_cache count: {len(getattr(pc, '_player_stats_cache', {}))}")

                # Test actual lookup with current row data
                test_key = (home_team, season)
                print(f"  TEST: Looking up ({home_team}, {season}) in injury cache...")
                if test_key in inj_cache:
                    print(f"    FOUND: {len(inj_cache[test_key])} player records")
                else:
                    print(f"    NOT FOUND! Available keys sample: {list(inj_cache.keys())[:5]}")
                print()

            try:
                # Get game doc for injury data from PRELOADED data (no DB call!)
                game_doc = None
                if self.all_games is not None:
                    # Use preloaded games - structure is (games_home, games_away)
                    # games_home[season][date][team] = game_doc
                    games_home, games_away = self.all_games
                    if season in games_home and game_date_str in games_home[season]:
                        if home_team in games_home[season][game_date_str]:
                            game_doc = games_home[season][game_date_str][home_team]
                else:
                    # Fallback to DB query only if preloaded data not available
                    try:
                        game_doc = self._games_repo.find_one({
                            'homeTeam.name': home_team,
                            'awayTeam.name': away_team,
                            'season': season,
                            'date': game_date_str
                        })
                    except Exception:
                        pass

                # Pass precomputed_season_severity dict even if empty - lazy caching happens inside
                # stat_handler._get_season_injury_severity() and uses its own cache
                injury_features = self.stat_handler.get_injury_features(
                    home_team, away_team, season, year, month, day,
                    game_doc=game_doc,
                    per_calculator=self.per_calculator,
                    precomputed_season_severity=None  # Let stat_handler use its internal lazy cache
                )
                if injury_features:
                    for fname in self.feature_names:
                        if fname.startswith('inj_'):
                            if fname in injury_features:
                                features_dict[fname] = injury_features[fname]
                            else:
                                features_dict[fname] = 0.0

                # SPECIAL HANDLING: Calculate share features from existing raw values
                # If share features are 0 but we have existing raw values in the row,
                # calculate shares using existing numerators + freshly calculated denominators
                if existing_row_data and self.per_calculator:
                    share_features_to_fix = [
                        ('inj_per_share|none|top3_sum|home', 'inj_per|none|top3_sum|home'),
                        ('inj_per_share|none|top3_sum|away', 'inj_per|none|top3_sum|away'),
                        ('inj_per_weighted_share|none|weighted_MIN|home', 'inj_per|none|weighted_MIN|home'),
                        ('inj_per_weighted_share|none|weighted_MIN|away', 'inj_per|none|weighted_MIN|away'),
                    ]

                    # Get team PER baselines for denominators (if not already done)
                    home_top3_sum = 0.0
                    away_top3_sum = 0.0
                    home_per_weighted = 0.0
                    away_per_weighted = 0.0
                    denominators_fetched = False

                    for share_feature, raw_feature in share_features_to_fix:
                        if share_feature not in self.feature_names:
                            continue

                        # Check if share is 0 but raw value exists
                        current_share = features_dict.get(share_feature, 0.0)
                        existing_raw = existing_row_data.get(raw_feature, 0.0)

                        if current_share == 0.0 and existing_raw != 0.0:
                            # Fetch denominators once
                            if not denominators_fetched:
                                try:
                                    home_per = self.per_calculator.compute_team_per_features(
                                        home_team, season, game_date_str
                                    )
                                    away_per = self.per_calculator.compute_team_per_features(
                                        away_team, season, game_date_str
                                    )
                                    if home_per:
                                        home_top3_sum = home_per.get('top3_sum', 0.0)
                                        home_per_weighted = home_per.get('per_weighted', 0.0)
                                    if away_per:
                                        away_top3_sum = away_per.get('top3_sum', 0.0)
                                        away_per_weighted = away_per.get('per_weighted', 0.0)
                                except Exception:
                                    pass
                                denominators_fetched = True

                            # Calculate share from existing raw value
                            EPS = 1e-6
                            if 'top3_sum|home' in share_feature:
                                if home_top3_sum > 0:
                                    features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (home_top3_sum + EPS)))
                            elif 'top3_sum|away' in share_feature:
                                if away_top3_sum > 0:
                                    features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (away_top3_sum + EPS)))
                            elif 'weighted_MIN|home' in share_feature:
                                if home_per_weighted > 0:
                                    features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (home_per_weighted + EPS)))
                            elif 'weighted_MIN|away' in share_feature:
                                if away_per_weighted > 0:
                                    features_dict[share_feature] = min(1.5, max(0.0, existing_raw / (away_per_weighted + EPS)))

                    # Calculate diff features from home/away
                    if 'inj_per_share|none|top3_sum|diff' in self.feature_names:
                        home_share = features_dict.get('inj_per_share|none|top3_sum|home', 0.0)
                        away_share = features_dict.get('inj_per_share|none|top3_sum|away', 0.0)
                        features_dict['inj_per_share|none|top3_sum|diff'] = home_share - away_share

                    if 'inj_per_weighted_share|none|weighted_MIN|diff' in self.feature_names:
                        home_share = features_dict.get('inj_per_weighted_share|none|weighted_MIN|home', 0.0)
                        away_share = features_dict.get('inj_per_weighted_share|none|weighted_MIN|away', 0.0)
                        features_dict['inj_per_weighted_share|none|weighted_MIN|diff'] = home_share - away_share

            except Exception:
                # Default injury features to 0 on error
                for fname in self.feature_names:
                    if fname.startswith('inj_'):
                        features_dict[fname] = 0.0

        # Ensure all requested features have a value
        for fname in self.feature_names:
            if fname not in features_dict:
                features_dict[fname] = 0.0

        return features_dict


def validate_features_against_registry(feature_names: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Validate feature names against FeatureRegistry (SSoT).

    All features must be defined in the FeatureRegistry before they can be
    added to the master training CSV. This ensures consistency and prevents
    orphan features.

    For prediction features (pred_*), also validates that the model type
    matches the selected points model in model_config_points_nba.

    Args:
        feature_names: List of feature names to validate

    Returns:
        Tuple of (valid_features, invalid_features_with_errors)
        - valid_features: List of features that passed validation
        - invalid_features_with_errors: List of (feature_name, error_message) tuples
    """
    # Basic metadata columns that don't need registry validation
    metadata_cols = {'game_id', 'home_points', 'away_points'}

    valid_features = []
    invalid_features = []

    # Check if we have any prediction features that need model type validation
    pred_features_registry_format = [
        f for f in feature_names
        if is_pred_feature(f) and not parse_pred_feature(f).get("is_legacy", True)
    ]

    selected_points_model_type = None
    if pred_features_registry_format:
        # Load selected points model config for validation
        mongo = Mongo()
        db = mongo.db
        selected_config = db.model_config_points_nba.find_one({'selected': True})
        if selected_config:
            selected_points_model_type = selected_config.get('model_type')
        else:
            # No selected model - registry-format pred features are invalid
            for feature in pred_features_registry_format:
                invalid_features.append((
                    feature,
                    "No points model selected. Select a points model in Points Model Config first, "
                    "or use legacy format (e.g., 'pred_margin' instead of 'pred_margin|none|ridge|none')."
                ))
            # Remove from feature_names to avoid double-processing
            feature_names = [f for f in feature_names if f not in pred_features_registry_format]

    for feature in feature_names:
        # Skip basic metadata columns
        if feature in metadata_cols:
            valid_features.append(feature)
            continue

        # Handle prediction features
        if is_pred_feature(feature):
            parsed = parse_pred_feature(feature)
            if not parsed:
                invalid_features.append((feature, f"Invalid prediction feature format: {feature}"))
                continue

            # Legacy format - always valid (no model type validation)
            if parsed["is_legacy"]:
                valid_features.append(feature)
                continue

            # Registry format - validate model type matches selected model
            if selected_points_model_type:
                is_valid, error = validate_pred_feature_model_type(feature, selected_points_model_type)
                if is_valid:
                    valid_features.append(feature)
                else:
                    invalid_features.append((feature, error))
            continue

        # Validate regular features against FeatureRegistry
        is_valid, error = FeatureRegistry.validate_feature(feature)
        if is_valid:
            valid_features.append(feature)
        else:
            invalid_features.append((feature, error))

    return valid_features, invalid_features


def _infer_nba_model_flags(feature_names: List[str]) -> Dict[str, bool]:
    """
    Infer which heavyweight BballModel components are actually needed for this feature batch.

    This is important for the master-training "add columns" workflow: we don't want to
    preload player stats / PER calculator when regenerating team-level-only features
    like travel/rest.
    """
    names = [f or "" for f in feature_names]
    names_lower = [f.lower() for f in names]

    def _stat_name(f: str) -> str:
        return f.split("|", 1)[0].lower() if "|" in f else f.lower()

    stat_names = [_stat_name(f) for f in names]

    # Player/PER features
    needs_per = any(
        f.startswith("player_") or f.startswith("per_available") or _stat_name(f).endswith("_per")
        for f in names_lower
    )

    # Injury features
    needs_injuries = any(f.startswith("inj_") for f in names_lower)

    # Elo features
    needs_elo = any(sn.startswith("elo") for sn in stat_names)

    # Preloading all game docs is expensive and (for small team-level batches) unnecessary.
    # Heuristic: preload when batch is large or when we know we need more global context.
    preload_data = (len(feature_names) >= 50) or needs_per or needs_injuries

    return {
        "needs_per": needs_per,
        "needs_injuries": needs_injuries,
        "needs_elo": needs_elo,
        "preload_data": preload_data,
    }


def update_job_progress(job_id: str, progress: int, message: str = None, db=None):
    """
    Update job progress and message.

    Args:
        job_id: Job ID
        progress: Progress percentage (0-100)
        message: Optional status message
        db: MongoDB database connection
    """
    if db is None:
        print(f"WARNING: update_job_progress called with db=None for job {job_id}")
        return
    
    update_doc = {
        'progress': max(0, min(100, progress)),
        'updated_at': datetime.utcnow()
    }
    if message:
        update_doc['message'] = message
    
    db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': update_doc}
    )


def complete_job(job_id: str, message: str = 'Job completed successfully', db=None):
    """Mark job as completed."""
    if db is None:
        print(f"WARNING: complete_job called with db=None for job {job_id}. Job will remain in 'running' state!")
        return
    
    db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'status': 'completed',
            'progress': 100,
            'message': message,
            'updated_at': datetime.utcnow()
        }}
    )


def fail_job(job_id: str, error: str, message: str = None, db=None):
    """Mark job as failed with error message."""
    if db is None:
        print(f"WARNING: fail_job called with db=None for job {job_id}. Job will remain in 'running' state!")
        return
    
    db.jobs_nba.update_one(
        {'_id': ObjectId(job_id)},
        {'$set': {
            'status': 'failed',
            'error': error,
            'message': message or f'Job failed: {error}',
            'updated_at': datetime.utcnow()
        }}
    )


def find_features_by_substrings(features: List[str], substrings: List[str], match_mode: str = 'OR') -> List[str]:
    """
    Find features that match any or all of the given substrings.
    
    Args:
        features: List of all feature names to search
        substrings: List of substrings to match against (case-insensitive)
        match_mode: 'OR' to match features containing ANY substring, 'AND' to match features containing ALL substrings
        
    Returns:
        List of matching feature names
    """
    if not substrings:
        return []
    
    # Normalize substrings to lowercase for case-insensitive matching
    substrings_lower = [s.lower().strip() for s in substrings if s.strip()]
    
    if not substrings_lower:
        return []
    
    matching_features = []
    
    for feature in features:
        feature_lower = feature.lower()
        
        if match_mode.upper() == 'AND':
            # AND mode: feature must contain ALL substrings
            if all(substring in feature_lower for substring in substrings_lower):
                matching_features.append(feature)
        else:
            # OR mode (default): feature must contain ANY substring
            if any(substring in feature_lower for substring in substrings_lower):
                matching_features.append(feature)
    
    return matching_features


def extract_metadata_from_mongodb(
    df: pd.DataFrame,
    columns: List[str],
    db
) -> pd.DataFrame:
    """
    Extract metadata columns (game_id, home_points, away_points) from MongoDB.
    
    Args:
        df: DataFrame with existing master training data
        columns: List of metadata columns to extract
        db: MongoDB database connection
        
    Returns:
        DataFrame with new columns added
    """
    print(f"Loading games from MongoDB for {len(df)} rows...")
    
    # Get date range from dataframe for efficient batch query
    min_year = int(df['Year'].min())
    max_year = int(df['Year'].max())
    print(f"  Date range: {min_year}-{max_year}")
    
    # Build lookup maps: load all games in date range once and create lookup by (year, month, day, home, away)
    game_lookup_by_date_teams = {}  # (year, month, day, home, away) -> game doc
    game_lookup_by_id = {}  # game_id -> game doc
    
    # Get game_ids from dataframe if available
    game_ids_in_df = set()
    if 'game_id' in df.columns:
        for game_id in df['game_id'].dropna():
            if game_id and str(game_id).strip():
                game_ids_in_df.add(str(game_id).strip())
    
    # Strategy: Query all games in the date range, then build lookup maps in memory
    # This is much faster than querying per combination
    print("  Loading all games from MongoDB in date range (single batch query)...")
    query = {
        'year': {'$gte': min_year, '$lte': max_year}
    }
    
    # Add game_id filter if we have game_ids (can help with indexing)
    if game_ids_in_df and len(game_ids_in_df) < 50000:  # Only if reasonable number
        query['$or'] = [
            {'year': {'$gte': min_year, '$lte': max_year}},
            {'game_id': {'$in': list(game_ids_in_df)}}
        ]
    
    all_games = list(db.stats_nba.find(query))
    print(f"  Loaded {len(all_games)} games from MongoDB")
    
    # Build lookup maps from loaded games
    print("  Building lookup maps...")
    for game in all_games:
        year = game.get('year')
        month = game.get('month')
        day = game.get('day')
        home_team = game.get('homeTeam', {}).get('name', '')
        away_team = game.get('awayTeam', {}).get('name', '')
        
        if year and month and day and home_team and away_team:
            key = (year, month, day, home_team, away_team)
            game_lookup_by_date_teams[key] = game
        
        # Also index by game_id if available
        game_id = game.get('game_id', '') or ''
        if game_id:
            game_lookup_by_id[game_id] = game
    
    print(f"  Built lookup maps: {len(game_lookup_by_date_teams)} by date/team, {len(game_lookup_by_id)} by game_id")
    
    # Now populate columns using the lookup maps
    print(f"\nPopulating metadata columns: {columns}")
    total_rows = len(df)
    
    # Initialize columns
    for col in columns:
        if col == 'game_id':
            df['game_id'] = ''
        elif col == 'home_points':
            df['home_points'] = 0
        elif col == 'away_points':
            df['away_points'] = 0
    
    # Populate values using lookup maps
    matched_count = 0
    for idx, row in df.iterrows():
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_rows:
            print(f"  Progress: {idx + 1}/{total_rows} rows ({100 * (idx + 1) / total_rows:.1f}%)")
        
        year = int(row['Year'])
        month = int(row['Month'])
        day = int(row['Day'])
        home_team = row['Home']
        away_team = row['Away']
        
        # Try to find game: first by game_id if available, then by date/teams
        game = None
        
        # Try game_id lookup first if game_id exists in row
        if 'game_id' in row and pd.notna(row['game_id']) and row['game_id']:
            game_id = str(row['game_id'])
            game = game_lookup_by_id.get(game_id)
        
        # Fallback to date/team lookup
        if not game:
            key = (year, month, day, home_team, away_team)
            game = game_lookup_by_date_teams.get(key)
        
        if game:
            matched_count += 1
            # Extract requested columns
            for col in columns:
                if col == 'game_id':
                    game_id = game.get('game_id', '') or ''
                    df.at[idx, 'game_id'] = game_id
                elif col == 'home_points':
                    home_points = game.get('homeTeam', {}).get('points', 0) or 0
                    df.at[idx, 'home_points'] = home_points
                elif col == 'away_points':
                    away_points = game.get('awayTeam', {}).get('points', 0) or 0
                    df.at[idx, 'away_points'] = away_points
    
    print(f"\n  Matched {matched_count}/{total_rows} rows ({100 * matched_count / total_rows:.1f}%)")
    
    return df


def _process_feature_chunk(
    chunk_df: pd.DataFrame,
    chunk_idx: int,
    feature_names: List[str],
    total_rows: int,
    start_idx: int,
    progress_callback: callable = None,
    shared_context: 'SharedFeatureContext' = None
) -> pd.DataFrame:
    """
    Process a chunk of DataFrame rows and calculate features.
    This function is designed to be called in parallel by multiple threads.

    Args:
        chunk_df: DataFrame slice with rows to process
        chunk_idx: Index of this chunk (for logging)
        feature_names: List of feature names to calculate
        total_rows: Total number of rows being processed (for progress calculation)
        start_idx: Starting row index for this chunk (for progress calculation)
        progress_callback: Optional callback function to report progress (processed, total)
        shared_context: SharedFeatureContext with pre-loaded data (REQUIRED for optimized processing)

    Returns:
        DataFrame with calculated feature columns
    """
    # OPTIMIZED PATH: Use shared context (pre-loaded data, no new MongoDB connections)
    if shared_context is not None:
        # Initialize feature columns if needed
        for feature_name in feature_names:
            if feature_name not in chunk_df.columns:
                chunk_df[feature_name] = 0.0

        # Calculate features for each row using shared context
        chunk_size = len(chunk_df)
        last_reported = 0
        report_interval = max(1, chunk_size // 10) if chunk_size > 10 else 1

        for row_idx, (idx, row) in enumerate(chunk_df.iterrows()):
            year = int(row['Year'])
            month = int(row['Month'])
            day = int(row['Day'])
            home_team = row['Home']
            away_team = row['Away']
            game_id = row.get('game_id') if 'game_id' in row.index else None

            # Determine season from year/month
            if month >= 10:
                season = f"{year}-{year+1}"
            else:
                season = f"{year-1}-{year}"

            # Use shared context to calculate features (NO MongoDB calls here!)
            # Convert row to dict for passing existing values to calculate share features
            existing_row_data = row.to_dict()
            features_dict = shared_context.calculate_features_for_row(
                home_team=home_team,
                away_team=away_team,
                season=season,
                year=year,
                month=month,
                day=day,
                game_id=str(game_id) if game_id else None,
                existing_row_data=existing_row_data
            )

            # Update feature values
            for feature_name in feature_names:
                if features_dict and feature_name in features_dict:
                    value = features_dict[feature_name]
                    # Handle NaN/Inf
                    if pd.isna(value) or (isinstance(value, float) and (value != value or abs(value) == float('inf'))):
                        value = 0.0
                    chunk_df.at[idx, feature_name] = value

            # Report progress periodically
            rows_processed_in_chunk = row_idx + 1
            if progress_callback and (rows_processed_in_chunk - last_reported >= report_interval or rows_processed_in_chunk == chunk_size):
                progress_callback(rows_processed_in_chunk - last_reported, total_rows)
                last_reported = rows_processed_in_chunk

        return chunk_df

    # LEGACY PATH: Create BballModel per thread (kept for backward compatibility)
    # This path is SLOW and should be avoided - use shared_context instead
    print(f"  [Chunk {chunk_idx}] WARNING: Using legacy path (no shared context) - this is slow!")
    flags = _infer_nba_model_flags(feature_names)
    model = BballModel(
        classifier_features=feature_names,
        include_elo=flags["needs_elo"],
        include_per_features=flags["needs_per"],
        include_injuries=flags["needs_injuries"],
        preload_data=flags["preload_data"]
    )
    model.feature_names = feature_names

    # Pre-load venue_guids for all game_ids in this chunk (for travel features)
    venue_guid_cache = {}
    if 'game_id' in chunk_df.columns:
        game_ids = [str(gid) for gid in chunk_df['game_id'].dropna().unique().tolist()]
        if game_ids:
            from bball_app.core.mongo import Mongo
            db = Mongo().db
            games_with_venue = db.stats_nba.find(
                {'game_id': {'$in': game_ids}},
                {'game_id': 1, 'venue_guid': 1}
            )
            for g in games_with_venue:
                if g.get('venue_guid'):
                    venue_guid_cache[str(g['game_id'])] = g['venue_guid']

    # Initialize feature columns if needed
    for feature_name in feature_names:
        if feature_name not in chunk_df.columns:
            chunk_df[feature_name] = 0.0

    # Calculate features for each row in the chunk
    chunk_size = len(chunk_df)
    last_reported = 0
    report_interval = max(1, chunk_size // 10) if chunk_size > 10 else 1

    for row_idx, (idx, row) in enumerate(chunk_df.iterrows()):
        year = int(row['Year'])
        month = int(row['Month'])
        day = int(row['Day'])
        home_team = row['Home']
        away_team = row['Away']
        game_id = row.get('game_id') if 'game_id' in row.index else None
        venue_guid = venue_guid_cache.get(str(game_id)) if game_id else None

        # Determine season from year/month
        if month >= 10:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"

        # Build features dict (pass venue_guid for travel features)
        features_dict = model._build_features_dict(
            home_team, away_team, season, year, month, day,
            target_venue_guid=venue_guid
        )

        # Update feature values
        for feature_name in feature_names:
            if features_dict and feature_name in features_dict:
                value = features_dict[feature_name]
                # Handle NaN/Inf
                if pd.isna(value) or (isinstance(value, float) and (value != value or abs(value) == float('inf'))):
                    value = 0.0
                chunk_df.at[idx, feature_name] = value

        # Report progress periodically during chunk processing
        rows_processed_in_chunk = row_idx + 1
        if progress_callback and (rows_processed_in_chunk - last_reported >= report_interval or rows_processed_in_chunk == chunk_size):
            progress_callback(rows_processed_in_chunk - last_reported, total_rows)
            last_reported = rows_processed_in_chunk

    return chunk_df


def calculate_feature_columns_chunked(
    df: pd.DataFrame,
    feature_names: List[str],
    db,
    job_id: str = None,
    chunk_size: int = 500,
    progress_callback: callable = None
) -> pd.DataFrame:
    """
    Calculate multiple feature columns in chunks with parallel processing.
    Uses ThreadPoolExecutor with a shared context for efficient parallel processing.

    OPTIMIZED:
    1. Pre-loads all data ONCE in main thread (games, player stats, PER cache)
    2. Pre-computes expensive calculations (season severity) to avoid O(n^2) complexity
    3. Threads share the pre-loaded context (no duplicate DB connections)

    Args:
        df: DataFrame with master training data
        feature_names: List of feature names to calculate
        db: MongoDB database connection
        job_id: Optional job ID for progress updates
        chunk_size: Number of rows to process per chunk
        progress_callback: Optional callback function(current, total, progress_pct)

    Returns:
        DataFrame with calculated feature columns added
    """
    total_rows = len(df)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size

    print(f"Calculating {len(feature_names)} features in {total_chunks} chunks of ~{chunk_size} rows...")

    # =========================================================================
    # OPTIMIZATION: Pre-load all data ONCE before processing
    # - Games, player stats, PER cache loaded into memory
    # - Season severity uses lazy caching (computed on first access, cached for reuse)
    # - All threads share the same caches (no duplicate computation)
    # =========================================================================
    print("\n[OPTIMIZATION] Creating shared feature context (loads data ONCE)...")
    shared_context = SharedFeatureContext(feature_names, preload_data=True)

    # Pre-load venue GUIDs for all games in the DataFrame
    if 'game_id' in df.columns:
        game_ids = [str(gid) for gid in df['game_id'].dropna().unique().tolist()]
        if game_ids:
            print(f"Pre-loading venue GUIDs for {len(game_ids)} games...")
            shared_context.preload_venue_guids(game_ids)

    print(f"\nStarting parallel chunk processing with ThreadPoolExecutor...")

    # Update progress: starting chunked processing
    if job_id:
        update_job_progress(job_id, 1, f"[STEP 8/8] Starting chunked processing: {len(feature_names)} features, {total_rows} rows, {total_chunks} chunks", db)

    # Initialize feature columns
    for feature_name in feature_names:
        if feature_name not in df.columns:
            df[feature_name] = 0.0

    # Split DataFrame into chunks
    chunk_data = []
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk_df = df.iloc[start_idx:end_idx].copy()
        chunk_data.append((chunk_idx, chunk_df, start_idx))

    # Thread-safe progress tracker
    progress_lock = threading.Lock()
    progress_state = {'processed': 0, 'last_logged_threshold': 0.0}

    def log_progress_if_needed(processed, total):
        with progress_lock:
            progress_state['processed'] += processed
            current = min(progress_state['processed'], total)
            current_progress_pct = (current / total) * 100 if total > 0 else 0
            if current_progress_pct >= progress_state['last_logged_threshold'] + 2.5 or current == total:
                progress_state['last_logged_threshold'] = current_progress_pct
                if job_id and db is not None:
                    current_feature = feature_names[0] if len(feature_names) == 1 else f"{len(feature_names)} features"
                    message = f"Processing: {current_feature} ({current}/{total} rows, {current_progress_pct:.1f}%)"
                    update_job_progress(job_id, min(int(current_progress_pct), 85), message, db)

    # Process chunks in parallel using ThreadPoolExecutor
    # Threads share the pre-loaded context (memory) and single MongoDB connection pool
    chunk_results = []

    with ThreadPoolExecutor(max_workers=None) as executor:  # None = use default (optimal for I/O-bound)
        future_to_chunk = {
            executor.submit(
                _process_feature_chunk,
                chunk_df, chunk_idx, feature_names, total_rows, start_idx,
                log_progress_if_needed, shared_context
            ): (chunk_idx, start_idx)
            for chunk_idx, chunk_df, start_idx in chunk_data
        }

        for future in as_completed(future_to_chunk):
            chunk_idx, start_idx = future_to_chunk[future]
            try:
                result_chunk_df = future.result()
                chunk_results.append((chunk_idx, result_chunk_df))
                rows_in_chunk = len(result_chunk_df)
                log_progress_if_needed(rows_in_chunk, total_rows)
                print(f"  Completed chunk {chunk_idx + 1}/{total_chunks} ({rows_in_chunk} rows)")
            except Exception as e:
                print(f"  [Chunk {chunk_idx}] Failed: {e}")
                import traceback
                traceback.print_exc()
                raise
    
    # Sort results by chunk index to maintain order
    chunk_results.sort(key=lambda x: x[0])

    # Optimized chunk merge: use pd.concat() instead of iterative loc assignments
    # This is 3-5x faster for large feature sets
    print(f"\nCombining {len(chunk_results)} chunks into main DataFrame...")
    if job_id:
        update_job_progress(job_id, 85, "Combining processed chunks (optimized concat)...", db)

    # Extract just the feature columns from each chunk and concat all at once
    feature_chunks = [result_chunk_df[feature_names] for _, result_chunk_df in chunk_results]
    combined_features = pd.concat(feature_chunks, axis=0)

    # Single assignment to update all features at once
    df.loc[combined_features.index, feature_names] = combined_features

    if job_id:
        update_job_progress(job_id, 88, f"Chunks combined. {len(feature_names)} features updated.", db)

    # Final progress update (90% - feature calculation complete, but CSV writing still pending)
    if job_id:
        current_feature = feature_names[0] if len(feature_names) == 1 else f"{len(feature_names)} features"
        message = f"Feature calculation complete: {current_feature} ({total_rows}/{total_rows} rows). Preparing to write CSV..."
        update_job_progress(job_id, 90, message, db)
    
    if progress_callback:
        progress_callback(total_rows, total_rows, 100.0)
    
    print(f"  Completed: {total_rows}/{total_rows} rows (100.0%)")

    # Print diagnostic stats about team name normalizations
    shared_context.print_normalization_stats()

    return df


def calculate_feature_column(
    df: pd.DataFrame,
    feature_name: str,
    db
) -> pd.Series:
    """
    Calculate a feature column on-demand using BballModel.
    
    Args:
        df: DataFrame with master training data
        feature_name: Feature name to calculate (e.g., 'points|season|avg|diff')
        db: MongoDB database connection
        
    Returns:
        Series with feature values
    """
    print(f"Calculating feature: {feature_name}")
    
    # Create BballModel instance
    flags = _infer_nba_model_flags([feature_name])
    model = BballModel(
        classifier_features=[feature_name],
        include_elo=flags["needs_elo"],
        include_per_features=flags["needs_per"],
        include_injuries=flags["needs_injuries"],
        preload_data=flags["preload_data"]
    )
    model.feature_names = [feature_name]

    # Pre-load venue_guids for all game_ids (for travel features)
    venue_guid_cache = {}
    if 'game_id' in df.columns:
        # Convert to strings - MongoDB stores game_id as string, CSV has int
        game_ids = [str(gid) for gid in df['game_id'].dropna().unique().tolist()]
        if game_ids:
            from bball_app.core.mongo import Mongo
            mongo_db = Mongo().db
            games_with_venue = mongo_db.stats_nba.find(
                {'game_id': {'$in': game_ids}},
                {'game_id': 1, 'venue_guid': 1}
            )
            for g in games_with_venue:
                if g.get('venue_guid'):
                    venue_guid_cache[str(g['game_id'])] = g['venue_guid']

    # Calculate feature for each row
    feature_values = []
    total_rows = len(df)

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0 or (idx + 1) == total_rows:
            print(f"  Progress: {idx + 1}/{total_rows} rows ({100 * (idx + 1) / total_rows:.1f}%)")

        year = int(row['Year'])
        month = int(row['Month'])
        day = int(row['Day'])
        home_team = row['Home']
        away_team = row['Away']
        game_id = row.get('game_id') if 'game_id' in row.index else None
        venue_guid = venue_guid_cache.get(str(game_id)) if game_id else None

        # Determine season from year/month
        if month >= 10:
            season = f"{year}-{year+1}"
        else:
            season = f"{year-1}-{year}"

        # Build features dict (pass venue_guid for travel features)
        features_dict = model._build_features_dict(
            home_team, away_team, season, year, month, day,
            target_venue_guid=venue_guid
        )
        
        if features_dict and feature_name in features_dict:
            value = features_dict[feature_name]
            # Handle NaN/Inf
            if pd.isna(value) or (isinstance(value, float) and (value != value or abs(value) == float('inf'))):
                value = 0.0
            feature_values.append(value)
        else:
            feature_values.append(0.0)
    
    return pd.Series(feature_values, index=df.index)


def extract_predictions_from_selected_model(
    df: pd.DataFrame,
    db
) -> pd.DataFrame:
    """
    Extract point predictions from the selected point model config.
    
    Args:
        df: DataFrame with master training data
        db: MongoDB database connection
        
    Returns:
        DataFrame with added columns: pred_home_points, pred_away_points, pred_margin, pred_point_total
    """
    from bball_app.core.models.points_regression import PointsRegressionTrainer
    
    print("Loading selected point model config from MongoDB...")
    
    # Query for selected config
    selected_config = db.model_config_points_nba.find_one({'selected': True})
    if not selected_config:
        raise ValueError(
            "No selected point model config found in model_config_points_nba collection. "
            "Please select a point model config in the UI or via MongoDB."
        )
    
    model_path = selected_config.get('model_path')
    if not model_path:
        raise ValueError(
            "Selected point model config does not have 'model_path' field. "
            "The config may be incomplete or the model was not saved properly."
        )

    # Convert relative paths to absolute (for backward compatibility)
    if not os.path.isabs(model_path):
        # Relative paths are relative to nba_app directory
        nba_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(nba_app_dir, model_path)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Selected point model file not found: {model_path}. "
            "The model may have been deleted or moved."
        )

    # Extract model name from path (remove .pkl extension)
    model_name = os.path.basename(model_path).replace('.pkl', '')

    # Derive output_dir from model_path (go up from artifacts directory)
    # model_path: .../models/point_regression/artifacts/model.pkl
    # output_dir: .../models/point_regression
    artifacts_dir = os.path.dirname(model_path)
    output_dir = os.path.dirname(artifacts_dir)

    print(f"  Selected config found: {selected_config.get('name', 'Unnamed')}")
    print(f"  Model path: {model_path}")
    print(f"  Model name: {model_name}")
    print(f"  Output dir: {output_dir}")

    # Load model using PointsRegressionTrainer
    print(f"\nLoading point model: {model_name}")
    trainer = PointsRegressionTrainer(db=db, output_dir=output_dir)
    try:
        trainer.load_model(model_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load point model '{model_name}': {e}. "
            "Make sure the model files exist in the artifacts directory."
        ) from e

    # Determine target type from loaded model structure
    target_type = getattr(trainer, 'target_type', 'home_away')
    if not hasattr(trainer, 'target_type') or trainer.target_type is None:
        if isinstance(trainer.model, dict) and 'home' in trainer.model and 'away' in trainer.model:
            target_type = 'home_away'
        else:
            target_type = 'margin'
    
    # CRITICAL: Use feature_names from loaded model to ensure correct feature order
    # The model was trained with features in a specific order (trainer.feature_names)
    # We must use this exact order for predictions to work correctly
    if not hasattr(trainer, 'feature_names') or not trainer.feature_names:
        raise ValueError(
            "Loaded model does not have feature_names. Model may be corrupted or incompatible."
        )
    
    model_feature_names = trainer.feature_names
    print(f"  Model expects {len(model_feature_names)} features")
    
    # Log config features if available (for informational purposes)
    config_features = selected_config.get('features', [])
    if config_features:
        print(f"  Config lists {len(config_features)} features (model uses its saved feature order)")
    
    print(f"  Model target type: {target_type}")

    def _build_feature_matrix(feature_names: List[str]) -> np.ndarray:
        """Build dense matrix (rows=len(df), cols=len(feature_names)) in the given order."""
        matrix = np.zeros((len(df), len(feature_names)))
        for idx, feature_name in enumerate(feature_names):
            if feature_name in df.columns:
                matrix[:, idx] = df[feature_name].fillna(0.0).values

        if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):
            nan_count = np.sum(np.isnan(matrix) | np.isinf(matrix))
            print(f"  WARNING: Found {nan_count} NaN or Inf values in feature matrix. Replacing with 0.0")
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return matrix

    # Prefer the same perspective-split workflow used in training:
    # home model uses features with perspective in ['home','diff','none']
    # away model uses features with perspective in ['away','diff','none']
    use_perspective_split = bool(
        target_type == 'home_away'
        and isinstance(trainer.model, dict)
        and getattr(trainer, 'home_feature_names', None)
        and getattr(trainer, 'away_feature_names', None)
        and getattr(trainer, 'home_scaler', None) is not None
        and getattr(trainer, 'away_scaler', None) is not None
    )

    # Validate that required features exist in master CSV (missing -> filled with 0.0)
    master_features = set(df.columns)
    if use_perspective_split:
        missing_home = [f for f in trainer.home_feature_names if f not in master_features]
        missing_away = [f for f in trainer.away_feature_names if f not in master_features]
        if missing_home:
            print(f"  WARNING: {len(missing_home)} home features not found in master CSV (will use 0.0). First 20: {missing_home[:20]}")
        if missing_away:
            print(f"  WARNING: {len(missing_away)} away features not found in master CSV (will use 0.0). First 20: {missing_away[:20]}")
    else:
        missing_features = [f for f in model_feature_names if f not in master_features]
        available_features = [f for f in model_feature_names if f in master_features]

        if missing_features:
            print(f"  WARNING: {len(missing_features)} features not found in master CSV:")
            print(f"  Missing features (first 20): {missing_features[:20]}")
            print(f"  Available features: {len(available_features)}/{len(model_feature_names)}")
            print(f"  Missing features will be set to 0.0 for prediction")

        if len(available_features) == 0:
            raise ValueError(
                f"None of the required features ({len(model_feature_names)} total) are found in master CSV. "
                "Master CSV must contain the features used to train the point model."
            )

    # Prepare features for vectorized prediction
    print(f"\nGenerating predictions for {len(df)} games using vectorized prediction...")

    # Precompute scaled matrices (either per-side or global)
    feature_matrix_scaled = None
    home_matrix_scaled = None
    away_matrix_scaled = None

    if use_perspective_split:
        home_matrix = _build_feature_matrix(list(trainer.home_feature_names))
        away_matrix = _build_feature_matrix(list(trainer.away_feature_names))

        try:
            home_matrix_scaled = trainer.home_scaler.transform(home_matrix)
            away_matrix_scaled = trainer.away_scaler.transform(away_matrix)
        except Exception as e:
            raise RuntimeError(
                f"Failed to scale perspective-split features: {e}. "
                "Make sure home/away scalers were saved correctly with the model."
            ) from e
    else:
        feature_matrix = _build_feature_matrix(model_feature_names)
        try:
            feature_matrix_scaled = trainer.scaler.transform(feature_matrix)
        except Exception as e:
            raise RuntimeError(
                f"Failed to scale features: {e}. "
                "Make sure the scaler was saved correctly with the model."
            ) from e
    
    # Initialize prediction arrays
    pred_home_points = np.zeros(len(df))
    pred_away_points = np.zeros(len(df))
    pred_margin = np.zeros(len(df))
    pred_point_total = np.zeros(len(df))
    
    try:
        if target_type == 'margin':
            # Margin-only model: single model that predicts margin directly
            if feature_matrix_scaled is None:
                raise ValueError("Margin model requires a single scaled feature matrix, but none was built.")
            pred_margin_all = trainer.model.predict(feature_matrix_scaled)
            pred_margin_all = np.clip(pred_margin_all, -60, 60)  # Clamp to reasonable range
            
            pred_margin[:] = pred_margin_all
            # For margin-only, home/away are None (not predicted)
            pred_home_points[:] = np.nan
            pred_away_points[:] = np.nan
            pred_point_total[:] = np.nan
        else:
            # Home/away models: dict with 'home' and 'away' keys
            if not isinstance(trainer.model, dict) or 'home' not in trainer.model or 'away' not in trainer.model:
                raise ValueError(
                    f"Invalid model structure for target='home_away'. "
                    "Expected dict with 'home' and 'away' keys."
                )

            if use_perspective_split:
                pred_home_all = trainer.model['home'].predict(home_matrix_scaled)
                pred_away_all = trainer.model['away'].predict(away_matrix_scaled)
            else:
                pred_home_all = trainer.model['home'].predict(feature_matrix_scaled)
                pred_away_all = trainer.model['away'].predict(feature_matrix_scaled)
            
            # Clamp to reasonable range
            pred_home_all = np.clip(pred_home_all, 0, 200)
            pred_away_all = np.clip(pred_away_all, 0, 200)
            
            pred_home_points[:] = pred_home_all
            pred_away_points[:] = pred_away_all
            pred_margin[:] = pred_home_all - pred_away_all
            pred_point_total[:] = pred_home_all + pred_away_all
    except Exception as e:
        raise RuntimeError(
            f"Failed to generate predictions: {e}. "
            "Make sure the model structure matches the expected format."
        ) from e
    
    # Add prediction columns to dataframe
    df['pred_home_points'] = pred_home_points
    df['pred_away_points'] = pred_away_points
    df['pred_margin'] = pred_margin
    df['pred_point_total'] = pred_point_total
    
    print(f"  Generated predictions for {len(df)} games")
    if target_type == 'margin':
        print(f"  Margin range: {pred_margin.min():.2f} to {pred_margin.max():.2f}")
    else:
        print(f"  Home points range: {pred_home_points.min():.2f} to {pred_home_points.max():.2f}")
        print(f"  Away points range: {pred_away_points.min():.2f} to {pred_away_points.max():.2f}")
        print(f"  Margin range: {pred_margin.min():.2f} to {pred_margin.max():.2f}")
        print(f"  Total points range: {pred_point_total.min():.2f} to {pred_point_total.max():.2f}")
    
    return df


def populate_columns(
    master_csv_path: str,
    columns: List[str] = None,
    feature_substrings: List[str] = None,
    match_mode: str = 'OR',
    overwrite: bool = False,
    backup: bool = True,
    job_id: str = None,
    chunk_size: int = 500,
    progress_callback: callable = None
) -> str:
    """
    Populate additional columns in master training CSV.
    
    Args:
        master_csv_path: Path to master training CSV
        columns: List of column names to add (optional if feature_substrings provided)
        feature_substrings: List of substrings to match features (optional if columns provided)
        match_mode: 'OR' to match features containing ANY substring, 'AND' to match features containing ALL substrings
        overwrite: If True, overwrite existing columns
        backup: If True, create backup before modifying
        job_id: Optional job ID for progress updates
        chunk_size: Batch size for processing (default: 500 rows)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to updated CSV
    """
    # Connect to MongoDB
    if job_id:
        try:
            mongo = Mongo()
            db = mongo.db
            update_job_progress(job_id, 0, "[STEP 4/8] Connected to MongoDB in populate_columns. Checking CSV...", db=db)
        except Exception as e:
            # If we can't connect, try anyway
            mongo = Mongo()
            db = mongo.db
    else:
        mongo = Mongo()
        db = mongo.db
    
    if job_id:
        update_job_progress(job_id, 0, f"[STEP 5/8] Checking if CSV exists: {master_csv_path}", db=db)
    
    if not os.path.exists(master_csv_path):
        error_msg = f"Master training CSV not found: {master_csv_path}"
        if job_id:
            fail_job(job_id, error_msg, db=db)
        raise FileNotFoundError(error_msg)
    
    if job_id:
        update_job_progress(job_id, 0, "[STEP 6/8] CSV exists. Processing feature_substrings or columns...", db=db)
    
    # If feature_substrings provided, match features by substring
    if feature_substrings:
        print(f"Matching features by substrings: {feature_substrings}")
        # Read CSV to get all features
        df_temp = pd.read_csv(master_csv_path)
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 
                        'home_points', 'away_points', 'pred_home_points', 'pred_away_points', 
                        'pred_margin', 'pred_point_total', 'pred_total']
        all_features = [c for c in df_temp.columns if c not in metadata_cols]
        
        # Match features with match mode
        matching_features = find_features_by_substrings(all_features, feature_substrings, match_mode)
        
        if not matching_features:
            error_msg = f"No features found matching substrings: {feature_substrings}"
            if job_id:
                fail_job(job_id, error_msg, db=db)
            raise ValueError(error_msg)
        
        print(f"  Found {len(matching_features)} matching features")
        columns = matching_features
        
        # Resolve dependencies
        print("Resolving feature dependencies...")
        from bball_app.core.features.dependencies import resolve_dependencies
        all_features_set, dependency_map = resolve_dependencies(matching_features, include_transitive=True)
        categorized = categorize_features(matching_features, all_features_set)
        
        print(f"  Requested: {len(categorized['requested'])} features")
        print(f"  Dependencies: {len(categorized['dependencies'])} features")
        print(f"  Total to regenerate: {len(categorized['all'])} features")
        
        # Use all features including dependencies
        columns = categorized['all']
    
    if not columns:
        error_msg = "No columns specified for regeneration"
        if job_id:
            fail_job(job_id, error_msg, db=db)
        raise ValueError(error_msg)

    # Validate features against FeatureRegistry (SSoT)
    # All features must be defined in the registry before they can be added
    print(f"\nValidating {len(columns)} columns against FeatureRegistry...")
    valid_features, invalid_features = validate_features_against_registry(columns)

    if invalid_features:
        print(f"\n  ERROR: {len(invalid_features)} features failed validation:")
        for feature, error in invalid_features[:10]:  # Show first 10
            print(f"    - {feature}: {error}")
        if len(invalid_features) > 10:
            print(f"    ... and {len(invalid_features) - 10} more")

        error_msg = (
            f"{len(invalid_features)} features are not defined in FeatureRegistry. "
            f"Features must be added to the registry before they can be generated. "
            f"First invalid feature: {invalid_features[0][0]} ({invalid_features[0][1]})"
        )
        if job_id:
            fail_job(job_id, error_msg, db=db)
        raise ValueError(error_msg)

    print(f"  All {len(valid_features)} features validated successfully")

    # Update job: starting
    if job_id:
        update_job_progress(job_id, 0, f"[STEP 7/8] Starting column processing. Columns to process: {len(columns)}", db)
    
    # Create backup if requested
    if backup:
        backup_path = f"{master_csv_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating backup: {backup_path}")
        shutil.copy2(master_csv_path, backup_path)
    
    # Read existing CSV
    print(f"Reading master training CSV: {master_csv_path}")
    df = pd.read_csv(master_csv_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Separate columns into metadata, prediction, and feature columns
    metadata_cols = []
    prediction_cols = []
    feature_cols = []

    for col in columns:
        if col in ['game_id', 'home_points', 'away_points']:
            metadata_cols.append(col)
        elif is_pred_feature(col):
            # Handles both legacy format (pred_margin) and registry format (pred_margin|none|ridge|none)
            prediction_cols.append(col)
        else:
            feature_cols.append(col)
    
    # Check which columns already exist
    existing_cols = [col for col in columns if col in df.columns]
    if existing_cols:
        if overwrite:
            print(f"Overwriting existing columns: {existing_cols}")
        else:
            print(f"Skipping existing columns (use --overwrite to update): {existing_cols}")
            columns = [col for col in columns if col not in existing_cols]
    
    if not columns:
        print("No new columns to add")
        return master_csv_path
    
    # Add metadata columns
    if metadata_cols:
        print(f"\nExtracting metadata columns from MongoDB: {metadata_cols}")
        df = extract_metadata_from_mongodb(df, metadata_cols, db)
    
    # Add prediction columns (all prediction columns are added together)
    if prediction_cols:
        # Convert requested prediction cols to internal column names
        # Both legacy (pred_margin) and registry format (pred_margin|none|ridge|none)
        # map to the same internal columns (pred_margin, pred_home_points, etc.)
        requested_internal_cols = get_pred_internal_columns(prediction_cols)

        # Check existing internal columns
        existing_prediction_cols = []
        for col in ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total']:
            if col in df.columns:
                existing_prediction_cols.append(col)

        if existing_prediction_cols and not overwrite:
            print(f"Skipping existing prediction columns (use --overwrite to update): {existing_prediction_cols}")
            # Check if any requested columns are missing
            missing_prediction_cols = [c for c in requested_internal_cols if c not in df.columns]

            if missing_prediction_cols:
                print(f"  But these prediction columns are missing: {missing_prediction_cols}")
                print(f"  Generating all prediction columns...")
                df = extract_predictions_from_selected_model(df, db)
        else:
            print(f"\nExtracting prediction columns from selected point model: {prediction_cols}")
            if overwrite and existing_prediction_cols:
                print(f"  Overwriting existing prediction columns: {existing_prediction_cols}")
            df = extract_predictions_from_selected_model(df, db)
        
        # Handle pred_total alias mapping (function generates pred_point_total)
        # If user requested pred_total (legacy or registry format), create alias if it doesn't exist
        pred_total_requested = any(
            col == 'pred_total' or (is_pred_feature(col) and parse_pred_feature(col).get('stat_name') == 'pred_total')
            for col in prediction_cols
        )
        if pred_total_requested:
            if 'pred_point_total' in df.columns:
                if 'pred_total' not in df.columns:
                    df['pred_total'] = df['pred_point_total']
                elif overwrite:
                    df['pred_total'] = df['pred_point_total']
    
    # Calculate feature columns
    if feature_cols:
        print(f"\nCalculating feature columns: {len(feature_cols)} features")
        
        # Filter to only features that need calculation
        features_to_calculate = [f for f in feature_cols if f not in df.columns or overwrite]
        
        if features_to_calculate:
            if job_id or chunk_size < len(df):
                # Use chunked processing
                df = calculate_feature_columns_chunked(
                    df, features_to_calculate, db, 
                    job_id=job_id, 
                    chunk_size=chunk_size,
                    progress_callback=progress_callback
                )
            else:
                # Use original per-feature processing (backward compatibility)
                for feature_name in features_to_calculate:
                    if feature_name not in df.columns or overwrite:
                        df[feature_name] = calculate_feature_column(df, feature_name, db)
                        # Update progress after each feature
                        if job_id:
                            features_done = feature_cols.index(feature_name) + 1
                            progress_pct = (features_done / len(feature_cols)) * 100
                            update_job_progress(job_id, int(progress_pct), 
                                              f"Calculated {feature_name} ({features_done}/{len(feature_cols)} features)", db)
    
    # Reorder columns to match expected format:
    # [Year, Month, Day, Home, Away, game_id, ...features..., pred_home_points, pred_away_points, pred_margin, pred_point_total, HomeWon, home_points, away_points]
    meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away']
    if 'game_id' in df.columns:
        meta_cols.append('game_id')
    
    # Prediction columns (if they exist)
    pred_cols = []
    for pred_col in ['pred_home_points', 'pred_away_points', 'pred_margin', 'pred_point_total', 'pred_total']:
        if pred_col in df.columns and pred_col not in pred_cols:
            pred_cols.append(pred_col)
    
    target_cols = []
    if 'HomeWon' in df.columns:
        target_cols.append('HomeWon')
    if 'home_points' in df.columns:
        target_cols.append('home_points')
    if 'away_points' in df.columns:
        target_cols.append('away_points')
    
    # Feature columns are everything else (exclude metadata, predictions, and targets)
    excluded_cols = meta_cols + pred_cols + target_cols
    feature_cols_in_df = [c for c in df.columns if c not in excluded_cols]
    
    # Reorder
    new_column_order = meta_cols + sorted(feature_cols_in_df) + pred_cols + target_cols
    df = df[new_column_order]
    
    # Write updated CSV
    print(f"\nWriting updated CSV: {master_csv_path}")
    if job_id:
        update_job_progress(job_id, 95, "Writing updated CSV to disk...", db)
    
    # Write to temporary file first, then swap (atomic operation)
    temp_file_path = f"{master_csv_path}.tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    df.to_csv(temp_file_path, index=False)
    
    # Swap files
    shutil.move(temp_file_path, master_csv_path)
    
    print(f"  Updated: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Regenerated columns: {len(columns)}")

    # Update job: completed
    # Refresh MongoDB connection in case the original timed out during long-running job
    if job_id:
        try:
            fresh_mongo = Mongo()
            fresh_db = fresh_mongo.db
            complete_job(job_id, f"Successfully regenerated {len(columns)} features", fresh_db)
            print(f"  Job {job_id} marked as completed")
        except Exception as e:
            print(f"WARNING: Failed to mark job {job_id} as completed: {e}")
            # Fall back to original db connection
            complete_job(job_id, f"Successfully regenerated {len(columns)} features", db)
    
    return master_csv_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Populate additional columns in master training CSV',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli/populate_master_training_cols.py --columns "game_id,home_points,away_points"
  python cli/populate_master_training_cols.py --columns "points|season|avg|diff" --overwrite
  python cli/populate_master_training_cols.py --columns "pred_home_points,pred_away_points,pred_margin,pred_total"
  python cli/populate_master_training_cols.py --columns "game_id,home_points,away_points" --master-csv /path/to/csv
  python cli/populate_master_training_cols.py --feature-substrings "inj_min_lost,player_per_1" --overwrite --job-id JOB_ID
        """
    )
    
    parser.add_argument(
        '--columns',
        type=str,
        default=None,
        help='Comma-separated list of column names to add (e.g., "game_id,home_points,away_points,points|season|avg|diff,pred_home_points,pred_away_points,pred_margin,pred_total")'
    )
    
    parser.add_argument(
        '--feature-substrings',
        type=str,
        default=None,
        help='Comma-separated list of substrings to match features (e.g., "inj_min_lost,player_per_1"). Matches all features containing any substring.'
    )
    
    parser.add_argument(
        '--match-mode',
        type=str,
        choices=['OR', 'AND'],
        default='OR',
        help='Match mode: OR matches features containing ANY substring, AND matches features containing ALL substrings (default: OR)'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing columns if they already exist'
    )
    
    parser.add_argument(
        '--backup',
        action='store_true',
        default=True,
        help='Create backup before modifying (default: True)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup'
    )
    
    parser.add_argument(
        '--master-csv',
        type=str,
        default=MASTER_TRAINING_PATH,
        help=f'Path to master training CSV (default: {MASTER_TRAINING_PATH})'
    )
    
    parser.add_argument(
        '--job-id',
        type=str,
        default=None,
        help='Job ID for progress updates (optional)'
    )

    parser.add_argument(
        '--league',
        type=str,
        default='nba',
        help='League ID for database collections (default: nba)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=500,
        help='Batch size for processing rows (default: 500)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate without regenerating (shows what would be regenerated)'
    )
    
    args = parser.parse_args()
    
    # Connect to MongoDB if job_id provided (do this early for logging)
    db = None
    if args.job_id:
        try:
            from bball_app.core.league_config import load_league_config

            # LeagueDbProxy for CLI scripts - maps db.jobs_nba to the league's jobs collection
            class LeagueDbProxy:
                _NBA_ATTR_TO_KEY = {
                    "stats_nba": "games",
                    "stats_nba_players": "player_stats",
                    "players_nba": "players",
                    "teams_nba": "teams",
                    "nba_venues": "venues",
                    "nba_rosters": "rosters",
                    "model_config_nba": "model_config_classifier",
                    "model_config_points_nba": "model_config_points",
                    "master_training_data_nba": "master_training_metadata",
                    "cached_league_stats": "cached_league_stats",
                    "nba_cached_elo_ratings": "elo_cache",
                    "experiment_runs": "experiment_runs",
                    "jobs_nba": "jobs",
                }

                def __init__(self, db, league):
                    self._db = db
                    self._league = league

                def __getitem__(self, name: str):
                    return self._db[name]

                def __getattr__(self, name: str):
                    key = self._NBA_ATTR_TO_KEY.get(name)
                    if key is not None:
                        coll_name = self._league.collections.get(key)
                        if coll_name:
                            return self._db[coll_name]
                    return getattr(self._db, name)

            mongo = Mongo()
            league = load_league_config(args.league)
            db = LeagueDbProxy(mongo.db, league)
            print(f"Connected to MongoDB (league: {args.league}, jobs collection: {league.collections.get('jobs')})")
            update_job_progress(args.job_id, 0, "[STEP 1/8] Connected to MongoDB. Validating arguments...", db=db)
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    # Validate arguments
    if not args.columns and not args.feature_substrings:
        error_msg = "Error: Either --columns or --feature-substrings must be specified"
        print(error_msg)
        if args.job_id and db is not None:
            fail_job(args.job_id, error_msg, db=db)
        sys.exit(1)
    
    if args.columns and args.feature_substrings:
        error_msg = "Error: Cannot specify both --columns and --feature-substrings. Use one or the other."
        print(error_msg)
        if args.job_id and db is not None:
            fail_job(args.job_id, error_msg, db=db)
        sys.exit(1)
    
    # Parse columns or feature_substrings
    columns = None
    feature_substrings = None
    
    if args.columns:
        if args.job_id and db is not None:
            update_job_progress(args.job_id, 0, "[STEP 2/8] Parsing columns...", db=db)
        columns = [c.strip() for c in args.columns.split(',') if c.strip()]
        if not columns:
            error_msg = "Error: No columns specified"
            print(error_msg)
            if args.job_id and db is not None:
                fail_job(args.job_id, error_msg, db=db)
            sys.exit(1)
    
    if args.feature_substrings:
        if args.job_id and db is not None:
            update_job_progress(args.job_id, 0, "[STEP 2/8] Parsing feature substrings...", db=db)
        feature_substrings = [s.strip() for s in args.feature_substrings.split(',') if s.strip()]
        if not feature_substrings:
            error_msg = "Error: No feature substrings specified"
            print(error_msg)
            if args.job_id and db is not None:
                fail_job(args.job_id, error_msg, db=db)
            sys.exit(1)
    
    # Handle backup flag
    backup = args.backup and not args.no_backup

    # Get league-aware master CSV path if using default
    from bball_app.core.services.training_data import get_master_training_path
    if args.master_csv == MASTER_TRAINING_PATH:
        # Using default path - get league-specific path instead
        args.master_csv = get_master_training_path(args.league)
        print(f"Using league-aware master CSV path: {args.master_csv}")

    try:
        if args.job_id and db is not None:
            update_job_progress(args.job_id, 0, "[STEP 3/8] Calling populate_columns function...", db=db)
        result = populate_columns(
            master_csv_path=args.master_csv,
            columns=columns,
            feature_substrings=feature_substrings,
            match_mode=args.match_mode,
            overwrite=args.overwrite,
            backup=backup if not args.dry_run else False,
            job_id=args.job_id,
            chunk_size=args.chunk_size
        )
        
        if args.dry_run:
            print("\n✓ Dry run completed (no changes made)")
        else:
            print("\n✓ Successfully populated columns")
    except Exception as e:
        error_msg = str(e)
        print(f"\n✗ Error: {error_msg}")

        # Mark job as failed if job_id provided
        # Use fresh connection in case original timed out during long-running job
        if args.job_id:
            try:
                fresh_mongo = Mongo()
                fresh_db = fresh_mongo.db
                fail_job(args.job_id, error_msg, db=fresh_db)
                print(f"  Job {args.job_id} marked as failed")
            except Exception as mongo_err:
                print(f"WARNING: Failed to mark job {args.job_id} as failed: {mongo_err}")
                # Fall back to original db connection if available
                if db is not None:
                    fail_job(args.job_id, error_msg, db=db)

        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
