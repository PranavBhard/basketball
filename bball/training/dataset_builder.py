"""
Dataset Builder Tool - Wraps BballModel.create_training_data() with caching
"""

import sys
import os
import hashlib
import json
from typing import Dict, Optional
from datetime import datetime

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from bball.mongo import Mongo
from bball.models.bball_model import BballModel
from bball.features.sets import filter_features_by_model_type
from bball.services.training_data import MASTER_TRAINING_PATH, extract_features_from_master, check_master_needs_regeneration, get_all_possible_features, get_master_training_path
from bball.training.schemas import DatasetSpec


class DatasetBuilder:
    """Builds training datasets with caching"""

    def __init__(self, db=None, league=None):
        """
        Initialize DatasetBuilder.

        Args:
            db: MongoDB database instance (optional)
            league: LeagueConfig instance for league-specific paths
        """
        if db is None:
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league
        self.league_id = league.league_id if league else "nba"
        # Get league-specific master training path
        self.master_training_path = get_master_training_path(league) if league else MASTER_TRAINING_PATH

        # Cache directory for datasets — namespaced by league to prevent cross-league collisions
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        self.cache_dir = os.path.join(project_root, 'model_output', 'dataset_cache', self.league_id)
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def merge_point_predictions(self, df, point_model_id: str):
        """
        Merge point predictions into a dataframe.
        
        Args:
            df: DataFrame with master training data
            point_model_id: Model identifier for predictions to merge
            
        Returns:
            DataFrame with added columns: pred_home_points, pred_away_points, pred_margin, pred_point_total
            Note: Only pred_margin is included as a feature by default. Other prediction columns
            (pred_home_points, pred_away_points, pred_point_total) are available in the dataframe
            for reference/analysis but are not used as features in classification training.
        """
        from bball.data.cache import PointPredictionCache
        cache = PointPredictionCache(db=self.db)
        
        try:
            df = cache.merge_predictions_into_dataframe(df, point_model_id)
            print(f"  [INFO] Merged point predictions from model_id: {point_model_id}")
            return df
        except ValueError as e:
            raise ValueError(
                f"Cannot merge point predictions: {e}. "
                f"Make sure a points regression experiment has been run with this model_id first."
            ) from e
    
    def _hash_spec(self, spec: Dict) -> str:
        """
        Create a hash of the dataset spec for caching.

        Args:
            spec: Dataset specification dict

        Returns:
            Hash string
        """
        # Create a normalized version of the spec (sorted keys, no None values)
        normalized = {}
        for key, value in sorted(spec.items()):
            if value is not None:
                normalized[key] = value

        # Include league ID in hash to prevent cross-league cache collisions
        # (e.g., CBB and WCBB with same features must not share cached datasets)
        normalized['_league_id'] = self.league_id

        spec_str = json.dumps(normalized, sort_keys=True)
        return hashlib.sha256(spec_str.encode()).hexdigest()[:16]
    
    def build_dataset(self, dataset_spec: Dict) -> Dict:
        """
        Build a training dataset from specification.
        
        Args:
            dataset_spec: Dataset specification dict with:
                - feature_blocks: List of feature set names
                - individual_features: Optional list of specific features (overrides blocks)
                - begin_year: Optional start year
                - end_year: Optional end year
                - begin_date: Optional start date (YYYY-MM-DD)
                - end_date: Optional end date (YYYY-MM-DD)
                - min_games_played: Optional minimum games filter
                - exclude_preseason: Whether to exclude preseason games
                - include_per: Whether to include PER features
                - diff_mode: 'home_minus_away', 'away_minus_home', or 'absolute'
        
        Returns:
            Dict with:
                - dataset_id: Unique identifier
                - schema: Feature names
                - row_count: Number of games
                - feature_count: Number of features
                - csv_path: Path to CSV file
                - cached: Whether this was loaded from cache
                - dropped_features: (Optional) List of features that were requested but not available in master CSV
                - requested_feature_count: (Optional) Total number of features originally requested
        """
        # Extract force_rebuild flag (not part of spec hash)
        force_rebuild = dataset_spec.get('force_rebuild', False)
        if force_rebuild:
            print(f"[DatasetBuilder] Force rebuild requested - will bypass cache")
        # Remove from spec dict copy to avoid validation errors
        dataset_spec_clean = {k: v for k, v in dataset_spec.items() if k != 'force_rebuild'}

        # Validate spec
        try:
            spec = DatasetSpec(**dataset_spec_clean)
        except Exception as e:
            raise ValueError(f"Invalid dataset spec: {e}")

        # Create hash for caching
        spec_dict = spec.dict(exclude_none=True)
        dataset_id = self._hash_spec(spec_dict)

        # Check cache
        cache_file = os.path.join(self.cache_dir, f'dataset_{dataset_id}.csv')
        cache_meta_file = os.path.join(self.cache_dir, f'dataset_{dataset_id}_meta.json')

        # Delete cache if force_rebuild is requested
        if force_rebuild:
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"[DatasetBuilder] Force rebuild: deleted cached dataset {cache_file}")
            if os.path.exists(cache_meta_file):
                os.remove(cache_meta_file)

        if os.path.exists(cache_file) and os.path.exists(cache_meta_file):
            # Load from cache
            with open(cache_meta_file, 'r') as f:
                metadata = json.load(f)
            
            # Verify cache is still valid (check file exists and has rows)
            if os.path.getsize(cache_file) > 0:
                import pandas as pd
                try:
                    df = pd.read_csv(cache_file)
                    # Check if DataFrame has actual rows (not just headers)
                    if df.empty:
                        # Cache is invalid (empty CSV), rebuild
                        print(f"Cache file {cache_file} is empty, rebuilding dataset...")
                        os.remove(cache_file)
                        os.remove(cache_meta_file)
                    elif metadata.get('row_count', 0) == 0:
                        # Metadata says 0 rows, but CSV has rows - invalid cache
                        print(f"Cache metadata indicates 0 rows but CSV has data, rebuilding dataset...")
                        os.remove(cache_file)
                        os.remove(cache_meta_file)
                    else:
                        # Cache is valid
                        result = {
                            'dataset_id': dataset_id,
                            'schema': metadata['schema'],
                            'row_count': metadata['row_count'],
                            'feature_count': metadata['feature_count'],
                            'csv_path': cache_file,
                            'cached': True
                        }
                        # Include dropped features info if present in metadata
                        if 'dropped_features' in metadata:
                            result['dropped_features'] = metadata['dropped_features']
                            result['requested_feature_count'] = metadata.get('requested_feature_count', metadata['feature_count'])
                        return result
                except Exception as e:
                    # Error reading cache, rebuild
                    print(f"Error reading cache file {cache_file}: {e}, rebuilding dataset...")
                    if os.path.exists(cache_file):
                        os.remove(cache_file)
                    if os.path.exists(cache_meta_file):
                        os.remove(cache_meta_file)
        
        # Build dataset
        # Determine feature list
        if spec.individual_features:
            features = spec.individual_features
        elif spec.feature_blocks:
            # Get features from blocks using master CSV (not FEATURE_SETS)
            # Read master CSV to get available features
            import pandas as pd
            master_features = set()
            if os.path.exists(self.master_training_path):
                try:
                    master_df = pd.read_csv(self.master_training_path, nrows=0)
                    meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 'home_points', 'away_points']
                    master_features = set([c for c in master_df.columns if c not in meta_cols])
                except Exception as e:
                    import logging
                    logging.error(f"Failed to read master CSV to get features: {e}")
                    master_features = set()
            
            # Map master CSV features to blocks (same logic as support_tools)
            from collections import defaultdict
            features_by_block = defaultdict(list)
            for feature in master_features:
                feature_lower = feature.lower()
                if '|' in feature:
                    parts = feature.split('|')
                    stat_name = parts[0].lower()
                    if 'inj' in stat_name or 'inj' in feature_lower:
                        features_by_block['injuries'].append(feature)
                    elif 'player_per' in stat_name or 'team_per' in stat_name or 'starters_per' in stat_name or \
                         'per1' in stat_name or 'per2' in stat_name or 'per3' in stat_name or 'per_available' in feature_lower:
                        features_by_block['player_talent'].append(feature)
                    elif 'elo' in stat_name:
                        features_by_block['elo_strength'].append(feature)
                    elif 'rel' in feature_lower:
                        features_by_block['era_normalization'].append(feature)
                    elif 'off_rtg' in stat_name or 'assists_ratio' in stat_name:
                        features_by_block['offensive_engine'].append(feature)
                    elif 'def_rtg' in stat_name or 'blocks' in stat_name or 'reb_total' in stat_name or 'reb_' in stat_name or 'turnovers' in stat_name:
                        features_by_block['defensive_engine'].append(feature)
                    elif 'efg' in stat_name or 'ts' in stat_name or 'three' in stat_name:
                        features_by_block['shooting_efficiency'].append(feature)
                    elif 'points' in stat_name or 'wins' in stat_name:
                        features_by_block['outcome_strength'].append(feature)
                    elif 'pace' in stat_name or 'std' in feature_lower:
                        features_by_block['pace_volatility'].append(feature)
                    elif 'b2b' in stat_name or 'travel' in stat_name or 'rest' in feature_lower:
                        features_by_block['schedule_fatigue'].append(feature)
                    elif 'games_played' in stat_name:
                        if 'days' in feature_lower or 'diff' in feature_lower:
                            features_by_block['schedule_fatigue'].append(feature)
                        else:
                            features_by_block['sample_size'].append(feature)
                    else:
                        features_by_block['other'].append(feature)
                else:
                    # Non-standard feature names (fallback categorization)
                    if 'Per' in feature or ('per' in feature_lower and 'percent' not in feature_lower):
                        features_by_block['player_talent'].append(feature)
                    elif 'Inj' in feature or 'inj' in feature_lower:
                        features_by_block['injuries'].append(feature)
                    elif 'Pace' in feature or 'pace' in feature_lower:
                        features_by_block['pace_volatility'].append(feature)
                    elif 'B2B' in feature or 'Travel' in feature or 'GamesLast' in feature or 'rest' in feature_lower:
                        features_by_block['schedule_fatigue'].append(feature)
                    elif 'Rel' in feature or 'rel' in feature_lower:
                        features_by_block['era_normalization'].append(feature)
                    elif 'Elo' in feature or 'elo' in feature_lower:
                        features_by_block['elo_strength'].append(feature)
                    elif 'GamesPlayed' in feature:
                        features_by_block['sample_size'].append(feature)
                    else:
                        features_by_block['other'].append(feature)
            
            # Get features for requested blocks
            features = []
            for block_name in spec.feature_blocks:
                if block_name in features_by_block:
                    features.extend(features_by_block[block_name])
            
            # Debug: Log if no features found
            if len(features) == 0 and len(master_features) > 0:
                import logging
                logging.warning(
                    f"No features found for blocks {spec.feature_blocks}. "
                    f"Master CSV has {len(master_features)} features. "
                    f"Available blocks: {sorted([b for b, f in features_by_block.items() if len(f) > 0])}. "
                    f"Requested blocks have features: {[(b, len(features_by_block[b])) for b in spec.feature_blocks if b in features_by_block]}"
                )
        else:
            # Use all features from registry SSoT
            features = get_all_possible_features()
        
        # Filter features based on diff_mode
        # FEATURE_SETS contain both diff and home/away features, so we need to filter to the right ones
        # Exception: If individual_features is explicitly provided, don't filter (user wants specific features)
        if spec.diff_mode and not spec.individual_features:
            from bball.features.parser import parse_feature_name
            filtered_features = []
            for feature in features:
                components = parse_feature_name(feature)
                if components:
                    # Filter based on diff_mode
                    if spec.diff_mode == 'home_minus_away':
                        # Keep only diff features
                        if components.home_away_diff == 'diff':
                            filtered_features.append(feature)
                    elif spec.diff_mode == 'away_minus_home':
                        # Also use diff features (the calculation is just reversed)
                        if components.home_away_diff == 'diff':
                            filtered_features.append(feature)
                    elif spec.diff_mode == 'absolute':
                        # Keep only home and away features (per-team)
                        if components.home_away_diff in ['home', 'away']:
                            filtered_features.append(feature)
                    elif spec.diff_mode == 'mixed' or spec.diff_mode == 'all':
                        # Keep all features (diff, home, away) - no filtering
                        filtered_features.append(feature)
                else:
                    # Can't parse as pipe-delimited format
                    # Keep only known special features (pred_*, etc.)
                    if feature.startswith('pred_'):
                        filtered_features.append(feature)
                    else:
                        print(f"Warning: Skipping unrecognized feature format: {feature}")
            features = filtered_features
        
        # Validate that we have features to work with
        if len(features) == 0:
            # Get list of valid feature blocks from master CSV for better error message
            # Use the same helper function as support_tools to ensure consistency
            valid_blocks = []
            if os.path.exists(self.master_training_path):
                try:
                    import pandas as pd
                    master_df = pd.read_csv(self.master_training_path, nrows=0)
                    meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon']
                    master_features = set([c for c in master_df.columns if c not in meta_cols])
                    
                    # Use same mapping logic as support_tools._map_master_features_to_blocks
                    from collections import defaultdict
                    temp_blocks = defaultdict(list)
                    for feature in master_features:
                        feature_lower = feature.lower()
                        if '|' in feature:
                            parts = feature.split('|')
                            stat_name = parts[0].lower()
                            if 'inj' in stat_name or 'inj' in feature_lower:
                                temp_blocks['injuries'].append(feature)
                            elif 'player_per' in stat_name or 'team_per' in stat_name or 'starters_per' in stat_name or \
                                 'per1' in stat_name or 'per2' in stat_name or 'per3' in stat_name or 'per_available' in feature_lower:
                                temp_blocks['player_talent'].append(feature)
                            elif 'elo' in stat_name:
                                temp_blocks['elo_strength'].append(feature)
                            elif 'rel' in feature_lower:
                                temp_blocks['era_normalization'].append(feature)
                            elif 'off_rtg' in stat_name or 'assists_ratio' in stat_name:
                                temp_blocks['offensive_engine'].append(feature)
                            elif 'def_rtg' in stat_name or 'blocks' in stat_name or 'reb_total' in stat_name or 'reb_' in stat_name or 'turnovers' in stat_name:
                                temp_blocks['defensive_engine'].append(feature)
                            elif 'efg' in stat_name or 'ts' in stat_name or 'three' in stat_name:
                                temp_blocks['shooting_efficiency'].append(feature)
                            elif 'points' in stat_name or 'wins' in stat_name:
                                temp_blocks['outcome_strength'].append(feature)
                            elif 'pace' in stat_name or 'std' in feature_lower:
                                temp_blocks['pace_volatility'].append(feature)
                            elif 'b2b' in stat_name or 'travel' in stat_name or 'rest' in feature_lower:
                                temp_blocks['schedule_fatigue'].append(feature)
                            elif 'games_played' in stat_name:
                                if 'days' in feature_lower or 'diff' in feature_lower:
                                    temp_blocks['schedule_fatigue'].append(feature)
                                else:
                                    temp_blocks['sample_size'].append(feature)
                            else:
                                temp_blocks['other'].append(feature)
                        else:
                            # Non-standard feature names (fallback categorization)
                            if 'Per' in feature or ('per' in feature_lower and 'percent' not in feature_lower):
                                temp_blocks['player_talent'].append(feature)
                            elif 'Inj' in feature or 'inj' in feature_lower:
                                temp_blocks['injuries'].append(feature)
                            elif 'Pace' in feature or 'pace' in feature_lower:
                                temp_blocks['pace_volatility'].append(feature)
                            elif 'B2B' in feature or 'Travel' in feature or 'GamesLast' in feature or 'rest' in feature_lower:
                                temp_blocks['schedule_fatigue'].append(feature)
                            elif 'Rel' in feature or 'rel' in feature_lower:
                                temp_blocks['era_normalization'].append(feature)
                            elif 'Elo' in feature or 'elo' in feature_lower:
                                temp_blocks['elo_strength'].append(feature)
                            elif 'GamesPlayed' in feature:
                                temp_blocks['sample_size'].append(feature)
                            else:
                                temp_blocks['other'].append(feature)
                    valid_blocks = sorted([b for b, f in temp_blocks.items() if len(f) > 0])
                except Exception as e:
                    # Log the exception for debugging but still show empty list
                    import logging
                    logging.warning(f"Failed to read master CSV to get valid blocks: {e}")
                    valid_blocks = []
            
            invalid_blocks = []
            if spec.feature_blocks:
                invalid_blocks = [b for b in spec.feature_blocks if b not in valid_blocks]
            
            error_msg = (
                f"No features specified or available. "
                f"Feature blocks requested: {spec.feature_blocks}, "
                f"Individual features: {spec.individual_features}, "
                f"include_per: {spec.include_per}. "
            )
            
            if invalid_blocks:
                error_msg += (
                    f"Invalid feature blocks: {invalid_blocks}. "
                    f"Valid feature blocks are: {valid_blocks}. "
                )
            else:
                error_msg += (
                    f"Valid feature blocks are: {valid_blocks}. "
                    f"Check that feature_blocks or individual_features are provided and valid."
                )
            
            raise ValueError(error_msg)
        
        # CRITICAL: Check if master training CSV exists and has all requested features
        # We MUST use master - never create from scratch (except for new feature testing)
        if not os.path.exists(self.master_training_path):
            raise ValueError(
                f"Master training CSV not found at {self.master_training_path}. "
                f"Cannot build dataset without pre-computed features. "
                f"Please generate the master training CSV first."
            )
        
        use_master = False
        missing_in_master = []
        try:
            import pandas as pd
            # Quick check: read just the header
            master_df = pd.read_csv(self.master_training_path, nrows=0)
            # Metadata and target columns (not features)
            meta_target_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'game_id', 'HomeWon', 'home_points', 'away_points']
            master_features = [c for c in master_df.columns if c not in meta_target_cols]
            master_features_set = set(master_features)
            
            # Check which requested features exist in master
            missing_in_master = [f for f in features if f not in master_features_set]
            available_features = [f for f in features if f in master_features_set]
            
            if missing_in_master:
                # Filter out missing features and continue with available ones
                print(f"  [WARNING] {len(missing_in_master)} requested features not found in master CSV and will be dropped:")
                print(f"  [WARNING] Missing features (first 20): {missing_in_master[:20]}")
                print(f"  [WARNING] Total requested: {len(features)}, Available in master: {len(available_features)}")
                
                # Update features list to only include available features
                features = available_features
                
                # If all features were dropped, raise an error
                if len(features) == 0:
                    error_msg = (
                        f"Cannot build dataset: All {len(missing_in_master)} requested features are not in the master training CSV. "
                        f"Missing features (first 10): {missing_in_master[:10]}. "
                        f"\n\nThis dataset builder only supports extracting from pre-computed features. "
                        f"Please use `get_available_features()` or `get_features_by_block()` to see which features are actually available, "
                        f"or regenerate the master training CSV to include the missing features."
                    )
                    raise ValueError(error_msg)
            else:
                print(f"  [INFO] All {len(features)} requested features found in pre-computed feature set. Using pre-computed features (fast).")
            
            use_master = True
        except ValueError:
            # Re-raise ValueError (our error about missing features)
            raise
        except Exception as e:
            raise ValueError(
                f"Error checking master training CSV: {e}. "
                f"Cannot proceed without verifying features exist in master CSV."
            ) from e
        
        if use_master:
            # Carve from pre-computed features (much faster)
            import pandas as pd
            from bball.services.training_data import extract_features_from_master
            
            # Read pre-computed features CSV
            master_df = pd.read_csv(self.master_training_path)
            
            # Apply date/year filters
            # Default to 2012 (2012-2013 season) if not specified
            # IMPORTANT: begin_year represents SeasonStartYear, not calendar Year
            # NBA seasons span two calendar years (e.g., 2012-2013 season includes Oct-Dec 2012 and Jan-Jun 2013)
            begin_year = spec.begin_year if spec.begin_year is not None else 2012
            if begin_year:
                # Calculate SeasonStartYear: Oct-Dec belong to that calendar year's season,
                # Jan-Jun belong to the previous calendar year's season
                import numpy as np
                master_df = master_df.copy()
                master_df['SeasonStartYear'] = np.where(
                    master_df['Month'].astype(int) >= 10,
                    master_df['Year'].astype(int),
                    master_df['Year'].astype(int) - 1
                )
                master_df = master_df[master_df['SeasonStartYear'] >= int(begin_year)]
                # Drop SeasonStartYear column after filtering (it's a helper column)
                master_df = master_df.drop('SeasonStartYear', axis=1)
            if spec.exclude_seasons:
                import numpy as np
                if 'SeasonStartYear' not in master_df.columns:
                    master_df = master_df.copy()
                    master_df['SeasonStartYear'] = np.where(
                        master_df['Month'].astype(int) >= 10,
                        master_df['Year'].astype(int),
                        master_df['Year'].astype(int) - 1
                    )
                before_exclude = len(master_df)
                master_df = master_df[~master_df['SeasonStartYear'].isin(spec.exclude_seasons)]
                print(f"  [INFO] Excluded seasons {spec.exclude_seasons}: {before_exclude} -> {len(master_df)} games")
                master_df = master_df.drop('SeasonStartYear', axis=1)
            if spec.end_year:
                # For end_year, also use SeasonStartYear logic
                if 'SeasonStartYear' not in master_df.columns:
                    import numpy as np
                    master_df = master_df.copy()
                    master_df['SeasonStartYear'] = np.where(
                        master_df['Month'].astype(int) >= 10,
                        master_df['Year'].astype(int),
                        master_df['Year'].astype(int) - 1
                    )
                master_df = master_df[master_df['SeasonStartYear'] <= int(spec.end_year)]
                master_df = master_df.drop('SeasonStartYear', axis=1)
            if spec.begin_date:
                # Convert date string to comparable format
                master_df['Date'] = pd.to_datetime(master_df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
                begin_dt = pd.to_datetime(spec.begin_date)
                master_df = master_df[master_df['Date'] >= begin_dt]
                master_df = master_df.drop('Date', axis=1)
            if spec.end_date:
                if 'Date' not in master_df.columns:
                    master_df['Date'] = pd.to_datetime(master_df[['Year', 'Month', 'Day']].astype(str).agg('-'.join, axis=1))
                end_dt = pd.to_datetime(spec.end_date)
                master_df = master_df[master_df['Date'] <= end_dt]
                master_df = master_df.drop('Date', axis=1)
            
            # Apply min_games filter if specified
            # This filter ensures both teams have played at least min_games_played games
            # in the same season before the target game (prevents using early-season games with insufficient data)
            # min_games_played=0 or None means no filter
            if spec.min_games_played is not None and spec.min_games_played > 0:
                MIN_GAMES_PLAYED = int(spec.min_games_played)
                before_mgp = len(master_df)
                
                # Calculate Season column (e.g., "2018-2019" for games from Oct 2018 to Jun 2019)
                # If Month >= 10 (Oct-Dec), season is Year-Year+1
                # If Month < 10 (Jan-Jun), season is Year-1-Year
                import numpy as np
                master_df = master_df.copy()
                master_df['Season'] = np.where(master_df['Month'].astype(int) >= 10,
                                                master_df['Year'].astype(int).astype(str) + '-' + (master_df['Year'].astype(int) + 1).astype(str),
                                                (master_df['Year'].astype(int) - 1).astype(str) + '-' + master_df['Year'].astype(int).astype(str))
                
                # Build a sortable date key
                master_df['_date_key'] = (master_df['Year'].astype(int) * 10000) + (master_df['Month'].astype(int) * 100) + master_df['Day'].astype(int)
                
                # Home prior counts per season (group by Season, not Year)
                home_keys = ['Year', 'Month', 'Day', 'Home']
                home_seq = master_df[home_keys + ['Season', '_date_key']].copy()
                home_seq = home_seq.sort_values(['Season', 'Home', '_date_key'])
                home_seq['_homePrior'] = home_seq.groupby(['Season', 'Home']).cumcount()
                master_df = master_df.merge(
                    home_seq[home_keys + ['_homePrior']],
                    on=home_keys,
                    how='left'
                )
                
                # Away prior counts per season (group by Season, not Year)
                away_keys = ['Year', 'Month', 'Day', 'Away']
                away_seq = master_df[away_keys + ['Season', '_date_key']].copy()
                away_seq = away_seq.sort_values(['Season', 'Away', '_date_key'])
                away_seq['_awayPrior'] = away_seq.groupby(['Season', 'Away']).cumcount()
                master_df = master_df.merge(
                    away_seq[away_keys + ['_awayPrior']],
                    on=away_keys,
                    how='left'
                )
                
                # Apply filter: both teams must have played at least MIN_GAMES_PLAYED prior same-season games
                master_df = master_df[(master_df['_homePrior'] >= MIN_GAMES_PLAYED) & (master_df['_awayPrior'] >= MIN_GAMES_PLAYED)].copy()
                
                # Drop helper columns
                master_df.drop(columns=[c for c in ['_date_key', '_homePrior', '_awayPrior', 'Season'] if c in master_df.columns], inplace=True)
                
                after_mgp = len(master_df)
                print(f"  [INFO] Applied min_games_played filter (>= {MIN_GAMES_PLAYED}): {before_mgp} -> {after_mgp} games")
                
                if len(master_df) == 0:
                    raise ValueError(
                        f'No training data available after applying min_games_played >= {MIN_GAMES_PLAYED}. '
                        f'This filter requires both teams to have played at least {MIN_GAMES_PLAYED} games '
                        f'in the same season before the target game. Try reducing min_games_played or check your data.'
                    )
            
            # Extract requested features
            # Metadata columns: Year, Month, Day, Home, Away, game_id (if present)
            # Target columns: HomeWon, home_points, away_points (if present)
            meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away']
            if 'game_id' in master_df.columns:
                meta_cols.append('game_id')
            target_cols = []
            if 'HomeWon' in master_df.columns:
                target_cols.append('HomeWon')
            if 'home_points' in master_df.columns:
                target_cols.append('home_points')
            if 'away_points' in master_df.columns:
                target_cols.append('away_points')
            
            csv_feature_cols = [c for c in master_df.columns if c not in meta_cols + target_cols]
            ordered_features = [f for f in csv_feature_cols if f in features]
            columns_to_extract = meta_cols + ordered_features + target_cols
            extracted_df = master_df[columns_to_extract].copy()
            
            # Merge point predictions if requested
            if spec.point_model_id:
                extracted_df = self.merge_point_predictions(extracted_df, spec.point_model_id)
                # Only include pred_margin as a feature by default (other prediction columns remain in dataframe for reference)
                ordered_features.append('pred_margin')
            
            # Write to cache file
            extracted_df.to_csv(cache_file, index=False)
            clf_csv = cache_file
            count = len(extracted_df)
            
            print(f"  [INFO] Extracted {count} rows from pre-computed features with {len(ordered_features)} features")
        else:
            # This should never happen - we raise an error above if features are missing
            raise ValueError(
                "Internal error: use_master is False but we should have raised an error earlier. "
                "This indicates a logic error in the dataset builder."
            )
        
        # At this point, we've extracted from master and have clf_csv and count
        # Check if any games were processed
        if count == 0:
            query_info = f"Master CSV filter (year: {spec.begin_year}-{spec.end_year}, date: {spec.begin_date}-{spec.end_date})"
            raise ValueError(
                f"No training data generated after filtering master CSV. "
                f"{query_info}. "
                f"The date/year filters may be too restrictive, or the master CSV may not have data for the requested range. "
                f"Check the master CSV to verify it contains data for the requested date range."
            )
        
        # Read schema from CSV
        import pandas as pd
        try:
            df = pd.read_csv(clf_csv)
        except Exception as e:
            raise ValueError(
                f"Failed to read generated CSV file {clf_csv}: {e}. "
                f"Game count reported: {count}, but CSV could not be read."
            )
        
        # Check if DataFrame is empty
        if df.empty:
            raise ValueError(
                f"Generated CSV file {clf_csv} is empty (no rows). "
                f"Game count reported: {count}, but no data rows were written. "
                f"This may indicate all games were filtered out during processing (missing season, homeWon, etc.)."
            )
        
        # Metadata columns: Year, Month, Day, Home, Away, game_id (if present)
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away']
        if 'game_id' in df.columns:
            meta_cols.append('game_id')
        
        # Target columns: HomeWon, home_points, away_points (if present)
        target_cols = []
        if 'HomeWon' in df.columns:
            target_cols.append('HomeWon')
        if 'home_points' in df.columns:
            target_cols.append('home_points')
        if 'away_points' in df.columns:
            target_cols.append('away_points')
        
        # Check if at least one target column exists (HomeWon for classification)
        if 'HomeWon' not in df.columns and len(target_cols) == 0:
            available_cols = list(df.columns)
            raise ValueError(
                f"Target column 'HomeWon' not found in CSV. "
                f"Available columns: {available_cols}. "
                f"CSV file: {clf_csv}, Game count: {count}"
            )
        
        feature_cols = [c for c in df.columns if c not in meta_cols + target_cols]
        
        # Validate that we have features
        if len(feature_cols) == 0:
            available_cols = list(df.columns)
            raise ValueError(
                f"Dataset created with 0 features. "
                f"CSV has {len(df)} rows but only metadata columns: {available_cols}. "
                f"This likely means: "
                f"(1) The feature list was empty, "
                f"(2) All features were filtered out during processing, or "
                f"(3) Features were not calculated correctly. "
                f"CSV file: {clf_csv}. "
                f"Requested features: {features[:20] if len(features) > 0 else 'NONE'} "
                f"({len(features)} total). "
                f"Check BballModel.create_training_data logs to see why features are missing."
            )
        
        # Save metadata
        metadata = {
            'dataset_id': dataset_id,
            'spec': spec_dict,
            'schema': feature_cols,
            'row_count': len(df),
            'feature_count': len(feature_cols),
            'created_at': datetime.utcnow().isoformat()
        }
        
        # Include dropped features info if any were dropped
        if missing_in_master:
            metadata['dropped_features'] = missing_in_master
            metadata['requested_feature_count'] = len(missing_in_master) + len(feature_cols)
        
        with open(cache_meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        result = {
            'dataset_id': dataset_id,
            'schema': feature_cols,
            'row_count': len(df),
            'feature_count': len(feature_cols),
            'csv_path': clf_csv,
            'cached': False
        }
        
        # Include dropped features info in return value
        if missing_in_master:
            result['dropped_features'] = missing_in_master
            result['requested_feature_count'] = len(missing_in_master) + len(feature_cols)
        
        return result

