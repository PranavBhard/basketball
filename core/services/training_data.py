"""
Training Data Service

Core service for master training data generation, management, and updates.
This is the Single Source of Truth for all training data operations.

Both CLI and web endpoints should call this service.
"""

import os
import pandas as pd
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union, TYPE_CHECKING

from bball_app.core.mongo import Mongo
from bball_app.core.league_config import load_league_config

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig


# Get project root (assuming this file is in nba_app/core/services/)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Consolidated: place master training in parent ../master_training/
_PARENT_ROOT = os.path.dirname(_PROJECT_ROOT)

# =============================================================================
# Module-level constants for backward compatibility
# =============================================================================
MASTER_TRAINING_PATH = os.path.join(_PARENT_ROOT, 'master_training', 'MASTER_TRAINING.csv')
MASTER_COLLECTION = 'master_training_data_nba'


class TrainingDataService:
    """
    Core service for master training data operations.

    Provides unified interface for:
    - Generating master training CSV with all features
    - Updating training data incrementally
    - Regenerating specific seasons
    - Full regeneration from scratch
    - Extracting feature subsets
    """

    def __init__(
        self,
        db=None,
        league: Union[str, "LeagueConfig", None] = None,
        job_id: str = None
    ):
        """
        Initialize the training data service.

        Args:
            db: MongoDB database connection (uses Mongo() if None)
            league: League config or ID (defaults to 'nba')
            job_id: Optional job ID for progress tracking
        """
        self.db = db if db is not None else Mongo().db
        self.league = self._resolve_league(league)
        self.job_id = job_id

        # Paths
        self.master_path = self._get_master_path()
        self.collection_name = self._get_collection_name()

    def _resolve_league(self, league: Union[str, "LeagueConfig", None]) -> "LeagueConfig":
        """Resolve league parameter to LeagueConfig instance."""
        if league is None:
            return load_league_config("nba")
        if isinstance(league, str):
            return load_league_config(league)
        return league

    def _get_master_path(self) -> str:
        """Get the master training CSV path for the current league."""
        return self.league.master_training_csv if self.league else os.path.join(
            _PARENT_ROOT, 'master_training', 'MASTER_TRAINING.csv'
        )

    def _get_collection_name(self) -> str:
        """Get the master training metadata collection name."""
        league_id = self.league.league_id if self.league else "nba"
        return f"master_training_data_{league_id}"

    @property
    def _exclude_game_types(self) -> list:
        """Get excluded game types from league config, with fallback."""
        return self.league.exclude_game_types if self.league else ['preseason', 'allstar']

    def _update_progress(self, current: int, total: int, pct: int, message: str = ""):
        """Update job progress in MongoDB if job_id is set."""
        if self.job_id:
            from bson import ObjectId
            self.db.jobs_nba.update_one(
                {'_id': ObjectId(self.job_id)},
                {'$set': {
                    'progress': pct,
                    'message': message,
                    'updated_at': datetime.utcnow()
                }}
            )

    # =========================================================================
    # Feature Registry
    # =========================================================================

    def get_all_possible_features(self, no_player: bool = False) -> List[str]:
        """
        Get all possible features from the feature registry.

        Args:
            no_player: If True, exclude player-level features (PER, injuries)

        Returns:
            List of feature names
        """
        from bball_app.core.features.registry import FeatureGroups

        if no_player:
            # Get all groups except player-related ones
            all_groups = FeatureGroups.get_all_group_definitions(league=self.league)
            player_groups = {FeatureGroups.PLAYER_TALENT, FeatureGroups.INJURIES}
            all_features = []
            for group_name in all_groups.keys():
                if group_name not in player_groups:
                    all_features.extend(
                        FeatureGroups.get_features_for_group(group_name, include_side=True, league=self.league)
                    )
            return sorted(set(all_features))
        else:
            return FeatureGroups.get_all_features_flat(include_side=True, league=self.league)

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    def get_metadata(self) -> Optional[Dict]:
        """Get master training metadata from MongoDB."""
        return self.db[self.collection_name].find_one({'type': 'master_meta'})

    def update_metadata(
        self,
        features: List[str],
        feature_count: int,
        last_date: str,
        options: Dict = None
    ) -> None:
        """Create or update master training metadata."""
        meta_doc = {
            'type': 'master_meta',
            'master_path': self.master_path,
            'features': features,
            'feature_count': feature_count,
            'last_date_updated': last_date,
            'options': options or {},
            'updated_at': datetime.utcnow()
        }

        self.db[self.collection_name].update_one(
            {'type': 'master_meta'},
            {'$set': meta_doc},
            upsert=True
        )

    # =========================================================================
    # Season Operations
    # =========================================================================

    def get_available_seasons(self) -> Dict[str, Dict]:
        """
        Get available seasons from MongoDB with game counts and master training status.

        Returns:
            Dict mapping season strings to info dicts:
            {
                '2024-2025': {'count': 567, 'in_master': True, 'master_count': 500},
                ...
            }
        """
        games_collection = self.league.collections.get("games", "stats_nba")

        # Get season counts from MongoDB
        pipeline = [
            {'$match': {
                'season': {'$exists': True},
                'homeTeam.points': {'$gt': 0},
                'awayTeam.points': {'$gt': 0},
                'game_type': {'$nin': self._exclude_game_types}
            }},
            {'$group': {'_id': '$season', 'count': {'$sum': 1}}},
            {'$sort': {'_id': 1}}
        ]

        mongo_seasons = {}
        for doc in self.db[games_collection].aggregate(pipeline):
            mongo_seasons[doc['_id']] = {'count': doc['count'], 'in_master': False}

        # Check which seasons are in master training CSV
        if os.path.exists(self.master_path):
            try:
                df = pd.read_csv(self.master_path, usecols=['Year', 'Month'])

                def get_season(row):
                    year = int(row['Year'])
                    month = int(row['Month'])
                    if month > 8:
                        return f"{year}-{year+1}"
                    else:
                        return f"{year-1}-{year}"

                df['season'] = df.apply(get_season, axis=1)
                master_season_counts = df['season'].value_counts().to_dict()

                for season in mongo_seasons:
                    if season in master_season_counts:
                        mongo_seasons[season]['in_master'] = True
                        mongo_seasons[season]['master_count'] = master_season_counts[season]
            except Exception as e:
                print(f"Warning: Could not read master training CSV: {e}")

        return mongo_seasons

    def regenerate_seasons(
        self,
        seasons: List[str],
        no_player: bool = False,
        progress_callback: callable = None
    ) -> Tuple[int, str]:
        """
        Regenerate specific seasons in master training CSV.

        Generates training data for the specified seasons, removes existing rows
        for those seasons from the master CSV, and inserts new rows in the correct
        order (sorted by date).

        Args:
            seasons: List of season strings to regenerate (e.g., ['2019-2020', '2020-2021'])
            no_player: If True, skip player-level features
            progress_callback: Optional callback function(current, total, pct, message)

        Returns:
            Tuple of (games_added_count, updated_master_path)
        """
        from bball_app.core.models.bball_model import BballModel

        if not seasons:
            raise ValueError("No seasons provided")

        def update_progress(current, total, pct, message=""):
            if progress_callback:
                progress_callback(current, total, pct, message)
            self._update_progress(current, total, pct, message)

        update_progress(0, 100, 0, f"Starting regeneration for seasons: {', '.join(seasons)}")

        # Build query for the specified seasons
        query = BballModel.DEFAULT_QUERY.copy()
        query['season'] = {'$in': seasons}

        # Get all possible features
        all_features = self.get_all_possible_features(no_player=no_player)

        update_progress(0, 100, 5, "Creating model...")

        # Create BballModel with all features
        model = BballModel(
            classifier_features=all_features,
            points_features=all_features,
            include_elo=True,
            use_exponential_weighting=False,
            include_era_normalization=True,
            include_per_features=not no_player,
            include_injuries=not no_player,
            recency_decay_k=15.0,
            output_dir=os.path.dirname(self.master_path),
            preload_data=True,
            master_training_mode=not no_player,
            league=self.league
        )
        model.feature_names = all_features

        update_progress(0, 100, 10, f"Generating training data for {len(seasons)} seasons...")

        # Generate training data for specified seasons
        temp_csv = self.master_path.replace('.csv', f'_temp_seasons_{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')

        def model_progress_callback(current, total, pct):
            scaled_pct = 10 + int(pct * 0.7)
            update_progress(current, total, scaled_pct, f"Processing games: {current}/{total}")

        game_count, clf_csv, _ = model.create_training_data(
            query=query,
            classifier_csv=temp_csv,
            min_games_filter=0,  # Master training includes ALL games; filtering done at model training time
            progress_callback=model_progress_callback
        )

        if game_count == 0:
            update_progress(0, 100, 100, f"No games found for seasons: {', '.join(seasons)}")
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            return 0, self.master_path

        update_progress(0, 100, 85, f"Generated {game_count} games. Merging with master CSV...")

        # Read new season data
        new_df = pd.read_csv(clf_csv)

        # Check if master CSV exists
        if os.path.exists(self.master_path):
            master_df = pd.read_csv(self.master_path)

            # Calculate season for each row to filter out regenerated seasons
            def get_season_from_row(row):
                year = int(row['Year'])
                month = int(row['Month'])
                if month > 8:
                    return f"{year}-{year+1}"
                else:
                    return f"{year-1}-{year}"

            master_df['_season'] = master_df.apply(get_season_from_row, axis=1)
            master_df = master_df[~master_df['_season'].isin(seasons)]
            master_df = master_df.drop(columns=['_season'])

            update_progress(0, 100, 90, "Aligning columns...")

            # Ensure column alignment
            master_cols = set(master_df.columns)
            new_cols = set(new_df.columns)

            for col in master_cols - new_cols:
                new_df[col] = 0
            for col in new_cols - master_cols:
                master_df[col] = 0

            all_cols = list(master_df.columns)
            for col in new_df.columns:
                if col not in all_cols:
                    all_cols.append(col)

            master_df = master_df.reindex(columns=all_cols, fill_value=0)
            new_df = new_df.reindex(columns=all_cols, fill_value=0)

            combined_df = pd.concat([master_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        update_progress(0, 100, 95, "Sorting and saving...")

        # Remove duplicates and sort
        combined_df = combined_df.drop_duplicates(
            subset=['Year', 'Month', 'Day', 'Home', 'Away'],
            keep='last'
        )
        combined_df = combined_df.sort_values(['Year', 'Month', 'Day', 'Home', 'Away'])

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.master_path), exist_ok=True)

        # Write updated master CSV
        combined_df.to_csv(self.master_path, index=False)

        # Clean up temp file
        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        update_progress(0, 100, 100, f"Completed! Added/updated {game_count} games for {len(seasons)} seasons")

        # Update metadata
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id',
                        'home_points', 'away_points', 'pred_home_points', 'pred_away_points',
                        'pred_margin', 'pred_point_total', 'pred_total']
        actual_features = [col for col in combined_df.columns if col not in metadata_cols]

        self.update_metadata(
            actual_features,
            len(actual_features),
            datetime.now().strftime('%Y-%m-%d'),
            {'no_player': no_player, 'regenerated_seasons': seasons}
        )

        return game_count, self.master_path

    def regenerate_full(
        self,
        no_player: bool = False,
        progress_callback: callable = None
    ) -> Tuple[int, str, List[str]]:
        """
        Regenerate the entire master training CSV from scratch.

        Args:
            no_player: If True, skip player-level features
            progress_callback: Optional callback function(current, total, pct, message)

        Returns:
            Tuple of (game_count, csv_path, feature_list)
        """
        from bball_app.core.models.bball_model import BballModel

        def update_progress(current, total, pct, message=""):
            if progress_callback:
                progress_callback(current, total, pct, message)
            self._update_progress(current, total, pct, message)

        update_progress(0, 100, 0, "Starting full master training regeneration...")

        # Backup existing file
        if os.path.exists(self.master_path):
            backup_path = self.master_path.replace(
                '.csv',
                f'.csv.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            import shutil
            shutil.copy2(self.master_path, backup_path)
            update_progress(0, 100, 2, f"Backed up existing master to: {os.path.basename(backup_path)}")

        # Get all possible features
        all_features = self.get_all_possible_features(no_player=no_player)

        update_progress(0, 100, 5, f"Creating model with {len(all_features)} features...")

        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.master_path), exist_ok=True)

        # Create BballModel
        model = BballModel(
            classifier_features=all_features,
            points_features=all_features,
            include_elo=True,
            use_exponential_weighting=False,
            include_era_normalization=True,
            include_per_features=not no_player,
            include_injuries=not no_player,
            recency_decay_k=15.0,
            output_dir=os.path.dirname(self.master_path),
            preload_data=True,
            master_training_mode=not no_player,
            league=self.league
        )
        model.feature_names = all_features

        # Generate training data
        def model_progress_callback(current, total, pct):
            scaled_pct = 5 + int(pct * 0.93)
            update_progress(current, total, scaled_pct, f"Processing games: {current}/{total}")

        game_count, clf_csv, _ = model.create_training_data(
            query=BballModel.DEFAULT_QUERY.copy(),
            classifier_csv=self.master_path,
            min_games_filter=0,  # Master training includes ALL games; filtering done at model training time
            progress_callback=model_progress_callback
        )

        update_progress(0, 100, 100, f"Completed! Generated {game_count} games with {len(all_features)} features")

        # Update metadata
        metadata_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id',
                        'home_points', 'away_points', 'pred_home_points', 'pred_away_points',
                        'pred_margin', 'pred_point_total', 'pred_total']
        actual_features = [f for f in all_features if f not in metadata_cols]

        self.update_metadata(
            actual_features,
            len(actual_features),
            datetime.now().strftime('%Y-%m-%d'),
            {'no_player': no_player, 'full_regeneration': True}
        )

        return game_count, self.master_path, all_features

    # =========================================================================
    # Incremental Updates
    # =========================================================================

    def update_incremental(
        self,
        start_date: str,
        end_date: str,
        progress_callback: callable = None
    ) -> Tuple[int, str]:
        """
        Incrementally update master training data with new games.

        Args:
            start_date: Start date for new games (YYYY-MM-DD, exclusive)
            end_date: End date for new games (YYYY-MM-DD, inclusive)
            progress_callback: Optional callback function(current, total, pct, message)

        Returns:
            Tuple of (games_added_count, updated_master_path)
        """
        from bball_app.core.models.bball_model import BballModel

        # Get master metadata
        master_meta = self.get_metadata()
        if not master_meta:
            raise ValueError("Master training data does not exist. Generate it first.")

        # Build query for date range
        if start_date == end_date:
            date_query = {'$eq': start_date}
        else:
            date_query = {'$gt': start_date, '$lte': end_date}

        query = {
            'date': date_query,
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': self._exclude_game_types}
        }

        # Get options from metadata
        options = master_meta.get('options', {})
        no_player = options.get('no_player', False)

        all_features = self.get_all_possible_features(no_player=no_player)

        # Create model
        model = BballModel(
            classifier_features=all_features,
            points_features=all_features,
            include_elo=True,
            use_exponential_weighting=False,
            include_era_normalization=True,
            include_per_features=not no_player,
            include_injuries=not no_player,
            recency_decay_k=15.0,
            output_dir=os.path.dirname(self.master_path),
            master_training_mode=not no_player,
            league=self.league
        )

        # Generate training data for new games
        temp_csv = self.master_path.replace('.csv', '_temp_incremental.csv')
        game_count, clf_csv, _ = model.create_training_data(
            query=query,
            classifier_csv=temp_csv,
            progress_callback=progress_callback
        )

        if game_count == 0:
            if os.path.exists(temp_csv):
                os.remove(temp_csv)
            return 0, self.master_path

        # Read and merge
        master_df = pd.read_csv(self.master_path)
        new_df = pd.read_csv(clf_csv)

        # Align columns
        master_cols = list(master_df.columns)
        for col in master_cols:
            if col not in new_df.columns:
                new_df[col] = 0
        new_df = new_df[master_cols]

        # Combine
        combined_df = pd.concat([master_df, new_df], ignore_index=True)
        combined_df = combined_df.drop_duplicates(
            subset=['Year', 'Month', 'Day', 'Home', 'Away'],
            keep='last'
        )
        combined_df = combined_df.sort_values(['Year', 'Month', 'Day', 'Home', 'Away'])

        combined_df.to_csv(self.master_path, index=False)

        if os.path.exists(temp_csv):
            os.remove(temp_csv)

        # Update metadata
        actual_features = [col for col in master_cols if col not in
                         ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon']]
        self.update_metadata(actual_features, len(actual_features), end_date, options)

        return game_count, self.master_path

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def extract_features(
        self,
        requested_features: List[str] = None,
        output_path: str = None
    ) -> str:
        """
        Extract selected features from master training CSV.

        Args:
            requested_features: List of feature names to extract (None = extract all)
            output_path: Output path for extracted CSV (defaults to temp file)

        Returns:
            Path to extracted CSV file
        """
        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Master training CSV not found: {self.master_path}")

        df = pd.read_csv(self.master_path)

        # Meta columns that should always be included
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon', 'game_id']

        if requested_features is None or len(requested_features) == 0:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.master_path.replace('MASTER_TRAINING.csv', f'extracted_training_{timestamp}.csv')
            df.to_csv(output_path, index=False)
            return output_path

        # Check which requested features exist
        available_features = [f for f in requested_features if f in df.columns]
        missing_features = [f for f in requested_features if f not in df.columns]

        if missing_features:
            raise ValueError(
                f"Requested features not found in master CSV: {missing_features}. "
                f"Master needs to be regenerated to include all possible features."
            )

        # Preserve CSV column order
        csv_feature_cols = [c for c in df.columns if c not in meta_cols]
        ordered_features = [f for f in csv_feature_cols if f in available_features]
        columns_to_extract = meta_cols + ordered_features
        extracted_df = df[columns_to_extract]

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.master_path.replace('MASTER_TRAINING.csv', f'extracted_training_{timestamp}.csv')

        extracted_df.to_csv(output_path, index=False)
        return output_path

    def extract_features_for_points(
        self,
        requested_features: List[str] = None,
        output_path: str = None,
        begin_year: int = None,
        min_games_played: int = 15
    ) -> str:
        """
        Extract selected features from master training CSV for points regression.

        Args:
            requested_features: List of feature names to extract (None = extract all)
            output_path: Output path for extracted CSV
            begin_year: Minimum season start year to include
            min_games_played: Minimum games played filter (default 15)

        Returns:
            Path to extracted CSV file
        """
        import numpy as np

        if not os.path.exists(self.master_path):
            raise FileNotFoundError(f"Master training CSV not found: {self.master_path}")

        df = pd.read_csv(self.master_path)

        # Filter by begin_year if provided
        if begin_year is not None:
            df = df.copy()
            df['SeasonStartYear'] = np.where(
                df['Month'].astype(int) >= 10,
                df['Year'].astype(int),
                df['Year'].astype(int) - 1
            )
            df = df[df['SeasonStartYear'] >= int(begin_year)]
            df = df.drop('SeasonStartYear', axis=1)

        # Apply min_games_played filter
        if min_games_played is not None and min_games_played > 0:
            before_count = len(df)
            df = df.copy()

            df['Season'] = np.where(
                df['Month'].astype(int) >= 10,
                df['Year'].astype(int).astype(str) + '-' + (df['Year'].astype(int) + 1).astype(str),
                (df['Year'].astype(int) - 1).astype(str) + '-' + df['Year'].astype(int).astype(str)
            )

            df['_date_key'] = (df['Year'].astype(int) * 10000) + (df['Month'].astype(int) * 100) + df['Day'].astype(int)

            # Home prior counts
            home_keys = ['Year', 'Month', 'Day', 'Home']
            home_seq = df[home_keys + ['Season', '_date_key']].copy()
            home_seq = home_seq.sort_values(['Season', 'Home', '_date_key'])
            home_seq['_homePrior'] = home_seq.groupby(['Season', 'Home']).cumcount()
            df = df.merge(home_seq[home_keys + ['_homePrior']], on=home_keys, how='left')

            # Away prior counts
            away_keys = ['Year', 'Month', 'Day', 'Away']
            away_seq = df[away_keys + ['Season', '_date_key']].copy()
            away_seq = away_seq.sort_values(['Season', 'Away', '_date_key'])
            away_seq['_awayPrior'] = away_seq.groupby(['Season', 'Away']).cumcount()
            df = df.merge(away_seq[away_keys + ['_awayPrior']], on=away_keys, how='left')

            df = df[(df['_homePrior'] >= min_games_played) & (df['_awayPrior'] >= min_games_played)].copy()
            df.drop(columns=[c for c in ['_date_key', '_homePrior', '_awayPrior', 'Season'] if c in df.columns], inplace=True)

            if len(df) == 0:
                raise ValueError(
                    f'No training data after min_games_played >= {min_games_played}. '
                    f'Try reducing min_games_played or check your data.'
                )

        # Meta columns for points regression
        base_meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away']
        meta_cols = base_meta_cols.copy()

        if 'game_id' in df.columns:
            meta_cols.append('game_id')
        if 'home_points' in df.columns:
            meta_cols.append('home_points')
        if 'away_points' in df.columns:
            meta_cols.append('away_points')

        if 'home_points' not in df.columns or 'away_points' not in df.columns:
            raise ValueError("Master CSV missing home_points or away_points columns")

        if requested_features is None or len(requested_features) == 0:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.master_path.replace('MASTER_TRAINING.csv', f'extracted_points_{timestamp}.csv')
            columns_to_extract = [col for col in df.columns if col not in ['HomeWon']]
            df[columns_to_extract].to_csv(output_path, index=False)
            return output_path

        # Check features
        available_features = [f for f in requested_features if f in df.columns]
        missing_features = [f for f in requested_features if f not in df.columns]

        if missing_features:
            raise ValueError(f"Requested features not found: {missing_features}")

        csv_feature_cols = [c for c in df.columns if c not in meta_cols and c != 'HomeWon']
        ordered_features = [f for f in csv_feature_cols if f in available_features]
        columns_to_extract = meta_cols + ordered_features
        extracted_df = df[columns_to_extract]

        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.master_path.replace('MASTER_TRAINING.csv', f'extracted_points_{timestamp}.csv')

        extracted_df.to_csv(output_path, index=False)
        return output_path

    # =========================================================================
    # Validation & Registration
    # =========================================================================

    def check_needs_regeneration(self, requested_features: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if master training data needs regeneration based on requested features.

        Args:
            requested_features: List of requested feature names

        Returns:
            Tuple of (needs_regeneration: bool, missing_features: List[str])
        """
        # Check actual CSV header first
        try:
            if os.path.exists(self.master_path):
                header_df = pd.read_csv(self.master_path, nrows=0)
                master_cols = set(header_df.columns)
                requested_set = set(requested_features or [])

                missing = list(requested_set - master_cols)
                if missing:
                    return True, missing
                return False, []
        except Exception:
            pass

        # Fallback to metadata
        master_meta = self.get_metadata()
        if not master_meta:
            return True, requested_features

        master_features = set(master_meta.get('features', []))
        requested_set = set(requested_features)
        missing = list(requested_set - master_features)

        return bool(missing), missing

    def register_existing_csv(self, master_path: str = None, options: Dict = None) -> Dict:
        """
        Register an existing master training CSV file in MongoDB.

        Args:
            master_path: Path to existing master CSV (defaults to self.master_path)
            options: Optional configuration options

        Returns:
            Metadata dict that was saved
        """
        master_path = master_path or self.master_path

        if not os.path.exists(master_path):
            raise FileNotFoundError(f"Master training CSV not found: {master_path}")

        print(f"Reading existing master CSV: {master_path}")

        # Read CSV with error handling
        try:
            df = pd.read_csv(master_path, on_bad_lines='skip', engine='python')
        except TypeError:
            try:
                df = pd.read_csv(master_path, error_bad_lines=False, warn_bad_lines=True, engine='python')
            except Exception:
                import csv as csv_module
                rows = []
                with open(master_path, 'r', encoding='utf-8') as f:
                    reader = csv_module.reader(f)
                    header = next(reader)
                    for row in reader:
                        if len(row) == len(header):
                            rows.append(row)
                df = pd.DataFrame(rows, columns=header)
                for col in df.columns:
                    if col not in ['Home', 'Away']:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception:
                            pass

        # Meta columns
        meta_cols = ['Year', 'Month', 'Day', 'Home', 'Away', 'HomeWon']

        # Extract feature list
        feature_list = [col for col in df.columns if col not in meta_cols]
        feature_count = len(feature_list)

        # Find latest date
        if len(df) > 0:
            df_sorted = df.sort_values(['Year', 'Month', 'Day'], ascending=False)
            latest_row = df_sorted.iloc[0]
            last_date = f"{int(latest_row['Year'])}-{int(latest_row['Month']):02d}-{int(latest_row['Day']):02d}"
        else:
            last_date = datetime.now().strftime('%Y-%m-%d')

        print(f"Found {len(df)} games, {feature_count} features, latest date: {last_date}")

        # Update metadata
        self.update_metadata(feature_list, feature_count, last_date, options)

        print(f"Registered master training data in MongoDB")
        print(f"  Features: {feature_count}")
        print(f"  Games: {len(df)}")
        print(f"  Last updated: {last_date}")

        return {
            'file_path': master_path,
            'feature_list': feature_list,
            'feature_count': feature_count,
            'last_date_updated': last_date,
            'options': options or {},
            'game_count': len(df)
        }


# =============================================================================
# Convenience Functions (Backward Compatibility)
# =============================================================================
# These standalone functions provide backward compatibility with the old
# master_training_data interface. New code should use
# TrainingDataService directly.

def get_master_training_path(league: Union[str, "LeagueConfig", None] = None) -> str:
    """Get the league-specific master training CSV path."""
    service = TrainingDataService(league=league)
    return service.master_path


def get_master_collection_name(league: Union[str, "LeagueConfig", None] = None) -> str:
    """Get the league-specific Mongo collection for master training metadata."""
    service = TrainingDataService(league=league)
    return service.collection_name


def get_all_possible_features(no_player: bool = False) -> List[str]:
    """Get all possible features from the feature registry."""
    service = TrainingDataService()
    return service.get_all_possible_features(no_player=no_player)


def get_available_seasons(league: Union[str, "LeagueConfig", None] = None) -> Dict[str, Dict]:
    """Get available seasons from MongoDB with game counts."""
    service = TrainingDataService(league=league)
    return service.get_available_seasons()


def extract_features_from_master(
    master_path: str,
    requested_features: List[str] = None,
    output_path: str = None
) -> str:
    """Extract selected features from master training CSV."""
    service = TrainingDataService()
    # Override the master_path if provided
    if master_path:
        service.master_path = master_path
    return service.extract_features(requested_features, output_path)


def extract_features_from_master_for_points(
    master_path: str,
    requested_features: List[str] = None,
    output_path: str = None,
    begin_year: int = None,
    min_games_played: int = 15
) -> str:
    """Extract selected features from master training CSV for points regression."""
    service = TrainingDataService()
    if master_path:
        service.master_path = master_path
    return service.extract_features_for_points(
        requested_features, output_path, begin_year, min_games_played
    )


def check_master_needs_regeneration(
    db,
    requested_features: List[str],
    league: Union[str, "LeagueConfig", None] = None
) -> Tuple[bool, List[str]]:
    """Check if master training data needs regeneration."""
    service = TrainingDataService(db=db, league=league)
    return service.check_needs_regeneration(requested_features)


def register_existing_master_csv(
    db,
    master_path: str = None,
    options: Dict = None,
    league: Union[str, "LeagueConfig", None] = None
) -> Dict:
    """Register an existing master training CSV file in MongoDB."""
    service = TrainingDataService(db=db, league=league)
    return service.register_existing_csv(master_path, options)


def get_master_training_metadata(
    db=None,
    league: Union[str, "LeagueConfig", None] = None
) -> Optional[Dict]:
    """Get master training metadata from MongoDB."""
    service = TrainingDataService(db=db, league=league)
    return service.get_metadata()


def generate_master_training_data(
    no_player: bool = False,
    progress_callback: callable = None,
    league: Union[str, "LeagueConfig", None] = None
) -> Tuple[str, List[str], int]:
    """
    Generate master training data from scratch.

    Args:
        no_player: If True, skip player-level features
        progress_callback: Optional callback function(current, total, pct, message)
        league: League config or ID (defaults to 'nba')

    Returns:
        Tuple of (master_path, feature_list, game_count)
    """
    service = TrainingDataService(league=league)
    game_count, master_path, features = service.regenerate_full(
        no_player=no_player,
        progress_callback=progress_callback
    )
    return master_path, features, game_count
