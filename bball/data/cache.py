"""
Cache Repositories - Data access for cached/precomputed data.

Handles:
- cached_league_stats: Precomputed league-level statistics for PER calculation
- nba_cached_elo_ratings: Cached Elo ratings per team/date/season
- point_predictions_cache: Cached point predictions for classifier features
"""

import pandas as pd
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from pymongo import ASCENDING, DESCENDING
from sportscore.db.base_repository import BaseRepository

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


class LeagueStatsCache(BaseRepository):
    """Repository for cached_league_stats collection."""

    collection_name = 'cached_league_stats'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["cached_league_stats"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_season(self, season: str) -> Optional[Dict]:
        """Get cached league stats for a season."""
        return self.find_one({'season': season})

    def find_all_seasons(self) -> List[Dict]:
        """Get cached stats for all seasons."""
        return self.find({}, sort=[('season', -1)])

    def get_cached_seasons(self) -> List[str]:
        """Get list of seasons that have cached stats."""
        return self.distinct('season')

    def get_league_constants(self, season: str) -> Optional[Dict]:
        """Get league constants (factor, VOP, DRB%) for a season."""
        doc = self.find_by_season(season)
        if doc and 'league_constants' in doc:
            return doc['league_constants']
        return None

    def get_team_pace(self, season: str, team: str) -> Optional[float]:
        """Get average pace for a team in a season."""
        doc = self.find_by_season(season)
        if doc and 'team_pace' in doc:
            return doc['team_pace'].get(team)
        return None

    def get_league_pace(self, season: str) -> Optional[float]:
        """Get league average pace for a season."""
        doc = self.find_by_season(season)
        if doc:
            return doc.get('lg_pace')
        return None

    # --- Update Methods ---

    def upsert_season_stats(
        self,
        season: str,
        league_totals: Dict,
        league_constants: Dict,
        lg_pace: float,
        team_pace: Dict[str, float],
        team_games: Dict[str, int]
    ) -> bool:
        """Insert or update cached stats for a season."""
        result = self.update_one(
            {'season': season},
            {'$set': {
                'season': season,
                'league_totals': league_totals,
                'league_constants': league_constants,
                'lg_pace': lg_pace,
                'team_pace': team_pace,
                'team_games': team_games,
                'computed_at': datetime.utcnow()
            }},
            upsert=True
        )
        return result.acknowledged

    def delete_season(self, season: str) -> bool:
        """Delete cached stats for a season."""
        result = self.delete_one({'season': season})
        return result.deleted_count > 0

    def clear_all(self) -> int:
        """Clear all cached stats. Returns number of docs deleted."""
        result = self.delete_many({})
        return result.deleted_count


class EloRatingsCache(BaseRepository):
    """Repository for nba_cached_elo_ratings collection."""

    collection_name = 'nba_cached_elo_ratings'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["elo_cache"]
        super().__init__(db, collection_name=effective)
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Create required indexes if they don't exist."""
        # Unique compound index for lookups
        self._collection.create_index(
            [("team", ASCENDING), ("game_date", ASCENDING), ("season", ASCENDING)],
            unique=True,
            name="team_date_season_unique"
        )
        # Index for season-based queries
        self._collection.create_index(
            [("season", ASCENDING)],
            name="season_idx"
        )
        # Index for finding latest ratings
        self._collection.create_index(
            [("game_date", DESCENDING)],
            name="game_date_desc_idx"
        )

    # --- Query Methods ---

    def get_elo(
        self,
        team: str,
        game_date: str,
        season: str
    ) -> Optional[float]:
        """Get Elo rating for a team on a specific date."""
        doc = self.find_one({
            'team': team,
            'game_date': game_date,
            'season': season
        })
        return doc.get('elo') if doc else None

    def get_latest_elo(self, team: str, season: str) -> Optional[Dict]:
        """Get most recent Elo rating for a team in a season."""
        results = self.find(
            {'team': team, 'season': season},
            sort=[('game_date', DESCENDING)],
            limit=1
        )
        return results[0] if results else None

    def get_elos_before_date(
        self,
        team: str,
        game_date: str,
        season: str
    ) -> Optional[Dict]:
        """Get the most recent Elo rating for a team before a date."""
        results = self.find(
            {'team': team, 'season': season, 'game_date': {'$lt': game_date}},
            sort=[('game_date', DESCENDING)],
            limit=1
        )
        return results[0] if results else None

    def get_all_teams_elo(
        self,
        game_date: str,
        season: str
    ) -> Dict[str, float]:
        """Get Elo ratings for all teams on a specific date."""
        results = self.find({'game_date': game_date, 'season': season})
        return {doc['team']: doc['elo'] for doc in results}

    def get_season_elos(self, season: str) -> List[Dict]:
        """Get all Elo ratings for a season."""
        return self.find({'season': season}, sort=[('game_date', 1), ('team', 1)])

    def get_cached_seasons(self) -> List[str]:
        """Get list of seasons with cached Elo ratings."""
        return sorted(self.distinct('season'), reverse=True)

    def get_cached_dates(self, season: str) -> List[str]:
        """Get list of dates with cached ratings for a season."""
        return sorted(self.distinct('game_date', {'season': season}))

    # --- Update Methods ---

    def upsert_elo(
        self,
        team: str,
        game_date: str,
        season: str,
        elo: float
    ) -> bool:
        """Insert or update an Elo rating."""
        result = self.update_one(
            {'team': team, 'game_date': game_date, 'season': season},
            {'$set': {
                'team': team,
                'game_date': game_date,
                'season': season,
                'elo': elo,
                'created_at': datetime.utcnow()
            }},
            upsert=True
        )
        return result.acknowledged

    def bulk_upsert_elos(self, elo_records: List[Dict]) -> int:
        """
        Bulk upsert Elo ratings.

        Args:
            elo_records: List of dicts with keys: team, game_date, season, elo

        Returns:
            Number of records processed
        """
        if not elo_records:
            return 0

        from pymongo import UpdateOne
        operations = []
        for record in elo_records:
            operations.append(UpdateOne(
                {
                    'team': record['team'],
                    'game_date': record['game_date'],
                    'season': record['season']
                },
                {'$set': {
                    'team': record['team'],
                    'game_date': record['game_date'],
                    'season': record['season'],
                    'elo': record['elo'],
                    'created_at': datetime.utcnow()
                }},
                upsert=True
            ))

        if operations:
            result = self._collection.bulk_write(operations, ordered=False)
            return result.upserted_count + result.modified_count
        return 0

    def delete_season(self, season: str) -> int:
        """Delete all cached Elo ratings for a season."""
        result = self.delete_many({'season': season})
        return result.deleted_count

    def clear_all(self) -> int:
        """Clear all cached Elo ratings."""
        result = self.delete_many({})
        return result.deleted_count

    # --- Utility Methods ---

    def has_elo(self, team: str, game_date: str, season: str) -> bool:
        """Check if Elo rating exists for this team/date/season."""
        return self.exists({
            'team': team,
            'game_date': game_date,
            'season': season
        })

    def count_by_season(self, season: str) -> int:
        """Count cached ratings for a season."""
        return self.count({'season': season})


class PointPredictionCache(BaseRepository):
    """
    Repository for point_predictions_cache collection.

    Manages caching and retrieval of point predictions for use as classifier features.
    Predictions are stored in MongoDB and can be merged into datasets on-demand.
    """

    collection_name = 'point_predictions_cache'

    def __init__(self, db, collection_name: Optional[str] = None):
        super().__init__(db, collection_name=collection_name)
        self._ensure_indexes()

    def _ensure_indexes(self):
        """Ensure indexes exist for fast lookups."""
        try:
            self._collection.create_index(
                [('game_id', ASCENDING), ('model_id', ASCENDING)],
                unique=True,
                name="game_model_unique"
            )
            self._collection.create_index(
                [('model_id', ASCENDING)],
                name="model_id_idx"
            )
        except Exception:
            pass

    # --- Write Methods ---

    def cache_predictions(
        self,
        model_id: str,
        predictions: List[Dict],
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Cache point predictions for a model.

        Args:
            model_id: Unique identifier for the point prediction model
            predictions: List of prediction dicts, each with:
                - game_id: str (ESPN game_id)
                - pred_home_points: float
                - pred_away_points: float
                - year: int (optional, for fallback matching)
                - month: int (optional, for fallback matching)
                - day: int (optional, for fallback matching)
                - home_team: str (optional, for fallback matching)
                - away_team: str (optional, for fallback matching)
            metadata: Optional metadata about the model (e.g., training config)

        Returns:
            Number of predictions cached
        """
        from pymongo import UpdateOne

        now = datetime.utcnow()
        operations = []

        for pred in predictions:
            game_id = pred.get('game_id', '')

            pred_home = pred.get('pred_home_points')
            pred_away = pred.get('pred_away_points')
            pred_margin = pred.get('pred_margin')

            if pred_home is None and pred_away is None and pred_margin is not None:
                doc = {
                    'game_id': game_id,
                    'model_id': model_id,
                    'pred_home_points': None,
                    'pred_away_points': None,
                    'pred_margin': float(pred_margin),
                    'created_at': now,
                    'metadata': {
                        'year': pred.get('year'),
                        'month': pred.get('month'),
                        'day': pred.get('day'),
                        'home_team': pred.get('home_team', ''),
                        'away_team': pred.get('away_team', ''),
                        'target_type': 'margin'
                    }
                }
            else:
                doc = {
                    'game_id': game_id,
                    'model_id': model_id,
                    'pred_home_points': float(pred_home) if pred_home is not None else 0.0,
                    'pred_away_points': float(pred_away) if pred_away is not None else 0.0,
                    'created_at': now,
                    'metadata': {
                        'year': pred.get('year'),
                        'month': pred.get('month'),
                        'day': pred.get('day'),
                        'home_team': pred.get('home_team', ''),
                        'away_team': pred.get('away_team', ''),
                        'target_type': 'home_away'
                    }
                }
                if pred_home is not None and pred_away is not None:
                    doc['pred_margin'] = float(pred_home) - float(pred_away)

            if metadata:
                doc['model_metadata'] = metadata

            operations.append(
                UpdateOne(
                    {'game_id': game_id, 'model_id': model_id},
                    {'$set': doc},
                    upsert=True
                )
            )

        batch_size = 1000
        cached_count = 0
        for i in range(0, len(operations), batch_size):
            batch = operations[i:i + batch_size]
            result = self._collection.bulk_write(batch, ordered=False)
            cached_count += result.upserted_count + result.modified_count

        return cached_count

    # --- Read Methods ---

    def load_predictions(
        self,
        model_id: str,
        game_ids: Optional[List[str]] = None
    ) -> "pd.DataFrame":
        """
        Load cached point predictions.

        Args:
            model_id: Model identifier
            game_ids: Optional list of game_ids to filter

        Returns:
            DataFrame with columns: game_id, pred_home_points, pred_away_points, metadata
        """
        query = {'model_id': model_id}
        if game_ids:
            query['game_id'] = {'$in': game_ids}

        predictions = self.find(query)

        if not predictions:
            return pd.DataFrame()

        data = []
        for pred in predictions:
            data.append({
                'game_id': pred.get('game_id', ''),
                'pred_home_points': pred.get('pred_home_points'),
                'pred_away_points': pred.get('pred_away_points'),
                'pred_margin': pred.get('pred_margin'),
                'year': pred.get('metadata', {}).get('year'),
                'month': pred.get('metadata', {}).get('month'),
                'day': pred.get('metadata', {}).get('day'),
                'home_team': pred.get('metadata', {}).get('home_team', ''),
                'away_team': pred.get('metadata', {}).get('away_team', ''),
                'target_type': pred.get('metadata', {}).get('target_type', 'home_away')
            })

        return pd.DataFrame(data)

    def merge_predictions_into_dataframe(
        self,
        df: "pd.DataFrame",
        model_id: str
    ) -> "pd.DataFrame":
        """
        Merge point predictions into a dataframe.

        Args:
            df: DataFrame with master training data (must have game_id or Year/Month/Day/Home/Away)
            model_id: Model identifier for predictions to merge

        Returns:
            DataFrame with added columns: pred_home_points, pred_away_points, pred_margin, pred_point_total
        """
        pred_count = self.count({'model_id': model_id})
        if pred_count == 0:
            raise ValueError(f"No predictions found for model_id: {model_id}")

        pred_df = self.load_predictions(model_id)

        if pred_df.empty:
            raise ValueError(f"No predictions loaded for model_id: {model_id}")

        has_game_id = 'game_id' in df.columns

        merge_cols = ['game_id', 'pred_margin']
        if 'pred_home_points' in pred_df.columns:
            merge_cols.append('pred_home_points')
        if 'pred_away_points' in pred_df.columns:
            merge_cols.append('pred_away_points')

        if has_game_id:
            df = df.merge(
                pred_df[merge_cols],
                on='game_id',
                how='left'
            )
        else:
            if not all(col in df.columns for col in ['Year', 'Month', 'Day', 'Home', 'Away']):
                raise ValueError("DataFrame must have game_id or (Year, Month, Day, Home, Away) columns")

            merge_cols_with_meta = merge_cols + ['year', 'month', 'day', 'home_team', 'away_team']
            merge_cols_with_meta = [col for col in merge_cols_with_meta if col in pred_df.columns]

            df = df.merge(
                pred_df[merge_cols_with_meta],
                left_on=['Year', 'Month', 'Day', 'Home', 'Away'],
                right_on=['year', 'month', 'day', 'home_team', 'away_team'],
                how='left'
            )
            df = df.drop(columns=['year', 'month', 'day', 'home_team', 'away_team'], errors='ignore')

        if 'pred_home_points' not in df.columns:
            df['pred_home_points'] = None
        if 'pred_away_points' not in df.columns:
            df['pred_away_points'] = None
        if 'pred_margin' not in df.columns:
            df['pred_margin'] = None

        has_home_away = df['pred_home_points'].notna().any() or df['pred_away_points'].notna().any()

        if has_home_away:
            df['pred_home_points'] = df['pred_home_points'].fillna(0.0)
            df['pred_away_points'] = df['pred_away_points'].fillna(0.0)

            mask_margin_na = df['pred_margin'].isna()
            if mask_margin_na.any():
                df.loc[mask_margin_na, 'pred_margin'] = (
                    df.loc[mask_margin_na, 'pred_home_points'] -
                    df.loc[mask_margin_na, 'pred_away_points']
                )

            df['pred_point_total'] = df['pred_home_points'] + df['pred_away_points']
        else:
            df['pred_point_total'] = None

        return df

    # --- Query Methods ---

    def get_model_ids(self) -> List[str]:
        """Get list of all model_ids with cached predictions."""
        return self.distinct('model_id')

    def delete_predictions(self, model_id: str) -> int:
        """Delete all predictions for a model."""
        result = self.delete_many({'model_id': model_id})
        return result.deleted_count

    def get_stats(self, model_id: Optional[str] = None) -> Dict:
        """
        Get cache statistics.

        Args:
            model_id: Optional model_id to filter

        Returns:
            Dict with statistics
        """
        query = {'model_id': model_id} if model_id else {}

        total = self.count(query)
        models = self.distinct('model_id', query) if not model_id else [model_id]

        return {
            'total_predictions': total,
            'model_ids': models,
            'model_count': len(models)
        }
