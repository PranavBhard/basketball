"""
Models Repository - Data access for model configuration collections.

Handles:
- model_config_nba: Win/loss classifier model configurations
- model_config_points_nba: Points regression model configurations

Note: For business logic (hash generation, validation), use ModelConfigManager.
This repository provides pure data access operations.
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
from datetime import datetime
from bson import ObjectId
from sportscore.db.base_repository import BaseRepository

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


class ClassifierConfigRepository(BaseRepository):
    """Repository for model_config_nba collection (classifier models)."""

    collection_name = 'nba_model_config'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["model_config_classifier"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_selected(self) -> Optional[Dict]:
        """Get the currently selected classifier config."""
        return self.find_one({'selected': True})

    def find_by_hash(self, config_hash: str) -> Optional[Dict]:
        """Find config by its unique hash."""
        return self.find_one({'config_hash': config_hash})

    def find_by_id(self, config_id: str) -> Optional[Dict]:
        """Find config by MongoDB ObjectId."""
        return self.find_one({'_id': ObjectId(config_id)})

    def find_all(
        self,
        trained_only: bool = False,
        limit: int = 0
    ) -> List[Dict]:
        """Get all classifier configs."""
        query = {}
        if trained_only:
            query['model_artifact_path'] = {'$exists': True}
        return self.find(query, sort=[('trained_at', -1)], limit=limit)

    def find_ensembles(self) -> List[Dict]:
        """Get all ensemble configurations."""
        return self.find({'ensemble_models': {'$exists': True}}, sort=[('trained_at', -1)])

    def find_by_ids(self, ids: List[str], projection: Optional[Dict] = None) -> List[Dict]:
        """Batch-fetch documents by a list of string ObjectId values."""
        if not ids:
            return []
        return self.find(
            {'_id': {'$in': [ObjectId(i) for i in ids]}},
            projection=projection,
        )

    def find_by_model_type(self, model_type: str) -> List[Dict]:
        """Find configs by model type."""
        return self.find({'model_type': model_type}, sort=[('trained_at', -1)])

    def find_by_run_id(self, run_id: str) -> Optional[Dict]:
        """Find config associated with a specific run ID."""
        return self.find_one({'run_id': run_id})

    # --- Update Methods ---

    def upsert_config(self, config_hash: str, config_data: Dict) -> bool:
        """Insert or update a config by hash."""
        config_data['updated_at'] = datetime.utcnow()
        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': config_data},
            upsert=True
        )
        return result.acknowledged

    def set_selected(self, config_hash: str) -> bool:
        """Set a config as the selected one (deselects others)."""
        # Deselect all others first
        self.update_many({}, {'$set': {'selected': False}})
        # Select the specified config
        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def set_selected_by_id(self, config_id: str) -> bool:
        """Set a config as selected by its ObjectId."""
        self.update_many({}, {'$set': {'selected': False}})
        result = self.update_one(
            {'_id': ObjectId(config_id)},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def update_training_results(
        self,
        config_hash: str,
        accuracy: float,
        log_loss: float = None,
        brier: float = None,
        model_artifact_path: str = None,
        scaler_artifact_path: str = None,
        features_path: str = None
    ) -> bool:
        """Update training results for a config."""
        update_data = {
            'accuracy': accuracy,
            'trained_at': datetime.utcnow()
        }
        if log_loss is not None:
            update_data['log_loss'] = log_loss
        if brier is not None:
            update_data['brier'] = brier
        if model_artifact_path:
            update_data['model_artifact_path'] = model_artifact_path
        if scaler_artifact_path:
            update_data['scaler_artifact_path'] = scaler_artifact_path
        if features_path:
            update_data['features_path'] = features_path

        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def update_training_csv(self, config_hash: str, csv_path: str) -> bool:
        """Update the training CSV path for a config."""
        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': {'training_csv': csv_path}}
        )
        return result.modified_count > 0

    def delete_config(self, config_hash: str) -> bool:
        """Delete a config by hash."""
        result = self.delete_one({'config_hash': config_hash})
        return result.deleted_count > 0

    # --- Utility Methods ---

    def config_exists(self, config_hash: str) -> bool:
        """Check if a config with this hash exists."""
        return self.exists({'config_hash': config_hash})

    def has_selected(self) -> bool:
        """Check if any config is currently selected."""
        return self.exists({'selected': True})


class PointsConfigRepository(BaseRepository):
    """Repository for model_config_points_nba collection (points regression models)."""

    collection_name = 'model_config_points_nba'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["model_config_points"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_selected(self) -> Optional[Dict]:
        """Get the currently selected points config."""
        return self.find_one({'selected': True})

    def find_by_hash(self, config_hash: str) -> Optional[Dict]:
        """Find config by its unique hash."""
        return self.find_one({'config_hash': config_hash})

    def find_by_id(self, config_id: str) -> Optional[Dict]:
        """Find config by MongoDB ObjectId."""
        return self.find_one({'_id': ObjectId(config_id)})

    def find_all(
        self,
        trained_only: bool = False,
        limit: int = 0
    ) -> List[Dict]:
        """Get all points configs."""
        query = {}
        if trained_only:
            query['model_artifact_path'] = {'$exists': True}
        return self.find(query, sort=[('trained_at', -1)], limit=limit)

    def find_by_target(self, target: str) -> List[Dict]:
        """Find configs by prediction target (e.g., 'home_points', 'total')."""
        return self.find({'target': target}, sort=[('trained_at', -1)])

    def find_by_model_type(self, model_type: str) -> List[Dict]:
        """Find configs by model type."""
        return self.find({'model_type': model_type}, sort=[('trained_at', -1)])

    # --- Update Methods ---

    def upsert_config(self, config_hash: str, config_data: Dict) -> bool:
        """Insert or update a config by hash."""
        config_data['updated_at'] = datetime.utcnow()
        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': config_data},
            upsert=True
        )
        return result.acknowledged

    def set_selected(self, config_hash: str) -> bool:
        """Set a config as the selected one (deselects others)."""
        self.update_many({}, {'$set': {'selected': False}})
        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def set_selected_by_id(self, config_id: str) -> bool:
        """Set a config as selected by its ObjectId."""
        self.update_many({}, {'$set': {'selected': False}})
        result = self.update_one(
            {'_id': ObjectId(config_id)},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def update_training_results(
        self,
        config_hash: str,
        mae: float = None,
        rmse: float = None,
        r2: float = None,
        model_artifact_path: str = None,
        scaler_artifact_path: str = None,
        features_path: str = None
    ) -> bool:
        """Update training results for a points config."""
        update_data = {'trained_at': datetime.utcnow()}
        if mae is not None:
            update_data['mae'] = mae
        if rmse is not None:
            update_data['rmse'] = rmse
        if r2 is not None:
            update_data['r2'] = r2
        if model_artifact_path:
            update_data['model_artifact_path'] = model_artifact_path
        if scaler_artifact_path:
            update_data['scaler_artifact_path'] = scaler_artifact_path
        if features_path:
            update_data['features_path'] = features_path

        result = self.update_one(
            {'config_hash': config_hash},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def delete_config(self, config_hash: str) -> bool:
        """Delete a config by hash."""
        result = self.delete_one({'config_hash': config_hash})
        return result.deleted_count > 0

    # --- Utility Methods ---

    def config_exists(self, config_hash: str) -> bool:
        """Check if a config with this hash exists."""
        return self.exists({'config_hash': config_hash})

    def has_selected(self) -> bool:
        """Check if any config is currently selected."""
        return self.exists({'selected': True})


class ExperimentRunsRepository(BaseRepository):
    """Repository for experiment_runs collection (experiment tracking)."""

    collection_name = 'experiment_runs'

    def __init__(
        self,
        db,
        league: Optional["LeagueConfig"] = None,
        collection_name: Optional[str] = None,
    ):
        effective = collection_name
        if league is not None:
            effective = effective or league.collections["experiment_runs"]
        super().__init__(db, collection_name=effective)

    # --- Query Methods ---

    def find_by_run_id(self, run_id: str) -> Optional[Dict]:
        """Find a run by its unique run_id."""
        return self.find_one({'run_id': run_id})

    def find_by_session(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict]:
        """Get all runs for a session."""
        return self.find(
            {'session_id': session_id},
            sort=[('created_at', -1)],
            limit=limit
        )

    def find_baseline(self, session_id: str) -> Optional[Dict]:
        """Get the baseline run for a session."""
        return self.find_one({
            'session_id': session_id,
            'baseline': True
        })

    def find_by_model_type(
        self,
        model_type: str,
        limit: int = 100
    ) -> List[Dict]:
        """Find runs by model type."""
        return self.find(
            {'model_type': model_type},
            sort=[('created_at', -1)],
            limit=limit
        )

    def find_by_date_range(
        self,
        date_from: datetime = None,
        date_to: datetime = None,
        session_id: str = None,
        model_type: str = None,
        limit: int = 100
    ) -> List[Dict]:
        """Find runs within a date range with optional filters."""
        query = {}

        if session_id:
            query['session_id'] = session_id

        if model_type:
            query['model_type'] = model_type

        if date_from or date_to:
            query['created_at'] = {}
            if date_from:
                query['created_at']['$gte'] = date_from
            if date_to:
                query['created_at']['$lte'] = date_to

        return self.find(query, sort=[('created_at', -1)], limit=limit)

    # --- Create/Update Methods ---

    def create_run(self, run_doc: Dict) -> str:
        """Insert a new run document."""
        result = self.insert_one(run_doc)
        return str(result.inserted_id)

    def update_run(self, run_id: str, update_data: Dict) -> bool:
        """Update a run by run_id."""
        result = self.update_one(
            {'run_id': run_id},
            {'$set': update_data}
        )
        return result.modified_count > 0

    def set_baseline(self, run_id: str, session_id: str) -> bool:
        """Set a run as the baseline for a session (clears other baselines)."""
        # Unset all other baselines for this session
        self.update_many(
            {'session_id': session_id, 'baseline': True},
            {'$set': {'baseline': False}}
        )
        # Set this run as baseline
        result = self.update_one(
            {'run_id': run_id, 'session_id': session_id},
            {'$set': {'baseline': True}}
        )
        return result.modified_count > 0

    def clear_baselines(self, session_id: str) -> int:
        """Clear all baselines for a session."""
        result = self.update_many(
            {'session_id': session_id, 'baseline': True},
            {'$set': {'baseline': False}}
        )
        return result.modified_count

    # --- Utility Methods ---

    def count_by_session(self, session_id: str) -> int:
        """Get the number of runs for a session."""
        return self.count({'session_id': session_id})
