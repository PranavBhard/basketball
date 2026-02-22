"""
Models Repository - Data access for model configuration collections.

Handles:
- model_config_nba: Win/loss classifier model configurations
- model_config_points_nba: Points regression model configurations

Note: For business logic (validation, creation), use ModelConfigManager.
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

    def find_trained_non_ensemble(self) -> List[Dict]:
        """Find all trained models that are not ensembles."""
        return self.find(
            {'trained_at': {'$exists': True}, 'ensemble_models': {'$exists': False}},
            sort=[('trained_at', -1)]
        )

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

    def set_selected_by_id(self, config_id: str) -> bool:
        """Set a config as selected by its ObjectId."""
        self.update_many({}, {'$set': {'selected': False}})
        result = self.update_one(
            {'_id': ObjectId(config_id)},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def delete_config(self, config_id: str) -> bool:
        """Delete a config by _id."""
        result = self.delete_one({'_id': ObjectId(config_id)})
        return result.deleted_count > 0

    # --- Utility Methods ---

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

    def set_selected_by_id(self, config_id: str) -> bool:
        """Set a config as selected by its ObjectId."""
        self.update_many({}, {'$set': {'selected': False}})
        result = self.update_one(
            {'_id': ObjectId(config_id)},
            {'$set': {'selected': True}}
        )
        return result.modified_count > 0

    def delete_config(self, config_id: str) -> bool:
        """Delete a config by _id."""
        result = self.delete_one({'_id': ObjectId(config_id)})
        return result.deleted_count > 0

    # --- Utility Methods ---

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
