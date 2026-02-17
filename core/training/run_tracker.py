"""
Run Tracker - MongoDB-based experiment run tracking
"""

import sys
import os
from datetime import datetime
from typing import Dict, List, Optional
import uuid

# Add parent directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from bball_app.core.mongo import Mongo
from bball_app.core.data import ExperimentRunsRepository


class RunTracker:
    """Manages experiment run tracking in MongoDB"""

    def __init__(self, db=None, league=None):
        """
        Initialize RunTracker.

        Args:
            db: MongoDB database instance (optional, will create if not provided)
            league: LeagueConfig instance for league-specific collections
        """
        if db is None:
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league

        # Initialize repository with league awareness
        self._repo = ExperimentRunsRepository(self.db, league=league)
    
    def create_run(
        self,
        config: Dict,
        dataset_id: Optional[str],
        model_type: str,
        session_id: str,
        baseline: bool = False
    ) -> str:
        """
        Create a new experiment run.
        
        Args:
            config: Full experiment configuration dict
            dataset_id: ID of the dataset used (optional, can be None for stacking)
            model_type: Type of model (e.g., 'LogisticRegression', 'Stacked')
            session_id: Chat session ID
            baseline: Whether this is the baseline run
            
        Returns:
            run_id: Unique identifier for the run
        """
        run_id = str(uuid.uuid4())

        # If this is set as baseline, unset all other baselines for this session
        if baseline:
            self._repo.clear_baselines(session_id)

        run_doc = {
            'run_id': run_id,
            'created_at': datetime.utcnow(),
            'config': config,
            'dataset_id': dataset_id,
            'model_type': model_type,
            'metrics': {},
            'diagnostics': {},
            'artifacts': {},
            'baseline': baseline,
            'session_id': session_id,
            'status': 'created'  # created, running, completed, failed
        }

        self._repo.insert_one(run_doc)
        return run_id
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """
        Get a run by ID.

        Args:
            run_id: Run identifier

        Returns:
            Run document or None if not found
        """
        run = self._repo.find_by_run_id(run_id)
        if run:
            # Convert ObjectId to string for JSON serialization
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()
        return run
    
    def update_run(
        self,
        run_id: str,
        metrics: Optional[Dict] = None,
        diagnostics: Optional[Dict] = None,
        artifacts: Optional[Dict] = None,
        status: Optional[str] = None
    ) -> bool:
        """
        Update a run with results.
        
        Args:
            run_id: Run identifier
            metrics: Metrics dict (accuracy, log_loss, brier, auc, etc.)
            diagnostics: Diagnostics dict (feature_importances, calibration_curve, etc.)
            artifacts: Artifacts dict (model_path, dataset_path, report_path)
            status: Status string (created, running, completed, failed)
            
        Returns:
            True if update was successful
        """
        update_dict = {}
        
        if metrics is not None:
            update_dict['metrics'] = metrics
        
        if diagnostics is not None:
            update_dict['diagnostics'] = diagnostics
        
        if artifacts is not None:
            update_dict['artifacts'] = artifacts
        
        if status is not None:
            update_dict['status'] = status
        
        if not update_dict:
            return False

        return self._repo.update_run(run_id, update_dict)
    
    def set_baseline(self, run_id: str, session_id: str) -> bool:
        """
        Set a run as the baseline for a session.

        Args:
            run_id: Run identifier
            session_id: Chat session ID

        Returns:
            True if successful
        """
        return self._repo.set_baseline(run_id, session_id)
    
    def get_baseline(self, session_id: str) -> Optional[Dict]:
        """
        Get the baseline run for a session.

        Args:
            session_id: Chat session ID

        Returns:
            Baseline run document or None
        """
        run = self._repo.find_baseline(session_id)

        if run:
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()

        return run
    
    def list_runs(
        self,
        session_id: Optional[str] = None,
        model_type: Optional[str] = None,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        List runs with optional filters.

        Args:
            session_id: Filter by session ID
            model_type: Filter by model type
            date_from: Filter runs created after this date
            date_to: Filter runs created before this date
            limit: Maximum number of runs to return

        Returns:
            List of run documents (summaries)
        """
        runs = self._repo.find_by_date_range(
            date_from=date_from,
            date_to=date_to,
            session_id=session_id,
            model_type=model_type,
            limit=limit
        )

        # Convert ObjectIds and datetimes for JSON serialization
        for run in runs:
            run['_id'] = str(run['_id'])
            if 'created_at' in run and isinstance(run['created_at'], datetime):
                run['created_at'] = run['created_at'].isoformat()

        return runs
    
    def get_run_count(self, session_id: str) -> int:
        """
        Get the number of runs for a session.

        Args:
            session_id: Chat session ID

        Returns:
            Number of runs
        """
        return self._repo.count_by_session(session_id)

