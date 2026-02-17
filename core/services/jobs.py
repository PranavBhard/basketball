"""
Job infrastructure for background task management.

This module provides job tracking functions that can be used from both
Flask request handlers and background threads. Uses _raw_db for thread-safe
MongoDB access without Flask context.

Usage:
    from bball_app.core.services.jobs import create_job, update_job_progress, complete_job, fail_job

    job_id = create_job('predict_all', league=league, metadata={'date': '2026-01-26'})
    update_job_progress(job_id, 50, 'Processing...', league=league)
    complete_job(job_id, 'Completed 10 predictions', league=league)
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, TYPE_CHECKING

from bson import ObjectId

from bball_app.core.mongo import Mongo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig

# Thread-safe MongoDB connection (not Flask-request-bound)
_mongo = None


def _get_raw_db():
    """Get thread-safe MongoDB database instance."""
    global _mongo
    if _mongo is None:
        _mongo = Mongo()
    return _mongo.db


def _get_jobs_collection(league: Optional["LeagueConfig"] = None):
    """Get the jobs collection for the given league."""
    db = _get_raw_db()
    if league:
        collection_name = league.collections.get('jobs', 'jobs_nba')
    else:
        collection_name = 'jobs_nba'
    return db[collection_name]


def create_job(
    job_type: str,
    league: Optional["LeagueConfig"] = None,
    config_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Create a new job in MongoDB.

    Args:
        job_type: Type of job ('predict_all', 'train', etc.)
        league: Optional league config for multi-league support
        config_id: Optional model config ID this job is associated with
        metadata: Optional metadata dict (e.g., {'date': '2026-01-26'})

    Returns:
        Job ID as string
    """
    utc = timezone.utc
    now = datetime.now(utc)

    job_doc = {
        'type': job_type,
        'config_id': config_id,
        'status': 'running',
        'progress': 0,
        'message': 'Starting...',
        'error': None,
        'metadata': metadata or {},
        'created_at': now,
        'updated_at': now
    }

    jobs_collection = _get_jobs_collection(league)
    collection_name = jobs_collection.name
    result = jobs_collection.insert_one(job_doc)
    job_id = str(result.inserted_id)
    logger.info(f"[JOBS] create_job(type={job_type}) in {collection_name}: job_id={job_id}")
    return job_id


def update_job_progress(
    job_id: str,
    progress: int,
    message: Optional[str] = None,
    league: Optional["LeagueConfig"] = None
):
    """
    Update job progress and message.

    Args:
        job_id: Job ID
        progress: Progress percentage (0-100)
        message: Optional status message
        league: Optional league config for multi-league support

    Note: Uses thread-safe MongoDB connection for background thread compatibility.
    """
    utc = timezone.utc
    update_doc = {
        'progress': max(0, min(100, progress)),
        'updated_at': datetime.now(utc)
    }
    if message:
        update_doc['message'] = message

    try:
        jobs_collection = _get_jobs_collection(league)
        collection_name = jobs_collection.name
        result = jobs_collection.update_one(
            {'_id': ObjectId(job_id)},
            {'$set': update_doc}
        )
        logger.info(f"[JOBS] update_job_progress({job_id}, {progress}%, '{message}') in {collection_name}: matched={result.matched_count}, modified={result.modified_count}")
        if result.matched_count == 0:
            logger.error(f"[JOBS] Job {job_id} not found in {collection_name} - update failed!")
    except Exception as e:
        logger.error(f"[JOBS] Exception in update_job_progress({job_id}): {e}")


def complete_job(
    job_id: str,
    message: str = 'Job completed successfully',
    league: Optional["LeagueConfig"] = None
):
    """
    Mark job as completed.

    Args:
        job_id: Job ID
        message: Completion message
        league: Optional league config for multi-league support
    """
    utc = timezone.utc
    try:
        jobs_collection = _get_jobs_collection(league)
        collection_name = jobs_collection.name
        result = jobs_collection.update_one(
            {'_id': ObjectId(job_id)},
            {'$set': {
                'status': 'completed',
                'progress': 100,
                'message': message,
                'updated_at': datetime.now(utc)
            }}
        )
        logger.info(f"[JOBS] complete_job({job_id}) in {collection_name}: matched={result.matched_count}, modified={result.modified_count}")
        if result.matched_count == 0:
            logger.error(f"[JOBS] Job {job_id} not found in {collection_name} - complete failed!")
    except Exception as e:
        logger.error(f"[JOBS] Exception in complete_job({job_id}): {e}")


def fail_job(
    job_id: str,
    error: str,
    message: Optional[str] = None,
    league: Optional["LeagueConfig"] = None
):
    """
    Mark job as failed.

    Args:
        job_id: Job ID
        error: Error description
        message: Optional user-facing message
        league: Optional league config for multi-league support
    """
    utc = timezone.utc
    try:
        jobs_collection = _get_jobs_collection(league)
        collection_name = jobs_collection.name
        result = jobs_collection.update_one(
            {'_id': ObjectId(job_id)},
            {'$set': {
                'status': 'failed',
                'error': error,
                'message': message or f'Job failed: {error}',
                'updated_at': datetime.now(utc)
            }}
        )
        logger.error(f"[JOBS] fail_job({job_id}, '{error}') in {collection_name}: matched={result.matched_count}, modified={result.modified_count}")
        if result.matched_count == 0:
            logger.error(f"[JOBS] Job {job_id} not found in {collection_name} - fail update failed!")
    except Exception as e:
        logger.error(f"[JOBS] Exception in fail_job({job_id}): {e}")


def get_job(job_id: str, league: Optional["LeagueConfig"] = None) -> Optional[Dict]:
    """
    Get job by ID.

    Args:
        job_id: Job ID
        league: Optional league config for multi-league support

    Returns:
        Job document or None if not found
    """
    jobs_collection = _get_jobs_collection(league)
    job = jobs_collection.find_one({'_id': ObjectId(job_id)})
    if job:
        job['_id'] = str(job['_id'])
    return job
