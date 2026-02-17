#!/usr/bin/env python3
"""
Validate injury feature generation via the SAME code path as the Master Training UI "Add Columns".

This script:
1) Samples games with populated injured_players from stats_nba
2) Builds a temporary master training CSV with those rows
3) Runs cli/populate_master_training_cols.populate_columns() with inj_* columns (overwrite)
4) Verifies that each target column has at least one non-zero value

If any target column remains all zeros/NaN in the sample, the script fails and
prints a diagnostic hint to increase sample size.

Usage:
  source venv/bin/activate
  PYTHONPATH=/Users/pranav/Documents/NBA \
    python tests/test_injury_features_add_columns_flow.py

Optional environment variables:
  INJ_SAMPLE_SIZE=2000   # max rows to include in temp CSV (default 2000)
  INJ_GAME_LIMIT=10000   # max game_ids to pull from stats_nba (default 10000)
  INJ_USE_JOB_ID=1       # if set, create a jobs_nba entry so the code path
                         # matches the master-training UI (chunked shared context)
"""

import os
import sys
import tempfile
from typing import List, Dict

import pandas as pd

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _load_injured_game_ids(db, limit: int) -> List[str]:
    query = {
        '$or': [
            {'homeTeam.injured_players': {'$exists': True, '$ne': []}},
            {'awayTeam.injured_players': {'$exists': True, '$ne': []}}
        ]
    }
    cursor = db.stats_nba.find(query, {'game_id': 1}).limit(limit)
    return [str(doc.get('game_id')) for doc in cursor if doc.get('game_id')]


def _build_sample_csv(master_csv_path: str, game_ids: List[str], sample_size: int) -> str:
    """Create a temp master CSV containing only rows with game_id in game_ids."""
    if not game_ids:
        raise RuntimeError("No game_ids found with injured_players.")

    game_id_set = set(game_ids)
    rows = []
    columns = None

    for chunk in pd.read_csv(master_csv_path, chunksize=200_000):
        if 'game_id' not in chunk.columns:
            raise RuntimeError("master CSV missing game_id column.")
        columns = list(chunk.columns)
        filtered = chunk[chunk['game_id'].astype(str).isin(game_id_set)]
        if not filtered.empty:
            rows.append(filtered)
            if sum(len(r) for r in rows) >= sample_size:
                break

    if not rows:
        raise RuntimeError("No master training rows matched game_ids with injuries.")

    df = pd.concat(rows, ignore_index=True)
    if len(df) > sample_size:
        df = df.iloc[:sample_size].copy()

    temp_dir = tempfile.mkdtemp(prefix="inj_feature_sample_")
    temp_csv = os.path.join(temp_dir, "MASTER_TRAINING_inj_sample.csv")
    df.to_csv(temp_csv, index=False)
    return temp_csv


def _scan_nonzero(df: pd.DataFrame, targets: List[str]) -> Dict[str, int]:
    counts = {}
    for col in targets:
        if col not in df.columns:
            counts[col] = 0
            continue
        s = df[col].fillna(0).astype(float)
        counts[col] = int((s != 0.0).sum())
    return counts


def main() -> int:
    from bball_app.core.mongo import Mongo
    from bball_app.core.services.training_data import MASTER_TRAINING_PATH
    from bball_app.cli.populate_master_training_cols import populate_columns

    sample_size = int(os.environ.get("INJ_SAMPLE_SIZE", "2000"))
    game_limit = int(os.environ.get("INJ_GAME_LIMIT", "10000"))
    use_job_id = os.environ.get("INJ_USE_JOB_ID", "").strip() in ("1", "true", "True")

    # Connect to Mongo
    try:
        mongo = Mongo()
        db = mongo.db
    except Exception as e:
        print(f"SKIP: Mongo not configured or not reachable: {e}")
        return 0

    print("Finding games with injured_players...")
    game_ids = _load_injured_game_ids(db, game_limit)
    print(f"  Found {len(game_ids)} game_ids with injuries (limit {game_limit})")

    print("Building sample CSV from master training...")
    sample_csv = _build_sample_csv(MASTER_TRAINING_PATH, game_ids, sample_size)
    print(f"  Sample CSV: {sample_csv}")

    targets = [
        'inj_min_lost|none|raw|home',
        'inj_min_lost|none|raw|away',
        'inj_min_lost|none|raw|diff',
        'inj_severity|none|raw|home',
        'inj_severity|none|raw|away',
        'inj_severity|none|raw|diff',
        'inj_severity|season|raw|home',
        'inj_severity|season|raw|away',
        'inj_severity|season|raw|diff',
        'inj_per_share|none|top1_avg|home',
        'inj_per_share|none|top1_avg|away',
        'inj_per_share|none|top1_avg|diff',
    ]

    # Run the SAME code path as the UI add-columns workflow
    print("Running populate_columns() with overwrite=True...")
    job_id = None
    if use_job_id:
        job_doc = {
            'config_id': None,
            'type': 'add_features',
            'progress': 0,
            'status': 'running',
            'error': None,
            'message': 'Starting test column addition...',
            'metadata': {
                'feature_names': targets,
                'test_mode': True
            },
            'created_at': pd.Timestamp.utcnow().to_pydatetime(),
            'updated_at': pd.Timestamp.utcnow().to_pydatetime()
        }
        result = db.jobs_nba.insert_one(job_doc)
        job_id = str(result.inserted_id)

    populate_columns(
        master_csv_path=sample_csv,
        columns=targets,
        feature_substrings=None,
        match_mode="OR",
        overwrite=True,
        backup=False,
        job_id=job_id,
        chunk_size=500,
    )

    print("Validating non-zero values...")
    df_out = pd.read_csv(sample_csv)
    counts = _scan_nonzero(df_out, targets)

    failed = [col for col, cnt in counts.items() if cnt == 0]
    if failed:
        print("\nFAILED: These columns are still all 0/NaN in the sample:")
        for col in failed:
            print(f"  - {col}")
        print("\nHints:")
        print("  • Increase INJ_SAMPLE_SIZE or INJ_GAME_LIMIT to widen the sample.")
        print("  • Verify injured_players are present for these game_ids.")
        print("  • Verify PERCalculator is preloading player stats (needed for top1_avg).")
        return 1

    print("PASS: All target columns have non-zero values in the sample.")
    for col, cnt in counts.items():
        print(f"  {col}: {cnt} non-zero rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
