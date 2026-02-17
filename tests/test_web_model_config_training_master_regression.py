#!/usr/bin/env python3
"""Regression test for model-config training (use_master=True).

This catches a class of failures where the background training job crashes due to
missing imports / undefined helper functions in web/app.py (e.g. NameError for
master-training extraction helpers).

This test does NOT train a real model; it stubs the expensive parts and focuses
on verifying that the use_master branch executes far enough to produce a
training CSV without raising.

Usage:
    python tests/test_web_model_config_training_master_regression.py
"""

import sys
import os
import tempfile
import importlib
from types import SimpleNamespace
from unittest import mock


# Add parent of nba_app to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))  # nba_app/tests/
nba_app_dir = os.path.dirname(script_dir)  # nba_app/
project_root = os.path.dirname(nba_app_dir)  # parent of nba_app/
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class _Collection:
    def __init__(self):
        self._docs = []

    def insert_one(self, doc):
        self._docs.append(doc)
        return SimpleNamespace(inserted_id='job1')

    def update_one(self, *args, **kwargs):
        return None

    def update_many(self, *args, **kwargs):
        return None

    def find_one(self, *args, **kwargs):
        return None


class _FakeDB:
    def __init__(self):
        self.jobs_nba = _Collection()
        self.model_config_nba = _Collection()


def test_use_master_training_branch_no_nameerror() -> bool:
    fake_db = _FakeDB()

    # Patch Mongo before importing web.app because web.app creates Mongo() at import time.
    with mock.patch('bball_app.core.mongo.Mongo', autospec=True) as MongoPatched:
        MongoPatched.return_value = SimpleNamespace(db=fake_db)

        if 'bball_app.web.app' in sys.modules:
            del sys.modules['bball_app.web.app']
        web_app = importlib.import_module('bball_app.web.app')

    # Stub job progress update to avoid needing real job rows.
    web_app.update_job_progress = lambda *args, **kwargs: None

    # Ensure the feature-flag helper exists (this previously regressed via a wrong function name)
    assert hasattr(web_app, 'infer_feature_flags_from_features')

    # Create a temp master CSV and a fake extracted CSV output.
    with tempfile.TemporaryDirectory() as td:
        master_path = os.path.join(td, 'MASTER_TRAINING.csv')
        with open(master_path, 'w') as f:
            f.write('Year,Month,Day,Home,Away,HomeWon,game_id,feat1\n')
            f.write('2026,1,13,NO,DEN,1,401810422,0.5\n')

        def _fake_generate_master_training_data():
            return (master_path, ['feat1'], 1)

        def _fake_check_master_needs_regeneration(db, requested_features):
            return (False, [])

        def _fake_extract_features_from_master(master_path_in, requested_features=None, output_path=None):
            # Just copy master to output path
            if output_path is None:
                output_path = os.path.join(td, 'extracted.csv')
            with open(master_path_in, 'r') as src, open(output_path, 'w') as dst:
                dst.write(src.read())
            return output_path

        # Patch the imported-in-function module symbols via sys.modules trick:
        # We patch the actual core.services.training_data module functions/constants.
        import bball_app.core.services.training_data as mtd
        mtd.MASTER_TRAINING_PATH = master_path
        mtd.generate_master_training_data = _fake_generate_master_training_data
        mtd.check_master_needs_regeneration = _fake_check_master_needs_regeneration
        mtd.extract_features_from_master = _fake_extract_features_from_master

        # web/app.py now uses get_master_training_path() (reads league config)
        # instead of MASTER_TRAINING_PATH directly — patch it to return the temp path.
        web_app.get_master_training_path = lambda: master_path

        # Now call run_training_job with use_master=True. We expect it to proceed
        # past the master extraction step (no NameError).
        try:
            web_app.run_training_job(
                job_id='job1',
                config_id='cfg1',
                model_types=['LogisticRegression'],
                c_values=[0.1],
                feature_sets=[],
                features=['feat1'],
                use_time_calibration=False,
                calibration_method=None,
                begin_year=None,
                calibration_years=None,
                evaluation_year=None,
                use_master=True,
                include_injuries=False,
                recency_decay_k=None,
                min_games_played=0,
                point_model_id=None,
            )
        except NameError as e:
            raise AssertionError(f"Unexpected NameError in use_master training flow: {e}")
        except Exception:
            # We only care about NameError regressions here.
            pass

    return True


def main() -> int:
    try:
        ok = test_use_master_training_branch_no_nameerror()
        print("✅ test_use_master_training_branch_no_nameerror: PASSED" if ok else "❌ test_use_master_training_branch_no_nameerror: FAILED")
        return 0 if ok else 1
    except Exception as e:
        import traceback
        print("❌ test_use_master_training_branch_no_nameerror: FAILED")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
