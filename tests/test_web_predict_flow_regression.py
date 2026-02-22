#!/usr/bin/env python3
"""Regression test for web /api/predict wiring issues.

This test is designed to fail fast if the web prediction flow regresses into
common runtime wiring errors (e.g., NameError/AttributeError from missing
module globals, incorrect FeatureManager API usage, etc.).

It uses lightweight fakes to avoid requiring a real MongoDB instance or trained
model artifacts.

Usage:
    python tests/test_web_predict_flow_regression.py
"""

import sys
import os
import tempfile
import importlib
from types import SimpleNamespace
from unittest import mock


# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class _UpdateResult:
    def __init__(self, matched_count: int, modified_count: int, upserted_id=None):
        self.matched_count = matched_count
        self.modified_count = modified_count
        self.upserted_id = upserted_id


class _Collection:
    def __init__(self, initial=None, keyed_by: str = None):
        self._docs = []
        self._keyed_by = keyed_by
        if initial:
            if isinstance(initial, list):
                self._docs.extend(initial)
            else:
                self._docs.append(initial)

    def _match(self, doc, query: dict) -> bool:
        for k, v in query.items():
            if doc.get(k) != v:
                return False
        return True

    def find_one(self, query: dict, projection=None):
        for doc in self._docs:
            if self._match(doc, query):
                return doc
        return None

    def find(self, query: dict = None, projection=None):
        if query is None:
            results = list(self._docs)
        else:
            results = [d for d in self._docs if self._match(d, query)]
        return _Cursor(results)

    def update_one(self, query: dict, update_doc: dict, upsert: bool = False):
        doc = self.find_one(query)
        if doc is None:
            if not upsert:
                return _UpdateResult(0, 0, None)
            doc = {}
            self._docs.append(doc)
            matched = 0
            upserted_id = query.get(self._keyed_by) if self._keyed_by else "upserted"
        else:
            matched = 1
            upserted_id = None

        set_doc = update_doc.get('$set', {})
        for k, v in set_doc.items():
            # For this regression test we don't need to interpret dot-notation.
            # We only need the saved key to exist in the retrieved doc.
            doc[k] = v

        modified = 1
        return _UpdateResult(matched, modified, upserted_id)


class _Cursor(list):
    """Minimal MongoDB cursor that supports .sort() chaining."""

    def sort(self, key_or_list, direction=None):
        return self


class _FakeDB:
    def __init__(self, selected_classifier_config: dict):
        self.model_configs = _Collection(initial=[], keyed_by="_id")
        self.model_config_nba = _Collection(initial=[selected_classifier_config], keyed_by="_id")
        # nba_model_config is the collection name used by ClassifierConfigRepository
        self.nba_model_config = self.model_config_nba
        self.model_config_points_nba = _Collection(initial=[], keyed_by="_id")
        self.stats_nba = _Collection(initial=[], keyed_by="game_id")
        self._collections = {}

    def __getitem__(self, name):
        """Support db[collection_name] access for repositories."""
        if hasattr(self, name):
            return getattr(self, name)
        return self._collections.setdefault(name, _Collection(initial=[], keyed_by="_id"))


class _FakeMongo:
    def __init__(self, *args, **kwargs):
        self.db = kwargs.get("db")


class _FakeBballModel:
    def __init__(
        self,
        classifier_features,
        points_features=None,
        include_elo=True,
        use_exponential_weighting=False,
        include_era_normalization=False,
        include_per_features=True,
        preload_data=False,
        **kwargs,
    ):
        self.classifier_features = classifier_features
        self.points_features = points_features or []
        self.include_elo = include_elo
        self.use_exponential_weighting = use_exponential_weighting
        self.include_era_normalization = include_era_normalization
        self.include_per_features = include_per_features
        self.preload_data = preload_data

        self.classifier_model = None
        self.scaler = None
        self.feature_names = []

        # Keep these attributes to satisfy web/app.py optional access.
        self.per_calculator = None
        self.recency_decay_k = 15.0
        self.stat_handler = None

    def load_cached_model(self, no_per: bool = False):
        # For this regression test, we do not exercise the cached model path.
        raise RuntimeError("cached model not available in regression test")

    def predict_with_player_config(
        self,
        home_team,
        away_team,
        season,
        game_date_str,
        player_filters,
        use_calibrated: bool = False,
        additional_features=None,
    ):
        # Minimal prediction payload expected by /api/predict.
        return {
            'predicted_winner': home_team,
            'home_win_prob': 55.0,
            'odds': '-122',
            'features_dict': additional_features or {},
        }


def test_web_predict_flow_regression(game_id: str = "401810422") -> bool:
    # Create a real file on disk so web/app.py's training_csv existence checks pass.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        training_csv_path = f.name
        f.write("Year,Month,Day,Home,Away,HomeWon\n")

    selected_classifier_config = {
        '_id': 'fake_selected',
        'selected': True,
        'training_csv': training_csv_path,
        'model_type': 'LogisticRegression',
        'ensemble': False,
    }

    fake_db = _FakeDB(selected_classifier_config=selected_classifier_config)

    # Patch Mongo before importing the web module, since it instantiates Mongo() at import time.
    with mock.patch('bball.mongo.Mongo', autospec=True) as MongoPatched:
        MongoPatched.return_value = SimpleNamespace(db=fake_db)

        # Import web.app fresh (in case a prior import exists in the interpreter)
        if 'web.app' in sys.modules:
            del sys.modules['web.app']
        web_app = importlib.import_module('web.app')

    # Hard-override heavyweight pieces to keep the test deterministic.
    web_app.BballModel = _FakeBballModel

    def _fake_create_model(config, use_artifacts: bool = True):
        import numpy as np
        dummy_model = SimpleNamespace(
            predict_proba=lambda X: np.array([[0.45, 0.55]]),
            predict=lambda X: np.array([1]),
        )
        dummy_scaler = SimpleNamespace(
            transform=lambda X: X,
        )
        feature_names = ['points|season|avg|diff']
        return dummy_model, dummy_scaler, feature_names

    web_app.ArtifactLoader.create_model = staticmethod(_fake_create_model)

    def _fake_build_player_lists_for_prediction(home_team, away_team, season, game_id, db):
        return {
            home_team: {'playing': [], 'starters': []},
            away_team: {'playing': [], 'starters': []},
        }

    web_app.build_player_lists_for_prediction = _fake_build_player_lists_for_prediction

    # Sanity-check wiring globals exist (these were the repeated NameError sources).
    assert hasattr(web_app, 'config_manager')
    assert hasattr(web_app, '_nba_model')
    assert hasattr(web_app, '_nba_model_config_version')

    # Exercise get_bball_model() directly.
    model = web_app.get_bball_model()
    assert model is not None

    # Exercise the Flask endpoint flow.
    client = web_app.app.test_client()
    payload = {
        'game_id': game_id,
        'game_date': '2026-01-13',
        'home_team': 'NO',
        'away_team': 'DEN',
        'player_config': {},
    }
    resp = client.post('/api/predict', json=payload)

    assert resp.status_code == 200, resp.get_data(as_text=True)
    data = resp.get_json()
    assert data and data.get('success') is True, data

    # Verify prediction persisted to the (fake) DB's predictions collection.
    saved = web_app.db['nba_model_predictions'].find_one({'game_id': game_id})
    assert saved is not None

    return True


def main() -> int:
    try:
        ok = test_web_predict_flow_regression()
        print("✅ test_web_predict_flow_regression: PASSED" if ok else "❌ test_web_predict_flow_regression: FAILED")
        return 0 if ok else 1
    except Exception as e:
        import traceback
        print("❌ test_web_predict_flow_regression: FAILED")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
