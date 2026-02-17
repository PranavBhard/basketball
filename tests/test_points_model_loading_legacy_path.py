#!/usr/bin/env python3
"""
Regression test: PredictionService should load selected points models that use legacy `model_path`.

Historically, model_config_points_nba stored:
- model_path: "<artifacts_dir>/<model_name>.pkl"

But PredictionService._load_points_model() originally only supported:
- model_artifact_path / scaler_artifact_path / features_path

This test creates a minimal synthetic points model + scaler + feature names using the
same naming conventions as PointsRegressionTrainer.save_model()/load_model(), then
verifies PredictionService can load it when only `model_path` is provided.
"""

import os
import sys
import json
import pickle
import tempfile

import numpy as np


# Add parent of nba_app to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


class DummyScaler:
    """Minimal scaler with sklearn-like API."""

    def __init__(self, n_features_in_: int):
        self.n_features_in_ = n_features_in_

    def transform(self, X):
        # No-op scaling
        return np.asarray(X)


class DummyRegressor:
    """Minimal regressor with sklearn-like API."""

    def __init__(self, value: float):
        self._value = float(value)

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._value] * X.shape[0], dtype=float)


def main() -> int:
    from bball_app.core.services.prediction import PredictionService

    with tempfile.TemporaryDirectory() as td:
        model_name = "unit_test_points_model"
        model_path = os.path.join(td, f"{model_name}.pkl")
        scaler_path = os.path.join(td, f"{model_name}_scaler.pkl")
        features_path = os.path.join(td, f"{model_name}_features.json")

        # Create a minimal "home_away" model structure expected by PointsRegressionTrainer.predict()
        model_obj = {"home": DummyRegressor(112.3), "away": DummyRegressor(107.9)}
        scaler_obj = DummyScaler(n_features_in_=2)
        feature_names = ["f1", "f2"]

        with open(model_path, "wb") as f:
            pickle.dump(model_obj, f)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler_obj, f)
        with open(features_path, "w") as f:
            json.dump(feature_names, f)

        # IMPORTANT: Don't call PredictionService() here. Its __init__ wires up Mongo
        # repositories, which requires a valid mongo_conn_str in this environment.
        # We only need _load_points_model(), which doesn't depend on the repos.
        service = PredictionService.__new__(PredictionService)
        # Any non-None sentinel prevents PointsRegressionTrainer from trying to
        # auto-connect to Mongo in its __init__.
        service.db = object()
        service._points_model_cache = {}

        trainer = service._load_points_model({"model_path": model_path})

        if trainer is None:
            print("FAIL: _load_points_model returned None for legacy model_path config")
            return 1

        ok = (
            hasattr(trainer, "model")
            and isinstance(trainer.model, dict)
            and "home" in trainer.model
            and "away" in trainer.model
            and hasattr(trainer, "scaler")
            and hasattr(trainer, "feature_names")
            and trainer.feature_names == feature_names
        )

        if not ok:
            print("FAIL: trainer loaded but missing expected attributes/structure")
            print(f"  model type: {type(getattr(trainer, 'model', None))}")
            print(f"  feature_names: {getattr(trainer, 'feature_names', None)}")
            return 1

        print("PASS: PredictionService loaded legacy points model_path config successfully")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

