#!/usr/bin/env python3
"""Regression test for ArtifactLoader._train_from_data numeric coercion.

This reproduces the failure seen in the web UI when ArtifactLoader falls back to
training from a CSV that includes a non-numeric feature column.

Historically this crashed with:
    TypeError: ufunc 'isnan' not supported for the input types

Usage:
    python tests/test_model_factory_numeric_coercion.py
"""

import sys
import os
import tempfile


# Add parent of nba_app to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))  # nba_app/tests/
nba_app_dir = os.path.dirname(script_dir)  # nba_app/
project_root = os.path.dirname(nba_app_dir)  # parent of nba_app/
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_model_factory_train_from_data_coerces_to_numeric() -> bool:
    from bball_app.core.models.artifact_loader import ArtifactLoader

    # Create a CSV with one non-numeric feature column that previously caused
    # np.isnan to crash due to object dtype.
    csv_content = """Year,Month,Day,Home,Away,HomeWon,game_id,feat_numeric,feat_bad
2026,1,13,NO,DEN,1,401810422,1.5,abc
2026,1,13,NO,DEN,0,401810423,2.5,def
"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(csv_content)
        path = f.name

    config = {
        'model_type': 'LogisticRegression',
        'best_c_value': 0.1,
        'training_csv': path,
    }

    model, scaler, feature_names = ArtifactLoader.create_model(config, use_artifacts=False)

    assert model is not None
    assert scaler is not None
    assert isinstance(feature_names, list)
    assert 'feat_numeric' in feature_names
    assert 'feat_bad' not in feature_names  # dropped as non-numeric

    return True


def main() -> int:
    try:
        ok = test_model_factory_train_from_data_coerces_to_numeric()
        print("✅ test_model_factory_train_from_data_coerces_to_numeric: PASSED" if ok else "❌ test_model_factory_train_from_data_coerces_to_numeric: FAILED")
        return 0 if ok else 1
    except Exception as e:
        import traceback
        print("❌ test_model_factory_train_from_data_coerces_to_numeric: FAILED")
        print(f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
