#!/usr/bin/env python
"""
Shared Context Experiment

Tests whether adding "anchor" features (margin, wins) to base models improves
the ensemble by giving the meta-learner agreement/disagreement signals.

Experiments:
  A: Baseline (current ensemble)
  B: +margin to player_talent + injuries (targeted)
  C: +margin to matchup, player_talent, injuries, h2h (broad)
  D: +margin+wins to player_talent + injuries (dual anchor, targeted)

Usage:
    python cli/scripts/shared_context_experiment.py
"""

import time
import sys

# ── Current base model IDs ──
BASE1_ID = "699a71cdea5f72bb698be47e"   # LR elo/strength  (has margin already)
BASE2_ID = "699a8054660be9ba66746327"   # LR matchup       (19 features)
BASE3_ID = "699a83751fc5d0b09f31490c"   # GB recent form   (has margin in recent window)
BASE4_ID = "699a86871fc5d0b09f314917"   # GB player_talent  (17 features)
BASE5_ID = "6998ef07660be9ba6668d9f8"   # GB injuries       (8 features)
BASE6_ID = "699a8bae1fc5d0b09f31492b"   # LR h2h            (6 features)

# ── Anchor features ──
MARGIN = "margin|season|avg|diff"
WINS = "wins|season|avg|diff"

# ── Ensemble settings (match current) ──
META_MODEL = "LR"
META_C = 0.1
STACKING_MODE = "informed"
USE_LOGIT = True
META_FEATURES = [
    "b2b|none|raw|diff",
    "travel|days_5|sum|diff",
    "games_played|days_7|raw|diff",
    "road_games|days_10|raw|diff",
]

# ── Time config (must match all base models) ──
BEGIN_YEAR = 2010
CAL_YEARS = [2021, 2022, 2023]
EVAL_YEAR = 2024
MIN_GAMES = 20


def get_base_features(service, config_id):
    """Get the feature list from an existing base model config."""
    config = service.resolve_model(config_id)
    return config.get('features', [])


def train_variant(service, original_id, extra_features, name_prefix):
    """Train a new base model = original features + extra anchors."""
    config = service.resolve_model(original_id)
    model_type = config.get('model_type', 'LogisticRegression')
    original_features = config.get('features', [])

    # Add anchors (skip duplicates)
    features = list(original_features)
    for f in extra_features:
        if f not in features:
            features.append(f)

    # Use short alias for model type
    mt_alias = 'LR' if 'Logistic' in model_type else 'GB' if 'Gradient' in model_type else model_type

    print(f"  Training {name_prefix} ({mt_alias}, {len(features)} features, +{len(features)-len(original_features)} anchors)")

    result = service.train_model_grid(
        model_types=[mt_alias],
        c_values=[0.01, 0.1, 1.0],
        features=features,
        begin_year=BEGIN_YEAR,
        calibration_years=CAL_YEARS,
        evaluation_year=EVAL_YEAR,
        min_games_played=MIN_GAMES,
        calibration_method='sigmoid',
        name_prefix=name_prefix,
        use_master=True,
    )

    # Get best config_id
    for mt, mt_data in result.get('model_type_results', {}).items():
        best = mt_data.get('best')
        if best:
            acc = best.get('accuracy', 0)
            print(f"    -> {best['config_id']}  acc={acc:.2f}%")
            return best['config_id'], acc

    raise RuntimeError(f"Training failed for {name_prefix}")


def train_ensemble_variant(service, base_ids, name):
    """Train an ensemble with given base model IDs."""
    print(f"\n  Training ensemble: {name}")
    print(f"    Base models: {base_ids}")

    result = service.train_ensemble(
        meta_model_type=META_MODEL,
        base_model_names_or_ids=base_ids,
        meta_c_value=META_C,
        extra_features=META_FEATURES,
        stacking_mode=STACKING_MODE,
        use_disagree=False,
        use_conf=False,
        use_logit=USE_LOGIT,
        name=name,
    )

    metrics = result.get('metrics', {})
    acc = metrics.get('accuracy_mean', 0)
    ll = metrics.get('log_loss_mean', 0)
    brier = metrics.get('brier_mean', 0)

    print(f"    -> acc={acc:.2f}%  log_loss={ll:.4f}  brier={brier:.4f}")
    print(f"    -> config_id={result.get('config_id')}")

    return {
        'config_id': result.get('config_id'),
        'accuracy': acc,
        'log_loss': ll,
        'brier': brier,
        'diagnostics': result.get('diagnostics', {}),
    }


def main():
    from bball.league_config import load_league_config
    from bball.services.training_service import TrainingService

    league = load_league_config('nba')
    service = TrainingService(league=league)

    results = {}

    # ================================================================
    # Experiment A: Baseline (retrain current ensemble for fair comparison)
    # ================================================================
    print("=" * 70)
    print("EXPERIMENT A: Baseline (current base models, retrained ensemble)")
    print("=" * 70)
    t0 = time.time()
    r = train_ensemble_variant(
        service,
        [BASE1_ID, BASE2_ID, BASE3_ID, BASE4_ID, BASE5_ID, BASE6_ID],
        "EXP-A-baseline",
    )
    results['A: baseline'] = r
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ================================================================
    # Experiment B: +margin to player_talent + injuries (targeted)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT B: +margin to player_talent + injuries")
    print("=" * 70)
    t0 = time.time()

    b4_margin_id, _ = train_variant(service, BASE4_ID, [MARGIN], "SC-players+margin")
    b5_margin_id, _ = train_variant(service, BASE5_ID, [MARGIN], "SC-injuries+margin")

    r = train_ensemble_variant(
        service,
        [BASE1_ID, BASE2_ID, BASE3_ID, b4_margin_id, b5_margin_id, BASE6_ID],
        "EXP-B-margin-targeted",
    )
    results['B: +margin targeted'] = r
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ================================================================
    # Experiment C: +margin to matchup, players, injuries, h2h (broad)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT C: +margin to matchup, players, injuries, h2h")
    print("=" * 70)
    t0 = time.time()

    b2_margin_id, _ = train_variant(service, BASE2_ID, [MARGIN], "SC-matchup+margin")
    b6_margin_id, _ = train_variant(service, BASE6_ID, [MARGIN], "SC-h2h+margin")
    # Reuse b4_margin_id, b5_margin_id from experiment B

    r = train_ensemble_variant(
        service,
        [BASE1_ID, b2_margin_id, BASE3_ID, b4_margin_id, b5_margin_id, b6_margin_id],
        "EXP-C-margin-broad",
    )
    results['C: +margin broad'] = r
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ================================================================
    # Experiment D: +margin+wins to player_talent + injuries (dual anchor)
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT D: +margin+wins to player_talent + injuries")
    print("=" * 70)
    t0 = time.time()

    b4_dual_id, _ = train_variant(service, BASE4_ID, [MARGIN, WINS], "SC-players+margin+wins")
    b5_dual_id, _ = train_variant(service, BASE5_ID, [MARGIN, WINS], "SC-injuries+margin+wins")

    r = train_ensemble_variant(
        service,
        [BASE1_ID, BASE2_ID, BASE3_ID, b4_dual_id, b5_dual_id, BASE6_ID],
        "EXP-D-dual-targeted",
    )
    results['D: +margin+wins targeted'] = r
    print(f"  elapsed: {time.time()-t0:.1f}s")

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Acc%':>7}  {'LogLoss':>8}  {'Brier':>8}  {'Config ID'}")
    print("-" * 90)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    for name, r in sorted_results:
        print(f"{name:<30} {r['accuracy']:>7.2f}  {r['log_loss']:>8.4f}  {r['brier']:>8.4f}  {r['config_id']}")

    # Delta from baseline
    baseline_acc = results.get('A: baseline', {}).get('accuracy', 0)
    print()
    for name, r in sorted_results:
        if name != 'A: baseline':
            delta = r['accuracy'] - baseline_acc
            print(f"  {name}: {'+' if delta >= 0 else ''}{delta:.2f}pp vs baseline")

    # Print meta-feature importances for best
    best_name, best_r = sorted_results[0]
    diag = best_r.get('diagnostics', {})
    meta_fi = diag.get('meta_feature_importances', {})
    if meta_fi:
        print(f"\nMeta-feature importances for best ({best_name}):")
        for fname, score in sorted(meta_fi.items(), key=lambda x: abs(x[1]), reverse=True):
            print(f"  {fname}: {score:.4f}")


if __name__ == "__main__":
    main()
