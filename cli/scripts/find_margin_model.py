#!/usr/bin/env python
"""
Find Optimal NBA Margin Regression Model

Sweeps 5 feature-set variants × 3 model types (Ridge, ElasticNet, XGBoost)
= 15 experiments to find the best margin predictor.

Uses same time config as classifier ensemble:
  begin_year=2010, cal=[2021,2022,2023], eval=2024, min_games=20

Usage:
    source venv/bin/activate && python cli/scripts/find_margin_model.py
"""

import os
import sys
import time
import tempfile
import numpy as np

# ── Feature set building blocks ──

CORE = [
    "margin|season|avg|diff",
    "wins|season|avg|diff",
    "elo|season|raw|diff",
    "off_rtg|season|avg|diff",
    "def_rtg|season|avg|diff",
    "efg|season|avg|diff",
    "ts|season|avg|diff",
    "three_pct|season|avg|diff",
    "pace_interaction|season|harmonic_mean|none",
    "exp_points_matchup|season|derived|diff",
    "efg_net|season|avg|diff",
    "ts_net|season|avg|diff",
]

PLAYMAKING = [
    "assists|season|avg|diff",
    "turnovers|season|avg|diff",
    "three_made|season|avg|diff",
    "reb_total|season|avg|diff",
    "blocks|season|avg|diff",
    "steals|season|avg|diff",
    "off_rtg_net|season|avg|diff",
    "three_pct_net|season|avg|diff",
]

MU7_SEASON = [
    "mu_pace_delta|season|avg|diff",
    "mu_oreb_vs_dreb|season|avg|diff",
    "mu_to_vs_steals|season|avg|diff",
    "mu_three_exposure|season|avg|diff",
    "mu_ft_draw_vs_fouls|season|avg|diff",
    "mu_paint_vs_blocks|season|avg|diff",
    "mu_ast_vs_steals|season|avg|diff",
]

SOS = [
    "sos_opp_margin|season|avg|diff",
    "sos_opp_elo|season|avg|diff",
    "sos_opp_efg|season|avg|diff",
    "sos_opp_off_rtg|season|avg|diff",
    "sos_opp_def_rtg|season|avg|diff",
    "sos_opp_net_rtg|season|avg|diff",
    "sos_opp_pace|season|avg|diff",
    "sos_opp_win_pct|season|avg|diff",
]

FORM_GAMES12 = [
    "margin|blend:games_12:0.70/games_25:0.30|avg|diff",
    "efg|games_12|avg|diff",
    "off_rtg|games_12|avg|diff",
    "turnovers|games_12|avg|diff",
]

SCHEDULE = [
    "b2b|season|avg|diff",
    "rest|none|raw|diff",
    "travel|season|avg|diff",
    "road_games|days_10|raw|diff",
    "games_played|season|raw|diff",
]

DELTAS = [
    "margin|delta:games_12-season|avg|diff",
    "efg_net|delta:games_12-season|avg|diff",
    "off_rtg_net|delta:games_12-season|avg|diff",
    "turnovers|delta:games_12-season|avg|diff",
    "def_rtg|delta:games_12-season|avg|diff",
]

PLAYERS = [
    "player_team_per|season|weighted_MPG|diff",
    "player_team_per|season|avg|diff",
    "player_star_score|season|top1|diff",
    "player_star_score|season|top3_avg|diff",
    "player_star_score|season|top3_sum|diff",
    "player_star_share|season|top1_share|diff",
    "player_star_share|season|top3_share|diff",
    "player_rotation_per|season|weighted_MPG|diff",
    "player_per|season|top1_weighted_MPG|diff",
    "player_per|season|top2_weighted_MPG|diff",
    "player_per|season|top3_weighted_MPG|diff",
    "player_bench_per|season|weighted_MPG|diff",
    "player_starters_per|season|avg|diff",
    "player_starter_per|season|avg|diff",
    "player_starter_bench_per_gap|season|derived(k=35)|diff",
    "player_continuity|season|avg|diff",
]

# ── 5 feature-set variants ──
VARIANTS = [
    {
        "id": "A",
        "name": "core",
        "features": CORE,
    },
    {
        "id": "B",
        "name": "playmaking",
        "features": CORE + PLAYMAKING,
    },
    {
        "id": "C",
        "name": "matchup_sos",
        "features": CORE + PLAYMAKING + MU7_SEASON + SOS,
    },
    {
        "id": "D",
        "name": "form_sched",
        "features": CORE + PLAYMAKING + MU7_SEASON + SOS + FORM_GAMES12 + SCHEDULE + DELTAS,
    },
    {
        "id": "E",
        "name": "players",
        "features": CORE + PLAYMAKING + MU7_SEASON + SOS + FORM_GAMES12 + SCHEDULE + DELTAS + PLAYERS,
    },
]

# ── Grid config ──
RIDGE_ALPHAS = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
ELASTICNET_ALPHAS = [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
ELASTICNET_L1_RATIO = 0.5
XGBOOST_PARAMS = {"n_estimators": 200, "max_depth": 6}

BEGIN_YEAR = 2010
CALIBRATION_YEARS = [2021, 2022, 2023]
EVALUATION_YEAR = 2024
MIN_GAMES = 20

MASTER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "master_training", "MASTER_TRAINING.csv"
)


def extract_variant_csv(features, variant_name):
    """Extract features from master CSV for a variant, return path to temp CSV."""
    from bball.services.training_data import extract_features_from_master_for_points

    output_path = os.path.join(
        tempfile.gettempdir(), f"margin_exp_{variant_name}.csv"
    )
    return extract_features_from_master_for_points(
        master_path=MASTER_PATH,
        requested_features=features,
        output_path=output_path,
        begin_year=BEGIN_YEAR,
        min_games_played=MIN_GAMES,
    )


def train_ridge(trainer, csv_path, features):
    """Train Ridge with internal alpha grid search. Returns (result_dict, alpha)."""
    result = trainer.train(
        model_type="Ridge",
        alphas=RIDGE_ALPHAS,
        selected_features=features,
        training_csv=csv_path,
        target="margin",
        use_time_calibration=True,
        calibration_years=CALIBRATION_YEARS,
        evaluation_year=EVALUATION_YEAR,
        begin_year=BEGIN_YEAR,
    )
    return result, result.get("selected_alpha")


def train_elasticnet(trainer, csv_path, features):
    """Train ElasticNet with external alpha sweep. Returns (result_dict, best_alpha)."""
    best_mae = float("inf")
    best_result = None
    best_alpha = None

    for alpha in ELASTICNET_ALPHAS:
        result = trainer.train(
            model_type="ElasticNet",
            selected_features=features,
            training_csv=csv_path,
            target="margin",
            use_time_calibration=True,
            calibration_years=CALIBRATION_YEARS,
            evaluation_year=EVALUATION_YEAR,
            begin_year=BEGIN_YEAR,
            alpha=alpha,
            l1_ratio=ELASTICNET_L1_RATIO,
        )
        fm = result["final_metrics"]
        mae = fm.get("margin_mae", float("inf"))
        if not np.isnan(mae) and mae < best_mae:
            best_mae = mae
            best_result = result
            best_alpha = alpha

    return best_result, best_alpha


def train_xgboost(trainer, csv_path, features):
    """Train XGBoost. Returns (result_dict, None)."""
    result = trainer.train(
        model_type="XGBoost",
        selected_features=features,
        training_csv=csv_path,
        target="margin",
        use_time_calibration=True,
        calibration_years=CALIBRATION_YEARS,
        evaluation_year=EVALUATION_YEAR,
        begin_year=BEGIN_YEAR,
        **XGBOOST_PARAMS,
    )
    return result, None


def _sanitize(v):
    """Replace NaN/Inf with None for MongoDB storage."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    return v


def save_config(db, config_mgr, model_type, features, alpha, variant_name, result):
    """Save a points config with metrics matching the web UI format."""
    from datetime import datetime, timezone
    from bson import ObjectId

    name = f"margin-{variant_name}-{model_type}"
    if alpha is not None:
        name += f"-a{alpha}"

    config_id, _ = config_mgr.create_points_config(
        model_type=model_type,
        features=features,
        target="margin",
        alpha=alpha or 1.0,
        l1_ratio=ELASTICNET_L1_RATIO if model_type == "ElasticNet" else None,
        begin_year=BEGIN_YEAR,
        calibration_years=CALIBRATION_YEARS,
        evaluation_year=EVALUATION_YEAR,
        min_games_played=MIN_GAMES,
        use_master=True,
        name=name,
    )

    # Update the config with metrics/training_stats matching the web UI format
    fm = result["final_metrics"]
    points_collection = config_mgr._points_repo.collection.name
    update = {
        "metrics": {
            "diff_mae": _sanitize(fm.get("margin_mae")),
            "diff_rmse": _sanitize(fm.get("margin_rmse")),
            "diff_r2": _sanitize(fm.get("margin_r2")),
            # margin-only models don't have separate home/away/total metrics
            "home_mae": None,
            "away_mae": None,
            "total_mae": None,
        },
        "training_stats": {
            "n_samples": result.get("n_samples"),
            "n_features": result.get("n_features"),
        },
        "trained_at": datetime.now(timezone.utc),
        "use_time_calibration": True,
        "begin_year": BEGIN_YEAR,
        "calibration_years": CALIBRATION_YEARS,
        "evaluation_year": EVALUATION_YEAR,
    }
    if alpha is not None and model_type in ("Ridge", "ElasticNet"):
        update["best_alpha"] = alpha
    if result.get("alphas_tested"):
        update["alphas_tested"] = result["alphas_tested"]

    db[points_collection].update_one(
        {"_id": ObjectId(config_id)},
        {"$set": update},
    )
    return config_id


def main():
    from bball.mongo import Mongo
    from bball.models.points_regression import PointsRegressionTrainer
    from bball.services.config_manager import ModelConfigManager
    from bball.league_config import load_league_config

    league = load_league_config("nba")
    db = Mongo().db
    config_mgr = ModelConfigManager(db, league=league)

    model_types = [
        ("Ridge", train_ridge),
        ("ElasticNet", train_elasticnet),
        ("XGBoost", train_xgboost),
    ]

    total = len(VARIANTS) * len(model_types)
    print(f"Running {len(VARIANTS)} variants x {len(model_types)} model types = {total} experiments")
    print(f"  begin_year={BEGIN_YEAR}, cal={CALIBRATION_YEARS}, eval={EVALUATION_YEAR}, min_games={MIN_GAMES}")
    print(f"  Ridge alphas: {RIDGE_ALPHAS}")
    print(f"  ElasticNet alphas: {ELASTICNET_ALPHAS}, l1_ratio={ELASTICNET_L1_RATIO}")
    print(f"  XGBoost: {XGBOOST_PARAMS}")
    print()

    all_rows = []

    for variant in VARIANTS:
        vid = variant["id"]
        vname = variant["name"]
        features = variant["features"]

        print(f"{'=' * 70}")
        print(f"Variant {vid}: {vname}  ({len(features)} features)")
        print(f"{'=' * 70}")

        # Extract CSV once per variant
        try:
            csv_path = extract_variant_csv(features, vname)
            print(f"  Extracted CSV: {csv_path}")
        except Exception as e:
            print(f"  ERROR extracting features: {e}")
            for mt_name, _ in model_types:
                all_rows.append({
                    "variant": f"{vid}:{vname}",
                    "model": mt_name,
                    "alpha": "-",
                    "mae": "-",
                    "rmse": "-",
                    "r2": "-",
                    "config_id": f"ERR: {str(e)[:40]}",
                })
            continue

        for mt_name, mt_fn in model_types:
            print(f"\n  Training {mt_name}...")
            t0 = time.time()
            try:
                trainer = PointsRegressionTrainer(db=db)
                result, alpha = mt_fn(trainer, csv_path, features)
                elapsed = time.time() - t0

                fm = result["final_metrics"]
                mae = fm.get("margin_mae", float("nan"))
                rmse = fm.get("margin_rmse", float("nan"))
                r2 = fm.get("margin_r2", float("nan"))
                alpha_s = alpha if alpha is not None else "-"

                print(f"    {mt_name} done in {elapsed:.1f}s  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  alpha={alpha_s}")

                # Save config with metrics for web UI
                config_id = save_config(db, config_mgr, mt_name, features, alpha, vname, result)
                print(f"    Saved config: {config_id}")

                all_rows.append({
                    "variant": f"{vid}:{vname}",
                    "model": mt_name,
                    "alpha": alpha_s,
                    "mae": mae,
                    "rmse": rmse,
                    "r2": r2,
                    "config_id": config_id,
                })
            except Exception as e:
                elapsed = time.time() - t0
                print(f"    ERROR ({elapsed:.1f}s): {e}")
                all_rows.append({
                    "variant": f"{vid}:{vname}",
                    "model": mt_name,
                    "alpha": "-",
                    "mae": "-",
                    "rmse": "-",
                    "r2": "-",
                    "config_id": f"ERR: {str(e)[:40]}",
                })

    # ── Print sorted comparison table ──
    print()
    print("=" * 120)
    print("RESULTS (sorted by margin MAE ascending — lower is better)")
    print("=" * 120)

    def sort_key(row):
        mae = row["mae"]
        return mae if isinstance(mae, (int, float)) and not np.isnan(mae) else 9999
    all_rows.sort(key=sort_key)

    header = f"{'Variant':<20} {'Model':<14} {'Alpha':>7}  {'MAE':>8}  {'RMSE':>8}  {'R²':>8}  {'Config ID'}"
    print(header)
    print("-" * 120)

    for row in all_rows:
        mae = row["mae"]
        rmse = row["rmse"]
        r2 = row["r2"]
        mae_s = f"{mae:.3f}" if isinstance(mae, (int, float)) else str(mae)
        rmse_s = f"{rmse:.3f}" if isinstance(rmse, (int, float)) else str(rmse)
        r2_s = f"{r2:.4f}" if isinstance(r2, (int, float)) else str(r2)
        alpha_s = f"{row['alpha']}" if row["alpha"] != "-" else "-"

        print(f"{row['variant']:<20} {row['model']:<14} {alpha_s:>7}  {mae_s:>8}  {rmse_s:>8}  {r2_s:>8}  {row['config_id']}")

    print()
    print(f"Total experiments: {len(all_rows)}")

    # Highlight best overall
    numeric_rows = [r for r in all_rows if isinstance(r["mae"], (int, float)) and not np.isnan(r["mae"])]
    if numeric_rows:
        best = min(numeric_rows, key=lambda r: r["mae"])
        print(f"\nBest: {best['variant']}  {best['model']}  alpha={best['alpha']}  "
              f"MAE={best['mae']:.3f}  RMSE={best['rmse']:.3f}  R²={best['r2']:.4f}  "
              f"config={best['config_id']}")


if __name__ == "__main__":
    main()
