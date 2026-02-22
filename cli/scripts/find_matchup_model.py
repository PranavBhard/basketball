#!/usr/bin/env python
"""
Find Optimal Matchup Style Base Model

Sweeps 6 feature-set variants × (LR C∈{0.01,0.1,1.0} + GB) = 24 experiments
to find the best matchup-focused base model for the ensemble.

Uses same time config as existing ensemble:
  begin_year=2010, cal=[2021,2022,2023], eval=2024, min_games=20

Usage:
    python cli/scripts/find_matchup_model.py
"""

import sys
import time

# ── mu_ stats (7, excluding mu_off_vs_def which duplicates off_rtg+def_rtg diff) ──
MU7_SEASON = [
    "mu_pace_delta|season|avg|diff",
    "mu_oreb_vs_dreb|season|avg|diff",
    "mu_to_vs_steals|season|avg|diff",
    "mu_three_exposure|season|avg|diff",
    "mu_ft_draw_vs_fouls|season|avg|diff",
    "mu_paint_vs_blocks|season|avg|diff",
    "mu_ast_vs_steals|season|avg|diff",
]

MU7_GAMES12 = [
    "mu_pace_delta|games_12|avg|diff",
    "mu_oreb_vs_dreb|games_12|avg|diff",
    "mu_to_vs_steals|games_12|avg|diff",
    "mu_three_exposure|games_12|avg|diff",
    "mu_ft_draw_vs_fouls|games_12|avg|diff",
    "mu_paint_vs_blocks|games_12|avg|diff",
    "mu_ast_vs_steals|games_12|avg|diff",
]

DERIVED = [
    "pace_interaction|season|harmonic_mean|none",
    "exp_points_matchup|season|derived|diff",
]

NETS = [
    "efg_net|season|avg|diff",
    "ts_net|season|avg|diff",
    "three_pct_net|season|avg|diff",
]

# ── 6 feature-set variants ──
VARIANTS = [
    {
        "id": "A",
        "name": "pure_mu7",
        "features": MU7_SEASON,
    },
    {
        "id": "B",
        "name": "full_mu8",
        "features": MU7_SEASON + ["mu_off_vs_def|season|avg|diff"],
    },
    {
        "id": "C",
        "name": "mu7_derived",
        "features": MU7_SEASON + DERIVED,
    },
    {
        "id": "D",
        "name": "mu7_dual12",
        "features": MU7_SEASON + MU7_GAMES12,
    },
    {
        "id": "E",
        "name": "mu7_nets",
        "features": MU7_SEASON + NETS,
    },
    {
        "id": "F",
        "name": "broad",
        "features": MU7_SEASON + MU7_GAMES12 + DERIVED + NETS,
    },
]

# ── Grid config ──
MODEL_TYPES = ["LR", "GB"]
C_VALUES = [0.01, 0.1, 1.0]
BEGIN_YEAR = 2010
CALIBRATION_YEARS = [2021, 2022, 2023]
EVALUATION_YEAR = 2024
MIN_GAMES = 20


def main():
    from bball.league_config import load_league_config
    from bball.services.training_service import TrainingService

    league = load_league_config("nba")
    service = TrainingService(league=league)

    total = len(VARIANTS) * (len(C_VALUES) + 1)  # LR×3 + GB×1 per variant
    print(f"Running {len(VARIANTS)} variants × {len(C_VALUES)+1} combos = {total} experiments")
    print(f"  begin_year={BEGIN_YEAR}, cal={CALIBRATION_YEARS}, eval={EVALUATION_YEAR}, min_games={MIN_GAMES}")
    print()

    all_rows = []

    for variant in VARIANTS:
        vid = variant["id"]
        vname = variant["name"]
        features = variant["features"]
        prefix = f"MU-{vname}"

        print(f"{'='*60}")
        print(f"Variant {vid}: {vname}  ({len(features)} features)")
        print(f"  prefix: {prefix}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            result = service.train_model_grid(
                model_types=MODEL_TYPES,
                c_values=C_VALUES,
                features=features,
                begin_year=BEGIN_YEAR,
                calibration_years=CALIBRATION_YEARS,
                evaluation_year=EVALUATION_YEAR,
                min_games_played=MIN_GAMES,
                calibration_method="sigmoid",
                name_prefix=prefix,
                use_master=True,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0
        print(f"  completed in {elapsed:.1f}s")

        # Collect rows from results
        for mt, mt_data in result.get("model_type_results", {}).items():
            for entry in mt_data.get("all", []):
                if "error" in entry:
                    all_rows.append({
                        "variant": f"{vid}:{vname}",
                        "model": mt,
                        "C": entry.get("c_value", "-"),
                        "accuracy": "-",
                        "log_loss": "-",
                        "brier": "-",
                        "config_id": f"ERR: {entry['error'][:40]}",
                    })
                else:
                    metrics = entry.get("metrics", {})
                    all_rows.append({
                        "variant": f"{vid}:{vname}",
                        "model": mt,
                        "C": entry.get("c_value", "-"),
                        "accuracy": metrics.get("accuracy_mean", 0),
                        "log_loss": metrics.get("log_loss_mean", 0),
                        "brier": metrics.get("brier_mean", 0),
                        "config_id": entry.get("config_id", "?"),
                    })

    # ── Print sorted comparison table ──
    print()
    print("=" * 110)
    print("RESULTS (sorted by accuracy desc)")
    print("=" * 110)

    # Sort: numeric accuracy descending, errors at bottom
    def sort_key(row):
        acc = row["accuracy"]
        return acc if isinstance(acc, (int, float)) else -999

    all_rows.sort(key=sort_key, reverse=True)

    header = f"{'Variant':<18} {'Model':<22} {'C':>5}  {'Acc%':>7}  {'LogLoss':>8}  {'Brier':>8}  {'Config ID'}"
    print(header)
    print("-" * 110)

    for row in all_rows:
        acc = row["accuracy"]
        ll = row["log_loss"]
        br = row["brier"]
        acc_s = f"{acc:.2f}" if isinstance(acc, (int, float)) else acc
        ll_s = f"{ll:.4f}" if isinstance(ll, (int, float)) else ll
        br_s = f"{br:.4f}" if isinstance(br, (int, float)) else br
        c_s = f"{row['C']}" if row["C"] != "-" else "-"

        print(f"{row['variant']:<18} {row['model']:<22} {c_s:>5}  {acc_s:>7}  {ll_s:>8}  {br_s:>8}  {row['config_id']}")

    print()
    print(f"Total experiments: {len(all_rows)}")

    # Highlight best overall
    numeric_rows = [r for r in all_rows if isinstance(r["accuracy"], (int, float))]
    if numeric_rows:
        best = max(numeric_rows, key=lambda r: r["accuracy"])
        print(f"\nBest: {best['variant']}  {best['model']}  C={best['C']}  acc={best['accuracy']:.2f}%  config={best['config_id']}")


if __name__ == "__main__":
    main()
