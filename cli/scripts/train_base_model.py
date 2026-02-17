#!/usr/bin/env python
"""
Train Base Model CLI Script

Thin wrapper around TrainingService.train_base_model().

Usage:
    python -m bball_app.cli.scripts.train_base_model <league> [options]

Examples:
    python -m bball_app.cli.scripts.train_base_model nba --model LR --c-value 0.1 \
        --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \
        --calibration-seasons 2023 --evaluation-seasons 2024 \
        --features "points|season|avg|diff,assists|season|avg|diff" \
        --name "test_lr_model"
"""

import argparse
import sys


def parse_seasons(seasons_str: str) -> list:
    """Parse comma-separated season years into a list of ints."""
    if not seasons_str:
        return []
    return [int(s.strip()) for s in seasons_str.split(',')]


def parse_features(features_str: str) -> list:
    """Parse comma-separated feature names."""
    if not features_str:
        return []
    return [f.strip() for f in features_str.split(',') if f.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='Train a base classifier model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a LogisticRegression model:
    %(prog)s nba --model LR --c-value 0.1 \\
        --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \\
        --calibration-seasons 2023 --evaluation-seasons 2024 \\
        --features "points|season|avg|diff,assists|season|avg|diff" \\
        --name "test_lr_model"

  Train a GradientBoosting model:
    %(prog)s nba --model GB \\
        --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \\
        --calibration-seasons 2023 --evaluation-seasons 2024 \\
        --features "points|season|avg|diff" --name "test_gb_model"

Model Types:
  LR  = LogisticRegression
  GB  = GradientBoosting
  SVM = Support Vector Machine
        """
    )

    parser.add_argument('league', help='League ID (e.g., nba, cbb)')
    parser.add_argument('--model', required=True, choices=['LR', 'GB', 'SVM'],
                        help='Model type: LR (LogisticRegression), GB (GradientBoosting), SVM')
    parser.add_argument('--c-value', type=float, default=0.1,
                        help='Regularization parameter C (default: 0.1)')
    parser.add_argument('--time-method', default='sigmoid', choices=['sigmoid', 'isotonic'],
                        help='Calibration method (default: sigmoid)')
    parser.add_argument('--train-seasons', required=True,
                        help='Comma-separated training seasons (e.g., 2012,2013,2014,...)')
    parser.add_argument('--calibration-seasons', required=True,
                        help='Comma-separated calibration seasons (e.g., 2023)')
    parser.add_argument('--evaluation-seasons', required=True,
                        help='Evaluation season year (e.g., 2024)')
    parser.add_argument('--features', required=True,
                        help='Comma-separated feature names')
    parser.add_argument('--name', help='Model name (auto-generated if not provided)')
    parser.add_argument('--min-games', type=int, default=20,
                        help='Minimum games played filter (default: 20)')
    parser.add_argument('--include-injuries', action='store_true',
                        help='Include injury features')
    parser.add_argument('--no-master', action='store_true',
                        help='Do not use master training CSV')

    args = parser.parse_args()

    # Import here to avoid slow imports on --help
    from bball_app.core.league_config import load_league_config
    from bball_app.core.services.training_service import TrainingService

    # Load league config
    league = load_league_config(args.league)

    # Parse arguments
    train_seasons = parse_seasons(args.train_seasons)
    calibration_seasons = parse_seasons(args.calibration_seasons)
    evaluation_season = int(args.evaluation_seasons)
    features = parse_features(args.features)

    if not features:
        print("Error: No features specified", file=sys.stderr)
        sys.exit(1)

    if not train_seasons:
        print("Error: No training seasons specified", file=sys.stderr)
        sys.exit(1)

    # Initialize service
    service = TrainingService(league=league)

    print(f"Training {args.model} model for {args.league}...")
    print(f"  Train seasons: {train_seasons}")
    print(f"  Calibration seasons: {calibration_seasons}")
    print(f"  Evaluation season: {evaluation_season}")
    print(f"  Features: {len(features)} features")
    print(f"  C-value: {args.c_value}")
    print(f"  Calibration method: {args.time_method}")
    print(f"  Min games played: {args.min_games}")
    print()

    # Train the model
    try:
        result = service.train_base_model(
            model_type=args.model,
            features=features,
            train_seasons=train_seasons,
            calibration_seasons=calibration_seasons,
            evaluation_season=evaluation_season,
            c_value=args.c_value,
            calibration_method=args.time_method,
            name=args.name,
            min_games_played=args.min_games,
            include_injuries=args.include_injuries,
            use_master=not args.no_master,
        )
    except Exception as e:
        print(f"Error training model: {e}", file=sys.stderr)
        sys.exit(1)

    # Print results
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print()
    print(f"Config ID: {result['config_id']}")
    print(f"Run ID: {result['run_id']}")
    print(f"Samples: {result['n_samples']}")
    print(f"Features: {result['n_features']}")
    print()

    # Print metrics
    metrics = result.get('metrics', {})
    print("METRICS:")
    print("-" * 40)
    if metrics.get('accuracy_mean'):
        print(f"  Accuracy:   {metrics['accuracy_mean']:.2f}%")
    if metrics.get('log_loss_mean'):
        print(f"  Log Loss:   {metrics['log_loss_mean']:.4f}")
    if metrics.get('brier_mean'):
        print(f"  Brier:      {metrics['brier_mean']:.4f}")
    if metrics.get('auc') or metrics.get('auc_mean'):
        auc = metrics.get('auc') or metrics.get('auc_mean')
        print(f"  AUC:        {auc:.4f}")
    print()

    # Print F-scores (ANOVA F-test)
    f_scores = result.get('f_scores', {})
    if f_scores:
        print("F-SCORES (ANOVA F-test):")
        print("-" * 50)
        sorted_f = sorted(f_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, score) in enumerate(sorted_f, 1):
            print(f"  {i:2d}. {feat}: {score:.4f}")
        print()

    # Print model importance scores (coefficients/feature_importances_)
    feature_importances = result.get('feature_importances', {})
    if feature_importances:
        print("IMPORTANCE SCORES (model coefficients):")
        print("-" * 50)
        sorted_imp = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
        for i, (feat, importance) in enumerate(sorted_imp, 1):
            print(f"  {i:2d}. {feat}: {importance:.6f}")
        print()

    print(f"Model saved. Use 'model_results.sh {args.league} --model {result['config_id']}' to view details.")


if __name__ == '__main__':
    main()
