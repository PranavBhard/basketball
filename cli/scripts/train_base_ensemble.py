#!/usr/bin/env python
"""
Train Base Ensemble CLI Script

Thin wrapper around TrainingService.train_ensemble().

IMPORTANT: Temporal parameters (begin_year, calibration_years, evaluation_year)
are derived from the base models to ensure platform-wide consistency. All base
models must have identical temporal configurations.

Usage:
    python -m bball_app.cli.scripts.train_base_ensemble <league> [options]

Examples:
    python -m bball_app.cli.scripts.train_base_ensemble nba --model LR --c-value 0.1 \
        --models "model1_name,model2_name" --extra-features "pred_margin"
"""

import argparse
import sys


def parse_list(list_str: str) -> list:
    """Parse comma-separated values into a list."""
    if not list_str:
        return []
    return [s.strip() for s in list_str.split(',') if s.strip()]


def main():
    parser = argparse.ArgumentParser(
        description='Train an ensemble (stacking) model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a LogisticRegression ensemble:
    %(prog)s nba --model LR --c-value 0.1 \\
        --models "model1_name,model2_name" \\
        --extra-features "pred_margin"

  Train with disagreement and confidence features:
    %(prog)s nba --model LR --c-value 0.1 \\
        --models "model1,model2,model3" \\
        --use-disagree --use-conf

Model Types:
  LR  = LogisticRegression
  GB  = GradientBoosting
  SVM = Support Vector Machine

Stacking Modes:
  naive    = Use only base model predictions
  informed = Include additional features (disagree, conf, meta-features)

Temporal Configuration:
  The meta-model's temporal split (begin_year, calibration_years, evaluation_year)
  is DERIVED from the base models. All base models must have identical temporal
  configurations. This ensures platform-wide consistency in how time-calibration
  is handled.
        """
    )

    parser.add_argument('league', help='League ID (e.g., nba, cbb)')
    parser.add_argument('--model', required=True, choices=['LR', 'GB', 'SVM'],
                        help='Meta-model type: LR (LogisticRegression), GB (GradientBoosting), SVM')
    parser.add_argument('--c-value', type=float, default=0.1,
                        help='Regularization parameter C for meta-model (default: 0.1)')
    parser.add_argument('--models', required=True,
                        help='Comma-separated base model names or IDs')
    parser.add_argument('--extra-features',
                        help='Comma-separated additional features for meta-model (e.g., pred_margin)')
    parser.add_argument('--stacking-mode', default='informed', choices=['naive', 'informed'],
                        help='Stacking mode (default: informed)')
    parser.add_argument('--use-disagree', action='store_true',
                        help='Include pairwise disagreement features')
    parser.add_argument('--use-conf', action='store_true',
                        help='Include confidence features for each base model')

    args = parser.parse_args()

    # Import here to avoid slow imports on --help
    from bball_app.core.league_config import load_league_config
    from bball_app.core.services.training_service import TrainingService

    # Load league config
    league = load_league_config(args.league)

    # Parse arguments
    base_models = parse_list(args.models)
    extra_features = parse_list(args.extra_features) if args.extra_features else None

    if len(base_models) < 2:
        print("Error: At least 2 base models are required for ensemble", file=sys.stderr)
        sys.exit(1)

    # Initialize service
    service = TrainingService(league=league)

    # Resolve first model to display temporal config (will be validated by train_ensemble)
    first_model = service.resolve_model(base_models[0])
    if not first_model:
        print(f"Error: Base model not found: {base_models[0]}", file=sys.stderr)
        sys.exit(1)

    # Display temporal config derived from base models
    calibration_seasons = first_model.get('calibration_years', [])
    evaluation_season = first_model.get('evaluation_year')
    begin_year = first_model.get('begin_year')

    print(f"Training {args.model} ensemble for {args.league}...")
    print(f"  Base models: {base_models}")
    print(f"  Temporal config (from base models):")
    print(f"    Begin year: {begin_year}")
    print(f"    Calibration seasons: {calibration_seasons}")
    print(f"    Evaluation season: {evaluation_season}")
    print(f"  Meta C-value: {args.c_value}")
    print(f"  Stacking mode: {args.stacking_mode}")
    if extra_features:
        print(f"  Extra features: {extra_features}")
    if args.use_disagree:
        print(f"  Using disagreement features")
    if args.use_conf:
        print(f"  Using confidence features")
    print()

    # Train the ensemble (temporal params derived from base models internally)
    try:
        result = service.train_ensemble(
            meta_model_type=args.model,
            base_model_names_or_ids=base_models,
            meta_c_value=args.c_value,
            extra_features=extra_features,
            stacking_mode=args.stacking_mode,
            use_disagree=args.use_disagree,
            use_conf=args.use_conf,
        )
    except Exception as e:
        print(f"Error training ensemble: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print results
    print("=" * 60)
    print("ENSEMBLE TRAINING COMPLETE")
    print("=" * 60)
    print()
    print(f"Config ID: {result.get('config_id', 'N/A')}")
    print(f"Run ID: {result['run_id']}")
    print()

    # Print metrics
    metrics = result.get('metrics', {})
    print("ENSEMBLE METRICS:")
    print("-" * 40)
    if metrics.get('accuracy_mean'):
        print(f"  Accuracy:   {metrics['accuracy_mean']:.2f}%")
    if metrics.get('log_loss_mean'):
        print(f"  Log Loss:   {metrics['log_loss_mean']:.4f}")
    if metrics.get('brier_mean'):
        print(f"  Brier:      {metrics['brier_mean']:.4f}")
    if metrics.get('auc_mean'):
        print(f"  AUC:        {metrics['auc_mean']:.4f}")
    print()

    # Print base model summaries
    base_models_info = result.get('base_models', [])
    if base_models_info:
        print("BASE MODELS:")
        print("-" * 40)
        for i, bm in enumerate(base_models_info, 1):
            print(f"  {i}. {bm.get('name', 'N/A')} ({bm.get('model_type', 'N/A')})")
            print(f"     ID: {bm.get('id', 'N/A')}")
        print()

    # Print base model metrics from diagnostics
    diagnostics = result.get('diagnostics', {})
    base_summaries = diagnostics.get('base_models_summary', [])
    if base_summaries:
        print("BASE MODEL PERFORMANCE (on evaluation set):")
        print("-" * 40)
        for i, bm in enumerate(base_summaries, 1):
            bm_metrics = bm.get('metrics', {})
            accuracy = bm_metrics.get('accuracy_mean', 'N/A')
            if isinstance(accuracy, (int, float)):
                accuracy = f"{accuracy:.2f}%"
            print(f"  {i}. {bm.get('run_id', 'N/A')[:8]}...")
            print(f"     Accuracy: {accuracy}")
            print(f"     Features: {len(bm.get('feature_names', []))}")
        print()

    # Print meta-feature importances
    meta_importances = diagnostics.get('meta_feature_importances', {})
    if meta_importances:
        print("META-MODEL FEATURE IMPORTANCES:")
        print("-" * 40)
        sorted_features = sorted(meta_importances.items(), key=lambda x: x[1], reverse=True)
        for feat, importance in sorted_features:
            print(f"  {feat}: {importance:.4f}")
        print()


if __name__ == '__main__':
    main()
