#!/usr/bin/env python
"""
Model Results CLI Script

Thin wrapper around TrainingService.get_model_results().

Usage:
    python -m bball_app.cli.scripts.model_results <league> --model <name_or_id>

Examples:
    python -m bball_app.cli.scripts.model_results nba --model "test_lr_model"
    python -m bball_app.cli.scripts.model_results nba --model "67a1b2c3d4e5f6g7h8i9j0k1"
    python -m bball_app.cli.scripts.model_results nba --list
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='View model details and results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  View model by name:
    %(prog)s nba --model "test_lr_model"

  View model by ID:
    %(prog)s nba --model "67a1b2c3d4e5f6g7h8i9j0k1"

  List all models:
    %(prog)s nba --list

  List trained models only:
    %(prog)s nba --list --trained-only

  List ensemble models:
    %(prog)s nba --list --ensemble-only
        """
    )

    parser.add_argument('league', help='League ID (e.g., nba, cbb)')
    parser.add_argument('--model', help='Model name or MongoDB _id')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--trained-only', action='store_true', help='Only show trained models')
    parser.add_argument('--ensemble-only', action='store_true', help='Only show ensemble models')
    parser.add_argument('--top-features', type=int, default=20,
                        help='Number of top features to show (default: 20)')
    parser.add_argument('--all-features', action='store_true',
                        help='Show all feature importances')

    args = parser.parse_args()

    if not args.model and not args.list:
        print("Error: Either --model or --list is required", file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    # Import here to avoid slow imports on --help
    from bball_app.core.league_config import load_league_config
    from bball_app.core.services.training_service import TrainingService

    # Load league config
    league = load_league_config(args.league)

    # Initialize service
    service = TrainingService(league=league)

    # List models
    if args.list:
        models = service.list_models(
            ensemble_only=args.ensemble_only,
            trained_only=args.trained_only,
        )

        if not models:
            print("No models found.")
            return

        print(f"Found {len(models)} models:")
        print()
        print(f"{'ID':<26} {'Name':<30} {'Type':<18} {'Trained':<8} {'Accuracy'}")
        print("-" * 100)

        for m in models:
            model_id = m.get('id', 'N/A')[:24]
            name = m.get('name', 'N/A')[:28]
            model_type = m.get('model_type', 'N/A')
            if m.get('is_ensemble'):
                model_type = f"Ensemble({model_type})"
            model_type = model_type[:16]
            trained = 'Yes' if m.get('trained') else 'No'
            accuracy = m.get('accuracy')
            if accuracy:
                accuracy = f"{accuracy:.2f}%"
            else:
                accuracy = 'N/A'
            print(f"{model_id:<26} {name:<30} {model_type:<18} {trained:<8} {accuracy}")

        print()
        return

    # Get model results
    result = service.get_model_results(args.model)

    if not result:
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    is_ensemble = result.get('is_ensemble', False)

    # Print header
    print("=" * 60)
    if is_ensemble:
        print("ENSEMBLE MODEL DETAILS")
    else:
        print("MODEL DETAILS")
    print("=" * 60)
    print()

    # Print config details
    print("CONFIGURATION:")
    print("-" * 40)
    print(f"  Config ID:     {result.get('config_id', 'N/A')}")
    print(f"  Name:          {result.get('name', 'N/A')}")
    print(f"  Model Type:    {result.get('model_type', 'N/A')}")
    print(f"  Is Ensemble:   {is_ensemble}")
    print(f"  Begin Year:    {result.get('begin_year', 'N/A')}")
    print(f"  Calibration:   {result.get('calibration_years', 'N/A')}")
    print(f"  Evaluation:    {result.get('evaluation_year', 'N/A')}")
    print(f"  C-value:       {result.get('c_value', 'N/A')}")
    print(f"  Calibration:   {result.get('calibration_method', 'N/A')}")
    print(f"  Features:      {result.get('feature_count', 0)}")
    print()

    # Print metrics
    metrics = result.get('metrics', {})
    print("METRICS:")
    print("-" * 40)
    accuracy = metrics.get('accuracy')
    log_loss = metrics.get('log_loss')
    brier = metrics.get('brier_score')
    auc = metrics.get('auc')

    if accuracy is not None:
        print(f"  Accuracy:   {accuracy:.2f}%")
    else:
        print(f"  Accuracy:   N/A")

    if log_loss is not None:
        print(f"  Log Loss:   {log_loss:.4f}")
    else:
        print(f"  Log Loss:   N/A")

    if brier is not None:
        print(f"  Brier:      {brier:.4f}")
    else:
        print(f"  Brier:      N/A")

    if auc is not None:
        print(f"  AUC:        {auc:.4f}")
    else:
        print(f"  AUC:        N/A")
    print()

    # For ensemble models, print base model details
    if is_ensemble:
        base_models = result.get('base_models', [])
        if base_models:
            print("BASE MODELS:")
            print("-" * 40)
            for i, bm in enumerate(base_models, 1):
                print(f"  {i}. {bm.get('name', 'N/A')} ({bm.get('model_type', 'N/A')})")
                print(f"     ID: {bm.get('id', 'N/A')}")
                print(f"     Features: {len(bm.get('features', []))}")

                # Print base model feature importances
                bm_importances = bm.get('feature_importances', {})
                if bm_importances:
                    print(f"     Top Features:")
                    sorted_features = sorted(bm_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                    for feat, importance in sorted_features:
                        print(f"       - {feat}: {importance:.4f}")
                print()
    else:
        # Print feature importances for base models
        feature_importances = result.get('feature_importances', {})
        if feature_importances:
            n_features = args.top_features if not args.all_features else len(feature_importances)
            print(f"FEATURE IMPORTANCES (top {n_features}):")
            print("-" * 40)
            sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:n_features]
            for i, (feat, importance) in enumerate(sorted_features, 1):
                print(f"  {i:3d}. {feat}: {importance:.6f}")
            print()

    # Print feature list
    features = result.get('features', [])
    if features and not is_ensemble:
        print(f"FEATURES ({len(features)} total):")
        print("-" * 40)
        # Show first 10 and last 5
        if len(features) <= 15:
            for feat in features:
                print(f"  - {feat}")
        else:
            for feat in features[:10]:
                print(f"  - {feat}")
            print(f"  ... ({len(features) - 15} more features)")
            for feat in features[-5:]:
                print(f"  - {feat}")
        print()


if __name__ == '__main__':
    main()
