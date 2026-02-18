"""ModelsCommand — basketball models nba --list | --model <name_or_id>"""

import argparse
from sportscore.cli.base import BaseCommand, format_table, print_metrics, print_feature_importances


class ModelsCommand(BaseCommand):
    name = "models"
    help = "List models or view details for a model name or ID"
    description = "List models or view config, metrics, feature importances, and feature list. No --select/--delete (basketball TrainingService does not expose those)."
    epilog = """
Examples:
  basketball models nba --list
  basketball models nba --list --trained-only --ensemble-only
  basketball models nba --model "test_lr_model"
  basketball models nba --model 67a1b2c3d4e5f6a7b8c9d0e1 --top-features 30 --all-features
"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--list", action="store_true", help="List models")
        parser.add_argument("--model", type=str, help="View details for model name or ID")
        parser.add_argument("--trained-only", action="store_true", help="Filter to trained only")
        parser.add_argument("--ensemble-only", action="store_true", help="Filter to ensembles only")
        parser.add_argument("--top-features", type=int, default=20, help="Number of features to show (default: 20)")
        parser.add_argument("--all-features", action="store_true", help="Show all feature importances")

    def handle(self, args: argparse.Namespace, league, db) -> None:
        from bball.services.training_service import TrainingService

        service = TrainingService(league=league, db=db)

        if args.list:
            models = service.list_models(
                ensemble_only=args.ensemble_only,
                trained_only=args.trained_only,
            )
            if not models:
                print("No models found.")
                return
            rows = []
            for m in models:
                model_id = str(m.get("id", "N/A"))[:24]
                name = (m.get("name") or "N/A")[:28]
                model_type = (m.get("model_type") or "N/A")
                if m.get("is_ensemble"):
                    model_type = f"Ensemble({model_type})"
                model_type = model_type[:16]
                trained = "Yes" if m.get("trained") else "No"
                acc = m.get("accuracy")
                acc_str = f"{acc:.2f}%" if acc is not None else "N/A"
                rows.append([model_id, name, model_type, trained, acc_str])
            print(format_table(["ID", "Name", "Type", "Trained", "Accuracy"], rows))
            return

        if args.model:
            result = service.get_model_results(args.model)
            if not result:
                self.error(f"Model not found: {args.model}")
            is_ensemble = result.get("is_ensemble", False)

            print("=" * 60)
            print("ENSEMBLE MODEL DETAILS" if is_ensemble else "MODEL DETAILS")
            print("=" * 60)
            print()
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

            metrics = result.get("metrics", {})
            if metrics:
                m = {
                    "accuracy_mean": metrics.get("accuracy"),
                    "log_loss_mean": metrics.get("log_loss"),
                    "brier_mean": metrics.get("brier_score"),
                    "auc": metrics.get("auc"),
                }
                print_metrics(m, "METRICS")

            if is_ensemble:
                base_models = result.get("base_models", [])
                if base_models:
                    print("BASE MODELS:")
                    print("-" * 40)
                    for i, bm in enumerate(base_models, 1):
                        print(f"  {i}. {bm.get('name', 'N/A')} ({bm.get('model_type', 'N/A')})")
                        print(f"     ID: {bm.get('id', 'N/A')}")
                        print(f"     Features: {len(bm.get('features', []))}")
                        bm_importances = bm.get("feature_importances", {})
                        if bm_importances:
                            print("     Top Features:")
                            sorted_f = sorted(bm_importances.items(), key=lambda x: x[1], reverse=True)[:5]
                            for feat, importance in sorted_f:
                                print(f"       - {feat}: {importance:.4f}")
                        print()
            else:
                imp = result.get("feature_importances", {})
                if imp:
                    n = len(imp) if args.all_features else min(args.top_features, len(imp))
                    print_feature_importances(imp, f"FEATURE IMPORTANCES (top {n})", top_n=n)
                features = result.get("features", [])
                if features:
                    print(f"FEATURES ({len(features)} total):")
                    print("-" * 40)
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
            return

        self.error("Use --list or --model <name_or_id>")