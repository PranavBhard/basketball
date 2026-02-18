"""EnsembleCommand — basketball ensemble nba --models id1,id2 ..."""

import argparse
from sportscore.cli.base import BaseCommand, parse_list, print_metrics, print_feature_importances


class EnsembleCommand(BaseCommand):
    name = "ensemble"
    help = "Train a stacking ensemble from base model names/IDs"
    description = "Train an ensemble. Temporal config is derived from base models. Requires at least 2 base models."
    epilog = """
Examples:
  basketball ensemble nba --model LR --c-value 0.1 --models "model1,model2"
  basketball ensemble nba --model LR --models "id1,id2" --stacking-mode informed --use-disagree --use-conf
"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True, choices=["LR", "GB", "SVM"], help="Meta-model type")
        parser.add_argument("--c-value", type=float, default=0.1, help="Meta C-value (default: 0.1)")
        parser.add_argument(
            "--models",
            type=str,
            required=True,
            help="Comma-separated base model names or IDs",
        )
        parser.add_argument(
            "--extra-features",
            type=str,
            default=None,
            help="Comma-separated additional features for meta-model (e.g., pred_margin)",
        )
        parser.add_argument(
            "--stacking-mode",
            default="informed",
            choices=["naive", "informed"],
            help="Stacking mode (default: informed)",
        )
        parser.add_argument("--use-disagree", action="store_true", help="Include pairwise disagreement features")
        parser.add_argument("--use-conf", action="store_true", help="Include confidence features")

    def handle(self, args: argparse.Namespace, league, db) -> None:
        from bball.services.training_service import TrainingService

        base_models = parse_list(args.models)
        if len(base_models) < 2:
            self.error("At least 2 base models are required")

        extra_features = parse_list(args.extra_features) if args.extra_features else None

        service = TrainingService(league=league, db=db)
        result = service.train_ensemble(
            meta_model_type=args.model,
            base_model_names_or_ids=base_models,
            meta_c_value=args.c_value,
            extra_features=extra_features,
            stacking_mode=args.stacking_mode,
            use_disagree=args.use_disagree,
            use_conf=args.use_conf,
        )

        print("=" * 60)
        print("ENSEMBLE TRAINING COMPLETE")
        print("=" * 60)
        print(f"Config ID: {result.get('config_id', 'N/A')}")
        print(f"Run ID: {result.get('run_id', 'N/A')}")
        print()

        metrics = result.get("metrics", {})
        if metrics:
            m = {
                "accuracy_mean": metrics.get("accuracy_mean"),
                "log_loss_mean": metrics.get("log_loss_mean"),
                "brier_mean": metrics.get("brier_mean"),
                "auc": metrics.get("auc_mean") or metrics.get("auc"),
            }
            print_metrics(m, "ENSEMBLE METRICS")

        base_models_info = result.get("base_models", [])
        if base_models_info:
            print("BASE MODELS:")
            print("-" * 40)
            for i, bm in enumerate(base_models_info, 1):
                print(f"  {i}. {bm.get('name', 'N/A')} ({bm.get('model_type', 'N/A')})")
                print(f"     ID: {bm.get('id', 'N/A')}")
            print()

        diagnostics = result.get("diagnostics", {})
        base_summaries = diagnostics.get("base_models_summary", [])
        if base_summaries:
            print("BASE MODEL PERFORMANCE (on evaluation set):")
            print("-" * 40)
            for i, bm in enumerate(base_summaries, 1):
                bm_metrics = bm.get("metrics", {})
                accuracy = bm_metrics.get("accuracy_mean", "N/A")
                if isinstance(accuracy, (int, float)):
                    accuracy = f"{accuracy:.2f}%"
                print(f"  {i}. {bm.get('run_id', 'N/A')[:8]}...")
                print(f"     Accuracy: {accuracy}")
                print(f"     Features: {len(bm.get('feature_names', []))}")
            print()

        meta_importances = diagnostics.get("meta_feature_importances", {})
        if meta_importances:
            print_feature_importances(meta_importances, "META-MODEL FEATURE IMPORTANCES", top_n=50)
