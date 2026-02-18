"""TrainCommand — basketball train nba --features pts_per_game,ast_per_game ..."""

import argparse
from sportscore.cli.base import BaseCommand, parse_list, parse_seasons, print_metrics, print_feature_importances


class TrainCommand(BaseCommand):
    name = "train"
    help = "Train a base classifier model"
    description = "Train a base classifier using explicit features or feature sets. Uses TrainingService.train_base_model()."
    epilog = """
Examples:
  basketball train nba --model LR --c-value 0.1 \\
    --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \\
    --calibration-seasons 2023 --evaluation-season 2024 \\
    --features "points|season|avg|diff,assists|season|avg|diff" --name "test_lr"
  basketball train nba --model LR --feature-sets pts_per_game,ast_per_game \\
    --train-seasons 2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022 \\
    --calibration-seasons 2023 --evaluation-season 2024
"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", required=True, choices=["LR", "GB", "SVM"], help="Model type: LR, GB, SVM")
        parser.add_argument("--c-value", type=float, default=0.1, help="Regularization C (default: 0.1)")
        parser.add_argument(
            "--calibration-method",
            default="sigmoid",
            choices=["sigmoid", "isotonic"],
            help="Calibration method (default: sigmoid)",
        )
        parser.add_argument(
            "--train-seasons",
            required=True,
            help="Comma-separated training seasons (e.g., 2012,2013,...,2022)",
        )
        parser.add_argument(
            "--calibration-seasons",
            required=True,
            help="Comma-separated calibration seasons (e.g., 2023)",
        )
        parser.add_argument(
            "--evaluation-season",
            required=True,
            type=int,
            help="Evaluation season year (e.g., 2024)",
        )
        parser.add_argument("--features", type=str, default=None, help="Comma-separated feature names")
        parser.add_argument(
            "--feature-sets",
            type=str,
            default=None,
            help="Comma-separated feature set names (resolved via get_features_by_sets)",
        )
        parser.add_argument("--name", default=None, help="Model name (auto-generated if not provided)")
        parser.add_argument("--min-games", type=int, default=20, help="Minimum games played filter (default: 20)")
        parser.add_argument("--include-injuries", action="store_true", help="Include injury features")
        parser.add_argument("--no-master", action="store_true", help="Do not use master training CSV")

    def handle(self, args: argparse.Namespace, league, db) -> None:
        from bball.services.training_service import TrainingService
        from bball.features.sets import get_features_by_sets

        features = []
        if args.features:
            features.extend(parse_list(args.features))
        if args.feature_sets:
            sets_list = parse_list(args.feature_sets)
            features.extend(get_features_by_sets(sets_list))
        features = list(dict.fromkeys(features))  # dedupe, preserve order

        if not features:
            self.error("At least one of --features or --feature-sets is required")

        train_seasons = parse_seasons(args.train_seasons)
        calibration_seasons = parse_seasons(args.calibration_seasons)
        if not train_seasons:
            self.error("--train-seasons is required and must be non-empty")
        if not calibration_seasons:
            self.error("--calibration-seasons is required and must be non-empty")

        service = TrainingService(league=league, db=db)
        result = service.train_base_model(
            model_type=args.model,
            features=features,
            train_seasons=train_seasons,
            calibration_seasons=calibration_seasons,
            evaluation_season=args.evaluation_season,
            c_value=args.c_value,
            calibration_method=args.calibration_method,
            name=args.name,
            min_games_played=args.min_games,
            include_injuries=args.include_injuries,
            use_master=not args.no_master,
        )

        print("=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Config ID: {result['config_id']}")
        print(f"Run ID: {result['run_id']}")
        print(f"Samples: {result.get('n_samples', 'N/A')}")
        print(f"Features: {result.get('n_features', 'N/A')}")
        metrics = result.get("metrics", {})
        if metrics:
            m = {
                "accuracy_mean": metrics.get("accuracy_mean"),
                "log_loss_mean": metrics.get("log_loss_mean"),
                "brier_mean": metrics.get("brier_mean"),
                "auc": metrics.get("auc_mean") or metrics.get("auc"),
            }
            print_metrics(m, "METRICS")
        if result.get("f_scores"):
            print_feature_importances(
                {k: v for k, v in result["f_scores"].items()},
                "F-SCORES (ANOVA)",
                top_n=20,
            )
        if result.get("feature_importances"):
            print_feature_importances(result["feature_importances"], "FEATURE IMPORTANCES", top_n=20)
        print(f"Model saved. Use 'basketball models {args.league} --model {result['config_id']}' to view details.")
