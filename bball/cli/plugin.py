"""Basketball plugin for the unified sportscore CLI."""

from typing import Dict, List

from sportscore.cli.discovery import SportCommand, SportPlugin


class FullDataPipelineCommand(SportCommand):
    name = "full_data_pipeline"
    help = "Full data pipeline (ESPN pull -> enrichment -> training -> registration)"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--max-workers", type=int, default=None,
                            help="Max parallel workers for ESPN pull")
        parser.add_argument("--seasons", type=str, default=None,
                            help="Comma-separated seasons (e.g., '2022-2023,2023-2024')")
        parser.add_argument("--skip-espn", action="store_true",
                            help="Skip ESPN data pull")
        parser.add_argument("--skip-training", action="store_true",
                            help="Skip master training generation")
        parser.add_argument("--skip-post", action="store_true",
                            help="Skip post-processing (venues, players)")
        parser.add_argument("--skip-injuries", action="store_true",
                            help="Skip injury computation")
        parser.add_argument("--skip-rosters", action="store_true",
                            help="Skip roster build")
        parser.add_argument("--skip-odds", action="store_true",
                            help="Skip odds backfill")
        parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be done without modifying data")
        parser.add_argument("--verbose", "-v", action="store_true",
                            help="Show detailed output")

    def run(self, args) -> int:
        from bball.pipeline.full_pipeline import main as fp_main

        # Build argv for the pipeline's own parser
        argv = [args.league]
        if args.max_workers:
            argv += ["--max-workers", str(args.max_workers)]
        if args.seasons:
            argv += ["--seasons", args.seasons]
        if args.skip_espn:
            argv.append("--skip-espn")
        if args.skip_training:
            argv.append("--skip-training")
        if args.skip_post:
            argv.append("--skip-post")
        if args.skip_injuries:
            argv.append("--skip-injuries")
        if args.skip_rosters:
            argv.append("--skip-rosters")
        if args.skip_odds:
            argv.append("--skip-odds")
        if args.dry_run:
            argv.append("--dry-run")
        if args.verbose:
            argv.append("--verbose")

        return fp_main(argv)


class GenerateTrainingDataCommand(SportCommand):
    name = "generate_training_data"
    help = "Generate master training CSV with features"

    def add_arguments(self, parser) -> None:
        parser.add_argument("--workers", type=int, default=None,
                            help="Number of parallel workers")
        parser.add_argument("--chunk-size", type=int, default=None,
                            help="Rows per chunk")
        parser.add_argument("--season", type=str, default=None,
                            help="Specific season (e.g., '2023-2024')")
        parser.add_argument("--seasons", type=str, default=None,
                            help="Comma-separated season range (e.g., '2023-2024,2024-2025')")
        parser.add_argument("--min-season", type=str, default=None,
                            help="Minimum season to include")
        parser.add_argument("--no-player", action="store_true",
                            help="Skip player-level features")
        parser.add_argument("--limit", type=int, default=None,
                            help="Limit number of games (testing)")
        parser.add_argument("--output", type=str, default=None,
                            help="Output CSV path")
        parser.add_argument("--dry-run", action="store_true",
                            help="Show what would be done without generating")
        parser.add_argument("--features", type=str, default=None,
                            help="Comma-separated feature names or patterns")
        parser.add_argument("--exclude-features", type=str, default=None,
                            help="Comma-separated features to exclude")
        parser.add_argument("--add", action="store_true",
                            help="Add/update to existing CSV")

    def run(self, args) -> int:
        from bball.pipeline.training_pipeline import main as tp_main

        # Build argv for the training pipeline's own parser
        argv = [args.league]
        if args.workers:
            argv += ["--workers", str(args.workers)]
        if args.chunk_size:
            argv += ["--chunk-size", str(args.chunk_size)]
        if args.season:
            argv += ["--season", args.season]
        if args.seasons:
            argv += ["--seasons", args.seasons]
        if args.min_season:
            argv += ["--min-season", args.min_season]
        if args.no_player:
            argv.append("--no-player")
        if args.limit:
            argv += ["--limit", str(args.limit)]
        if args.output:
            argv += ["--output", args.output]
        if args.dry_run:
            argv.append("--dry-run")
        if args.features:
            argv += ["--features", args.features]
        if args.exclude_features:
            argv += ["--exclude-features", args.exclude_features]
        if args.add:
            argv.append("--add")

        return tp_main(argv)


class BasketballPlugin(SportPlugin):
    def get_leagues(self) -> List[str]:
        from bball.league_config import get_available_leagues
        return get_available_leagues()

    def get_commands(self) -> Dict[str, SportCommand]:
        return {
            "full_data_pipeline": FullDataPipelineCommand(),
            "generate_training_data": GenerateTrainingDataCommand(),
        }

    def get_league_loader(self):
        from bball.league_config import load_league_config
        return load_league_config

    def get_db_factory(self):
        from bball.mongo import Mongo
        return lambda: Mongo().db
