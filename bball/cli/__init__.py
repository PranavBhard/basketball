"""Basketball CLI — subcommand entry point using sportscore SportsCLI."""

from sportscore.cli.base import SportsCLI


def create_cli() -> SportsCLI:
    from bball.league_config import load_league_config, get_available_leagues
    from bball.mongo import Mongo
    from bball.cli.commands.train import TrainCommand
    from bball.cli.commands.ensemble import EnsembleCommand
    from bball.cli.commands.models import ModelsCommand
    from bball.cli.commands.predict import PredictCommand

    cli = SportsCLI(
        prog="basketball",
        description="Basketball analytics and prediction platform",
        league_loader=load_league_config,
        db_factory=lambda: Mongo().db,
        available_leagues=get_available_leagues,
    )
    cli.register(TrainCommand())
    cli.register(EnsembleCommand())
    cli.register(ModelsCommand())
    cli.register(PredictCommand())
    return cli


def main() -> None:
    create_cli().run()
