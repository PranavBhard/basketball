"""PredictCommand — basketball predict nba --date 2024-03-15 [--home LAL --away BOS] [--save]"""

import argparse
from datetime import datetime
from sportscore.cli.base import BaseCommand, format_table


class PredictCommand(BaseCommand):
    name = "predict"
    help = "Run predictions for a date or single matchup"
    description = "Full date: predict all games. Single matchup: use --home and --away with --date. Optionally --save to persist."
    epilog = """
Examples:
  basketball predict nba --date 2024-03-15
  basketball predict nba --date 2024-03-15 --home LAL --away BOS
  basketball predict nba --date 2024-03-15 --save
"""

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--date", type=str, required=True, help="Game date (YYYY-MM-DD)")
        parser.add_argument("--home", type=str, default=None, help="Home team abbrev (single matchup)")
        parser.add_argument("--away", type=str, default=None, help="Away team abbrev (single matchup)")
        parser.add_argument("--save", action="store_true", help="Save predictions to DB")

    def handle(self, args: argparse.Namespace, league, db) -> None:
        from bball.services.prediction import PredictionService

        service = PredictionService(db=db, league=league)

        if args.home and args.away:
            result = service.predict_game(
                home_team=args.home,
                away_team=args.away,
                game_date=args.date,
            )
            if result.error:
                self.error(result.error)
            print(f"  {result.away_team} @ {result.home_team}  ({result.game_date})")
            print(f"  Winner: {result.predicted_winner}  |  Home: {result.home_win_prob:.1f}%  Away: {result.away_win_prob:.1f}%")
            print(f"  Odds: Home {result.home_odds}  Away {result.away_odds}")
            if result.home_points_pred is not None and result.away_points_pred is not None:
                print(f"  Points: {result.home_points_pred} - {result.away_points_pred}")
            if args.save and result.game_id:
                game_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
                service.save_prediction(result, result.game_id, game_date_obj, args.home, args.away)
                print("  Saved.")
            return

        results = service.predict_date(args.date)
        if not results:
            print("No games found for this date.")
            return

        rows = []
        for r in results:
            if r.error:
                rows.append([r.home_team, r.away_team, "ERROR", r.error[:40]])
            else:
                pts = f"{r.home_points_pred or '-'}-{r.away_points_pred or '-'}" if (r.home_points_pred is not None or r.away_points_pred is not None) else "-"
                rows.append([
                    r.home_team,
                    r.away_team,
                    r.predicted_winner,
                    f"{r.home_win_prob:.0f}%",
                    f"{r.away_win_prob:.0f}%",
                    f"{r.home_odds}/{r.away_odds}",
                    pts,
                ])
            if args.save and r.game_id and not r.error:
                game_date_obj = datetime.strptime(args.date, "%Y-%m-%d").date()
                service.save_prediction(r, r.game_id, game_date_obj, r.home_team, r.away_team)
        headers = ["Home", "Away", "Winner", "Home%", "Away%", "Odds", "Points"]
        print(format_table(headers, rows))
        if args.save:
            print("\nPredictions saved to DB.")