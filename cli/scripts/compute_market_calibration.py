#!/usr/bin/env python
"""
Compute Market Calibration CLI Script

Reads the master training CSV, computes per-season market Brier score and
log-loss from historical vegas implied probabilities, and stores results
in the market_calibration MongoDB collection.

Usage:
    python cli/scripts/compute_market_calibration.py <league>
    python cli/scripts/compute_market_calibration.py nba --rolling-seasons 3
    python cli/scripts/compute_market_calibration.py cbb --min-coverage 0.50
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Compute market calibration (Brier/log-loss) from historical odds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Compute market calibration for NBA:
    %(prog)s nba

  Compute with 5-season rolling window:
    %(prog)s nba --rolling-seasons 5

  Compute for CBB with higher coverage threshold:
    %(prog)s cbb --min-coverage 0.60
        """
    )

    parser.add_argument('league', help='League ID (e.g., nba, cbb)')
    parser.add_argument('--rolling-seasons', type=int, default=3,
                        help='Number of recent seasons for rolling aggregate (default: 3)')
    parser.add_argument('--min-coverage', type=float, default=0.50,
                        help='Minimum odds coverage ratio to include a season (default: 0.50)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print results without writing to MongoDB')

    args = parser.parse_args()

    # Import here to avoid slow imports on --help
    from bball.league_config import load_league_config
    from bball.mongo import Mongo
    from bball.services.market_calibration_service import compute_and_store_market_calibration

    # Load league config
    league = load_league_config(args.league)

    print(f"League: {league.display_name} ({league.league_id})")
    print(f"Master training CSV: {league.master_training_csv}")
    print()

    db = Mongo().db

    try:
        stats = compute_and_store_market_calibration(
            db, league,
            rolling_seasons=args.rolling_seasons,
            min_coverage=args.min_coverage,
            dry_run=args.dry_run,
            verbose=True,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print()
    print(f"Seasons computed: {stats['seasons_computed']}")
    if stats['rolling_brier'] is not None:
        print(f"Rolling Brier:    {stats['rolling_brier']:.4f}")
        print(f"Rolling Log-Loss: {stats['rolling_log_loss']:.4f}")
    if stats['n_written']:
        print(f"Docs written:     {stats['n_written']} to {stats['collection_name']}")


if __name__ == '__main__':
    main()
