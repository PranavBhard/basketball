#!/usr/bin/env python3
"""
Cache League Stats for PER Calculation - Core Infrastructure

Computes and caches per-season league constants required for PER calculation:
- factor, VOP, DRB% (league-level constants)
- lg_pace (league average pace)
- team_pace (per-team average pace)

Stores results in MongoDB collection `cached_league_stats`.
"""

import argparse
from datetime import datetime
from typing import Dict, List, Optional
from bball_app.core.mongo import Mongo
from bball_app.core.league_config import load_league_config


# Defaults (NBA). Multi-league callers should pass a league config.
DEFAULT_CACHE_COLLECTION = 'cached_league_stats'
DEFAULT_GAMES_COLLECTION = 'stats_nba'

# Backward-compatible constant used throughout this module.
COLLECTION_NAME = DEFAULT_CACHE_COLLECTION


def get_db():
    """Get MongoDB database connection."""
    mongo = Mongo()
    return mongo.db


def get_all_seasons(db, league=None) -> List[str]:
    """Get all unique seasons from the league's games collection."""
    league = league or load_league_config("nba")
    games_coll = league.collections["games"] if league else DEFAULT_GAMES_COLLECTION
    seasons = db[games_coll].distinct('season', {
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': (league.exclude_game_types if league else ['preseason', 'allstar'])}
    })
    return sorted(seasons)


def compute_team_possessions(team: dict) -> float:
    """
    Compute possessions for a team in a single game.
    Formula: FGA - OREB + TO + 0.44 * FTA
    """
    fga = team.get('FG_att', 0)
    oreb = team.get('off_reb', 0)
    to = team.get('TO', 0)
    fta = team.get('FT_att', 0)
    return fga - oreb + to + 0.44 * fta


def compute_season_stats(db, season: str, as_of_date: Optional[str] = None, league=None) -> Optional[Dict]:
    """
    Compute all league and team stats for a given season.

    Returns dict with:
    - league_totals: aggregated stats across all teams/games
    - league_constants: factor, VOP, DRB%
    - lg_pace: league average possessions per game
    - team_pace: dict of team_name -> avg possessions per game
    - team_games: dict of team_name -> games played
    - computed_at: timestamp
    """
    league = league or load_league_config("nba")
    games_coll = league.collections["games"] if league else DEFAULT_GAMES_COLLECTION
    query = {
        'season': season,
        'homeTeam.points': {'$gt': 0},
        'awayTeam.points': {'$gt': 0},
        'game_type': {'$nin': (league.exclude_game_types if league else ['preseason', 'allstar'])}
    }
    if as_of_date:
        # Date is stored as YYYY-MM-DD string in stats_nba; string ordering matches chronological ordering.
        query['date'] = {'$lte': as_of_date}

    games = list(db[games_coll].find(query))

    if not games:
        print(f"  No games found for season {season}")
        return None

    # Initialize accumulators
    lg_totals = {
        'AST': 0, 'FG': 0, 'FGA': 0, 'FT': 0, 'FTA': 0,
        'PTS': 0, 'TRB': 0, 'ORB': 0, 'TOV': 0, 'PF': 0,
        'possessions': 0, 'team_games': 0
    }

    team_possessions_sum = {}
    team_games_count = {}

    for game in games:
        for side in ['homeTeam', 'awayTeam']:
            team = game.get(side, {})
            team_name = team.get('name', '')

            if not team_name:
                continue

            # Compute possessions for this team-game
            poss = compute_team_possessions(team)

            # Aggregate league totals
            lg_totals['AST'] += team.get('assists', 0)
            lg_totals['FG'] += team.get('FG_made', 0)
            lg_totals['FGA'] += team.get('FG_att', 0)
            lg_totals['FT'] += team.get('FT_made', 0)
            lg_totals['FTA'] += team.get('FT_att', 0)
            lg_totals['PTS'] += team.get('points', 0)
            lg_totals['TRB'] += team.get('total_reb', 0)
            lg_totals['ORB'] += team.get('off_reb', 0)
            lg_totals['TOV'] += team.get('TO', 0)
            lg_totals['PF'] += team.get('PF', 0)
            lg_totals['possessions'] += poss
            lg_totals['team_games'] += 1

            # Accumulate per-team stats
            if team_name not in team_possessions_sum:
                team_possessions_sum[team_name] = 0
                team_games_count[team_name] = 0

            team_possessions_sum[team_name] += poss
            team_games_count[team_name] += 1

    # Compute league constants
    # Avoid division by zero
    if lg_totals['FG'] == 0 or lg_totals['FT'] == 0 or lg_totals['TRB'] == 0:
        print(f"  Warning: Zero denominator in league stats for {season}")
        return None

    # factor = (2/3) - (0.5 * (lg_AST / lg_FG)) / (2 * (lg_FG / lg_FT))
    ast_fg_ratio = lg_totals['AST'] / lg_totals['FG']
    fg_ft_ratio = lg_totals['FG'] / lg_totals['FT']
    factor = (2 / 3) - (0.5 * ast_fg_ratio) / (2 * fg_ft_ratio)

    # VOP = lg_PTS / (lg_FGA - lg_ORB + lg_TOV + 0.44 * lg_FTA)
    vop_denom = lg_totals['FGA'] - lg_totals['ORB'] + lg_totals['TOV'] + 0.44 * lg_totals['FTA']
    if vop_denom == 0:
        print(f"  Warning: VOP denominator is zero for {season}")
        return None
    VOP = lg_totals['PTS'] / vop_denom

    # DRB% = (lg_TRB - lg_ORB) / lg_TRB
    DRB_pct = (lg_totals['TRB'] - lg_totals['ORB']) / lg_totals['TRB']

    # League pace (average possessions per team-game)
    lg_pace = lg_totals['possessions'] / lg_totals['team_games'] if lg_totals['team_games'] > 0 else 0

    # Per-team pace
    team_pace = {}
    for team_name in team_possessions_sum:
        if team_games_count[team_name] > 0:
            team_pace[team_name] = team_possessions_sum[team_name] / team_games_count[team_name]
        else:
            team_pace[team_name] = lg_pace  # fallback to league average

    result = {
        'season': season,
        'league_totals': lg_totals,
        'league_constants': {
            'factor': factor,
            'VOP': VOP,
            'DRB_pct': DRB_pct
        },
        'lg_pace': lg_pace,
        'team_pace': team_pace,
        'team_games': team_games_count,
        'game_count': len(games),
        'computed_at': datetime.now().isoformat(),
        'version': 2  # v2 supports optional as_of_date snapshots
    }
    if as_of_date:
        result['as_of_date'] = as_of_date
    return result


def cache_season(db, season: str, force: bool = False, as_of_date: Optional[str] = None, league=None) -> bool:
    """
    Compute and cache stats for a single season in MongoDB.
    Returns True if successful.
    """
    league = league or load_league_config("nba")
    cache_coll = league.collections["cached_league_stats"] if league else DEFAULT_CACHE_COLLECTION
    collection = db[cache_coll]

    # Check if already cached (base season doc vs as-of snapshot docs)
    query_key = {'season': season, 'as_of_date': as_of_date} if as_of_date else {'season': season, 'as_of_date': {'$exists': False}}
    existing = collection.find_one(query_key)
    if existing and not force:
        msg = f"  Season {season} already cached"
        if as_of_date:
            msg += f" (as_of_date={as_of_date})"
        msg += " (use --force to recalculate)"
        print(msg)
        return True

    print(f"  Computing stats for {season}" + (f" (as_of_date={as_of_date})" if as_of_date else "") + "...")
    stats = compute_season_stats(db, season, as_of_date=as_of_date, league=league)

    if stats:
        # Upsert into MongoDB
        collection.update_one(
            query_key,
            {'$set': stats},
            upsert=True
        )
        print(f"    Cached {stats['game_count']} games, {len(stats['team_pace'])} teams")
        print(f"    factor={stats['league_constants']['factor']:.4f}, "
              f"VOP={stats['league_constants']['VOP']:.4f}, "
              f"DRB%={stats['league_constants']['DRB_pct']:.4f}")
        print(f"    lg_pace={stats['lg_pace']:.2f} possessions/game")
        return True
    return False


def list_cached_seasons(db, league=None):
    """List all cached seasons with summary info."""
    league = league or load_league_config("nba")
    cache_coll = league.collections["cached_league_stats"] if league else DEFAULT_CACHE_COLLECTION
    collection = db[cache_coll]
    cached = list(collection.find({}).sort('season', 1))

    if not cached:
        print("No seasons cached yet.")
        return

    print(f"\nCached seasons in '{cache_coll}' collection:\n")
    print(f"{'Season':<12} {'Games':>8} {'Teams':>8} {'lg_pace':>10} {'VOP':>8} {'Computed At':<20}")
    print("-" * 70)

    for s in cached:
        print(f"{s['season']:<12} {s['game_count']:>8} {len(s['team_pace']):>8} "
              f"{s['lg_pace']:>10.2f} {s['league_constants']['VOP']:>8.4f} "
              f"{s['computed_at'][:19]:<20}")


# =============================================================================
# UTILITY FUNCTIONS FOR USE BY OTHER MODULES
# =============================================================================

def get_season_stats(season: str, db=None, as_of_date: Optional[str] = None, league=None) -> Optional[Dict]:
    """
    Retrieve cached stats for a season from MongoDB.

    Args:
        season: Season string (e.g., '2024-2025')
        db: Optional database connection (creates new if None)

    Returns:
        Dict with league stats, or None if not cached
    """
    if db is None:
        db = get_db()
    league = league or load_league_config("nba")
    cache_coll = league.collections["cached_league_stats"] if league else DEFAULT_CACHE_COLLECTION

    # IMPORTANT: keep a single canonical per-season doc for consumers (no as_of_date field).
    # as_of_date snapshots are stored as separate docs and only returned if explicitly requested.
    if as_of_date:
        return db[cache_coll].find_one({'season': season, 'as_of_date': as_of_date})
    return db[cache_coll].find_one({'season': season, 'as_of_date': {'$exists': False}})


def get_previous_season(season: str) -> Optional[str]:
    """
    Get the previous season string.
    E.g., '2024-2025' -> '2023-2024'
    """
    try:
        start_year = int(season.split('-')[0])
        return f"{start_year - 1}-{start_year}"
    except (ValueError, IndexError):
        return None


def get_season_stats_with_fallback(season: str, db=None, league=None) -> Optional[Dict]:
    """
    Get season stats, falling back to previous season if not available.
    Useful for early-season predictions.

    Args:
        season: Season string (e.g., '2024-2025')
        db: Optional database connection

    Returns:
        Dict with league stats (may be from previous season)
    """
    if db is None:
        db = get_db()

    league = league or load_league_config("nba")
    stats = get_season_stats(season, db, as_of_date=None, league=league)
    if stats:
        return stats

    # Try previous season as fallback
    prev_season = get_previous_season(season)
    if prev_season:
        stats = get_season_stats(prev_season, db, as_of_date=None, league=league)
        if stats:
            print(f"  Using {prev_season} stats as fallback for {season}")
            # Mark as fallback
            stats['is_fallback'] = True
            stats['original_season'] = prev_season
            return stats

    return None


def get_league_constants(season: str, db=None, league=None) -> Optional[Dict]:
    """
    Get just the league constants (factor, VOP, DRB_pct) for a season.

    Args:
        season: Season string
        db: Optional database connection

    Returns:
        Dict with factor, VOP, DRB_pct, lg_pace
    """
    stats = get_season_stats_with_fallback(season, db, league=league)
    if not stats:
        return None

    return {
        'factor': stats['league_constants']['factor'],
        'VOP': stats['league_constants']['VOP'],
        'DRB_pct': stats['league_constants']['DRB_pct'],
        'lg_pace': stats['lg_pace'],
        'lg_FT': stats['league_totals']['FT'],
        'lg_FTA': stats['league_totals']['FTA'],
        'lg_PF': stats['league_totals']['PF'],
        'season': stats['season'],
        'is_fallback': stats.get('is_fallback', False)
    }


def get_team_pace(season: str, team: str, db=None, league=None) -> float:
    """
    Get pace for a specific team in a season.

    Args:
        season: Season string
        team: Team name
        db: Optional database connection

    Returns:
        Team's average possessions per game, or league average if not found
    """
    stats = get_season_stats_with_fallback(season, db, league=league)
    if not stats:
        return 95.0  # Default NBA pace if no data

    return stats['team_pace'].get(team, stats['lg_pace'])


def ensure_season_cached(season: str, db=None, league=None) -> bool:
    """
    Ensure a season is cached. Computes if not already cached.

    Args:
        season: Season string
        db: Optional database connection

    Returns:
        True if season is now cached, False on failure
    """
    if db is None:
        db = get_db()
    league = league or load_league_config("nba")

    existing = get_season_stats(season, db, league=league)
    if existing:
        return True

    # Try to compute and cache
    print(f"Season {season} not cached, computing...")
    return cache_season(db, season, force=False, league=league)


def main():
    """CLI entry point for cache_league_stats."""
    parser = argparse.ArgumentParser(
        description='Cache league stats for PER calculation in MongoDB',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bball_app.core.cache_league_stats                     # Cache all seasons
  python -m bball_app.core.cache_league_stats --season 2024-2025  # Cache specific season
  python -m bball_app.core.cache_league_stats --list              # List cached seasons
  python -m bball_app.core.cache_league_stats --force             # Force recalculation
        """
    )

    parser.add_argument(
        '--season', '-s',
        type=str,
        help='Specific season to cache (e.g., 2024-2025)'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all cached seasons'
    )

    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force recalculation even if already cached'
    )

    args = parser.parse_args()

    # Connect to MongoDB
    print("Connecting to MongoDB...")
    db = get_db()

    if args.list:
        list_cached_seasons(db)
        return

    if args.season:
        # Cache specific season
        print(f"\nCaching season {args.season}...")
        success = cache_season(db, args.season, force=args.force)
        if success:
            print(f"\nSeason {args.season} cached successfully.")
    else:
        # Cache all seasons
        print("\nDiscovering seasons...")
        seasons = get_all_seasons(db)
        print(f"Found {len(seasons)} seasons: {', '.join(seasons)}\n")

        cached_count = 0
        for season in seasons:
            if cache_season(db, season, force=args.force):
                cached_count += 1

        print(f"\nCached {cached_count} seasons total.")

    # Print summary
    list_cached_seasons(db)


if __name__ == '__main__':
    main()
