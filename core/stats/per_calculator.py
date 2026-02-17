#!/usr/bin/env python3
"""
PER Calculator - Player Efficiency Rating computation module

Implements Hollinger's PER formula:
1. uPER (unadjusted PER) - raw per-minute efficiency
2. aPER (pace-adjusted PER) - adjusted for team pace
3. PER (normalized) - scaled so league average = 15

Optimizations:
- Preloads stats_nba_players into memory to avoid 27k+ DB queries
- Caches computed PER features per game to MongoDB

Usage:
    from per_calculator import PERCalculator
    
    # With preloading (fast for batch processing)
    calc = PERCalculator(preload=True)
    
    # Without preloading (for single queries)
    calc = PERCalculator(preload=False)
    
    # Get team PER aggregates for game prediction
    features = calc.get_game_per_features(home_team, away_team, season, game_date)
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
import numpy as np

from bball_app.core.mongo import Mongo
from typing import TYPE_CHECKING
from bball_app.core.data import (
    GamesRepository, PlayerStatsRepository, PlayersRepository,
    RostersRepository, LeagueStatsCache
)
from bball_app.core.stats.league_cache import (
    get_season_stats_with_fallback,
    get_league_constants,
    get_team_pace,
    ensure_season_cached
)

if TYPE_CHECKING:
    from bball_app.core.league_config import LeagueConfig

# MongoDB collection for cached PER features
CACHED_PER_COLLECTION = 'cached_per_features'

# =============================================================================
# PLAYER SUBSET CONSTANTS (from documentation/player_feature_updates.md)
# =============================================================================
# These constants define how player subsets are filtered for feature calculations

MPG_THRESH = 10        # Minimum minutes per game to be considered "rotation" quality
GP_FLOOR = 10          # Minimum games played threshold floor
GP_RATIO = 0.15        # Games played threshold as ratio of team games
ROTATION_SIZE = 10     # Maximum players in rotation subset
EPS = 1e-6            # Small constant to avoid division by zero


class PERCalculator:
    """
    Calculate Player Efficiency Rating (PER) using Hollinger's formula.
    
    PER measures a player's per-minute productivity, normalized so that
    the league average is always 15.0.
    
    Supports two modes:
    - preload=True: Loads all player stats into memory for fast batch processing
    - preload=False: Queries DB on demand (slower but uses less memory)
    """
    
    def __init__(self, db=None, preload: bool = True, league: Optional["LeagueConfig"] = None, preload_seasons: list = None):
        """
        Initialize PER Calculator.

        Args:
            db: Optional MongoDB database connection. Creates new if None.
            preload: If True, preload stats_nba_players into memory for fast access
            league: League configuration object
            preload_seasons: Optional list of seasons to preload (e.g., ['2024-2025', '2023-2024']).
                           If None and preload=True, loads all seasons.
        """
        self._preload_seasons = preload_seasons

        if db is None:
            mongo = Mongo()
            self.db = mongo.db
        else:
            self.db = db

        self.league = league

        # Initialize repositories (league-aware)
        self._games_repo = GamesRepository(self.db, league=league)
        self._players_repo = PlayerStatsRepository(self.db, league=league)
        self._players_dir_repo = PlayersRepository(self.db, league=league)
        self._rosters_repo = RostersRepository(self.db, league=league)
        self._league_cache_repo = LeagueStatsCache(self.db, league=league)

        # Cache for computed values
        self._league_cache = {}
        self._team_pace_cache = {}
        self._player_per_cache = {}  # {(player_id, team, season, before_date): PER_value} - cache computed PER values
        self._league_aper_cache = {}  # {(season, before_date): lg_aper} - cache league average aPER
        self._lg_aper_by_season = {}  # {season: lg_aper} - season-level cache for performance
        
        # Preloaded data caches
        self._player_stats_cache = None  # {(team, season): [player_games sorted by date]}
        self._team_stats_cache = None    # {(team, season): [game_stats sorted by date]}
        self._per_features_cache = {}    # {game_key: features} - in-memory cache
        self._team_players_agg_cache = {}  # {(team, season, before_date): [aggregated player data]} - cache aggregated results

        # Cross-team caches for training (traded player support)
        self._player_stats_by_player = None  # {(player_id, season): [player_games]} - for cross-team aggregation
        self._game_to_players_cache = None   # {game_id: {team: [player_ids]}} - maps game to actual participants
        
        self._preloaded = False
        if preload:
            self._preload_data()

    @property
    def _exclude_game_types(self) -> list:
        """Get excluded game types from league config, with fallback."""
        return self.league.exclude_game_types if self.league else ['preseason', 'allstar']

    def _preload_data(self):
        """
        Preload stats_nba_players and stats_nba into memory for fast access.
        This converts 27k+ DB queries into a single load + in-memory filtering.

        If preload_seasons is set, only loads those seasons (faster for web UI).
        """
        seasons_desc = f"seasons {self._preload_seasons}" if self._preload_seasons else "all seasons"
        print(f"  Preloading player stats into memory ({seasons_desc})...")

        # Build query - optionally filter by season, exclude preseason/allstar
        query = {
            'stats.min': {'$gt': 0},
            'game_type': {'$nin': self._exclude_game_types}
        }
        if self._preload_seasons:
            query['season'] = {'$in': self._preload_seasons}

        # Load player stats with relevant fields (no sort - do it in Python to avoid MongoDB memory limit)
        player_stats = self._players_repo.find(
            query,
            projection={
                'player_id': 1,
                'player_name': 1,
                'game_id': 1,
                'date': 1,
                'season': 1,
                'team': 1,
                'home': 1,
                'starter': 1,
                'stats': 1
            }
        )
        
        print(f"    Loaded {len(player_stats)} player-game records")
        
        # Sort in Python (faster and avoids MongoDB 32MB sort limit)
        player_stats.sort(key=lambda x: x.get('date', ''))
        
        # Index by (team, season) for fast lookup
        self._player_stats_cache = defaultdict(list)
        # Also build cross-team indices for training (traded player support)
        self._player_stats_by_player = defaultdict(list)
        self._game_to_players_cache = defaultdict(lambda: defaultdict(list))

        for ps in player_stats:
            team = ps.get('team')
            season = ps.get('season')
            key = (team, season)
            self._player_stats_cache[key].append(ps)

            # Index by (player_id, season) for cross-team lookups
            player_id = str(ps.get('player_id'))
            player_key = (player_id, season)
            self._player_stats_by_player[player_key].append(ps)

            # Index by game_id -> team -> [player_ids] for finding game participants
            game_id = ps.get('game_id')
            if game_id and team:
                self._game_to_players_cache[game_id][team].append(player_id)

        print(f"    Indexed into {len(self._player_stats_cache)} team-season combinations")
        print(f"    Cross-team indices: {len(self._player_stats_by_player)} player-season keys, {len(self._game_to_players_cache)} games")

        # Load team stats from stats_nba (no sort - do it in Python)
        print("  Preloading team stats into memory...")
        games_query = {
            'homeTeam.points': {'$gt': 0},
            'awayTeam.points': {'$gt': 0},
            'game_type': {'$nin': self._exclude_game_types}
        }
        if self._preload_seasons:
            games_query['season'] = {'$in': self._preload_seasons}

        games = self._games_repo.find(
            games_query,
            projection={
                'game_id': 1,
                'date': 1,
                'season': 1,
                'homeTeam': 1,
                'awayTeam': 1
            }
        )

        print(f"    Loaded {len(games)} games")
        
        # Sort in Python (avoids MongoDB 32MB sort limit)
        games.sort(key=lambda x: x.get('date', ''))
        
        # Index by (team, season) for fast lookup
        self._team_stats_cache = defaultdict(list)
        for game in games:
            home_team = game['homeTeam']['name']
            away_team = game['awayTeam']['name']
            season = game.get('season')
            
            self._team_stats_cache[(home_team, season)].append({
                'game_id': game.get('game_id'),
                'date': game['date'],
                'team_data': game['homeTeam'],
                'is_home': True
            })
            self._team_stats_cache[(away_team, season)].append({
                'game_id': game.get('game_id'),
                'date': game['date'],
                'team_data': game['awayTeam'],
                'is_home': False
            })
        
        print(f"    Indexed into {len(self._team_stats_cache)} team-season combinations")
        
        # Load cached PER features from MongoDB
        self._load_per_cache()
        
        self._preloaded = True
        print("  Preloading complete!")
    
    def _load_per_cache(self):
        """Load cached PER features from MongoDB into memory."""
        cached_docs = list(self.db[CACHED_PER_COLLECTION].find({}))
        
        for doc in cached_docs:
            season = doc.get('season')
            game_features = doc.get('game_features', {})
            for game_key, features in game_features.items():
                self._per_features_cache[f"{season}|{game_key}"] = features
        
        print(f"    Loaded {len(self._per_features_cache)} cached PER feature sets")
    
    def _save_per_cache(self, season: str, game_features: dict):
        """Save computed PER features to MongoDB cache."""
        self.db[CACHED_PER_COLLECTION].update_one(
            {'season': season},
            {
                '$set': {
                    'season': season,
                    'game_features': game_features,
                    'game_count': len(game_features),
                    'updated_at': datetime.utcnow()
                }
            },
            upsert=True
        )
    
    # =========================================================================
    # CORE PER FORMULAS
    # =========================================================================
    
    def compute_uper(
        self,
        player_stats: dict,
        team_stats: dict,
        league_constants: dict
    ) -> float:
        """
        Compute unadjusted PER (uPER) for a single player-game.
        
        This is the raw per-minute efficiency rating before pace adjustment
        and league normalization.
        """
        # Extract player stats
        MP = player_stats.get('min', 0)
        if MP == 0:
            return 0
        
        # Player counting stats
        FG = player_stats.get('fg_made', 0)
        FGA = player_stats.get('fg_att', 0)
        THREE = player_stats.get('three_made', 0)
        FT = player_stats.get('ft_made', 0)
        FTA = player_stats.get('ft_att', 0)
        TRB = player_stats.get('reb', 0)
        ORB = player_stats.get('oreb', 0)
        AST = player_stats.get('ast', 0)
        STL = player_stats.get('stl', 0)
        BLK = player_stats.get('blk', 0)
        TOV = player_stats.get('to', 0)
        PF = player_stats.get('pf', 0)
        
        # Team stats
        team_AST = team_stats.get('assists', 1)
        team_FG = team_stats.get('FG_made', 1)
        
        # Avoid division by zero
        if team_FG == 0:
            team_FG = 1
        
        # League constants
        factor = league_constants.get('factor', 0.5)
        VOP = league_constants.get('VOP', 1.0)
        DRB_pct = league_constants.get('DRB_pct', 0.75)
        lg_FT = league_constants.get('lg_FT', 1)
        lg_PF = league_constants.get('lg_PF', 1)
        lg_FTA = league_constants.get('lg_FTA', 1)
        
        # Compute team AST ratio
        team_ast_ratio = team_AST / team_FG
        
        try:
            # Positive contributions
            three_pts = THREE
            ast_term = (2/3) * AST
            fg_term = (2 - factor * team_ast_ratio) * FG
            ft_term = FT * 0.5 * (1 + (1 - team_ast_ratio) + (2/3) * team_ast_ratio)
            
            # Negative contributions
            tov_term = VOP * TOV
            missed_fg_term = VOP * DRB_pct * (FGA - FG)
            missed_ft_term = VOP * 0.44 * (0.44 + (0.56 * DRB_pct)) * (FTA - FT)
            
            # Rebounding
            drb_term = VOP * (1 - DRB_pct) * (TRB - ORB)
            orb_term = VOP * DRB_pct * ORB
            
            # Defense
            stl_term = VOP * STL
            blk_term = VOP * DRB_pct * BLK
            
            # Fouls
            foul_value = (lg_FT / lg_PF) - 0.44 * (lg_FTA / lg_PF) * VOP if lg_PF > 0 else 0
            pf_term = PF * foul_value
            
            uper_numerator = (
                three_pts + ast_term + fg_term + ft_term -
                tov_term - missed_fg_term - missed_ft_term +
                drb_term + orb_term + stl_term + blk_term - pf_term
            )
            
            uper = uper_numerator / MP
            
        except (ZeroDivisionError, TypeError):
            uper = 0
        
        return uper
    
    def compute_aper(self, uper: float, team_pace: float, lg_pace: float) -> float:
        """Compute pace-adjusted PER (aPER)."""
        if team_pace == 0:
            return uper
        return uper * (lg_pace / team_pace)
    
    def compute_per(self, aper: float, lg_aper: float) -> float:
        """
        Compute normalized PER from aPER.
        
        Normalizes aPER so that league average = 15.0.
        Formula: PER = aPER * (15 / lg_aPER)
        
        Args:
            aper: Pace-adjusted PER
            lg_aper: League average aPER for the season
            
        Returns:
            Normalized PER (league average = 15.0)
        """
        if lg_aper == 0:
            return aper  # Fallback if league average not available
        return aper * (15.0 / lg_aper)
    
    def compute_league_average_aper(self, season: str, before_date: Optional[str] = None) -> float:
        """
        Compute league average aPER for a season.
        
        Calculates aPER for all players in the league and returns
        the minutes-weighted average.
        
        Uses cache to avoid expensive recomputation for the same season/date.
        
        Args:
            season: Season string
            before_date: Optional date to compute stats before (YYYY-MM-DD)
                        If None, uses all games in season
            
        Returns:
            League average aPER (minutes-weighted)
        """
        logger = logging.getLogger(__name__)
        
        # Check cache first - this is expensive to compute
        cache_key = (season, before_date)
        if cache_key in self._league_aper_cache:
            return self._league_aper_cache[cache_key]
        
        # Get league constants
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            logger.warning(f"[PER] Cannot compute lg_aPER: no league constants for {season}")
            self._league_aper_cache[cache_key] = 0.0
            return 0.0
        
        lg_pace = league_constants.get('lg_pace', 95)
        
        # Get all player stats for the season
        # Use preloaded data if available (much faster than DB query)
        if self._preloaded and self._player_stats_cache:
            # Collect player games from preloaded cache
            player_games = []
            for (team, season_key), player_list in self._player_stats_cache.items():
                if season_key == season:
                    # Filter by date if before_date is specified
                    if before_date:
                        player_games.extend([pg for pg in player_list if pg.get('date', '') < before_date])
                    else:
                        player_games.extend(player_list)
        else:
            # Fallback to DB query
            query = {
                'season': season,
                'stats.min': {'$gt': 0},
                'game_type': {'$nin': self._exclude_game_types}
            }
            if before_date:
                query['date'] = {'$lt': before_date}

            player_games = self._players_repo.find(query)
        
        if not player_games:
            logger.warning(f"[PER] No player games found for {season}")
            self._league_aper_cache[cache_key] = 0.0
            return 0.0
        
        # Aggregate by player
        player_agg = defaultdict(lambda: {
            'total_min': 0,
            'total_pts': 0,
            'total_fg_made': 0,
            'total_fg_att': 0,
            'total_three_made': 0,
            'total_ft_made': 0,
            'total_ft_att': 0,
            'total_reb': 0,
            'total_oreb': 0,
            'total_ast': 0,
            'total_stl': 0,
            'total_blk': 0,
            'total_to': 0,
            'total_pf': 0,
            'team': None
        })
        
        team_stats_agg = defaultdict(lambda: {
            'assists': 0,
            'FG_made': 0
        })
        
        for pg in player_games:
            pid = pg['player_id']
            team = pg.get('team')
            stats = pg.get('stats', {})
            
            min_played = stats.get('min', 0)
            if min_played == 0:
                continue
            
            player_agg[pid]['total_min'] += min_played
            player_agg[pid]['total_pts'] += stats.get('pts', 0)
            player_agg[pid]['total_fg_made'] += stats.get('fg_made', 0)
            player_agg[pid]['total_fg_att'] += stats.get('fg_att', 0)
            player_agg[pid]['total_three_made'] += stats.get('three_made', 0)
            player_agg[pid]['total_ft_made'] += stats.get('ft_made', 0)
            player_agg[pid]['total_ft_att'] += stats.get('ft_att', 0)
            player_agg[pid]['total_reb'] += stats.get('reb', 0)
            player_agg[pid]['total_oreb'] += stats.get('oreb', 0)
            player_agg[pid]['total_ast'] += stats.get('ast', 0)
            player_agg[pid]['total_stl'] += stats.get('stl', 0)
            player_agg[pid]['total_blk'] += stats.get('blk', 0)
            player_agg[pid]['total_to'] += stats.get('to', 0)
            player_agg[pid]['total_pf'] += stats.get('pf', 0)
            if not player_agg[pid]['team']:
                player_agg[pid]['team'] = team
            
            # Note: Team stats will be computed per-team when calculating PER
        
        # Calculate aPER for each player and compute weighted average
        total_minutes = 0
        weighted_aper_sum = 0.0
        
        for pid, player in player_agg.items():
            if player['total_min'] == 0:
                continue
            
            team = player['team']
            if not team:
                continue
            
            # Get team stats for PER calculation (aggregated before date if specified)
            if before_date:
                team_stats = self._get_team_stats_before_date_cached(team, season, before_date)
            else:
                # Get full season team stats
                team_stats = self._get_team_stats_before_date_cached(team, season, '9999-12-31')
            
            # Ensure we have valid team stats
            if team_stats['FG_made'] == 0:
                team_stats['FG_made'] = 1  # Avoid division by zero
            
            # Get team pace
            team_pace = get_team_pace(season, team, self.db, league=self.league)
            if team_pace == 0:
                team_pace = lg_pace
            
            # Aggregate stats for PER calculation
            agg_stats = {
                'min': player['total_min'],
                'pts': player['total_pts'],
                'fg_made': player['total_fg_made'],
                'fg_att': player['total_fg_att'],
                'three_made': player['total_three_made'],
                'ft_made': player['total_ft_made'],
                'ft_att': player['total_ft_att'],
                'reb': player['total_reb'],
                'oreb': player['total_oreb'],
                'ast': player['total_ast'],
                'stl': player['total_stl'],
                'blk': player['total_blk'],
                'to': player['total_to'],
                'pf': player['total_pf']
            }
            
            # Compute uPER and aPER
            uper = self.compute_uper(agg_stats, team_stats, league_constants)
            aper = self.compute_aper(uper, team_pace, lg_pace)
            
            # Add to weighted sum
            total_minutes += player['total_min']
            weighted_aper_sum += aper * player['total_min']
        
        # Return minutes-weighted average
        if total_minutes == 0:
            self._league_aper_cache[cache_key] = 0.0
            return 0.0
        
        lg_aper = weighted_aper_sum / total_minutes
        logger.debug(f"[PER] League average aPER for {season}: {lg_aper:.4f}")
        
        # Cache the result
        self._league_aper_cache[cache_key] = lg_aper
        return lg_aper
    
    # =========================================================================
    # OPTIMIZED DATA ACCESS (uses preloaded cache)
    # =========================================================================

    def _aggregate_player_games(self, player_games: List[dict]) -> dict:
        """
        Aggregate a list of player game docs into summary stats.

        Args:
            player_games: List of player-game documents with 'stats', 'player_id', etc.

        Returns:
            Aggregated dict matching the format of _get_team_players_before_date_cached output
        """
        if not player_games:
            return {}

        first = player_games[0]
        total_min = sum(pg.get('stats', {}).get('min', 0) for pg in player_games)
        games = len(player_games)

        if games == 0 or total_min == 0:
            return {}

        return {
            '_id': str(first.get('player_id')),
            'player_name': first.get('player_name', 'Unknown'),
            'games': games,
            'games_5min': sum(1 for pg in player_games if pg.get('stats', {}).get('min', 0) >= 5),
            'total_min': total_min,
            'total_pts': sum(pg.get('stats', {}).get('pts', 0) for pg in player_games),
            'total_fg_made': sum(pg.get('stats', {}).get('fg_made', 0) for pg in player_games),
            'total_fg_att': sum(pg.get('stats', {}).get('fg_att', 0) for pg in player_games),
            'total_three_made': sum(pg.get('stats', {}).get('three_made', 0) for pg in player_games),
            'total_ft_made': sum(pg.get('stats', {}).get('ft_made', 0) for pg in player_games),
            'total_ft_att': sum(pg.get('stats', {}).get('ft_att', 0) for pg in player_games),
            'total_reb': sum(pg.get('stats', {}).get('reb', 0) for pg in player_games),
            'total_oreb': sum(pg.get('stats', {}).get('oreb', 0) for pg in player_games),
            'total_ast': sum(pg.get('stats', {}).get('ast', 0) for pg in player_games),
            'total_stl': sum(pg.get('stats', {}).get('stl', 0) for pg in player_games),
            'total_blk': sum(pg.get('stats', {}).get('blk', 0) for pg in player_games),
            'total_to': sum(pg.get('stats', {}).get('to', 0) for pg in player_games),
            'total_pf': sum(pg.get('stats', {}).get('pf', 0) for pg in player_games),
            'avg_min': total_min / games,
            'starter_games': sum(1 for pg in player_games if pg.get('starter', False))
        }

    def _get_team_players_before_date_cached(
        self,
        team: str,
        season: str,
        before_date: str,
        min_games: int = 1
    ) -> List[dict]:
        """
        Get aggregated player stats for a team before a date.
        Uses preloaded cache for O(n) in-memory filtering instead of DB query.
        """
        if not self._preloaded:
            # Fall back to DB query
            return self._get_team_players_before_date_db(team, season, before_date, min_games)
        
        # Check cache first
        cache_key = (team, season, before_date, min_games)
        if cache_key in self._team_players_agg_cache:
            return self._team_players_agg_cache[cache_key]
        
        key = (team, season)
        if key not in self._player_stats_cache:
            # Key not in cache - fall back to DB query for this team/season
            # This happens when prediction context was loaded for a different scope
            return self._get_team_players_before_date_db(team, season, before_date, min_games)

# Filter in memory
        player_games = [
            pg for pg in self._player_stats_cache[key]
            if pg.get('date') and pg['date'] < before_date
        ]

        if not player_games:
            self._team_players_agg_cache[cache_key] = []
            return []
        
        # Aggregate by player
        player_agg = defaultdict(lambda: {
            'games': 0,
            'games_5min': 0,  # Games where player played > 5 minutes
            'total_min': 0,
            'total_pts': 0,
            'total_fg_made': 0,
            'total_fg_att': 0,
            'total_three_made': 0,
            'total_ft_made': 0,
            'total_ft_att': 0,
            'total_reb': 0,
            'total_oreb': 0,
            'total_ast': 0,
            'total_stl': 0,
            'total_blk': 0,
            'total_to': 0,
            'total_pf': 0,
            'starter_games': 0,
            'player_name': None
        })
        
        for pg in player_games:
            pid = pg['player_id']
            stats = pg.get('stats', {})
            
            # Skip players with no stats.min or stats.min == 0 (did not play)
            min_played = stats.get('min', 0)
            if not min_played or min_played == 0:
                continue
            
            player_agg[pid]['games'] += 1
            player_agg[pid]['total_min'] += min_played
            # Count games where player played > 5 minutes
            if min_played > 5:
                player_agg[pid]['games_5min'] += 1
            player_agg[pid]['total_pts'] += stats.get('pts', 0)
            player_agg[pid]['total_fg_made'] += stats.get('fg_made', 0)
            player_agg[pid]['total_fg_att'] += stats.get('fg_att', 0)
            player_agg[pid]['total_three_made'] += stats.get('three_made', 0)
            player_agg[pid]['total_ft_made'] += stats.get('ft_made', 0)
            player_agg[pid]['total_ft_att'] += stats.get('ft_att', 0)
            player_agg[pid]['total_reb'] += stats.get('reb', 0)
            player_agg[pid]['total_oreb'] += stats.get('oreb', 0)
            player_agg[pid]['total_ast'] += stats.get('ast', 0)
            player_agg[pid]['total_stl'] += stats.get('stl', 0)
            player_agg[pid]['total_blk'] += stats.get('blk', 0)
            player_agg[pid]['total_to'] += stats.get('to', 0)
            player_agg[pid]['total_pf'] += stats.get('pf', 0)
            if pg.get('starter'):
                player_agg[pid]['starter_games'] += 1
            if not player_agg[pid]['player_name']:
                player_agg[pid]['player_name'] = pg.get('player_name', 'Unknown')
        
        # Convert to list format matching DB aggregation output
        result = []
        for pid, agg in player_agg.items():
            if agg['games'] >= min_games and agg['total_min'] > 0:
                result.append({
                    '_id': pid,
                    'player_name': agg['player_name'],
                    'games': agg['games'],
                    'games_5min': agg['games_5min'],  # Games where player played > 5 minutes
                    'total_min': agg['total_min'],
                    'total_pts': agg['total_pts'],
                    'total_fg_made': agg['total_fg_made'],
                    'total_fg_att': agg['total_fg_att'],
                    'total_three_made': agg['total_three_made'],
                    'total_ft_made': agg['total_ft_made'],
                    'total_ft_att': agg['total_ft_att'],
                    'total_reb': agg['total_reb'],
                    'total_oreb': agg['total_oreb'],
                    'total_ast': agg['total_ast'],
                    'total_stl': agg['total_stl'],
                    'total_blk': agg['total_blk'],
                    'total_to': agg['total_to'],
                    'total_pf': agg['total_pf'],
                    'avg_min': agg['total_min'] / agg['games'],
                    'starter_games': agg['starter_games']
                })
        
        # Sort by total minutes descending
        result.sort(key=lambda x: x['total_min'], reverse=True)
        
        # Cache the result
        self._team_players_agg_cache[cache_key] = result
        return result
    
    def _get_team_players_before_date_db(
        self,
        team: str,
        season: str,
        before_date: str,
        min_games: int = 1
    ) -> List[dict]:
        """Fallback: Get player stats via DB aggregation pipeline."""
        pipeline = [
            {
                '$match': {
                    'team': team,
                    'season': season,
                    'date': {'$lt': before_date},
                    'stats.min': {'$gt': 0},
                    'game_type': {'$nin': self._exclude_game_types}
                }
            },
            {
                '$group': {
                    '_id': '$player_id',
                    'player_name': {'$first': '$player_name'},
                    'games': {'$sum': 1},
                    'games_5min': {'$sum': {'$cond': [{'$gt': ['$stats.min', 5]}, 1, 0]}},  # Games where player played > 5 minutes
                    'total_min': {'$sum': '$stats.min'},
                    'total_pts': {'$sum': '$stats.pts'},
                    'total_fg_made': {'$sum': '$stats.fg_made'},
                    'total_fg_att': {'$sum': '$stats.fg_att'},
                    'total_three_made': {'$sum': '$stats.three_made'},
                    'total_ft_made': {'$sum': '$stats.ft_made'},
                    'total_ft_att': {'$sum': '$stats.ft_att'},
                    'total_reb': {'$sum': '$stats.reb'},
                    'total_oreb': {'$sum': '$stats.oreb'},
                    'total_ast': {'$sum': '$stats.ast'},
                    'total_stl': {'$sum': '$stats.stl'},
                    'total_blk': {'$sum': '$stats.blk'},
                    'total_to': {'$sum': '$stats.to'},
                    'total_pf': {'$sum': '$stats.pf'},
                    'avg_min': {'$avg': '$stats.min'},
                    'starter_games': {'$sum': {'$cond': ['$starter', 1, 0]}}
                }
            },
            {'$match': {'games': {'$gte': min_games}}},
            {'$sort': {'total_min': -1}}
        ]
        result = self._players_repo.aggregate(pipeline)
        return result

    def _get_players_for_game_cross_team(
        self,
        game_id: str,
        team: str,
        season: str,
        before_date: str,
        min_games: int = 1
    ) -> List[dict]:
        """
        Get aggregated stats for players who played in a specific game,
        aggregating their stats across ALL teams (cross-team).

        Used for training to include traded players' full season history.
        Returns same shape as _get_team_players_before_date_cached().

        Args:
            game_id: The game ID to find players for
            team: Team name to get players for (home or away)
            season: Season string
            before_date: Date before which to compute stats
            min_games: Minimum games played threshold

        Returns:
            List of aggregated player dicts, sorted by total_min descending
        """
        if not self._game_to_players_cache or not self._player_stats_by_player:
            # Cross-team caches not available, fall back to team-specific
            return self._get_team_players_before_date_cached(team, season, before_date, min_games)

        # Get player IDs who actually played in this game for this team
        player_ids = self._game_to_players_cache.get(game_id, {}).get(team, [])
        if not player_ids:
            return []

        result = []
        for player_id in set(player_ids):  # dedupe
            player_key = (str(player_id), season)

            # Get ALL games for this player this season (any team)
            all_player_games = self._player_stats_by_player.get(player_key, [])

            # Filter to before_date and min > 0
            games_before = [
                pg for pg in all_player_games
                if pg.get('date', '') < before_date and pg.get('stats', {}).get('min', 0) > 0
            ]

            if len(games_before) < min_games:
                continue

            # Aggregate using helper
            agg = self._aggregate_player_games(games_before)
            if agg and agg.get('total_min', 0) > 0:
                result.append(agg)

        result.sort(key=lambda x: x['total_min'], reverse=True)
        return result

    def _get_players_cross_team_by_ids(
        self,
        player_ids: List[str],
        season: str,
        before_date: str,
        min_games: int = 1
    ) -> List[dict]:
        """
        Aggregate stats for specified players across ALL teams.
        Used for prediction when player list comes from rosters.

        This method provides training/prediction parity for traded players:
        - Training uses _get_players_for_game_cross_team (players from game doc)
        - Prediction uses this method (players from roster)
        Both aggregate stats across all teams the player played for.

        Args:
            player_ids: List of player IDs (strings) to aggregate
            season: Season string
            before_date: Date before which to compute stats
            min_games: Minimum games played threshold

        Returns:
            List of aggregated player dicts, sorted by total_min descending
        """
        if not self._player_stats_by_player:
            # Cross-team cache not available
            return []

        result = []
        for player_id in set(player_ids):  # dedupe
            player_key = (str(player_id), season)

            # Get ALL games for this player this season (any team)
            all_player_games = self._player_stats_by_player.get(player_key, [])

            # Filter to before_date and min > 0
            games_before = [
                pg for pg in all_player_games
                if pg.get('date', '') < before_date and pg.get('stats', {}).get('min', 0) > 0
            ]

            if len(games_before) < min_games:
                continue

            # Aggregate using helper
            agg = self._aggregate_player_games(games_before)
            if agg and agg.get('total_min', 0) > 0:
                result.append(agg)

        result.sort(key=lambda x: x['total_min'], reverse=True)
        return result

    def _get_team_stats_before_date_cached(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> dict:
        """
        Get aggregated team stats before a date.
        Uses preloaded cache for fast access.
        """
        if not self._preloaded:
            return self._get_team_stats_before_date_db(team, season, before_date)

        key = (team, season)
        if not self._team_stats_cache or key not in self._team_stats_cache:
            # Key not in cache - fall back to DB query for this team/season
            return self._get_team_stats_before_date_db(team, season, before_date)
        
        # Filter in memory
        team_games = [
            tg for tg in self._team_stats_cache[key]
            if tg['date'] < before_date
        ]
        
        if not team_games:
            return self._empty_team_totals()
        
        # Aggregate
        totals = self._empty_team_totals()
        for tg in team_games:
            td = tg['team_data']
            totals['assists'] += td.get('assists', 0)
            totals['FG_made'] += td.get('FG_made', 0)
            totals['FG_att'] += td.get('FG_att', 0)
            totals['FT_made'] += td.get('FT_made', 0)
            totals['FT_att'] += td.get('FT_att', 0)
            totals['total_reb'] += td.get('total_reb', 0)
            totals['off_reb'] += td.get('off_reb', 0)
            totals['TO'] += td.get('TO', 0)
        
        return totals
    
    def _get_team_stats_before_date_db(self, team: str, season: str, before_date: str) -> dict:
        """Fallback: Get team stats via DB query."""
        pipeline = [
            {
                '$match': {
                    '$or': [
                        {'homeTeam.name': team},
                        {'awayTeam.name': team}
                    ],
                    'season': season,
                    'date': {'$lt': before_date},
                    'game_type': {'$nin': self._exclude_game_types}
                }
            }
        ]

        games = self._games_repo.aggregate(pipeline)
        totals = self._empty_team_totals()
        
        for game in games:
            if game['homeTeam']['name'] == team:
                td = game['homeTeam']
            else:
                td = game['awayTeam']
            
            totals['assists'] += td.get('assists', 0)
            totals['FG_made'] += td.get('FG_made', 0)
            totals['FG_att'] += td.get('FG_att', 0)
            totals['FT_made'] += td.get('FT_made', 0)
            totals['FT_att'] += td.get('FT_att', 0)
            totals['total_reb'] += td.get('total_reb', 0)
            totals['off_reb'] += td.get('off_reb', 0)
            totals['TO'] += td.get('TO', 0)
        
        return totals
    
    def _empty_team_totals(self) -> dict:
        return {
            'assists': 0, 'FG_made': 0, 'FG_att': 0,
            'FT_made': 0, 'FT_att': 0, 'total_reb': 0,
            'off_reb': 0, 'TO': 0
        }

    # =========================================================================
    # PLAYER SUBSET FEATURES (from documentation/player_feature_updates.md)
    # =========================================================================

    def compute_player_subset_features(
        self,
        team: str,
        season: str,
        before_date: str,
        player_pers: list,
        starters_set: set = None,
        team_games_played: int = None,
        game_id: Optional[str] = None
    ) -> dict:
        """
        Compute all player subset features for a team.

        Uses player subsets as defined in documentation/player_feature_updates.md:
        - {ROTATION}: top active players by MPG, max ROTATION_SIZE (10)
        - {STARTERS}: starter players from the rotation
        - {BENCH}: {ROTATION} - {STARTERS}

        Args:
            team: Team name
            season: Season string
            before_date: Date string (YYYY-MM-DD)
            player_pers: List of player dicts with 'player_id', 'per', 'mpg', 'games', 'is_starter'
            starters_set: Optional set of starter player IDs (if None, uses is_starter flag)
            team_games_played: Number of team games played (for GP_THRESH calc)
            game_id: Optional game ID for training mode (enables cross-team aggregation)

        Returns:
            Dict with all player subset features
        """
        import math
        from datetime import datetime

        logger = logging.getLogger(__name__)

        result = {}

        if not player_pers:
            # Return zeros for all features if no players
            return self._empty_player_subset_features()

        # Calculate GP_THRESH based on team games
        if team_games_played is None:
            team_games_played = 82  # Default to full season
        # Cap GP threshold at team games played (can't require more games than played)
        ratio_thresh = math.ceil(GP_RATIO * team_games_played)
        gp_thresh = min(team_games_played, max(GP_FLOOR, ratio_thresh))

        # Filter players based on MPG and GP thresholds
        # {USAGE_PLAYERS} = {ROSTER} - {LOW_MPG} - {LOW_GP}
        usage_players = [
            p for p in player_pers
            if p.get('mpg', 0) >= MPG_THRESH and p.get('games', 0) >= gp_thresh
        ]

        # Sort by MPG to get rotation
        usage_players_sorted = sorted(usage_players, key=lambda x: x.get('mpg', 0), reverse=True)

        # {ROTATION} = top N by MPG, up to ROTATION_SIZE
        rotation = usage_players_sorted[:ROTATION_SIZE]

        # Determine starters in rotation
        if starters_set is None:
            starters_set = {str(p['player_id']) for p in player_pers if p.get('is_starter', False)}

        # {STARTERS} from rotation
        starters = [p for p in rotation if str(p['player_id']) in starters_set]

        # {BENCH} = {ROTATION} - {STARTERS}
        starter_ids = {str(p['player_id']) for p in starters}
        bench = [p for p in rotation if str(p['player_id']) not in starter_ids]

        # =========================================================================
        # player_starter_per: Average PER for starters
        # =========================================================================
        if starters:
            starter_per_avg = np.mean([p['per'] for p in starters])
        else:
            starter_per_avg = np.mean([p['per'] for p in rotation[:5]]) if rotation else 0.0
        result['player_starter_per_avg'] = float(starter_per_avg)

        # =========================================================================
        # player_bench_per: Weighted PER for bench players
        # =========================================================================
        if bench:
            # Static MPG-weighted (no recency)
            bench_weighted_mpg = self._compute_weighted_per(bench, weight_key='mpg')
            result['player_bench_per_weighted_MPG'] = float(bench_weighted_mpg)

            # Recency-weighted with different k values
            cross_team = bool(game_id)
            for k in [35, 40, 45, 50]:
                bench_recency = self._compute_subset_recency_weighted_per(
                    [str(p['player_id']) for p in bench], team, season, before_date, k,
                    cross_team=cross_team
                )
                result[f'player_bench_per_weighted_MIN_REC_k{k}'] = float(bench_recency)
        else:
            result['player_bench_per_weighted_MPG'] = 0.0
            for k in [35, 40, 45, 50]:
                result[f'player_bench_per_weighted_MIN_REC_k{k}'] = 0.0

        # =========================================================================
        # player_rotation_per: Weighted PER for rotation players
        # =========================================================================
        if rotation:
            # Static MPG-weighted (no recency)
            rotation_weighted_mpg = self._compute_weighted_per(rotation, weight_key='mpg')
            result['player_rotation_per_weighted_MPG'] = float(rotation_weighted_mpg)

            # Recency-weighted with different k values (lower k = faster decay)
            cross_team = bool(game_id)
            for k in [20, 25, 30, 35]:
                rotation_recency = self._compute_subset_recency_weighted_per(
                    [str(p['player_id']) for p in rotation], team, season, before_date, k,
                    cross_team=cross_team
                )
                result[f'player_rotation_per_weighted_MIN_REC_k{k}'] = float(rotation_recency)
        else:
            result['player_rotation_per_weighted_MPG'] = 0.0
            for k in [20, 25, 30, 35]:
                result[f'player_rotation_per_weighted_MIN_REC_k{k}'] = 0.0

        # =========================================================================
        # player_starter_bench_per_gap: Starter PER minus Bench PER
        # =========================================================================
        for k in [35, 40, 45, 50]:
            bench_k = result.get(f'player_bench_per_weighted_MIN_REC_k{k}', 0.0)
            gap = starter_per_avg - bench_k
            result[f'player_starter_bench_per_gap_derived_k{k}'] = float(gap)

        # =========================================================================
        # player_star_score: PER × MPG aggregations
        # =========================================================================
        # Compute star scores for rotation players
        star_scores = []
        for p in rotation:
            star_score = p.get('per', 0) * p.get('mpg', 0)
            star_scores.append({'player_id': p['player_id'], 'star_score': star_score})

        # Sort by star score
        star_scores_sorted = sorted(star_scores, key=lambda x: x['star_score'], reverse=True)

        # top1, top3_avg, top3_sum (season averages)
        if star_scores_sorted:
            result['player_star_score_top1'] = float(star_scores_sorted[0]['star_score'])
            top3 = star_scores_sorted[:3]
            result['player_star_score_top3_avg'] = float(np.mean([s['star_score'] for s in top3]))
            result['player_star_score_top3_sum'] = float(sum(s['star_score'] for s in top3))
        else:
            result['player_star_score_top1'] = 0.0
            result['player_star_score_top3_avg'] = 0.0
            result['player_star_score_top3_sum'] = 0.0

        # Recency-weighted star scores
        cross_team = bool(game_id)
        for k in [20, 25, 30, 35]:
            recency_star_scores = self._compute_recency_star_scores(
                [str(p['player_id']) for p in rotation], team, season, before_date, k,
                cross_team=cross_team
            )
            if recency_star_scores:
                recency_sorted = sorted(recency_star_scores, key=lambda x: x['star_score'], reverse=True)
                result[f'player_star_score_top1_MIN_REC_k{k}'] = float(recency_sorted[0]['star_score'])
                top3_rec = recency_sorted[:3]
                result[f'player_star_score_top3_MIN_REC_k{k}'] = float(sum(s['star_score'] for s in top3_rec))
            else:
                result[f'player_star_score_top1_MIN_REC_k{k}'] = 0.0
                result[f'player_star_score_top3_MIN_REC_k{k}'] = 0.0

        # =========================================================================
        # player_star_share: Star concentration metrics
        # =========================================================================
        total_star_mass = sum(s['star_score'] for s in star_scores)

        # top1_share, top3_share (season)
        if total_star_mass > EPS:
            result['player_star_share_top1_share'] = float(star_scores_sorted[0]['star_score'] / (total_star_mass + EPS)) if star_scores_sorted else 0.0
            top3_mass = sum(s['star_score'] for s in star_scores_sorted[:3])
            result['player_star_share_top3_share'] = float(top3_mass / (total_star_mass + EPS))
        else:
            result['player_star_share_top1_share'] = 0.0
            result['player_star_share_top3_share'] = 0.0

        # Recency-weighted star shares
        cross_team = bool(game_id)
        for k in [20, 25, 30, 35]:
            recency_star_scores = self._compute_recency_star_scores(
                [str(p['player_id']) for p in rotation], team, season, before_date, k,
                cross_team=cross_team
            )
            total_recency = sum(s['star_score'] for s in recency_star_scores) if recency_star_scores else 0.0
            if total_recency > EPS and recency_star_scores:
                recency_sorted = sorted(recency_star_scores, key=lambda x: x['star_score'], reverse=True)
                result[f'player_star_share_top1_share_MIN_REC_k{k}'] = float(recency_sorted[0]['star_score'] / (total_recency + EPS))
                top3_recency = sum(s['star_score'] for s in recency_sorted[:3])
                result[f'player_star_share_top3_share_MIN_REC_k{k}'] = float(top3_recency / (total_recency + EPS))
            else:
                result[f'player_star_share_top1_share_MIN_REC_k{k}'] = 0.0
                result[f'player_star_share_top3_share_MIN_REC_k{k}'] = 0.0

        # =========================================================================
        # player_star_score_all: Top star from {USAGE_PLAYERS} (includes injured in sorting)
        # =========================================================================
        all_star_scores = []
        for p in usage_players_sorted:
            all_star_scores.append(p.get('per', 0) * p.get('mpg', 0))
        result['player_star_score_all_top1'] = float(max(all_star_scores)) if all_star_scores else 0.0

        # =========================================================================
        # player_rotation_count: Depth indicator
        # =========================================================================
        result['player_rotation_count'] = len(rotation)

        # =========================================================================
        # player_continuity: Rotation cohesion proxy
        # =========================================================================
        if rotation:
            continuity_scores = []
            for p in rotation:
                gp = p.get('games', 0)
                continuity = min(1.0, gp / gp_thresh) if gp_thresh > 0 else 1.0
                continuity_scores.append(continuity)
            result['player_continuity_avg'] = float(np.mean(continuity_scores))
        else:
            result['player_continuity_avg'] = 0.0

        return result

    def _compute_weighted_per(self, players: list, weight_key: str = 'mpg') -> float:
        """Compute weighted PER for a list of players."""
        if not players:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for p in players:
            weight = p.get(weight_key, 0)
            if weight > 0:
                weighted_sum += p.get('per', 0) * weight
                total_weight += weight

        return weighted_sum / (total_weight + EPS) if total_weight > 0 else 0.0

    def _compute_subset_recency_weighted_per(
        self,
        player_ids: list,
        team: str,
        season: str,
        before_date: str,
        recency_decay_k: float,
        cross_team: bool = False
    ) -> float:
        """
        Compute recency-weighted PER for a subset of players.

        Formula: Σ_p Σ_g [ PER_g(p) × MIN_g(p) × weight_g ] / Σ_p Σ_g [ MIN_g(p) × weight_g ]
        where weight_g = exp(-days_since_game / k)

        PER is computed from box score stats for each game (not stored in DB).

        Args:
            player_ids: List of player IDs
            team: Team name
            season: Season string
            before_date: Date string
            recency_decay_k: Decay constant for recency weighting
            cross_team: If True, include games from all teams (for traded players in training)
        """
        import math
        from datetime import datetime

        if not player_ids:
            return 0.0
        before_date_obj = datetime.strptime(before_date, '%Y-%m-%d').date()

        # Get league constants and average aPER once (used for all games)
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            return 0.0
        lg_pace = league_constants.get('lg_pace', 95)

        # PERFORMANCE FIX: Use cached season-level lg_aper instead of computing per-date
        # The lg_aper doesn't vary significantly within a season, and computing per-date
        # triggers a full league-wide aggregation (very slow)
        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            # Try MongoDB cache first
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                # Last resort: compute it (slow but caches for future use)
                lg_aper = self.compute_league_average_aper(season, None)  # Use None for full season
                self._lg_aper_by_season[season] = lg_aper
        use_normalization = lg_aper > 0

        # Get team pace once
        team_pace = get_team_pace(season, team, self.db, league=self.league)
        if team_pace == 0:
            team_pace = lg_pace

        # Build game_id -> team_data lookup from cache for fast access
        game_team_data = {}
        key = (team, season)
        if self._preloaded and self._team_stats_cache and key in self._team_stats_cache:
            for tg in self._team_stats_cache[key]:
                game_team_data[tg['game_id']] = tg['team_data']

        total_weighted_per = 0.0
        total_weight = 0.0

        for player_id in player_ids:
            # Get player's per-game data
            player_games = self._get_player_games_for_subset(player_id, team, season, before_date, cross_team=cross_team)

            for pg in player_games:
                game_date_str = pg.get('date', '')
                if not game_date_str:
                    continue

                try:
                    game_date = datetime.strptime(str(game_date_str), '%Y-%m-%d').date()
                    days_since = (before_date_obj - game_date).days
                    if days_since < 0:
                        continue

                    stats = pg.get('stats', {})
                    min_g = stats.get('min', 0)
                    if min_g <= 0:
                        continue

                    # Get team stats for this game
                    game_id = pg.get('game_id')
                    team_data = game_team_data.get(game_id)
                    if not team_data:
                        # Fallback to DB query if not in cache
                        game_doc = self._games_repo.find_one({'game_id': game_id})
                        if game_doc:
                            is_home = pg.get('home', False)
                            team_data = game_doc.get('homeTeam' if is_home else 'awayTeam', {})

                    if not team_data:
                        continue

                    # Prepare team stats for PER calculation
                    team_stats = {
                        'assists': team_data.get('assists', 1),
                        'FG_made': team_data.get('FG_made', 1)
                    }

                    # Build player game stats dict
                    player_game_stats = {
                        'min': min_g,
                        'pts': stats.get('pts', 0),
                        'fg_made': stats.get('fg_made', 0),
                        'fg_att': stats.get('fg_att', 0),
                        'three_made': stats.get('three_made', 0),
                        'ft_made': stats.get('ft_made', 0),
                        'ft_att': stats.get('ft_att', 0),
                        'reb': stats.get('reb', 0),
                        'oreb': stats.get('oreb', 0),
                        'ast': stats.get('ast', 0),
                        'stl': stats.get('stl', 0),
                        'blk': stats.get('blk', 0),
                        'to': stats.get('to', 0),
                        'pf': stats.get('pf', 0)
                    }

                    # Compute PER: uPER -> aPER -> PER
                    uper = self.compute_uper(player_game_stats, team_stats, league_constants)
                    aper = self.compute_aper(uper, team_pace, lg_pace)
                    if use_normalization:
                        per_g = self.compute_per(aper, lg_aper)
                    else:
                        per_g = aper

                    # Apply recency weighting: MIN * exp(-days_since / k)
                    weight = math.exp(-days_since / recency_decay_k) * min_g
                    total_weighted_per += per_g * weight
                    total_weight += weight
                except (ValueError, TypeError):
                    continue
        return total_weighted_per / (total_weight + EPS) if total_weight > 0 else 0.0

    def _compute_recency_star_scores(
        self,
        player_ids: list,
        team: str,
        season: str,
        before_date: str,
        recency_decay_k: float,
        cross_team: bool = False
    ) -> list:
        """
        Compute recency-weighted star scores (PER × MIN) for a list of players.

        Returns list of {'player_id': id, 'star_score': score}

        PER is computed from box score stats for each game (not stored in DB).

        Args:
            player_ids: List of player IDs
            team: Team name
            season: Season string
            before_date: Date string
            recency_decay_k: Decay constant for recency weighting
            cross_team: If True, include games from all teams (for traded players in training)
        """
        import math
        from datetime import datetime

        results = []
        before_date_obj = datetime.strptime(before_date, '%Y-%m-%d').date()

        # Get league constants and average aPER once (used for all games)
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            # Return empty scores if no league constants
            return [{'player_id': pid, 'star_score': 0.0} for pid in player_ids]
        lg_pace = league_constants.get('lg_pace', 95)

        # PERFORMANCE FIX: Use cached season-level lg_aper
        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                lg_aper = self.compute_league_average_aper(season, None)
                self._lg_aper_by_season[season] = lg_aper
        use_normalization = lg_aper > 0

        # Get team pace once
        team_pace = get_team_pace(season, team, self.db, league=self.league)
        if team_pace == 0:
            team_pace = lg_pace

        # Build game_id -> team_data lookup from cache for fast access
        game_team_data = {}
        key = (team, season)
        if self._preloaded and self._team_stats_cache and key in self._team_stats_cache:
            for tg in self._team_stats_cache[key]:
                game_team_data[tg['game_id']] = tg['team_data']

        for player_id in player_ids:
            player_games = self._get_player_games_for_subset(player_id, team, season, before_date, cross_team=cross_team)

            nums = []
            denoms = []

            for pg in player_games:
                game_date_str = pg.get('date', '')
                if not game_date_str:
                    continue

                try:
                    game_date = datetime.strptime(str(game_date_str), '%Y-%m-%d').date()
                    days_since = (before_date_obj - game_date).days
                    if days_since < 0:
                        continue

                    stats = pg.get('stats', {})
                    min_g = stats.get('min', 0)
                    if min_g <= 0:
                        continue

                    # Get team stats for this game
                    game_id = pg.get('game_id')
                    team_data = game_team_data.get(game_id)
                    if not team_data:
                        # Fallback to DB query if not in cache
                        game_doc = self._games_repo.find_one({'game_id': game_id})
                        if game_doc:
                            is_home = pg.get('home', False)
                            team_data = game_doc.get('homeTeam' if is_home else 'awayTeam', {})

                    if not team_data:
                        continue

                    # Prepare team stats for PER calculation
                    team_stats = {
                        'assists': team_data.get('assists', 1),
                        'FG_made': team_data.get('FG_made', 1)
                    }

                    # Build player game stats dict
                    player_game_stats = {
                        'min': min_g,
                        'pts': stats.get('pts', 0),
                        'fg_made': stats.get('fg_made', 0),
                        'fg_att': stats.get('fg_att', 0),
                        'three_made': stats.get('three_made', 0),
                        'ft_made': stats.get('ft_made', 0),
                        'ft_att': stats.get('ft_att', 0),
                        'reb': stats.get('reb', 0),
                        'oreb': stats.get('oreb', 0),
                        'ast': stats.get('ast', 0),
                        'stl': stats.get('stl', 0),
                        'blk': stats.get('blk', 0),
                        'to': stats.get('to', 0),
                        'pf': stats.get('pf', 0)
                    }

                    # Compute PER: uPER -> aPER -> PER
                    uper = self.compute_uper(player_game_stats, team_stats, league_constants)
                    aper = self.compute_aper(uper, team_pace, lg_pace)
                    if use_normalization:
                        per_g = self.compute_per(aper, lg_aper)
                    else:
                        per_g = aper

                    # Star score = PER × MIN
                    star_g = per_g * min_g
                    w = math.exp(-days_since / recency_decay_k)
                    nums.append(star_g * w)
                    denoms.append(w)
                except (ValueError, TypeError):
                    continue

            # Recency-weighted star score per game
            star_score = sum(nums) / (sum(denoms) + EPS) if denoms else 0.0
            results.append({'player_id': player_id, 'star_score': star_score})

        return results

    def _get_player_games_for_subset(
        self,
        player_id: str,
        team: str,
        season: str,
        before_date: str,
        cross_team: bool = False
    ) -> list:
        """
        Get a player's games for subset feature calculations.

        Args:
            player_id: Player ID
            team: Team name (used for team-specific lookup)
            season: Season string
            before_date: Date before which to get games
            cross_team: If True, get games across ALL teams (for training with traded players)
        """
        if cross_team and self._player_stats_by_player:
            # Cross-team: return ALL games for this player this season (any team)
            player_key = (str(player_id), season)
            all_games = self._player_stats_by_player.get(player_key, [])
            return [
                pg for pg in all_games
                if pg.get('date', '') < before_date and pg.get('stats', {}).get('min', 0) > 0
            ]

        # Team-specific lookup (original behavior)
        key = (team, season)

        # Use preloaded cache if available
        if self._preloaded and self._player_stats_cache and key in self._player_stats_cache:
            return [
                pg for pg in self._player_stats_cache[key]
                if (str(pg.get('player_id')) == str(player_id)) and
                   pg.get('date') and pg['date'] < before_date and
                   pg.get('stats', {}).get('min', 0) > 0
            ]

        # Fallback to DB query
        return self._players_repo.find({
            'player_id': player_id,
            'team': team,
            'season': season,
            'date': {'$lt': before_date},
            'stats.min': {'$gt': 0},
            'game_type': {'$nin': self._exclude_game_types}
        })

    def _empty_player_subset_features(self) -> dict:
        """Return empty dict with all player subset features set to 0."""
        result = {
            'player_starter_per_avg': 0.0,
            'player_bench_per_weighted_MPG': 0.0,
            'player_rotation_per_weighted_MPG': 0.0,
            'player_star_score_top1': 0.0,
            'player_star_score_top3_avg': 0.0,
            'player_star_score_top3_sum': 0.0,
            'player_star_share_top1_share': 0.0,
            'player_star_share_top3_share': 0.0,
            'player_star_score_all_top1': 0.0,
            'player_rotation_count': 0,
            'player_continuity_avg': 0.0,
        }
        # Add k-variations
        for k in [35, 40, 45, 50]:
            result[f'player_bench_per_weighted_MIN_REC_k{k}'] = 0.0
            result[f'player_starter_bench_per_gap_derived_k{k}'] = 0.0
        for k in [20, 25, 30, 35]:
            result[f'player_rotation_per_weighted_MIN_REC_k{k}'] = 0.0
            result[f'player_star_score_top1_MIN_REC_k{k}'] = 0.0
            result[f'player_star_score_top3_MIN_REC_k{k}'] = 0.0
            result[f'player_star_share_top1_share_MIN_REC_k{k}'] = 0.0
            result[f'player_star_share_top3_share_MIN_REC_k{k}'] = 0.0
        return result

    # =========================================================================
    # TEAM PER FEATURES (with caching)
    # =========================================================================
    
    def compute_team_per_features(
        self,
        team: str,
        season: str,
        before_date: str,
        top_n: int = 8,
        player_filters: Optional[Dict] = None,
        injured_players: Optional[List[str]] = None,
        game_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Compute team-level PER features for game prediction.
        Uses preloaded cache for fast computation.

        Args:
            team: Team name
            season: Season string
            before_date: Date before which to compute stats
            top_n: Number of top players to include
            player_filters: Optional dict with keys:
                - 'playing': List of player_ids that are playing (exclude others)
                - 'starters': List of player_ids that are starters
            injured_players: Optional list of injured player IDs (strings).
                If provided, these players will be excluded from PER calculations.
                Training: Should pass stats_nba.{home/away}Team.injured_players
                Prediction: Should pass injured players from nba_rosters.injured flag
            game_id: Optional game ID for training mode. When provided, uses cross-team
                aggregation to include traded players' full season history.
        """
        logger = logging.getLogger(__name__)
        # Get all players who have played for this team
        # Training mode: use cross-team aggregation for traded player support
        if game_id and self._game_to_players_cache:
            players = self._get_players_for_game_cross_team(game_id, team, season, before_date, min_games=1)
        elif player_filters and self._player_stats_by_player:
            # Prediction mode with roster data: get player IDs from roster, aggregate cross-team
            # This ensures traded players' full season stats are included (matching training behavior)
            player_ids = [str(pid) for pid in player_filters.get('playing', [])]
            if player_ids:
                players = self._get_players_cross_team_by_ids(player_ids, season, before_date, min_games=1)
            else:
                # Empty roster, fall back to team-specific lookup
                players = self._get_team_players_before_date_cached(team, season, before_date, min_games=1)
        else:
            # Fallback: team-specific lookup (no cross-team cache available)
            players = self._get_team_players_before_date_cached(team, season, before_date, min_games=1)
        if not players:
            logger.debug(f"[PER] No players found for {team} before {before_date}")
            return None
        
        initial_player_count = len(players)
        initial_player_ids = {str(p.get('_id', '')) for p in players}
        logger.debug(f"[PER] {team}: Found {initial_player_count} players before {before_date}")
        
        # DEBUG: If player_filters provided, check for mismatches early (before any filtering)
        if player_filters:
            playing_list_check = player_filters.get('playing', [])
            if playing_list_check:
                playing_set_check = {str(pid) for pid in playing_list_check}
                unmatched_early = playing_set_check - initial_player_ids
                if unmatched_early:
                    logger.warning(
                        f"[PER] {team}: DEBUG - {len(unmatched_early)} players in filter but NOT in initial players list (before any filtering): "
                        f"{list(unmatched_early)[:5]}. This suggests they don't have stats before {before_date} or weren't found by aggregation."
                    )
        
        # Phase 2.1: Filter out injured players if explicitly provided
        # CRITICAL: Injured players are excluded from ALL PER feature calculations:
        # - perAvg, perWeighted, per1/per2/per3, and startersPerAvg
        # This filtering happens BEFORE any PER calculations, ensuring injured players
        # are never included in any PER features
        if injured_players:
            # Convert to set of strings for efficient lookup
            injured_players_set = {str(pid) for pid in injured_players}
            players_before_injury = len(players)
            players = [p for p in players if str(p['_id']) not in injured_players_set]
            players_after_injury = len(players)
            logger.debug(f"[PER] {team}: Filtering out {len(injured_players_set)} injured players: {players_before_injury} -> {players_after_injury}")
            
            # DEBUG: Check if any players were removed by injury filter
            if players_before_injury > players_after_injury:
                removed_by_injury = players_before_injury - players_after_injury
                logger.debug(f"[PER] {team}: Removed {removed_by_injury} injured players from PER calculations")
        
        db_filtered_count = len(players)
        logger.debug(f"[PER] {team}: {db_filtered_count} players after filtering")
        
        if not players:
            # Debug level - this is expected for non-NBA teams (All-Star, preseason international)
            logger.debug(f"[PER] {team}: No players remaining after filtering")
            return None
        
        # Store filter sets for later use
        playing_set = None
        starters_set = None

        # Phase 2.3: Apply player filters if provided and validate IDs

        if player_filters:
            playing_list = player_filters.get('playing', [])
            starters_list = player_filters.get('starters', [])

            # If player_filters provided but playing list is empty, roster data is missing
            # Fall back to using all historical players (like training mode)
            if not playing_list:
                logger.warning(f"[PER] {team}: player_filters provided but playing list is empty - falling back to historical players")
                player_filters = None  # Disable filtering, use all historical players
            else:
                # Filter to roster players
                playing_set = set(playing_list)

                # Hard ID mismatch check
                # Normalize both to strings to handle type mismatches (player_id can be int or str in DB)
                available_ids = {str(p['_id']) for p in players}
                playing_set_str = {str(pid) for pid in playing_list}
                unmatched = playing_set_str - available_ids
                if unmatched:
                    # Look up player names for unmatched IDs
                    unmatched_list = list(unmatched)
                    player_name_map = {}
                    try:
                        # Query stats_nba_players to get player names
                        unmatched_docs = self._players_repo.find(
                            {'player_id': {'$in': unmatched_list}},
                            projection={'player_id': 1, 'player_name': 1},
                            limit=20
                        )
                        player_name_map = {doc['player_id']: doc.get('player_name', 'Unknown') for doc in unmatched_docs}
                    except Exception as e:
                        logger.debug(f"Error looking up player names: {e}")

                    # Build player info strings (ID: Name if available, otherwise just ID)
                    player_info_list = []
                    for pid in unmatched_list[:10]:  # Show up to 10
                        name = player_name_map.get(pid, 'Unknown')
                        if name and name != 'Unknown':
                            player_info_list.append(f"{pid} ({name})")
                        else:
                            player_info_list.append(pid)

                    # DEBUG: Check why these players are unmatched
                    debug_info = []
                    for unmatched_id in unmatched_list[:5]:  # Check first 5 unmatched
                        # Check if they have stats before the date
                        stats_check = self._players_repo.find(
                            {
                                'player_id': unmatched_id,
                                'team': team,
                                'season': season,
                                'date': {'$lt': before_date},
                                'stats.min': {'$gt': 0},
                                'game_type': {'$nin': self._exclude_game_types}
                            },
                            projection={'date': 1, 'game_id': 1},
                            sort=[('date', -1)],
                            limit=3
                        )
                        debug_info.append(f"{unmatched_id}: {len(stats_check)} games before {before_date}")

                        # Check if they're in the initial players list (before filtering)
                        initial_players_check = [p for p in players if str(p.get('_id', '')) == unmatched_id]
                        if initial_players_check:
                            debug_info.append(f"  -> Found in players list after filtering")
                        else:
                            debug_info.append(f"  -> NOT in players list after filtering")

                    # Build detailed warning message
                    warning_msg = (
                        f"[PER] {team}: Some player_filters IDs not found in available players. "
                        f"Game date: {before_date}, Season: {season}. "
                        f"Total unmatched: {len(unmatched)}. "
                        f"Filter had {len(playing_list)} players, available had {len(available_ids)} players. "
                        f"Unmatched IDs: {', '.join(player_info_list)}. "
                        f"DEBUG: {'; '.join(debug_info)}"
                    )
                    if len(unmatched) > 10:
                        warning_msg += f" (showing first 10 of {len(unmatched)})"

                    logger.warning(warning_msg)

                # Filter to only players who are playing (normalize _id to string for comparison)
                players = [p for p in players if str(p['_id']) in playing_set_str]
                logger.debug(f"[PER] {team}: {len(players)} players after applying 'playing' filter")

                if starters_list:
                    starters_set = set(starters_list)
                    logger.debug(f"[PER] {team}: {len(starters_set)} starters specified in filter")

        # Get league constants
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            return None

        lg_pace = league_constants.get('lg_pace', 95)
        team_pace = get_team_pace(season, team, self.db, league=self.league)
        
        # Get league average aPER for normalization
        # Try to get from cache first, otherwise compute it
        # PERFORMANCE FIX: Use season-only cache key since lg_aper doesn't vary much by date
        lg_aper = None

        # Check class-level in-memory cache first (fastest)
        if not hasattr(self, '_lg_aper_by_season'):
            self._lg_aper_by_season = {}

        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            # Try MongoDB cache
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                # Compute on-demand (can be slow, but ensures accuracy)
                # Use end of season as date to get full season average (more stable)
                lg_aper = self.compute_league_average_aper(season, None)  # None = full season
                self._lg_aper_by_season[season] = lg_aper

                # Cache it for future use (optional optimization)
                if lg_aper > 0 and cached_stats:
                    try:
                        self._league_cache_repo.update_one(
                            {'season': season},
                            {'$set': {'lg_aper': lg_aper}},
                            upsert=False
                        )
                    except Exception as e:
                        logger.debug(f"[PER] Could not cache lg_aper: {e}")
        
        # Fallback: if lg_aper is still 0 or None, don't normalize
        use_normalization = lg_aper is not None and lg_aper > 0

        # Get team totals for PER calculation
        team_stats = self._get_team_stats_before_date_cached(team, season, before_date)

        player_pers = []

        # Process all players (no top_n cutoff - assume player_filters provides correct players)
        for player in players:
            total_min = player['total_min']
            if total_min == 0:
                continue
            
            # Aggregate stats to compute PER
            agg_stats = {
                'min': total_min,
                'pts': player['total_pts'],
                'fg_made': player['total_fg_made'],
                'fg_att': player['total_fg_att'],
                'three_made': player['total_three_made'],
                'ft_made': player['total_ft_made'],
                'ft_att': player['total_ft_att'],
                'reb': player['total_reb'],
                'oreb': player['total_oreb'],
                'ast': player['total_ast'],
                'stl': player['total_stl'],
                'blk': player['total_blk'],
                'to': player['total_to'],
                'pf': player['total_pf']
            }
            
            uper = self.compute_uper(agg_stats, team_stats, league_constants)
            aper = self.compute_aper(uper, team_pace, lg_pace)
            
            # Normalize to PER (league average = 15.0)
            if use_normalization:
                per = self.compute_per(aper, lg_aper)
            else:
                per = aper  # Use aPER if normalization not available
            
            # Determine if player is starter (for individual player dict, not used in startersPerAvg calculation)
            # For prediction (player_filters provided): Use nba_rosters.starter as authority
            # For training (no player_filters): Use historical heuristic from stats_nba_players
            # Determine starter status
            # CRITICAL: When player_filters is provided, use starters from player_filters (not nba_rosters)
            # - Training: player_filters['starters'] comes from stats_nba_players.starter for that game
            # - Prediction: player_filters['starters'] comes from UI (which syncs with nba_rosters.starter)
            if player_filters:
                # Use starter status from player_filters (source of truth)
                # player_filters is already the team-level dict: {'playing': [...], 'starters': [...], ...}
                starters_list = player_filters.get('starters', [])
                if starters_list:
                    is_starter = str(player['_id']) in {str(sid) for sid in starters_list}
                else:
                    # Fallback: if no starters in player_filters, use historical heuristic
                    is_starter = player['starter_games'] > player['games'] / 2
            else:
                # Training mode (no player_filters): Use historical heuristic
                is_starter = player['starter_games'] > player['games'] / 2
            
            # Calculate minutes per game (MPG) for sorting
            games_5min = player.get('games_5min', 0)
            mpg = total_min / games_5min if games_5min > 0 else 0
            
            player_pers.append({
                'player_id': player['_id'],
                'player_name': player['player_name'],
                'games': player['games'],
                'games_5min': games_5min,  # Games where player played > 5 minutes
                'total_min': total_min,
                'avg_min': player['avg_min'],
                'mpg': mpg,  # Minutes per game (total_min / games_5min)
                'uper': uper,
                'aper': aper,
                'per': per,  # Normalized PER (or aPER if normalization not available)
                'is_starter': is_starter
            })

        if not player_pers:
            logger.warning(f"[PER] {team}: No players with valid PER after filtering")
            return None

        # Phase 2.1: Log final player count
        logger.debug(f"[PER] {team}: Computing PER for {len(player_pers)} players")
        if use_normalization:
            logger.debug(f"[PER] {team}: Using normalized PER (lg_aPER={lg_aper:.4f})")
        else:
            logger.debug(f"[PER] {team}: Using aPER (normalization not available)")
        
        # Use all players (no sorting/cutoff - assume player_filters provides correct players)
        # Calculate perAvg as simple average of all players' PER
        per_avg = np.mean([p['per'] for p in player_pers]) if player_pers else 0
        
        # Calculate perWeighted using minutes per game (MPG) as weight
        # MPG = total_min / games_5min (games where player played > 5 minutes)
        # If games_5min is 0, weight is 0 (MPG already calculated and stored in player dict)
        weighted_sum = 0.0
        total_mpg = 0.0
        for p in player_pers:
            mpg = p.get('mpg', 0)
            if mpg > 0:  # Only include players with games_5min > 0 (i.e., played > 5 min in at least one game)
                weighted_sum += p['per'] * mpg
                total_mpg += mpg
        per_weighted = weighted_sum / total_mpg if total_mpg > 0 else 0
        
        if player_filters:
            # Use starters from player_filters (already sourced from rosters)
            starter_ids = set(str(sid) for sid in player_filters.get('starters', []))
            starters = [p for p in player_pers if str(p['player_id']) in starter_ids][:5]
            logger.debug(f"[PER] {team}: Using {len(starters)} starters from player_filters (prediction mode)")
        else:
            # Training mode: Use historical heuristic (is_starter from aggregated stats)
            starters = [p for p in player_pers if p['is_starter']][:5]
            logger.debug(f"[PER] {team}: Using historical starter heuristic ({len(starters)} starters, training mode)")
        
        starters_avg = np.mean([p['per'] for p in starters]) if starters else per_avg
        
        # For per1, per2, per3: sort by PER (descending) and take top 3
        # These features represent the top 1st, 2nd, and 3rd players by PER value
        player_pers_sorted_by_per = sorted(player_pers, key=lambda x: x.get('per', 0), reverse=True)
        slot_pers = [p['per'] for p in player_pers_sorted_by_per[:3]]
        # Only keep per1, per2, per3
        while len(slot_pers) < 3:
            slot_pers.append(0)
        
        # Calculate recency-weighted PER for top player by MPG (for player_per_1|none|weighted_MIN_REC feature)
        # Note: This uses top player by MPG, not by PER
        player_pers_sorted_by_mpg = sorted(player_pers, key=lambda x: x.get('mpg', 0), reverse=True)
        per1_recency_weighted = 0.0
        if player_pers_sorted_by_mpg and len(player_pers_sorted_by_mpg) > 0:
            top_player_by_mpg_id = player_pers_sorted_by_mpg[0]['player_id']
            recency_per = self.get_player_recency_weighted_per(
                str(top_player_by_mpg_id), team, season, before_date, recency_decay_k=14.0
            )
            per1_recency_weighted = recency_per if recency_per is not None else 0.0
        
        # Phase 2.1: Log final PER values
        logger.debug(
            f"[PER] {team}: PER_avg={per_avg:.2f}, PER_weighted={per_weighted:.2f}, "
            f"Starters_avg={starters_avg:.2f}, Player_count={len(player_pers)}"
        )
        
        # Build player lists for each feature
        # perAvg: all players used in calculation
        per_avg_players = [{'player_id': str(p['player_id']), 'player_name': p['player_name'], 'per': p['per']} for p in player_pers]
        
        # perWeighted: all players with MPG > 0 (used in weighted calculation)
        per_weighted_players = [{'player_id': str(p['player_id']), 'player_name': p['player_name'], 'per': p['per'], 'mpg': p.get('mpg', 0)} 
                                for p in player_pers if p.get('mpg', 0) > 0]
        
        # startersPerAvg: starter players
        starters_players = [{'player_id': str(p['player_id']), 'player_name': p['player_name'], 'per': p['per']} for p in starters]
        
        # per1, per2, per3: top 3 players by PER (not MPG)
        per1_player = [{'player_id': str(player_pers_sorted_by_per[0]['player_id']), 'player_name': player_pers_sorted_by_per[0]['player_name'], 'per': player_pers_sorted_by_per[0]['per'], 'mpg': player_pers_sorted_by_per[0].get('mpg', 0)}] if len(player_pers_sorted_by_per) > 0 else []
        per2_player = [{'player_id': str(player_pers_sorted_by_per[1]['player_id']), 'player_name': player_pers_sorted_by_per[1]['player_name'], 'per': player_pers_sorted_by_per[1]['per'], 'mpg': player_pers_sorted_by_per[1].get('mpg', 0)}] if len(player_pers_sorted_by_per) > 1 else []
        per3_player = [{'player_id': str(player_pers_sorted_by_per[2]['player_id']), 'player_name': player_pers_sorted_by_per[2]['player_name'], 'per': player_pers_sorted_by_per[2]['per'], 'mpg': player_pers_sorted_by_per[2].get('mpg', 0)}] if len(player_pers_sorted_by_per) > 2 else []
        
        # Calculate topN_avg features (average PER of top N players)
        top1_avg = slot_pers[0] if len(slot_pers) > 0 else 0.0
        top2_avg = np.mean(slot_pers[:2]) if len(slot_pers) >= 2 else (top1_avg if len(slot_pers) >= 1 else 0.0)
        top3_avg = np.mean(slot_pers[:3]) if len(slot_pers) >= 3 else (np.mean(slot_pers) if len(slot_pers) > 0 else 0.0)
        
        # Calculate topN_weighted_MPG features (MPG-weighted average PER of top N players)
        top1_weighted = 0.0
        top2_weighted = 0.0
        top3_weighted = 0.0
        
        if len(player_pers_sorted_by_per) > 0:
            top1_players = player_pers_sorted_by_per[:1]
            top1_weighted = np.average([p['per'] for p in top1_players], weights=[p.get('mpg', 0) for p in top1_players]) if any(p.get('mpg', 0) > 0 for p in top1_players) else top1_avg
        
        if len(player_pers_sorted_by_per) >= 2:
            top2_players = player_pers_sorted_by_per[:2]
            top2_weighted = np.average([p['per'] for p in top2_players], weights=[p.get('mpg', 0) for p in top2_players]) if any(p.get('mpg', 0) > 0 for p in top2_players) else top2_avg
        
        if len(player_pers_sorted_by_per) >= 3:
            top3_players = player_pers_sorted_by_per[:3]
            top3_weighted = np.average([p['per'] for p in top3_players], weights=[p.get('mpg', 0) for p in top3_players]) if any(p.get('mpg', 0) > 0 for p in top3_players) else top3_avg

        # Compute player subset features (bench, rotation, star scores, etc.)
        # Build starters_set from the starters we identified
        starters_set = {str(p['player_id']) for p in starters}

        # Get team games played for GP_THRESH calculation
        # Use max games any player has played (approximates team's total games)
        team_games_played = max([p.get('games', 0) for p in player_pers], default=0)
        if team_games_played == 0:
            team_games_played = 82  # Default

        subset_features = self.compute_player_subset_features(
            team=team,
            season=season,
            before_date=before_date,
            player_pers=player_pers,
            starters_set=starters_set,
            team_games_played=team_games_played,
            game_id=game_id
        )

        return {
            'team': team,
            'season': season,
            'before_date': before_date,
            'per_avg': float(per_avg),
            'per_weighted': float(per_weighted),
            'starters_avg': float(starters_avg),
            'per1': slot_pers[0] if len(slot_pers) > 0 else 0,
            'per2': slot_pers[1] if len(slot_pers) > 1 else 0,
            'per3': slot_pers[2] if len(slot_pers) > 2 else 0,
            'per1_recency_weighted': float(per1_recency_weighted),
            'top1_avg': float(top1_avg),
            'top2_avg': float(top2_avg),
            'top3_avg': float(top3_avg),
            'top1_weighted_MPG': float(top1_weighted),
            'top2_weighted_MPG': float(top2_weighted),
            'top3_weighted_MPG': float(top3_weighted),
            # Sum of top 3 PERs (safe denominator for normalized injury features)
            'top3_sum': float(slot_pers[0] if len(slot_pers) > 0 else 0) +
                        float(slot_pers[1] if len(slot_pers) > 1 else 0) +
                        float(slot_pers[2] if len(slot_pers) > 2 else 0),
            'player_count': len(player_pers),
            'players': player_pers[:top_n],
            # Player lists for each feature
            'per_avg_players': per_avg_players,
            'per_weighted_players': per_weighted_players,
            'starters_players': starters_players,
            'per1_player': per1_player,
            'per2_player': per2_player,
            'per3_player': per3_player,
            # Player subset features (bench, rotation, star scores, etc.)
            **subset_features
        }

    def get_game_per_features(
        self,
        home_team: str,
        away_team: str,
        season: str,
        game_date: str,
        use_cache: bool = True,
        player_filters: Optional[Dict] = None,
        injured_players: Optional[Dict] = None,
        game_id: Optional[str] = None
    ) -> Optional[dict]:
        """
        Get PER-based features for a game prediction.

        Args:
            home_team: Home team name
            away_team: Away team name
            season: Season string
            game_date: Date of the game (YYYY-MM-DD)
            use_cache: If True, check MongoDB cache first
            player_filters: Optional dict with team names as keys:
                {team: {'playing': [player_ids], 'starters': [player_ids]}}
            injured_players: Optional dict with team names as keys:
                {team: [player_ids]} - List of injured player IDs for each team
                Training: Should pass stats_nba.{home/away}Team.injured_players
                Prediction: Should pass injured players from nba_rosters.injured flag
            game_id: Optional game ID for training mode. When provided, uses cross-team
                aggregation to include traded players' full season history.

        Returns:
            Dict with differential PER features for the matchup
        """
        # Don't use cache if player filters, injured players, or game_id are provided
        if player_filters or injured_players or game_id:
            use_cache = False

        # Check in-memory cache first (only if no filters)
        cache_key = f"{season}|{home_team}|{away_team}|{game_date}"
        if use_cache and cache_key in self._per_features_cache:
            return self._per_features_cache[cache_key]

        # Get player filters and injured players for each team
        home_filters = player_filters.get(home_team, {}) if player_filters else None
        away_filters = player_filters.get(away_team, {}) if player_filters else None
        home_injured = injured_players.get(home_team, []) if injured_players else None
        away_injured = injured_players.get(away_team, []) if injured_players else None

        # Compute features
        home_per = self.compute_team_per_features(
            home_team, season, game_date,
            player_filters=home_filters,
            injured_players=home_injured,
            game_id=game_id
        )
        away_per = self.compute_team_per_features(
            away_team, season, game_date,
            player_filters=away_filters,
            injured_players=away_injured,
            game_id=game_id
        )
        
        if not home_per or not away_per:
            return None
        
        # Return features in pipe-delimited format
        features = {
            'player_team_per|season|avg|home': home_per['per_avg'],
            'player_team_per|season|avg|away': away_per['per_avg'],
            'player_team_per|season|avg|diff': home_per['per_avg'] - away_per['per_avg'],
            
            'player_team_per|season|weighted_MPG|home': home_per['per_weighted'],
            'player_team_per|season|weighted_MPG|away': away_per['per_weighted'],
            'player_team_per|season|weighted_MPG|diff': home_per['per_weighted'] - away_per['per_weighted'],
            
            'player_starters_per|season|avg|home': home_per['starters_avg'],
            'player_starters_per|season|avg|away': away_per['starters_avg'],
            'player_starters_per|season|avg|diff': home_per['starters_avg'] - away_per['starters_avg'],
            
            # Format: player_per|season|topN_avg (average PER of top N players)
            'player_per|season|top1_avg|home': home_per.get('top1_avg', 0),
            'player_per|season|top2_avg|home': home_per.get('top2_avg', 0),
            'player_per|season|top3_avg|home': home_per.get('top3_avg', 0),
            
            'player_per|season|top1_avg|away': away_per.get('top1_avg', 0),
            'player_per|season|top2_avg|away': away_per.get('top2_avg', 0),
            'player_per|season|top3_avg|away': away_per.get('top3_avg', 0),
            
            'player_per|season|top1_avg|diff': home_per.get('top1_avg', 0) - away_per.get('top1_avg', 0),
            'player_per|season|top2_avg|diff': home_per.get('top2_avg', 0) - away_per.get('top2_avg', 0),
            'player_per|season|top3_avg|diff': home_per.get('top3_avg', 0) - away_per.get('top3_avg', 0),
            
            # MPG-weighted versions
            'player_per|season|top1_weighted_MPG|home': home_per.get('top1_weighted_MPG', 0),
            'player_per|season|top2_weighted_MPG|home': home_per.get('top2_weighted_MPG', 0),
            'player_per|season|top3_weighted_MPG|home': home_per.get('top3_weighted_MPG', 0),
            
            'player_per|season|top1_weighted_MPG|away': away_per.get('top1_weighted_MPG', 0),
            'player_per|season|top2_weighted_MPG|away': away_per.get('top2_weighted_MPG', 0),
            'player_per|season|top3_weighted_MPG|away': away_per.get('top3_weighted_MPG', 0),
            
            'player_per|season|top1_weighted_MPG|diff': home_per.get('top1_weighted_MPG', 0) - away_per.get('top1_weighted_MPG', 0),
            'player_per|season|top2_weighted_MPG|diff': home_per.get('top2_weighted_MPG', 0) - away_per.get('top2_weighted_MPG', 0),
            'player_per|season|top3_weighted_MPG|diff': home_per.get('top3_weighted_MPG', 0) - away_per.get('top3_weighted_MPG', 0),

            # Sum of top 3 PERs (safe denominator for normalized injury features)
            'player_per|season|top3_sum|home': home_per.get('top3_sum', 0),
            'player_per|season|top3_sum|away': away_per.get('top3_sum', 0),
            'player_per|season|top3_sum|diff': home_per.get('top3_sum', 0) - away_per.get('top3_sum', 0),

            # Player lists for each feature (for display in feature modal)
            '_player_lists': {
                'player_team_per|season|avg|home': home_per.get('per_avg_players', []),
                'player_team_per|season|avg|away': away_per.get('per_avg_players', []),
                'player_team_per|season|weighted_MPG|home': home_per.get('per_weighted_players', []),
                'player_team_per|season|weighted_MPG|away': away_per.get('per_weighted_players', []),
                'player_starters_per|season|avg|home': home_per.get('starters_players', []),
                'player_starters_per|season|avg|away': away_per.get('starters_players', []),
                'player_per_1|season|top1_avg|home': home_per.get('per1_player', []),
                'player_per_1|season|top1_avg|away': away_per.get('per1_player', []),
                'player_per_2|season|top1_avg|home': home_per.get('per2_player', []),
                'player_per_2|season|top1_avg|away': away_per.get('per2_player', []),
                'player_per_3|season|top1_avg|home': home_per.get('per3_player', []),
                'player_per_3|season|top1_avg|away': away_per.get('per3_player', [])
            },
            
            # New features for master training
            'player_per_1|none|weighted_MIN_REC|home': home_per.get('per1_recency_weighted', 0),
            'player_per_1|none|weighted_MIN_REC|away': away_per.get('per1_recency_weighted', 0),
            'player_per_1|none|weighted_MIN_REC|diff': home_per.get('per1_recency_weighted', 0) - away_per.get('per1_recency_weighted', 0),
            
            'player_per_1|season|raw|home': home_per['per1'],  # Using existing aggregated PER
            'player_per_1|season|raw|away': away_per['per1'],
            'player_per_1|season|raw|diff': home_per['per1'] - away_per['per1'],
            
            'player_per_2|season|raw|home': home_per['per2'],  # Using existing aggregated PER
            'player_per_2|season|raw|away': away_per['per2'],
            'player_per_2|season|raw|diff': home_per['per2'] - away_per['per2'],
            
            'player_per_3|season|raw|home': home_per['per3'],  # Using existing aggregated PER
            'player_per_3|season|raw|away': away_per['per3'],
            'player_per_3|season|raw|diff': home_per['per3'] - away_per['per3'],

            # =========================================================================
            # NEW PLAYER SUBSET FEATURES (from documentation/player_feature_updates.md)
            # =========================================================================

            # player_starter_per: Average PER for starters
            'player_starter_per|season|avg|home': home_per.get('player_starter_per_avg', 0),
            'player_starter_per|season|avg|away': away_per.get('player_starter_per_avg', 0),
            'player_starter_per|season|avg|diff': home_per.get('player_starter_per_avg', 0) - away_per.get('player_starter_per_avg', 0),

            # player_bench_per: Weighted PER for bench players
            'player_bench_per|season|weighted_MPG|home': home_per.get('player_bench_per_weighted_MPG', 0),
            'player_bench_per|season|weighted_MPG|away': away_per.get('player_bench_per_weighted_MPG', 0),
            'player_bench_per|season|weighted_MPG|diff': home_per.get('player_bench_per_weighted_MPG', 0) - away_per.get('player_bench_per_weighted_MPG', 0),

            'player_bench_per|season|weighted_MIN_REC(k=35)|home': home_per.get('player_bench_per_weighted_MIN_REC_k35', 0),
            'player_bench_per|season|weighted_MIN_REC(k=35)|away': away_per.get('player_bench_per_weighted_MIN_REC_k35', 0),
            'player_bench_per|season|weighted_MIN_REC(k=35)|diff': home_per.get('player_bench_per_weighted_MIN_REC_k35', 0) - away_per.get('player_bench_per_weighted_MIN_REC_k35', 0),

            'player_bench_per|season|weighted_MIN_REC(k=40)|home': home_per.get('player_bench_per_weighted_MIN_REC_k40', 0),
            'player_bench_per|season|weighted_MIN_REC(k=40)|away': away_per.get('player_bench_per_weighted_MIN_REC_k40', 0),
            'player_bench_per|season|weighted_MIN_REC(k=40)|diff': home_per.get('player_bench_per_weighted_MIN_REC_k40', 0) - away_per.get('player_bench_per_weighted_MIN_REC_k40', 0),

            'player_bench_per|season|weighted_MIN_REC(k=45)|home': home_per.get('player_bench_per_weighted_MIN_REC_k45', 0),
            'player_bench_per|season|weighted_MIN_REC(k=45)|away': away_per.get('player_bench_per_weighted_MIN_REC_k45', 0),
            'player_bench_per|season|weighted_MIN_REC(k=45)|diff': home_per.get('player_bench_per_weighted_MIN_REC_k45', 0) - away_per.get('player_bench_per_weighted_MIN_REC_k45', 0),

            'player_bench_per|season|weighted_MIN_REC(k=50)|home': home_per.get('player_bench_per_weighted_MIN_REC_k50', 0),
            'player_bench_per|season|weighted_MIN_REC(k=50)|away': away_per.get('player_bench_per_weighted_MIN_REC_k50', 0),
            'player_bench_per|season|weighted_MIN_REC(k=50)|diff': home_per.get('player_bench_per_weighted_MIN_REC_k50', 0) - away_per.get('player_bench_per_weighted_MIN_REC_k50', 0),

            # player_rotation_per: Weighted PER for rotation players
            'player_rotation_per|season|weighted_MPG|home': home_per.get('player_rotation_per_weighted_MPG', 0),
            'player_rotation_per|season|weighted_MPG|away': away_per.get('player_rotation_per_weighted_MPG', 0),
            'player_rotation_per|season|weighted_MPG|diff': home_per.get('player_rotation_per_weighted_MPG', 0) - away_per.get('player_rotation_per_weighted_MPG', 0),

            'player_rotation_per|season|weighted_MIN_REC(k=20)|home': home_per.get('player_rotation_per_weighted_MIN_REC_k20', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=20)|away': away_per.get('player_rotation_per_weighted_MIN_REC_k20', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=20)|diff': home_per.get('player_rotation_per_weighted_MIN_REC_k20', 0) - away_per.get('player_rotation_per_weighted_MIN_REC_k20', 0),

            'player_rotation_per|season|weighted_MIN_REC(k=25)|home': home_per.get('player_rotation_per_weighted_MIN_REC_k25', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=25)|away': away_per.get('player_rotation_per_weighted_MIN_REC_k25', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=25)|diff': home_per.get('player_rotation_per_weighted_MIN_REC_k25', 0) - away_per.get('player_rotation_per_weighted_MIN_REC_k25', 0),

            'player_rotation_per|season|weighted_MIN_REC(k=30)|home': home_per.get('player_rotation_per_weighted_MIN_REC_k30', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=30)|away': away_per.get('player_rotation_per_weighted_MIN_REC_k30', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=30)|diff': home_per.get('player_rotation_per_weighted_MIN_REC_k30', 0) - away_per.get('player_rotation_per_weighted_MIN_REC_k30', 0),

            'player_rotation_per|season|weighted_MIN_REC(k=35)|home': home_per.get('player_rotation_per_weighted_MIN_REC_k35', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=35)|away': away_per.get('player_rotation_per_weighted_MIN_REC_k35', 0),
            'player_rotation_per|season|weighted_MIN_REC(k=35)|diff': home_per.get('player_rotation_per_weighted_MIN_REC_k35', 0) - away_per.get('player_rotation_per_weighted_MIN_REC_k35', 0),

            # player_starter_bench_per_gap: Starter PER minus Bench PER
            'player_starter_bench_per_gap|season|derived(k=35)|home': home_per.get('player_starter_bench_per_gap_derived_k35', 0),
            'player_starter_bench_per_gap|season|derived(k=35)|away': away_per.get('player_starter_bench_per_gap_derived_k35', 0),
            'player_starter_bench_per_gap|season|derived(k=35)|diff': home_per.get('player_starter_bench_per_gap_derived_k35', 0) - away_per.get('player_starter_bench_per_gap_derived_k35', 0),

            'player_starter_bench_per_gap|season|derived(k=40)|home': home_per.get('player_starter_bench_per_gap_derived_k40', 0),
            'player_starter_bench_per_gap|season|derived(k=40)|away': away_per.get('player_starter_bench_per_gap_derived_k40', 0),
            'player_starter_bench_per_gap|season|derived(k=40)|diff': home_per.get('player_starter_bench_per_gap_derived_k40', 0) - away_per.get('player_starter_bench_per_gap_derived_k40', 0),

            'player_starter_bench_per_gap|season|derived(k=45)|home': home_per.get('player_starter_bench_per_gap_derived_k45', 0),
            'player_starter_bench_per_gap|season|derived(k=45)|away': away_per.get('player_starter_bench_per_gap_derived_k45', 0),
            'player_starter_bench_per_gap|season|derived(k=45)|diff': home_per.get('player_starter_bench_per_gap_derived_k45', 0) - away_per.get('player_starter_bench_per_gap_derived_k45', 0),

            'player_starter_bench_per_gap|season|derived(k=50)|home': home_per.get('player_starter_bench_per_gap_derived_k50', 0),
            'player_starter_bench_per_gap|season|derived(k=50)|away': away_per.get('player_starter_bench_per_gap_derived_k50', 0),
            'player_starter_bench_per_gap|season|derived(k=50)|diff': home_per.get('player_starter_bench_per_gap_derived_k50', 0) - away_per.get('player_starter_bench_per_gap_derived_k50', 0),

            # player_star_score: Star score (PER × MPG) aggregations
            'player_star_score|season|top1|home': home_per.get('player_star_score_top1', 0),
            'player_star_score|season|top1|away': away_per.get('player_star_score_top1', 0),
            'player_star_score|season|top1|diff': home_per.get('player_star_score_top1', 0) - away_per.get('player_star_score_top1', 0),

            'player_star_score|season|top3_avg|home': home_per.get('player_star_score_top3_avg', 0),
            'player_star_score|season|top3_avg|away': away_per.get('player_star_score_top3_avg', 0),
            'player_star_score|season|top3_avg|diff': home_per.get('player_star_score_top3_avg', 0) - away_per.get('player_star_score_top3_avg', 0),

            'player_star_score|season|top3_sum|home': home_per.get('player_star_score_top3_sum', 0),
            'player_star_score|season|top3_sum|away': away_per.get('player_star_score_top3_sum', 0),
            'player_star_score|season|top3_sum|diff': home_per.get('player_star_score_top3_sum', 0) - away_per.get('player_star_score_top3_sum', 0),

            'player_star_score|season|top1_MIN_REC(k=20)|home': home_per.get('player_star_score_top1_MIN_REC_k20', 0),
            'player_star_score|season|top1_MIN_REC(k=20)|away': away_per.get('player_star_score_top1_MIN_REC_k20', 0),
            'player_star_score|season|top1_MIN_REC(k=20)|diff': home_per.get('player_star_score_top1_MIN_REC_k20', 0) - away_per.get('player_star_score_top1_MIN_REC_k20', 0),

            'player_star_score|season|top1_MIN_REC(k=25)|home': home_per.get('player_star_score_top1_MIN_REC_k25', 0),
            'player_star_score|season|top1_MIN_REC(k=25)|away': away_per.get('player_star_score_top1_MIN_REC_k25', 0),
            'player_star_score|season|top1_MIN_REC(k=25)|diff': home_per.get('player_star_score_top1_MIN_REC_k25', 0) - away_per.get('player_star_score_top1_MIN_REC_k25', 0),

            'player_star_score|season|top1_MIN_REC(k=30)|home': home_per.get('player_star_score_top1_MIN_REC_k30', 0),
            'player_star_score|season|top1_MIN_REC(k=30)|away': away_per.get('player_star_score_top1_MIN_REC_k30', 0),
            'player_star_score|season|top1_MIN_REC(k=30)|diff': home_per.get('player_star_score_top1_MIN_REC_k30', 0) - away_per.get('player_star_score_top1_MIN_REC_k30', 0),

            'player_star_score|season|top1_MIN_REC(k=35)|home': home_per.get('player_star_score_top1_MIN_REC_k35', 0),
            'player_star_score|season|top1_MIN_REC(k=35)|away': away_per.get('player_star_score_top1_MIN_REC_k35', 0),
            'player_star_score|season|top1_MIN_REC(k=35)|diff': home_per.get('player_star_score_top1_MIN_REC_k35', 0) - away_per.get('player_star_score_top1_MIN_REC_k35', 0),

            'player_star_score|season|top3_MIN_REC(k=20)|home': home_per.get('player_star_score_top3_MIN_REC_k20', 0),
            'player_star_score|season|top3_MIN_REC(k=20)|away': away_per.get('player_star_score_top3_MIN_REC_k20', 0),
            'player_star_score|season|top3_MIN_REC(k=20)|diff': home_per.get('player_star_score_top3_MIN_REC_k20', 0) - away_per.get('player_star_score_top3_MIN_REC_k20', 0),

            'player_star_score|season|top3_MIN_REC(k=25)|home': home_per.get('player_star_score_top3_MIN_REC_k25', 0),
            'player_star_score|season|top3_MIN_REC(k=25)|away': away_per.get('player_star_score_top3_MIN_REC_k25', 0),
            'player_star_score|season|top3_MIN_REC(k=25)|diff': home_per.get('player_star_score_top3_MIN_REC_k25', 0) - away_per.get('player_star_score_top3_MIN_REC_k25', 0),

            'player_star_score|season|top3_MIN_REC(k=30)|home': home_per.get('player_star_score_top3_MIN_REC_k30', 0),
            'player_star_score|season|top3_MIN_REC(k=30)|away': away_per.get('player_star_score_top3_MIN_REC_k30', 0),
            'player_star_score|season|top3_MIN_REC(k=30)|diff': home_per.get('player_star_score_top3_MIN_REC_k30', 0) - away_per.get('player_star_score_top3_MIN_REC_k30', 0),

            'player_star_score|season|top3_MIN_REC(k=35)|home': home_per.get('player_star_score_top3_MIN_REC_k35', 0),
            'player_star_score|season|top3_MIN_REC(k=35)|away': away_per.get('player_star_score_top3_MIN_REC_k35', 0),
            'player_star_score|season|top3_MIN_REC(k=35)|diff': home_per.get('player_star_score_top3_MIN_REC_k35', 0) - away_per.get('player_star_score_top3_MIN_REC_k35', 0),

            # player_star_share: Star concentration metrics
            'player_star_share|season|top1_share|home': home_per.get('player_star_share_top1_share', 0),
            'player_star_share|season|top1_share|away': away_per.get('player_star_share_top1_share', 0),
            'player_star_share|season|top1_share|diff': home_per.get('player_star_share_top1_share', 0) - away_per.get('player_star_share_top1_share', 0),

            'player_star_share|season|top3_share|home': home_per.get('player_star_share_top3_share', 0),
            'player_star_share|season|top3_share|away': away_per.get('player_star_share_top3_share', 0),
            'player_star_share|season|top3_share|diff': home_per.get('player_star_share_top3_share', 0) - away_per.get('player_star_share_top3_share', 0),

            'player_star_share|season|top1_share_MIN_REC(k=20)|home': home_per.get('player_star_share_top1_share_MIN_REC_k20', 0),
            'player_star_share|season|top1_share_MIN_REC(k=20)|away': away_per.get('player_star_share_top1_share_MIN_REC_k20', 0),
            'player_star_share|season|top1_share_MIN_REC(k=20)|diff': home_per.get('player_star_share_top1_share_MIN_REC_k20', 0) - away_per.get('player_star_share_top1_share_MIN_REC_k20', 0),

            'player_star_share|season|top1_share_MIN_REC(k=25)|home': home_per.get('player_star_share_top1_share_MIN_REC_k25', 0),
            'player_star_share|season|top1_share_MIN_REC(k=25)|away': away_per.get('player_star_share_top1_share_MIN_REC_k25', 0),
            'player_star_share|season|top1_share_MIN_REC(k=25)|diff': home_per.get('player_star_share_top1_share_MIN_REC_k25', 0) - away_per.get('player_star_share_top1_share_MIN_REC_k25', 0),

            'player_star_share|season|top1_share_MIN_REC(k=30)|home': home_per.get('player_star_share_top1_share_MIN_REC_k30', 0),
            'player_star_share|season|top1_share_MIN_REC(k=30)|away': away_per.get('player_star_share_top1_share_MIN_REC_k30', 0),
            'player_star_share|season|top1_share_MIN_REC(k=30)|diff': home_per.get('player_star_share_top1_share_MIN_REC_k30', 0) - away_per.get('player_star_share_top1_share_MIN_REC_k30', 0),

            'player_star_share|season|top1_share_MIN_REC(k=35)|home': home_per.get('player_star_share_top1_share_MIN_REC_k35', 0),
            'player_star_share|season|top1_share_MIN_REC(k=35)|away': away_per.get('player_star_share_top1_share_MIN_REC_k35', 0),
            'player_star_share|season|top1_share_MIN_REC(k=35)|diff': home_per.get('player_star_share_top1_share_MIN_REC_k35', 0) - away_per.get('player_star_share_top1_share_MIN_REC_k35', 0),

            'player_star_share|season|top3_share_MIN_REC(k=20)|home': home_per.get('player_star_share_top3_share_MIN_REC_k20', 0),
            'player_star_share|season|top3_share_MIN_REC(k=20)|away': away_per.get('player_star_share_top3_share_MIN_REC_k20', 0),
            'player_star_share|season|top3_share_MIN_REC(k=20)|diff': home_per.get('player_star_share_top3_share_MIN_REC_k20', 0) - away_per.get('player_star_share_top3_share_MIN_REC_k20', 0),

            'player_star_share|season|top3_share_MIN_REC(k=25)|home': home_per.get('player_star_share_top3_share_MIN_REC_k25', 0),
            'player_star_share|season|top3_share_MIN_REC(k=25)|away': away_per.get('player_star_share_top3_share_MIN_REC_k25', 0),
            'player_star_share|season|top3_share_MIN_REC(k=25)|diff': home_per.get('player_star_share_top3_share_MIN_REC_k25', 0) - away_per.get('player_star_share_top3_share_MIN_REC_k25', 0),

            'player_star_share|season|top3_share_MIN_REC(k=30)|home': home_per.get('player_star_share_top3_share_MIN_REC_k30', 0),
            'player_star_share|season|top3_share_MIN_REC(k=30)|away': away_per.get('player_star_share_top3_share_MIN_REC_k30', 0),
            'player_star_share|season|top3_share_MIN_REC(k=30)|diff': home_per.get('player_star_share_top3_share_MIN_REC_k30', 0) - away_per.get('player_star_share_top3_share_MIN_REC_k30', 0),

            'player_star_share|season|top3_share_MIN_REC(k=35)|home': home_per.get('player_star_share_top3_share_MIN_REC_k35', 0),
            'player_star_share|season|top3_share_MIN_REC(k=35)|away': away_per.get('player_star_share_top3_share_MIN_REC_k35', 0),
            'player_star_share|season|top3_share_MIN_REC(k=35)|diff': home_per.get('player_star_share_top3_share_MIN_REC_k35', 0) - away_per.get('player_star_share_top3_share_MIN_REC_k35', 0),

            # player_star_score_all: Top star from {USAGE_PLAYERS} (includes injured in sorting)
            'player_star_score_all|season|top1|home': home_per.get('player_star_score_all_top1', 0),
            'player_star_score_all|season|top1|away': away_per.get('player_star_score_all_top1', 0),
            'player_star_score_all|season|top1|diff': home_per.get('player_star_score_all_top1', 0) - away_per.get('player_star_score_all_top1', 0),

            # player_rotation_count: Depth indicator
            'player_rotation_count|season|raw|home': home_per.get('player_rotation_count', 0),
            'player_rotation_count|season|raw|away': away_per.get('player_rotation_count', 0),
            'player_rotation_count|season|raw|diff': home_per.get('player_rotation_count', 0) - away_per.get('player_rotation_count', 0),

            # player_continuity: Rotation cohesion proxy
            'player_continuity|season|avg|home': home_per.get('player_continuity_avg', 0),
            'player_continuity|season|avg|away': away_per.get('player_continuity_avg', 0),
            'player_continuity|season|avg|diff': home_per.get('player_continuity_avg', 0) - away_per.get('player_continuity_avg', 0),

            # Internal metadata (not used as features)
            'homePlayerCount': home_per['player_count'],
            'awayPlayerCount': away_per['player_count']
        }
        
        # Store in in-memory cache
        self._per_features_cache[cache_key] = features
        
        return features
    
    def save_per_features_to_db(self, season: str = None):
        """
        Save all computed PER features from memory to MongoDB.
        
        Args:
            season: If provided, only save features for this season
        """
        # Group features by season
        season_features = defaultdict(dict)
        
        for cache_key, features in self._per_features_cache.items():
            parts = cache_key.split('|')
            if len(parts) >= 4:
                key_season = parts[0]
                game_key = '|'.join(parts[1:])
                
                if season is None or key_season == season:
                    season_features[key_season][game_key] = features
        
        # Save each season
        for szn, game_features in season_features.items():
            self._save_per_cache(szn, game_features)
            print(f"  Saved {len(game_features)} PER features for {szn}")
    
    def clear_cache(self, season: str = None):
        """Clear cached PER features from MongoDB."""
        if season:
            self.db[CACHED_PER_COLLECTION].delete_one({'season': season})
        else:
            self.db[CACHED_PER_COLLECTION].delete_many({})
        
        # Clear in-memory cache
        if season:
            self._per_features_cache = {
                k: v for k, v in self._per_features_cache.items()
                if not k.startswith(f"{season}|")
            }
        else:
            self._per_features_cache = {}
    
    # =========================================================================
    # DATABASE QUERY HELPERS
    # =========================================================================
    
    def get_player_game_stats(self, player_id: str, game_id: str) -> Optional[dict]:
        """Get player stats for a specific game."""
        return self._players_repo.find_one({
            'player_id': player_id,
            'game_id': game_id
        })

    def get_team_game_stats(self, game_id: str, is_home: bool) -> Optional[dict]:
        """Get team stats for a specific game."""
        game = self._games_repo.find_one({'game_id': game_id})
        if not game:
            return None
        return game['homeTeam'] if is_home else game['awayTeam']
    
    def get_team_players_before_date(
        self,
        team: str,
        season: str,
        before_date: str,
        min_games: int = 3
    ) -> List[dict]:
        """Get all players who have played for a team before a given date."""
        return self._get_team_players_before_date_cached(team, season, before_date, min_games)
    
    def get_player_recency_weighted_per(
        self,
        player_id: str,
        team: str,
        season: str,
        before_date: str,
        recency_decay_k: float = 14.0
    ) -> Optional[float]:
        """
        Get a player's recency-weighted PER using game-level PER calculations.
        
        For each game the player played:
        - Compute PER using that game's box score and team context
        - Weight by: MIN_g * exp(-days_since_game / k)
        - Aggregate: Σ (PER_g * w_g) / Σ w_g
        
        Args:
            player_id: Player ID (as string)
            team: Team name
            season: Season string
            before_date: Date string (YYYY-MM-DD) - stats up to but not including this date
            recency_decay_k: Decay constant for recency weighting (default 14.0)
        
        Returns:
            Recency-weighted PER (float) or None if player not found or insufficient data
        """
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        # Get all games for this player before the date
        player_games = None
        key = (team, season)

        # Try preloaded cache first if available
        if self._preloaded and self._player_stats_cache and key in self._player_stats_cache:
            player_games = [
                pg for pg in self._player_stats_cache[key]
                if (pg.get('player_id') == player_id or str(pg.get('player_id')) == str(player_id)) and
                   pg.get('date') and pg['date'] < before_date and
                   pg.get('stats', {}).get('min', 0) > 0
            ]

        # Fall back to DB query if cache miss or not preloaded
        if player_games is None:
            player_games = self._players_repo.find({
                'player_id': player_id,
                'team': team,
                'season': season,
                'date': {'$lt': before_date},
                'stats.min': {'$gt': 0},
                'game_type': {'$nin': self._exclude_game_types}
            })
        
        if not player_games:
            return None
        
        # Sort by date
        player_games.sort(key=lambda x: x.get('date', ''))
        
        # Get league constants (same for all games in season)
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            return None
        
        lg_pace = league_constants.get('lg_pace', 95)

        # PERFORMANCE FIX: Use cached season-level lg_aper
        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                lg_aper = self.compute_league_average_aper(season, None)
                self._lg_aper_by_season[season] = lg_aper
        use_normalization = lg_aper > 0

        # Parse game date for recency calculation
        try:
            game_date_obj = datetime.strptime(before_date, '%Y-%m-%d').date()
        except:
            return None
        
        # Calculate PER for each game and apply recency weighting
        weighted_per_sum = 0.0
        total_weight = 0.0
        
        for pg in player_games:
            game_date_str = pg.get('date')
            if not game_date_str:
                continue
            
            try:
                pg_date_obj = datetime.strptime(str(game_date_str), '%Y-%m-%d').date() if isinstance(game_date_str, str) else game_date_str
                if isinstance(pg_date_obj, str):
                    pg_date_obj = datetime.strptime(pg_date_obj, '%Y-%m-%d').date()
            except:
                continue
            
            # Calculate days since game
            days_since = (game_date_obj - pg_date_obj).days
            if days_since < 0:
                continue  # Skip future games (shouldn't happen, but safety check)
            
            # Get player stats for this game
            stats = pg.get('stats', {})
            min_played = stats.get('min', 0)
            if min_played == 0:
                continue
            
            # Get team stats for this specific game
            game_id = pg.get('game_id')
            if not game_id:
                continue
            
            game_doc = self._games_repo.find_one({'game_id': game_id})
            if not game_doc:
                continue
            
            # Determine if player was on home or away team
            is_home = pg.get('home', False)
            team_data = game_doc.get('homeTeam' if is_home else 'awayTeam', {})
            
            # Get team stats for this game
            team_stats = {
                'assists': team_data.get('assists', 1),
                'FG_made': team_data.get('FG_made', 1)
            }
            
            # Get team pace for this game's team
            team_pace = get_team_pace(season, team, self.db, league=self.league)
            if team_pace == 0:
                team_pace = lg_pace
            
            # Calculate PER for this game
            player_game_stats = {
                'min': min_played,
                'pts': stats.get('pts', 0),
                'fg_made': stats.get('fg_made', 0),
                'fg_att': stats.get('fg_att', 0),
                'three_made': stats.get('three_made', 0),
                'ft_made': stats.get('ft_made', 0),
                'ft_att': stats.get('ft_att', 0),
                'reb': stats.get('reb', 0),
                'oreb': stats.get('oreb', 0),
                'ast': stats.get('ast', 0),
                'stl': stats.get('stl', 0),
                'blk': stats.get('blk', 0),
                'to': stats.get('to', 0),
                'pf': stats.get('pf', 0)
            }
            
            uper = self.compute_uper(player_game_stats, team_stats, league_constants)
            aper = self.compute_aper(uper, team_pace, lg_pace)
            
            # Normalize to PER
            if use_normalization:
                per = self.compute_per(aper, lg_aper)
            else:
                per = aper
            
            # Calculate weight: MIN * exp(-days_since / k)
            recency_weight = np.exp(-days_since / recency_decay_k)
            weight = min_played * recency_weight
            
            # Add to weighted sum
            weighted_per_sum += per * weight
            total_weight += weight
        
        # Return weighted average
        if total_weight == 0:
            return None
        
        return weighted_per_sum / total_weight
    
    def get_player_recency_weighted_per(
        self,
        player_id: str,
        team: str,
        season: str,
        before_date: str,
        recency_decay_k: float = 14.0
    ) -> Optional[float]:
        """
        Get a player's recency-weighted PER using game-level PER calculations.
        
        For each game the player played:
        - Compute PER using that game's box score and team context
        - Weight by: MIN_g * exp(-days_since_game / k)
        - Aggregate: Σ (PER_g * w_g) / Σ w_g
        
        Args:
            player_id: Player ID (as string)
            team: Team name
            season: Season string
            before_date: Date string (YYYY-MM-DD) - stats up to but not including this date
            recency_decay_k: Decay constant for recency weighting (default 14.0)
        
        Returns:
            Recency-weighted PER (float) or None if player not found or insufficient data
        """
        from datetime import datetime
        
        logger = logging.getLogger(__name__)
        
        # Get all games for this player before the date
        player_games = None
        key = (team, season)

        # Try preloaded cache first if available
        if self._preloaded and self._player_stats_cache and key in self._player_stats_cache:
            player_games = [
                pg for pg in self._player_stats_cache[key]
                if (pg.get('player_id') == player_id or str(pg.get('player_id')) == str(player_id)) and
                   pg.get('date') and pg['date'] < before_date and
                   pg.get('stats', {}).get('min', 0) > 0
            ]

        # Fall back to DB query if cache miss or not preloaded
        if player_games is None:
            player_games = self._players_repo.find({
                'player_id': player_id,
                'team': team,
                'season': season,
                'date': {'$lt': before_date},
                'stats.min': {'$gt': 0},
                'game_type': {'$nin': self._exclude_game_types}
            })
        
        if not player_games:
            return None
        
        # Sort by date
        player_games.sort(key=lambda x: x.get('date', ''))
        
        # Get league constants (same for all games in season)
        league_constants = get_league_constants(season, self.db, league=self.league)
        if not league_constants:
            return None
        
        lg_pace = league_constants.get('lg_pace', 95)

        # PERFORMANCE FIX: Use cached season-level lg_aper
        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                lg_aper = self.compute_league_average_aper(season, None)
                self._lg_aper_by_season[season] = lg_aper
        use_normalization = lg_aper > 0

        # Parse game date for recency calculation
        try:
            game_date_obj = datetime.strptime(before_date, '%Y-%m-%d').date()
        except:
            return None
        
        # Calculate PER for each game and apply recency weighting
        weighted_per_sum = 0.0
        total_weight = 0.0
        
        for pg in player_games:
            game_date_str = pg.get('date')
            if not game_date_str:
                continue
            
            try:
                pg_date_obj = datetime.strptime(str(game_date_str), '%Y-%m-%d').date() if isinstance(game_date_str, str) else game_date_str
                if isinstance(pg_date_obj, str):
                    pg_date_obj = datetime.strptime(pg_date_obj, '%Y-%m-%d').date()
            except:
                continue
            
            # Calculate days since game
            days_since = (game_date_obj - pg_date_obj).days
            if days_since < 0:
                continue  # Skip future games (shouldn't happen, but safety check)
            
            # Get player stats for this game
            stats = pg.get('stats', {})
            min_played = stats.get('min', 0)
            if min_played == 0:
                continue
            
            # Get team stats for this specific game
            game_id = pg.get('game_id')
            if not game_id:
                continue
            
            game_doc = self._games_repo.find_one({'game_id': game_id})
            if not game_doc:
                continue
            
            # Determine if player was on home or away team
            is_home = pg.get('home', False)
            team_data = game_doc.get('homeTeam' if is_home else 'awayTeam', {})
            
            # Get team stats for this game
            team_stats = {
                'assists': team_data.get('assists', 1),
                'FG_made': team_data.get('FG_made', 1)
            }
            
            # Get team pace for this game's team
            team_pace = get_team_pace(season, team, self.db, league=self.league)
            if team_pace == 0:
                team_pace = lg_pace
            
            # Calculate PER for this game
            player_game_stats = {
                'min': min_played,
                'pts': stats.get('pts', 0),
                'fg_made': stats.get('fg_made', 0),
                'fg_att': stats.get('fg_att', 0),
                'three_made': stats.get('three_made', 0),
                'ft_made': stats.get('ft_made', 0),
                'ft_att': stats.get('ft_att', 0),
                'reb': stats.get('reb', 0),
                'oreb': stats.get('oreb', 0),
                'ast': stats.get('ast', 0),
                'stl': stats.get('stl', 0),
                'blk': stats.get('blk', 0),
                'to': stats.get('to', 0),
                'pf': stats.get('pf', 0)
            }
            
            uper = self.compute_uper(player_game_stats, team_stats, league_constants)
            aper = self.compute_aper(uper, team_pace, lg_pace)
            
            # Normalize to PER
            if use_normalization:
                per = self.compute_per(aper, lg_aper)
            else:
                per = aper
            
            # Calculate weight: MIN * exp(-days_since / k)
            recency_weight = np.exp(-days_since / recency_decay_k)
            weight = min_played * recency_weight
            
            # Add to weighted sum
            weighted_per_sum += per * weight
            total_weight += weight
        
        # Return weighted average
        if total_weight == 0:
            return None
        
        return weighted_per_sum / total_weight
    
    def get_player_per_before_date(
        self,
        player_id: str,
        team: str,
        season: str,
        before_date: str,
        cross_team: bool = False
    ) -> Optional[float]:
        """
        Get a single player's PER (season-to-date) before a given date.
        Uses cache to avoid recalculating PER for the same player/team/season/date.

        Args:
            player_id: Player ID (as string)
            team: Team name
            season: Season string
            before_date: Date string (YYYY-MM-DD) - stats up to but not including this date
            cross_team: If True, aggregate stats across ALL teams the player played for
                       (for traded players). Default False for backward compatibility.

        Returns:
            Player's PER (float) or None if player not found or insufficient data
        """
        # Check cache first - this is critical for performance when computing PER for many injured players
        cache_key = (str(player_id), team, season, before_date, cross_team)
        if cache_key in self._player_per_cache:
            cached_per = self._player_per_cache[cache_key]
            # Return None if cached value indicates no data (we use None for this)
            return cached_per if cached_per is not None else None

        # Get player data - either cross-team or team-specific
        if cross_team:
            # Aggregate stats across ALL teams for traded player support
            players = self._get_players_cross_team_by_ids([str(player_id)], season, before_date, min_games=1)
        else:
            # Get all players for the team before the date
            players = self._get_team_players_before_date_cached(team, season, before_date, min_games=1)

        # Find the specific player
        player_data = None
        for p in players:
            if str(p['_id']) == str(player_id):
                player_data = p
                break

        if not player_data or player_data['total_min'] == 0:
            # Cache None to indicate no data found
            self._player_per_cache[cache_key] = None
            return None
        
        # Get league constants and team stats
        league_constants = get_league_constants(season, self.db, league=self.league)
        team_stats = self._get_team_stats_before_date_cached(team, season, before_date)
        
        # Get team pace
        team_pace = get_team_pace(season, team, self.db, league=self.league)
        lg_pace = get_team_pace(season, None, self.db, league=self.league)  # League average pace
        if team_pace == 0:
            team_pace = lg_pace
        if lg_pace == 0:
            lg_pace = 100.0  # Fallback
        
        # Aggregate stats for PER calculation
        agg_stats = {
            'min': player_data['total_min'],
            'pts': player_data['total_pts'],
            'fg_made': player_data['total_fg_made'],
            'fg_att': player_data['total_fg_att'],
            'three_made': player_data['total_three_made'],
            'ft_made': player_data['total_ft_made'],
            'ft_att': player_data['total_ft_att'],
            'reb': player_data['total_reb'],
            'oreb': player_data['total_oreb'],
            'ast': player_data['total_ast'],
            'stl': player_data['total_stl'],
            'blk': player_data['total_blk'],
            'to': player_data['total_to'],
            'pf': player_data['total_pf']
        }
        
        # Compute PER
        uper = self.compute_uper(agg_stats, team_stats, league_constants)
        aper = self.compute_aper(uper, team_pace, lg_pace)
        
        # PERFORMANCE FIX: Use cached season-level lg_aper
        if season in self._lg_aper_by_season:
            lg_aper = self._lg_aper_by_season[season]
        else:
            cached_stats = get_season_stats_with_fallback(season, self.db, league=self.league)
            if cached_stats and 'lg_aper' in cached_stats:
                lg_aper = cached_stats['lg_aper']
                self._lg_aper_by_season[season] = lg_aper
            else:
                lg_aper = self.compute_league_average_aper(season, None)
                self._lg_aper_by_season[season] = lg_aper
        if lg_aper > 0:
            per = self.compute_per(aper, lg_aper)
        else:
            per = aper  # Use aPER if normalization not available
        
        # Cache the computed PER value
        self._player_per_cache[cache_key] = per
        
        return per
    
    def _get_approximate_team_stats(
        self,
        team: str,
        season: str,
        before_date: str
    ) -> dict:
        """Get approximate team season totals for PER calculation."""
        return self._get_team_stats_before_date_cached(team, season, before_date)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='PER Calculator CLI')
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Cache command
    cache_parser = subparsers.add_parser('cache', help='Manage PER feature cache')
    cache_parser.add_argument('action', choices=['show', 'clear', 'rebuild'],
                              help='Action to perform')
    cache_parser.add_argument('--season', '-s', type=str, help='Limit to specific season')
    
    # Team PER command
    team_parser = subparsers.add_parser('team', help='Get team PER features')
    team_parser.add_argument('team', type=str, help='Team name')
    team_parser.add_argument('--season', '-s', type=str, required=True, help='Season')
    team_parser.add_argument('--before', '-b', type=str, required=True, help='Before date')
    
    # Game PER command
    game_parser = subparsers.add_parser('game', help='Get game PER features')
    game_parser.add_argument('home', type=str, help='Home team')
    game_parser.add_argument('away', type=str, help='Away team')
    game_parser.add_argument('--season', '-s', type=str, required=True, help='Season')
    game_parser.add_argument('--date', '-d', type=str, required=True, help='Game date')
    
    args = parser.parse_args()
    
    if args.command == 'cache':
        calc = PERCalculator(preload=False)
        
        if args.action == 'show':
            cached = list(calc.db[CACHED_PER_COLLECTION].find({}))
            print(f"\nCached PER Features:")
            for doc in cached:
                print(f"  {doc['season']}: {doc.get('game_count', 0)} games, updated {doc.get('updated_at', 'N/A')}")
            if not cached:
                print("  (no cached data)")
        
        elif args.action == 'clear':
            calc.clear_cache(args.season)
            print(f"Cleared PER cache{' for ' + args.season if args.season else ''}")
        
        elif args.action == 'rebuild':
            print("Rebuilding PER cache requires running training. Use train.py train")
    
    elif args.command == 'team':
        print("Loading data...")
        calc = PERCalculator(preload=True)
        result = calc.compute_team_per_features(args.team, args.season, args.before)
        if result:
            print(f"\n{args.team} Team PER Features (before {args.before}):")
            print(f"  Team PER Average: {result['per_avg']:.2f}")
            print(f"  Team PER Weighted: {result['per_weighted']:.2f}")
            print(f"  Starters Average: {result['starters_avg']:.2f}")
            print(f"\n  Top 8 Players:")
            for i, p in enumerate(result['players'][:8], 1):
                starter = '*' if p['is_starter'] else ' '
                print(f"    {i}. {starter}{p['player_name']}: {p['aper']:.2f} PER ({p['games']} games, {p['avg_min']:.1f} mpg)")
        else:
            print("No data found")
    
    elif args.command == 'game':
        print("Loading data...")
        calc = PERCalculator(preload=True)
        result = calc.get_game_per_features(args.home, args.away, args.season, args.date)
        if result:
            print(f"\nGame PER Features: {args.away} @ {args.home}")
            print(f"  Home Team PER Avg: {result.get('player_team_per|season|avg|home', 0):.2f}")
            print(f"  Away Team PER Avg: {result.get('player_team_per|season|avg|away', 0):.2f}")
            print(f"  PER Differential: {result.get('player_team_per|season|avg|diff', 0):.2f}")
            print(f"\n  Weighted Differential: {result.get('player_team_per|season|weighted_MPG|diff', 0):.2f}")
            print(f"  Starters Differential: {result.get('player_starters_per|season|avg|diff', 0):.2f}")
        else:
            print("No data found")
    
    else:
        parser.print_help()

