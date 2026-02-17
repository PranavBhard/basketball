"""
Feature Registry - Single Source of Truth for NBA Feature Definitions

This module provides:
1. Complete catalog of all valid stat names with metadata
2. Valid time periods, calc weights, and perspectives
3. Stat categorization (rate stats, side-splittable stats, etc.)
4. DB field mappings for stat computation
5. Feature name generation and validation
6. Feature groupings for semantic organization

All other modules (StatHandlerV2, BballModel, feature_sets, feature_name_parser)
should import from this registry rather than defining their own stat lists.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


class StatCategory(Enum):
    """Categories of stats based on computation method."""
    BASIC = "basic"           # Simple counting stats (points, assists)
    RATE = "rate"             # Efficiency stats computed from ratios (efg, ts)
    NET = "net"               # Opponent-adjusted stats (points_net, efg_net)
    DERIVED = "derived"       # Computed from other stats (pace_interaction)
    SPECIAL = "special"       # Special handling required (elo, injuries)


class CalcWeight(Enum):
    """Valid calculation weight methods."""
    RAW = "raw"                    # Raw aggregation (sum for basic, compute-from-agg for rate)
    AVG = "avg"                    # Per-game average
    WEIGHTED_MPG = "weighted_MPG"  # Weighted by minutes played (for PER)
    HARMONIC_MEAN = "harmonic_mean"  # For pace interaction features
    DERIVED = "derived"            # Computed/derived features
    # Note: blend:* is handled separately as it's a dynamic format


class Perspective(Enum):
    """Valid perspectives for feature computation."""
    DIFF = "diff"    # Home minus away differential
    HOME = "home"    # Absolute home team value
    AWAY = "away"    # Absolute away team value
    NONE = "none"    # Matchup-level (not split by team)


@dataclass
class StatDefinition:
    """Definition of a single statistic with all its metadata."""
    name: str                          # Canonical stat name (e.g., "points", "efg")
    category: StatCategory             # Computation category
    db_field: Optional[str] = None     # MongoDB field name (if different from name)
    description: str = ""              # Human-readable description
    supports_side_split: bool = False  # Can be computed for home/away sides separately
    supports_net: bool = False         # Has opponent-adjusted version (*_net)
    valid_calc_weights: Set[str] = field(default_factory=lambda: {"raw", "avg"})
    valid_time_periods: Set[str] = field(default_factory=set)  # Empty = all periods valid
    valid_perspectives: Set[str] = field(default_factory=set)  # Empty = all perspectives valid (diff/home/away/none)
    requires_aggregation: bool = False  # For rate stats: aggregate before computing


class FeatureRegistry:
    """
    Single Source of Truth for all feature definitions.

    Usage:
        from bball_app.core.feature_registry import FeatureRegistry

        # Get all valid stat names
        stats = FeatureRegistry.get_all_stat_names()

        # Validate a feature name
        valid, error = FeatureRegistry.validate_feature("points|season|avg|diff")

        # Get stat definition
        stat_def = FeatureRegistry.get_stat_definition("efg")

        # Get DB field mapping
        db_field = FeatureRegistry.get_db_field("efg")  # Returns "effective_fg_perc"
    """

    # ==========================================================================
    # CORE STAT DEFINITIONS
    # ==========================================================================

    STAT_DEFINITIONS: Dict[str, StatDefinition] = {
        # ---- Basic Counting Stats ----
        "points": StatDefinition(
            name="points",
            category=StatCategory.BASIC,
            db_field="points",
            description="Total points scored",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},  # std for volatility features
            supports_net=True,
        ),
        "wins": StatDefinition(
            name="wins",
            category=StatCategory.BASIC,
            db_field="homeWon",  # Derived from homeWon boolean
            description="Number of wins",
            supports_side_split=True,
        ),
        "assists": StatDefinition(
            name="assists",
            category=StatCategory.BASIC,
            db_field="assists",
            description="Total assists",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "blocks": StatDefinition(
            name="blocks",
            category=StatCategory.BASIC,
            db_field="blocks",
            description="Total blocks",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "steals": StatDefinition(
            name="steals",
            category=StatCategory.BASIC,
            db_field="steals",
            description="Total steals",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "reb_total": StatDefinition(
            name="reb_total",
            category=StatCategory.BASIC,
            db_field="total_reb",  # Mapped from reb_total
            description="Total rebounds (offensive + defensive)",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "turnovers": StatDefinition(
            name="turnovers",
            category=StatCategory.BASIC,
            db_field="TO_metric",
            description="Turnovers",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "three_made": StatDefinition(
            name="three_made",
            category=StatCategory.BASIC,
            db_field="three_made",
            description="Three-pointers made",
            supports_side_split=True,
            supports_net=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),

        # ---- Rate/Efficiency Stats ----
        "efg": StatDefinition(
            name="efg",
            category=StatCategory.RATE,
            db_field="effective_fg_perc",
            description="Effective field goal percentage",
            supports_side_split=True,
            supports_net=True,
            requires_aggregation=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "ts": StatDefinition(
            name="ts",
            category=StatCategory.RATE,
            db_field="true_shooting_perc",
            description="True shooting percentage",
            supports_side_split=True,
            supports_net=True,
            requires_aggregation=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "three_pct": StatDefinition(
            name="three_pct",
            category=StatCategory.RATE,
            db_field="three_perc",
            description="Three-point percentage",
            supports_side_split=True,
            supports_net=True,
            requires_aggregation=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "off_rtg": StatDefinition(
            name="off_rtg",
            category=StatCategory.RATE,
            db_field="off_rtg",
            description="Offensive rating (points per 100 possessions)",
            supports_side_split=True,
            supports_net=True,
            requires_aggregation=True,
        ),
        "def_rtg": StatDefinition(
            name="def_rtg",
            category=StatCategory.RATE,
            db_field="def_rtg",
            description="Defensive rating (points allowed per 100 possessions)",
            supports_side_split=True,
        ),
        "assists_ratio": StatDefinition(
            name="assists_ratio",
            category=StatCategory.RATE,
            db_field="assists_ratio",
            description="Assist ratio (assists per 100 possessions)",
            supports_side_split=True,
            supports_net=True,
            requires_aggregation=True,
        ),
        "ast_to_ratio": StatDefinition(
            name="ast_to_ratio",
            category=StatCategory.RATE,
            description="Assist-to-turnover ratio (assists / turnovers)",
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "to_metric": StatDefinition(
            name="to_metric",
            category=StatCategory.RATE,
            db_field="TO_metric",
            description="Turnover rate",
            supports_side_split=True,
            requires_aggregation=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),

        # ---- Net Stats (opponent-adjusted) ----
        "points_net": StatDefinition(
            name="points_net",
            category=StatCategory.NET,
            description="Points minus opponent points allowed to teams",
            supports_side_split=True,
        ),
        "efg_net": StatDefinition(
            name="efg_net",
            category=StatCategory.NET,
            description="EFG% minus opponent EFG% against",
            supports_side_split=True,
        ),
        "ts_net": StatDefinition(
            name="ts_net",
            category=StatCategory.NET,
            description="TS% minus opponent TS% against",
            supports_side_split=True,
        ),
        "three_pct_net": StatDefinition(
            name="three_pct_net",
            category=StatCategory.NET,
            description="3P% minus opponent 3P% against",
            supports_side_split=True,
        ),
        "off_rtg_net": StatDefinition(
            name="off_rtg_net",
            category=StatCategory.NET,
            description="Offensive rating minus opponent defensive rating",
            supports_side_split=True,
        ),
        "assists_ratio_net": StatDefinition(
            name="assists_ratio_net",
            category=StatCategory.NET,
            description="Assist ratio minus opponent assist ratio against",
            supports_side_split=True,
        ),
        "three_made_net": StatDefinition(
            name="three_made_net",
            category=StatCategory.NET,
            description="3PM minus opponent 3PM against",
            supports_side_split=True,
        ),

        # ---- Derived/Matchup Stats ----
        "pace": StatDefinition(
            name="pace",
            category=StatCategory.DERIVED,
            db_field="pace",
            description="Pace (possessions per 48 minutes)",
        ),
        "pace_interaction": StatDefinition(
            name="pace_interaction",
            category=StatCategory.DERIVED,
            description="Matchup-level pace interaction (harmonic mean of both teams' pace). Single value per game.",
            valid_calc_weights={"harmonic_mean"},
            # Requires aggregated pace data - 'none' time period is invalid
            valid_time_periods={"season", "games_5", "games_10", "games_12", "games_20", "games_50", "months_1", "months_2"},
            # Matchup-level metric - only 'none' perspective is valid (not home/away/diff)
            valid_perspectives={"none"},
        ),
        "est_possessions": StatDefinition(
            name="est_possessions",
            category=StatCategory.DERIVED,
            description="Estimated possessions for matchup (pace_interaction value). Single value per game.",
            valid_calc_weights={"derived"},
            # Depends on pace_interaction which requires aggregated data
            valid_time_periods={"season", "games_5", "games_10", "games_12", "games_20", "games_50", "months_1", "months_2"},
            # Matchup-level metric - only 'none' perspective is valid (not home/away/diff)
            valid_perspectives={"none"},
        ),
        "exp_points_off": StatDefinition(
            name="exp_points_off",
            category=StatCategory.DERIVED,
            description="Expected offensive points (possessions × offensive efficiency)",
            valid_calc_weights={"derived"},
            # Depends on est_possessions and off_rtg which require aggregated data
            valid_time_periods={"season", "games_5", "games_10", "games_12", "games_20", "games_50", "months_1", "months_2"},
            # Team-specific values: home/away/diff all valid
            valid_perspectives={"home", "away", "diff"},
        ),
        "exp_points_def": StatDefinition(
            name="exp_points_def",
            category=StatCategory.DERIVED,
            description="Expected defensive points allowed (possessions × opponent efficiency)",
            valid_calc_weights={"derived"},
            # Depends on est_possessions and def_rtg which require aggregated data
            valid_time_periods={"season", "games_5", "games_10", "games_12", "games_20", "games_50", "months_1", "months_2"},
            # Team-specific values: home/away/diff all valid
            valid_perspectives={"home", "away", "diff"},
        ),
        "exp_points_matchup": StatDefinition(
            name="exp_points_matchup",
            category=StatCategory.DERIVED,
            description="Expected points from matchup ((exp_points_off + opp_exp_points_def) / 2)",
            valid_calc_weights={"derived"},
            # Depends on exp_points_off/def which require aggregated data
            valid_time_periods={"season", "games_5", "games_10", "games_12", "games_20", "games_50", "months_1", "months_2"},
            # Team-specific values: home/away/diff all valid
            valid_perspectives={"home", "away", "diff"},
        ),

        # ---- Shot Mix Stats ----
        "three_rate": StatDefinition(
            name="three_rate",
            category=StatCategory.DERIVED,
            description="Three-point attempt rate (3PA / FGA)",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "ft_rate": StatDefinition(
            name="ft_rate",
            category=StatCategory.DERIVED,
            description="Free throw rate (FTA / FGA)",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "reb_off_pct": StatDefinition(
            name="reb_off_pct",
            category=StatCategory.RATE,
            description="Offensive rebound percentage",
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "three_pct_allowed": StatDefinition(
            name="three_pct_allowed",
            category=StatCategory.RATE,
            description="Opponent 3-point percentage allowed (defensive metric)",
            valid_calc_weights={"raw", "avg", "std"},
        ),
        "three_pct_matchup": StatDefinition(
            name="three_pct_matchup",
            category=StatCategory.DERIVED,
            description="Three-point matchup metric (team 3PT% vs opp 3PT% defense)",
            valid_calc_weights={"derived"},
            valid_perspectives={"home", "away"},  # No diff - team-specific
        ),

        # ---- Schedule/Situational Stats ----
        "home_court": StatDefinition(
            name="home_court",
            category=StatCategory.SPECIAL,
            description="Binary home court indicator",
            valid_time_periods={"none"},
            valid_calc_weights={"raw"},
            valid_perspectives={"home"},  # Only relevant for home team
        ),
        "travel": StatDefinition(
            name="travel",
            category=StatCategory.SPECIAL,
            description="Travel distance (miles)",
            valid_calc_weights={"raw", "avg", "sum"},
        ),
        "road_games": StatDefinition(
            name="road_games",
            category=StatCategory.SPECIAL,
            description="Away/road games played in period",
        ),
        "b2b": StatDefinition(
            name="b2b",
            category=StatCategory.SPECIAL,
            description="Back-to-back indicator (playing second game in consecutive days)",
            valid_time_periods={"none"},
            valid_calc_weights={"raw"},
        ),
        "first_of_b2b": StatDefinition(
            name="first_of_b2b",
            category=StatCategory.SPECIAL,
            description="First of back-to-back indicator (team plays again tomorrow)",
            valid_time_periods={"none"},
            valid_calc_weights={"raw"},
        ),
        "days_rest": StatDefinition(
            name="days_rest",
            category=StatCategory.SPECIAL,
            description="Days of rest before game",
        ),
        "games_played": StatDefinition(
            name="games_played",
            category=StatCategory.SPECIAL,
            description="Games played in period (sample size)",
        ),
        "postseason": StatDefinition(
            name="postseason",
            category=StatCategory.SPECIAL,
            description="Binary: 1 if game is postseason/playoffs, 0 otherwise",
            supports_side_split=False,
            valid_calc_weights={"binary"},
            valid_time_periods={"none"},
            valid_perspectives={"none"},
        ),

        # ---- Special Stats ----
        "elo": StatDefinition(
            name="elo",
            category=StatCategory.SPECIAL,
            db_field="elo",
            description="Elo rating",
            valid_calc_weights={"raw"},
            valid_time_periods={"season", "none"},  # 'none' for point-in-time elo
        ),
        "inj_impact": StatDefinition(
            name="inj_impact",
            category=StatCategory.SPECIAL,
            description="Injury impact blend: 0.45*severity + 0.35*top1_per + 0.20*rotation",
            valid_calc_weights={"blend:severity:0.45/top1_per:0.35/rotation:0.20"},
        ),
        "inj_severity": StatDefinition(
            name="inj_severity",
            category=StatCategory.SPECIAL,
            description="Proportion of rotation minutes lost: inj_min_lost / team_rotation_mpg",
            valid_calc_weights={"raw"},
            valid_time_periods={"none", "season"},  # 'none' for point-in-time, 'season' for STD weighted avg
        ),
        "inj_min_lost": StatDefinition(
            name="inj_min_lost",
            category=StatCategory.SPECIAL,
            description="Total expected minutes lost to injury (sum of injured players' MPG)",
            valid_calc_weights={"raw"},
        ),
        "inj_per": StatDefinition(
            name="inj_per",
            category=StatCategory.SPECIAL,
            description="PER value lost to injuries. DEPRECATED: top3_sum confounds team quality with injury impact; use inj_per_share|none|top3_sum instead",
            valid_calc_weights={"top1_avg", "top3_sum", "weighted_MIN"},
        ),
        # NORMALIZED INJURY FEATURES (deconfounded from team quality)
        "inj_per_share": StatDefinition(
            name="inj_per_share",
            category=StatCategory.SPECIAL,
            description="Fraction of top-team PER that is injured: inj_top3_sum / team_top3_sum. Measures 'how much of their best talent is missing' instead of raw talent value.",
            valid_calc_weights={"top3_sum", "top1_avg"},
        ),
        "inj_per_weighted_share": StatDefinition(
            name="inj_per_weighted_share",
            category=StatCategory.SPECIAL,
            description="Normalized weighted PER lost: inj_per_weighted_MIN / team_per_weighted_MPG. Minutes-weighted share of team talent missing.",
            valid_calc_weights={"weighted_MIN"},
        ),
        "inj_rotation_per": StatDefinition(
            name="inj_rotation_per",
            category=StatCategory.SPECIAL,
            description="Proportion of rotation players injured: injured_rotation_mpg / total_rotation_mpg",
            valid_calc_weights={"raw"},
        ),

        # ---- New Injury Features (from player_feature_updates.md) ----
        # These use {USAGE_PLAYERS} and {INJ_PLAYERS} sets

        "inj_star_share": StatDefinition(
            name="inj_star_share",
            category=StatCategory.SPECIAL,
            description="Injured star's share of top-3 star mass (0 if star not out): star_score(STAR) / sum(top3 star_scores)",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},
        ),
        "inj_star_score_share": StatDefinition(
            name="inj_star_score_share",
            category=StatCategory.SPECIAL,
            description="Top-3 injured star mass as share of team top-3: sum(inj_top3 star_scores) / sum(team_top3 star_scores), clipped to [0, 1.5]",
            valid_calc_weights={"top3_sum"},
            valid_time_periods={"none"},
        ),
        "inj_top1_star_out": StatDefinition(
            name="inj_top1_star_out",
            category=StatCategory.SPECIAL,
            description="Binary: is top-1 usage star injured? 1.0 if top1({USAGE_PLAYERS}) in {INJ_PLAYERS} else 0.0",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},
        ),

        "player_team_per": StatDefinition(
            name="player_team_per",
            category=StatCategory.SPECIAL,
            description="Team-level PER aggregation",
            valid_calc_weights={"weighted_MPG", "avg"},
        ),
        "player_starters_per": StatDefinition(
            name="player_starters_per",
            category=StatCategory.SPECIAL,
            description="Starting lineup average PER (top 5 by starts or MPG)",
            valid_calc_weights={"avg"},
        ),
        "player_per_1": StatDefinition(
            name="player_per_1",
            category=StatCategory.SPECIAL,
            description="PER of #1 player by MPG",
            valid_calc_weights={"raw", "weighted_MIN_REC"},
        ),
        "player_per_2": StatDefinition(
            name="player_per_2",
            category=StatCategory.SPECIAL,
            description="PER of #2 player by MPG",
            valid_calc_weights={"raw"},
        ),
        "player_per_3": StatDefinition(
            name="player_per_3",
            category=StatCategory.SPECIAL,
            description="PER of #3 player by MPG",
            valid_calc_weights={"raw"},
        ),
        "player_per": StatDefinition(
            name="player_per",
            category=StatCategory.SPECIAL,
            description="Team PER aggregations (topN avg/weighted/sum)",
            valid_calc_weights={"top1_avg", "top2_avg", "top3_avg",
                               "top1_weighted_MPG", "top2_weighted_MPG", "top3_weighted_MPG",
                               "top3_sum"},  # top3_sum: safe denominator for normalized injury features
        ),

        # ---- Player Talent Features (from player_feature_updates.md) ----
        # These features use player subsets: ROSTER, ROTATION, BENCH, STARTERS, USAGE_PLAYERS
        # See documentation/player_feature_updates.md for subset definitions and calc formulas

        "player_starter_per": StatDefinition(
            name="player_starter_per",
            category=StatCategory.SPECIAL,
            description="Average PER for starting players ({STARTERS} set)",
            valid_calc_weights={"avg"},
            valid_time_periods={"season"},
        ),
        "player_bench_per": StatDefinition(
            name="player_bench_per",
            category=StatCategory.SPECIAL,
            description="Minute-weighted PER for bench players ({BENCH} = {ROTATION} - {STARTERS})",
            # weighted_MPG: static minute weights, weighted_MIN_REC(k=N): recency-decayed minute weights
            # Bench uses higher k values (35-50) for slower decay
            valid_calc_weights={"weighted_MPG", "weighted_MIN_REC",
                               "weighted_MIN_REC(k=35)", "weighted_MIN_REC(k=40)",
                               "weighted_MIN_REC(k=45)", "weighted_MIN_REC(k=50)"},
            valid_time_periods={"season"},
        ),
        "player_rotation_per": StatDefinition(
            name="player_rotation_per",
            category=StatCategory.SPECIAL,
            description="Minute-weighted PER for rotation players ({ROTATION} = top active by MPG, max 10)",
            # weighted_MPG: static minute weights, weighted_MIN_REC(k=N): recency-decayed minute weights
            # Rotation uses lower k values (20-35) for faster decay
            valid_calc_weights={"weighted_MPG", "weighted_MIN_REC",
                               "weighted_MIN_REC(k=20)", "weighted_MIN_REC(k=25)",
                               "weighted_MIN_REC(k=30)", "weighted_MIN_REC(k=35)"},
            valid_time_periods={"season"},
        ),
        "player_starter_bench_per_gap": StatDefinition(
            name="player_starter_bench_per_gap",
            category=StatCategory.SPECIAL,
            description="Gap between starter and bench PER: player_starter_per|avg - player_bench_per|weighted_MIN_REC(k=N)",
            # derived(k=N) indicates which bench k value was used in the subtraction
            valid_calc_weights={"derived(k=35)", "derived(k=40)", "derived(k=45)", "derived(k=50)"},
            valid_time_periods={"season"},
        ),
        "player_star_score": StatDefinition(
            name="player_star_score",
            category=StatCategory.SPECIAL,
            description="Star score (PER × MIN) aggregations for {ROTATION}. top1/top3 = season avg, *_MIN_REC = recency-weighted",
            # top1: season PER×MPG for top player, top3_avg/sum: season averages
            # top1_MIN_REC(k=N), top3_MIN_REC(k=N): recency-weighted variants
            valid_calc_weights={"top1", "top3_avg", "top3_sum",
                               "top1_MIN_REC(k=20)", "top1_MIN_REC(k=25)",
                               "top1_MIN_REC(k=30)", "top1_MIN_REC(k=35)",
                               "top3_MIN_REC(k=20)", "top3_MIN_REC(k=25)",
                               "top3_MIN_REC(k=30)", "top3_MIN_REC(k=35)"},
            valid_time_periods={"season"},
        ),
        "player_star_share": StatDefinition(
            name="player_star_share",
            category=StatCategory.SPECIAL,
            description="Star concentration: top1 or top3 star_score as share of total rotation star_score",
            # top1_share/top3_share: season averages, *_MIN_REC(k=N): recency-weighted variants
            valid_calc_weights={"top1_share", "top3_share",
                               "top1_share_MIN_REC(k=20)", "top1_share_MIN_REC(k=25)",
                               "top1_share_MIN_REC(k=30)", "top1_share_MIN_REC(k=35)",
                               "top3_share_MIN_REC(k=20)", "top3_share_MIN_REC(k=25)",
                               "top3_share_MIN_REC(k=30)", "top3_share_MIN_REC(k=35)"},
            valid_time_periods={"season"},
        ),
        "player_star_score_all": StatDefinition(
            name="player_star_score_all",
            category=StatCategory.SPECIAL,
            description="Top star score from {USAGE_PLAYERS} (includes injured in sorting)",
            valid_calc_weights={"top1"},
            valid_time_periods={"season"},
        ),
        "player_rotation_count": StatDefinition(
            name="player_rotation_count",
            category=StatCategory.SPECIAL,
            description="Number of players in {ROTATION} (depth indicator)",
            valid_calc_weights={"raw"},
            valid_time_periods={"season"},
        ),
        "player_continuity": StatDefinition(
            name="player_continuity",
            category=StatCategory.SPECIAL,
            description="Rotation cohesion proxy: avg(min(1, GP/GP_THRESH)) for {ROTATION}",
            valid_calc_weights={"avg"},
            valid_time_periods={"season"},
        ),

        "rest": StatDefinition(
            name="rest",
            category=StatCategory.SPECIAL,
            description="Rest days before game",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},
        ),

        # ---- GB-4 Matchup & Chemistry Stats ----
        # Added for ensemble model improvement (+0.40pp with margin_avg subset)
        "margin": StatDefinition(
            name="margin",
            category=StatCategory.DERIVED,
            description="Point differential (home_score - away_score) per game",
            supports_side_split=True,
            valid_calc_weights={"raw", "avg", "std"},  # raw for single game, avg/std for rolling
            valid_time_periods={"season", "games_5", "games_10", "games_20"},  # H2H margin now uses margin_h2h stat
        ),
        "close_win_pct": StatDefinition(
            name="close_win_pct",
            category=StatCategory.DERIVED,
            description="Win percentage in close games (decided by <=5 points)",
            supports_side_split=True,
            valid_calc_weights={"avg"},  # Only avg makes sense (win rate)
            valid_time_periods={"season", "games_close5"},  # Season or last 5 close games
        ),
        "h2h_win_pct": StatDefinition(
            name="h2h_win_pct",
            category=StatCategory.DERIVED,
            description="Head-to-head win percentage (team's wins / total H2H games)",
            supports_side_split=True,  # |side filters to games where home was home, away was away
            valid_calc_weights={"raw", "beta"},  # raw=simple pct, beta=Beta-prior-smoothed
            valid_time_periods={"season", "last_3", "last_5"},  # season=this season only, last_N=cross-season
            valid_perspectives={"home", "away", "diff"},
        ),
        "margin_h2h": StatDefinition(
            name="margin_h2h",
            category=StatCategory.DERIVED,
            description="Point margin in head-to-head games",
            supports_side_split=True,  # |side filters to games where home was home, away was away
            valid_calc_weights={"avg", "eb", "logw"},  # avg=simple, eb=Empirical Bayes shrinkage, logw=log-weighted
            valid_time_periods={"season", "last_3", "last_5"},  # season=this season only, last_N=cross-season
            valid_perspectives={"home", "away", "diff"},  # diff = home - away = 2*home_margin
        ),
        "h2h_games_count": StatDefinition(
            name="h2h_games_count",
            category=StatCategory.SPECIAL,
            description="Number of head-to-head games played (sample size indicator)",
            supports_side_split=True,  # |side filters to games where home was home, away was away
            valid_calc_weights={"raw"},
            valid_time_periods={"season", "last_3", "last_5"},  # season=this season only, last_N=cross-season
            valid_perspectives={"none"},  # Matchup-level, not team-specific
        ),

        # ---- League-Specific Features ----
        "conf_wins": StatDefinition(
            name="conf_wins",
            category=StatCategory.DERIVED,
            description="Conference win percentage (wins vs same-conference opponents / total conference games)",
            supports_side_split=False,
            valid_calc_weights={"avg"},
            valid_time_periods={"season"},
            valid_perspectives={"home", "away", "diff"},
        ),
        "same_conf": StatDefinition(
            name="same_conf",
            category=StatCategory.DERIVED,
            description="Binary: 1 if both teams are in the same conference, 0 otherwise",
            supports_side_split=False,
            valid_calc_weights={"binary"},
            valid_time_periods={"none"},
            valid_perspectives={"none"},
        ),
        "conf_gp": StatDefinition(
            name="conf_gp",
            category=StatCategory.DERIVED,
            description="Number of conference games played this season",
            supports_side_split=False,
            valid_calc_weights={"raw"},
            valid_time_periods={"season"},
            valid_perspectives={"home", "away", "diff"},
        ),
        "conf_margin": StatDefinition(
            name="conf_margin",
            category=StatCategory.DERIVED,
            description="Average margin per game against same-conference opponents this season",
            supports_side_split=False,
            valid_calc_weights={"avg"},
            valid_time_periods={"season"},
            valid_perspectives={"home", "away", "diff"},
        ),

        # ---- Prediction Features (from Points Regression Models) ----
        # Format: pred_margin|none|ridge|none (calc_weight = model type)
        "pred_home": StatDefinition(
            name="pred_home",
            category=StatCategory.SPECIAL,
            description="Predicted home team points from selected points model",
            valid_calc_weights={"ridge", "elasticnet", "randomforest", "xgboost"},
            valid_time_periods={"none"},  # Predictions are point-in-time
            valid_perspectives={"none"},  # Matchup-level, not team-specific
        ),
        "pred_away": StatDefinition(
            name="pred_away",
            category=StatCategory.SPECIAL,
            description="Predicted away team points from selected points model",
            valid_calc_weights={"ridge", "elasticnet", "randomforest", "xgboost"},
            valid_time_periods={"none"},
            valid_perspectives={"none"},
        ),
        "pred_margin": StatDefinition(
            name="pred_margin",
            category=StatCategory.SPECIAL,
            description="Predicted point differential (home - away) from selected points model",
            valid_calc_weights={"ridge", "elasticnet", "randomforest", "xgboost"},
            valid_time_periods={"none"},
            valid_perspectives={"none"},
        ),
        "pred_total": StatDefinition(
            name="pred_total",
            category=StatCategory.SPECIAL,
            description="Predicted total points (home + away) from selected points model",
            valid_calc_weights={"ridge", "elasticnet", "randomforest", "xgboost"},
            valid_time_periods={"none"},
            valid_perspectives={"none"},
        ),

        # ---- Vegas Betting Lines ----
        # Pulled from 'vegas' field in game documents (populated from nba_2008-2025.csv)
        "vegas_ML": StatDefinition(
            name="vegas_ML",
            category=StatCategory.SPECIAL,
            db_field="vegas",  # Source: vegas.home_ML / vegas.away_ML
            description="Vegas moneyline odds (American format, e.g., -150 for favorite, +130 for underdog)",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},  # Point-in-time pregame line
            valid_perspectives={"home", "away", "diff"},
        ),
        "vegas_spread": StatDefinition(
            name="vegas_spread",
            category=StatCategory.SPECIAL,
            db_field="vegas",  # Source: vegas.home_spread / vegas.away_spread
            description="Vegas point spread (home: raw value where negative = home favored; away: negated)",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},  # Point-in-time pregame line
            valid_perspectives={"home", "away"},  # home = raw spread, away = negated spread
        ),
        "vegas_ou": StatDefinition(
            name="vegas_ou",
            category=StatCategory.SPECIAL,
            db_field="vegas",  # Source: vegas.OU
            description="Vegas over/under total points line",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},  # Point-in-time pregame line
            valid_perspectives={"none"},  # Matchup-level, not team-specific
        ),
        "vegas_implied_prob": StatDefinition(
            name="vegas_implied_prob",
            category=StatCategory.DERIVED,
            description="Implied win probability from moneyline: if ML<0: (-ML)/(-ML+100), else: 100/(ML+100)",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},  # Point-in-time derived from pregame ML
            valid_perspectives={"home", "away", "diff"},
        ),
        "vegas_edge": StatDefinition(
            name="vegas_edge",
            category=StatCategory.DERIVED,
            description="Edge between implied prob and spread-derived prob: implied_prob - spread_prob, where spread_prob = 1/(1+exp(-spread/6.5))",
            valid_calc_weights={"raw"},
            valid_time_periods={"none"},  # Point-in-time derived from pregame lines
            valid_perspectives={"home", "away", "diff"},
        ),
    }

    # ==========================================================================
    # VALID TIME PERIODS
    # ==========================================================================

    # Base time periods (non-parameterized)
    BASE_TIME_PERIODS: Set[str] = {"season", "none"}

    # Parameterized time period prefixes (accept any integer N)
    # e.g., games_10, games_20, days_5, months_1, games_close5, last_3 (for H2H)
    TIME_PERIOD_PREFIXES: Set[str] = {"games_", "days_", "months_", "games_close", "last_"}

    @classmethod
    def is_valid_time_period(cls, tp: str) -> bool:
        """
        Check if a time period is valid.

        Valid formats:
        - "season", "none" (base periods)
        - "games_N", "days_N", "months_N" where N is any positive integer
        """
        if tp in cls.BASE_TIME_PERIODS:
            return True
        for prefix in cls.TIME_PERIOD_PREFIXES:
            if tp.startswith(prefix):
                suffix = tp[len(prefix):]
                if suffix.isdigit() and int(suffix) > 0:
                    return True
        # Composite time periods: blend and delta
        if tp.startswith('blend:'):
            parsed = cls.parse_blend_time_period(tp)
            if parsed is None:
                return False
            # Validate each component is a valid base time period and weights sum to ~1.0
            total_weight = 0.0
            for comp_tp, weight in parsed['components']:
                if not cls.is_valid_time_period(comp_tp):
                    return False
                total_weight += weight
            if abs(total_weight - 1.0) > 0.01:
                return False
            return True

        if tp.startswith('delta:'):
            parsed = cls.parse_delta_time_period(tp)
            if parsed is None:
                return False
            # Validate all leaf time periods
            for leaf_tp in cls._extract_leaf_time_periods(tp):
                if not cls.is_valid_time_period(leaf_tp):
                    return False
            return True

        return False

    @classmethod
    def parse_blend_time_period(cls, time_period: str) -> dict:
        """
        Parse a blend time_period string.

        Input:  "blend:games_5:0.70/games_10:0.30"
        Output: {"type": "blend", "components": [("games_5", 0.70), ("games_10", 0.30)]}

        Returns None if malformed.
        """
        if not time_period.startswith('blend:'):
            return None

        blend_str = time_period[6:]  # Remove 'blend:' prefix
        parts = blend_str.split('/')

        components = []
        for part in parts:
            if ':' not in part:
                return None
            # Split on last ':' to get (tp, weight) — handles tp names with colons
            idx = part.rfind(':')
            tp = part[:idx]
            weight_str = part[idx + 1:]
            try:
                weight = float(weight_str)
            except ValueError:
                return None
            components.append((tp, weight))

        if not components:
            return None

        return {"type": "blend", "components": components}

    @classmethod
    def parse_delta_time_period(cls, time_period: str) -> dict:
        """
        Parse a delta time_period string.

        Input:  "delta:games_5-season"
        Output: {"type": "delta", "recent": "games_5", "baseline": "season"}

        Input:  "delta:blend:games_5:0.70/games_10:0.30-season"
        Output: {"type": "blend_delta",
                 "recent": {"type": "blend", "components": [("games_5", 0.70), ("games_10", 0.30)]},
                 "baseline": "season"}

        Returns None if malformed.
        """
        if not time_period.startswith('delta:'):
            return None

        delta_str = time_period[6:]  # Remove 'delta:' prefix

        # Use rfind('-') to split recent from baseline (baseline is always a simple tp)
        sep_idx = delta_str.rfind('-')
        if sep_idx <= 0 or sep_idx == len(delta_str) - 1:
            return None

        recent_str = delta_str[:sep_idx]
        baseline = delta_str[sep_idx + 1:]

        # Check if recent is itself a blend
        if recent_str.startswith('blend:'):
            blend_parsed = cls.parse_blend_time_period(recent_str)
            if blend_parsed is None:
                return None
            return {
                "type": "blend_delta",
                "recent": blend_parsed,
                "baseline": baseline,
            }

        return {
            "type": "delta",
            "recent": recent_str,
            "baseline": baseline,
        }

    @classmethod
    def _extract_leaf_time_periods(cls, time_period: str) -> list:
        """
        Extract all base (leaf) time periods from any composite time period.

        "season"                                        -> ["season"]
        "blend:games_5:0.70/games_10:0.30"              -> ["games_5", "games_10"]
        "delta:games_5-season"                           -> ["games_5", "season"]
        "delta:blend:games_5:0.70/games_10:0.30-season"  -> ["games_5", "games_10", "season"]
        """
        if time_period.startswith('blend:'):
            parsed = cls.parse_blend_time_period(time_period)
            if parsed is None:
                return [time_period]
            return [tp for tp, _ in parsed['components']]

        if time_period.startswith('delta:'):
            parsed = cls.parse_delta_time_period(time_period)
            if parsed is None:
                return [time_period]

            leaves = []
            if parsed['type'] == 'blend_delta':
                # Recent is a blend
                leaves.extend(tp for tp, _ in parsed['recent']['components'])
            else:
                leaves.append(parsed['recent'])
            leaves.append(parsed['baseline'])
            return leaves

        return [time_period]

    # Legacy set for backwards compatibility (common time periods)
    VALID_TIME_PERIODS: Set[str] = {
        "season", "games_2", "games_3", "games_5", "games_10", "games_12", "games_20",
        "months_1", "days_2", "days_3", "days_5", "days_12", "none",
        "games_close5",
        "last_3", "last_5",  # H2H cross-season lookups
    }

    # ==========================================================================
    # VALID CALC WEIGHTS
    # ==========================================================================

    VALID_CALC_WEIGHTS: Set[str] = {
        "raw",
        "avg",
        "std",
        "weighted_MPG",
        "weighted_MIN",
        "weighted_MIN_REC",
        "harmonic_mean",
        "derived",
        "top1", "top1_avg", "top2_avg", "top3_avg",
        "top1_weighted_MPG", "top2_weighted_MPG", "top3_weighted_MPG",
        "top3_sum",
        "top1_share", "top3_share",  # Star share (non-recency)
        # Model types for prediction features (pred_margin|none|ridge|none)
        "ridge", "elasticnet", "randomforest", "xgboost",
        # H2H shrinkage/reliability calc weights
        "beta",  # Beta-prior-smoothed win probability
        "eb",    # Empirical Bayes shrinkage for margin
        "logw",  # Log-weighted margin (scales by sample size)
        # Binary features
        "binary",  # Binary indicator (0 or 1)
        # Aggregation types
        "sum",  # Sum/total (e.g., total travel distance)
    }

    # Parameterized calc weight patterns (accept integer k values)
    # e.g., weighted_MIN_REC(k=20), derived(k=50), top1_MIN_REC(k=25)
    CALC_WEIGHT_PATTERNS: Set[str] = {
        "weighted_MIN_REC",      # weighted_MIN_REC(k=N)
        "derived",               # derived(k=N)
        "top1_MIN_REC",          # top1_MIN_REC(k=N)
        "top3_MIN_REC",          # top3_MIN_REC(k=N)
        "top1_share_MIN_REC",    # top1_share_MIN_REC(k=N)
        "top3_share_MIN_REC",    # top3_share_MIN_REC(k=N)
    }

    @classmethod
    def is_parameterized_calc_weight(cls, calc_weight: str) -> bool:
        """
        Check if calc_weight is a valid parameterized format.

        Valid formats: pattern(k=N) where pattern is in CALC_WEIGHT_PATTERNS
        and N is a positive integer.
        """
        import re
        match = re.match(r'^(.+)\(k=(\d+)\)$', calc_weight)
        if not match:
            return False
        pattern, k_val = match.groups()
        return pattern in cls.CALC_WEIGHT_PATTERNS and int(k_val) > 0

    # Note: blend:* is validated separately via is_blend_format()

    # ==========================================================================
    # VALID PERSPECTIVES
    # ==========================================================================

    VALID_PERSPECTIVES: Set[str] = {"diff", "home", "away", "none"}

    # ==========================================================================
    # STAT CATEGORIES FOR COMPUTATION
    # ==========================================================================

    @classmethod
    def get_rate_stats(cls) -> Set[str]:
        """Get stats that require aggregation before computing rate."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.requires_aggregation or defn.category == StatCategory.RATE
        }

    @classmethod
    def get_side_splittable_stats(cls) -> Set[str]:
        """Get stats that can be split by home/away side."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.supports_side_split
        }

    @classmethod
    def get_net_stats(cls) -> Set[str]:
        """Get stats that have opponent-adjusted (*_net) versions."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.supports_net
        }

    @classmethod
    def get_basic_stats(cls) -> Set[str]:
        """Get basic counting stats."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.BASIC
        }

    @classmethod
    def get_derived_stats(cls) -> Set[str]:
        """Get derived/computed stats."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.DERIVED
        }

    @classmethod
    def get_special_stats(cls) -> Set[str]:
        """Get stats requiring special handling."""
        return {
            name for name, defn in cls.STAT_DEFINITIONS.items()
            if defn.category == StatCategory.SPECIAL
        }

    # ==========================================================================
    # DB FIELD MAPPING
    # ==========================================================================

    @classmethod
    def get_db_field(cls, stat_name: str) -> str:
        """
        Get the MongoDB field name for a stat.

        Args:
            stat_name: Canonical stat name (e.g., "efg")

        Returns:
            DB field name (e.g., "effective_fg_perc")
        """
        defn = cls.STAT_DEFINITIONS.get(stat_name)
        if defn and defn.db_field:
            return defn.db_field
        return stat_name  # Default: use stat name as-is

    @classmethod
    def get_stat_name_map(cls) -> Dict[str, str]:
        """
        Get the complete stat name to DB field mapping.

        Used by StatHandlerV2 for computation.

        Returns:
            Dict mapping stat names to DB field names
        """
        return {
            name: defn.db_field if defn.db_field else name
            for name, defn in cls.STAT_DEFINITIONS.items()
        }

    # ==========================================================================
    # FEATURE NAME GENERATION
    # ==========================================================================

    @classmethod
    def build_feature_name(
        cls,
        stat_name: str,
        time_period: str,
        calc_weight: str,
        perspective: str,
        side_split: bool = False
    ) -> str:
        """
        Build a valid feature name from components.

        Args:
            stat_name: Base stat name (e.g., "points")
            time_period: Time period (e.g., "season", "games_10")
            calc_weight: Calculation weight (e.g., "raw", "avg")
            perspective: Perspective (e.g., "diff", "home")
            side_split: Whether to add |side suffix

        Returns:
            Formatted feature name (e.g., "points|season|avg|diff|side")
        """
        parts = [stat_name, time_period, calc_weight, perspective]
        if side_split:
            parts.append("side")
        return "|".join(parts)

    @classmethod
    def parse_feature_name(cls, feature_name: str) -> Optional[Dict[str, str]]:
        """
        Parse a feature name into its components.

        Args:
            feature_name: Full feature name (e.g., "points|season|avg|diff|side")

        Returns:
            Dict with keys: stat_name, time_period, calc_weight, perspective, has_side
            Or None if parsing fails
        """
        parts = feature_name.split("|")

        if len(parts) < 4:
            return None

        result = {
            "stat_name": parts[0],
            "time_period": parts[1],
            "calc_weight": parts[2],
            "perspective": parts[3],
            "has_side": len(parts) >= 5 and parts[4] == "side"
        }

        return result

    # ==========================================================================
    # VALIDATION
    # ==========================================================================

    @classmethod
    def is_blend_format(cls, calc_weight: str) -> bool:
        """Check if calc_weight is a blend format (e.g., 'blend:season:0.8/games_10:0.2')."""
        return calc_weight.startswith("blend:")

    @classmethod
    def validate_blend_format(cls, calc_weight: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a blend format calc_weight.

        Args:
            calc_weight: e.g., "blend:season:0.80/games_20:0.10/games_12:0.10"

        Returns:
            (is_valid, error_message)
        """
        if not calc_weight.startswith("blend:"):
            return False, "Blend format must start with 'blend:'"

        blend_spec = calc_weight[6:]  # Remove "blend:" prefix

        # Handle special injury blend format
        if "/" in blend_spec and ":" in blend_spec:
            components = blend_spec.split("/")
            total_weight = 0.0

            for component in components:
                if ":" not in component:
                    return False, f"Invalid blend component: {component}"

                parts = component.split(":")
                if len(parts) != 2:
                    return False, f"Invalid blend component format: {component}"

                period_or_key, weight_str = parts
                try:
                    weight = float(weight_str)
                    total_weight += weight
                except ValueError:
                    return False, f"Invalid weight value: {weight_str}"

            # Weights should sum to approximately 1.0
            if abs(total_weight - 1.0) > 0.01:
                return False, f"Blend weights sum to {total_weight}, expected 1.0"

        return True, None

    @classmethod
    def validate_stat_name(cls, stat_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate a stat name.

        Args:
            stat_name: The stat name to validate

        Returns:
            (is_valid, error_message)
        """
        if stat_name in cls.STAT_DEFINITIONS:
            return True, None

        # Check if it's a valid _net variant
        if stat_name.endswith("_net"):
            base_stat = stat_name[:-4]
            if base_stat in cls.STAT_DEFINITIONS:
                defn = cls.STAT_DEFINITIONS[base_stat]
                if defn.supports_net:
                    return True, None
                return False, f"Stat '{base_stat}' does not support _net variant"

        return False, f"Unknown stat name: '{stat_name}'"

    @classmethod
    def validate_time_period(cls, time_period: str) -> Tuple[bool, Optional[str]]:
        """Validate a time period (including composite blend/delta formats)."""
        if cls.is_valid_time_period(time_period):
            return True, None

        # Provide specific error messages for malformed composite formats
        if time_period.startswith('blend:'):
            parsed = cls.parse_blend_time_period(time_period)
            if parsed is None:
                return False, f"Malformed blend time period: '{time_period}'. Expected format: 'blend:tp1:weight1/tp2:weight2'"
            total_weight = 0.0
            for comp_tp, weight in parsed['components']:
                if not cls.is_valid_time_period(comp_tp):
                    return False, f"Invalid component time period '{comp_tp}' in blend: '{time_period}'"
                total_weight += weight
            if abs(total_weight - 1.0) > 0.01:
                return False, f"Blend weights sum to {total_weight:.2f}, expected 1.0 in: '{time_period}'"
            return True, None

        if time_period.startswith('delta:'):
            parsed = cls.parse_delta_time_period(time_period)
            if parsed is None:
                return False, f"Malformed delta time period: '{time_period}'. Expected format: 'delta:recent_tp-baseline_tp'"
            for leaf_tp in cls._extract_leaf_time_periods(time_period):
                if not cls.is_valid_time_period(leaf_tp):
                    return False, f"Invalid leaf time period '{leaf_tp}' in delta: '{time_period}'"
            return True, None

        return False, f"Invalid time period: '{time_period}'. Valid formats: 'season', 'none', 'games_N', 'days_N', 'months_N', 'blend:...', 'delta:...' (N = positive integer)"

    @classmethod
    def validate_calc_weight(cls, calc_weight: str) -> Tuple[bool, Optional[str]]:
        """Validate a calc weight."""
        if calc_weight in cls.VALID_CALC_WEIGHTS:
            return True, None

        if cls.is_blend_format(calc_weight):
            return cls.validate_blend_format(calc_weight)

        if cls.is_parameterized_calc_weight(calc_weight):
            return True, None

        return False, f"Invalid calc weight: '{calc_weight}'. Valid: {sorted(cls.VALID_CALC_WEIGHTS)}, pattern(k=N), or blend:*"

    @classmethod
    def validate_perspective(cls, perspective: str) -> Tuple[bool, Optional[str]]:
        """Validate a perspective."""
        if perspective in cls.VALID_PERSPECTIVES:
            return True, None
        return False, f"Invalid perspective: '{perspective}'. Valid: {sorted(cls.VALID_PERSPECTIVES)}"

    @classmethod
    def validate_feature(cls, feature_name: str) -> Tuple[bool, Optional[str]]:
        """
        Fully validate a feature name.

        Args:
            feature_name: Full feature name (e.g., "points|season|avg|diff")

        Returns:
            (is_valid, error_message)
        """
        parsed = cls.parse_feature_name(feature_name)

        if not parsed:
            return False, f"Invalid feature format: '{feature_name}'. Expected: stat|period|weight|perspective[|side]"

        # Validate stat name
        valid, error = cls.validate_stat_name(parsed["stat_name"])
        if not valid:
            return False, error

        # Validate time period format
        valid, error = cls.validate_time_period(parsed["time_period"])
        if not valid:
            return False, error

        # Validate time period is allowed for this specific stat
        stat_name = parsed["stat_name"]
        base_stat = stat_name[:-4] if stat_name.endswith("_net") else stat_name
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_time_periods:  # If restrictions exist, enforce them
                # For composite time periods, validate each leaf tp
                leaf_tps = cls._extract_leaf_time_periods(parsed["time_period"])
                for tp in leaf_tps:
                    is_allowed = False
                    for allowed_tp in stat_def.valid_time_periods:
                        if tp == allowed_tp:
                            is_allowed = True
                            break
                        # Check if it matches a pattern like "games_10" matching "games_N"
                        if '_' in allowed_tp and '_' in tp:
                            prefix = allowed_tp.rsplit('_', 1)[0]
                            if tp.startswith(prefix + '_'):
                                is_allowed = True
                                break
                    if not is_allowed:
                        return False, f"Time period '{tp}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_time_periods)}"

        # Validate calc weight format
        valid, error = cls.validate_calc_weight(parsed["calc_weight"])
        if not valid:
            return False, error

        # Validate calc weight is allowed for this specific stat
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_calc_weights:  # If restrictions exist, enforce them
                calc_weight = parsed["calc_weight"]
                # Check for exact match first
                if calc_weight not in stat_def.valid_calc_weights:
                    # For blend stats, also check if blend format is allowed
                    if "blend" in stat_def.valid_calc_weights and cls.is_blend_format(calc_weight):
                        pass  # Blend format is allowed
                    else:
                        return False, f"Calc weight '{calc_weight}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_calc_weights)}"

        # Validate perspective format
        valid, error = cls.validate_perspective(parsed["perspective"])
        if not valid:
            return False, error

        # Validate perspective is allowed for this specific stat
        if base_stat in cls.STAT_DEFINITIONS:
            stat_def = cls.STAT_DEFINITIONS[base_stat]
            if stat_def.valid_perspectives:  # If restrictions exist, enforce them
                if parsed["perspective"] not in stat_def.valid_perspectives:
                    return False, f"Perspective '{parsed['perspective']}' not valid for stat '{stat_name}'. Valid: {sorted(stat_def.valid_perspectives)}"

        # Validate side split is allowed for this stat
        if parsed["has_side"]:
            stat_name = parsed["stat_name"]
            base_stat = stat_name[:-4] if stat_name.endswith("_net") else stat_name
            if base_stat in cls.STAT_DEFINITIONS:
                if not cls.STAT_DEFINITIONS[base_stat].supports_side_split:
                    return False, f"Stat '{stat_name}' does not support side split"

        return True, None

    @classmethod
    def validate_feature_list(cls, features: List[str]) -> Dict[str, any]:
        """
        Validate a list of features.

        Args:
            features: List of feature names

        Returns:
            Dict with 'valid', 'errors', 'warnings', 'invalid_features'
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "invalid_features": [],
            "valid_count": 0
        }

        for feature in features:
            valid, error = cls.validate_feature(feature)
            if valid:
                result["valid_count"] += 1
            else:
                result["valid"] = False
                result["invalid_features"].append(feature)
                result["errors"].append(f"{feature}: {error}")

        return result

    # ==========================================================================
    # CONVENIENCE METHODS
    # ==========================================================================

    @classmethod
    def get_all_stat_names(cls) -> List[str]:
        """Get all valid stat names."""
        return list(cls.STAT_DEFINITIONS.keys())

    @classmethod
    def get_stat_definition(cls, stat_name: str) -> Optional[StatDefinition]:
        """Get the definition for a stat."""
        return cls.STAT_DEFINITIONS.get(stat_name)

    @classmethod
    def get_all_time_periods(cls) -> List[str]:
        """Get all valid time periods."""
        return sorted(cls.VALID_TIME_PERIODS)

    @classmethod
    def get_all_calc_weights(cls) -> List[str]:
        """Get all valid calc weights (excluding blend)."""
        return sorted(cls.VALID_CALC_WEIGHTS)

    @classmethod
    def get_all_perspectives(cls) -> List[str]:
        """Get all valid perspectives."""
        return sorted(cls.VALID_PERSPECTIVES)

    @classmethod
    def generate_feature_variants(
        cls,
        stat_name: str,
        time_periods: Optional[List[str]] = None,
        calc_weights: Optional[List[str]] = None,
        perspectives: Optional[List[str]] = None,
        include_side: bool = False
    ) -> List[str]:
        """
        Generate all feature variants for a stat.

        Args:
            stat_name: Base stat name
            time_periods: List of time periods (default: all)
            calc_weights: List of calc weights (default: raw, avg)
            perspectives: List of perspectives (default: all)
            include_side: Whether to include side-split variants

        Returns:
            List of feature names
        """
        if time_periods is None:
            time_periods = list(cls.VALID_TIME_PERIODS - {"none"})
        if calc_weights is None:
            calc_weights = ["raw", "avg"]
        if perspectives is None:
            perspectives = ["diff", "home", "away"]

        features = []
        defn = cls.STAT_DEFINITIONS.get(stat_name)

        for period in time_periods:
            for weight in calc_weights:
                for perspective in perspectives:
                    feature = cls.build_feature_name(stat_name, period, weight, perspective)
                    features.append(feature)

                    # Add side split variant if applicable
                    if include_side and defn and defn.supports_side_split:
                        features.append(cls.build_feature_name(
                            stat_name, period, weight, perspective, side_split=True
                        ))

        return features


# =============================================================================
# FEATURE GROUPS (Semantic Organization)
# =============================================================================

class FeatureGroups:
    """
    Semantic groupings of features.

    These groups organize features by their predictive purpose,
    following a layered hierarchy from outcome signals to context.
    """

    OUTCOME_STRENGTH = "outcome_strength"
    SHOOTING_EFFICIENCY = "shooting_efficiency"
    OFFENSIVE_ENGINE = "offensive_engine"
    DEFENSIVE_ENGINE = "defensive_engine"
    PACE_VOLATILITY = "pace_volatility"
    SCHEDULE_FATIGUE = "schedule_fatigue"
    SAMPLE_SIZE = "sample_size"
    ELO_STRENGTH = "elo_strength"
    PLAYER_TALENT = "player_talent"
    INJURIES = "injuries"
    POINTS_DECOMPOSITION = "points_decomposition"

    # Group definitions with the stats they contain
    GROUP_DEFINITIONS: Dict[str, Dict] = {
        OUTCOME_STRENGTH: {
            "description": "Points, wins, and margins across multiple windows (raw outcome signals)",
            "stats": ["points", "wins", "points_net", "margin"],
            "layer": 1,
        },
        SHOOTING_EFFICIENCY: {
            "description": "eFG%, TS%, 3P%, shot-mix rates, matchup metrics",
            "stats": ["efg", "ts", "three_pct", "three_rate", "ft_rate", "three_pct_matchup", "efg_net", "ts_net", "three_pct_net"],
            "layer": 1,
        },
        OFFENSIVE_ENGINE: {
            "description": "Offensive rating, assists, playmaking",
            "stats": ["off_rtg", "assists_ratio", "three_made", "assists", "ast_to_ratio", "off_rtg_net", "assists_ratio_net", "three_made_net"],
            "layer": 1,
        },
        DEFENSIVE_ENGINE: {
            "description": "Defensive rating, rebounds, blocks, steals, turnovers",
            "stats": ["def_rtg", "reb_total", "blocks", "steals", "turnovers", "to_metric"],
            "layer": 1,
        },
        PACE_VOLATILITY: {
            "description": "Possession count, pace, interaction features",
            "stats": ["pace", "pace_interaction", "est_possessions"],
            "layer": 2,
        },
        SCHEDULE_FATIGUE: {
            "description": "Rest, back-to-backs, travel distance, road games, postseason indicator",
            "stats": ["days_rest", "b2b", "first_of_b2b", "travel", "road_games", "rest", "postseason"],
            "layer": 2,
        },
        SAMPLE_SIZE: {
            "description": "Games played reliability signals",
            "stats": ["games_played"],
            "layer": 2,
        },
        ELO_STRENGTH: {
            "description": "Elo rating (meta-strength summary)",
            "stats": ["elo"],
            "layer": 3,
        },
        PLAYER_TALENT: {
            "description": "PER-based team-level aggregations and player subset talent metrics",
            "stats": [
                # Legacy player PER stats
                "player_team_per", "player_per", "player_per_1", "player_per_2", "player_per_3", "player_starters_per",
                # New player talent features (see documentation/player_feature_updates.md)
                "player_starter_per",           # Starters avg PER
                "player_bench_per",             # Bench weighted PER
                "player_rotation_per",          # Rotation weighted PER
                "player_starter_bench_per_gap", # Starter-bench PER gap
                "player_star_score",            # Star score (PER×MIN) aggregations
                "player_star_share",            # Star concentration metrics
                "player_star_score_all",        # Top star from usage players
                "player_rotation_count",        # Rotation depth
                "player_continuity",            # Rotation cohesion proxy
            ],
            "layer": 4,
        },
        INJURIES: {
            "description": "Injury impact features (severity, minutes lost, star impact, normalized shares)",
            "stats": [
                # Legacy injury features
                "inj_impact", "inj_severity", "inj_min_lost", "inj_per", "inj_rotation_per",
                "inj_per_share", "inj_per_weighted_share",
                # New star-based injury features (see documentation/player_feature_updates.md)
                "inj_star_share",        # Injured star's share of top-3 star mass
                "inj_star_score_share",  # Top-3 injured star mass as share of team
                "inj_top1_star_out",     # Binary: is top-1 star injured?
            ],
            "layer": 4,
        },
        POINTS_DECOMPOSITION: {
            "description": "Expected points from possessions × efficiency",
            "stats": ["exp_points_off", "exp_points_def", "exp_points_matchup"],
            "layer": 1,
        },
    }

    # Additional group names for extended feature groups
    H2H = "h2h"  # Head-to-head matchup features
    CLOSE_GAMES = "close_games"  # Close game performance features
    PREDICTION_FEATURES = "prediction_features"  # Points model predictions
    VEGAS_LINES = "vegas_lines"  # Vegas betting lines (ML, spread, O/U)

    # Extended group definitions (added separately to avoid breaking existing layer-based queries)
    EXTENDED_GROUP_DEFINITIONS: Dict[str, Dict] = {
        H2H: {
            "description": "Head-to-head matchup history between teams (season-only)",
            "stats": ["h2h_win_pct", "margin_h2h", "h2h_games_count"],
            "filter_substring": "h2h",  # Only include features containing "h2h" in name
            "layer": 4,
        },
        CLOSE_GAMES: {
            "description": "Performance in close games (decided by <=5 points)",
            "stats": ["close_win_pct"],
            "filter_substring": "close",  # Only include features containing this substring
            "layer": 4,
        },
        PREDICTION_FEATURES: {
            "description": "Point predictions from trained regression models (format: pred_margin|none|ridge|none)",
            "stats": ["pred_home", "pred_away", "pred_margin", "pred_total"],
            "layer": 5,  # Highest layer - derived from other model
        },
        VEGAS_LINES: {
            "description": "Vegas pregame betting lines and derived metrics (moneylines, spreads, over/under, implied probability, edge)",
            "stats": ["vegas_ML", "vegas_spread", "vegas_ou", "vegas_implied_prob", "vegas_edge"],
            "layer": 0,  # Layer 0 - external market data, not derived from team stats
        },
    }

    @classmethod
    def get_league_extra_group(cls, league) -> Optional[Dict]:
        """Build a synthetic group from league's extra_features config."""
        if not league or not hasattr(league, 'extra_feature_stats'):
            return None
        stats = league.extra_feature_stats
        if not stats:
            return None
        return {
            "description": f"League-specific features for {league.league_id}",
            "stats": stats,
            "layer": 4,
        }

    @classmethod
    def get_all_group_definitions(cls, league=None) -> Dict[str, Dict]:
        """Get all group definitions (base + extended + league-specific)."""
        groups = {**cls.GROUP_DEFINITIONS, **cls.EXTENDED_GROUP_DEFINITIONS}
        extra = cls.get_league_extra_group(league)
        if extra:
            groups["league_extra"] = extra
        return groups

    @classmethod
    def get_group_stats(cls, group_name: str, league=None) -> List[str]:
        """Get stats in a feature group."""
        all_groups = cls.get_all_group_definitions(league=league)
        group = all_groups.get(group_name)
        return group["stats"] if group else []

    @classmethod
    def get_groups_by_layer(cls, layer: int, league=None) -> List[str]:
        """Get all groups in a specific layer."""
        all_groups = cls.get_all_group_definitions(league=league)
        return [
            name for name, defn in all_groups.items()
            if defn.get("layer") == layer
        ]

    @classmethod
    def get_all_groups(cls, league=None) -> List[str]:
        """Get all group names."""
        return list(cls.get_all_group_definitions(league=league).keys())

    # =========================================================================
    # ACTUAL FEATURES GENERATED BY CALCULATORS
    # =========================================================================
    # These are the exact features that PERCalculator and StatHandlerV2 return.
    # The registry was previously generating combinatoric expansions that didn't
    # match what calculators actually produce, causing all values to be 0.

    # Features returned by PERCalculator.get_game_per_features()
    # Updated to include all subset features from documentation/player_feature_updates.md
    ACTUAL_PLAYER_FEATURES = [
        # Legacy player features
        'player_per_1|none|weighted_MIN_REC|away',
        'player_per_1|none|weighted_MIN_REC|diff',
        'player_per_1|none|weighted_MIN_REC|home',
        'player_per_1|season|raw|away',
        'player_per_1|season|raw|diff',
        'player_per_1|season|raw|home',
        'player_per_2|season|raw|away',
        'player_per_2|season|raw|diff',
        'player_per_2|season|raw|home',
        'player_per_3|season|raw|away',
        'player_per_3|season|raw|diff',
        'player_per_3|season|raw|home',
        'player_per|season|top1_avg|away',
        'player_per|season|top1_avg|diff',
        'player_per|season|top1_avg|home',
        'player_per|season|top1_weighted_MPG|away',
        'player_per|season|top1_weighted_MPG|diff',
        'player_per|season|top1_weighted_MPG|home',
        'player_per|season|top2_avg|away',
        'player_per|season|top2_avg|diff',
        'player_per|season|top2_avg|home',
        'player_per|season|top2_weighted_MPG|away',
        'player_per|season|top2_weighted_MPG|diff',
        'player_per|season|top2_weighted_MPG|home',
        'player_per|season|top3_avg|away',
        'player_per|season|top3_avg|diff',
        'player_per|season|top3_avg|home',
        'player_per|season|top3_sum|away',
        'player_per|season|top3_sum|diff',
        'player_per|season|top3_sum|home',
        'player_per|season|top3_weighted_MPG|away',
        'player_per|season|top3_weighted_MPG|diff',
        'player_per|season|top3_weighted_MPG|home',
        'player_starters_per|season|avg|away',
        'player_starters_per|season|avg|diff',
        'player_starters_per|season|avg|home',
        'player_team_per|season|avg|away',
        'player_team_per|season|avg|diff',
        'player_team_per|season|avg|home',
        'player_team_per|season|weighted_MPG|away',
        'player_team_per|season|weighted_MPG|diff',
        'player_team_per|season|weighted_MPG|home',

        # NEW: player_starter_per (from player_feature_updates.md)
        'player_starter_per|season|avg|away',
        'player_starter_per|season|avg|diff',
        'player_starter_per|season|avg|home',

        # NEW: player_bench_per (from player_feature_updates.md)
        'player_bench_per|season|weighted_MPG|away',
        'player_bench_per|season|weighted_MPG|diff',
        'player_bench_per|season|weighted_MPG|home',
        'player_bench_per|season|weighted_MIN_REC(k=35)|away',
        'player_bench_per|season|weighted_MIN_REC(k=35)|diff',
        'player_bench_per|season|weighted_MIN_REC(k=35)|home',
        'player_bench_per|season|weighted_MIN_REC(k=40)|away',
        'player_bench_per|season|weighted_MIN_REC(k=40)|diff',
        'player_bench_per|season|weighted_MIN_REC(k=40)|home',
        'player_bench_per|season|weighted_MIN_REC(k=45)|away',
        'player_bench_per|season|weighted_MIN_REC(k=45)|diff',
        'player_bench_per|season|weighted_MIN_REC(k=45)|home',
        'player_bench_per|season|weighted_MIN_REC(k=50)|away',
        'player_bench_per|season|weighted_MIN_REC(k=50)|diff',
        'player_bench_per|season|weighted_MIN_REC(k=50)|home',

        # NEW: player_rotation_per (from player_feature_updates.md)
        'player_rotation_per|season|weighted_MPG|away',
        'player_rotation_per|season|weighted_MPG|diff',
        'player_rotation_per|season|weighted_MPG|home',
        'player_rotation_per|season|weighted_MIN_REC(k=20)|away',
        'player_rotation_per|season|weighted_MIN_REC(k=20)|diff',
        'player_rotation_per|season|weighted_MIN_REC(k=20)|home',
        'player_rotation_per|season|weighted_MIN_REC(k=25)|away',
        'player_rotation_per|season|weighted_MIN_REC(k=25)|diff',
        'player_rotation_per|season|weighted_MIN_REC(k=25)|home',
        'player_rotation_per|season|weighted_MIN_REC(k=30)|away',
        'player_rotation_per|season|weighted_MIN_REC(k=30)|diff',
        'player_rotation_per|season|weighted_MIN_REC(k=30)|home',
        'player_rotation_per|season|weighted_MIN_REC(k=35)|away',
        'player_rotation_per|season|weighted_MIN_REC(k=35)|diff',
        'player_rotation_per|season|weighted_MIN_REC(k=35)|home',

        # NEW: player_starter_bench_per_gap (from player_feature_updates.md)
        'player_starter_bench_per_gap|season|derived(k=35)|away',
        'player_starter_bench_per_gap|season|derived(k=35)|diff',
        'player_starter_bench_per_gap|season|derived(k=35)|home',
        'player_starter_bench_per_gap|season|derived(k=40)|away',
        'player_starter_bench_per_gap|season|derived(k=40)|diff',
        'player_starter_bench_per_gap|season|derived(k=40)|home',
        'player_starter_bench_per_gap|season|derived(k=45)|away',
        'player_starter_bench_per_gap|season|derived(k=45)|diff',
        'player_starter_bench_per_gap|season|derived(k=45)|home',
        'player_starter_bench_per_gap|season|derived(k=50)|away',
        'player_starter_bench_per_gap|season|derived(k=50)|diff',
        'player_starter_bench_per_gap|season|derived(k=50)|home',

        # NEW: player_star_score (from player_feature_updates.md)
        'player_star_score|season|top1|away',
        'player_star_score|season|top1|diff',
        'player_star_score|season|top1|home',
        'player_star_score|season|top3_avg|away',
        'player_star_score|season|top3_avg|diff',
        'player_star_score|season|top3_avg|home',
        'player_star_score|season|top3_sum|away',
        'player_star_score|season|top3_sum|diff',
        'player_star_score|season|top3_sum|home',
        'player_star_score|season|top1_MIN_REC(k=20)|away',
        'player_star_score|season|top1_MIN_REC(k=20)|diff',
        'player_star_score|season|top1_MIN_REC(k=20)|home',
        'player_star_score|season|top1_MIN_REC(k=25)|away',
        'player_star_score|season|top1_MIN_REC(k=25)|diff',
        'player_star_score|season|top1_MIN_REC(k=25)|home',
        'player_star_score|season|top1_MIN_REC(k=30)|away',
        'player_star_score|season|top1_MIN_REC(k=30)|diff',
        'player_star_score|season|top1_MIN_REC(k=30)|home',
        'player_star_score|season|top1_MIN_REC(k=35)|away',
        'player_star_score|season|top1_MIN_REC(k=35)|diff',
        'player_star_score|season|top1_MIN_REC(k=35)|home',
        'player_star_score|season|top3_MIN_REC(k=20)|away',
        'player_star_score|season|top3_MIN_REC(k=20)|diff',
        'player_star_score|season|top3_MIN_REC(k=20)|home',
        'player_star_score|season|top3_MIN_REC(k=25)|away',
        'player_star_score|season|top3_MIN_REC(k=25)|diff',
        'player_star_score|season|top3_MIN_REC(k=25)|home',
        'player_star_score|season|top3_MIN_REC(k=30)|away',
        'player_star_score|season|top3_MIN_REC(k=30)|diff',
        'player_star_score|season|top3_MIN_REC(k=30)|home',
        'player_star_score|season|top3_MIN_REC(k=35)|away',
        'player_star_score|season|top3_MIN_REC(k=35)|diff',
        'player_star_score|season|top3_MIN_REC(k=35)|home',

        # NEW: player_star_share (from player_feature_updates.md)
        'player_star_share|season|top1_share|away',
        'player_star_share|season|top1_share|diff',
        'player_star_share|season|top1_share|home',
        'player_star_share|season|top3_share|away',
        'player_star_share|season|top3_share|diff',
        'player_star_share|season|top3_share|home',
        'player_star_share|season|top1_share_MIN_REC(k=20)|away',
        'player_star_share|season|top1_share_MIN_REC(k=20)|diff',
        'player_star_share|season|top1_share_MIN_REC(k=20)|home',
        'player_star_share|season|top1_share_MIN_REC(k=25)|away',
        'player_star_share|season|top1_share_MIN_REC(k=25)|diff',
        'player_star_share|season|top1_share_MIN_REC(k=25)|home',
        'player_star_share|season|top1_share_MIN_REC(k=30)|away',
        'player_star_share|season|top1_share_MIN_REC(k=30)|diff',
        'player_star_share|season|top1_share_MIN_REC(k=30)|home',
        'player_star_share|season|top1_share_MIN_REC(k=35)|away',
        'player_star_share|season|top1_share_MIN_REC(k=35)|diff',
        'player_star_share|season|top1_share_MIN_REC(k=35)|home',
        'player_star_share|season|top3_share_MIN_REC(k=20)|away',
        'player_star_share|season|top3_share_MIN_REC(k=20)|diff',
        'player_star_share|season|top3_share_MIN_REC(k=20)|home',
        'player_star_share|season|top3_share_MIN_REC(k=25)|away',
        'player_star_share|season|top3_share_MIN_REC(k=25)|diff',
        'player_star_share|season|top3_share_MIN_REC(k=25)|home',
        'player_star_share|season|top3_share_MIN_REC(k=30)|away',
        'player_star_share|season|top3_share_MIN_REC(k=30)|diff',
        'player_star_share|season|top3_share_MIN_REC(k=30)|home',
        'player_star_share|season|top3_share_MIN_REC(k=35)|away',
        'player_star_share|season|top3_share_MIN_REC(k=35)|diff',
        'player_star_share|season|top3_share_MIN_REC(k=35)|home',

        # NEW: player_star_score_all (from player_feature_updates.md)
        'player_star_score_all|season|top1|away',
        'player_star_score_all|season|top1|diff',
        'player_star_score_all|season|top1|home',

        # NEW: player_rotation_count (from player_feature_updates.md)
        'player_rotation_count|season|raw|away',
        'player_rotation_count|season|raw|diff',
        'player_rotation_count|season|raw|home',

        # NEW: player_continuity (from player_feature_updates.md)
        'player_continuity|season|avg|away',
        'player_continuity|season|avg|diff',
        'player_continuity|season|avg|home',
    ]

    # Features returned by StatHandlerV2.get_injury_features()
    # Updated to include star-based injury features from documentation/player_feature_updates.md
    ACTUAL_INJURY_FEATURES = [
        'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|away',
        'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|diff',
        'inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|home',
        'inj_min_lost|none|raw|away',
        'inj_min_lost|none|raw|diff',
        'inj_min_lost|none|raw|home',
        'inj_per_share|none|top1_avg|away',
        'inj_per_share|none|top1_avg|diff',
        'inj_per_share|none|top1_avg|home',
        'inj_per_share|none|top3_sum|away',
        'inj_per_share|none|top3_sum|diff',
        'inj_per_share|none|top3_sum|home',
        'inj_per_weighted_share|none|weighted_MIN|away',
        'inj_per_weighted_share|none|weighted_MIN|diff',
        'inj_per_weighted_share|none|weighted_MIN|home',
        'inj_per|none|top1_avg|away',
        'inj_per|none|top1_avg|diff',
        'inj_per|none|top1_avg|home',
        'inj_per|none|top3_sum|away',
        'inj_per|none|top3_sum|diff',
        'inj_per|none|top3_sum|home',
        'inj_per|none|weighted_MIN|away',
        'inj_per|none|weighted_MIN|diff',
        'inj_per|none|weighted_MIN|home',
        'inj_rotation_per|none|raw|away',
        'inj_rotation_per|none|raw|diff',
        'inj_rotation_per|none|raw|home',
        'inj_severity|none|raw|away',
        'inj_severity|none|raw|diff',
        'inj_severity|none|raw|home',
        'inj_severity|season|raw|away',
        'inj_severity|season|raw|diff',
        'inj_severity|season|raw|home',

        # NEW: Star-based injury features (from player_feature_updates.md)
        'inj_star_share|none|raw|away',
        'inj_star_share|none|raw|diff',
        'inj_star_share|none|raw|home',
        'inj_star_score_share|none|top3_sum|away',
        'inj_star_score_share|none|top3_sum|diff',
        'inj_star_score_share|none|top3_sum|home',
        'inj_top1_star_out|none|raw|away',
        'inj_top1_star_out|none|raw|diff',
        'inj_top1_star_out|none|raw|home',
    ]

    @classmethod
    def get_features_for_group(cls, group_name: str, include_side: bool = True, league=None) -> List[str]:
        """
        Get ALL valid feature combinations for a feature group.

        Enumerates all valid features by combining:
        - Stats in the group
        - All valid time periods for those stats
        - All valid calc weights for those stats
        - All valid perspectives for those stats

        For PLAYER_TALENT and INJURIES groups, returns the actual features
        that calculators generate (not theoretical combinations).

        IMPORTANT: Features are excluded if they would be categorized into a
        different group based on filter_substring rules. For example,
        margin_h2h|season|* is excluded from outcome_strength because 'h2h'
        in the feature name means it belongs to the H2H group.

        Args:
            group_name: Name of the feature group (e.g., 'outcome_strength')
            include_side: Whether to include |side variants
            league: Optional LeagueConfig to include league-specific groups

        Returns:
            List of all valid feature names for the group
        """
        # For player and injury features, return what calculators actually generate
        # This prevents feature name mismatches that cause all values to be 0
        if group_name == cls.PLAYER_TALENT:
            return sorted(cls.ACTUAL_PLAYER_FEATURES)
        if group_name == cls.INJURIES:
            return sorted(cls.ACTUAL_INJURY_FEATURES)

        all_groups = cls.get_all_group_definitions(league=league)
        group = all_groups.get(group_name)
        if not group:
            return []

        stats = group.get("stats", [])
        features = []

        # Collect filter_substrings from OTHER groups to exclude features that belong elsewhere
        # This ensures SSoT: a feature with 'h2h' in it goes to H2H, not outcome_strength
        other_group_filters = []
        for other_name, other_def in all_groups.items():
            if other_name != group_name:
                filter_sub = other_def.get("filter_substring")
                if filter_sub:
                    other_group_filters.append(filter_sub)

        # Standard time periods to enumerate
        base_time_periods = ['season', 'none']
        param_time_periods = [
            'games_2', 'games_3', 'games_5', 'games_10', 'games_12', 'games_20', 'games_50',
            'days_2', 'days_3', 'days_5', 'days_12',
            'months_1', 'months_2',
            'games_close5',
            'last_3', 'last_5',  # H2H cross-season lookups
        ]
        # Composite time periods (blend, delta, blend-delta)
        composite_time_periods = [
            'blend:season:0.80/games_12:0.20',
            'blend:season:0.70/games_20:0.20/games_12:0.10',
            'blend:games_5:0.70/games_10:0.30',
            'delta:games_5-season',
            'delta:games_10-season',
            'delta:blend:games_5:0.70/games_10:0.30-season',
        ]
        all_time_periods = base_time_periods + param_time_periods + composite_time_periods

        # Default perspectives
        default_perspectives = ['diff', 'home', 'away']

        for stat_name in stats:
            stat_def = FeatureRegistry.STAT_DEFINITIONS.get(stat_name)
            if not stat_def:
                continue

            # Get valid combinations from stat definition
            calc_weights = stat_def.valid_calc_weights if stat_def.valid_calc_weights else {'raw', 'avg'}

            if stat_def.valid_time_periods:
                stat_time_periods = []
                for tp in all_time_periods:
                    if tp in stat_def.valid_time_periods:
                        stat_time_periods.append(tp)
                    elif tp.startswith(('blend:', 'delta:')):
                        # For composite tps, check all leaf tps are allowed
                        leaves = FeatureRegistry._extract_leaf_time_periods(tp)
                        if all(l in stat_def.valid_time_periods for l in leaves):
                            stat_time_periods.append(tp)
            else:
                stat_time_periods = all_time_periods

            stat_perspectives = list(stat_def.valid_perspectives) if stat_def.valid_perspectives else default_perspectives

            # Enumerate all combinations
            for time_period in stat_time_periods:
                for calc_weight in calc_weights:
                    for perspective in stat_perspectives:
                        feature = f'{stat_name}|{time_period}|{calc_weight}|{perspective}'

                        # Validate against registry
                        is_valid, _ = FeatureRegistry.validate_feature(feature)
                        if is_valid:
                            features.append(feature)

                        # Add side variant if supported
                        if include_side and stat_def.supports_side_split:
                            feature_side = f'{feature}|side'
                            is_valid, _ = FeatureRegistry.validate_feature(feature_side)
                            if is_valid:
                                features.append(feature_side)

        # Apply substring filter if specified (e.g., H2H group only includes features with "h2h")
        filter_substring = group.get("filter_substring")
        if filter_substring:
            # Keep only features that match the filter from our stats
            features = [f for f in features if filter_substring in f.lower()]

            # ALSO include features from OTHER stats that match our filter_substring
            # This ensures h2h features appear in H2H group based on filter_substring matching
            for other_name, other_def in all_groups.items():
                if other_name == group_name:
                    continue
                # Skip other filter_substring groups (they have their own logic)
                if other_def.get("filter_substring"):
                    continue

                # Check all stats from this other group for features matching our filter
                for other_stat in other_def.get("stats", []):
                    other_stat_def = FeatureRegistry.STAT_DEFINITIONS.get(other_stat)
                    if not other_stat_def:
                        continue

                    other_calc_weights = other_stat_def.valid_calc_weights if other_stat_def.valid_calc_weights else {'raw', 'avg'}
                    if other_stat_def.valid_time_periods:
                        other_time_periods = [tp for tp in all_time_periods if tp in other_stat_def.valid_time_periods]
                    else:
                        other_time_periods = all_time_periods
                    other_perspectives = list(other_stat_def.valid_perspectives) if other_stat_def.valid_perspectives else default_perspectives

                    for time_period in other_time_periods:
                        for calc_weight in other_calc_weights:
                            for perspective in other_perspectives:
                                feature = f'{other_stat}|{time_period}|{calc_weight}|{perspective}'
                                # Only add if it matches our filter_substring
                                if filter_substring in feature.lower():
                                    is_valid, _ = FeatureRegistry.validate_feature(feature)
                                    if is_valid:
                                        features.append(feature)

                                    if include_side and other_stat_def.supports_side_split:
                                        feature_side = f'{feature}|side'
                                        is_valid, _ = FeatureRegistry.validate_feature(feature_side)
                                        if is_valid:
                                            features.append(feature_side)

        # Exclude features that belong to OTHER groups based on their filter_substring
        # This is the SSoT enforcement: h2h features go to H2H, close features go to close_games
        if other_group_filters and not filter_substring:
            # Only apply exclusion if this group doesn't have its own filter
            features = [
                f for f in features
                if not any(other_filter in f.lower() for other_filter in other_group_filters)
            ]

        return sorted(set(features))

    @classmethod
    def get_all_features(cls, include_side: bool = True, league=None) -> Dict[str, List[str]]:
        """
        Get ALL valid features organized by group.

        Args:
            include_side: Whether to include |side variants
            league: Optional LeagueConfig to include league-specific groups

        Returns:
            Dict mapping group name to list of all valid feature names
        """
        all_groups = cls.get_all_group_definitions(league=league)
        return {
            group_name: cls.get_features_for_group(group_name, include_side=include_side, league=league)
            for group_name in all_groups.keys()
        }

    @classmethod
    def get_all_features_flat(cls, include_side: bool = True, league=None) -> List[str]:
        """
        Get ALL valid features as a flat list.

        Args:
            include_side: Whether to include |side variants
            league: Optional LeagueConfig to include league-specific groups

        Returns:
            List of all valid feature names across all groups
        """
        all_groups = cls.get_all_group_definitions(league=league)
        all_features = []
        for group_name in all_groups.keys():
            all_features.extend(cls.get_features_for_group(group_name, include_side=include_side, league=league))
        return sorted(set(all_features))

    @classmethod
    def get_group_description(cls, group_name: str, league=None) -> str:
        """Get the description for a feature group."""
        all_groups = cls.get_all_group_definitions(league=league)
        group = all_groups.get(group_name)
        return group.get("description", "") if group else ""

    @classmethod
    def get_group_layer(cls, group_name: str, league=None) -> int:
        """Get the layer number for a feature group."""
        all_groups = cls.get_all_group_definitions(league=league)
        group = all_groups.get(group_name)
        return group.get("layer", 0) if group else 0

    @classmethod
    def get_group_for_feature(cls, feature_name: str, league=None) -> str:
        """
        Determine which group a feature belongs to.

        This is the SSoT for feature categorization - used by web UI and other consumers.

        Args:
            feature_name: The feature name (e.g., 'margin|season|avg|diff')
            league: Optional LeagueConfig to include league-specific groups

        Returns:
            Group name (e.g., 'outcome_strength', 'h2h', 'other')
        """
        feature_lower = feature_name.lower()

        # Parse the stat name from the feature (first component)
        parts = feature_name.split('|')
        stat_name = parts[0] if parts else ''

        # Special prefixes that override stat-based categorization
        if feature_name.startswith('pred_'):
            return cls.PREDICTION_FEATURES
        if feature_name.startswith('inj_'):
            return cls.INJURIES
        if feature_name.startswith('player_'):
            return cls.PLAYER_TALENT
        if feature_name.startswith('vegas_'):
            return cls.VEGAS_LINES

        # Check groups with filter_substring first (most specific)
        # H2H: any feature with 'h2h' in the name
        if 'h2h' in feature_lower:
            return cls.H2H

        # CLOSE_GAMES: any feature with 'close' in the name
        if 'close' in feature_lower:
            return cls.CLOSE_GAMES

        # Check stat-based groups
        all_groups = cls.get_all_group_definitions(league=league)
        for group_name, group_def in all_groups.items():
            # Skip groups with filter_substring (already handled above)
            if group_def.get('filter_substring'):
                continue
            stats = group_def.get('stats', [])
            if stat_name in stats:
                return group_name

        # Additional substring-based fallbacks for common patterns
        if 'elo' in feature_lower:
            return cls.ELO_STRENGTH
        if 'per' in feature_lower and not feature_name.startswith('inj_'):
            return cls.PLAYER_TALENT
        if 'rel' in feature_lower:
            return 'era_normalization'

        return 'other'

    @classmethod
    def categorize_features(cls, feature_dict: dict) -> dict:
        """
        Categorize a dict of features into groups.

        This is the SSoT for bulk feature categorization - used by web UI.

        Args:
            feature_dict: Dict of {feature_name: value}

        Returns:
            Dict of {group_name: [{'name': feature_name, 'value': value}, ...]}
        """
        categories = {}

        for feature_name, value in feature_dict.items():
            # Skip internal metadata keys
            if isinstance(feature_name, str) and feature_name.startswith('_'):
                continue

            group = cls.get_group_for_feature(feature_name)
            if group not in categories:
                categories[group] = []
            categories[group].append({'name': feature_name, 'value': value})

        # Sort features within each category and remove empty categories
        return {k: sorted(v, key=lambda x: x['name']) for k, v in categories.items() if v}
