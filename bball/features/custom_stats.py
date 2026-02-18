"""
Custom stat handler functions for basketball.

Each handler computes stats that cannot be expressed as simple YAML db_field
extraction + standard aggregation:
- Rate/efficiency stats requiring multi-field formulas (eFG%, TS%, off/def RTG)
- Net stats (team stat minus opponent defensive stat)
- Derived matchup interactions (pace_interaction, exp_points, margin)
- Schedule/fatigue stats (travel, b2b, days_rest, pace, games_played)
- Head-to-head matchup stats (h2h_win_pct, margin_h2h)
- Conference-specific stats (conf_wins, conf_margin)
- Elo ratings (requires EloCache)
- Vegas betting lines (requires game document lookup)
- Game context (postseason, close_win_pct, home_court)
- Player PER features (delegates to PERCalculator)
- Injury impact features (delegates to injury computation)

Handler signature:
    handler(stat_name, time_period, calc_weight, perspective,
            home_team, away_team, home_games, away_games, **context)
"""

from collections import defaultdict
from datetime import date, datetime, timedelta
from math import log1p, exp, radians, sin, cos, sqrt, atan2
from typing import Dict, List, Optional, Tuple


# ============================================================================
# Shared helpers
# ============================================================================

def _apply_perspective(home_val, away_val, perspective):
    """Combine home/away values based on perspective."""
    if perspective == "diff":
        if home_val is not None and away_val is not None:
            return home_val - away_val
        return None
    elif perspective == "home":
        return home_val
    elif perspective == "away":
        return away_val
    elif perspective == "none":
        return home_val
    return None


def _get_team_data(game, team_name):
    """Return (team_data, opp_data, is_home) for a game."""
    is_home = game.get("homeTeam", {}).get("name") == team_name
    if is_home:
        return game.get("homeTeam", {}), game.get("awayTeam", {}), True
    return game.get("awayTeam", {}), game.get("homeTeam", {}), False


def _window_and_filter(games, time_period, reference_date, team_name,
                       has_side, side_type, engine=None):
    """Window games by time period and optionally filter to side-specific.

    Args:
        side_type: "home" or "away" - which side this team plays in the matchup
    """
    if engine:
        windowed = engine._window_games(games, time_period, reference_date)
    else:
        windowed = games

    if has_side:
        if side_type == "home":
            windowed = [g for g in windowed
                        if g.get("homeTeam", {}).get("name") == team_name]
        else:
            windowed = [g for g in windowed
                        if g.get("awayTeam", {}).get("name") == team_name]

    return windowed


def _std(values):
    """Sample standard deviation (n-1 denominator)."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5


def _count_wins(team, games):
    """Count wins for a team across games."""
    wins = 0
    for game in games:
        is_home = game.get("homeTeam", {}).get("name") == team
        home_won = game.get("homeWon", False)
        if (is_home and home_won) or (not is_home and not home_won):
            wins += 1
    return wins


def haversine_miles(lat1, lon1, lat2, lon2):
    """Great circle distance between two points on Earth (miles)."""
    R = 3959
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


# ============================================================================
# Rate stat computation infrastructure
# ============================================================================

# DB field names for pre-computed per-game values
_DB_FIELDS = {
    "efg": "effective_fg_perc",
    "ts": "true_shooting_perc",
    "three_pct": "three_perc",
    "off_rtg": "off_rtg",
    "def_rtg": "def_rtg",
    "assists_ratio": "assists_ratio",
    "to_metric": "TO_metric",
}

# Basic stat db_field mapping (for net stat sub-computations)
_BASIC_FIELDS = {
    "points": "points",
    "three_made": "three_made",
    "assists": "assists",
    "blocks": "blocks",
    "steals": "steals",
    "reb_total": "total_reb",
}

# Stats requiring multi-game aggregation (can't compute per-game independently)
_AGGREGATION_REQUIRED = frozenset({
    "off_rtg", "def_rtg", "assists_ratio",
    "opp_efg", "opp_ts", "opp_three_pct",
    "opp_assists_ratio", "opp_points",
})


def _per_game_rate(stat_name, team_data, opp_data, game):
    """Compute a rate/derived stat for a single game from raw components."""
    if stat_name in ("efg", "effective_fg_perc"):
        fg_made = team_data.get("FG_made", 0) or 0
        fg_att = team_data.get("FG_att", 0) or 0
        three_made = team_data.get("three_made", 0) or 0
        return ((fg_made + 0.5 * three_made) / fg_att * 100) if fg_att > 0 else None

    if stat_name in ("ts", "true_shooting_perc"):
        points = team_data.get("points", 0) or 0
        fg_att = team_data.get("FG_att", 0) or 0
        ft_att = team_data.get("FT_att", 0) or 0
        denom = 2 * (fg_att + 0.44 * ft_att)
        return (points / denom * 100) if denom > 0 else None

    if stat_name in ("three_pct", "three_perc"):
        three_made = team_data.get("three_made", 0) or 0
        three_att = team_data.get("three_att", 0) or 0
        return (three_made / three_att * 100) if three_att > 0 else None

    if stat_name in ("to_metric", "TO_metric"):
        to = team_data.get("TO", 0) or 0
        fg_att = team_data.get("FG_att", 0) or 0
        ft_att = team_data.get("FT_att", 0) or 0
        denom = fg_att + 0.44 * ft_att + to
        return (100 * to / denom) if denom > 0 else None

    if stat_name == "ast_to_ratio":
        assists = team_data.get("assists", 0) or 0
        to = team_data.get("TO", 0) or 0
        return (assists / to) if to > 0 else None

    if stat_name == "three_rate":
        three_att = team_data.get("three_att", 0) or 0
        fg_att = team_data.get("FG_att", 0) or 0
        return (float(three_att) / float(fg_att)) if fg_att > 0 else None

    if stat_name == "ft_rate":
        ft_att = team_data.get("FT_att", 0) or 0
        fg_att = team_data.get("FG_att", 0) or 0
        return (float(ft_att) / float(fg_att)) if fg_att > 0 else None

    if stat_name == "wins":
        team_name = team_data.get("name")
        if team_name is None:
            return None
        is_home = game.get("homeTeam", {}).get("name") == team_name
        home_won = game.get("homeWon", False)
        return 1.0 if (is_home and home_won) or (not is_home and not home_won) else 0.0

    if stat_name == "pace":
        fga = team_data.get("FG_att", 0) or 0
        oreb = team_data.get("off_reb", 0) or 0
        to = team_data.get("TO", 0) or 0
        fta = team_data.get("FT_att", 0) or 0
        return fga - oreb + to + 0.44 * fta

    if stat_name == "margin":
        team_name = team_data.get("name")
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        is_home = game.get("homeTeam", {}).get("name") == team_name
        return (home_pts - away_pts) if is_home else (away_pts - home_pts)

    # Opponent per-game stats
    if stat_name == "opp_efg":
        fg_made = opp_data.get("FG_made", 0) or 0
        fg_att = opp_data.get("FG_att", 0) or 0
        three_made = opp_data.get("three_made", 0) or 0
        return ((fg_made + 0.5 * three_made) / fg_att * 100) if fg_att > 0 else None

    if stat_name == "opp_ts":
        pts = opp_data.get("points", 0) or 0
        fg_att = opp_data.get("FG_att", 0) or 0
        ft_att = opp_data.get("FT_att", 0) or 0
        denom = 2 * (fg_att + 0.44 * ft_att)
        return (pts / denom * 100) if denom > 0 else None

    if stat_name == "opp_three_pct":
        three_made = opp_data.get("three_made", 0) or 0
        three_att = opp_data.get("three_att", 0) or 0
        return (three_made / three_att * 100) if three_att > 0 else None

    # Basic field extraction (for sub-computations like net stats)
    basic_field = _BASIC_FIELDS.get(stat_name)
    if basic_field:
        val = team_data.get(basic_field)
        return float(val) if val is not None else None

    # Opponent basic field extraction
    if stat_name.startswith("opp_"):
        base = stat_name[4:]
        basic_field = _BASIC_FIELDS.get(base)
        if basic_field:
            val = opp_data.get(basic_field)
            return float(val) if val is not None else None

    return None


def _build_aggregates(team, games):
    """Build team and opponent aggregate stat dicts from a list of games."""
    team_agg = defaultdict(float)
    opp_agg = defaultdict(float)

    for game in games:
        home_name = game.get("homeTeam", {}).get("name")
        away_name = game.get("awayTeam", {}).get("name")

        if home_name == team:
            td = game.get("homeTeam", {})
            od = game.get("awayTeam", {})
        elif away_name == team:
            td = game.get("awayTeam", {})
            od = game.get("homeTeam", {})
        else:
            continue

        for key, val in td.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                team_agg[key] += val
        for key, val in od.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                opp_agg[key] += val

    return team_agg, opp_agg


def _aggregate_rate(stat_name, team_agg, opp_agg, n_games, games=None):
    """Compute a stat from aggregated totals across multiple games."""
    if stat_name in ("efg", "effective_fg_perc"):
        fa = team_agg.get("FG_att", 0)
        if fa > 0:
            return ((team_agg.get("FG_made", 0) + 0.5 * team_agg.get("three_made", 0)) / fa) * 100
        return None

    if stat_name in ("ts", "true_shooting_perc"):
        fa = team_agg.get("FG_att", 0)
        fta = team_agg.get("FT_att", 0)
        d = 2 * (fa + 0.44 * fta)
        return (team_agg.get("points", 0) / d * 100) if d > 0 else None

    if stat_name in ("three_pct", "three_perc"):
        ta = team_agg.get("three_att", 0)
        return (team_agg.get("three_made", 0) / ta * 100) if ta > 0 else None

    if stat_name in ("to_metric", "TO_metric"):
        to = team_agg.get("TO", 0)
        fa = team_agg.get("FG_att", 0)
        fta = team_agg.get("FT_att", 0)
        d = fa + 0.44 * fta + to
        return (100 * to / d) if d > 0 else None

    if stat_name == "ast_to_ratio":
        to = team_agg.get("TO", 0)
        return (team_agg.get("assists", 0) / to) if to > 0 else None

    if stat_name == "off_rtg":
        fa = team_agg.get("FG_att", 0)
        poss = fa - team_agg.get("off_reb", 0) + team_agg.get("TO", 0) + 0.44 * team_agg.get("FT_att", 0)
        return (team_agg.get("points", 0) / poss * 100) if poss > 0 else None

    if stat_name == "def_rtg":
        fa = opp_agg.get("FG_att", 0)
        poss = fa - opp_agg.get("off_reb", 0) + opp_agg.get("TO", 0) + 0.44 * opp_agg.get("FT_att", 0)
        return (opp_agg.get("points", 0) / poss * 100) if poss > 0 else None

    if stat_name == "assists_ratio":
        ast = team_agg.get("assists", 0)
        fg_made = team_agg.get("FG_made", 0)
        if fg_made > 0 and games:
            total_min = sum(53 if g.get("OT", False) else 48 for g in games)
            d = (total_min / 5) * fg_made - fg_made
            return (100 * ast / d) if d > 0 else None
        return None

    if stat_name == "three_rate":
        fa = team_agg.get("FG_att", 0)
        return (team_agg.get("three_att", 0) / fa) if fa > 0 else 0.0

    if stat_name == "ft_rate":
        fa = team_agg.get("FG_att", 0)
        return (team_agg.get("FT_att", 0) / fa) if fa > 0 else 0.0

    if stat_name == "pace":
        fa = team_agg.get("FG_att", 0)
        poss = fa - team_agg.get("off_reb", 0) + team_agg.get("TO", 0) + 0.44 * team_agg.get("FT_att", 0)
        return poss / n_games if n_games > 0 else 0.0

    if stat_name == "wins":
        # Wins are counted via _count_wins, not aggregated from fields
        return None

    # Opponent aggregate stats
    if stat_name in ("opp_efg", "opp_effective_fg_perc"):
        fa = opp_agg.get("FG_att", 0)
        if fa > 0:
            return ((opp_agg.get("FG_made", 0) + 0.5 * opp_agg.get("three_made", 0)) / fa) * 100
        return None

    if stat_name in ("opp_ts", "opp_true_shooting_perc"):
        fa = opp_agg.get("FG_att", 0)
        fta = opp_agg.get("FT_att", 0)
        d = 2 * (fa + 0.44 * fta)
        return (opp_agg.get("points", 0) / d * 100) if d > 0 else None

    if stat_name in ("opp_three_pct", "opp_three_perc"):
        ta = opp_agg.get("three_att", 0)
        return (opp_agg.get("three_made", 0) / ta * 100) if ta > 0 else None

    if stat_name == "opp_assists_ratio":
        opp_ast = opp_agg.get("assists", 0)
        opp_fg_made = opp_agg.get("FG_made", 0)
        if opp_fg_made > 0 and games:
            total_min = sum(53 if g.get("OT", False) else 48 for g in games)
            d = (total_min / 5) * opp_fg_made - opp_fg_made
            return (100 * opp_ast / d) if d > 0 else None
        return None

    if stat_name == "opp_points":
        return opp_agg.get("points", 0) / n_games if n_games > 0 else None

    # Basic stat aggregation (raw sum)
    basic_field = _BASIC_FIELDS.get(stat_name)
    if basic_field:
        return team_agg.get(basic_field, 0)

    # Opponent basic stat aggregation
    if stat_name.startswith("opp_"):
        base = stat_name[4:]
        basic_field = _BASIC_FIELDS.get(base)
        if basic_field:
            return opp_agg.get(basic_field, 0)

    return None


def _collect_per_game_values(stat_name, team, games):
    """Collect per-game values for a stat."""
    db_field = _DB_FIELDS.get(stat_name) or _BASIC_FIELDS.get(stat_name)
    values = []

    for game in games:
        td, od, _ = _get_team_data(game, team)
        val = None

        # Try pre-computed db field first
        if db_field and db_field in td:
            raw = td.get(db_field)
            if raw is not None:
                val = float(raw)

        # Fall back to formula computation
        if val is None:
            val = _per_game_rate(stat_name, td, od, game)

        if val is not None:
            values.append(val)

    return values


def _compute_stat_for_team(stat_name, team, games, calc_weight):
    """Compute a stat value for a single team from a list of games.

    Returns the computed value, or 0.0 if computation fails.
    """
    if not games:
        return 0.0

    # Special: Wins
    if stat_name == "wins":
        wins = _count_wins(team, games)
        if calc_weight == "raw":
            return float(wins)
        elif calc_weight == "std":
            values = []
            for game in games:
                is_home = game.get("homeTeam", {}).get("name") == team
                home_won = game.get("homeWon", False)
                values.append(1.0 if (is_home and home_won) or (not is_home and not home_won) else 0.0)
            return _std(values)
        else:  # avg = win percentage
            return float(wins) / len(games)

    # For 'avg' and 'std': try per-game computation first
    if calc_weight in ("avg", "std"):
        if stat_name not in _AGGREGATION_REQUIRED:
            values = _collect_per_game_values(stat_name, team, games)
            if values:
                if calc_weight == "std":
                    return _std(values)
                return sum(values) / len(values)

    # Aggregation-required stats can't compute std
    if stat_name in _AGGREGATION_REQUIRED and calc_weight == "std":
        return 0.0

    # Aggregate computation (for raw, or fallback for avg)
    team_agg, opp_agg = _build_aggregates(team, games)
    result = _aggregate_rate(stat_name, team_agg, opp_agg, len(games), games)
    if result is not None:
        # For basic stats with 'avg', divide by game count
        if calc_weight == "avg" and stat_name in _BASIC_FIELDS:
            return result / len(games) if len(games) > 0 else 0.0
        return result

    return 0.0


# ============================================================================
# Handler: compute_basic_rate
# ============================================================================

def compute_basic_rate(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute rate/efficiency stats, wins, and pace.

    Handles: efg, ts, three_pct, to_metric, ast_to_ratio, off_rtg, def_rtg,
             assists_ratio, wins, reb_off_pct, three_pct_allowed, pace
    """
    engine = context.get("engine")
    reference_date = context.get("reference_date")
    has_side = context.get("has_side", False)

    h_games = _window_and_filter(
        home_games, time_period, reference_date, home_team, has_side, "home", engine)
    a_games = _window_and_filter(
        away_games, time_period, reference_date, away_team, has_side, "away", engine)

    home_val = _compute_stat_for_team(stat_name, home_team, h_games, calc_weight)
    away_val = _compute_stat_for_team(stat_name, away_team, a_games, calc_weight)

    return _apply_perspective(home_val, away_val, perspective)


# ============================================================================
# Handler: compute_net
# ============================================================================

# Map base stat (without _net) to (team_stat_name, opponent_stat_name)
_NET_STAT_MAP = {
    "efg": ("efg", "opp_efg"),
    "ts": ("ts", "opp_ts"),
    "three_pct": ("three_pct", "opp_three_pct"),
    "off_rtg": ("off_rtg", "def_rtg"),
    "def_rtg": ("def_rtg", "off_rtg"),
    "assists_ratio": ("assists_ratio", "opp_assists_ratio"),
    "points": ("points", "opp_points"),
    "three_made": ("three_made", "opp_three_made"),
}


def compute_net(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute net stats (team stat - opponent defensive stat).

    Handles: efg_net, ts_net, three_pct_net, off_rtg_net, def_rtg_net,
             assists_ratio_net, points_net, three_made_net
    """
    engine = context.get("engine")
    reference_date = context.get("reference_date")
    has_side = context.get("has_side", False)

    # Strip _net suffix
    base_stat = stat_name[:-4] if stat_name.endswith("_net") else stat_name

    if base_stat not in _NET_STAT_MAP:
        return 0.0

    team_stat_name, opp_stat_name = _NET_STAT_MAP[base_stat]

    h_games = _window_and_filter(
        home_games, time_period, reference_date, home_team, has_side, "home", engine)
    a_games = _window_and_filter(
        away_games, time_period, reference_date, away_team, has_side, "away", engine)

    # Home team net
    home_team_stat = _compute_stat_for_team(team_stat_name, home_team, h_games, calc_weight)
    home_opp_stat = _compute_stat_for_team(opp_stat_name, home_team, h_games, calc_weight)
    home_net = home_team_stat - home_opp_stat

    # Away team net
    away_team_stat = _compute_stat_for_team(team_stat_name, away_team, a_games, calc_weight)
    away_opp_stat = _compute_stat_for_team(opp_stat_name, away_team, a_games, calc_weight)
    away_net = away_team_stat - away_opp_stat

    return _apply_perspective(home_net, away_net, perspective)


# ============================================================================
# Handler: compute_derived
# ============================================================================

def compute_derived(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute derived matchup-level stats.

    Handles: pace_interaction, est_possessions, exp_points_off, exp_points_def,
             exp_points_matchup, margin, three_rate, ft_rate, three_pct_matchup
    """
    engine = context.get("engine")
    reference_date = context.get("reference_date")
    has_side = context.get("has_side", False)

    if stat_name == "pace_interaction":
        return _compute_pace_interaction(
            time_period, perspective, home_team, away_team,
            home_games, away_games, context)

    if stat_name == "est_possessions":
        return _compute_est_possessions(
            time_period, perspective, home_team, away_team,
            home_games, away_games, context)

    if stat_name in ("exp_points_off", "exp_points_def", "exp_points_matchup"):
        return _compute_exp_points(
            stat_name, time_period, perspective, home_team, away_team,
            home_games, away_games, context)

    if stat_name == "margin":
        return _compute_margin(
            time_period, calc_weight, perspective, has_side,
            home_team, away_team, home_games, away_games, context)

    if stat_name in ("three_rate", "ft_rate"):
        return _compute_attempt_rate(
            stat_name, time_period, calc_weight, perspective, has_side,
            home_team, away_team, home_games, away_games, context)

    if stat_name == "three_pct_matchup":
        return _compute_three_pct_matchup(
            time_period, perspective, home_team, away_team,
            home_games, away_games, context)

    return None


def _compute_pace_interaction(time_period, perspective, home_team, away_team,
                              home_games, away_games, context):
    """Harmonic mean of both teams' pace. Same value for home/away."""
    engine = context.get("engine")
    ctx = dict(context)

    p_home = engine.compute_feature(
        f"pace|{time_period}|avg|home", home_team, away_team,
        home_games, away_games, ctx) or 0.0
    p_away = engine.compute_feature(
        f"pace|{time_period}|avg|away", home_team, away_team,
        home_games, away_games, ctx) or 0.0

    p_home = float(p_home or 0.0)
    p_away = float(p_away or 0.0)

    interaction = 0.0
    if p_home > 0 and p_away > 0:
        interaction = (2.0 * p_home * p_away) / (p_home + p_away)
    elif (p_home > 0) != (p_away > 0):
        interaction = max(p_home, p_away)
    else:
        # Both zero — try fallback to season if using games_N
        if time_period != "season":
            season_interaction = _compute_pace_interaction(
                "season", perspective, home_team, away_team,
                home_games, away_games, context)
            if season_interaction and season_interaction > 0:
                interaction = float(season_interaction)
        if interaction == 0.0:
            interaction = context.get("league_pace_baseline", 100.0)

    if perspective == "diff":
        return 0.0
    return interaction


def _compute_est_possessions(time_period, perspective, home_team, away_team,
                             home_games, away_games, context):
    """Estimated possessions = pace_interaction value."""
    interaction = _compute_pace_interaction(
        time_period, "none", home_team, away_team,
        home_games, away_games, context)
    if perspective in ("home", "away", "none"):
        return float(interaction or 0.0)
    return 0.0


def _compute_exp_points(stat_name, time_period, perspective, home_team,
                        away_team, home_games, away_games, context):
    """Expected points from off/def ratings and estimated possessions."""
    engine = context.get("engine")
    ctx = dict(context)

    est = _compute_est_possessions(
        time_period, "none", home_team, away_team,
        home_games, away_games, context) or 0.0

    off_home = engine.compute_feature(
        f"off_rtg|{time_period}|raw|home", home_team, away_team,
        home_games, away_games, ctx) or 0.0
    off_away = engine.compute_feature(
        f"off_rtg|{time_period}|raw|away", home_team, away_team,
        home_games, away_games, ctx) or 0.0
    def_home = engine.compute_feature(
        f"def_rtg|{time_period}|raw|home", home_team, away_team,
        home_games, away_games, ctx) or 0.0
    def_away = engine.compute_feature(
        f"def_rtg|{time_period}|raw|away", home_team, away_team,
        home_games, away_games, ctx) or 0.0

    exp_off_home = (float(off_home) / 100.0) * float(est)
    exp_off_away = (float(off_away) / 100.0) * float(est)
    exp_def_home = (float(def_home) / 100.0) * float(est)
    exp_def_away = (float(def_away) / 100.0) * float(est)

    if stat_name == "exp_points_off":
        return _apply_perspective(exp_off_home, exp_off_away, perspective)
    if stat_name == "exp_points_def":
        return _apply_perspective(exp_def_home, exp_def_away, perspective)
    # exp_points_matchup
    home_exp = 0.5 * exp_off_home + 0.5 * exp_def_away
    away_exp = 0.5 * exp_off_away + 0.5 * exp_def_home
    return _apply_perspective(home_exp, away_exp, perspective)


def _compute_margin(time_period, calc_weight, perspective, has_side,
                    home_team, away_team, home_games, away_games, context):
    """Point differential per game."""
    engine = context.get("engine")
    reference_date = context.get("reference_date")

    h_games = _window_and_filter(
        home_games, time_period, reference_date, home_team, has_side, "home", engine)
    a_games = _window_and_filter(
        away_games, time_period, reference_date, away_team, has_side, "away", engine)

    def _get_margins(team, games):
        margins = []
        for game in games:
            home_pts = game.get("homeTeam", {}).get("points", 0) or 0
            away_pts = game.get("awayTeam", {}).get("points", 0) or 0
            is_home = game.get("homeTeam", {}).get("name") == team
            margins.append((home_pts - away_pts) if is_home else (away_pts - home_pts))
        return margins

    def _calc(margins):
        if not margins:
            return 0.0
        if calc_weight == "avg":
            return sum(margins) / len(margins)
        elif calc_weight == "std":
            return _std(margins)
        elif calc_weight == "raw":
            return float(sum(margins))
        return 0.0

    home_val = _calc(_get_margins(home_team, h_games))
    away_val = _calc(_get_margins(away_team, a_games))

    return _apply_perspective(home_val, away_val, perspective)


def _compute_attempt_rate(stat_name, time_period, calc_weight, perspective,
                          has_side, home_team, away_team, home_games,
                          away_games, context):
    """three_rate / ft_rate: ratio of totals with side filtering.

    For raw/avg: aggregate totals then compute ratio.
    For std: per-game computation then std dev.
    """
    engine = context.get("engine")
    reference_date = context.get("reference_date")

    # For std, use per-game approach via compute_basic_rate
    if calc_weight == "std":
        h_games = _window_and_filter(
            home_games, time_period, reference_date, home_team, has_side, "home", engine)
        a_games = _window_and_filter(
            away_games, time_period, reference_date, away_team, has_side, "away", engine)
        home_val = _compute_stat_for_team(stat_name, home_team, h_games, "std")
        away_val = _compute_stat_for_team(stat_name, away_team, a_games, "std")
        return _apply_perspective(home_val, away_val, perspective)

    # For raw/avg: ratio-of-totals with side split
    h_games = _window_and_filter(
        home_games, time_period, reference_date, home_team, False, "home", engine)
    a_games = _window_and_filter(
        away_games, time_period, reference_date, away_team, False, "away", engine)

    def _ratio_for(team, games, require_side):
        if require_side == "home":
            games = [g for g in games if g.get("homeTeam", {}).get("name") == team]
        elif require_side == "away":
            games = [g for g in games if g.get("awayTeam", {}).get("name") == team]

        total_fg_att = 0.0
        total_three_att = 0.0
        total_ft_att = 0.0

        for g in games:
            td, _, _ = _get_team_data(g, team)
            total_fg_att += float(td.get("FG_att", 0) or 0)
            total_three_att += float(td.get("three_att", 0) or 0)
            total_ft_att += float(td.get("FT_att", 0) or 0)

        if total_fg_att <= 0:
            return 0.0
        if stat_name == "three_rate":
            return total_three_att / total_fg_att
        return total_ft_att / total_fg_att

    home_val = _ratio_for(home_team, h_games, "home")
    away_val = _ratio_for(away_team, a_games, "away")

    return _apply_perspective(home_val, away_val, perspective)


def _compute_three_pct_matchup(time_period, perspective, home_team, away_team,
                               home_games, away_games, context):
    """Team 3PT% vs opponent 3PT% defense."""
    engine = context.get("engine")
    ctx = dict(context)

    # Team's 3PT%
    home_three_pct = engine.compute_feature(
        f"three_pct|{time_period}|avg|home", home_team, away_team,
        home_games, away_games, ctx) or 0.0
    away_three_pct = engine.compute_feature(
        f"three_pct|{time_period}|avg|away", home_team, away_team,
        home_games, away_games, ctx) or 0.0

    return _apply_perspective(float(home_three_pct), float(away_three_pct), perspective)


# ============================================================================
# Handler: compute_schedule
# ============================================================================

def compute_schedule(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute schedule/fatigue features.

    Handles: games_played, road_games, pace, days_rest, b2b, first_of_b2b,
             travel, rest
    """
    engine = context.get("engine")
    reference_date = context.get("reference_date")

    if stat_name == "games_played":
        h_games = _window_and_filter(
            home_games, time_period, reference_date, home_team, False, "home", engine)
        a_games = _window_and_filter(
            away_games, time_period, reference_date, away_team, False, "away", engine)
        return _apply_perspective(float(len(h_games)), float(len(a_games)), perspective)

    if stat_name == "road_games":
        h_games = _window_and_filter(
            home_games, time_period, reference_date, home_team, False, "home", engine)
        a_games = _window_and_filter(
            away_games, time_period, reference_date, away_team, False, "away", engine)
        home_val = float(sum(1 for g in h_games
                             if g.get("awayTeam", {}).get("name") == home_team))
        away_val = float(sum(1 for g in a_games
                             if g.get("awayTeam", {}).get("name") == away_team))
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "pace":
        has_side = context.get("has_side", False)
        h_games = _window_and_filter(
            home_games, time_period, reference_date, home_team, has_side, "home", engine)
        a_games = _window_and_filter(
            away_games, time_period, reference_date, away_team, has_side, "away", engine)
        home_val = _compute_stat_for_team("pace", home_team, h_games, "avg")
        away_val = _compute_stat_for_team("pace", away_team, a_games, "avg")
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name in ("days_rest", "rest"):
        home_rest = float(_get_days_rest(home_team, home_games, reference_date))
        away_rest = float(_get_days_rest(away_team, away_games, reference_date))
        return _apply_perspective(home_rest, away_rest, perspective)

    if stat_name == "b2b":
        home_rest = _get_days_rest(home_team, home_games, reference_date)
        away_rest = _get_days_rest(away_team, away_games, reference_date)
        home_b2b = 1.0 if home_rest == 1 else 0.0
        away_b2b = 1.0 if away_rest == 1 else 0.0
        return _apply_perspective(home_b2b, away_b2b, perspective)

    if stat_name == "first_of_b2b":
        home_val = 1.0 if _team_plays_tomorrow(
            home_team, reference_date, context) else 0.0
        away_val = 1.0 if _team_plays_tomorrow(
            away_team, reference_date, context) else 0.0
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "travel":
        home_val = _compute_travel_distance(
            home_team, home_games, reference_date, time_period, context)
        away_val = _compute_travel_distance(
            away_team, away_games, reference_date, time_period, context)
        return _apply_perspective(home_val, away_val, perspective)

    return None


def _get_days_rest(team, games, reference_date, cap=7):
    """Days since last game, capped at `cap`. Returns cap if no prior game."""
    if not reference_date:
        return cap
    try:
        target = datetime.strptime(reference_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return cap

    last_date = None
    for g in sorted(games, key=lambda x: x.get("date", ""), reverse=True):
        try:
            gd = datetime.strptime(g.get("date", ""), "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        if gd < target:
            last_date = gd
            break

    if last_date is None:
        return cap
    days = (target - last_date).days
    return min(max(days, 0), cap)


def _team_plays_tomorrow(team, reference_date, context):
    """Check if team plays the day after reference_date."""
    if not reference_date:
        return False
    try:
        target = datetime.strptime(reference_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return False
    tomorrow_str = (target + timedelta(days=1)).strftime("%Y-%m-%d")

    games_home = context.get("games_home")
    games_away = context.get("games_away")
    season = context.get("season", "")
    exclude = context.get("exclude_game_types", ["preseason", "allstar"])

    if games_home is not None and games_away is not None:
        # Check preloaded data
        for s_key in ([season] if season else list(games_home.keys())):
            if s_key in games_home:
                date_games = games_home[s_key].get(tomorrow_str, {})
                for t, g in date_games.items():
                    if t == team and g.get("game_type", "regseason") not in exclude:
                        return True
            if s_key in games_away:
                date_games = games_away[s_key].get(tomorrow_str, {})
                for t, g in date_games.items():
                    if t == team and g.get("game_type", "regseason") not in exclude:
                        return True
        return False

    # Fall back to DB query
    db = context.get("db")
    league = context.get("league")
    if db is not None and league is not None:
        games_coll = league.collections.get("games", "stats_nba")
        query = {
            "season": season,
            "date": tomorrow_str,
            "game_type": {"$nin": exclude},
            "$or": [{"homeTeam.name": team}, {"awayTeam.name": team}],
        }
        return db[games_coll].count_documents(query, limit=1) > 0

    return False


def _compute_travel_distance(team, games, reference_date, time_period, context):
    """Compute travel distance for a team. Delegates to venue cache."""
    venue_cache = context.get("venue_cache", {})
    db = context.get("db")
    if not venue_cache and db is None:
        return 0.0

    if not reference_date:
        return 0.0

    try:
        target = datetime.strptime(reference_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return 0.0

    # Parse days from time_period (travel uses days_N)
    if time_period.startswith("days_"):
        try:
            n_days = int(time_period.replace("days_", ""))
        except ValueError:
            return 0.0
    else:
        return 0.0

    start_date = target - timedelta(days=n_days)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = reference_date

    # Filter games in range
    range_games = [
        g for g in games
        if start_str <= g.get("date", "") < end_str
    ]

    if not range_games:
        return 0.0

    range_games.sort(key=lambda g: g.get("date", ""))
    total_distance = 0.0
    prev_lat, prev_lon = None, None

    for game in range_games:
        venue_guid = (
            game.get("venue_guid") or game.get("venueGuid") or
            game.get("arena_guid") or game.get("arenaId") or
            (game.get("venue") or {}).get("venue_guid") or
            (game.get("venue") or {}).get("guid")
        )
        if not venue_guid:
            continue

        coords = venue_cache.get(venue_guid)
        if coords is None:
            continue

        lat, lon = coords
        if prev_lat is not None:
            dist = haversine_miles(prev_lat, prev_lon, lat, lon)
            total_distance += dist
        prev_lat, prev_lon = lat, lon

    # Add distance to current game venue
    target_venue_guid = context.get("target_venue_guid")
    if target_venue_guid and prev_lat is not None:
        target_coords = venue_cache.get(target_venue_guid)
        if target_coords:
            total_distance += haversine_miles(prev_lat, prev_lon, *target_coords)

    return total_distance


# ============================================================================
# Handler: compute_h2h
# ============================================================================

def compute_h2h(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute head-to-head matchup features.

    Handles: h2h_win_pct, margin_h2h, h2h_games_count
    """
    reference_date = context.get("reference_date")
    has_side = context.get("has_side", False)

    season_only = (time_period == "season")
    n_games = 100
    if time_period.startswith("last_"):
        try:
            n_games = int(time_period.replace("last_", ""))
        except ValueError:
            pass

    h2h_games = _get_h2h_games(
        home_team, away_team, reference_date, n_games,
        side_filter=has_side, season_only=season_only, context=context)

    if stat_name == "h2h_games_count":
        return float(len(h2h_games))

    if stat_name == "h2h_win_pct":
        return _compute_h2h_win_pct(
            h2h_games, home_team, calc_weight, perspective)

    if stat_name == "margin_h2h":
        return _compute_margin_h2h(
            h2h_games, home_team, calc_weight, perspective)

    return None


def _get_h2h_games(home_team, away_team, reference_date, n_games,
                   side_filter, season_only, context):
    """Get head-to-head games between two teams before reference_date."""
    game_date = reference_date or ""
    season = context.get("season", "")
    games_home = context.get("games_home")
    exclude = context.get("exclude_game_types", ["preseason", "allstar"])

    if games_home is not None:
        h2h = []
        seasons_to_search = [season] if season_only else list(games_home.keys())

        for s_key in seasons_to_search:
            if s_key not in games_home:
                continue
            for date_str, date_games in games_home[s_key].items():
                if date_str >= game_date:
                    continue
                for g_home_team, game in date_games.items():
                    if game.get("game_type", "regseason") in exclude:
                        continue
                    g_away_team = game.get("awayTeam", {}).get("name", "")
                    is_h2h = (
                        (g_home_team == home_team and g_away_team == away_team) or
                        (g_home_team == away_team and g_away_team == home_team)
                    )
                    if not is_h2h:
                        continue
                    if side_filter:
                        if g_home_team != home_team or g_away_team != away_team:
                            continue
                    h2h.append(game)

        h2h.sort(key=lambda g: g.get("date", ""), reverse=True)
        h2h = h2h[:n_games]
        h2h.reverse()
        return h2h

    # Fallback to DB
    db = context.get("db")
    league = context.get("league")
    if db is None or league is None:
        return []

    try:
        coll = league.collections.get("games", "stats_nba")
        if side_filter:
            query = {
                "homeTeam.name": home_team,
                "awayTeam.name": away_team,
                "date": {"$lt": game_date},
                "game_type": {"$nin": exclude},
            }
        else:
            query = {
                "$or": [
                    {"homeTeam.name": home_team, "awayTeam.name": away_team},
                    {"homeTeam.name": away_team, "awayTeam.name": home_team},
                ],
                "date": {"$lt": game_date},
                "game_type": {"$nin": exclude},
            }
        if season_only:
            query["season"] = season

        h2h = list(db[coll].find(query).sort("date", -1).limit(n_games))
        h2h.reverse()
        return h2h
    except Exception:
        return []


def _compute_h2h_win_pct(h2h_games, home_team, calc_weight, perspective):
    """H2H win percentage for the home team."""
    if not h2h_games:
        return 0.5

    home_wins = 0
    for game in h2h_games:
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        g_home = game.get("homeTeam", {}).get("name", "")

        if g_home == home_team:
            if home_pts > away_pts:
                home_wins += 1
        else:
            if away_pts > home_pts:
                home_wins += 1

    n = len(h2h_games)
    if calc_weight == "beta":
        alpha, beta = 2.0, 2.0
        home_pct = (home_wins + alpha) / (n + alpha + beta)
    else:
        home_pct = float(home_wins) / n

    away_pct = 1.0 - home_pct
    return _apply_perspective(home_pct, away_pct, perspective)


def _compute_margin_h2h(h2h_games, home_team, calc_weight, perspective):
    """Point margin in H2H games."""
    if not h2h_games:
        return 0.0

    total_margin = 0.0
    for game in h2h_games:
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        g_home = game.get("homeTeam", {}).get("name", "")

        if g_home == home_team:
            total_margin += home_pts - away_pts
        else:
            total_margin += away_pts - home_pts

    n = len(h2h_games)
    avg_margin = total_margin / n

    if calc_weight == "eb":
        k = 5.0
        home_margin = (n / (n + k)) * avg_margin
    elif calc_weight == "logw":
        home_margin = avg_margin * log1p(n)
    else:
        home_margin = avg_margin

    if perspective == "home":
        return home_margin
    elif perspective == "away":
        return -home_margin
    else:  # diff
        return 2.0 * home_margin


# ============================================================================
# Handler: compute_conference
# ============================================================================

def compute_conference(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute conference-specific features.

    Handles: conf_wins, same_conf, conf_margin, conf_gp
    """
    if stat_name == "same_conf":
        home_conf = _get_team_conference(home_team, context)
        away_conf = _get_team_conference(away_team, context)
        return 1.0 if (home_conf and away_conf and home_conf == away_conf) else 0.0

    engine = context.get("engine")
    reference_date = context.get("reference_date")

    # All conf stats use season games
    h_games = _window_and_filter(
        home_games, "season", reference_date, home_team, False, "home", engine)
    a_games = _window_and_filter(
        away_games, "season", reference_date, away_team, False, "away", engine)

    if stat_name == "conf_wins":
        home_val = _get_conf_win_pct(home_team, h_games, context)
        away_val = _get_conf_win_pct(away_team, a_games, context)
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "conf_margin":
        home_val = _get_conf_avg_margin(home_team, h_games, context)
        away_val = _get_conf_avg_margin(away_team, a_games, context)
        return _apply_perspective(home_val, away_val, perspective)

    if stat_name == "conf_gp":
        home_val = float(_get_conf_games_count(home_team, h_games, context))
        away_val = float(_get_conf_games_count(away_team, a_games, context))
        return _apply_perspective(home_val, away_val, perspective)

    return None


def _get_team_conference(team, context):
    """Look up a team's conference. Uses cache from context."""
    conf_cache = context.get("conference_cache", {})
    if team in conf_cache:
        return conf_cache[team]

    db = context.get("db")
    league = context.get("league")
    if db is None or league is None:
        return None

    teams_coll = league.collections.get("teams", "cbb_teams")
    doc = db[teams_coll].find_one(
        {"$or": [{"abbreviation": team}, {"team_id": team}]},
        {"conference": 1},
    )
    conf = doc.get("conference") if doc else None
    conf_cache[team] = conf
    return conf


def _get_conference_teams(conference, context):
    """Get all team names/ids in a conference."""
    conf_teams_cache = context.get("conf_teams_cache", {})
    if conference in conf_teams_cache:
        return conf_teams_cache[conference]

    db = context.get("db")
    league = context.get("league")
    if db is None or league is None:
        return set()

    teams_coll = league.collections.get("teams", "cbb_teams")
    docs = db[teams_coll].find({"conference": conference}, {"abbreviation": 1, "team_id": 1})
    names = set()
    for d in docs:
        if d.get("abbreviation"):
            names.add(d["abbreviation"])
        if d.get("team_id"):
            names.add(str(d["team_id"]))
    conf_teams_cache[conference] = names
    return names


def _is_conf_game(team, game, conf_teams):
    """Check if a game is a conference game for the given team."""
    home_name = game.get("homeTeam", {}).get("name", "")
    away_name = game.get("awayTeam", {}).get("name", "")
    home_id = str(game.get("homeTeam", {}).get("team_id", ""))
    away_id = str(game.get("awayTeam", {}).get("team_id", ""))

    is_home = (home_name == team or home_id == team)
    opp_name = away_name if is_home else home_name
    opp_id = away_id if is_home else home_id

    return opp_name in conf_teams or opp_id in conf_teams


def _get_conf_win_pct(team, games, context):
    """Conference win percentage."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0.5
    conf_teams = _get_conference_teams(conf, context)

    wins, total = 0, 0
    for game in games:
        if not _is_conf_game(team, game, conf_teams):
            continue
        total += 1
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        is_home = game.get("homeTeam", {}).get("name") == team
        if (is_home and home_pts > away_pts) or (not is_home and away_pts > home_pts):
            wins += 1

    return wins / total if total > 0 else 0.5


def _get_conf_avg_margin(team, games, context):
    """Average point margin in conference games."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0.0
    conf_teams = _get_conference_teams(conf, context)

    total_margin, total = 0.0, 0
    for game in games:
        if not _is_conf_game(team, game, conf_teams):
            continue
        total += 1
        home_pts = game.get("homeTeam", {}).get("points", 0) or 0
        away_pts = game.get("awayTeam", {}).get("points", 0) or 0
        is_home = game.get("homeTeam", {}).get("name") == team
        total_margin += (home_pts - away_pts) if is_home else (away_pts - home_pts)

    return total_margin / total if total > 0 else 0.0


def _get_conf_games_count(team, games, context):
    """Count conference games played."""
    conf = _get_team_conference(team, context)
    if not conf:
        return 0
    conf_teams = _get_conference_teams(conf, context)
    return sum(1 for g in games if _is_conf_game(team, g, conf_teams))


# ============================================================================
# Handler: compute_elo
# ============================================================================

def compute_elo(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute Elo rating features. Requires context: elo_cache, season."""
    elo_cache = context.get("elo_cache")
    db = context.get("db")
    reference_date = context.get("reference_date", "")
    season = context.get("season", "")

    if elo_cache is None:
        if db is None:
            return 0.0
        try:
            from bball.stats.elo_cache import EloCache
            elo_cache = EloCache(db)
        except Exception:
            return 0.0

    try:
        home_elo = elo_cache.get_elo_for_game_with_fallback(
            home_team, reference_date, season)
        away_elo = elo_cache.get_elo_for_game_with_fallback(
            away_team, reference_date, season)
    except Exception:
        return 0.0

    return _apply_perspective(home_elo, away_elo, perspective)


# ============================================================================
# Handler: compute_vegas
# ============================================================================

def compute_vegas(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute Vegas betting line features.

    Handles: vegas_ML, vegas_spread, vegas_ou, vegas_implied_prob, vegas_edge
    """
    game = context.get("game_doc")

    # Try to find the game doc if not in context
    if game is None:
        game = _find_game_doc(home_team, away_team, context)
    if not game:
        return 0.0

    try:
        vegas = game.get("vegas")
        pregame = game.get("pregame_lines")

        if stat_name == "vegas_ML":
            return _get_vegas_ml(vegas, pregame, perspective)
        if stat_name == "vegas_spread":
            return _get_vegas_spread(vegas, pregame, perspective)
        if stat_name == "vegas_ou":
            return _get_vegas_ou(vegas, pregame)
        if stat_name == "vegas_implied_prob":
            return _get_vegas_implied_prob(vegas, pregame, perspective)
        if stat_name == "vegas_edge":
            return _get_vegas_edge(vegas, pregame, perspective)
    except Exception:
        return 0.0

    return 0.0


def _find_game_doc(home_team, away_team, context):
    """Find the game document from preloaded data or DB."""
    reference_date = context.get("reference_date", "")
    season = context.get("season", "")
    games_home = context.get("games_home")
    league = context.get("league")

    if games_home is not None and season in games_home:
        date_games = games_home[season].get(reference_date, {})
        return date_games.get(home_team)

    db = context.get("db")
    if db is not None and league is not None:
        coll = league.collections.get("games", "stats_nba")
        return db[coll].find_one(
            {"date": reference_date, "homeTeam.name": home_team,
             "awayTeam.name": away_team},
            {"vegas": 1, "pregame_lines": 1},
        )

    return None


def _ml_to_prob(ml):
    """Convert American moneyline to implied probability."""
    if ml is None:
        return None
    ml = float(ml)
    if ml < 0:
        return (-ml) / (-ml + 100)
    return 100 / (ml + 100)


def _get_ml(vegas, pregame, side):
    """Get moneyline for home or away."""
    val = None
    if vegas:
        val = vegas.get(f"{side}_ML")
    if val is None and pregame:
        val = pregame.get(f"{side}_ml")
    return val


def _get_vegas_ml(vegas, pregame, perspective):
    home_ml = _get_ml(vegas, pregame, "home")
    away_ml = _get_ml(vegas, pregame, "away")
    if perspective == "home":
        return float(home_ml) if home_ml is not None else 0.0
    elif perspective == "away":
        return float(away_ml) if away_ml is not None else 0.0
    else:
        if home_ml is not None and away_ml is not None:
            return float(home_ml) - float(away_ml)
        return 0.0


def _get_vegas_spread(vegas, pregame, perspective):
    if vegas:
        if perspective == "home":
            s = vegas.get("home_spread")
            if s is not None:
                return float(s)
        elif perspective == "away":
            s = vegas.get("away_spread")
            if s is not None:
                return float(s)
    if pregame:
        s = pregame.get("spread")
        if s is not None:
            if perspective == "home":
                return float(s)
            elif perspective == "away":
                return -float(s)
    return 0.0


def _get_vegas_ou(vegas, pregame):
    ou = None
    if vegas:
        ou = vegas.get("OU")
    if ou is None and pregame:
        ou = pregame.get("over_under")
    return float(ou) if ou is not None else 0.0


def _get_vegas_implied_prob(vegas, pregame, perspective):
    home_ml = _get_ml(vegas, pregame, "home")
    away_ml = _get_ml(vegas, pregame, "away")
    home_prob = _ml_to_prob(home_ml)
    away_prob = _ml_to_prob(away_ml)
    if perspective == "home":
        return home_prob if home_prob is not None else 0.0
    elif perspective == "away":
        return away_prob if away_prob is not None else 0.0
    else:
        if home_prob is not None and away_prob is not None:
            return home_prob - away_prob
        return 0.0


def _get_vegas_edge(vegas, pregame, perspective):
    import math
    K = 6.5

    home_ml = _get_ml(vegas, pregame, "home")
    away_ml = _get_ml(vegas, pregame, "away")

    home_spread = None
    if vegas:
        home_spread = vegas.get("home_spread")
    if home_spread is None and pregame:
        home_spread = pregame.get("spread")

    def spread_to_prob(s):
        if s is None:
            return None
        return 1 / (1 + math.exp(-float(s) / K))

    home_implied = _ml_to_prob(home_ml)
    away_implied = _ml_to_prob(away_ml)
    home_sp = spread_to_prob(home_spread)
    away_sp = spread_to_prob(-float(home_spread)) if home_spread is not None else None

    if perspective == "home":
        if home_implied is not None and home_sp is not None:
            return home_implied - home_sp
        return 0.0
    elif perspective == "away":
        if away_implied is not None and away_sp is not None:
            return away_implied - away_sp
        return 0.0
    else:
        h = (home_implied - home_sp) if (home_implied is not None and home_sp is not None) else None
        a = (away_implied - away_sp) if (away_implied is not None and away_sp is not None) else None
        if h is not None and a is not None:
            return h - a
        return 0.0


# ============================================================================
# Handler: compute_context
# ============================================================================

def compute_context(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Compute game context features.

    Handles: postseason, close_win_pct, home_court
    """
    if stat_name == "home_court":
        return 1.0 if perspective == "home" else 0.0

    if stat_name == "postseason":
        game = context.get("game_doc") or _find_game_doc(home_team, away_team, context)
        if game:
            game_type = game.get("game_type", "")
            return 1.0 if game_type in ("postseason", "playoffs") else 0.0

        # Fall back to DB lookup
        reference_date = context.get("reference_date", "")
        db = context.get("db")
        league = context.get("league")
        if db is not None and league is not None:
            coll = league.collections.get("games", "stats_nba")
            doc = db[coll].find_one(
                {"date": reference_date, "homeTeam.name": home_team,
                 "awayTeam.name": away_team},
                {"game_type": 1},
            )
            gt = doc.get("game_type") if doc else None
            return 1.0 if gt in ("postseason", "playoffs") else 0.0
        return 0.0

    if stat_name == "close_win_pct":
        return _compute_close_win_pct(
            time_period, perspective, home_team, away_team,
            home_games, away_games, context)

    return None


def _compute_close_win_pct(time_period, perspective, home_team, away_team,
                           home_games, away_games, context):
    """Win percentage in close games (decided by <=5 points)."""
    engine = context.get("engine")
    reference_date = context.get("reference_date")

    def _pct_for_team(team, games):
        # Get all season games
        all_games = _window_and_filter(
            games, "season", reference_date, team, False, "home", engine)

        # Filter to close games
        close = []
        for g in all_games:
            h_pts = g.get("homeTeam", {}).get("points", 0) or 0
            a_pts = g.get("awayTeam", {}).get("points", 0) or 0
            if abs(h_pts - a_pts) <= 5:
                close.append(g)

        if time_period == "games_close5":
            close = close[-5:] if len(close) >= 5 else close

        if not close:
            return 0.5

        wins = 0
        for g in close:
            h_pts = g.get("homeTeam", {}).get("points", 0) or 0
            a_pts = g.get("awayTeam", {}).get("points", 0) or 0
            g_home = g.get("homeTeam", {}).get("name", "")
            if g_home == team:
                if h_pts > a_pts:
                    wins += 1
            else:
                if a_pts > h_pts:
                    wins += 1
        return float(wins) / len(close)

    home_val = _pct_for_team(home_team, home_games)
    away_val = _pct_for_team(away_team, away_games)
    return _apply_perspective(home_val, away_val, perspective)


# ============================================================================
# Handler: compute_player
# ============================================================================

def compute_player(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Player PER features. Delegates to PERCalculator from context.

    Returns 0.0 as placeholder — real computation happens in
    SharedFeatureGenerator._calculate_per_features() which calls
    PERCalculator.get_game_per_features() in bulk.
    """
    return 0.0


# ============================================================================
# Handler: compute_injury
# ============================================================================

def compute_injury(
    stat_name, time_period, calc_weight, perspective,
    home_team, away_team, home_games, away_games,
    **context,
) -> Optional[float]:
    """Injury impact features. Delegates to injury computation from context.

    Returns 0.0 as placeholder — real computation happens in
    SharedFeatureGenerator._calculate_injury_features() which calls
    StatHandlerV2.get_injury_features() in bulk.
    """
    return 0.0


# ============================================================================
# Handler registry
# ============================================================================

CUSTOM_HANDLERS = {
    "compute_basic_rate": compute_basic_rate,
    "compute_net": compute_net,
    "compute_derived": compute_derived,
    "compute_schedule": compute_schedule,
    "compute_h2h": compute_h2h,
    "compute_conference": compute_conference,
    "compute_elo": compute_elo,
    "compute_vegas": compute_vegas,
    "compute_context": compute_context,
    "compute_player": compute_player,
    "compute_injury": compute_injury,
}
