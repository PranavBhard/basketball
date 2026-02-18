"""
Feature Groups - Semantic Organization of Features

This module provides FeatureGroups, which organizes features by their
predictive purpose following a layered hierarchy from outcome signals to context.

Extracted from registry.py to keep module sizes manageable.
"""

from typing import Dict, List, Optional


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
        # Import here to avoid circular imports
        from bball.features.registry import FeatureRegistry

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
