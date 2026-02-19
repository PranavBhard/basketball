"""
Injury Feature Calculator — standalone injury feature computation.

Computes injury-related features for basketball games:
- Per-player injury impact (PER-weighted, minutes-weighted)
- Team rotation impact (minutes lost, severity)
- Season-to-date injury severity
- Normalized/deconfounded injury metrics
- Star-based injury features

Usage:
    calculator = InjuryFeatureCalculator(db=db, league=league)
    # For prediction: inject preloaded data
    calculator.set_preloaded_data(games_home, games_away)
    # Compute features
    features = calculator.get_injury_features(
        'BOS', 'LAL', '2024-2025', 2025, 1, 15,
        game_doc=game_doc, per_calculator=per_calc
    )
"""

import bisect
import math
from collections import defaultdict
from datetime import date, datetime

from bball.data import (
    GamesRepository, PlayerStatsRepository, PlayersRepository,
    RostersRepository
)
from bball.league_config import LeagueConfig, load_league_config


class InjuryFeatureCalculator:
    """Standalone calculator for injury-impact features."""

    def __init__(self, db=None, league=None, batch_training_mode=False):
        if league is None:
            league = load_league_config("nba")
        self.league = league
        self.db = db
        self._batch_training_mode = batch_training_mode

        # Repositories (only if db provided)
        if db is not None:
            self._games_repo = GamesRepository(db, league=self.league)
            self._players_repo = PlayerStatsRepository(db, league=self.league)
            self._players_dir_repo = PlayersRepository(db, league=self.league)
            self._rosters_repo = RostersRepository(db, league=self.league)
        else:
            self._games_repo = None
            self._players_repo = None
            self._players_dir_repo = None
            self._rosters_repo = None

        # Exclude game types from league config
        self._exclude_game_types = (
            league.exclude_game_types if league else ["preseason", "allstar"]
        )

        # Preloaded game data
        self.games_home = None
        self.games_away = None

        # Team index for O(log N) bisect lookups
        self._team_games_index = {}   # {season: {team: [(date_str, game_doc), ...]}}
        self._team_dates_index = {}   # {season: {team: [date_str, ...]}}

        # Injury caches
        self._injury_player_stats_cache = {}   # (team, season, date_str) -> {pid: stats}
        self._injury_max_mpg_cache = {}        # (team, season, date_str) -> float
        self._injury_rotation_mpg_cache = {}   # (team, season, date_str) -> float
        self._injury_preloaded_players = {}    # (team, season) -> [player_records]
        self._injury_cache_loaded = False
        self._team_weighted_per_mass_cache = {}
        self._season_injury_severity_cache = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_preloaded_data(self, games_home, games_away):
        """Inject preloaded game data and build team index."""
        self.games_home = games_home
        self.games_away = games_away
        self._build_team_index()

    def _build_team_index(self):
        """Build sorted team game index for O(log N) bisect lookups."""
        if self.games_home is None:
            return

        team_index = {}
        dates_index = {}

        for season, date_dict in self.games_home.items():
            season_teams = team_index.setdefault(season, defaultdict(list))
            for date_str, teams_dict in date_dict.items():
                for home_team, game in teams_dict.items():
                    season_teams[home_team].append((date_str, game))
                    away_team = game.get("awayTeam", {}).get("name")
                    if away_team:
                        season_teams[away_team].append((date_str, game))

        for season, teams in team_index.items():
            dates_index[season] = {}
            for team, pairs in teams.items():
                pairs.sort(key=lambda p: p[0])
                dates_index[season][team] = [p[0] for p in pairs]

        self._team_games_index = team_index
        self._team_dates_index = dates_index

    def _get_team_games_in_range(
        self, team, season, begin_date_str=None, end_date_str=None,
        exclude_game_types=None
    ):
        """O(log G) bisect lookup of a team's games in [begin, end)."""
        dates = self._team_dates_index.get(season, {}).get(team)
        if not dates:
            return []

        pairs = self._team_games_index[season][team]
        lo = bisect.bisect_left(dates, begin_date_str) if begin_date_str else 0
        hi = bisect.bisect_left(dates, end_date_str) if end_date_str else len(dates)

        if exclude_game_types is None:
            exclude_game_types = self._exclude_game_types

        if not exclude_game_types:
            return [pair[1] for pair in pairs[lo:hi]]

        exclude_set = set(exclude_game_types)
        return [
            game for _, game in pairs[lo:hi]
            if game.get("game_type", "regseason") not in exclude_set
        ]

    # ------------------------------------------------------------------
    # Usage filter helpers
    # ------------------------------------------------------------------

    def _get_usage_thresholds(self):
        """Get player filter thresholds from league config, with defaults.

        Supports early/late season phases and OR-rule eligibility.
        """
        pf = {}
        if self.league:
            pf = self.league.raw.get("player_filters", {})
        gp_ratio = pf.get("gp_ratio", 0.15)
        return {
            "mpg_thresh": pf.get("mpg_thresh", 10),
            "rotation_size": pf.get("rotation_size", 10),
            "gp_ratio_early": pf.get("gp_ratio_early", gp_ratio),
            "gp_ratio_late": pf.get("gp_ratio_late", gp_ratio),
            "gp_floor_early": pf.get("gp_floor_early", 3),
            "gp_floor_late": pf.get("gp_floor_late", 10),
            "gp_floor_phase_games": pf.get("gp_floor_phase_games", 20),
            "min_team_games_for_filter": pf.get("min_team_games_for_filter", 5),
            "use_or_rule": pf.get("use_or_rule", True),
        }

    def _compute_gp_thresh(self, team_gp, thresholds):
        """Phase-aware GP threshold: lenient early, strict late."""
        phase_games = thresholds["gp_floor_phase_games"]
        if team_gp < phase_games:
            gp_floor = thresholds["gp_floor_early"]
            gp_ratio = thresholds["gp_ratio_early"]
        else:
            gp_floor = thresholds["gp_floor_late"]
            gp_ratio = thresholds["gp_ratio_late"]
        return min(team_gp, max(gp_floor, math.ceil(gp_ratio * team_gp)))

    def _passes_usage_filter(self, mpg, games_played, team_gp, thresholds):
        """Check if a player passes the USAGE filter (for PER/value features).

        Uses OR rule when configured: eligible if mpg >= threshold OR gp >= threshold.
        Skips GP filtering entirely if team hasn't played enough games.
        """
        passes_mpg = mpg >= thresholds["mpg_thresh"]

        # Too early in season for GP filter to be meaningful
        if team_gp < thresholds["min_team_games_for_filter"]:
            return passes_mpg

        gp_thresh = self._compute_gp_thresh(team_gp, thresholds)
        passes_gp = games_played >= gp_thresh

        if thresholds["use_or_rule"]:
            return passes_mpg or passes_gp
        return passes_mpg and passes_gp

    def _get_team_games_played(self, team, season, before_date):
        """Count distinct game dates for a team/season before a date."""
        cache_key = (team, season, before_date)

        # Use player stats cache — warmed cache has all players for this team
        if cache_key not in self._injury_player_stats_cache:
            self._warm_player_stats_cache(team, season, before_date)

        all_player_stats = self._injury_player_stats_cache.get(cache_key, {})
        if not all_player_stats:
            return 0

        # Approximate team GP as the max games_played among all team players
        return max(
            (s.get("games_played", 0) for s in all_player_stats.values()),
            default=0,
        )

    # ------------------------------------------------------------------
    # Batch preloading (training)
    # ------------------------------------------------------------------

    def preload_injury_cache(self, games=None):
        """Preload player stats for injury features (batch queries per season)."""
        if self.db is None:
            return

        if games is None:
            games = self._games_repo.find(
                {
                    "homeTeam.points": {"$exists": True, "$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={
                    "homeTeam.name": 1, "awayTeam.name": 1,
                    "season": 1, "date": 1,
                },
            )

        teams_by_season = defaultdict(set)
        for game in games:
            home = game.get("homeTeam", {}).get("name")
            away = game.get("awayTeam", {}).get("name")
            season = game.get("season")
            if home and season:
                teams_by_season[season].add(home)
            if away and season:
                teams_by_season[season].add(away)

        total_team_seasons = sum(len(t) for t in teams_by_season.values())
        print(
            f"Preloading injury cache for {total_team_seasons} team-season combinations "
            f"across {len(teams_by_season)} seasons..."
        )

        total_records = 0
        for season, teams in teams_by_season.items():
            all_records = list(self._players_repo.find(
                {
                    "season": season,
                    "team": {"$in": list(teams)},
                    "stats.min": {"$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={
                    "player_id": 1, "team": 1,
                    "season": 1, "date": 1, "stats.min": 1,
                },
            ))

            records_by_team = defaultdict(list)
            for record in all_records:
                team = record.get("team")
                if team:
                    records_by_team[team].append(record)

            for team, records in records_by_team.items():
                sorted_records = sorted(records, key=lambda r: r.get("date", ""))
                self._injury_preloaded_players[(team, season)] = sorted_records
                total_records += len(sorted_records)

        print(
            f"  Preloaded {total_records:,} player records for {total_team_seasons} "
            f"team-seasons ({len(teams_by_season)} batch queries)"
        )
        self._injury_cache_loaded = True

    def precompute_season_severity(self, games=None):
        """Precompute season injury severity for all (team, season, date) combos.

        O(G) incremental computation instead of O(G^2) on-demand.
        Must be called AFTER preload_injury_cache().
        """
        if not self._injury_cache_loaded:
            print(
                "Warning: precompute_season_severity called before injury cache loaded. "
                "Season severity cache will be empty."
            )
            return {}

        if games is None:
            if self.games_home is None:
                print("Warning: No games available for season severity precomputation.")
                return {}
            games = []
            for season_data in self.games_home.values():
                for date_data in season_data.values():
                    for game in date_data.values():
                        games.append(game)

        # Group games by (team, season)
        team_season_games = defaultdict(list)
        for game in games:
            home_team = game.get("homeTeam", {}).get("name")
            away_team = game.get("awayTeam", {}).get("name")
            season = game.get("season")
            game_date = game.get("date")

            if game.get("homeWon") is None:
                continue

            if home_team and season and game_date:
                team_season_games[(home_team, season)].append({
                    "date": game_date,
                    "is_home": True,
                    "injured_players": game.get("homeTeam", {}).get("injured_players", []),
                })
            if away_team and season and game_date:
                team_season_games[(away_team, season)].append({
                    "date": game_date,
                    "is_home": False,
                    "injured_players": game.get("awayTeam", {}).get("injured_players", []),
                })

        print(f"Precomputing season severity for {len(team_season_games)} team-season combinations...")

        severity_cache = {}
        EPS = 1e-6
        # Severity is a bodies/depth feature — MPG-only, no GP filter
        mpg_thresh = self._get_usage_thresholds()["mpg_thresh"]

        for (team, season), team_games in team_season_games.items():
            sorted_games = sorted(team_games, key=lambda g: g["date"])
            player_cumulative = defaultdict(lambda: {"total_min": 0.0, "games": 0})

            player_records = self._injury_preloaded_players.get((team, season), [])
            records_by_date = defaultdict(list)
            for record in player_records:
                records_by_date[record.get("date", "")].append(record)

            running_min_lost = 0.0
            running_rotation_mpg = 0.0

            for game_info in sorted_games:
                game_date = game_info["date"]
                injured_ids = (
                    [str(pid) for pid in game_info["injured_players"]]
                    if game_info["injured_players"] else []
                )

                # Store severity BEFORE this game
                if running_rotation_mpg > 0:
                    severity = running_min_lost / (running_rotation_mpg + EPS)
                else:
                    severity = 0.0
                severity_cache[(team, season, game_date)] = severity

                # Rotation MPG at this point (MPG-only filter)
                game_rotation_mpg = 0.0
                for player_id, stats in player_cumulative.items():
                    if stats["games"] > 0:
                        mpg = stats["total_min"] / stats["games"]
                        if mpg >= mpg_thresh:
                            game_rotation_mpg += mpg

                # Min lost for this game (MPG-only filter)
                game_min_lost = 0.0
                if injured_ids:
                    for pid in injured_ids:
                        if pid in player_cumulative:
                            stats = player_cumulative[pid]
                            if stats["games"] > 0:
                                mpg = stats["total_min"] / stats["games"]
                                if mpg >= mpg_thresh:
                                    game_min_lost += mpg

                running_rotation_mpg += game_rotation_mpg
                running_min_lost += game_min_lost

                # Update cumulative player stats
                for record in records_by_date.get(game_date, []):
                    pid = str(record.get("player_id", ""))
                    minutes = record.get("stats", {}).get("min", 0.0)
                    if minutes > 0:
                        player_cumulative[pid]["total_min"] += minutes
                        player_cumulative[pid]["games"] += 1

        print(f"  Precomputed {len(severity_cache)} season severity values")
        self._season_injury_severity_cache.update(severity_cache)
        return severity_cache

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_injury_features(
        self,
        HOME, AWAY, season,
        year, month, day,
        game_doc=None,
        per_calculator=None,
        recency_decay_k=15.0,
        precomputed_season_severity=None,
    ):
        """Calculate injury impact features for a game.

        Args:
            HOME: Home team name
            AWAY: Away team name
            season: Season string
            year, month, day: Game date
            game_doc: Optional game document (if None, will query)
            per_calculator: Optional PERCalculator instance
            recency_decay_k: Decay constant for recency weighting
            precomputed_season_severity: Optional precomputed severity cache

        Returns:
            Dict with injury features (home/away/diff)
        """
        if self.db is None:
            return {}

        if recency_decay_k is None:
            recency_decay_k = 15.0

        game_date = f"{year}-{month:02d}-{day:02d}"
        game_date_obj = date(year, month, day)

        # Get game document if not provided
        if game_doc is None and self._games_repo:
            game_doc = self._games_repo.find_one({
                "homeTeam.name": HOME,
                "awayTeam.name": AWAY,
                "season": season,
                "date": game_date,
            })

        if not game_doc:
            return {}

        # --- Determine injured player IDs ---
        home_injured_ids = game_doc.get("homeTeam", {}).get("injured_players", [])
        away_injured_ids = game_doc.get("awayTeam", {}).get("injured_players", [])
        home_injured_ids = [str(pid) for pid in home_injured_ids] if home_injured_ids else []
        away_injured_ids = [str(pid) for pid in away_injured_ids] if away_injured_ids else []

        # Fallback: roster lookup for future games (skip in batch training mode)
        skip_roster_lookup = self._batch_training_mode
        if not home_injured_ids and not away_injured_ids and not skip_roster_lookup:
            home_player_ids = game_doc.get("homeTeam", {}).get("players", [])
            away_player_ids = game_doc.get("awayTeam", {}).get("players", [])
            home_player_ids = [str(pid) for pid in home_player_ids] if home_player_ids else []
            away_player_ids = [str(pid) for pid in away_player_ids] if away_player_ids else []

            if self._rosters_repo is not None:
                try:
                    home_roster_doc = self._rosters_repo.find_roster(HOME, season)
                    if home_roster_doc:
                        home_roster = home_roster_doc.get("roster", [])
                        home_roster_map = {
                            str(entry.get("player_id")): entry.get("injured", False)
                            for entry in home_roster
                            if not entry.get("disabled", False)
                        }
                        if home_player_ids:
                            home_injured_ids = [pid for pid in home_player_ids if home_roster_map.get(pid, False)]
                        else:
                            home_injured_ids = [pid for pid, is_inj in home_roster_map.items() if is_inj]

                    away_roster_doc = self._rosters_repo.find_roster(AWAY, season)
                    if away_roster_doc:
                        away_roster = away_roster_doc.get("roster", [])
                        away_roster_map = {
                            str(entry.get("player_id")): entry.get("injured", False)
                            for entry in away_roster
                            if not entry.get("disabled", False)
                        }
                        if away_player_ids:
                            away_injured_ids = [pid for pid in away_player_ids if away_roster_map.get(pid, False)]
                        else:
                            away_injured_ids = [pid for pid, is_inj in away_roster_map.items() if is_inj]
                except Exception:
                    pass

        # --- Player names (skip in batch mode) ---
        home_injured_names = []
        away_injured_names = []
        skip_name_lookup = self._batch_training_mode
        if not skip_name_lookup and self._players_dir_repo is not None:
            try:
                if home_injured_ids:
                    docs = self._players_dir_repo.find(
                        {"player_id": {"$in": home_injured_ids}},
                        projection={"player_id": 1, "player_name": 1},
                    )
                    home_injured_names = [d.get("player_name", "Unknown") for d in docs]
                if away_injured_ids:
                    docs = self._players_dir_repo.find(
                        {"player_id": {"$in": away_injured_ids}},
                        projection={"player_id": 1, "player_name": 1},
                    )
                    away_injured_names = [d.get("player_name", "Unknown") for d in docs]
            except Exception:
                pass

        # --- Per-team features ---
        home_features = self._calculate_team_injury_features(
            HOME, season, game_date, game_date_obj, home_injured_ids,
            per_calculator, recency_decay_k,
        )
        away_features = self._calculate_team_injury_features(
            AWAY, season, game_date, game_date_obj, away_injured_ids,
            per_calculator, recency_decay_k,
        )

        # --- Build result dict ---
        features = {}

        home_inj_per_value = home_features.get("injPerValue", 0.0)
        away_inj_per_value = away_features.get("injPerValue", 0.0)
        home_inj_top1_per = home_features.get("injTop1Per", 0.0)
        away_inj_top1_per = away_features.get("injTop1Per", 0.0)
        home_inj_top3_sum = home_features.get("injTop3PerSum", 0.0)
        away_inj_top3_sum = away_features.get("injTop3PerSum", 0.0)
        home_inj_min_lost = home_features.get("injMinLost", 0.0)
        away_inj_min_lost = away_features.get("injMinLost", 0.0)
        home_injury_severity = home_features.get("injurySeverity", 0.0)
        away_injury_severity = away_features.get("injurySeverity", 0.0)
        home_inj_rotation = home_features.get("injRotation", 0.0)
        away_inj_rotation = away_features.get("injRotation", 0.0)
        home_inj_rotation_share = home_features.get("injRotationShare", 0.0)
        away_inj_rotation_share = away_features.get("injRotationShare", 0.0)

        # inj_per features
        features["inj_per|none|weighted_MIN|home"] = home_inj_per_value
        features["inj_per|none|weighted_MIN|away"] = away_inj_per_value
        features["inj_per|none|weighted_MIN|diff"] = home_inj_per_value - away_inj_per_value

        features["inj_per|none|top1_avg|home"] = home_inj_top1_per
        features["inj_per|none|top1_avg|away"] = away_inj_top1_per
        features["inj_per|none|top1_avg|diff"] = home_inj_top1_per - away_inj_top1_per

        features["inj_per|none|top3_sum|home"] = home_inj_top3_sum
        features["inj_per|none|top3_sum|away"] = away_inj_top3_sum
        features["inj_per|none|top3_sum|diff"] = home_inj_top3_sum - away_inj_top3_sum

        # inj_min_lost features
        features["inj_min_lost|none|raw|home"] = home_inj_min_lost
        features["inj_min_lost|none|raw|away"] = away_inj_min_lost
        features["inj_min_lost|none|raw|diff"] = home_inj_min_lost - away_inj_min_lost

        # inj_severity (point-in-time)
        features["inj_severity|none|raw|home"] = home_injury_severity
        features["inj_severity|none|raw|away"] = away_injury_severity
        features["inj_severity|none|raw|diff"] = home_injury_severity - away_injury_severity

        # inj_severity|season — season-to-date weighted average
        if precomputed_season_severity is not None:
            home_season_severity = precomputed_season_severity.get((HOME, season, game_date), 0.0)
            away_season_severity = precomputed_season_severity.get((AWAY, season, game_date), 0.0)
        else:
            home_season_severity = self._get_season_injury_severity(HOME, season, game_date)
            away_season_severity = self._get_season_injury_severity(AWAY, season, game_date)
        features["inj_severity|season|raw|home"] = home_season_severity
        features["inj_severity|season|raw|away"] = away_season_severity
        features["inj_severity|season|raw|diff"] = home_season_severity - away_season_severity

        # Player lists for UI display
        features["_player_lists"] = {
            "inj_per|none|weighted_MIN|home": home_features.get("injPerValue_players", []),
            "inj_per|none|weighted_MIN|away": away_features.get("injPerValue_players", []),
            "inj_per|none|top1_avg|home": home_features.get("injTop1Per_players", []),
            "inj_per|none|top1_avg|away": away_features.get("injTop1Per_players", []),
            "inj_per|none|top3_sum|home": home_features.get("injTop3PerSum_players", []),
            "inj_per|none|top3_sum|away": away_features.get("injTop3PerSum_players", []),
            "inj_min_lost|none|raw|home": home_features.get("injMinLost_players", []),
            "inj_min_lost|none|raw|away": away_features.get("injMinLost_players", []),
            "inj_rotation_per|none|raw|home": home_features.get("injRotation_players", []),
            "inj_rotation_per|none|raw|away": away_features.get("injRotation_players", []),
        }

        # inj_rotation_per features
        features["inj_rotation_per|none|raw|home"] = home_inj_rotation
        features["inj_rotation_per|none|raw|away"] = away_inj_rotation
        features["inj_rotation_per|none|raw|diff"] = home_inj_rotation - away_inj_rotation

        # inj_rotation_share features (fraction of rotation lost)
        features["inj_rotation_share|none|raw|home"] = home_inj_rotation_share
        features["inj_rotation_share|none|raw|away"] = away_inj_rotation_share
        features["inj_rotation_share|none|raw|diff"] = home_inj_rotation_share - away_inj_rotation_share

        # Injury blend feature
        blend_w_sev = 0.45
        blend_w_top1 = 0.35
        blend_w_rot = 0.20

        features["inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|home"] = (
            blend_w_sev * home_injury_severity
            + blend_w_top1 * home_inj_top1_per
            + blend_w_rot * home_inj_rotation
        )
        features["inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|away"] = (
            blend_w_sev * away_injury_severity
            + blend_w_top1 * away_inj_top1_per
            + blend_w_rot * away_inj_rotation
        )
        features["inj_impact|none|blend:severity:0.45/top1_per:0.35/rotation:0.20|diff"] = (
            blend_w_sev * (home_injury_severity - away_injury_severity)
            + blend_w_top1 * (home_inj_top1_per - away_inj_top1_per)
            + blend_w_rot * (home_inj_rotation - away_inj_rotation)
        )

        # =====================================================================
        # NORMALIZED INJURY FEATURES (deconfounded from team quality)
        # =====================================================================
        EPS = 1e-6

        home_top3_sum = 0.0
        away_top3_sum = 0.0
        home_per_features = None
        away_per_features = None

        # USAGE filter for per_features players (same thresholds as numerator)
        thresholds = self._get_usage_thresholds()

        def _filter_per_players_usage(per_feats, team):
            """Filter per_features['players'] to USAGE players only."""
            if not per_feats:
                return []
            players = per_feats.get("players", [])
            if not players:
                return []
            team_gp = self._get_team_games_played(team, season, game_date)
            return [
                p for p in players
                if self._passes_usage_filter(
                    p.get("mpg", 0), p.get("games", 0), team_gp, thresholds,
                )
            ]

        if per_calculator:
            try:
                home_per_features = per_calculator.compute_team_per_features(HOME, season, game_date)
                away_per_features = per_calculator.compute_team_per_features(AWAY, season, game_date)
                # Compute top3_sum from USAGE-filtered players (not raw per_features)
                home_usage_players = _filter_per_players_usage(home_per_features, HOME)
                away_usage_players = _filter_per_players_usage(away_per_features, AWAY)
                if home_usage_players:
                    home_sorted = sorted(home_usage_players, key=lambda p: p.get("per", 0), reverse=True)
                    home_top3_sum = sum(p.get("per", 0) for p in home_sorted[:3])
                if away_usage_players:
                    away_sorted = sorted(away_usage_players, key=lambda p: p.get("per", 0), reverse=True)
                    away_top3_sum = sum(p.get("per", 0) for p in away_sorted[:3])
            except Exception:
                home_usage_players = []
                away_usage_players = []

        # inj_per_share — fraction of top-team PER injured
        inj_per_share_home = min(1.5, max(0.0, home_inj_top3_sum / (home_top3_sum + EPS)))
        inj_per_share_away = min(1.5, max(0.0, away_inj_top3_sum / (away_top3_sum + EPS)))
        features["inj_per_share|none|top3_sum|home"] = inj_per_share_home
        features["inj_per_share|none|top3_sum|away"] = inj_per_share_away
        features["inj_per_share|none|top3_sum|diff"] = inj_per_share_home - inj_per_share_away

        # inj_per_share|top1_avg — binary: is top PER player (from USAGE pool) injured?
        home_top1_injured = 0.0
        away_top1_injured = 0.0
        if per_calculator:
            try:
                # Use USAGE-filtered players for top1 determination
                home_usage_sorted = sorted(
                    (home_usage_players if home_per_features else []),
                    key=lambda p: p.get("per", 0), reverse=True,
                )
                away_usage_sorted = sorted(
                    (away_usage_players if away_per_features else []),
                    key=lambda p: p.get("per", 0), reverse=True,
                )
                if home_usage_sorted:
                    home_top1_id = str(home_usage_sorted[0].get("player_id", ""))
                    if home_top1_id and home_top1_id in [str(pid) for pid in home_injured_ids]:
                        home_top1_injured = 1.0
                if away_usage_sorted:
                    away_top1_id = str(away_usage_sorted[0].get("player_id", ""))
                    if away_top1_id and away_top1_id in [str(pid) for pid in away_injured_ids]:
                        away_top1_injured = 1.0
            except Exception:
                pass

        features["inj_per_share|none|top1_avg|home"] = home_top1_injured
        features["inj_per_share|none|top1_avg|away"] = away_top1_injured
        features["inj_per_share|none|top1_avg|diff"] = home_top1_injured - away_top1_injured

        # inj_per_weighted_share — normalized weighted PER lost
        home_weighted_per_mass = self._get_team_weighted_per_mass(
            HOME, season, game_date, game_date_obj, per_calculator, recency_decay_k,
        )
        away_weighted_per_mass = self._get_team_weighted_per_mass(
            AWAY, season, game_date, game_date_obj, per_calculator, recency_decay_k,
        )
        inj_wshare_home = min(1.5, max(0.0, home_inj_per_value / (home_weighted_per_mass + EPS)))
        inj_wshare_away = min(1.5, max(0.0, away_inj_per_value / (away_weighted_per_mass + EPS)))
        features["inj_per_weighted_share|none|weighted_MIN|home"] = inj_wshare_home
        features["inj_per_weighted_share|none|weighted_MIN|away"] = inj_wshare_away
        features["inj_per_weighted_share|none|weighted_MIN|diff"] = inj_wshare_home - inj_wshare_away

        # =====================================================================
        # STAR-BASED INJURY FEATURES
        # =====================================================================

        def compute_star_injury_features(team, injured_ids, per_feats):
            result = {"inj_star_share": 0.0, "inj_star_score_share": 0.0, "inj_top1_star_out": 0.0}
            if not per_feats:
                return result
            # Use USAGE-filtered players for star features
            players = _filter_per_players_usage(per_feats, team)
            if not players:
                return result

            star_scores = []
            for p in players:
                per = p.get("per", 0)
                mpg = p.get("mpg", 0)
                star_scores.append({
                    "player_id": str(p.get("player_id", "")),
                    "star_score": per * mpg,
                })
            star_scores_sorted = sorted(star_scores, key=lambda x: x["star_score"], reverse=True)
            if not star_scores_sorted:
                return result

            top3_team = star_scores_sorted[:3]
            top3_team_sum = sum(s["star_score"] for s in top3_team)
            top1_player_id = star_scores_sorted[0]["player_id"]
            top1_star_score = star_scores_sorted[0]["star_score"]
            injured_set = {str(pid) for pid in injured_ids} if injured_ids else set()

            result["inj_top1_star_out"] = 1.0 if top1_player_id in injured_set else 0.0
            if top1_player_id in injured_set and top3_team_sum > EPS:
                result["inj_star_share"] = top1_star_score / (top3_team_sum + EPS)
            injured_top3_sum = sum(
                s["star_score"] for s in top3_team if s["player_id"] in injured_set
            )
            if top3_team_sum > EPS:
                result["inj_star_score_share"] = min(
                    1.5, max(0.0, injured_top3_sum / (top3_team_sum + EPS))
                )
            return result

        home_star = compute_star_injury_features(
            HOME, home_injured_ids, home_per_features if per_calculator else None,
        )
        away_star = compute_star_injury_features(
            AWAY, away_injured_ids, away_per_features if per_calculator else None,
        )

        features["inj_star_share|none|raw|home"] = home_star["inj_star_share"]
        features["inj_star_share|none|raw|away"] = away_star["inj_star_share"]
        features["inj_star_share|none|raw|diff"] = home_star["inj_star_share"] - away_star["inj_star_share"]

        features["inj_star_score_share|none|top3_sum|home"] = home_star["inj_star_score_share"]
        features["inj_star_score_share|none|top3_sum|away"] = away_star["inj_star_score_share"]
        features["inj_star_score_share|none|top3_sum|diff"] = (
            home_star["inj_star_score_share"] - away_star["inj_star_score_share"]
        )

        features["inj_top1_star_out|none|binary|home"] = home_star["inj_top1_star_out"]
        features["inj_top1_star_out|none|binary|away"] = away_star["inj_top1_star_out"]
        features["inj_top1_star_out|none|binary|diff"] = (
            home_star["inj_top1_star_out"] - away_star["inj_top1_star_out"]
        )

        # Injured player names (for UI)
        features["home_injured_players"] = home_injured_names
        features["away_injured_players"] = away_injured_names

        return features

    # ------------------------------------------------------------------
    # Per-team injury computation (public + private)
    # ------------------------------------------------------------------

    def compute_team_injury_features(
        self, team, season, game_date_str, injured_player_ids,
        per_calculator=None, recency_decay_k=15.0,
    ):
        """Compute injury features for a single team.

        Public API for views/services that need per-team injury breakdowns
        (as opposed to get_injury_features() which returns a full matchup).

        Args:
            team: Team abbreviation
            season: Season string (e.g. "2024-2025")
            game_date_str: YYYY-MM-DD
            injured_player_ids: List of injured player ID strings
            per_calculator: Optional PERCalculator instance
            recency_decay_k: Decay constant for recency weighting

        Returns:
            Dict of raw injury feature values for the team.
        """
        from datetime import datetime
        game_date_obj = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        return self._calculate_team_injury_features(
            team, season, game_date_str, game_date_obj,
            injured_player_ids, per_calculator, recency_decay_k,
        )

    def _calculate_team_injury_features(
        self, team, season, game_date, game_date_obj,
        injured_player_ids, per_calculator, recency_decay_k,
    ):
        """Calculate injury features for a single team."""
        empty = {
            "injPerValue": 0.0, "injTop1Per": 0.0, "injTop3PerSum": 0.0,
            "injMinLost": 0.0, "injurySeverity": 0.0, "injRotation": 0.0,
            "injRotationShare": 0.0,
            "injPerValue_players": [], "injTop1Per_players": [],
            "injTop3PerSum_players": [], "injMinLost_players": [],
            "injRotation_players": [],
        }
        if not injured_player_ids:
            return empty

        player_stats = self._get_player_season_stats(team, season, game_date, injured_player_ids)
        if not player_stats:
            return empty

        # Filter: only players whose last game was for this team
        team_players = [
            (pid, stats) for pid, stats in player_stats.items()
            if stats.get("last_game_team") == team
        ]
        if not team_players:
            return empty

        # USAGE thresholds from league config
        thresholds = self._get_usage_thresholds()
        mpg_thresh = thresholds["mpg_thresh"]
        team_gp = self._get_team_games_played(team, season, game_date)

        # Full USAGE filter (MPG + GP with OR rule) for PER/value features
        usage_players = [
            (pid, s) for pid, s in team_players
            if self._passes_usage_filter(
                s.get("mpg", 0), s.get("games_played", 0), team_gp, thresholds,
            )
        ]

        # MPG-only filter for bodies/depth features (min_lost, rotation, severity)
        # A mid-season trade playing real minutes IS part of the rotation
        rotation_players = [
            (pid, s) for pid, s in team_players
            if s.get("mpg", 0) >= mpg_thresh
        ]

        if not usage_players and not rotation_players:
            return empty

        # Get PER values for usage players (PER features need GP-stable estimates)
        if per_calculator:
            for player_id, stats in usage_players:
                per = per_calculator.get_player_per_before_date(
                    player_id, team, season, game_date,
                )
                stats["per"] = per if per else 0.0

        max_mpg = self._get_max_mpg_on_team(team, season, game_date)
        if max_mpg == 0:
            max_mpg = 1.0

        team_rotation_mpg = self._get_team_rotation_mpg(team, season, game_date)

        # Player names (skip in batch)
        player_names = {}
        if not self._batch_training_mode and self._players_dir_repo is not None:
            try:
                all_pids = {str(pid) for pid, _ in usage_players} | {str(pid) for pid, _ in rotation_players}
                ids = list(all_pids)
                if ids:
                    docs = self._players_dir_repo.find(
                        {"player_id": {"$in": ids}},
                        projection={"player_id": 1, "player_name": 1},
                    )
                    player_names = {str(d["player_id"]): d.get("player_name", "Unknown") for d in docs}
            except Exception:
                pass

        # 1. injPerValue: Weighted sum of injured players' PERs (USAGE filter)
        inj_per_values = []
        inj_per_value_players = []
        for player_id, stats in usage_players:
            per = stats.get("per", 0.0)
            mpg = stats.get("mpg", 0.0)
            last_played_date = stats.get("last_played_date")
            if per == 0.0 or mpg == 0.0 or last_played_date is None:
                continue
            mpg_weight = mpg / max_mpg
            days_since = (game_date_obj - last_played_date).days
            recency_weight = math.exp(-days_since / recency_decay_k)
            weighted_per = per * mpg_weight * recency_weight
            inj_per_values.append(weighted_per)
            inj_per_value_players.append({
                "player_id": str(player_id),
                "player_name": player_names.get(str(player_id), "Unknown"),
                "per": per, "mpg": mpg, "weighted_per": weighted_per,
            })

        inj_per_value = sum(inj_per_values) if inj_per_values else 0.0

        # 2. injTop1Per (USAGE filter — PER needs GP stability)
        injured_pers_with_players = [
            (s.get("per", 0.0), pid, s)
            for pid, s in usage_players if s.get("per", 0.0) > 0
        ]
        injured_pers = [per for per, _, _ in injured_pers_with_players]
        inj_top1_per = max(injured_pers) if injured_pers else 0.0

        inj_top1_per_players = []
        if injured_pers_with_players:
            top1 = max(injured_pers_with_players, key=lambda x: x[0])
            inj_top1_per_players.append({
                "player_id": str(top1[1]),
                "player_name": player_names.get(str(top1[1]), "Unknown"),
                "per": top1[0], "mpg": top1[2].get("mpg", 0.0),
            })

        # 2b. injTop3PerSum
        inj_top3_per_sum_players = []
        if injured_pers_with_players:
            sorted_pers = sorted(injured_pers_with_players, key=lambda x: x[0], reverse=True)
            inj_top3_per_sum = sum(per for per, _, _ in sorted_pers[:3])
            for per, pid, s in sorted_pers[:3]:
                inj_top3_per_sum_players.append({
                    "player_id": str(pid),
                    "player_name": player_names.get(str(pid), "Unknown"),
                    "per": per, "mpg": s.get("mpg", 0.0),
                })
        else:
            inj_top3_per_sum = 0.0

        # 3. injMinLost
        inj_min_lost = sum(s.get("mpg", 0.0) for _, s in rotation_players)
        inj_min_lost_players = [{
            "player_id": str(pid),
            "player_name": player_names.get(str(pid), "Unknown"),
            "mpg": s.get("mpg", 0.0), "per": s.get("per", 0.0),
        } for pid, s in rotation_players]

        # 4. injurySeverity
        injury_severity = inj_min_lost / team_rotation_mpg if team_rotation_mpg > 0 else 0.0

        # 5. injRotation
        inj_rotation = len(rotation_players)
        inj_rotation_players = [{
            "player_id": str(pid),
            "player_name": player_names.get(str(pid), "Unknown"),
            "mpg": s.get("mpg", 0.0), "per": s.get("per", 0.0),
        } for pid, s in rotation_players]

        # 6. injRotationShare: fraction of rotation players injured (bodies feature)
        team_rotation_count = self._get_team_rotation_count(team, season, game_date)
        inj_rotation_share = inj_rotation / team_rotation_count if team_rotation_count > 0 else 0.0

        return {
            "injPerValue": inj_per_value,
            "injTop1Per": inj_top1_per,
            "injTop3PerSum": inj_top3_per_sum,
            "injMinLost": inj_min_lost,
            "injurySeverity": injury_severity,
            "injRotation": float(inj_rotation),
            "injRotationShare": inj_rotation_share,
            "injPerValue_players": inj_per_value_players,
            "injTop1Per_players": inj_top1_per_players,
            "injTop3PerSum_players": inj_top3_per_sum_players,
            "injMinLost_players": inj_min_lost_players,
            "injRotation_players": inj_rotation_players,
        }

    # ------------------------------------------------------------------
    # Player stats helpers
    # ------------------------------------------------------------------

    def _get_player_season_stats(self, team, season, before_date, player_ids):
        """Get season-to-date stats for players (MPG, last game date/team)."""
        if not player_ids:
            return {}

        player_ids_set = set(str(pid) for pid in player_ids)
        cache_key = (team, season, before_date)

        if cache_key in self._injury_player_stats_cache:
            cached = self._injury_player_stats_cache[cache_key]
            return {pid: cached[pid] for pid in player_ids_set if pid in cached}

        # Use preloaded data or fall back to DB
        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            player_records = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get("date", "") < before_date
            ]
        else:
            if not hasattr(self, "_db_fallback_player_stats"):
                self._db_fallback_player_stats = 0
            self._db_fallback_player_stats += 1
            if self._db_fallback_player_stats <= 3:
                print(
                    f"[DB FALLBACK] _get_player_season_stats #{self._db_fallback_player_stats}: "
                    f"team={team}, season={season}"
                )
            player_records = self._players_repo.find(
                {
                    "team": team, "season": season,
                    "date": {"$lt": before_date},
                    "stats.min": {"$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={"player_id": 1, "team": 1, "date": 1, "stats.min": 1},
                sort=[("date", 1)],
            )

        if not player_records:
            self._injury_player_stats_cache[cache_key] = {}
            return {}

        player_agg = defaultdict(lambda: {
            "total_minutes": 0.0, "games_played": 0,
            "last_played_date": None, "last_game_team": None,
        })
        for record in player_records:
            pid = str(record.get("player_id"))
            game_date_str = record.get("date")
            minutes = record.get("stats", {}).get("min", 0.0)
            if minutes > 0:
                player_agg[pid]["total_minutes"] += minutes
                player_agg[pid]["games_played"] += 1
                gd = datetime.strptime(game_date_str, "%Y-%m-%d").date()
                if player_agg[pid]["last_played_date"] is None or gd > player_agg[pid]["last_played_date"]:
                    player_agg[pid]["last_played_date"] = gd
                    player_agg[pid]["last_game_team"] = record.get("team")

        result = {}
        for pid, agg in player_agg.items():
            if agg["games_played"] == 0:
                continue
            result[pid] = {
                "mpg": agg["total_minutes"] / agg["games_played"],
                "games_played": agg["games_played"],
                "per": 0.0,
                "last_played_date": agg["last_played_date"],
                "last_game_team": agg["last_game_team"],
            }

        self._injury_player_stats_cache[cache_key] = result
        return {pid: result[pid] for pid in player_ids_set if pid in result}

    def _get_max_mpg_on_team(self, team, season, before_date):
        """Max MPG among all players on the team (season-to-date)."""
        cache_key = (team, season, before_date)
        if cache_key in self._injury_max_mpg_cache:
            return self._injury_max_mpg_cache[cache_key]

        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            all_players = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get("date", "") < before_date
            ]
        else:
            all_players = self._players_repo.find(
                {
                    "team": team, "season": season,
                    "date": {"$lt": before_date},
                    "stats.min": {"$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={"player_id": 1, "stats.min": 1},
            )

        if not all_players:
            self._injury_max_mpg_cache[cache_key] = 0.0
            return 0.0

        player_mpg = defaultdict(lambda: {"total_min": 0.0, "games": 0})
        for record in all_players:
            pid = str(record.get("player_id"))
            minutes = record.get("stats", {}).get("min", 0.0)
            player_mpg[pid]["total_min"] += minutes
            player_mpg[pid]["games"] += 1

        max_mpg = 0.0
        for pid, agg in player_mpg.items():
            if agg["games"] > 0:
                mpg = agg["total_min"] / agg["games"]
                max_mpg = max(max_mpg, mpg)

        self._injury_max_mpg_cache[cache_key] = max_mpg
        return max_mpg

    def _get_team_rotation_mpg(self, team, season, before_date):
        """Sum of MPG for rotation players (mpg >= mpg_thresh) on the team.

        Bodies/depth denominator — uses MPG-only, no GP filter.
        A mid-season addition playing real minutes is part of the rotation.
        """
        mpg_thresh = self._get_usage_thresholds()["mpg_thresh"]

        cache_key = (team, season, before_date)
        if cache_key in self._injury_rotation_mpg_cache:
            return self._injury_rotation_mpg_cache[cache_key]

        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            all_players = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get("date", "") < before_date
            ]
        else:
            if not hasattr(self, "_db_fallback_rotation_mpg"):
                self._db_fallback_rotation_mpg = 0
            self._db_fallback_rotation_mpg += 1
            if self._db_fallback_rotation_mpg <= 3:
                print(
                    f"[DB FALLBACK] _get_team_rotation_mpg #{self._db_fallback_rotation_mpg}: "
                    f"team={team}, season={season}"
                )
            all_players = self._players_repo.find(
                {
                    "team": team, "season": season,
                    "date": {"$lt": before_date},
                    "stats.min": {"$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={"player_id": 1, "stats.min": 1},
            )

        if not all_players:
            self._injury_rotation_mpg_cache[cache_key] = 0.0
            return 0.0

        player_mpg = defaultdict(lambda: {"total_min": 0.0, "games": 0})
        for record in all_players:
            pid = str(record.get("player_id"))
            minutes = record.get("stats", {}).get("min", 0.0)
            player_mpg[pid]["total_min"] += minutes
            player_mpg[pid]["games"] += 1

        total_rotation_mpg = 0.0
        for pid, agg in player_mpg.items():
            if agg["games"] > 0:
                mpg = agg["total_min"] / agg["games"]
                if mpg >= mpg_thresh:
                    total_rotation_mpg += mpg

        self._injury_rotation_mpg_cache[cache_key] = total_rotation_mpg
        return total_rotation_mpg

    def _get_team_rotation_count(self, team, season, before_date):
        """Number of rotation players (mpg >= mpg_thresh) on the team.

        Bodies/depth denominator — uses MPG-only, no GP filter.
        """
        mpg_thresh = self._get_usage_thresholds()["mpg_thresh"]

        # Warm the full-team player stats cache
        cache_key = (team, season, before_date)
        if cache_key not in self._injury_player_stats_cache:
            self._warm_player_stats_cache(team, season, before_date)

        all_player_stats = self._injury_player_stats_cache.get(cache_key, {})
        return sum(
            1 for s in all_player_stats.values()
            if s.get("mpg", 0.0) >= mpg_thresh and s.get("last_game_team") == team
        )

    def _get_team_weighted_per_mass(
        self, team, season, game_date, game_date_obj,
        per_calculator, recency_decay_k=15.0,
    ):
        """Compute team's weighted PER mass (denominator for inj_per_weighted_share)."""
        if per_calculator is None:
            return 0.0

        cache_key = (team, season, game_date, recency_decay_k)
        if cache_key in self._team_weighted_per_mass_cache:
            return self._team_weighted_per_mass_cache[cache_key]

        thresholds = self._get_usage_thresholds()
        mpg_thresh = thresholds["mpg_thresh"]

        max_mpg = self._get_max_mpg_on_team(team, season, game_date)
        if max_mpg == 0:
            max_mpg = 1.0

        player_stats_cache_key = (team, season, game_date)
        if player_stats_cache_key not in self._injury_player_stats_cache:
            self._warm_player_stats_cache(team, season, game_date)

        all_player_stats = self._injury_player_stats_cache.get(player_stats_cache_key, {})
        if not all_player_stats:
            self._team_weighted_per_mass_cache[cache_key] = 0.0
            return 0.0

        # USAGE filter (PER denominator — same OR rule as numerator)
        team_gp = max(
            (s.get("games_played", 0) for s in all_player_stats.values()),
            default=0,
        )

        rotation_players = [
            (pid, s) for pid, s in all_player_stats.items()
            if s.get("last_game_team") == team
            and self._passes_usage_filter(
                s.get("mpg", 0.0), s.get("games_played", 0), team_gp, thresholds,
            )
        ]
        if not rotation_players:
            self._team_weighted_per_mass_cache[cache_key] = 0.0
            return 0.0

        weighted_per_mass = 0.0
        for pid, stats in rotation_players:
            mpg = stats.get("mpg", 0.0)
            last_played_date = stats.get("last_played_date")
            if mpg == 0.0 or last_played_date is None:
                continue
            per = per_calculator.get_player_per_before_date(pid, team, season, game_date)
            if per is None or per <= 0:
                continue
            mpg_weight = mpg / max_mpg
            days_since = (game_date_obj - last_played_date).days
            recency_weight = math.exp(-days_since / recency_decay_k)
            weighted_per_mass += per * mpg_weight * recency_weight

        self._team_weighted_per_mass_cache[cache_key] = weighted_per_mass
        return weighted_per_mass

    def _warm_player_stats_cache(self, team, season, before_date):
        """Warm the player stats cache for a team/season/date."""
        cache_key = (team, season, before_date)
        if cache_key in self._injury_player_stats_cache:
            return

        if self._injury_cache_loaded and (team, season) in self._injury_preloaded_players:
            player_records = [
                r for r in self._injury_preloaded_players[(team, season)]
                if r.get("date", "") < before_date
            ]
        else:
            player_records = self._players_repo.find(
                {
                    "team": team, "season": season,
                    "date": {"$lt": before_date},
                    "stats.min": {"$gt": 0},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={"player_id": 1, "team": 1, "date": 1, "stats.min": 1},
                sort=[("date", 1)],
            )

        if not player_records:
            self._injury_player_stats_cache[cache_key] = {}
            return

        player_agg = defaultdict(lambda: {
            "total_minutes": 0.0, "games_played": 0,
            "last_played_date": None, "last_game_team": None,
        })
        for record in player_records:
            pid = str(record.get("player_id"))
            game_date_str = record.get("date")
            minutes = record.get("stats", {}).get("min", 0.0)
            if minutes > 0:
                player_agg[pid]["total_minutes"] += minutes
                player_agg[pid]["games_played"] += 1
                gd = datetime.strptime(game_date_str, "%Y-%m-%d").date()
                if player_agg[pid]["last_played_date"] is None or gd > player_agg[pid]["last_played_date"]:
                    player_agg[pid]["last_played_date"] = gd
                    player_agg[pid]["last_game_team"] = record.get("team")

        result = {}
        for pid, agg in player_agg.items():
            if agg["games_played"] == 0:
                continue
            result[pid] = {
                "mpg": agg["total_minutes"] / agg["games_played"],
                "games_played": agg["games_played"],
                "per": 0.0,
                "last_played_date": agg["last_played_date"],
                "last_game_team": agg["last_game_team"],
            }

        self._injury_player_stats_cache[cache_key] = result

    def _get_season_injury_severity(self, team, season, before_date):
        """Season-to-date injury severity: ratio-of-sums over prior games."""
        EPS = 1e-6

        cache_key = (team, season, before_date, "season_inj_severity")
        if cache_key in self._season_injury_severity_cache:
            return self._season_injury_severity_cache[cache_key]

        prior_games = []
        if self.games_home is not None and self.games_away is not None:
            all_games = self._get_team_games_in_range(
                team, season, end_date_str=before_date, exclude_game_types=[],
            )
            prior_games = [g for g in all_games if g.get("homeWon") is not None]
        else:
            if self._games_repo is None:
                self._season_injury_severity_cache[cache_key] = 0.0
                return 0.0
            if not hasattr(self, "_db_fallback_season_severity"):
                self._db_fallback_season_severity = 0
            self._db_fallback_season_severity += 1
            if self._db_fallback_season_severity <= 3:
                print(
                    f"[DB FALLBACK] _get_season_injury_severity #{self._db_fallback_season_severity}: "
                    f"team={team}, season={season}"
                )
            prior_games = list(self._games_repo.find(
                {
                    "season": season,
                    "date": {"$lt": before_date},
                    "$or": [
                        {"homeTeam.name": team},
                        {"awayTeam.name": team},
                    ],
                    "homeWon": {"$exists": True},
                    "game_type": {"$nin": self._exclude_game_types},
                },
                projection={
                    "date": 1,
                    "homeTeam.name": 1, "homeTeam.injured_players": 1,
                    "awayTeam.name": 1, "awayTeam.injured_players": 1,
                },
            ))

        if not prior_games:
            self._season_injury_severity_cache[cache_key] = 0.0
            return 0.0

        total_inj_min_lost = 0.0
        total_rotation_mpg = 0.0
        # Severity is a bodies/depth feature — MPG-only, no GP filter
        _mpg_thresh = self._get_usage_thresholds()["mpg_thresh"]

        for game in prior_games:
            g_date = game.get("date")
            if not g_date:
                continue
            home_team = game.get("homeTeam", {}).get("name")
            is_home = (home_team == team)
            if is_home:
                injured_ids = game.get("homeTeam", {}).get("injured_players", [])
            else:
                injured_ids = game.get("awayTeam", {}).get("injured_players", [])
            injured_ids = [str(pid) for pid in injured_ids] if injured_ids else []

            game_rotation_mpg = self._get_team_rotation_mpg(team, season, g_date)
            total_rotation_mpg += game_rotation_mpg

            if not injured_ids:
                continue

            player_stats = self._get_player_season_stats(team, season, g_date, injured_ids)
            game_inj_min_lost = 0.0
            for pid, stats in player_stats.items():
                mpg = stats.get("mpg", 0.0)
                if mpg >= _mpg_thresh and stats.get("last_game_team") == team:
                    game_inj_min_lost += mpg
            total_inj_min_lost += game_inj_min_lost

        if total_rotation_mpg <= 0:
            result = 0.0
        else:
            result = total_inj_min_lost / (total_rotation_mpg + EPS)

        self._season_injury_severity_cache[cache_key] = result
        return result
