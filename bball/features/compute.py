"""
Basketball Feature Computer — primary orchestrator for regular stat features.

Uses sportscore's StatEngine for standard stats and dispatches custom stats
to handler functions registered in custom_stats.py.

Usage:
    computer = BasketballFeatureComputer(db, league=league)
    computer.set_preloaded_data(games_home, games_away, venue_cache)

    results = computer.compute_matchup_features(
        feature_names, home_team, away_team,
        season, game_date, venue_guid=venue_guid
    )
"""

import bisect
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from sportscore.features.stat_engine import StatEngine

from bball.features.registry import FeatureRegistry
from bball.features.custom_stats import CUSTOM_HANDLERS
from bball.features.parser import parse_feature_name


class BasketballFeatureComputer:
    """Primary orchestrator for regular feature computation.

    PER and injury features remain handled externally by
    SharedFeatureGenerator (bulk computation via PERCalculator
    and InjuryFeatureCalculator).
    """

    def __init__(self, db=None, league=None, recency_alpha=0.0):
        self.db = db
        self.league = league

        # Core engine
        self.engine = StatEngine(
            stat_definitions=FeatureRegistry.STAT_DEFINITIONS,
            custom_handlers=CUSTOM_HANDLERS,
            recency_alpha=recency_alpha,
            recency_mode="date",
        )

        # Preloaded game data (set via set_preloaded_data or injected)
        self.games_home = None   # {season: {date: {home_team: game_doc}}}
        self.games_away = None   # {season: {date: {away_team: game_doc}}}
        self._venue_cache = {}   # {venue_guid: (lat, lon)}

        # Team index for fast bisect lookups
        self._team_games_index = {}  # {season: {team: [(date_str, game_doc), ...]}}
        self._team_dates_index = {}  # {season: {team: [date_str, ...]}}

        # Caches
        self._team_games_cache = {}   # (team, season, date_str) -> [game_doc]
        self._elo_cache = None
        self._conference_cache = {}   # {team_abbrev: conference_name}
        self._conf_teams_cache = {}   # {conference_name: set(team_abbrevs/ids)}

        # Exclude game types from league config
        if league:
            self._exclude_game_types = league.exclude_game_types
        else:
            self._exclude_game_types = ["preseason", "allstar"]

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def set_preloaded_data(self, games_home, games_away, venue_cache=None):
        """Inject preloaded game data and build indexes."""
        self.games_home = games_home
        self.games_away = games_away
        if venue_cache:
            self._venue_cache.update(venue_cache)
        self._build_team_index()

    def preload_venue_cache(self):
        """Preload venue coordinates from DB."""
        if self.db is None:
            return
        venues_coll = "nba_venues"
        if self.league:
            venues_coll = self.league.collections.get("venues", venues_coll)
        venues = list(self.db[venues_coll].find(
            {},
            {"venue_guid": 1, "location.lat": 1, "location.lon": 1, "location.long": 1},
        ))
        for v in venues:
            guid = v.get("venue_guid")
            loc = v.get("location", {})
            if guid and loc:
                lat = loc.get("lat")
                lon = loc.get("lon") if "lon" in loc else loc.get("long")
                if lat is not None and lon is not None:
                    self._venue_cache[guid] = (lat, lon)

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

    # ------------------------------------------------------------------
    # Game retrieval
    # ------------------------------------------------------------------

    def _get_team_season_games(self, team: str, season: str, game_date: str) -> List[Dict]:
        """Get all games for a team before game_date in the given season.

        Uses preloaded index when available, falls back to DB query.
        """
        cache_key = (team, season, game_date)
        if cache_key in self._team_games_cache:
            return self._team_games_cache[cache_key]

        games = []

        if self._team_dates_index and self._team_dates_index.get(season, {}).get(team):
            # Fast bisect lookup on preloaded data
            dates = self._team_dates_index[season][team]
            pairs = self._team_games_index[season][team]
            hi = bisect.bisect_left(dates, game_date)
            exclude_set = set(self._exclude_game_types)
            games = [
                g for _, g in pairs[:hi]
                if g.get("game_type", "regseason") not in exclude_set
            ]
        elif self.db is not None:
            # Lazy DB query
            from bball.data import GamesRepository
            repo = GamesRepository(self.db, league=self.league)
            query = {
                "season": season,
                "date": {"$lt": game_date},
                "game_type": {"$nin": self._exclude_game_types},
                "$or": [
                    {"homeTeam.name": team},
                    {"awayTeam.name": team},
                ],
                "homeTeam.points": {"$exists": True, "$gt": 0},
                "awayTeam.points": {"$exists": True, "$gt": 0},
            }
            games = repo.find(query, sort=[("date", 1)])
            for g in games:
                g["_id"] = str(g["_id"])

        self._team_games_cache[cache_key] = games
        return games

    # ------------------------------------------------------------------
    # Main computation
    # ------------------------------------------------------------------

    def compute_matchup_features(
        self,
        feature_names: List[str],
        home_team: str,
        away_team: str,
        season: str,
        game_date: str,
        venue_guid: Optional[str] = None,
    ) -> Dict[str, Optional[float]]:
        """Compute all requested features for a matchup.

        Args:
            feature_names: Pipe-delimited feature names
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season string (e.g. "2024-2025")
            game_date: YYYY-MM-DD
            venue_guid: Optional venue GUID for travel features

        Returns:
            Dict mapping feature name -> computed value (or 0.0)
        """
        # Fetch season games for each team
        home_games = self._get_team_season_games(home_team, season, game_date)
        away_games = self._get_team_season_games(away_team, season, game_date)

        # Build context dict for custom handlers
        context = self._build_context(
            home_team, away_team, season, game_date, venue_guid,
        )

        results: Dict[str, Optional[float]] = {}

        # Categorize features: regular vs composite time periods
        regular = []
        composite = []

        for fname in feature_names:
            parts = fname.split("|")
            if len(parts) >= 4:
                time_period = parts[1]
                if (time_period.startswith("blend:") or
                        time_period.startswith("delta:") or
                        time_period.startswith("blend-delta:")):
                    composite.append(fname)
                else:
                    regular.append(fname)
            else:
                regular.append(fname)

        # Compute regular features via engine
        for fname in regular:
            try:
                val = self.engine.compute_feature(
                    fname, home_team, away_team,
                    home_games, away_games, context,
                )
                results[fname] = val if val is not None else 0.0
            except Exception as e:
                print(f"[BasketballFeatureComputer] Error computing {fname}: {e}")
                results[fname] = 0.0

        # Compute composite time period features
        for fname in composite:
            try:
                val = self._compute_composite(
                    fname, home_team, away_team,
                    home_games, away_games, context,
                )
                results[fname] = val if val is not None else 0.0
            except Exception as e:
                print(f"[BasketballFeatureComputer] Error computing composite {fname}: {e}")
                results[fname] = 0.0

        return results

    def _build_context(
        self, home_team, away_team, season, game_date, venue_guid=None,
    ) -> Dict:
        """Build context dict passed to custom handlers."""
        # Find game doc for vegas/postseason lookups
        game_doc = None
        if self.games_home and season in self.games_home:
            date_games = self.games_home[season].get(game_date, {})
            game_doc = date_games.get(home_team)

        return {
            "engine": self.engine,
            "reference_date": game_date,
            "season": season,
            "db": self.db,
            "league": self.league,
            "elo_cache": self._elo_cache,
            "venue_cache": self._venue_cache,
            "games_home": self.games_home,
            "games_away": self.games_away,
            "team_games_index": self._team_games_index,
            "team_dates_index": self._team_dates_index,
            "game_doc": game_doc,
            "target_venue_guid": venue_guid,
            "exclude_game_types": self._exclude_game_types,
            "conference_cache": self._conference_cache,
            "conf_teams_cache": self._conf_teams_cache,
            # Shared SOS cache: persists across all SOS feature calls for this game.
            # Key: (opponent, inner_stat, inner_weight) -> computed value.
            # Avoids recomputing the same opponent's season stat across perspectives,
            # across SOS stat types sharing inner stats (off_rtg, def_rtg → net_rtg),
            # and across 2nd-order SOS lookups.
            "sos_cache": {},
        }

    # ------------------------------------------------------------------
    # Composite time periods (blend / delta / blend-delta)
    # ------------------------------------------------------------------

    def _compute_composite(
        self, feature_name, home_team, away_team,
        home_games, away_games, context,
    ) -> Optional[float]:
        """Handle blend: and delta: composite time periods.

        blend:tp1:w1/tp2:w2 — weighted combination of time periods
        delta:recent-baseline — difference between two time periods
        blend-delta:tp1:w1/tp2:w2-baseline — blend first, then delta
        """
        parts = feature_name.split("|")
        if len(parts) == 5:
            stat_name, time_period, calc_weight, perspective, _ = parts
        elif len(parts) == 4:
            stat_name, time_period, calc_weight, perspective = parts
        else:
            return None

        if time_period.startswith("blend-delta:"):
            return self._compute_blend_delta(
                stat_name, time_period, calc_weight, perspective,
                home_team, away_team, home_games, away_games, context,
            )

        if time_period.startswith("blend:"):
            return self._compute_blend(
                stat_name, time_period, calc_weight, perspective,
                home_team, away_team, home_games, away_games, context,
            )

        if time_period.startswith("delta:"):
            return self._compute_delta(
                stat_name, time_period, calc_weight, perspective,
                home_team, away_team, home_games, away_games, context,
            )

        return None

    def _compute_blend(
        self, stat_name, time_period, calc_weight, perspective,
        home_team, away_team, home_games, away_games, context,
    ) -> Optional[float]:
        """blend:tp1:w1/tp2:w2[/tp3:w3] — weighted sum of sub-features."""
        spec = time_period[len("blend:"):]
        segments = spec.split("/")

        total = 0.0
        total_weight = 0.0

        for seg in segments:
            parts = seg.rsplit(":", 1)
            if len(parts) != 2:
                continue
            tp, w_str = parts
            try:
                w = float(w_str)
            except ValueError:
                continue

            sub_feature = f"{stat_name}|{tp}|{calc_weight}|{perspective}"
            val = self.engine.compute_feature(
                sub_feature, home_team, away_team,
                home_games, away_games, context,
            )
            if val is not None:
                total += float(val) * w
                total_weight += w

        return total / total_weight if total_weight > 0 else None

    def _compute_delta(
        self, stat_name, time_period, calc_weight, perspective,
        home_team, away_team, home_games, away_games, context,
    ) -> Optional[float]:
        """delta:recent-baseline — recent minus baseline."""
        spec = time_period[len("delta:"):]
        parts = spec.split("-", 1)
        if len(parts) != 2:
            return None

        recent_tp, baseline_tp = parts

        recent_val = self._compute_sub_feature(
            stat_name, recent_tp, calc_weight, perspective,
            home_team, away_team, home_games, away_games, context,
        )
        baseline_val = self._compute_sub_feature(
            stat_name, baseline_tp, calc_weight, perspective,
            home_team, away_team, home_games, away_games, context,
        )

        if recent_val is not None and baseline_val is not None:
            return float(recent_val) - float(baseline_val)
        return None

    def _compute_sub_feature(
        self, stat_name, tp, calc_weight, perspective,
        home_team, away_team, home_games, away_games, context,
    ) -> Optional[float]:
        """Compute a sub-feature value, routing blend: time periods correctly."""
        if tp.startswith("blend:"):
            return self._compute_blend(
                stat_name, tp, calc_weight, perspective,
                home_team, away_team, home_games, away_games, context,
            )
        feature = f"{stat_name}|{tp}|{calc_weight}|{perspective}"
        return self.engine.compute_feature(
            feature, home_team, away_team,
            home_games, away_games, context,
        )

    def _compute_blend_delta(
        self, stat_name, time_period, calc_weight, perspective,
        home_team, away_team, home_games, away_games, context,
    ) -> Optional[float]:
        """blend-delta:tp1:w1/tp2:w2-baseline — blend then subtract baseline."""
        spec = time_period[len("blend-delta:"):]

        # Split on last '-' that separates blend spec from baseline
        # But blend spec may contain '-' in time period names, so we need
        # to find the baseline which is after the last segment separator
        # Format: tp1:w1/tp2:w2-baseline
        # The baseline is after the last '-' that follows a weight
        parts = spec.rsplit("-", 1)
        if len(parts) != 2:
            return None

        blend_spec, baseline_tp = parts

        # Compute blend
        blend_feature_tp = f"blend:{blend_spec}"
        blend_val = self._compute_blend(
            stat_name, blend_feature_tp, calc_weight, perspective,
            home_team, away_team, home_games, away_games, context,
        )

        # Compute baseline
        baseline_feature = f"{stat_name}|{baseline_tp}|{calc_weight}|{perspective}"
        baseline_val = self.engine.compute_feature(
            baseline_feature, home_team, away_team,
            home_games, away_games, context,
        )

        if blend_val is not None and baseline_val is not None:
            return float(blend_val) - float(baseline_val)
        return None
