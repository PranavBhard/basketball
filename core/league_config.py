"""
League configuration loader.

The app supports multiple basketball leagues (NBA, CBB, ...). All league-specific
variables live in YAML files under `nba_app/leagues/<league_id>.yaml`.

This module loads and validates those configs and provides a small, typed API
to access common fields (collection names, ESPN endpoints, paths).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import yaml


class LeagueConfigError(ValueError):
    pass


def _repo_root() -> str:
    """
    Return filesystem path to the repo root (the directory containing `nba_app/`).
    """
    # core/league_config.py -> nba_app/core -> nba_app -> repo root
    nba_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(nba_app_dir)


def _nba_app_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise LeagueConfigError(f"League config must be a mapping at top-level: {path}")
        return data
    except FileNotFoundError as e:
        raise LeagueConfigError(f"League config not found: {path}") from e
    except yaml.YAMLError as e:
        raise LeagueConfigError(f"Failed to parse YAML: {path}: {e}") from e


def _require(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise LeagueConfigError(f"Missing required key '{key}' in {ctx}")
    return d[key]


def _as_str(x: Any, ctx: str) -> str:
    if not isinstance(x, str) or not x.strip():
        raise LeagueConfigError(f"Expected non-empty string for {ctx}")
    return x


def _as_dict(x: Any, ctx: str) -> Dict[str, Any]:
    if not isinstance(x, dict):
        raise LeagueConfigError(f"Expected mapping for {ctx}")
    return x


def _normalize_path(path_str: str) -> str:
    """
    Normalize a config path. If relative, treat it as relative to repo root.
    """
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(_repo_root(), path_str)


class _SafeFormatDict(dict):
    """
    Allows partial .format_map() substitution.

    Any placeholder not provided in the mapping is preserved as `{key}` instead of
    raising KeyError (e.g., `{YYYYMMDD}`, `{game_id}`).
    """

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@dataclass(frozen=True)
class LeagueConfig:
    raw: Dict[str, Any]
    config_path: str

    @property
    def league_id(self) -> str:
        return _as_str(_require(_require(self.raw, "meta", self.config_path), "league_id", self.config_path), "meta.league_id")

    @property
    def display_name(self) -> str:
        meta = _require(self.raw, "meta", self.config_path)
        return _as_str(_require(meta, "display_name", self.config_path), "meta.display_name")

    @property
    def logo_url(self) -> Optional[str]:
        meta = _require(self.raw, "meta", self.config_path)
        logo = meta.get("logo_url")
        if not logo:
            return None
        return _as_str(logo, "meta.logo_url")

    # -----------------------
    # Mongo collections
    # -----------------------
    @property
    def collections(self) -> Dict[str, str]:
        mongo = _as_dict(_require(self.raw, "mongo", self.config_path), "mongo")
        cols = _as_dict(_require(mongo, "collections", self.config_path), "mongo.collections")
        # Validate common required collections
        required = [
            "games",
            "player_stats",
            "players",
            "teams",
            "venues",
            "rosters",
            "model_config_classifier",
            "model_config_points",
            "master_training_metadata",
            "cached_league_stats",
            "elo_cache",
            "experiment_runs",
            "jobs",
        ]
        out: Dict[str, str] = {}
        for k in required:
            out[k] = _as_str(_require(cols, k, f"{self.config_path} mongo.collections"), f"mongo.collections.{k}")
        # Preserve any additional collections
        for k, v in cols.items():
            if k not in out:
                if isinstance(v, str):
                    out[k] = v
        return out

    # -----------------------
    # Paths
    # -----------------------
    @property
    def master_training_csv(self) -> str:
        paths = _as_dict(_require(self.raw, "paths", self.config_path), "paths")
        p = _as_str(_require(paths, "master_training_csv", self.config_path), "paths.master_training_csv")
        return _normalize_path(p)

    @property
    def team_id_map_file(self) -> Optional[str]:
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return None
        f = teams.get("team_id_map_file")
        if not f:
            return None
        if not isinstance(f, str):
            raise LeagueConfigError(f"Expected string for teams.team_id_map_file in {self.config_path}")
        return _normalize_path(f)

    @property
    def team_primary_identifier(self) -> str:
        """
        Primary identifier for teams: 'name' (abbreviation) or 'id' (ESPN team_id).

        For NBA, 'name' (abbreviation like 'LAL') is sufficient since team names are unique.
        For CBB, 'id' (ESPN team_id) is preferred since multiple schools may share abbreviations.

        Returns 'name' by default for backward compatibility.
        """
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return "name"
        identifier = teams.get("primary_identifier", "name")
        if identifier not in ("name", "id"):
            raise LeagueConfigError(
                f"teams.primary_identifier must be 'name' or 'id', got '{identifier}' in {self.config_path}"
            )
        return identifier

    @property
    def include_team_id(self) -> bool:
        """Whether to include team_id in games and player stats."""
        teams = self.raw.get("teams") or {}
        if not isinstance(teams, dict):
            return False
        return bool(teams.get("include_team_id", False))

    # -----------------------
    # Season rules
    # -----------------------
    @property
    def season_rules(self) -> Dict[str, Any]:
        season = self.raw.get("season") or {}
        if not isinstance(season, dict):
            raise LeagueConfigError(f"Expected mapping for season in {self.config_path}")
        return season

    @property
    def timezone(self) -> str:
        season = self.season_rules
        tz = season.get("timezone", "America/New_York")
        return _as_str(tz, "season.timezone")

    @property
    def season_cutover_month(self) -> int:
        season = self.season_rules
        m = season.get("season_cutover_month", 8)
        try:
            m_int = int(m)
        except Exception as e:
            raise LeagueConfigError(f"Expected integer for season.season_cutover_month in {self.config_path}") from e
        if m_int < 1 or m_int > 12:
            raise LeagueConfigError(f"season.season_cutover_month out of range (1-12) in {self.config_path}")
        return m_int

    @property
    def exclude_game_types(self) -> list:
        season = self.season_rules
        v = season.get("exclude_game_types", [])
        if v is None:
            return []
        if not isinstance(v, list):
            raise LeagueConfigError(f"Expected list for season.exclude_game_types in {self.config_path}")
        return v

    # -----------------------
    # Extra features (league-specific)
    # -----------------------
    @property
    def extra_feature_stats(self) -> List[str]:
        """League-specific extra stat names from YAML config."""
        ef = self.raw.get("extra_features") or {}
        stats = ef.get("stats", [])
        return stats if isinstance(stats, list) else []

    # -----------------------
    # ESPN endpoint templates
    # -----------------------
    @property
    def espn(self) -> Dict[str, Any]:
        return _as_dict(_require(self.raw, "espn", self.config_path), "espn")

    def espn_endpoint(self, key: str) -> str:
        espn = self.espn
        endpoints = _as_dict(_require(espn, "endpoints", self.config_path), "espn.endpoints")
        template = _as_str(_require(endpoints, key, f"{self.config_path} espn.endpoints"), f"espn.endpoints.{key}")

        base_url_web = _as_str(_require(espn, "base_url_web", self.config_path), "espn.base_url_web")
        base_url_site = _as_str(_require(espn, "base_url_site", self.config_path), "espn.base_url_site")
        league_slug = _as_str(_require(espn, "league_slug", self.config_path), "espn.league_slug")
        sport_path = _as_str(_require(espn, "sport_path", self.config_path), "espn.sport_path")

        # Important: templates may include additional placeholders like {YYYYMMDD} or {game_id}.
        # We intentionally only fill league constants here and keep other placeholders intact.
        return template.format_map(_SafeFormatDict(
            base_url_web=base_url_web,
            base_url_site=base_url_site,
            league_slug=league_slug,
            sport_path=sport_path,
        ))

    def espn_page(self, key: str) -> str:
        """
        Return a formatted ESPN public website page URL from `espn.pages.*`.

        Example:
            league.espn_page("teams_template") -> "https://www.espn.com/nba/teams"
        """
        espn = self.espn
        pages = _as_dict(_require(espn, "pages", self.config_path), "espn.pages")
        template = _as_str(_require(pages, key, f"{self.config_path} espn.pages"), f"espn.pages.{key}")

        base_url_public = espn.get("base_url_public", "https://www.espn.com")
        base_url_public = _as_str(base_url_public, "espn.base_url_public")
        sport_path = _as_str(_require(espn, "sport_path", self.config_path), "espn.sport_path")

        return template.format_map(_SafeFormatDict(
            base_url_public=base_url_public,
            sport_path=sport_path,
        ))

    # -----------------------
    # Pipeline configuration
    # -----------------------
    @property
    def pipeline(self) -> Dict[str, Any]:
        """Pipeline configuration section."""
        return self.raw.get("pipeline") or {}

    @property
    def min_season(self) -> str:
        """First season with data (e.g., '2007-2008')."""
        return self.pipeline.get("min_season", "2007-2008")

    @property
    def season_start_month(self) -> int:
        return int(self.pipeline.get("season_start_month", 10))

    @property
    def season_start_day(self) -> int:
        return int(self.pipeline.get("season_start_day", 1))

    @property
    def season_end_month(self) -> int:
        return int(self.pipeline.get("season_end_month", 6))

    @property
    def season_end_day(self) -> int:
        return int(self.pipeline.get("season_end_day", 30))

    @property
    def start_year(self) -> int:
        """Extract start year from min_season (e.g., 2007 from '2007-2008')."""
        return int(self.min_season.split("-")[0])

    # -----------------------
    # Pipelines configuration
    # -----------------------
    @property
    def pipelines(self) -> Dict[str, Any]:
        """
        Pipeline configuration section.

        Returns the `pipelines` section from the league YAML, or empty dict
        if not configured. This section can contain:
        - full: Full pipeline steps configuration
        - training: Training-specific settings (workers, chunk_size, etc.)

        Example YAML:
            pipelines:
              full:
                steps:
                  - espn_pull:
                      parallel: true
                      max_workers: 4
                  - generate_training:
                      workers: 32
                      chunk_size: 500
              training:
                workers: 32
                chunk_size: 500
                include_player_features: true
        """
        return self.raw.get("pipelines") or {}


_CACHE: Dict[str, LeagueConfig] = {}


def get_available_leagues() -> List[str]:
    """Return list of available league IDs from YAML files in leagues/ dir."""
    leagues_dir = os.path.join(_nba_app_dir(), "leagues")
    if not os.path.isdir(leagues_dir):
        return []
    return [f[:-5] for f in os.listdir(leagues_dir) if f.endswith(".yaml")]


def get_league_config_path(league_id: str) -> str:
    return os.path.join(_nba_app_dir(), "leagues", f"{league_id}.yaml")


def load_league_config(league_id: str = "nba", *, use_cache: bool = True) -> LeagueConfig:
    league_id = (league_id or "nba").strip().lower()
    if use_cache and league_id in _CACHE:
        return _CACHE[league_id]

    path = get_league_config_path(league_id)
    raw = _load_yaml(path)
    cfg = LeagueConfig(raw=raw, config_path=path)

    # Trigger validation early by reading required properties
    _ = cfg.league_id
    _ = cfg.display_name
    _ = cfg.collections
    _ = cfg.master_training_csv
    # ESPN endpoints are validated when requested; validate base required keys exist:
    _ = cfg.espn_endpoint("scoreboard_header_template")
    _ = cfg.espn_endpoint("scoreboard_site_template")
    _ = cfg.espn_endpoint("game_summary_template")

    if use_cache:
        _CACHE[league_id] = cfg
    return cfg

