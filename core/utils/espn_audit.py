"""
ESPN <-> Mongo audit utilities.

Used by:
- tests/test_espn_db_audit_games.py (integration audit script)
- web UI (Data Caches -> ESPN DB Audit)

The goal is to compare "games ESPN says exist for a team/season/date-range" with
"games we have stored in Mongo" and report missing/extra game_ids, plus record
counts where possible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests

from bball_app.core.data import ESPNClient, GamesRepository
from bball_app.core.league_config import LeagueConfig


NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


ProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class EspnTeamGame:
    game_id: str
    et_date: str  # YYYY-MM-DD
    scoreboard_query_date: str  # YYYY-MM-DD (the YYYYMMDD used on the scoreboard endpoint)
    utc_datetime: Optional[str]  # raw ESPN UTC datetime string if present
    home: str
    away: str
    completed: bool
    winner: Optional[str]  # 'HOME' | 'AWAY' | None
    season: str  # 'YYYY-YYYY'
    season_type: Optional[int]  # 1 preseason, 2 reg, 3 playoffs (best effort)


def _season_from_et_date(d: date, league: Optional[LeagueConfig] = None) -> str:
    """Get season string from date using league config's season_cutover_month."""
    cutover_month = league.season_rules.get('season_cutover_month', 8) if league else 8
    return f"{d.year}-{d.year + 1}" if d.month > cutover_month else f"{d.year - 1}-{d.year}"


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_season_type_from_scoreboard_event(event: Dict[str, Any]) -> Optional[int]:
    """
    Best-effort extraction of season type (1 preseason / 2 reg / 3 playoffs).
    ESPN response shape differs across endpoints/years, so we check multiple places.
    """
    season_obj = event.get("season")
    if isinstance(season_obj, dict):
        t = _safe_int(season_obj.get("type"))
        if t in (1, 2, 3):
            return t

    for k in ("seasonType", "seasonTypeId"):
        t = _safe_int(event.get(k))
        if t in (1, 2, 3):
            return t

    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        comp = competitions[0]
        season_obj = comp.get("season")
        if isinstance(season_obj, dict):
            t = _safe_int(season_obj.get("type"))
            if t in (1, 2, 3):
                return t

    return None


def _extract_event_competitors(event: Dict[str, Any]) -> List[Dict[str, Any]]:
    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        return competitions[0].get("competitors") or []
    return event.get("competitors") or []


def _extract_event_status_completed(event: Dict[str, Any]) -> bool:
    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        comp = competitions[0]
        status = comp.get("status") or {}
        status_type = status.get("type") or {}
        completed = status_type.get("completed")
        if isinstance(completed, bool):
            return completed

    status = event.get("status") or {}
    status_type = status.get("type") or {}
    completed = status_type.get("completed")
    if isinstance(completed, bool):
        return completed

    return False


def _extract_winner_side(event: Dict[str, Any]) -> Optional[str]:
    for comp in _extract_event_competitors(event):
        if not isinstance(comp, dict):
            continue
        if comp.get("winner") is True:
            side = comp.get("homeAway")
            if side == "home":
                return "HOME"
            if side == "away":
                return "AWAY"
    return None


def _extract_home_away_abbrevs(event: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    home = None
    away = None
    for comp in _extract_event_competitors(event):
        if not isinstance(comp, dict):
            continue
        side = comp.get("homeAway")
        team = comp.get("team") or {}
        abbr = (team.get("abbreviation") or "").upper()
        if not abbr:
            continue
        if side == "home":
            home = abbr
        elif side == "away":
            away = abbr
    if home and away:
        return home, away
    return None


def fetch_espn_games_for_date_range(
    start_date: date,
    end_date: date,
    timeout_s: int = 30,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, EspnTeamGame]:
    """
    Fetch ALL ESPN games for a date range (all teams).

    This is more efficient than calling fetch_espn_games_for_team() multiple times
    since the ESPN scoreboard API returns all games for a date, not just one team's.

    Returns:
        Dict mapping game_id -> EspnTeamGame for ALL games in the date range.
    """
    all_games: Dict[str, EspnTeamGame] = {}

    total_days = max(1, (end_date - start_date).days + 1)
    for idx, d in enumerate(_iter_dates(start_date, end_date), start=1):
        if progress_callback:
            pct = int(5 + (70 * (idx - 1) / total_days))
            progress_callback(pct, f"Fetching ESPN scoreboard for {d}…")

        url = f"{NBA_SCOREBOARD_URL}?dates={d.strftime('%Y%m%d')}"
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_s)
        resp.raise_for_status()
        scoreboard = resp.json()
        events = scoreboard.get("events") or []

        for event in events:
            if not isinstance(event, dict):
                continue
            game_id = event.get("id")
            if not game_id:
                continue

            ha = _extract_home_away_abbrevs(event)
            if not ha:
                continue
            home, away = ha

            completed = _extract_event_status_completed(event)

            event_date_str = event.get("date")
            et_dt = ESPNClient.parse_espn_utc_to_eastern(event_date_str) if event_date_str else None
            if et_dt is None:
                competitions = event.get("competitions") or []
                if competitions and isinstance(competitions[0], dict):
                    comp_date = competitions[0].get("date")
                    et_dt = ESPNClient.parse_espn_utc_to_eastern(comp_date) if comp_date else None

            et_date = et_dt.date() if et_dt else d
            event_season = _season_from_et_date(et_date)

            all_games[str(game_id)] = EspnTeamGame(
                game_id=str(game_id),
                et_date=et_date.strftime("%Y-%m-%d"),
                scoreboard_query_date=d.strftime("%Y-%m-%d"),
                utc_datetime=str(event_date_str) if event_date_str else None,
                home=home,
                away=away,
                completed=completed,
                winner=_extract_winner_side(event),
                season=event_season,
                season_type=_extract_season_type_from_scoreboard_event(event),
            )

    if progress_callback:
        progress_callback(75, f"Finished ESPN scan: {len(all_games)} games found.")

    return all_games


def filter_games_for_team(
    all_games: Dict[str, EspnTeamGame],
    team: str,
    season: str,
    completed_only: bool,
) -> Dict[str, EspnTeamGame]:
    """
    Filter a pre-fetched games dict to only games for a specific team/season.

    This is used after fetch_espn_games_for_date_range() to efficiently
    extract one team's games without additional API calls.
    """
    team = team.upper().strip()
    filtered: Dict[str, EspnTeamGame] = {}

    for game_id, game in all_games.items():
        if team not in (game.home, game.away):
            continue
        if game.season != season:
            continue
        if completed_only and not game.completed:
            continue
        filtered[game_id] = game

    return filtered


def fetch_espn_games_for_team(
    team: str,
    season: str,
    start_date: date,
    end_date: date,
    completed_only: bool,
    timeout_s: int = 30,
    progress_callback: Optional[ProgressCallback] = None,
    prefetched_games: Optional[Dict[str, EspnTeamGame]] = None,
) -> Dict[str, EspnTeamGame]:
    """
    Fetch ESPN games for a specific team.

    Args:
        prefetched_games: If provided, filter from this cache instead of making API calls.
                         This is much faster when auditing multiple teams.
    """
    team = team.upper().strip()

    # OPTIMIZED PATH: Use pre-fetched games cache
    if prefetched_games is not None:
        if progress_callback:
            progress_callback(75, f"Filtering {len(prefetched_games)} cached games for {team}…")
        return filter_games_for_team(prefetched_games, team, season, completed_only)

    # ORIGINAL PATH: Fetch from ESPN API (for single-team audits)
    expected: Dict[str, EspnTeamGame] = {}

    total_days = max(1, (end_date - start_date).days + 1)
    for idx, d in enumerate(_iter_dates(start_date, end_date), start=1):
        if progress_callback:
            pct = int(5 + (70 * (idx - 1) / total_days))
            progress_callback(pct, f"Fetching ESPN scoreboard for {d}…")

        url = f"{NBA_SCOREBOARD_URL}?dates={d.strftime('%Y%m%d')}"
        resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout_s)
        resp.raise_for_status()
        scoreboard = resp.json()
        events = scoreboard.get("events") or []

        for event in events:
            if not isinstance(event, dict):
                continue
            game_id = event.get("id")
            if not game_id:
                continue

            ha = _extract_home_away_abbrevs(event)
            if not ha:
                continue
            home, away = ha
            if team not in (home, away):
                continue

            completed = _extract_event_status_completed(event)
            if completed_only and not completed:
                continue

            event_date_str = event.get("date")
            et_dt = ESPNClient.parse_espn_utc_to_eastern(event_date_str) if event_date_str else None
            if et_dt is None:
                competitions = event.get("competitions") or []
                if competitions and isinstance(competitions[0], dict):
                    comp_date = competitions[0].get("date")
                    et_dt = ESPNClient.parse_espn_utc_to_eastern(comp_date) if comp_date else None

            et_date = et_dt.date() if et_dt else d
            event_season = _season_from_et_date(et_date)
            if event_season != season:
                continue

            expected[str(game_id)] = EspnTeamGame(
                game_id=str(game_id),
                et_date=et_date.strftime("%Y-%m-%d"),
                scoreboard_query_date=d.strftime("%Y-%m-%d"),
                utc_datetime=str(event_date_str) if event_date_str else None,
                home=home,
                away=away,
                completed=completed,
                winner=_extract_winner_side(event),
                season=event_season,
                season_type=_extract_season_type_from_scoreboard_event(event),
            )

    if progress_callback:
        progress_callback(75, "Finished ESPN scan.")

    return expected


def fetch_db_games_for_team(
    db,
    team: str,
    season: str,
    completed_only: bool,
) -> Dict[str, Dict[str, Any]]:
    repo = GamesRepository(db)
    games = repo.find_by_season(season=season, team=team.upper().strip(), completed_only=completed_only)
    return {str(g.get("game_id")): g for g in games if g.get("game_id")}


def compute_db_record(team: str, games: List[Dict[str, Any]]) -> Tuple[int, int]:
    team = team.upper().strip()
    wins = 0
    losses = 0
    for g in games:
        home = (((g.get("homeTeam") or {}).get("name")) or "").upper()
        away = (((g.get("awayTeam") or {}).get("name")) or "").upper()
        home_won = g.get("homeWon")
        if home_won is None:
            continue
        if team == home:
            wins += 1 if home_won else 0
            losses += 0 if home_won else 1
        elif team == away:
            wins += 0 if home_won else 1
            losses += 1 if home_won else 0
    return wins, losses


def compute_espn_record(team: str, games: List[EspnTeamGame]) -> Tuple[int, int]:
    team = team.upper().strip()
    wins = 0
    losses = 0
    for g in games:
        if not g.completed:
            continue
        if team not in (g.home, g.away):
            continue
        if g.winner is None:
            continue
        if g.winner == "HOME":
            wins += 1 if team == g.home else 0
            losses += 1 if team == g.away else 0
        elif g.winner == "AWAY":
            wins += 1 if team == g.away else 0
            losses += 1 if team == g.home else 0
    return wins, losses


def run_espn_vs_db_audit(
    *,
    db,
    team: str,
    season: str,
    start_date: date,
    end_date: date,
    completed_only: bool = True,
    timeout_s: int = 30,
    progress_callback: Optional[ProgressCallback] = None,
    prefetched_games: Optional[Dict[str, EspnTeamGame]] = None,
) -> Dict[str, Any]:
    """
    Returns a JSON-serializable result dict.

    Args:
        prefetched_games: If provided, use this cache instead of fetching from ESPN API.
                         This enables batch mode where we fetch all games once upfront.
    """
    team = team.upper().strip()
    season = season.strip()

    espn_map = fetch_espn_games_for_team(
        team=team,
        season=season,
        start_date=start_date,
        end_date=end_date,
        completed_only=completed_only,
        timeout_s=timeout_s,
        progress_callback=progress_callback,
        prefetched_games=prefetched_games,
    )
    db_map = fetch_db_games_for_team(db=db, team=team, season=season, completed_only=completed_only)

    espn_games = list(espn_map.values())
    db_games = list(db_map.values())

    espn_w, espn_l = compute_espn_record(team, espn_games)
    db_w, db_l = compute_db_record(team, db_games)

    espn_ids: Set[str] = set(espn_map.keys())
    db_ids: Set[str] = set(db_map.keys())
    missing_in_db = sorted(espn_ids - db_ids)
    extra_in_db = sorted(db_ids - espn_ids)

    date_mismatch_like_pull_script = sum(1 for g in espn_games if g.scoreboard_query_date != g.et_date)

    missing_details = []
    for gid in missing_in_db:
        g = espn_map[gid]
        missing_details.append(
            {
                "game_id": gid,
                "et_date": g.et_date,
                "scoreboard_query_date": g.scoreboard_query_date,
                "utc_datetime": g.utc_datetime,
                "matchup": f"{g.away}@{g.home}",
                "home": g.home,
                "away": g.away,
                "completed": g.completed,
                "winner": g.winner,
                "season_type": g.season_type,
            }
        )

    extra_details = []
    for gid in extra_in_db:
        g = db_map.get(gid) or {}
        extra_details.append(
            {
                "game_id": gid,
                "date": g.get("date"),
                "season": g.get("season"),
                "game_type": g.get("game_type"),
                "matchup": f"{(g.get('awayTeam') or {}).get('name')}@{(g.get('homeTeam') or {}).get('name')}",
            }
        )

    result: Dict[str, Any] = {
        "params": {
            "team": team,
            "season": season,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "completed_only": bool(completed_only),
        },
        "summary": {
            "espn_games": len(espn_games),
            "db_games": len(db_games),
            "espn_record": f"{espn_w}-{espn_l}",
            "db_record": f"{db_w}-{db_l}",
            "missing_in_db": len(missing_in_db),
            "extra_in_db": len(extra_in_db),
            "espn_scoreboard_query_date_ne_et_date": int(date_mismatch_like_pull_script),
        },
        "missing_in_db": missing_details,
        "extra_in_db": extra_details,
    }

    if progress_callback:
        progress_callback(95, f"Computed diff: missing={len(missing_in_db)} extra={len(extra_in_db)}.")

    return result

