#!/usr/bin/env python3
"""
Audit test: compare ESPN "source of truth" games vs MongoDB stored games.

Purpose
-------
When a team's record/count in the app doesn't match ESPN, this script identifies
which *specific games* are missing from the DB (by game_id) and prints enough
context (date/opponent/status) to debug the pull/aggregation logic.

This intentionally runs as a lightweight integration test (Mongo + live ESPN).

Usage
-----
source venv/bin/activate
PYTHONPATH=/Users/pranav/Documents/NBA \
  NBA_AUDIT_LIVE_ESPN=1 \
  NBA_AUDIT_TEAM=MIL \
  NBA_AUDIT_SEASON=2025-2026 \
  python tests/test_espn_db_audit_games.py

Optional environment variables
------------------------------
- NBA_AUDIT_START_DATE=YYYY-MM-DD   (default: Oct 1 of season start year)
- NBA_AUDIT_END_DATE=YYYY-MM-DD     (default: today)
- NBA_AUDIT_COMPLETED_ONLY=1|0      (default: 1)

Notes
-----
- This test is skipped unless NBA_AUDIT_LIVE_ESPN=1 is set.
- ESPN endpoint used: https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD
"""

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import requests


# Add project root to path (match other tests' convention)
script_dir = os.path.dirname(os.path.abspath(__file__))
nba_app_dir = os.path.dirname(script_dir)
project_root = os.path.dirname(nba_app_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from bball_app.core.mongo import Mongo
from bball_app.core.data import GamesRepository, ESPNClient
from bball_app.config import config as app_config


NBA_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}


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


def _parse_bool_env(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip() not in ("0", "false", "False", "")


def _season_from_et_date(d: date) -> str:
    return f"{d.year}-{d.year + 1}" if d.month > 8 else f"{d.year - 1}-{d.year}"


def _iter_dates(start: date, end: date) -> Iterable[date]:
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def _safe_int(x) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _extract_season_type_from_scoreboard_event(event: Dict) -> Optional[int]:
    """
    Best-effort extraction of season type (1 preseason / 2 reg / 3 playoffs).
    ESPN response shape differs across endpoints/years, so we check multiple places.
    """
    # Common: event["season"]["type"] on some endpoints
    season_obj = event.get("season")
    if isinstance(season_obj, dict):
        t = _safe_int(season_obj.get("type"))
        if t in (1, 2, 3):
            return t
    # Sometimes: event["seasonType"] or event["seasonTypeId"]
    for k in ("seasonType", "seasonTypeId"):
        t = _safe_int(event.get(k))
        if t in (1, 2, 3):
            return t
    # Sometimes within first competition
    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        comp = competitions[0]
        season_obj = comp.get("season")
        if isinstance(season_obj, dict):
            t = _safe_int(season_obj.get("type"))
            if t in (1, 2, 3):
                return t
    return None


def _extract_event_competitors(event: Dict) -> List[Dict]:
    competitions = event.get("competitions") or []
    if competitions and isinstance(competitions[0], dict):
        return competitions[0].get("competitors") or []
    # Some shapes may have competitors at top-level (rare)
    return event.get("competitors") or []


def _extract_event_status_completed(event: Dict) -> bool:
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


def _extract_winner_side(event: Dict) -> Optional[str]:
    """
    Returns 'HOME' or 'AWAY' if winner is present, else None.
    """
    competitors = _extract_event_competitors(event)
    for comp in competitors:
        if not isinstance(comp, dict):
            continue
        if comp.get("winner") is True:
            side = comp.get("homeAway")
            if side == "home":
                return "HOME"
            if side == "away":
                return "AWAY"
    return None


def _extract_home_away_abbrevs(event: Dict) -> Optional[Tuple[str, str]]:
    competitors = _extract_event_competitors(event)
    home = None
    away = None
    for comp in competitors:
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


def fetch_espn_games_for_team(
    team: str,
    season: str,
    start_date: date,
    end_date: date,
    completed_only: bool,
    timeout_s: int = 30,
) -> Dict[str, EspnTeamGame]:
    """
    Pulls games from ESPN scoreboard-by-date and returns a mapping of game_id -> EspnTeamGame.
    """
    team = team.upper().strip()
    expected: Dict[str, EspnTeamGame] = {}

    for d in _iter_dates(start_date, end_date):
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

            # Parse ESPN UTC datetime and derive ET date for season matching & debug printing.
            event_date_str = event.get("date")
            et_dt = ESPNClient.parse_espn_utc_to_eastern(event_date_str) if event_date_str else None
            # Fallback: attempt competition date if present
            if et_dt is None:
                competitions = event.get("competitions") or []
                if competitions and isinstance(competitions[0], dict):
                    comp_date = competitions[0].get("date")
                    et_dt = ESPNClient.parse_espn_utc_to_eastern(comp_date) if comp_date else None

            if et_dt is None:
                # Keep it, but mark date as unknown to avoid dropping games silently.
                # Use the queried date as a last-resort placeholder.
                et_date = d
            else:
                et_date = et_dt.date()

            event_season = _season_from_et_date(et_date)
            if event_season != season:
                continue

            season_type = _extract_season_type_from_scoreboard_event(event)
            winner_side = _extract_winner_side(event)

            expected[str(game_id)] = EspnTeamGame(
                game_id=str(game_id),
                et_date=et_date.strftime("%Y-%m-%d"),
                scoreboard_query_date=d.strftime("%Y-%m-%d"),
                utc_datetime=str(event_date_str) if event_date_str else None,
                home=home,
                away=away,
                completed=completed,
                winner=winner_side,
                season=event_season,
                season_type=season_type,
            )

    return expected


def fetch_db_games_for_team(
    db,
    team: str,
    season: str,
    completed_only: bool,
) -> Dict[str, Dict]:
    repo = GamesRepository(db)
    games = repo.find_by_season(season=season, team=team.upper().strip(), completed_only=completed_only)
    return {str(g.get("game_id")): g for g in games if g.get("game_id")}


def _compute_db_record(team: str, games: List[Dict]) -> Tuple[int, int]:
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


def _compute_espn_record(team: str, games: List[EspnTeamGame]) -> Tuple[int, int]:
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


def test_audit_espn_vs_db_team_season_games() -> List[Tuple[str, bool]]:
    print("\n" + "=" * 80)
    print("TEST: ESPN -> DB audit (team season games)")
    print("=" * 80)

    live = os.environ.get("NBA_AUDIT_LIVE_ESPN", "").strip() == "1"
    if not live:
        print("  SKIP: set NBA_AUDIT_LIVE_ESPN=1 to enable live ESPN audit")
        return [("audit_espn_vs_db_team_season_games", True)]

    team = os.environ.get("NBA_AUDIT_TEAM", "").strip().upper()
    season = os.environ.get("NBA_AUDIT_SEASON", "").strip()
    if not team or not season:
        print("  SKIP: set NBA_AUDIT_TEAM and NBA_AUDIT_SEASON (e.g., MIL and 2025-2026)")
        return [("audit_espn_vs_db_team_season_games", True)]

    completed_only = _parse_bool_env("NBA_AUDIT_COMPLETED_ONLY", True)

    try:
        start_year = int(season.split("-")[0])
    except Exception:
        raise ValueError(f"Invalid NBA_AUDIT_SEASON: {season} (expected YYYY-YYYY)")
    end_year = start_year + 1

    start_date_str = os.environ.get("NBA_AUDIT_START_DATE")
    end_date_str = os.environ.get("NBA_AUDIT_END_DATE")
    if start_date_str:
        start_d = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    else:
        # ESPN season start date is often Oct 1; this safely includes any early-season games.
        start_d = date(start_year, 10, 1)
    if end_date_str:
        end_d = datetime.strptime(end_date_str, "%Y-%m-%d").date()
    else:
        # Default to "end of season window" so past seasons don't scan into future years.
        # Using Sep 30 is a safe boundary that includes playoffs/Finals without requiring
        # us to model the exact end date each year.
        season_window_end = date(end_year, 9, 30)
        end_d = min(date.today(), season_window_end)

    print(f"  Team: {team}")
    print(f"  Season: {season}")
    print(f"  Date range: {start_d} -> {end_d}")
    print(f"  Completed only: {completed_only}")

    if not (app_config.get("mongo_conn_str") or "").strip():
        print("  SKIP: Mongo is not configured (set MONGO_CONN_STR or MONGODB_URI)")
        return [("audit_espn_vs_db_team_season_games", True)]

    mongo = Mongo()
    db = mongo.db

    # DB side
    db_games_map = fetch_db_games_for_team(db=db, team=team, season=season, completed_only=completed_only)
    db_games = list(db_games_map.values())
    db_w, db_l = _compute_db_record(team, db_games)

    # ESPN side
    espn_games_map = fetch_espn_games_for_team(
        team=team,
        season=season,
        start_date=start_d,
        end_date=end_d,
        completed_only=completed_only,
    )
    espn_games = list(espn_games_map.values())
    espn_w, espn_l = _compute_espn_record(team, espn_games)
    date_mismatch_like_pull_script = sum(
        1 for g in espn_games if g.scoreboard_query_date != g.et_date
    )

    print("\n  Counts / record")
    print(f"    ESPN games: {len(espn_games)}; record (best-effort): {espn_w}-{espn_l}")
    print(f"    ESPN games where scoreboard query date != ET date: {date_mismatch_like_pull_script}")
    print(f"    DB games:   {len(db_games)}; record:              {db_w}-{db_l}")

    espn_ids: Set[str] = set(espn_games_map.keys())
    db_ids: Set[str] = set(db_games_map.keys())

    missing_in_db = sorted(espn_ids - db_ids)
    extra_in_db = sorted(db_ids - espn_ids)

    # Print high-signal diffs
    if missing_in_db:
        print(f"\n  Missing in DB ({len(missing_in_db)}):")
        for gid in missing_in_db:
            g = espn_games_map[gid]
            matchup = f"{g.away}@{g.home}"
            st = f"seasonType={g.season_type}" if g.season_type is not None else "seasonType=?"
            utc = f"utc={g.utc_datetime}" if g.utc_datetime else "utc=?"
            qd = g.scoreboard_query_date
            print(
                f"    - et_date={g.et_date} (scoreboard_query_date={qd})  {matchup}  game_id={gid}  "
                f"completed={g.completed}  winner={g.winner}  {st}  {utc}"
            )

    if extra_in_db:
        print(f"\n  Extra in DB (not found in ESPN query) ({len(extra_in_db)}):")
        for gid in extra_in_db[:20]:
            g = db_games_map[gid]
            matchup = f"{(g.get('awayTeam') or {}).get('name')}@{(g.get('homeTeam') or {}).get('name')}"
            print(f"    - {g.get('date')}  {matchup}  game_id={gid}")
        if len(extra_in_db) > 20:
            print(f"    ... and {len(extra_in_db) - 20} more")

    is_valid = (len(missing_in_db) == 0 and len(extra_in_db) == 0)
    if is_valid:
        print("\n  PASS: ESPN and DB game sets match for this team/season/range")
    else:
        print("\n  FAIL: ESPN and DB game sets differ (see missing/extra above)")

    return [("audit_espn_vs_db_team_season_games", is_valid)]


def main() -> int:
    results = test_audit_espn_vs_db_team_season_games()
    failed = [name for name, ok in results if not ok]
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

