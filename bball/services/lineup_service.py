"""
Lineup tools for matchup agents.

Goal: return starters/bench/injured in the shape used by stats + research agents:
{
  "starters": [{id, name, pos}, ...],
  "bench": [{id, name, pos}, ...],
  "injured": [{id, name, pos}, ...]
}

This is intended to be sourced from the same roster/inactives source used by
prediction (rosters collection + players lookup).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from bball.mongo import Mongo
from bball.utils import get_season_from_date


def _get_coll(league, key: str, fallback: str) -> str:
    if league is not None and getattr(league, "collections", None):
        return league.collections.get(key) or fallback
    return fallback


def _team_abbrev_from_team_id(db, league, team_id: str) -> Optional[str]:
    teams_coll = _get_coll(league, "teams", "teams_nba")
    doc = db[teams_coll].find_one({"team_id": team_id}) or db[teams_coll].find_one({"id": team_id}) or {}
    abbr = doc.get("abbreviation") or doc.get("abbr")
    return str(abbr) if abbr else None


def get_lineups(team_id: str, *, game_id: Optional[str] = None, db=None, league=None) -> Dict[str, List[Dict[str, Any]]]:
    if db is None:
        db = Mongo().db

    # Determine season from the game if possible
    season = None
    if game_id:
        games_coll = _get_coll(league, "games", "stats_nba")
        g = db[games_coll].find_one({"game_id": game_id}) or {}
        if g.get("season"):
            season = g.get("season")
        elif g.get("date"):
            try:
                season = get_season_from_date(str(g["date"])[:10])
            except Exception:
                season = None

    team_abbrev = _team_abbrev_from_team_id(db, league, str(team_id)) or str(team_id)

    rosters_coll = _get_coll(league, "rosters", "nba_rosters")
    roster_doc = None
    if season:
        roster_doc = db[rosters_coll].find_one({"season": season, "team": team_abbrev})
    if roster_doc is None:
        roster_doc = db[rosters_coll].find_one({"team": team_abbrev})

    roster_list = (roster_doc or {}).get("roster") or []

    players_coll = _get_coll(league, "players", "players_nba")
    player_ids = [str(p.get("player_id", "")) for p in roster_list if p.get("player_id") is not None]
    players = list(db[players_coll].find({"player_id": {"$in": player_ids}}, {"player_id": 1, "player_name": 1, "position": 1}))
    name_map = {str(p.get("player_id")): p.get("player_name") for p in players}
    pos_map = {str(p.get("player_id")): p.get("position") for p in players}

    starters: List[Dict[str, Any]] = []
    bench: List[Dict[str, Any]] = []
    injured: List[Dict[str, Any]] = []

    for entry in roster_list:
        pid = str(entry.get("player_id", ""))
        obj = {"id": pid, "name": name_map.get(pid, ""), "pos": pos_map.get(pid, "")}
        if entry.get("injured"):
            injured.append(obj)
        elif entry.get("starter"):
            starters.append(obj)
        else:
            bench.append(obj)

    return {"starters": starters, "bench": bench, "injured": injured}


def drop_player_from_roster(db, player_id: str, team: str, season: str,
                            league=None) -> Dict[str, Any]:
    """
    Remove a player from a team's roster entirely.

    Args:
        db: MongoDB database instance
        player_id: Player ID string
        team: Team identifier
        season: Season string
        league: LeagueConfig for collection names

    Returns:
        Dict with 'success' bool and 'message' (or 'error' string)
    """
    from bball.data.rosters import RostersRepository

    rosters_repo = RostersRepository(db, league=league)
    pid_str = str(player_id)

    rosters_repo.remove_player_from_roster(team, season, pid_str)

    return {'success': True, 'message': f'Player {pid_str} dropped from {team}'}


def auto_set_lineups(db, league_config, game_date_str: str) -> Dict[str, Any]:
    """
    Auto-set lineups for all teams playing on a given date based on their
    most recent previous game. Starters carry over, other players who played
    go to bench, and roster players who didn't play get marked injured.

    Args:
        db: MongoDB database instance
        league_config: LeagueConfig instance
        game_date_str: Date string in YYYY-MM-DD format

    Returns:
        Dict with teams_updated, teams_skipped, and per-team details
    """
    from bball.data.games import GamesRepository
    from bball.data.rosters import RostersRepository

    games_coll = league_config.collections["games"]
    player_stats_coll = league_config.collections["player_stats"]

    # 1. Find all games on the given date
    todays_games = list(db[games_coll].find({"date": game_date_str}))
    if not todays_games:
        return {"teams_updated": 0, "teams_skipped": 0, "details": []}

    # 2. Collect unique team abbreviations
    teams = set()
    for game in todays_games:
        home = (game.get("homeTeam") or {}).get("name")
        away = (game.get("awayTeam") or {}).get("name")
        if home:
            teams.add(home)
        if away:
            teams.add(away)

    # 3. Determine season from the date
    parsed_date = date.fromisoformat(game_date_str)
    season = get_season_from_date(parsed_date, league=league_config)

    games_repo = GamesRepository(db, league=league_config)
    rosters_repo = RostersRepository(db, league=league_config)

    teams_updated = 0
    teams_skipped = 0
    details = []

    for team in sorted(teams):
        # 4a. Find up to 3 most recent completed games before this date
        prev_games = games_repo.find_team_games(team, before_date=game_date_str, season=season, limit=3)
        if not prev_games:
            teams_skipped += 1
            continue

        n_prev_games = len(prev_games)

        # 4b. Track per-player activity across recent games
        # Most recent game determines starter/bench classification
        # Inactivity across multiple games determines injury status
        prev_starters = set()
        prev_bench = set()

        for i, prev_game in enumerate(prev_games):
            prev_game_id = prev_game.get("game_id")
            player_stats = list(db[player_stats_coll].find({
                "game_id": prev_game_id,
                "team": team,
            }))

            active_pids = set()
            for ps in player_stats:
                pid = str(ps.get("player_id", ""))
                if not pid:
                    continue
                if ps.get("didNotPlay"):
                    continue
                active_pids.add(pid)
                # Only use the most recent game for starter/bench classification
                if i == 0:
                    if ps.get("starter"):
                        prev_starters.add(pid)
                    else:
                        prev_bench.add(pid)

            # Store active sets for inactivity calculation below
            prev_games[i]["_active_pids"] = active_pids

        # 4c. Get current roster
        roster_doc = rosters_repo.find_roster(team, season)
        if not roster_doc or not roster_doc.get("roster"):
            teams_skipped += 1
            continue

        # 4d. Update each player's lineup flags
        # A player is marked injured only if inactive in >= 2 of the last N games
        updated_roster = []
        n_starters = 0
        n_bench = 0
        n_injured = 0

        for player in roster_doc["roster"]:
            pid = str(player.get("player_id", ""))
            if pid in prev_starters:
                player["starter"] = True
                player["injured"] = False
                n_starters += 1
            elif pid in prev_bench:
                player["starter"] = False
                player["injured"] = False
                n_bench += 1
            else:
                # Count how many of the recent games this player was inactive
                games_inactive = sum(
                    1 for pg in prev_games if pid not in pg.get("_active_pids", set())
                )
                if games_inactive >= 2 or n_prev_games == 1:
                    player["starter"] = False
                    player["injured"] = True
                    n_injured += 1
                else:
                    # Inactive in only 1 of 2-3 games — likely a rest day, not injured
                    player["starter"] = False
                    player["injured"] = False
                    n_bench += 1
            updated_roster.append(player)

        # 4e. Write updated roster back
        rosters_repo.update_one(
            {"team": team, "season": season},
            {"$set": {"roster": updated_roster, "updated_at": datetime.utcnow()}},
        )

        teams_updated += 1
        details.append({
            "team": team,
            "starters": n_starters,
            "bench": n_bench,
            "injured": n_injured,
        })

    return {
        "teams_updated": teams_updated,
        "teams_skipped": teams_skipped,
        "details": details,
    }

