"""Helper functions for the basketball web blueprint.

Extracted from web/app.py so the blueprint routes stay thin.
"""

from datetime import date, datetime
from typing import Dict, List

from flask import g


# ESPN team ID -> abbreviation mapping for NBA
TEAM_ABBREV_MAP = {
    '1': 'ATL', '2': 'BOS', '3': 'BKN', '4': 'CHA', '5': 'CHI',
    '6': 'CLE', '7': 'DAL', '8': 'DEN', '9': 'DET', '10': 'GS',
    '11': 'HOU', '12': 'IND', '13': 'LAC', '14': 'LAL', '15': 'MEM',
    '16': 'MIA', '17': 'MIL', '18': 'MIN', '19': 'NO', '20': 'NY',
    '21': 'OKC', '22': 'ORL', '23': 'PHI', '24': 'PHX', '25': 'POR',
    '26': 'SAC', '27': 'SA', '28': 'TOR', '29': 'UTA', '30': 'WAS',
}


def get_master_training_path() -> str:
    """Get master training CSV path from current league config."""
    league = getattr(g, "league", None)
    if league:
        return league.master_training_csv
    from bball.league_config import load_league_config
    return load_league_config("nba").master_training_csv


def get_season_from_date(game_date, league=None):
    """Proxy to core util, accepting optional league."""
    from bball.utils import get_season_from_date as _core
    if league:
        return _core(game_date, league=league)
    return _core(game_date)


def get_logo_url(team_data: dict) -> str | None:
    """Extract logo URL from team data, handling both 'logo' field and 'logos' array."""
    if team_data.get("logo"):
        return team_data["logo"]
    logos = team_data.get("logos", [])
    if isinstance(logos, list) and len(logos) > 0:
        first_logo = logos[0]
        if isinstance(first_logo, dict) and first_logo.get("href"):
            return first_logo["href"]
    return None


def get_position_sort_order(pos_name: str) -> int:
    """Get sort order: 0=Guard, 1=Forward, 2=Center, 3=unknown."""
    if not pos_name:
        return 3
    pos_lower = pos_name.lower()
    if "guard" in pos_lower:
        return 0
    elif "forward" in pos_lower:
        return 1
    elif "center" in pos_lower:
        return 2
    return 3


def get_position_abbreviation(pos_display_name: str) -> str:
    """Get position abbreviation: G, F, or C."""
    if not pos_display_name:
        return ""
    pos_lower = pos_display_name.lower()
    if "guard" in pos_lower:
        return "G"
    elif "forward" in pos_lower:
        return "F"
    elif "center" in pos_lower:
        return "C"
    return ""


def calculate_player_stats(db, player_id: str, team: str, season: str, before_date: str) -> Dict:
    """Calculate per-game stats for a player from player_stats collection."""
    player_games = list(db[g.league.collections.get("player_stats", "nba_player_stats")].find(
        {
            "player_id": player_id,
            "team": team,
            "season": season,
            "date": {"$lt": before_date},
            "stats.min": {"$gt": 0},
        },
        {"stats.pts": 1, "stats.ast": 1, "stats.reb": 1, "stats.min": 1},
    ))

    if not player_games:
        return {"gp": 0, "ppg": 0.0, "apg": 0.0, "rpg": 0.0, "mpg": 0.0, "per": 0.0}

    total_pts = sum(g_.get("stats", {}).get("pts", 0) for g_ in player_games)
    total_ast = sum(g_.get("stats", {}).get("ast", 0) for g_ in player_games)
    total_reb = sum(g_.get("stats", {}).get("reb", 0) for g_ in player_games)
    total_min = sum(g_.get("stats", {}).get("min", 0) for g_ in player_games)
    gp = len(player_games)

    ppg = round(total_pts / gp, 1) if gp > 0 else 0.0
    apg = round(total_ast / gp, 1) if gp > 0 else 0.0
    rpg = round(total_reb / gp, 1) if gp > 0 else 0.0
    mpg = round(total_min / gp, 1) if gp > 0 else 0.0

    per = round((ppg + rpg + apg) / mpg * 15, 1) if mpg > 0 else 0.0

    return {
        "gp": gp, "ppg": ppg, "apg": apg, "rpg": rpg,
        "mpg": mpg, "per": per, "min": total_min,
    }


def get_team_players_for_game(db, team: str, season: str, game_date: date) -> List[Dict]:
    """Get players for a team from rosters collection with stats."""
    before_date = game_date.strftime("%Y-%m-%d")

    rosters_collection = g.league.collections.get("rosters", "nba_rosters")
    roster_doc = db[rosters_collection].find_one({"season": season, "team": team})
    if not roster_doc:
        return []

    roster = roster_doc.get("roster", [])
    if not roster:
        return []

    player_ids = [str(p["player_id"]) for p in roster]

    players = list(db[g.league.collections.get("players", "nba_players")].find(
        {"player_id": {"$in": player_ids}},
        {
            "player_id": 1, "player_name": 1, "headshot": 1,
            "pos_name": 1, "pos_display_name": 1,
            "injured": 1, "injury_status": 1,
            "injury_date": 1, "injury_details": 1,
        },
    ))

    roster_map = {str(p["player_id"]): p for p in roster}

    valid_players = []
    for player in players:
        player_name = player.get("player_name")
        if not player_name or not player_name.strip():
            continue

        player_id = str(player.get("player_id", ""))
        roster_entry = roster_map.get(player_id, {})
        is_starter = roster_entry.get("starter", False)
        is_injured = roster_entry.get("injured", False)

        stats = calculate_player_stats(db, player_id, team, season, before_date)
        mpg = stats.get("mpg", 0.0)
        total_minutes = stats.get("min", 0)
        if mpg < 5.0 or total_minutes == 0:
            continue

        valid_players.append({
            "player_id": player_id,
            "player_name": player_name,
            "headshot": player.get("headshot"),
            "pos_name": player.get("pos_name"),
            "pos_display_name": player.get("pos_display_name"),
            "injured": is_injured,
            "injury_status": player.get("injury_status"),
            "was_starter": is_starter,
            "stats": stats,
        })

    valid_players.sort(key=lambda p: (
        not p.get("was_starter", False),
        get_position_sort_order(p.get("pos_name", "")),
        p.get("player_name", ""),
    ))

    return valid_players


def get_player_game_status(db, game_id: str) -> Dict:
    """Get player game status from database."""
    status_docs = list(db.player_game_status.find({"game_id": game_id}))
    status_dict = {}
    for doc in status_docs:
        key = f"{doc['team']}:{doc['player_id']}"
        status_dict[key] = {
            "is_playing": doc.get("is_playing", True),
            "is_starter": doc.get("is_starter", False),
        }
    return status_dict


def calculate_team_records(db, team: str, season: str, game_date: date, league=None) -> Dict:
    """Calculate team W-L records (overall, home, away, last 10)."""
    before_date = game_date.strftime("%Y-%m-%d")

    games_collection = "stats_nba"
    exclude_game_types = ["preseason", "allstar"]
    if league:
        games_collection = league.collections.get("games", "stats_nba")
        exclude_game_types = getattr(league, "exclude_game_types", exclude_game_types)

    games = list(db[games_collection].find(
        {
            "season": season,
            "date": {"$lt": before_date},
            "$or": [{"homeTeam.name": team}, {"awayTeam.name": team}],
            "homeTeam.points": {"$gt": 0},
            "awayTeam.points": {"$gt": 0},
            "game_type": {"$nin": exclude_game_types},
        },
        {"homeTeam.name": 1, "awayTeam.name": 1, "homeWon": 1, "date": 1},
    ))

    empty = {"wins": 0, "losses": 0, "record": "0-0"}
    if not games:
        return {"overall": dict(empty), "home": dict(empty), "away": dict(empty), "last10": dict(empty)}

    games.sort(key=lambda g_: g_.get("date", ""), reverse=True)
    last_10 = games[:10]

    ow, ol, hw, hl, aw, al, lw, ll = 0, 0, 0, 0, 0, 0, 0, 0
    for gm in games:
        is_home = gm.get("homeTeam", {}).get("name") == team
        won = (is_home and gm.get("homeWon", False)) or (not is_home and not gm.get("homeWon", False))
        if won:
            ow += 1
        else:
            ol += 1
        if is_home:
            if won: hw += 1
            else: hl += 1
        else:
            if won: aw += 1
            else: al += 1

    for gm in last_10:
        is_home = gm.get("homeTeam", {}).get("name") == team
        won = (is_home and gm.get("homeWon", False)) or (not is_home and not gm.get("homeWon", False))
        if won: lw += 1
        else: ll += 1

    return {
        "overall": {"wins": ow, "losses": ol, "record": f"{ow}-{ol}"},
        "home": {"wins": hw, "losses": hl, "record": f"{hw}-{hl}"},
        "away": {"wins": aw, "losses": al, "record": f"{aw}-{al}"},
        "last10": {"wins": lw, "losses": ll, "record": f"{lw}-{ll}"},
    }


def extract_and_update_teams(db, game_summary: Dict, teams_collection: str = "nba_teams"):
    """Extract team info from ESPN game summary and upsert to teams collection."""
    header = game_summary.get("header", {})
    competitions = header.get("competitions", [])
    if not competitions:
        return

    for competition in competitions:
        competitors = competition.get("competitors", [])
        for competitor in competitors:
            team = competitor.get("team", {})
            team_id = team.get("id")
            if not team_id:
                continue

            update_data = {"team_id": str(team_id), "last_update": datetime.utcnow()}

            for key, value in team.items():
                if value is not None and value != "" and value is not False:
                    if key == "logos" and isinstance(value, list) and len(value) > 0:
                        first_logo = value[0]
                        if isinstance(first_logo, dict) and first_logo.get("href"):
                            update_data["logo"] = first_logo["href"]
                        update_data["logos"] = value
                    else:
                        update_data[key] = value

            if "abbreviation" in update_data and update_data["abbreviation"]:
                update_data["abbreviation"] = str(update_data["abbreviation"]).upper()

            if "logo" not in update_data and "logos" in update_data:
                logos = update_data.get("logos", [])
                if isinstance(logos, list) and len(logos) > 0:
                    first_logo = logos[0]
                    if isinstance(first_logo, dict) and first_logo.get("href"):
                        update_data["logo"] = first_logo["href"]

            db[teams_collection].update_one(
                {"team_id": str(team_id)}, {"$set": update_data}, upsert=True,
            )


def extract_and_update_player_roster(db, game_summary: Dict, home_team: str, away_team: str):
    """Extract player roster from ESPN game summary and update players collection."""
    boxscore = game_summary.get("boxscore", {})
    players = boxscore.get("players", [])
    if not players:
        return

    header = game_summary.get("header", {})
    competitors = header.get("competitions", [{}])[0].get("competitors", [])
    home_team_id = None
    away_team_id = None
    for comp in competitors:
        comp_team = comp.get("team", {})
        comp_team_id = comp_team.get("id")
        if comp.get("homeAway") == "home":
            home_team_id = comp_team_id
        else:
            away_team_id = comp_team_id

    players_coll = g.league.collections.get("players", "nba_players")

    for team_players in players:
        team_info = team_players.get("team", {})
        team_id = team_info.get("id")

        matching_team = None
        if team_id == home_team_id:
            matching_team = home_team
        elif team_id == away_team_id:
            matching_team = away_team
        else:
            team_abbrev_api = team_info.get("abbreviation", "").upper()
            team_abbrev = TEAM_ABBREV_MAP.get(str(team_id)) or team_abbrev_api
            for game_team in [home_team, away_team]:
                if (team_abbrev == game_team or team_abbrev_api == game_team
                        or team_abbrev.upper() == game_team.upper()
                        or team_abbrev_api.upper() == game_team.upper()):
                    matching_team = game_team
                    break

        if not matching_team:
            continue

        statistics = team_players.get("statistics", [])
        for stat_group in statistics:
            athletes = stat_group.get("athletes", [])
            for athlete_data in athletes:
                athlete = athlete_data.get("athlete", {})
                player_id = athlete.get("id")
                if not player_id:
                    continue

                update_data = {"team": matching_team, "last_roster_update": datetime.utcnow()}

                if athlete.get("headshot"):
                    headshot_url = athlete["headshot"].get("href")
                    if headshot_url:
                        update_data["headshot"] = headshot_url
                if athlete.get("displayName"):
                    update_data["player_name"] = athlete["displayName"]
                if athlete.get("shortName"):
                    update_data["short_name"] = athlete["shortName"]
                if athlete.get("position"):
                    pos = athlete["position"]
                    if pos.get("displayName"):
                        update_data["pos_display_name"] = pos["displayName"]
                    if pos.get("name"):
                        update_data["pos_name"] = pos["name"]
                    if pos.get("abbreviation"):
                        update_data["pos_abbreviation"] = pos["abbreviation"]
                if athlete.get("jersey"):
                    update_data["jersey"] = athlete["jersey"]
                if "starter" in athlete_data:
                    update_data["is_starter"] = athlete_data.get("starter", False)

                db[players_coll].update_one(
                    {"player_id": str(player_id)}, {"$set": update_data}, upsert=True,
                )


def extract_and_update_injuries(db, game_summary: Dict, home_team: str, away_team: str):
    """Extract injury data from ESPN game summary and update players collection."""
    injuries = game_summary.get("injuries", [])
    if not injuries:
        return

    players_coll = g.league.collections.get("players", "nba_players")

    # Reset injury statuses for these teams
    db[players_coll].update_many(
        {"team": {"$in": [home_team, away_team]}},
        {"$set": {"injured": False, "injury_status": None}},
    )

    for team_injury_group in injuries:
        team_info = team_injury_group.get("team", {})
        team_abbrev_api = team_info.get("abbreviation", "").upper()
        team_id = team_info.get("id")
        team_abbrev = TEAM_ABBREV_MAP.get(str(team_id)) or team_abbrev_api

        matching_team = None
        for game_team in [home_team, away_team]:
            if (team_abbrev == game_team or team_abbrev_api == game_team
                    or team_abbrev.upper() == game_team.upper()
                    or team_abbrev_api.upper() == game_team.upper()):
                matching_team = game_team
                break

        if not matching_team:
            continue

        injury_list = team_injury_group.get("injuries", [])
        for injury in injury_list:
            athlete = injury.get("athlete", {})
            player_id = athlete.get("id")
            status = injury.get("status", "")
            if not player_id:
                continue

            details = injury.get("details", {})
            fantasy_status = details.get("fantasyStatus", {})
            fantasy_desc = fantasy_status.get("description", "").upper()

            if status.lower() == "out":
                injury_status = "Out"
                is_injured = True
            elif fantasy_desc == "GTD" or "GTD" in fantasy_desc:
                injury_status = "GTD"
                is_injured = False
            else:
                injury_status = status
                is_injured = False

            update_data = {
                "injured": is_injured,
                "injury_status": injury_status,
                "injury_date": injury.get("date"),
                "injury_details": details,
                "team": matching_team,
                "last_injury_update": datetime.utcnow(),
            }

            headshot_url = None
            if athlete.get("headshot"):
                headshot_url = athlete["headshot"].get("href")
            if headshot_url:
                update_data["headshot"] = headshot_url
            if athlete.get("displayName"):
                update_data["player_name"] = athlete["displayName"]
            if athlete.get("shortName"):
                update_data["short_name"] = athlete["shortName"]
            if athlete.get("position"):
                pos = athlete["position"]
                if pos.get("displayName"):
                    update_data["pos_display_name"] = pos["displayName"]
                if pos.get("name"):
                    update_data["pos_name"] = pos["name"]

            db[players_coll].update_one(
                {"player_id": str(player_id)}, {"$set": update_data}, upsert=True,
            )
