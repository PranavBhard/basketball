"""Basketball-specific Flask Blueprint for the unified sportscore web app.

Contains all routes that are specific to basketball:
  - Game list (index) page  /<league_id>/
  - Game detail page        /<league_id>/game/<game_id>
  - Injuries manager page   /<league_id>/injuries-manager
  - Player management APIs  (move, search, add, drop, update, etc.)
  - Prediction APIs         (predict, predict-all, betting-report, etc.)
  - Live games API          (ESPN scoreboard polling)
  - Portfolio positions API  (Kalshi fills/positions per game)
  - Matchup chat APIs       (AI-assisted matchup analysis)
"""

import logging
import os
import threading
from datetime import date, datetime, timedelta
from typing import Dict, List

from bson import ObjectId
from flask import Blueprint, g, jsonify, render_template, request, send_file
from pytz import timezone, utc

from .helpers import (
    calculate_player_stats,
    calculate_team_records,
    get_logo_url,
    get_master_training_path,
    get_player_game_status,
    get_position_abbreviation,
    get_position_sort_order,
    get_season_from_date,
    get_team_players_for_game,
)

logger = logging.getLogger(__name__)

bp = Blueprint(
    "basketball",
    __name__,
    template_folder="templates",
)


# ---------------------------------------------------------------------------
# Global caches (process-lifetime, same as original app.py)
# ---------------------------------------------------------------------------
_per_calculator = None
_per_calculator_league_id = None

# Matchup chat session storage (in-memory, per-process)
_matchup_agent_sessions: dict = {}
_session_lock = threading.Lock()


def _get_db():
    """Return the MongoDB database from the request context."""
    return g.db


def _get_per_calculator():
    """Lazy-create PERCalculator for the current league."""
    global _per_calculator, _per_calculator_league_id
    league = g.league
    lid = league.league_id
    if _per_calculator is None or _per_calculator_league_id != lid:
        from bball.stats.per_calculator import PERCalculator
        _per_calculator = PERCalculator(db=_get_db(), league=league)
        _per_calculator_league_id = lid
    return _per_calculator


# ======================================================================
#  INDEX (GAME LIST) PAGE
# ======================================================================

def index(league_id=None):
    """Game list page — shows today's games with scores, predictions, market data."""
    from bball.data.espn_client import ESPNClient

    db = _get_db()
    league_config = g.league

    # Determine date
    date_str = request.args.get("date")
    if date_str:
        try:
            game_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            game_date = date.today()
    else:
        game_date = date.today()

    date_str = game_date.strftime("%Y-%m-%d")
    date_yyyymmdd = date_str.replace("-", "")

    # ── Fetch from ESPN API ──
    games = []
    try:
        espn_url = league_config.espn_endpoint("scoreboard_site_template").format(YYYYMMDD=date_yyyymmdd)
    except Exception:
        espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_yyyymmdd}"

    espn_client = ESPNClient(league=league_config)
    games_collection = league_config.collections.get("games", "stats_nba")

    try:
        import requests as _requests
        response = _requests.get(espn_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        events = data.get("events", [])

        for event in events:
            game_id = event.get("id")
            if not game_id:
                continue

            competitions = event.get("competitions", [])
            if not competitions:
                continue

            competition = competitions[0]
            competitors = competition.get("competitors", [])

            home_team = away_team = ""
            home_team_obj = away_team_obj = {}
            for comp in competitors:
                team_data = comp.get("team", {})
                abbrev = team_data.get("abbreviation", "").upper()
                if comp.get("homeAway") == "home":
                    home_team = abbrev
                    home_team_obj = comp
                else:
                    away_team = abbrev
                    away_team_obj = comp

            if not home_team or not away_team:
                continue

            # Extract odds
            pregame_lines = None
            odds_list = competition.get("odds", [])
            if odds_list and isinstance(odds_list, list):
                odds = odds_list[0] if odds_list else {}
                spread = odds.get("spread")
                over_under = odds.get("overUnder")
                home_ml = odds.get("homeTeamOdds", {}).get("moneyLine")
                away_ml = odds.get("awayTeamOdds", {}).get("moneyLine")
                if spread is not None or over_under is not None:
                    pregame_lines = {
                        "spread": spread, "over_under": over_under,
                        "home_ml": home_ml, "away_ml": away_ml,
                    }

            # Gametime
            espn_date_str = competition.get("date", "")
            gametime = None
            if espn_date_str:
                try:
                    gametime = datetime.fromisoformat(espn_date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass

            # Game status
            status_obj = event.get("status", {})
            game_status = "pre"
            game_completed = False
            period = None
            clock = None
            if isinstance(status_obj, dict):
                status_type = status_obj.get("type", {})
                if isinstance(status_type, dict):
                    raw = status_type.get("name", "").lower()
                    game_completed = status_type.get("completed", False)
                    if game_completed or "final" in raw or "post" in raw:
                        game_status = "post"
                    elif "progress" in raw or "halftime" in raw or raw == "in":
                        game_status = "in"
                period = status_obj.get("period")
                clock = status_obj.get("displayClock")

            # Scores
            home_points = None
            away_points = None
            try:
                home_score_str = home_team_obj.get("score")
                away_score_str = away_team_obj.get("score")
                if home_score_str:
                    home_points = int(home_score_str)
                if away_score_str:
                    away_points = int(away_score_str)
            except (ValueError, TypeError):
                pass

            # Upsert game into DB
            try:
                update_doc = {
                    "game_id": game_id, "date": date_str,
                    "homeTeam.name": home_team, "awayTeam.name": away_team,
                }
                if gametime:
                    update_doc["gametime"] = gametime
                if pregame_lines:
                    update_doc["pregame_lines"] = pregame_lines
                if home_points is not None:
                    update_doc["homeTeam.points"] = home_points
                if away_points is not None:
                    update_doc["awayTeam.points"] = away_points

                venue = competition.get("venue", {})
                if venue:
                    update_doc["venue"] = {
                        "name": venue.get("fullName"),
                        "city": venue.get("address", {}).get("city"),
                        "state": venue.get("address", {}).get("state"),
                    }

                db[games_collection].update_one(
                    {"game_id": game_id}, {"$set": update_doc}, upsert=True,
                )
            except Exception as e:
                print(f"Error upserting game {game_id}: {e}")

            # Prediction
            predictions_collection = league_config.collections.get("model_predictions", "nba_model_predictions")
            prediction_doc = db[predictions_collection].find_one({"game_id": game_id})
            last_prediction = None
            if prediction_doc:
                last_prediction = {k: v for k, v in prediction_doc.items() if k != "_id"}

            # Injured player names
            home_injured_ids = home_team_obj.get("injuries", []) if isinstance(home_team_obj, dict) else []
            away_injured_ids = away_team_obj.get("injuries", []) if isinstance(away_team_obj, dict) else []
            # Look up from DB games doc instead (ESPN injuries field may not exist)
            game_doc = db[games_collection].find_one({"game_id": game_id})
            home_injured_names = []
            away_injured_names = []
            if game_doc:
                _lookup_injured = lambda ids_list: []
                h_ids = game_doc.get("homeTeam", {}).get("injured_players", [])
                a_ids = game_doc.get("awayTeam", {}).get("injured_players", [])
                players_coll = league_config.collections.get("players", "nba_players")
                if h_ids:
                    try:
                        home_injured_names = [
                            p.get("player_name", f"Player {p.get('player_id')}")
                            for p in db[players_coll].find({"player_id": {"$in": [str(i) for i in h_ids]}}, {"player_name": 1, "player_id": 1})
                        ]
                    except Exception:
                        pass
                if a_ids:
                    try:
                        away_injured_names = [
                            p.get("player_name", f"Player {p.get('player_id')}")
                            for p in db[players_coll].find({"player_id": {"$in": [str(i) for i in a_ids]}}, {"player_name": 1, "player_id": 1})
                        ]
                    except Exception:
                        pass

            games.append({
                "game_id": game_id, "home_team": home_team, "away_team": away_team,
                "date": date_str, "last_prediction": last_prediction,
                "pregame_lines": pregame_lines,
                "home_points": home_points, "away_points": away_points,
                "home_injured_players": home_injured_names,
                "away_injured_players": away_injured_names,
                "gametime": gametime,
                "status": game_status, "completed": game_completed,
                "period": period, "clock": clock,
            })

    except Exception:
        import traceback
        traceback.print_exc()
        # Fallback: load from DB
        try:
            db_games = list(db[games_collection].find(
                {"date": date_str},
                {"game_id": 1, "homeTeam": 1, "awayTeam": 1, "last_prediction": 1, "pregame_lines": 1, "gametime": 1},
            ).sort("gametime", 1))

            players_coll = league_config.collections.get("players", "nba_players")
            predictions_collection = league_config.collections.get("model_predictions", "nba_model_predictions")

            for game_doc in db_games:
                gid = game_doc.get("game_id")
                if not gid:
                    continue
                ht = game_doc.get("homeTeam", {}).get("name", "").upper()
                at = game_doc.get("awayTeam", {}).get("name", "").upper()
                if not ht or not at:
                    continue

                pred_doc = db[predictions_collection].find_one({"game_id": gid})
                lp = None
                if pred_doc:
                    lp = {k: v for k, v in pred_doc.items() if k != "_id"}

                hp = game_doc.get("homeTeam", {}).get("points")
                ap = game_doc.get("awayTeam", {}).get("points")
                gc = hp is not None and ap is not None

                games.append({
                    "game_id": gid, "home_team": ht, "away_team": at,
                    "date": date_str, "last_prediction": lp,
                    "pregame_lines": game_doc.get("pregame_lines"),
                    "home_points": hp, "away_points": ap,
                    "home_injured_players": [], "away_injured_players": [],
                    "gametime": game_doc.get("gametime"),
                    "status": "post" if gc else "pre", "completed": gc,
                    "period": None, "clock": None,
                })
        except Exception as e2:
            import traceback
            traceback.print_exc()

    # Team logos & colors
    teams_collection = league_config.collections.get("teams", "nba_teams")
    for game in games:
        htd = db[teams_collection].find_one({"abbreviation": game["home_team"]}) or {}
        atd = db[teams_collection].find_one({"abbreviation": game["away_team"]}) or {}
        game["home_team_logo"] = get_logo_url(htd)
        game["away_team_logo"] = get_logo_url(atd)
        game["home_team_color"] = htd.get("color", "667eea")
        game["home_team_alternate_color"] = htd.get("alternateColor", "764ba2")
        game["away_team_color"] = atd.get("color", "666666")
        game["away_team_alternate_color"] = atd.get("alternateColor", "999999")

    prev_date = (game_date - timedelta(days=1)).strftime("%Y-%m-%d")
    next_date = (game_date + timedelta(days=1)).strftime("%Y-%m-%d")
    games.sort(key=lambda g_: g_.get("gametime", "") or "")

    # Game-pull stats
    api_count = len(games)
    db_count = db[games_collection].count_documents({"date": date_str})
    total_games = max(api_count, db_count) if (api_count or db_count) else 0
    games_with_homewon = db[games_collection].count_documents({"date": date_str, "homeWon": {"$exists": True}})

    # Kalshi team-abbreviation maps
    kalshi_abbrev_map = league_config.raw.get("market", {}).get("team_abbrev_map", {})
    kalshi_reverse_map = {v: k for k, v in kalshi_abbrev_map.items()}

    # Chat message counts
    chat_message_counts = {}
    try:
        sessions_coll = league_config.collections.get("matchup_sessions", "nba_matchup_sessions")
        game_ids = [g_["game_id"] for g_ in games if g_.get("game_id")]
        if game_ids:
            pipeline = [
                {"$match": {"game_id": {"$in": game_ids}}},
                {"$project": {
                    "game_id": 1,
                    "user_message_count": {
                        "$size": {
                            "$filter": {
                                "input": {"$ifNull": ["$messages", []]},
                                "as": "msg",
                                "cond": {"$eq": ["$$msg.role", "user"]},
                            }
                        }
                    },
                }},
            ]
            for doc in db[sessions_coll].aggregate(pipeline):
                if doc.get("user_message_count", 0) > 0:
                    chat_message_counts[doc["game_id"]] = doc["user_message_count"]
    except Exception:
        pass

    return render_template(
        "basketball/game_list.html",
        games=games,
        game_date=game_date,
        prev_date=prev_date,
        next_date=next_date,
        games_pulled=games_with_homewon,
        total_games=total_games,
        kalshi_abbrev_map=kalshi_abbrev_map,
        kalshi_reverse_map=kalshi_reverse_map,
        chat_message_counts=chat_message_counts,
    )


# ======================================================================
#  GAME DETAIL PAGE
# ======================================================================

@bp.route("/<league_id>/game/<game_id>")
def game_detail(game_id, league_id=None):
    """Game detail page with player management."""
    db = _get_db()

    date_param = request.args.get("date")
    game_date = None
    if date_param:
        try:
            game_date = datetime.strptime(date_param, "%Y-%m-%d").date()
        except Exception:
            pass

    games_collection = g.league.collections.get("games", "stats_nba")
    game = db[games_collection].find_one({"game_id": game_id})

    home_team = away_team = ""

    if not game or not game.get("homeTeam") or not game.get("awayTeam"):
        from bball.data.espn_client import get_game_summary
        game_summary = get_game_summary(game_id)
        if not game_summary:
            return f"Game {game_id} not found", 404
        header = game_summary.get("header", {})
        competitors = header.get("competitions", [{}])[0].get("competitors", [])
        if len(competitors) != 2:
            return "Invalid game data", 404
        for comp in competitors:
            if comp.get("homeAway") == "home":
                home_team = comp.get("team", {}).get("abbreviation", "").upper()
            else:
                away_team = comp.get("team", {}).get("abbreviation", "").upper()
        if game_date is None:
            espn_date = header.get("competitions", [{}])[0].get("date", "")
            if espn_date:
                try:
                    game_date = datetime.strptime(espn_date[:10], "%Y-%m-%d").date()
                except Exception:
                    game_date = date.today()
            else:
                game_date = date.today()
    else:
        home_team = game.get("homeTeam", {}).get("name", "")
        away_team = game.get("awayTeam", {}).get("name", "")
        if not home_team or not away_team:
            from bball.data.espn_client import get_game_summary
            game_summary = get_game_summary(game_id)
            if game_summary:
                header = game_summary.get("header", {})
                for comp in header.get("competitions", [{}])[0].get("competitors", []):
                    if comp.get("homeAway") == "home":
                        home_team = comp.get("team", {}).get("abbreviation", "").upper()
                    else:
                        away_team = comp.get("team", {}).get("abbreviation", "").upper()
        if game_date is None:
            if game.get("date"):
                try:
                    game_date = datetime.strptime(game["date"], "%Y-%m-%d").date()
                except Exception:
                    game_date = date.today()
            else:
                game_date = date.today()

    season = get_season_from_date(game_date)

    home_players = get_team_players_for_game(db, home_team, season, game_date)
    away_players = get_team_players_for_game(db, away_team, season, game_date)
    game_status = get_player_game_status(db, game_id)

    # Merge player data with game status
    for player in home_players + away_players:
        pid = player["player_id"]
        team_for_key = home_team if player in home_players else away_team
        status = game_status.get(f"{team_for_key}:{pid}", {})
        player["is_playing"] = status.get("is_playing", True)
        player["is_starter"] = status.get("is_starter", player.get("was_starter", False))
        player["is_injured"] = player.get("injured", False)
        player["is_gtd"] = player.get("injury_status") == "GTD"

    def sort_key(p):
        if p.get("is_starter", False):
            return (0, get_position_sort_order(p.get("pos_name", "")), 0)
        return (1, 0, -p.get("stats", {}).get("mpg", 0.0))

    home_players.sort(key=sort_key)
    away_players.sort(key=sort_key)

    teams_collection = g.league.collections.get("teams", "nba_teams")
    htd = db[teams_collection].find_one({"abbreviation": home_team}) or {}
    atd = db[teams_collection].find_one({"abbreviation": away_team}) or {}

    home_records = calculate_team_records(db, home_team, season, game_date, league=g.league)
    away_records = calculate_team_records(db, away_team, season, game_date, league=g.league)

    # Store player IDs in games collection
    if game_id:
        try:
            db[games_collection].update_one(
                {"game_id": game_id},
                {"$set": {
                    "homeTeam.players": [str(p.get("player_id")) for p in home_players],
                    "awayTeam.players": [str(p.get("player_id")) for p in away_players],
                }},
            )
        except Exception:
            pass

    # Last prediction & pregame lines
    last_prediction = pregame_lines = None
    game_doc = db[games_collection].find_one({"game_id": game_id}, {"last_prediction": 1, "pregame_lines": 1})
    if game_doc:
        lp = game_doc.get("last_prediction")
        if lp:
            last_prediction = dict(lp) if hasattr(lp, "to_dict") else lp
        pl = game_doc.get("pregame_lines")
        if pl:
            pregame_lines = dict(pl) if hasattr(pl, "to_dict") else pl

    return render_template(
        "basketball/game_detail.html",
        game_id=game_id, home_team=home_team, away_team=away_team,
        game_date=game_date.strftime("%Y-%m-%d"), season=season,
        get_position_abbreviation=get_position_abbreviation,
        home_players=home_players, away_players=away_players,
        home_team_logo=get_logo_url(htd),
        home_team_color=htd.get("color", "667eea"),
        home_team_alternate_color=htd.get("alternateColor", "764ba2"),
        away_team_logo=get_logo_url(atd),
        away_team_color=atd.get("color", "666666"),
        away_team_alternate_color=atd.get("alternateColor", "999999"),
        home_records=home_records, away_records=away_records,
        last_prediction=last_prediction, pregame_lines=pregame_lines,
    )


# ======================================================================
#  INJURIES MANAGER PAGE
# ======================================================================

@bp.route("/<league_id>/injuries-manager")
def injuries_manager(league_id=None):
    return render_template("basketball/injuries_manager.html", stats={})


# ======================================================================
#  PLAYER MANAGEMENT APIs
# ======================================================================

@bp.route("/<league_id>/api/move-player", methods=["POST"])
def move_player(league_id=None):
    from bball.services.game_service import move_player_to_team
    data = request.json
    pid, to_team, season = data.get("player_id"), data.get("to_team"), data.get("season")
    if not all([pid, to_team, season]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        return jsonify(move_player_to_team(db=_get_db(), player_id=str(pid), to_team=to_team, season=season, league=g.league))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/teams-list", methods=["GET"])
def api_teams_list(league_id=None):
    db = _get_db()
    teams_coll = g.league.collections.get("teams", "nba_teams")
    teams = list(db[teams_coll].find({}, {"_id": 0, "abbreviation": 1, "displayName": 1, "name": 1, "id": 1, "team_id": 1, "logo": 1, "logos": 1}))
    result = []
    for t in teams:
        logo = t.get("logo")
        if not logo:
            logos = t.get("logos", [])
            if isinstance(logos, list) and logos:
                first = logos[0]
                if isinstance(first, dict):
                    logo = first.get("href")
        result.append({
            "team_id": t.get("id") or t.get("team_id") or t.get("abbreviation", ""),
            "abbreviation": t.get("abbreviation", ""),
            "display_name": t.get("displayName") or t.get("name", ""),
            "logo": logo,
        })
    return jsonify(result)


@bp.route("/<league_id>/api/player-search", methods=["GET"])
def player_search(league_id=None):
    from bball.services.game_service import search_players_for_roster
    q = request.args.get("q", "").strip()
    season = request.args.get("season", "")
    if len(q) < 2 or not season:
        return jsonify([])
    return jsonify(search_players_for_roster(_get_db(), q, season, g.league))


@bp.route("/<league_id>/api/add-player-to-roster", methods=["POST"])
def add_player_to_roster(league_id=None):
    from bball.services.game_service import add_player_to_team_roster
    data = request.json
    pid, team, season = data.get("player_id"), data.get("team"), data.get("season")
    game_date_str = data.get("game_date")
    if not all([pid, team, season]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        return jsonify(add_player_to_team_roster(_get_db(), pid, team, season, game_date_str, g.league))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/drop-player-from-roster", methods=["POST"])
def drop_player_from_roster_endpoint(league_id=None):
    from bball.services.lineup_service import drop_player_from_roster
    data = request.json
    pid, team, season = data.get("player_id"), data.get("team"), data.get("season")
    if not all([pid, team, season]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        return jsonify(drop_player_from_roster(_get_db(), pid, team, season, g.league))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/auto-set-lineups", methods=["POST"])
def auto_set_lineups_endpoint(league_id=None):
    from bball.services.lineup_service import auto_set_lineups
    data = request.json or {}
    game_date = data.get("game_date") or datetime.now().strftime("%Y-%m-%d")
    try:
        return jsonify({"success": True, **auto_set_lineups(_get_db(), g.league, game_date)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/remove-player", methods=["POST"])
def remove_player(league_id=None):
    """Remove a player from the game's player list (does NOT update rosters)."""
    db = _get_db()
    data = request.json
    gid, pid, team = data.get("game_id"), data.get("player_id"), data.get("team")
    if not all([gid, pid, team]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        games_coll = g.league.collections.get("games", "stats_nba")
        game = db[games_coll].find_one({"game_id": gid})
        if not game:
            return jsonify(success=False, error=f"Game {gid} not found"), 404
        team_key = "homeTeam" if team == game.get("homeTeam", {}).get("name") else "awayTeam"
        current = game.get(team_key, {}).get("players", [])
        updated = [p for p in current if str(p) != str(pid)]
        db[games_coll].update_one({"game_id": gid}, {"$set": {f"{team_key}.players": updated, "updated_at": datetime.utcnow()}})
        return jsonify(success=True, message=f"Player {pid} removed", remaining_players=len(updated))
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/load-players", methods=["POST"])
def load_players(league_id=None):
    """Load players from ESPN API and update players and injuries."""
    from bball.data.espn_client import ESPNClient
    db = _get_db()
    data = request.json
    gid = data.get("game_id")
    if not gid:
        return jsonify(success=False, error="Missing game_id"), 400
    try:
        games_coll = g.league.collections.get("games", "stats_nba")
        game = db[games_coll].find_one({"game_id": gid})
        if not game:
            return jsonify(success=False, error=f"Game {gid} not found"), 404
        ht = game.get("homeTeam", {}).get("name", "")
        at = game.get("awayTeam", {}).get("name", "")
        if not ht or not at:
            return jsonify(success=False, error="Game missing team information"), 400
        espn_client = ESPNClient(league=g.league)
        game_summary = espn_client.get_game_summary(gid)
        if not game_summary:
            return jsonify(success=False, error="Could not fetch game summary"), 404
        from bball.web.helpers import extract_and_update_teams, extract_and_update_player_roster, extract_and_update_injuries
        teams_coll = g.league.collections.get("teams", "nba_teams")
        extract_and_update_teams(db, game_summary, teams_coll)
        extract_and_update_player_roster(db, game_summary, ht, at)
        extract_and_update_injuries(db, game_summary, ht, at)
        return jsonify(success=True, message="Players loaded and updated successfully")
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/update-player", methods=["POST"])
def update_player(league_id=None):
    """Update player status (playing/starter/injured) for a game."""
    from bball.utils import get_season_from_date as get_season_from_date_core
    db = _get_db()
    data = request.json
    if not data:
        return jsonify(success=False, error="No JSON data received"), 400
    gid = data.get("game_id")
    pid = data.get("player_id")
    team = data.get("team")
    is_starter = data.get("is_starter")
    is_injured = data.get("is_injured")
    is_disabled = data.get("is_disabled")
    if not all([gid, pid, team]):
        return jsonify(success=False, error="Missing required fields"), 400
    league = g.league
    games_coll = league.collections.get("games", "stats_nba")
    players_coll = league.collections.get("players", "nba_players")
    rosters_coll = league.collections.get("rosters", "nba_rosters")
    gid_query = [gid]
    if str(gid).isdigit():
        gid_query += [int(gid), str(gid)]
    game = db[games_coll].find_one({"game_id": {"$in": gid_query}})
    if not game:
        return jsonify(success=False, error=f"Game not found: {gid}"), 404
    game_date_str = game.get("date")
    if not game_date_str:
        return jsonify(success=False, error="Game missing date"), 400
    try:
        gd = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        season = get_season_from_date_core(gd, league=league)
    except Exception:
        return jsonify(success=False, error="Invalid game date"), 400
    pid_query = [pid]
    if str(pid).isdigit():
        pid_query += [int(pid), str(pid)]
    if is_injured is not None:
        db[players_coll].update_one(
            {"player_id": {"$in": pid_query}},
            {"$set": {"injured": is_injured, "last_status_update": datetime.utcnow()}},
            upsert=False,
        )
    roster_doc = db[rosters_coll].find_one({"season": season, "team": team})
    if roster_doc:
        roster = roster_doc.get("roster", [])
        for entry in roster:
            if str(entry.get("player_id")) == str(pid):
                if is_starter is not None:
                    entry["starter"] = is_starter
                if is_injured is not None:
                    entry["injured"] = is_injured
                if is_disabled is not None:
                    entry["disabled"] = is_disabled
                break
        db[rosters_coll].update_one(
            {"season": season, "team": team},
            {"$set": {"roster": roster, "updated_at": datetime.utcnow()}},
        )
    return jsonify(success=True)


# ======================================================================
#  GAME DETAIL / PLAYER DETAIL APIs
# ======================================================================

@bp.route("/<league_id>/api/game-detail/<game_id>", methods=["GET"])
def api_game_detail(game_id, league_id=None):
    from bball.services.game_service import get_game_detail
    date_param = request.args.get("date")
    gd = None
    if date_param:
        try:
            gd = datetime.strptime(date_param, "%Y-%m-%d").date()
        except Exception:
            pass
    result = get_game_detail(_get_db(), game_id, gd, g.league)
    if not result.get("success"):
        code = 404 if "not found" in result.get("error", "").lower() else 400
        return jsonify(result), code
    return jsonify(result)


@bp.route("/<league_id>/api/player-detail", methods=["GET"])
def api_player_detail(league_id=None):
    from bball.services.game_service import get_player_detail
    pid = request.args.get("player_id")
    team = request.args.get("team")
    if not all([pid, team]):
        return jsonify(success=False, error="Missing player_id or team"), 400
    game_date_str = request.args.get("date")
    season = request.args.get("season")
    gd = None
    if game_date_str:
        try:
            gd = datetime.strptime(game_date_str, "%Y-%m-%d").date()
        except Exception:
            pass
    if not gd:
        gd = date.today()
    if not season:
        season = get_season_from_date(gd, league=g.league)
    return jsonify(get_player_detail(_get_db(), pid, team, season, gd, g.league))


@bp.route("/<league_id>/api/player-per", methods=["GET"])
def api_player_per(league_id=None):
    from bball.services.game_service import get_player_per
    pid = request.args.get("player_id")
    team = request.args.get("team")
    if not all([pid, team]):
        return jsonify(success=False, error="Missing player_id or team"), 400
    gd_str = request.args.get("date")
    season = request.args.get("season")
    gd = None
    if gd_str:
        try:
            gd = datetime.strptime(gd_str, "%Y-%m-%d").date()
        except Exception:
            pass
    if not gd:
        gd = date.today()
    if not season:
        season = get_season_from_date(gd, league=g.league)
    return jsonify(get_player_per(_get_db(), pid, team, season, gd, g.league))


@bp.route("/<league_id>/api/players-per-batch", methods=["POST"])
def api_players_per_batch(league_id=None):
    from bball.services.game_service import get_players_per_batch
    data = request.json
    if not data:
        return jsonify(success=False, error="No data provided"), 400
    players = data.get("players", [])
    if not players:
        return jsonify(success=True, per_values={})
    gd_str = data.get("date")
    season = data.get("season")
    gd = None
    if gd_str:
        try:
            gd = datetime.strptime(gd_str, "%Y-%m-%d").date()
        except Exception:
            pass
    if not gd:
        gd = date.today()
    if not season:
        season = get_season_from_date(gd, league=g.league)
    return jsonify(success=True, per_values=get_players_per_batch(_get_db(), players, season, gd, g.league))


@bp.route("/<league_id>/api/player-news", methods=["GET"])
def api_player_news(league_id=None):
    """Fetch player news from Rotowire."""
    import json
    from bball.services.webpage_parser import WebpageParser

    pid = request.args.get("player_id")
    if not pid:
        return jsonify(success=False, error="Missing player_id"), 400

    league_config = g.league
    news_sources = league_config.raw.get("news_sources", {})
    roto = news_sources.get("rotowire", {})
    slugs_file = roto.get("mappings", {}).get("player_slugs_file")
    if not slugs_file:
        return jsonify(success=False, error="Rotowire not available for this league"), 404

    patterns = roto.get("patterns", {})
    player_pattern = patterns.get("player")
    base_url = roto.get("base_url", "https://www.rotowire.com")
    if not player_pattern:
        return jsonify(success=False, error="Rotowire player URL pattern not configured"), 404

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    slugs_path = os.path.join(project_root, slugs_file)
    if not os.path.exists(slugs_path):
        return jsonify(success=False, error="Rotowire mapping file not found"), 404

    try:
        with open(slugs_path, "r") as f:
            player_slugs = json.load(f)
    except Exception as e:
        return jsonify(success=False, error=f"Failed to load mapping: {e}"), 500

    player_data = player_slugs.get(str(pid))
    if not player_data or not player_data.get("rotowire_slug"):
        return jsonify(success=False, error="Player not in Rotowire mapping"), 404

    player_url = base_url + player_pattern.replace("{player_slug}", player_data["rotowire_slug"])
    try:
        text = WebpageParser.extract_from_url(player_url, timeout=15)
        start_idx = text.find("Read Past Outlooks")
        end_idx = text.find("NBA Per Game Stats")
        if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
            text = text[start_idx + len("Read Past Outlooks"):end_idx].strip()
        elif start_idx != -1:
            text = text[start_idx + len("Read Past Outlooks"):].strip()
        elif end_idx != -1:
            text = text[:end_idx].strip()
        if len(text) > 2000:
            text = text[:2000] + "..."
        return jsonify(success=True, player_info=text, source_url=player_url, player_name=player_data.get("espn_name", ""))
    except Exception as e:
        return jsonify(success=False, error=f"Failed to fetch: {e}"), 500


# ======================================================================
#  PREDICTION APIs
# ======================================================================

@bp.route("/<league_id>/api/predict", methods=["POST"])
def predict(league_id=None):
    """Generate prediction for a game."""
    from bball.services.prediction import PredictionService
    db = _get_db()
    data = request.json
    if not data:
        return jsonify(success=False, error="No data received"), 400
    gid = data.get("game_id")
    gd_str = data.get("game_date")
    ht = data.get("home_team")
    at = data.get("away_team")
    missing = [k for k, v in {"game_id": gid, "game_date": gd_str, "home_team": ht, "away_team": at}.items() if not v]
    if missing:
        return jsonify(success=False, error=f"Missing: {', '.join(missing)}"), 400
    try:
        gd = datetime.strptime(gd_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify(success=False, error="Invalid date format"), 400
    try:
        svc = PredictionService(db=db, league=g.league)
        result = svc.predict_matchup(home_team=ht, away_team=at, game_date=gd_str, game_id=gid, include_points=True)
        if result.error:
            return jsonify(success=False, error=result.error), 400
        svc.save_prediction(result=result, game_id=gid, game_date=gd, home_team=ht, away_team=at)
        return jsonify(success=True, prediction={
            "predicted_winner": result.predicted_winner,
            "home_win_prob": result.home_win_prob, "away_win_prob": result.away_win_prob,
            "home_odds": result.home_odds, "away_odds": result.away_odds,
            "home_points_pred": result.home_points_pred, "away_points_pred": result.away_points_pred,
            "point_diff_pred": result.point_diff_pred,
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/pull-game-data", methods=["POST"])
def pull_game_data(league_id=None):
    """Pull game data from ESPN — spins off background sync job."""
    from bball.services.jobs import create_job, update_job_progress, complete_job, fail_job
    data = request.json
    date_str = data.get("date")
    if not date_str:
        return jsonify(success=False, error="Missing date parameter"), 400
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(success=False, error="Invalid date format"), 400

    league = g.league
    lid = league.league_id if league else "nba"
    job_id = create_job(job_type="sync", league=league, metadata={"date": date_str, "league": lid})

    def _run(job_id, date_str, league_id):
        from bball.pipeline.sync_pipeline import run_sync_pipeline
        from bball.league_config import load_league_config
        from bball.mongo import Mongo
        try:
            lc = load_league_config(league_id)
            bg_db = Mongo().db
            update_job_progress(job_id, 10, "Starting ESPN data sync...", league=lc)
            update_job_progress(job_id, 20, "Pulling games and player stats...", league=lc)
            gd = datetime.strptime(date_str, "%Y-%m-%d").date()
            def sync_progress(pct, msg):
                update_job_progress(job_id, 20 + int(pct * 0.65), msg, league=lc)
            run_sync_pipeline(league_config=lc, start_date=gd, end_date=gd, data_types={"games", "player_stats"}, dry_run=False, verbose=False, progress_callback=sync_progress)
            gc = lc.collections.get("games", "stats_nba")
            hw = bg_db[gc].count_documents({"date": date_str, "homeWon": {"$exists": True}})
            tg = bg_db[gc].count_documents({"date": date_str})
            complete_job(job_id, f"Synced {tg} games ({hw} completed)", league=lc)
        except Exception as e:
            import traceback; traceback.print_exc()
            try:
                fail_job(job_id, str(e), f"Sync failed: {e}", league=load_league_config(league_id))
            except Exception:
                pass

    threading.Thread(target=_run, args=(job_id, date_str, lid), daemon=True).start()
    return jsonify(success=True, job_id=job_id, message="Sync job started")


@bp.route("/<league_id>/api/predict-all", methods=["POST"])
def predict_all(league_id=None):
    """Generate predictions for all games on a given date."""
    from bball.services.jobs import create_job, complete_job, fail_job
    from bball.services.prediction import PredictionService
    from bball.services.config_manager import ModelConfigManager
    db = _get_db()
    data = request.json
    date_str = data.get("date")
    if not date_str:
        return jsonify(success=False, error="Missing date parameter"), 400
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(success=False, error="Invalid date format"), 400

    league = g.league
    cc = league.collections.get("model_config_classifier", "nba_model_config")
    sel = db[cc].find_one({"selected": True})
    is_valid, err = ModelConfigManager.validate_config_for_prediction(sel)
    if not is_valid:
        return jsonify(success=False, error=err), 400

    lid = league.league_id if league else "nba"
    job_id = create_job(job_type="predict_all", league=league, metadata={"date": date_str, "league": lid})

    def _run(job_id, date_str, league_id):
        from bball.league_config import load_league_config
        from bball.mongo import Mongo
        try:
            lc = load_league_config(league_id)
            bg_db = Mongo().db
            svc = PredictionService(db=bg_db, league=lc)
            gd = datetime.strptime(date_str, "%Y-%m-%d").date()
            ok = 0; fail_cnt = 0
            def on_pred(result, matchup):
                nonlocal ok, fail_cnt
                if result.error:
                    fail_cnt += 1; return
                try:
                    svc.save_prediction(result=result, game_id=result.game_id, game_date=gd, home_team=result.home_team, away_team=result.away_team)
                    ok += 1
                except Exception:
                    fail_cnt += 1
            results = svc.predict_date(game_date=date_str, include_points=True, job_id=job_id, on_prediction=on_pred)
            complete_job(job_id, f"Completed {ok}/{len(results)} predictions", league=lc)
        except Exception as e:
            import traceback; traceback.print_exc()
            try:
                fail_job(job_id, str(e), f"Prediction failed: {e}", league=load_league_config(league_id))
            except Exception:
                pass

    threading.Thread(target=_run, args=(job_id, date_str, lid), daemon=True).start()
    return jsonify(success=True, job_id=job_id, message="Prediction job started")


@bp.route("/<league_id>/api/predictions/<date_str>", methods=["GET"])
def get_predictions_by_date(date_str, league_id=None):
    """Get all predictions for a specific date."""
    try:
        pc = g.league.collections.get("model_predictions", "nba_model_predictions")
        preds = list(_get_db()[pc].find({"game_date": date_str}, {"_id": 0}))
        return jsonify(success=True, predictions=preds)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/game-features", methods=["POST"])
def get_game_features(league_id=None):
    """Get feature values used for a game prediction."""
    db = _get_db()
    data = request.json
    if not data:
        return jsonify(success=False, error="No data received"), 400
    gid = data.get("game_id")
    gd_str = data.get("game_date")
    ht = data.get("home_team")
    at = data.get("away_team")
    if not all([gid, gd_str, ht, at]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        league = g.league
        pc = league.collections.get("model_predictions", "nba_model_predictions")
        pred = db[pc].find_one({"game_id": gid})
        if not pred:
            return jsonify(success=False, error="No prediction found. Run predictions first."), 404
        features_dict = pred.get("features_dict")
        if not features_dict:
            return jsonify(success=False, error="No feature values stored."), 404
        feature_players = pred.get("feature_players", {})
        feature_names = [k for k in features_dict.keys() if not k.startswith("_")]

        from bball.features.groups import FeatureGroups
        filtered = {n: features_dict.get(n, 0.0) for n in feature_names if not n.startswith(("p_", "conf_", "disagree_"))}
        feature_categories = FeatureGroups.categorize_features(filtered)

        ensemble_breakdown = features_dict.get("_ensemble_breakdown")
        meta_normalized = ensemble_breakdown.get("meta_normalized_values", {}) if ensemble_breakdown else {}
        meta_companions = features_dict.get("_meta_companions", {})

        return jsonify(
            success=True, game_id=gid, home_team=ht, away_team=at, game_date=gd_str,
            feature_categories=feature_categories,
            home_injured_players=pred.get("home_injured_players", []),
            away_injured_players=pred.get("away_injured_players", []),
            feature_players=feature_players,
            total_features=len(feature_names),
            ensemble_breakdown=ensemble_breakdown,
            meta_normalized_values=meta_normalized,
            meta_companions=meta_companions,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/calculation-details", methods=["POST"])
def get_calculation_details(league_id=None):
    """Get calculation details for a team including players, features, and stats."""
    db = _get_db()
    data = request.json
    if not data:
        return jsonify(success=False, error="No data received"), 400
    team = data.get("team")
    opponent = data.get("opponent_team")
    gd_str = data.get("game_date")
    season = data.get("season")
    if not all([team, opponent, gd_str, season]):
        return jsonify(success=False, error="Missing required fields"), 400
    try:
        gd = datetime.strptime(gd_str, "%Y-%m-%d").date()
        per_calc = _get_per_calculator()
        from bball.features.injury import InjuryFeatureCalculator
        inj_calc = InjuryFeatureCalculator(db=db, league=g.league)

        rosters_coll = g.league.collections.get("rosters", "nba_rosters")
        roster_doc = db[rosters_coll].find_one({"season": season, "team": team})
        if not roster_doc:
            return jsonify(success=False, error=f"No roster for {team}"), 404

        roster = roster_doc.get("roster", [])
        all_ids = [str(p["player_id"]) for p in roster]
        playing_ids = [str(p["player_id"]) for p in roster if not p.get("injured")]
        injured_ids = [str(p["player_id"]) for p in roster if p.get("injured")]

        players_data = []
        for pid in all_ids:
            pdoc = db[g.league.collections.get("players", "nba_players")].find_one({"player_id": pid})
            if not pdoc:
                continue
            re = next((p for p in roster if str(p["player_id"]) == pid), {})
            stats = calculate_player_stats(db, pid, team, season, gd_str)
            pper = per_calc.get_player_per_before_date(pid, team, season, gd_str) if per_calc else 0.0
            players_data.append({
                "player_id": pid, "player_name": pdoc.get("player_name", "Unknown"),
                "position": pdoc.get("pos_display_name", ""),
                "is_starter": re.get("starter", False), "is_injured": re.get("injured", False),
                "stats": stats, "per": round(pper, 2) if pper else 0.0,
            })

        starters = [p["player_id"] for p in players_data if p["is_starter"] and not p["is_injured"]]
        pf = {"playing": playing_ids, "starters": starters}
        per_features = per_calc.compute_team_per_features(team, season, gd_str, top_n=8, player_filters=pf) if per_calc else None

        all_starters = [p["player_id"] for p in players_data if p["is_starter"]]
        pf_wi = {"playing": all_ids, "starters": all_starters}
        per_wi = per_calc.compute_team_per_features(team, season, gd_str, top_n=8, player_filters=pf_wi) if per_calc else None

        inj_features = inj_calc.compute_team_injury_features(team, season, gd_str, injured_ids, per_calculator=per_calc, recency_decay_k=15.0) if inj_calc else {}

        # Build player features dicts
        player_features = {}
        if per_features:
            player_features["player_team_per|season|avg"] = per_features.get("per_avg", 0.0)
            player_features["player_team_per|season|weighted_MPG"] = per_features.get("per_weighted", 0.0)
            player_features["player_starters_per|season|avg"] = per_features.get("starters_avg", 0.0)
            for i in range(1, 4):
                player_features[f"player_per_{i}|season|top1_avg"] = per_features.get(f"per{i}", 0.0)

        player_features_wi = {}
        if per_wi:
            player_features_wi["player_team_per|season|avg"] = per_wi.get("per_avg", 0.0)
            player_features_wi["player_team_per|season|weighted_MPG"] = per_wi.get("per_weighted", 0.0)
            player_features_wi["player_starters_per|season|avg"] = per_wi.get("starters_avg", 0.0)
            for i in range(1, 4):
                player_features_wi[f"player_per_{i}|season|top1_avg"] = per_wi.get(f"per{i}", 0.0)

        if inj_features:
            sev = inj_features.get("inj_severity|none|raw|home", 0.0)
            t1p = inj_features.get("inj_per|none|top1_avg|home", 0.0)
            rot = inj_features.get("inj_rotation_per|none|raw|home", 0.0)
            player_features["inj_impact|blend|raw"] = 0.45 * sev + 0.35 * t1p + 0.20 * rot
            player_features["inj_per|none|weighted_MIN|home"] = inj_features.get("inj_per|none|weighted_MIN|home", 0.0)
            player_features["inj_per|none|top1_avg|home"] = t1p
            player_features["inj_rotation_per|none|raw|home"] = rot
            player_features["inj_severity|none|raw|home"] = sev

        # Opponent
        opp_doc = db[rosters_coll].find_one({"season": season, "team": opponent})
        opp_injured = []
        opp_per = opp_per_wi = None
        if opp_doc:
            opp_roster = opp_doc.get("roster", [])
            opp_injured = [str(p["player_id"]) for p in opp_roster if p.get("injured")]
            opp_playing = [str(p["player_id"]) for p in opp_roster if not p.get("injured")]
            opp_all = [str(p["player_id"]) for p in opp_roster]
            opp_starters = [str(p["player_id"]) for p in opp_roster if p.get("starter") and not p.get("injured")]
            opp_all_starters = [str(p["player_id"]) for p in opp_roster if p.get("starter")]
            opp_per = per_calc.compute_team_per_features(opponent, season, gd_str, top_n=8, player_filters={"playing": opp_playing, "starters": opp_starters}) if per_calc else None
            opp_per_wi = per_calc.compute_team_per_features(opponent, season, gd_str, top_n=8, player_filters={"playing": opp_all, "starters": opp_all_starters}) if per_calc else None

        opp_inj = inj_calc.compute_team_injury_features(opponent, season, gd_str, opp_injured, per_calculator=per_calc, recency_decay_k=15.0) if inj_calc else {}

        # Diffs
        feature_diffs = {}
        feature_diffs_wi = {}
        if per_features and opp_per:
            for k in ["per_avg", "per_weighted", "starters_avg", "per1", "per2", "per3"]:
                dk = {"per_avg": "player_team_per|season|avg|diff", "per_weighted": "player_team_per|season|weighted_MPG|diff", "starters_avg": "player_starters_per|season|avg|diff"}.get(k, f"player_per_{k[-1]}|season|top1_avg|diff" if k.startswith("per") and k != "per_avg" and k != "per_weighted" else k)
                feature_diffs[dk] = per_features.get(k, 0.0) - opp_per.get(k, 0.0)
        if per_wi and opp_per_wi:
            for k in ["per_avg", "per_weighted", "starters_avg", "per1", "per2", "per3"]:
                dk = {"per_avg": "player_team_per|season|avg|diff", "per_weighted": "player_team_per|season|weighted_MPG|diff", "starters_avg": "player_starters_per|season|avg|diff"}.get(k, f"player_per_{k[-1]}|season|top1_avg|diff" if k.startswith("per") and k != "per_avg" and k != "per_weighted" else k)
                feature_diffs_wi[dk] = per_wi.get(k, 0.0) - opp_per_wi.get(k, 0.0)
        if inj_features and opp_inj:
            t_blend = 0.45 * inj_features.get("inj_severity|none|raw|home", 0) + 0.35 * inj_features.get("inj_per|none|top1_avg|home", 0) + 0.20 * inj_features.get("inj_rotation_per|none|raw|home", 0)
            o_blend = 0.45 * opp_inj.get("inj_severity|none|raw|home", 0) + 0.35 * opp_inj.get("inj_per|none|top1_avg|home", 0) + 0.20 * opp_inj.get("inj_rotation_per|none|raw|home", 0)
            feature_diffs["inj_impact|blend|raw|diff"] = t_blend - o_blend

        return jsonify(
            success=True, team=team, players=players_data,
            player_features=player_features, player_features_with_injured=player_features_wi,
            feature_diffs=feature_diffs, feature_diffs_with_injured=feature_diffs_wi,
            per_features=per_features, per_features_with_injured=per_wi,
            injury_features=inj_features,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


@bp.route("/<league_id>/api/betting-report", methods=["POST"])
def generate_betting_report_endpoint(league_id=None):
    """Generate betting report comparing model vs market odds."""
    try:
        db = _get_db()
        data = request.json
        date_str = data.get("date")
        bankroll = float(data.get("bankroll", 1000))
        edge_threshold = float(data.get("edge_threshold", 0.07))
        force_include = data.get("force_include_game_ids")
        if not date_str:
            return jsonify(success=False, error="Missing date parameter"), 400

        cc = g.league.collections.get("model_config_classifier", "nba_model_config")
        sel = db[cc].find_one({"selected": True})
        brier = sel.get("brier_score", 0.25) if sel else 0.25
        ll = sel.get("log_loss") if sel else None

        market_brier = market_ll = None
        cal_coll = g.league.collections.get("market_calibration")
        if cal_coll:
            cdoc = db[cal_coll].find_one({"league_id": g.league.league_id, "season": {"$regex": "^rolling_"}}, sort=[("computed_at", -1)])
            if cdoc:
                market_brier = cdoc.get("market_brier")
                market_ll = cdoc.get("market_log_loss")

        bin_trust = None
        btc = g.league.collections.get("bin_trust_weights")
        if btc:
            tdoc = db[btc].find_one({"league_id": g.league.league_id}, sort=[("computed_at", -1)])
            if tdoc:
                bin_trust = tdoc.get("bins", [])

        from bball.services.betting_report import generate_betting_report, group_parlay_fills
        recs = generate_betting_report(db, date_str, bankroll, brier, g.league, edge_threshold, force_include, log_loss_score=ll, market_brier=market_brier, market_log_loss=market_ll, bin_trust_weights=bin_trust)

        portfolio_data = data.get("portfolio_data")
        parlay_fills = group_parlay_fills(portfolio_data) if portfolio_data else []

        return jsonify(
            success=True,
            recommendations=[r.to_dict() for r in recs],
            parlay_fills=parlay_fills,
            brier_score=brier, log_loss=ll, bankroll=bankroll, edge_threshold=edge_threshold,
            market_brier=market_brier, market_log_loss=market_ll,
        )
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(success=False, error=str(e)), 500


# ======================================================================
#  LIVE GAMES (ESPN SCOREBOARD)
# ======================================================================

@bp.route("/<league_id>/api/live-games", methods=["GET"])
def get_live_games(league_id=None):
    """Get live game data from ESPN API for real-time polling."""
    date_str = request.args.get("date")
    if not date_str:
        return jsonify(success=False, error="Missing date parameter"), 400
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify(success=False, error="Invalid date format"), 400

    league = g.league
    date_yyyymmdd = date_str.replace("-", "")
    try:
        espn_url = league.espn_endpoint("scoreboard_site_template").format(YYYYMMDD=date_yyyymmdd)
    except Exception:
        espn_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={date_yyyymmdd}"

    live_games = {}
    try:
        import requests as _requests
        resp = _requests.get(espn_url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        for event in data.get("events", []):
            gid = event.get("id")
            if not gid:
                continue
            status_obj = event.get("status", {})
            gs = "pre"; gc = False; period = clock = status_detail = None
            if isinstance(status_obj, dict):
                st = status_obj.get("type", {})
                if isinstance(st, dict):
                    raw = st.get("name", "").lower()
                    gc = st.get("completed", False)
                    status_detail = st.get("shortDetail") or st.get("detail")
                    if gc or "final" in raw or "post" in raw:
                        gs = "post"
                    elif "progress" in raw or "halftime" in raw or raw == "in":
                        gs = "in"
                period = status_obj.get("period")
                clock = status_obj.get("displayClock")
            hs = as_ = None
            comps = event.get("competitions", [])
            if comps:
                for comp in comps[0].get("competitors", []):
                    sv = comp.get("score")
                    if sv is not None and sv != "":
                        try:
                            s = int(sv)
                            if comp.get("homeAway") == "home":
                                hs = s
                            else:
                                as_ = s
                        except (ValueError, TypeError):
                            pass
            live_games[gid] = {"status": gs, "completed": gc, "period": period, "clock": clock, "status_detail": status_detail, "home_score": hs, "away_score": as_}
    except Exception as e:
        return jsonify(success=False, error=str(e)), 500

    return jsonify(success=True, games=live_games)


# ======================================================================
#  PORTFOLIO / GAME POSITIONS (Kalshi)
# ======================================================================

@bp.route("/<league_id>/api/portfolio/game-positions", methods=["GET"])
def get_portfolio_game_positions(league_id=None):
    """Get portfolio positions, orders, and fills for games on a specific date."""
    import os
    from bball.market.connector import MarketConnector
    from bball.market.kalshi import match_portfolio_to_games
    from bball.league_config import load_league_config
    from sportscore.market import SimpleCache

    date_str = request.args.get("date")
    if not date_str:
        return jsonify(success=False, error="Missing date parameter"), 400
    try:
        game_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify(success=False, error="Invalid date format"), 400

    api_key = os.environ.get("KALSHI_API_KEY")
    pk_dir = os.environ.get("KALSHI_PRIVATE_KEY_DIR")
    if not api_key or not pk_dir:
        return jsonify(success=True, available=False, message="Kalshi API credentials not configured")

    try:
        connector = MarketConnector({"KALSHI_API_KEY": api_key, "KALSHI_PRIVATE_KEY_DIR": pk_dir})
    except Exception:
        return jsonify(success=True, available=False, message="Failed to connect to Kalshi API")

    league_config = g.league
    lid = league_config.league_id if league_config else "nba"
    games_coll = league_config.collections.get("games", "stats_nba")
    db = _get_db()

    games = list(db[games_coll].find({"date": date_str}, {"game_id": 1, "homeTeam.name": 1, "awayTeam.name": 1}))
    if not games:
        return jsonify(success=True, available=True, positions={}, message="No games for this date")

    try:
        lc = load_league_config(lid)
        series_ticker = lc.raw.get("market", {}).get("series_ticker", "KXNBAGAME")
        spread_series = lc.raw.get("market", {}).get("spread_series_ticker", "")
    except Exception:
        series_ticker = "KXNBAGAME"
        spread_series = "KXNBASPREAD"

    _cache = SimpleCache(default_ttl=60)
    _TTL = 60

    try:
        all_positions = _cache.get("gp_positions")
        if all_positions is None:
            all_positions = connector.get_positions(limit=200).get("market_positions", [])
            _cache.set("gp_positions", all_positions, ttl=_TTL)
    except Exception:
        all_positions = []

    try:
        all_fills = _cache.get("gp_fills")
        if all_fills is None:
            min_ts = int((datetime.now(utc).timestamp() - 86400) * 1000)
            all_fills = connector.get_fills(min_ts=min_ts, limit=200).get("fills", [])
            _cache.set("gp_fills", all_fills, ttl=_TTL)
    except Exception:
        all_fills = []

    try:
        all_orders = _cache.get("gp_orders")
        if all_orders is None:
            all_orders = connector.get_orders(status="resting", limit=200).get("orders", [])
            _cache.set("gp_orders", all_orders, ttl=_TTL)
    except Exception:
        all_orders = []

    try:
        sk = f"gp_settlements:{date_str}"
        all_settlements = _cache.get(sk)
        if all_settlements is None:
            min_ts = int((datetime.combine(game_date, datetime.min.time()).timestamp() - 86400) * 1000)
            all_settlements = connector.get_settlements(min_ts=min_ts, limit=200).get("settlements", [])
            _cache.set(sk, all_settlements, ttl=_TTL)
    except Exception:
        all_settlements = []

    def matches(item):
        t = item.get("ticker", "") or item.get("event_ticker", "")
        if t.startswith(series_ticker):
            return True
        if spread_series and t.startswith(spread_series):
            return True
        if "MULTIGAME" in t.upper():
            return True
        return False

    all_positions = [p for p in all_positions if matches(p)]
    all_fills = [f for f in all_fills if matches(f)]
    all_orders = [o for o in all_orders if matches(o)]
    all_settlements = [s for s in all_settlements if matches(s)]

    result = match_portfolio_to_games(
        games=games, positions=all_positions, fills=all_fills,
        orders=all_orders, settlements=all_settlements,
        game_date=game_date, league_id=lid,
    )

    return jsonify(
        success=True, available=True, date=date_str,
        positions=result.game_data,
        fetched_at=datetime.now(utc).isoformat(),
    )


# ======================================================================
#  MARKET PRICES (public Kalshi API)
# ======================================================================

# ======================================================================
#  MATCHUP CHAT APIs
# ======================================================================

@bp.route("/<league_id>/api/matchup-chat/sessions", methods=["POST"])
def create_matchup_chat_session(league_id=None):
    """Create or get existing matchup chat session for a game."""
    db = _get_db()
    data = request.json
    gid = data.get("game_id")
    if not gid:
        return jsonify(error="game_id is required"), 400

    games_coll = g.league.collections.get("games", "stats_nba")
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")

    existing = db[sessions_coll].find_one({"game_id": gid}, sort=[("updated_at", -1)])
    if existing:
        return jsonify(success=True, session_id=str(existing["_id"]), name=existing.get("name", f"Matchup Chat - {gid}"), game_id=gid, existing=True)

    game = db[games_coll].find_one({"game_id": gid})
    if not game:
        return jsonify(error=f"Game {gid} not found"), 404

    ht = data.get("home_team") or game.get("homeTeam", {}).get("name", "")
    at = data.get("away_team") or game.get("awayTeam", {}).get("name", "")
    gd = data.get("game_date") or game.get("date", "")
    if not ht or not at:
        return jsonify(error="Could not determine teams"), 400

    session_name = f"{at} @ {ht}"
    if gd:
        session_name += f" - {gd}"

    cc = g.league.collections.get("model_config_classifier", "nba_model_config")
    sel = db[cc].find_one({"selected": True})
    mcid = str(sel["_id"]) if sel else None

    season = None
    if gd:
        try:
            season = get_season_from_date(datetime.strptime(gd, "%Y-%m-%d").date())
        except Exception:
            pass

    new_session = {
        "game_id": gid, "name": session_name, "messages": [],
        "context": {"home_team": ht, "away_team": at, "game_date": gd, "season": season, "model_config_id": mcid},
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    result = db[sessions_coll].insert_one(new_session)
    return jsonify(success=True, session_id=str(result.inserted_id), name=session_name, game_id=gid, existing=False)


@bp.route("/<league_id>/api/matchup-chat", methods=["POST"])
def matchup_chat_api(league_id=None):
    """Matchup chat API endpoint."""
    from bball.services.matchup_chat import Controller as MatchupChatController
    from bball.services.matchup_chat.schemas import ControllerOptions

    db = _get_db()
    data = request.json
    message = data.get("message", "").strip()
    session_id = data.get("session_id")
    memory = data.get("memory")
    if not message:
        return jsonify(error="Message is required"), 400
    if not session_id:
        return jsonify(error="session_id is required"), 400

    try:
        sid = ObjectId(session_id)
    except Exception:
        return jsonify(error="Invalid session_id"), 400

    lc = g.league
    sessions_coll = lc.collections.get("matchup_sessions", "nba_matchup_sessions")
    session = db[sessions_coll].find_one({"_id": sid})
    if not session:
        return jsonify(error="Session not found"), 404
    gid = session.get("game_id")
    if not gid:
        return jsonify(error="Session missing game_id"), 400

    # Save user message
    db[sessions_coll].update_one({"_id": sid}, {
        "$push": {"messages": {"role": "user", "content": message, "timestamp": datetime.utcnow()}},
        "$set": {"updated_at": datetime.utcnow()},
    })

    # Build conversation history
    doc = db[sessions_coll].find_one({"_id": sid})
    history = [{"role": m["role"], "content": m["content"]} for m in doc.get("messages", []) if m.get("role") in ("user", "assistant")]
    if memory is not None:
        try:
            mi = int(memory)
            if mi > 0:
                history = history[-mi:]
        except (ValueError, TypeError):
            pass

    try:
        controller = MatchupChatController(db=db, league=lc, league_id=lc.league_id)
        options = ControllerOptions(show_agent_actions=data.get("show_agent_actions", False))
        result = controller.handle_user_message(game_id=gid, user_message=message, conversation_history=history, options=options)
        resp_text = result.get("response", "")
        agent_actions = result.get("agent_actions", [])
        turn_plan = result.get("turn_plan", {})

        db[sessions_coll].update_one({"_id": sid}, {
            "$push": {"messages": {"role": "assistant", "content": resp_text, "timestamp": datetime.utcnow(), "agent_actions": agent_actions, "turn_plan": turn_plan}},
            "$set": {"updated_at": datetime.utcnow()},
        })
        return jsonify(success=True, response=resp_text, agent_actions=agent_actions, turn_plan=turn_plan, session_id=session_id)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=f"Agent error: {e}", session_id=session_id), 500


@bp.route("/<league_id>/api/matchup-chat/sessions", methods=["GET"])
def get_matchup_chat_sessions(league_id=None):
    db = _get_db()
    gid_filter = request.args.get("game_id")
    query = {"game_id": gid_filter} if gid_filter else {}
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")
    sessions = list(db[sessions_coll].find(query, {"name": 1, "game_id": 1, "created_at": 1, "updated_at": 1, "messages": {"$slice": 1}}, sort=[("updated_at", -1)]))
    for s in sessions:
        s["_id"] = str(s["_id"])
        s["created_at"] = s["created_at"].isoformat() if s.get("created_at") else None
        s["updated_at"] = s["updated_at"].isoformat() if s.get("updated_at") else None
        full = db[sessions_coll].find_one({"_id": ObjectId(s["_id"])})
        s["message_count"] = len(full.get("messages", [])) if full else 0
    return jsonify(success=True, sessions=sessions)


@bp.route("/<league_id>/api/matchup-chat/sessions/<session_id>", methods=["GET"])
def get_matchup_chat_session(session_id, league_id=None):
    db = _get_db()
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")
    try:
        sid = ObjectId(session_id)
    except Exception:
        return jsonify(error="Invalid session_id"), 400
    session = db[sessions_coll].find_one({"_id": sid})
    if not session:
        return jsonify(error="Session not found"), 404
    session["_id"] = str(session["_id"])
    session["created_at"] = session["created_at"].isoformat() if session.get("created_at") else None
    session["updated_at"] = session["updated_at"].isoformat() if session.get("updated_at") else None
    for msg in session.get("messages", []):
        if "timestamp" in msg:
            msg["timestamp"] = msg["timestamp"].isoformat()
    return jsonify(success=True, session=session)


@bp.route("/<league_id>/api/matchup-chat/sessions/<session_id>/messages", methods=["DELETE"])
def delete_matchup_chat_message(session_id, league_id=None):
    db = _get_db()
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")
    try:
        sid = ObjectId(session_id)
    except Exception:
        return jsonify(error="Invalid session_id"), 400
    session = db[sessions_coll].find_one({"_id": sid})
    if not session:
        return jsonify(error="Session not found"), 404
    data = request.json
    mi = data.get("message_index")
    mc = data.get("message_content")
    mr = data.get("message_role")
    msgs = session.get("messages", [])
    deleted = False
    if mi is not None and 0 <= mi < len(msgs):
        msgs.pop(mi); deleted = True
    elif mc:
        for i, m in enumerate(msgs):
            if m.get("content") == mc and (mr is None or m.get("role") == mr):
                msgs.pop(i); deleted = True; break
    if not deleted:
        return jsonify(error="Message not found"), 400
    db[sessions_coll].update_one({"_id": sid}, {"$set": {"messages": msgs, "updated_at": datetime.utcnow()}})
    return jsonify(success=True)


@bp.route("/<league_id>/api/matchup-chat/sessions/<session_id>", methods=["DELETE"])
def delete_matchup_chat_session(session_id, league_id=None):
    db = _get_db()
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")
    try:
        sid = ObjectId(session_id)
    except Exception:
        return jsonify(error="Invalid session_id"), 400
    result = db[sessions_coll].delete_one({"_id": sid})
    if result.deleted_count == 0:
        return jsonify(error="Session not found"), 404
    with _session_lock:
        _matchup_agent_sessions.pop(session_id, None)
    return jsonify(success=True)


@bp.route("/<league_id>/api/matchup-chat/sessions/by-game/<game_id>", methods=["DELETE"])
def delete_matchup_chat_sessions_by_game(game_id, league_id=None):
    db = _get_db()
    sessions_coll = g.league.collections.get("matchup_sessions", "nba_matchup_sessions")
    sessions = list(db[sessions_coll].find({"game_id": game_id}, {"_id": 1}))
    sids = [str(s["_id"]) for s in sessions]
    result = db[sessions_coll].delete_many({"game_id": game_id})
    cleared = 0
    with _session_lock:
        for sid in sids:
            if sid in _matchup_agent_sessions:
                del _matchup_agent_sessions[sid]
                cleared += 1
    return jsonify(success=True, deleted_count=result.deleted_count, cleared_cache_count=cleared)
