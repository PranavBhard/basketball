from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from bball_app.core.services.matchup_chat.schemas import SharedContext
from bball_app.core.services.prediction import PredictionService
from bball_app.core.market.kalshi import get_game_market_data
from bball_app.core.services.espn_odds import get_live_pregame_lines


def _get_coll(league, key: str, fallback: str) -> str:
    if league is not None and getattr(league, "collections", None):
        return league.collections.get(key) or fallback
    return fallback


def _safe_date_str(d: Any) -> str:
    if not d:
        return ""
    s = str(d)
    return s[:10]


def _parse_date_yyyy_mm_dd(d: str) -> Optional[datetime.date]:
    try:
        return datetime.strptime(d[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _team_meta(db, league, abbrev: str) -> Dict[str, str]:
    teams_coll = _get_coll(league, "teams", "teams_nba")
    doc = db[teams_coll].find_one({"abbreviation": abbrev}) or {}
    team_id = doc.get("team_id") or doc.get("id") or ""
    full_name = doc.get("displayName") or doc.get("name") or ""
    return {
        "name": abbrev,
        "full_name": full_name,
        "team_id": str(team_id) if team_id is not None else "",
    }


def ensure_shared_context_baseline(
    *,
    db,
    league,
    game_id: str,
    league_id: str = "nba",
    existing: Optional[SharedContext] = None,
) -> Tuple[SharedContext, Dict[str, Any]]:
    """
    Ensure we have:
    - `game` metadata (home/away abbrev/full_name/team_id + date)
    - `ensemble_model.p_home` baseline win prob (if available)
    - `market_snapshot` (Kalshi + vegas odds) for this game (best-effort)
    - a persisted prediction document in the model predictions collection (SSoT)

    Returns:
      (fields_to_set_in_shared_context, prediction_info)
    """
    existing = existing or {}
    league_id = (league_id or "nba").lower()

    games_coll = _get_coll(league, "games", "stats_nba")
    game_doc = db[games_coll].find_one({"game_id": game_id}) or {}

    home_abbrev = (game_doc.get("homeTeam") or {}).get("name") or ""
    away_abbrev = (game_doc.get("awayTeam") or {}).get("name") or ""
    game_date = _safe_date_str(game_doc.get("date"))

    game_meta: Dict[str, Any] = {"date": game_date}
    if away_abbrev:
        game_meta["away"] = _team_meta(db, league, away_abbrev)
    if home_abbrev:
        game_meta["home"] = _team_meta(db, league, home_abbrev)

    # Prediction persistence is handled by PredictionService (SSoT)
    service = PredictionService(db=db, league=league)
    prediction_info = service.get_prediction_for_game(game_id) or {}

    if not prediction_info and home_abbrev and away_abbrev and game_date:
        # Compute + persist
        result = service.predict_game(home_team=home_abbrev, away_team=away_abbrev, game_date=game_date)
        # best-effort save (prediction doc contains rich info used by agents)
        date_obj = _parse_date_yyyy_mm_dd(game_date)
        if date_obj:
            prediction_info = service.save_prediction(
                result=result,
                game_id=game_id,
                game_date=date_obj,
                home_team=home_abbrev,
                away_team=away_abbrev,
            )
        else:
            # fallback to an unserialized dict view
            prediction_info = result.to_dict()

    # Normalize p_home from prediction_info if present (PredictionService stores win_prob as 0-100)
    p_home = None
    try:
        if "home_win_prob" in prediction_info and prediction_info["home_win_prob"] is not None:
            p_home = float(prediction_info["home_win_prob"]) / 100.0
    except Exception:
        p_home = None

    fields_to_set: SharedContext = {
        "game_id": game_id,
        "game": game_meta,
        "ensemble_model": {
            **({"p_home": p_home} if p_home is not None else {}),
        },
    }

    # Market snapshot (best-effort). Keep it small and explicitly distinguish it from model p_home.
    # Avoid repeated calls if we already have a recent snapshot.
    try:
        snap = existing.get("market_snapshot") if isinstance(existing, dict) else None
        snap_ts = None
        if isinstance(snap, dict):
            snap_ts = str(snap.get("timestamp") or "")
        is_stale = True
        try:
            if snap_ts:
                t = datetime.fromisoformat(snap_ts.replace("Z", "+00:00"))
                # refresh if older than 1 hour
                is_stale = (datetime.now(t.tzinfo) - t).total_seconds() > 3600
        except Exception:
            is_stale = True

        if not snap or is_stale:
            vegas_odds = get_live_pregame_lines(game_id, league=league) or (game_doc.get("pregame_lines") or game_doc.get("vegas") or {})
            vegas_odds_source = "espn_api" if (isinstance(vegas_odds, dict) and ("over_under" in vegas_odds or "spread" in vegas_odds or "home_ml" in vegas_odds or "away_ml" in vegas_odds)) else "db_snapshot"

            prediction_markets: Optional[Dict[str, Any]] = None
            try:
                date_str = str(game_doc.get("date") or "")[:10]
                game_date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                away = away_abbrev
                home = home_abbrev
                md = get_game_market_data(game_date=game_date_obj, away_team=away, home_team=home, league_id=league_id)
                prediction_markets = md.to_dict() if md is not None else None
            except Exception as e:
                prediction_markets = {"error": str(e)}

            fields_to_set["market_snapshot"] = {
                "timestamp": datetime.utcnow().replace(tzinfo=None).isoformat() + "+00:00",
                "source": "baseline",
                "prediction_markets": prediction_markets,
                "vegas_odds": vegas_odds,
                "vegas_odds_source": vegas_odds_source,
                "note": "market_snapshot reflects public market/vegas pricing; it is distinct from model p_home.",
            }
        else:
            fields_to_set["market_snapshot"] = snap
    except Exception:
        pass

    return fields_to_set, prediction_info

