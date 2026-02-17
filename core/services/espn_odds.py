"""
ESPN odds service (core layer).

Fetches fresh odds/lines for a game_id using ESPN game summary and extracts
`pregame_lines` in the same shape used elsewhere:
  { over_under, spread, home_ml, away_ml }

This is the Single Source of Truth for live ESPN odds fetching.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from bball_app.core.data.espn_client import ESPNClient


def extract_pregame_lines_from_summary(game_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract odds from ESPN game summary pickcenter and format as pregame_lines.

    ESPN provides odds data in the 'pickcenter' array of the game summary response.
    """
    pickcenter = game_summary.get("pickcenter", []) if isinstance(game_summary, dict) else []
    if not pickcenter:
        return None

    # Use first provider's odds (often DraftKings)
    odds = pickcenter[0] or {}
    pregame_lines: Dict[str, Any] = {}

    if odds.get("overUnder") is not None:
        pregame_lines["over_under"] = odds.get("overUnder")

    # spread is home team spread
    if odds.get("spread") is not None:
        pregame_lines["spread"] = odds.get("spread")

    home_odds = odds.get("homeTeamOdds") or {}
    if isinstance(home_odds, dict) and home_odds.get("moneyLine") is not None:
        pregame_lines["home_ml"] = home_odds.get("moneyLine")

    away_odds = odds.get("awayTeamOdds") or {}
    if isinstance(away_odds, dict) and away_odds.get("moneyLine") is not None:
        pregame_lines["away_ml"] = away_odds.get("moneyLine")

    return pregame_lines if pregame_lines else None


def get_live_pregame_lines(game_id: str, *, league=None) -> Optional[Dict[str, Any]]:
    """
    Fetch fresh ESPN game summary and extract pregame lines.

    Returns:
      pregame_lines dict or None if not available.
    """
    client = ESPNClient(league=league)
    summary = client.get_game_summary(str(game_id))
    if not summary:
        return None
    return extract_pregame_lines_from_summary(summary)

