#!/usr/bin/env python3
"""
Manual smoke test: Matchup multi-agent Controller flow.

This mirrors the web `/api/matchup-chat` path (Controller invocation) without
running the Flask server.

Usage:
  source venv/bin/activate
  PYTHONPATH=/Users/pranav/Documents/NBA python tests/test_matchup_chat_controller_flow.py --game-id <game_id>

Notes:
- This will use MongoDB via `nba_app.core.mongo.Mongo`.
- If the game does NOT already have a saved prediction in the model predictions
  collection, the PredictionService may attempt to compute and persist one.
"""

import argparse
import os
import sys


def _ensure_pythonpath():
    # tests/ is inside nba_app/, so parent of nba_app is one level up
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nba_app_dir = os.path.dirname(script_dir)
    parent = os.path.dirname(nba_app_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def main():
    _ensure_pythonpath()

    from bball_app.core.mongo import Mongo
    from bball_app.core.league_config import load_league_config
    from bball_app.core.services.matchup_chat import Controller
    from bball_app.core.services.matchup_chat.schemas import ControllerOptions

    parser = argparse.ArgumentParser()
    parser.add_argument("--league", default="nba")
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--message", default="What is the model thinking and what does the market say?")
    parser.add_argument("--force-web-refresh", action="store_true")
    parser.add_argument("--show-agent-actions", action="store_true")
    args = parser.parse_args()

    league = load_league_config(args.league, use_cache=False)
    db = Mongo().db

    controller = Controller(db=db, league=league, league_id=league.league_id)
    result = controller.handle_user_message(
        game_id=args.game_id,
        user_message=args.message,
        conversation_history=[],
        options=ControllerOptions(
            force_web_refresh=bool(args.force_web_refresh),
            show_agent_actions=bool(args.show_agent_actions),
        ),
    )

    print("\n=== RESPONSE ===\n")
    print(result.get("response", ""))

    actions = result.get("agent_actions") or []
    print(f"\n=== AGENT_ACTIONS ({len(actions)}) ===\n")
    for a in actions:
        kind = a.get("kind")
        agent = a.get("agent")
        name = a.get("name", "")
        print(f"- {kind} :: {agent} {(':: ' + name) if name else ''}")


if __name__ == "__main__":
    main()

