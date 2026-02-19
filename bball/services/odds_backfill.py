"""
Odds backfill service.

Re-fetches ESPN game summaries for games missing pregame_lines and
extracts odds via the existing extract_pregame_lines_from_summary().
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from bball.data.espn_client import ESPNClient
from bball.services.espn_odds import extract_pregame_lines_from_summary

if TYPE_CHECKING:
    from bball.league_config import LeagueConfig


def backfill_espn_odds(
    db,
    league: "LeagueConfig",
    *,
    max_workers: int = 4,
    dry_run: bool = False,
    min_season: Optional[str] = None,
) -> Dict[str, int]:
    """
    Re-fetch ESPN game summaries for games missing pregame_lines.home_ml.

    Returns dict with keys: games_checked, games_updated, games_no_odds, errors
    """
    games_coll = db[league.collections["games"]]

    # Build query for games missing odds
    query: Dict[str, Any] = {"pregame_lines.home_ml": {"$exists": False}}
    if min_season:
        query["season"] = {"$gte": min_season}

    # Only fetch game_id and season for the batch
    cursor = games_coll.find(query, {"game_id": 1, "season": 1, "_id": 0})
    game_docs = list(cursor)

    if not game_docs:
        print("  No games missing odds found.")
        return {"games_checked": 0, "games_updated": 0, "games_no_odds": 0, "errors": 0}

    game_ids = [str(doc["game_id"]) for doc in game_docs]
    print(f"  Found {len(game_ids)} games missing pregame_lines.home_ml")

    if dry_run:
        print(f"  [DRY RUN] Would fetch summaries for {len(game_ids)} games")
        return {"games_checked": len(game_ids), "games_updated": 0, "games_no_odds": 0, "errors": 0}

    # Fetch summaries and extract odds in parallel
    client = ESPNClient(league=league)
    results: List[Dict[str, Any]] = []
    stats = {"games_checked": len(game_ids), "games_updated": 0, "games_no_odds": 0, "errors": 0}

    def _fetch_one(game_id: str) -> Dict[str, Any]:
        """Fetch summary for one game and extract odds."""
        try:
            time.sleep(0.1)  # Rate limit: 100ms per request
            summary = client.get_game_summary(game_id)
            if not summary:
                return {"game_id": game_id, "lines": None, "error": None}
            lines = extract_pregame_lines_from_summary(summary)
            return {"game_id": game_id, "lines": lines, "error": None}
        except Exception as e:
            return {"game_id": game_id, "lines": None, "error": str(e)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_one, gid): gid for gid in game_ids}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Bulk update games that got odds
    updates = []
    for r in results:
        if r["error"]:
            stats["errors"] += 1
        elif r["lines"] and r["lines"].get("home_ml") is not None:
            updates.append(r)
        else:
            stats["games_no_odds"] += 1

    if updates:
        from pymongo import UpdateOne
        bulk_ops = [
            UpdateOne(
                {"game_id": int(u["game_id"]) if u["game_id"].isdigit() else u["game_id"]},
                {"$set": {"pregame_lines": u["lines"]}},
            )
            for u in updates
        ]
        result = games_coll.bulk_write(bulk_ops, ordered=False)
        stats["games_updated"] = result.modified_count
        print(f"  Updated {result.modified_count} games with odds")
    else:
        print("  No new odds found from ESPN")

    if stats["errors"]:
        print(f"  Errors: {stats['errors']}")

    return stats
