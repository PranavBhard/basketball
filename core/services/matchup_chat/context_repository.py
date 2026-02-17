from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from bball_app.core.services.matchup_chat.schemas import HistoryEntry, LatestPointer, SharedContext, utc_now_iso


class SharedContextRepository:
    """
    Mongo persistence for the matchup shared context document.

    Keyed by `game_id` (one conversation per game).
    """

    def __init__(self, db, league_id: str = "nba"):
        self.db = db
        self.league_id = (league_id or "nba").lower()

    @property
    def collection_name(self) -> str:
        # Per spec: `{league}_agent_shared_context`
        return f"{self.league_id}_agent_shared_context"

    def get(self, game_id: str) -> Optional[SharedContext]:
        doc = self.db[self.collection_name].find_one({"game_id": game_id})
        if not doc:
            return None
        doc.pop("_id", None)
        return doc

    def ensure_initialized(self, game_id: str, base_doc: SharedContext) -> SharedContext:
        """
        Create the shared context doc if missing. Always returns the current doc.
        """
        now = utc_now_iso()
        base_doc = dict(base_doc or {})
        base_doc.setdefault("game_id", game_id)
        base_doc.setdefault("history", [])
        base_doc.setdefault("latest_by_agent", {})

        self.db[self.collection_name].update_one(
            {"game_id": game_id},
            {"$setOnInsert": {**base_doc, "created_at": now}, "$set": {"updated_at": now}},
            upsert=True,
        )
        return self.get(game_id) or base_doc

    def update_fields(self, game_id: str, fields: Dict[str, Any]) -> None:
        self.db[self.collection_name].update_one(
            {"game_id": game_id},
            {"$set": {**(fields or {}), "updated_at": utc_now_iso()}},
            upsert=True,
        )

    def append_history(self, game_id: str, entry: HistoryEntry) -> Tuple[int, LatestPointer]:
        """
        Append a history entry and update latest_by_agent.

        Returns:
            (history_idx, latest_pointer)
        """
        existing = self.get(game_id) or {"history": [], "latest_by_agent": {}}
        history = existing.get("history") or []
        history_idx = len(history)

        ts = entry.get("timestamp") or utc_now_iso()
        agent = entry.get("agent") or "unknown"
        latest: LatestPointer = {"history_idx": history_idx, "timestamp": ts}

        self.db[self.collection_name].update_one(
            {"game_id": game_id},
            {
                "$push": {"history": {**entry, "timestamp": ts}},
                "$set": {f"latest_by_agent.{agent}": latest, "updated_at": utc_now_iso()},
            },
            upsert=True,
        )
        return history_idx, latest

    def delete(self, game_id: str) -> int:
        """
        Delete the shared context document for this game_id.
        Returns number of documents deleted (0 or 1).
        """
        res = self.db[self.collection_name].delete_one({"game_id": game_id})
        return int(getattr(res, "deleted_count", 0) or 0)

