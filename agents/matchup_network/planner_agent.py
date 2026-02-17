from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from bball_app.agents.matchup_network.base import load_rendered_system_message, simple_chat_completion
from bball_app.agents.utils.json_compression import encode_message_content


def _extract_json_object(text: str) -> str:
    """
    Best-effort extraction of a JSON object from LLM output.
    Handles code fences and extra prose.
    """
    if not text:
        return text
    t = text.strip()
    # Remove ```json fences if present
    t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*```$", "", t)
    # If still not pure JSON, slice from first { to last }
    if not t.lstrip().startswith("{"):
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            t = t[start : end + 1]
    return t


def _json_safe(obj: Any) -> Any:
    """
    Ensure dict/list structures are JSON-serializable for prompt building.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)


def plan_turn(*, shared_context: Dict[str, Any], conversation: List[Dict[str, str]], user_message: str) -> Dict[str, Any]:
    system = load_rendered_system_message("planner")
    shared_encoded = encode_message_content(_json_safe(shared_context))
    convo_encoded = encode_message_content(_json_safe(conversation))
    user = (
        "## Shared Context (toon-encoded JSON)\n"
        f"{shared_encoded}\n\n"
        "## Conversation (toon-encoded JSON)\n"
        f"{convo_encoded}\n\n"
        "## User Message\n"
        f"{user_message}\n"
    )
    out = simple_chat_completion(system=system, user=user, temperature=0.1, json_mode=True)
    try:
        return json.loads(_extract_json_object(out))
    except Exception:
        # Fallback: return a minimal plan if LLM output is not valid JSON
        return {
            "narrative": "Failed to parse planner JSON; using fallback plan.",
            "_raw_planner_output": out,
            "workflow": [
                {"agent": "model_inspector", "instruction": "Explain model prediction drivers."},
                {"agent": "stats_agent", "instruction": "Summarize stats/lineups/injuries."},
                {"agent": "research_media_agent", "instruction": "Summarize news context."},
            ],
            "final_synthesis_instructions": "Answer the user directly using agent findings.",
        }

