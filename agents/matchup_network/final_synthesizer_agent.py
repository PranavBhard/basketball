from __future__ import annotations

import json
from typing import Any, Dict, List

from bball_app.agents.matchup_network.base import load_rendered_system_message, simple_chat_completion

MAX_SECTION_CHARS = 6000
MAX_CONVO_ITEMS = 8


def _json_safe(obj: Any) -> Any:
    """
    Ensure dict/list structures are JSON-serializable for prompt building.
    """
    try:
        return json.loads(json.dumps(obj, default=str))
    except Exception:
        return str(obj)

def _truncate(text: Any, limit: int = MAX_SECTION_CHARS) -> str:
    s = text if isinstance(text, str) else str(text)
    if limit <= 0:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n…(truncated)…"


def _format_conversation_naturally(conversation: List[Dict[str, str]], max_items: int = MAX_CONVO_ITEMS) -> str:
    """
    Format conversation as natural back-and-forth instead of JSON.
    This helps the synthesizer "read the room" and understand context.
    """
    if not conversation:
        return "(no prior conversation)"

    # Keep only recent messages
    recent = conversation[-max_items:] if len(conversation) > max_items else conversation

    lines = []
    for msg in recent:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")

        # Truncate very long messages
        if len(content) > 1500:
            content = content[:1500] + "...(truncated)"

        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"You: {content}")
        else:
            lines.append(f"{role}: {content}")
        lines.append("")  # blank line between messages

    return "\n".join(lines).strip()


def synthesize(
    *,
    shared_context: Dict[str, Any],
    conversation: List[Dict[str, str]],
    turn_plan: Dict[str, Any],
    workflow_outputs: Dict[str, str],
) -> str:
    system = load_rendered_system_message("final_synthesizer")

    # IMPORTANT: keep this prompt small and human-readable.
    safe_shared = _json_safe(shared_context)
    safe_plan = _json_safe(turn_plan)

    final_instr = ""
    try:
        final_instr = str((safe_plan or {}).get("final_synthesis_instructions") or "").strip()
    except Exception:
        final_instr = ""

    # Format conversation naturally (not as JSON) so synthesizer can "read the room"
    formatted_convo = _format_conversation_naturally(conversation)

    parts: List[str] = []
    parts += [
        "## Final synthesis instructions (from planner)",
        final_instr or "(none provided)",
        "",
        "## Shared context (JSON)",
        _truncate(json.dumps(safe_shared, ensure_ascii=False, default=str), 8000),
        "",
        "## Conversation so far",
        "This is the conversation you're continuing. Read it to understand what the user has been asking about and what's already been covered.",
        "",
        formatted_convo,
        "",
        "## Workflow outputs (this turn)",
    ]

    # Present outputs as separate text sections; truncate each.
    for agent_name in ["model_inspector", "stats_agent", "research_media_agent"]:
        out = workflow_outputs.get(agent_name) or ""
        parts += ["", f"### {agent_name}", _truncate(out)]

    # Include any other outputs (e.g., extra agents) without blowing up.
    extra_agents = [k for k in (workflow_outputs or {}).keys() if k not in {"model_inspector", "stats_agent", "research_media_agent"}]
    for agent_name in extra_agents:
        out = workflow_outputs.get(agent_name) or ""
        parts += ["", f"### {agent_name}", _truncate(out)]

    user = "\n".join(parts).strip() + "\n"
    return simple_chat_completion(system=system, user=user, temperature=0.2)

