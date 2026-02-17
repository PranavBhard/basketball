from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from bball_app.config import config


try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    ChatOpenAI = None
    SystemMessage = None
    HumanMessage = None


def load_rendered_system_message(name: str) -> str:
    """
    Load rendered system message text produced by `agents/generate_system_messages.py`.
    """
    p = Path(__file__).parent / "system_messages" / "rendered" / f"{name}.txt"
    if not p.exists():
        # Fallback: load markdown template directly
        md = Path(__file__).parent / "system_messages" / f"{name}.md"
        return md.read_text(encoding="utf-8") if md.exists() else ""
    return p.read_text(encoding="utf-8")


def simple_chat_completion(
    *,
    system: str,
    user: str,
    model: str = "gpt-4o",
    temperature: float = 0.2,
    json_mode: bool = False,
) -> str:
    """
    Minimal LLM call wrapper (no tool calling).
    """
    if not LANGCHAIN_AVAILABLE:
        return "LangChain not available. Install langchain_openai to enable LLM agents."

    api_key = config.get("openai_api_key")
    if api_key:
        import os

        os.environ["OPENAI_API_KEY"] = api_key

    # Prefer JSON mode when asked (Planner).
    # IMPORTANT: only pass response_format when we actually want it.
    # Some langchain_openai versions will error if response_format=None is provided.
    llm_kwargs: Dict[str, Any] = {"model": model, "temperature": temperature}
    if json_mode:
        llm_kwargs["response_format"] = {"type": "json_object"}

    # Not all LangChain/OpenAI combinations support response_format; fall back gracefully.
    try:
        llm = ChatOpenAI(**llm_kwargs)
    except TypeError:
        # Retry without response_format (covers older clients).
        llm_kwargs.pop("response_format", None)
        llm = ChatOpenAI(**llm_kwargs)
    msgs = [SystemMessage(content=system), HumanMessage(content=user)]
    resp = llm.invoke(msgs)
    return getattr(resp, "content", str(resp))

