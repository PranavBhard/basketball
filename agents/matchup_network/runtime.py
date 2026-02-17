from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Tuple

from bball_app.config import config

try:
    from langchain.agents import create_agent
    from langchain.agents.middleware import AgentMiddleware, ToolCallLimitMiddleware
    from langchain_core.tools import StructuredTool
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False
    StructuredTool = None
    ChatOpenAI = None
    create_agent = None
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    AgentMiddleware = object  # type: ignore
    ToolCallLimitMiddleware = None


ToolFn = Callable[..., Any]


class SafeToolErrorsMiddleware(AgentMiddleware):  # type: ignore[misc]
    """
    Prevent tool-call exceptions (including validation errors) from crashing the agent run.

    Instead, return a ToolMessage with status='error' so the LLM can correct arguments
    and retry using the tool schema + error payload.
    """

    def __init__(self):
        self.tools = []

    def wrap_tool_call(self, request, handler):  # type: ignore[override]
        try:
            return handler(request)
        except Exception as e:
            tool_name = None
            tool_args = None
            tool_call_id = ""
            try:
                tool_name = (request.tool_call or {}).get("name")
                tool_args = (request.tool_call or {}).get("args")
                tool_call_id = (request.tool_call or {}).get("id") or ""
            except Exception:
                pass
            payload = {
                "error": "tool_call_failed",
                "tool": tool_name,
                "args": tool_args,
                "exception_type": type(e).__name__,
                "exception": str(e),
            }
            return ToolMessage(content=json.dumps(payload), tool_call_id=tool_call_id, status="error")


def build_tool(name: str, description: str, fn: ToolFn, *, args_schema: Any = None):
    if not LANGCHAIN_AVAILABLE:
        return None
    return StructuredTool.from_function(
        func=fn,
        name=name,
        description=description,
        args_schema=args_schema,
        infer_schema=(args_schema is None),
    )


def run_agent_with_tools(
    *,
    system_prompt: str,
    user_prompt: str,
    tools: List[Any],
    model: str = "gpt-4o",
    temperature: float = 0.2,
    return_messages: bool = False,
) -> Any:
    if not LANGCHAIN_AVAILABLE:
        msg = "LangChain not available. Install langchain + langchain_openai to enable tool-calling agents."
        return (msg, []) if return_messages else msg

    api_key = config.get("openai_api_key")
    if api_key:
        import os

        os.environ["OPENAI_API_KEY"] = api_key

    llm = ChatOpenAI(model=model, temperature=temperature)
    middleware = [SafeToolErrorsMiddleware()]
    # Hard guard against runaway tool loops (still allows plenty of calls).
    if ToolCallLimitMiddleware is not None:
        middleware.append(ToolCallLimitMiddleware(run_limit=20))

    agent_graph = create_agent(
        model=llm,
        tools=[t for t in tools if t is not None],
        system_prompt=system_prompt,
        middleware=middleware,
        debug=False,
    )

    result = agent_graph.invoke({"messages": [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]})

    # Extract last assistant content
    msgs = []
    if isinstance(result, dict):
        msgs = result.get("messages") or []
    elif hasattr(result, "messages"):
        msgs = result.messages
    else:
        msgs = [result]

    content = None
    for msg in reversed(msgs):
        if hasattr(msg, "content") and msg.content:
            content = msg.content
            break
        if isinstance(msg, dict) and msg.get("content"):
            content = str(msg["content"])
            break
    if content is None:
        content = str(result)

    return (content, msgs) if return_messages else content

