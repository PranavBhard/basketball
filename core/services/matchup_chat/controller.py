from __future__ import annotations

import json
import hashlib
import re
from typing import Any, Dict, List, Optional, Tuple, Callable

from bball_app.core.services.matchup_chat.context_repository import SharedContextRepository
from bball_app.core.services.matchup_chat.prediction_bootstrap import ensure_shared_context_baseline
from bball_app.core.services.matchup_chat.tool_cache import ToolCache
from bball_app.core.services.matchup_chat.schemas import (
    AgentAction,
    ControllerOptions,
    HistoryEntry,
    SharedContext,
    TurnPlan,
    utc_now_iso,
)

from bball_app.agents.utils.json_compression import encode_message_content, encode_tool_output


class Controller:
    """
    Code-level orchestrator (SSoT) for matchup multi-agent workflow.

    This class calls the Planner to produce a `turn_plan`, then runs the
    specialist agents in order, stores outputs into shared context, and
    returns the final synthesis.
    """

    def __init__(self, db, league=None, league_id: str = "nba"):
        self.db = db
        self.league = league
        self.league_id = (league_id or "nba").lower()
        self.repo = SharedContextRepository(db=db, league_id=self.league_id)
        # Mongo-backed cache for repeated DB tool calls within a matchup.
        self.tool_cache = ToolCache(db=db, league_id=self.league_id, ttl_s=60 * 60 * 12)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def handle_user_message(
        self,
        *,
        game_id: str,
        user_message: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None,
        options: Optional[ControllerOptions] = None,
    ) -> Dict[str, Any]:
        """
        Returns a dict for the web layer containing:
          - response: final synthesis text
          - agent_actions: chronological agent/tool events (if enabled)
          - turn_plan: planner output (best-effort)
        """
        options = options or ControllerOptions()
        conversation_history = conversation_history or []

        # 1) Ensure shared context exists + baseline prediction is present
        shared = self.repo.get(game_id) or {}
        self.repo.ensure_initialized(game_id, {"game_id": game_id, "history": [], "latest_by_agent": {}})

        baseline_fields, _prediction_info = ensure_shared_context_baseline(
            db=self.db, league=self.league, game_id=game_id, league_id=self.league_id, existing=shared
        )
        self.repo.update_fields(game_id, baseline_fields)
        shared = self.repo.get(game_id) or baseline_fields

        agent_actions: List[AgentAction] = []

        # 2) Planner (LLM) -> JSON plan (fallback if invalid/unavailable)
        turn_plan: TurnPlan
        try:
            from bball_app.agents.matchup_network.planner_agent import plan_turn

            planner_shared = self._shared_context_for_agent("planner", shared)
            turn_plan = plan_turn(
                shared_context=planner_shared,
                conversation=self._conversation_for_agent("planner", conversation_history),
                user_message=user_message,
            )
        except Exception:
            turn_plan = self._default_turn_plan(user_message=user_message)

        # If planner fell back, record raw output for debugging in agent actions.
        if isinstance(turn_plan, dict) and turn_plan.get("_raw_planner_output"):
            self._record_tool_event(
                game_id=game_id,
                agent="planner",
                tool_name="planner_raw_output",
                args={},
                output=str(turn_plan.get("_raw_planner_output")),
                agent_actions=agent_actions,
            )

        # Record planner output with system ref (for debugging/traceability)
        try:
            from bball_app.agents.matchup_network.base import load_rendered_system_message

            planner_system = load_rendered_system_message("planner")
            planner_system_ref = f"rendered:planner.txt sha256:{hashlib.sha256(planner_system.encode('utf-8')).hexdigest()[:12]}"
        except Exception:
            planner_system_ref = ""

        self._record_agent_output(
            game_id=game_id,
            agent="planner",
            output=json.dumps(turn_plan, indent=2),
            agent_actions=agent_actions,
            system=planner_system_ref,
        )

        # -----------------------------------------------------------------
        # First-turn outcome guardrail: if this is the first user turn of the
        # session and they are asking "who wins", force full coverage.
        # -----------------------------------------------------------------
        def _is_first_turn(convo: List[Dict[str, Any]]) -> bool:
            # If there are no assistant messages yet, treat as first turn.
            try:
                return not any((m or {}).get("role") == "assistant" for m in (convo or []))
            except Exception:
                return True

        def _looks_like_outcome_question(msg: str) -> bool:
            s = (msg or "").strip().lower()
            if not s:
                return False
            # conservative keyword set
            keys = [
                "who wins",
                "who's gonna win",
                "whos gonna win",
                "who will win",
                "who do you like",
                "pick",
                "winner",
                "win probability",
                "moneyline",
                "ml",
                "favored",
                "underdog",
                "who covers",
                "cover",
            ]
            return any(k in s for k in keys)

        try:
            if _is_first_turn(conversation_history) and _looks_like_outcome_question(user_message):
                desired = [
                    ("model_inspector", "Explain the model prediction drivers; include AuditChecklistJSON."),
                    ("stats_agent", "Execute the Model Inspector's AuditChecklistJSON audits using your tools. Report supports/contradicts/inconclusive + implications."),
                    ("research_media_agent", "Summarize relevant news/injury context and recency; cite sources/links."),
                ]
                wf = turn_plan.get("workflow", []) if isinstance(turn_plan, dict) else []
                wf = wf if isinstance(wf, list) else []
                existing = {((s or {}).get("agent") or "") for s in wf if isinstance(s, dict)}
                # Preserve any existing steps, but ensure all desired agents appear (and in preferred order at start).
                new_wf = []
                for a, instr in desired:
                    if a not in existing:
                        new_wf.append({"agent": a, "instruction": instr})
                # Append original workflow after, skipping duplicates
                for s in wf:
                    if not isinstance(s, dict):
                        continue
                    a = s.get("agent") or ""
                    if a in {d[0] for d in desired}:
                        # If planner included one of the desired agents, keep its instruction but maintain order by appending here.
                        pass
                    new_wf.append(s)
                # De-dupe while preserving order
                seen = set()
                deduped = []
                for s in new_wf:
                    if not isinstance(s, dict):
                        continue
                    a = s.get("agent") or ""
                    if not a or a in seen:
                        continue
                    seen.add(a)
                    deduped.append(s)
                turn_plan["workflow"] = deduped
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Guardrail: if model_inspector runs, ensure stats_agent follows it
        # to execute AuditChecklistJSON (unless stats is already scheduled).
        # -----------------------------------------------------------------
        try:
            wf = turn_plan.get("workflow", []) if isinstance(turn_plan, dict) else []
            if isinstance(wf, list) and wf:
                has_stats = any((s or {}).get("agent") == "stats_agent" for s in wf if isinstance(s, dict))
                if not has_stats:
                    new_wf = []
                    inserted = False
                    for s in wf:
                        new_wf.append(s)
                        if not inserted and isinstance(s, dict) and (s.get("agent") == "model_inspector"):
                            new_wf.append(
                                {
                                    "agent": "stats_agent",
                                    "instruction": "Execute the Model Inspector's AuditChecklistJSON audits for this matchup using your tools. Report supports/contradicts/inconclusive + implications.",
                                }
                            )
                            inserted = True
                    if inserted:
                        turn_plan["workflow"] = new_wf
        except Exception:
            pass

        def _extract_labeled_json_object(text: str, label: str) -> Optional[Dict[str, Any]]:
            """
            Extract a single JSON object that appears after a label line like:
              AuditResultsJSON: { ... }
            Best-effort: slices from first '{' after the label to the last matching '}'.
            """
            try:
                t = text or ""
                idx = t.find(label)
                if idx < 0:
                    return None
                tail = t[idx + len(label) :]
                start = tail.find("{")
                end = tail.rfind("}")
                if start < 0 or end < 0 or end <= start:
                    return None
                blob = tail[start : end + 1].strip()
                # Strip code fences if present (agents are instructed not to, but be robust)
                blob = re.sub(r"^```(?:json)?\s*", "", blob, flags=re.IGNORECASE)
                blob = re.sub(r"\s*```$", "", blob)
                return json.loads(blob)
            except Exception:
                return None

        # 3) Execute workflow steps (best-effort; LLM agents will replace these stubs)
        workflow_outputs: Dict[str, str] = {}
        did_contradiction_requeue = False
        for step in turn_plan.get("workflow", []):
            agent = step.get("agent") or ""
            instruction = step.get("instruction") or ""
            out = self._run_network_agent(
                game_id=game_id,
                agent=agent,
                instruction=instruction,
                shared=self.repo.get(game_id) or shared,
                conversation_history=self._conversation_for_agent(agent, conversation_history),
                prior_outputs=workflow_outputs,
                user_message=user_message,
                options=options,
                agent_actions=agent_actions,
            )
            workflow_outputs[agent] = out

            # Contradiction loop (bounded): after Stats executes audits, if it reports
            # high-severity contradictions, immediately requeue Model Inspector once.
            if agent == "stats_agent" and (not did_contradiction_requeue):
                audit = _extract_labeled_json_object(out or "", "AuditResultsJSON:")
                contradictions = []
                try:
                    contradictions = (audit or {}).get("contradictions") or []
                except Exception:
                    contradictions = []

                high = []
                if isinstance(contradictions, list):
                    for c in contradictions:
                        if not isinstance(c, dict):
                            continue
                        sev = (c.get("severity") or "").strip().lower()
                        if sev == "high":
                            high.append(c)

                if high:
                    did_contradiction_requeue = True
                    # Keep the packet small: pass only the contradiction list, not full stats output.
                    packet = {
                        "version": 1,
                        "game_id": game_id,
                        "contradictions": high,
                    }
                    followup_instruction = (
                        "Investigate the following high-severity contradiction(s) found by Stats.\n"
                        "Focus on the base model(s) implicated. Reconcile by:\n"
                        "- locating the exact base-model feature values in the prediction artifacts\n"
                        "- checking whether the Stats evidence is measuring the same construct/window\n"
                        "- correcting any swapped home/away interpretation\n"
                        "- explicitly concluding what to trust\n\n"
                        "ContradictionPacketJSON:\n"
                        f"{json.dumps(packet, ensure_ascii=False)}\n"
                    )
                    out2 = self._run_network_agent(
                        game_id=game_id,
                        agent="model_inspector",
                        instruction=followup_instruction,
                        shared=self.repo.get(game_id) or shared,
                        conversation_history=self._conversation_for_agent("model_inspector", conversation_history),
                        prior_outputs=workflow_outputs,
                        user_message=user_message,
                        options=options,
                        agent_actions=agent_actions,
                    )
                    # Overwrite model_inspector output for synthesis so the final answer uses the investigation result.
                    workflow_outputs["model_inspector"] = out2

        # 4) Final synthesis (LLM; fallback if unavailable)
        try:
            from bball_app.agents.matchup_network.final_synthesizer_agent import synthesize

            final_text = synthesize(
                shared_context=self._shared_context_for_agent("final_synthesizer", self.repo.get(game_id) or shared),
                conversation=self._conversation_for_agent("final_synthesizer", conversation_history),
                turn_plan=turn_plan,
                workflow_outputs=workflow_outputs,
            )
        except Exception as e:
            # Record the failure so we can debug why fallback was used.
            self._record_tool_event(
                game_id=game_id,
                agent="final_synthesizer",
                tool_name="final_synthesis_error",
                args={},
                output={"error": type(e).__name__, "message": str(e)},
                agent_actions=agent_actions,
            )
            final_text = self._synthesize_final(
                user_message=user_message,
                shared=self.repo.get(game_id) or shared,
                workflow_outputs=workflow_outputs,
            )
        try:
            from bball_app.agents.matchup_network.base import load_rendered_system_message

            fs_system = load_rendered_system_message("final_synthesizer")
            fs_system_ref = f"rendered:final_synthesizer.txt sha256:{hashlib.sha256(fs_system.encode('utf-8')).hexdigest()[:12]}"
        except Exception:
            fs_system_ref = ""
        self._record_agent_output(
            game_id=game_id,
            agent="final_synthesizer",
            output=final_text,
            agent_actions=agent_actions,
            system=fs_system_ref,
        )

        payload: Dict[str, Any] = {
            "response": final_text,
            "turn_plan": turn_plan,
            # Always return agent_actions so the UI can toggle visibility
            # without re-running the workflow.
            "agent_actions": agent_actions,
        }
        return payload

    # ---------------------------------------------------------------------
    # Internals
    # ---------------------------------------------------------------------
    def _record_agent_output(
        self,
        *,
        game_id: str,
        agent: str,
        output: str,
        agent_actions: List[AgentAction],
        tools: Optional[List[Dict[str, Any]]] = None,
        system: str = "",
    ) -> None:
        ts = utc_now_iso()
        entry: HistoryEntry = {
            "agent": agent,
            "system": system,
            "tools": tools or [],
            "output": output,
            "timestamp": ts,
        }
        self.repo.append_history(game_id, entry)

        agent_actions.append(
            {
                "kind": "agent_output",
                "agent": agent,
                "timestamp": ts,
                "text": output,
            }
        )
        for t in tools or []:
            agent_actions.append(
                {
                    "kind": "tool_call",
                    "agent": agent,
                    "timestamp": ts,
                    "name": t.get("name", ""),
                    "args": t.get("args", {}),
                    "output": t.get("output"),
                }
            )

    def _default_turn_plan(self, *, user_message: str) -> TurnPlan:
        # Temporary deterministic plan until PlannerAgent is implemented.
        return {
            "narrative": f"User asked: {user_message}. Gather core model context, market context, stats + news, then synthesize.",
            "workflow": [
                {"agent": "model_inspector", "instruction": "Explain model prediction drivers and any anomalies."},
                {"agent": "stats_agent", "instruction": "Summarize matchup stats, lineups, injuries, trends."},
                {"agent": "research_media_agent", "instruction": "Summarize news/injury updates and context."},
            ],
            "final_synthesis_instructions": "Answer the user's question directly, cite relevant agent findings, keep it concise.",
        }

    def _shared_context_for_agent(self, agent: str, shared: SharedContext) -> Dict[str, Any]:
        """
        Return the minimal shared-context slice needed for each agent.

        We do NOT pass the full shared context to every agent.
        """
        shared = shared or {}
        game = shared.get("game") or {}
        ensemble = shared.get("ensemble_model") or {}

        if agent == "model_inspector":
            # Model Inspector gets only a tiny baseline anchor from shared context.
            # All rich prediction artifacts are fetched via model_inspector tools.
            return {
                "game_id": shared.get("game_id"),
                "ensemble_model": {
                    "p_home": ensemble.get("p_home"),
                },
            }

        if agent == "stats_agent":
            return {
                "game_id": shared.get("game_id"),
                "game": game,
            }

        if agent == "research_media_agent":
            # No market tool; may reference market_snapshot read-only if present.
            return {
                "game_id": shared.get("game_id"),
                "game": game,
                "market_snapshot": shared.get("market_snapshot"),
            }

        if agent == "experimenter":
            # Needs matchup identifiers and a baseline anchor; it can mutate roster state
            # and rerun predictions via its tools.
            return {
                "game_id": shared.get("game_id"),
                "game": game,
                "ensemble_model": {
                    "p_home": ensemble.get("p_home"),
                },
            }

        if agent == "planner":
            # Planner needs enough to plan, but not the full history.
            return {
                "game_id": shared.get("game_id"),
                "game": game,
                "ensemble_model": {
                    "p_home": ensemble.get("p_home"),
                },
                "market_snapshot": shared.get("market_snapshot"),
                "latest_by_agent": shared.get("latest_by_agent") or {},
            }

        if agent == "final_synthesizer":
            # Needs broad context, but exclude the full history stack (too large).
            out = dict(shared)
            out.pop("history", None)
            out.pop("latest_by_agent", None)
            return out

        # Default: small safe slice
        return {"game_id": shared.get("game_id"), "game": game}

    def _conversation_for_agent(self, agent: str, conversation: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Return the minimal conversation slice for each agent.
        """
        conversation = conversation or []
        # Specialists generally do NOT need full chat history.
        if agent in {"model_inspector", "stats_agent", "experimenter"}:
            return []
        # Research may benefit from last user message context, but keep small.
        if agent == "research_media_agent":
            return conversation[-2:] if len(conversation) > 2 else conversation
        # Planner + final synthesizer can see full conversation (already user+assistant only).
        return conversation

    def _record_tool_event(
        self,
        *,
        game_id: str,
        agent: str,
        tool_name: str,
        args: Dict[str, Any],
        output: Any,
        agent_actions: List[AgentAction],
        system: str = "",
    ) -> None:
        """
        Persist a tool call immediately so the UI can stream it live via shared context history polling.
        """
        ts = utc_now_iso()
        entry: HistoryEntry = {
            "agent": agent,
            **({"system": system} if system else {}),
            "tools": [{"name": tool_name, "args": args or {}, "output": output}],
            "timestamp": ts,
        }
        self.repo.append_history(game_id, entry)
        agent_actions.append(
            {
                "kind": "tool_call",
                "agent": agent,
                "timestamp": ts,
                "name": tool_name,
                "args": args or {},
                "output": output,
            }
        )

        # Note: market_snapshot is now populated at baseline bootstrap time.

    def _run_network_agent(
        self,
        *,
        game_id: str,
        agent: str,
        instruction: str,
        shared: SharedContext,
        conversation_history: List[Dict[str, Any]],
        prior_outputs: Dict[str, str],
        user_message: str,
        options: ControllerOptions,
        agent_actions: List[AgentAction],
    ) -> str:
        """
        Run a matchup network agent.

        Uses LangChain tool calling when available; falls back to the old stub runner otherwise.
        """
        from bball_app.agents.matchup_network.base import load_rendered_system_message
        from bball_app.agents.matchup_network.runtime import LANGCHAIN_AVAILABLE, build_tool, run_agent_with_tools
        from pydantic import BaseModel

        # If tool-calling runtime isn't available, fall back to deterministic stub behavior.
        if not LANGCHAIN_AVAILABLE:
            out, tools = self._run_stub_agent(agent=agent, instruction=instruction, shared=shared, options=options)
            self._record_agent_output(game_id=game_id, agent=agent, output=out, tools=tools, agent_actions=agent_actions)
            return out

        system = load_rendered_system_message(agent)
        system_ref = f"rendered:{agent}.txt sha256:{hashlib.sha256(system.encode('utf-8')).hexdigest()[:12]}"

        shared_slice = self._shared_context_for_agent(agent, shared)
        game_meta = (shared_slice.get("game") or {}) if isinstance(shared_slice, dict) else {}
        home_meta = (game_meta.get("home") or {}) if isinstance(game_meta, dict) else {}
        away_meta = (game_meta.get("away") or {}) if isinstance(game_meta, dict) else {}
        home_team_id = home_meta.get("team_id")
        away_team_id = away_meta.get("team_id")

        # Encode JSON blocks with toon encoding to reduce tokens.
        shared_encoded = encode_message_content(shared_slice)
        convo_encoded = encode_message_content(conversation_history)

        user_prompt_parts = [
            "## User Message",
            user_message,
            "",
            "## Instruction",
            instruction,
            "",
            "## Shared Context (toon-encoded JSON)",
            shared_encoded if isinstance(shared_encoded, str) else str(shared_encoded),
        ]

        # Extra grounding to prevent home/away confusion in model analysis.
        if agent == "model_inspector":
            home_name = home_meta.get("full_name") or home_meta.get("name") or ""
            away_name = away_meta.get("full_name") or away_meta.get("name") or ""
            p_home = ((shared_slice.get("ensemble_model") or {}) if isinstance(shared_slice, dict) else {}).get("p_home")
            user_prompt_parts += [
                "",
                "## Home/Away grounding (critical)",
                f"- Away: {away_name} (team_id={away_team_id})",
                f"- Home: {home_name} (team_id={home_team_id})",
                f"- p_home (home win prob): {p_home}",
                "- Use this mapping consistently. Do not swap home/away mid-report.",
            ]

        # Explicit arg hints to reduce tool-call failures.
        if agent in {"stats_agent", "research_media_agent", "experimenter"}:
            hint_lines = [
                "## Tool argument hints",
                "- Team IDs are in shared context at `game.home.team_id` and `game.away.team_id`.",
                f"- home team_id: {home_team_id}",
                f"- away team_id: {away_team_id}",
            ]
            if agent == "stats_agent":
                hint_lines += [
                    '- Valid `window` examples: "days5", "games10", "games12", "games18", "season".',
                    "- Example calls:",
                    f'  - get_lineups(team_id="{home_team_id}")',
                    f'  - get_team_games(team_id="{home_team_id}", window="games10")',
                ]
            if agent == "research_media_agent":
                hint_lines += [
                    "- Use `get_team_news(team_id=...)` with one of the team_ids above.",
                ]
            if agent == "experimenter":
                hint_lines += [
                    '- Valid `bucket` values: "injured", "bench", "starter".',
                    "- Use `get_lineups(team_id=...)` first if you need a player_id.",
                    "- Reminder: roster changes persist platform-wide until changed again.",
                ]
            user_prompt_parts += ["", *hint_lines]

        # Provide minimal conversation only when this agent needs it.
        if conversation_history:
            user_prompt_parts += ["", "## Conversation (toon-encoded JSON)", convo_encoded if isinstance(convo_encoded, str) else str(convo_encoded)]

        # Provide Stats output to Research/Media as required dependency.
        if agent == "research_media_agent":
            stats_out = prior_outputs.get("stats_agent") or ""
            if stats_out:
                user_prompt_parts += ["", "## Stats Agent Output (this turn)", stats_out]

        # Provide Model Inspector output to Stats Agent for audit execution.
        if agent == "stats_agent":
            mi_out = prior_outputs.get("model_inspector") or ""
            if mi_out:
                user_prompt_parts += ["", "## Model Inspector Output (this turn)", mi_out]

        # Provide Situational Predictor output (scenario snapshot ids) to Model Inspector.
        if agent == "model_inspector":
            sp_out = prior_outputs.get("experimenter") or ""
            if sp_out:
                user_prompt_parts += ["", "## Situational Predictor Output (this turn)", sp_out]

        user_prompt = "\n".join(user_prompt_parts)

        # Tool wrapper helper that records calls immediately to shared context
        def recording(fn: Callable[..., Any], tool_name: str):
            def _wrapped(**kwargs):
                # Best-effort DB-backed cache for repeated Stats tool calls.
                cacheable_stats_tools = {"get_team_stats", "compare_team_stats", "get_rotation_stats", "get_team_games", "get_player_stats", "get_advanced_player_stats", "get_head_to_head_games", "get_head_to_head_stats"}
                cache_hit = False
                out_raw = None
                if agent == "stats_agent" and tool_name in cacheable_stats_tools:
                    try:
                        cached = self.tool_cache.get(game_id=game_id, tool=tool_name, args=kwargs)
                    except Exception:
                        cached = None
                    if cached is not None:
                        out_raw = cached
                        cache_hit = True

                if out_raw is None:
                    out_raw = fn(**kwargs)
                    if agent == "stats_agent" and tool_name in cacheable_stats_tools:
                        try:
                            self.tool_cache.set(game_id=game_id, tool=tool_name, args=kwargs, output=out_raw)
                        except Exception:
                            pass

                # Persist raw output for UI/debugging (annotate cache hits without changing tool return to LLM)
                record_output = {"cached": True, "value": out_raw} if cache_hit else out_raw
                self._record_tool_event(
                    game_id=game_id,
                    agent=agent,
                    tool_name=tool_name,
                    args=kwargs,
                    output=record_output,
                    agent_actions=agent_actions,
                    system=system_ref,
                )
                # Return toon-encoded tool output to keep the agent context window small.
                return encode_tool_output(out_raw)

            return _wrapped

        tools = []
        if agent == "research_media_agent":
            from bball_app.core.services.news_tools import get_game_news, get_team_news, get_player_news, web_search

            def _get_game_news(game_id: str = game_id, force_refresh: bool = False):
                return get_game_news(
                    game_id,
                    force_refresh=force_refresh or options.force_web_refresh,
                    league=self.league,
                    league_id=self.league_id,
                    db=self.db,
                )

            def _get_team_news(team_id: str, force_refresh: bool = False):
                return get_team_news(
                    team_id,
                    force_refresh=force_refresh or options.force_web_refresh,
                    league=self.league,
                    league_id=self.league_id,
                    db=self.db,
                )

            def _get_player_news(player_id: str, force_refresh: bool = False):
                return get_player_news(
                    player_id,
                    force_refresh=force_refresh or options.force_web_refresh,
                    league=self.league,
                    league_id=self.league_id,
                    db=self.db,
                )

            def _web_search(query: str, force_refresh: bool = False, num_results: int = 5):
                return web_search(
                    query,
                    force_refresh=force_refresh or options.force_web_refresh,
                    num_results=num_results,
                    league_id=self.league_id,
                )

            class _GameNewsArgs(BaseModel):
                game_id: str
                force_refresh: bool = False

            class _TeamNewsArgs(BaseModel):
                team_id: str
                force_refresh: bool = False

            class _PlayerNewsArgs(BaseModel):
                player_id: str
                force_refresh: bool = False

            class _WebSearchArgs(BaseModel):
                query: str
                force_refresh: bool = False
                num_results: int = 5

            tools = [
                build_tool(
                    "web_search",
                    "General web search (fallback / broader context).",
                    recording(_web_search, "web_search"),
                    args_schema=_WebSearchArgs,
                ),
                build_tool(
                    "get_game_news",
                    "Web search news for this game_id.",
                    recording(_get_game_news, "get_game_news"),
                    args_schema=_GameNewsArgs,
                ),
                build_tool(
                    "get_team_news",
                    "Web search news for a team_id. team_id is required (see shared context game.home.team_id / game.away.team_id).",
                    recording(_get_team_news, "get_team_news"),
                    args_schema=_TeamNewsArgs,
                ),
                build_tool(
                    "get_player_news",
                    "Web search news for a player_id. player_id is required (usually from get_lineups output).",
                    recording(_get_player_news, "get_player_news"),
                    args_schema=_PlayerNewsArgs,
                ),
            ]

        elif agent == "stats_agent":
            from bball_app.core.services.lineup_service import get_lineups
            from bball_app.agents.tools.team_game_window_tools import (
                get_team_games, get_team_stats, compare_team_stats,
                get_head_to_head_games, get_head_to_head_stats,
            )
            from bball_app.agents.tools.window_player_stats_tools import get_player_stats, get_advanced_player_stats, get_rotation_stats
            from bball_app.agents.tools.code_executor import CodeExecutor

            # Stats agent can optionally run code
            code_exec = CodeExecutor(db=self.db)

            def _get_lineups(team_id: str):
                return get_lineups(team_id, game_id=game_id, db=self.db, league=self.league)

            def _get_team_stats(team_id: str, window: str, split: Optional[str] = None):
                return get_team_stats(team_id, window, split=split, game_id=game_id, db=self.db, league=self.league)

            def _compare_team_stats(team_a_id: str, team_b_id: str, window: str):
                return compare_team_stats(team_a_id, team_b_id, window, game_id=game_id, db=self.db, league=self.league)

            def _get_team_games(team_id: str, window: str, split: Optional[str] = None):
                return get_team_games(team_id, window, split=split, game_id=game_id, db=self.db, league=self.league)

            def _get_head_to_head_games(team_a_id: str, team_b_id: str, window: str = "season"):
                return get_head_to_head_games(team_a_id, team_b_id, window, game_id=game_id, db=self.db, league=self.league)

            def _get_head_to_head_stats(team_a_id: str, team_b_id: str, window: str = "season"):
                return get_head_to_head_stats(team_a_id, team_b_id, window, game_id=game_id, db=self.db, league=self.league)

            def _get_player_stats(player_id: str, window: str, split: Optional[str] = None):
                return get_player_stats(player_id, window, split=split, game_id=game_id, db=self.db, league=self.league)

            def _get_adv_player_stats(player_id: str, window: str, split: Optional[str] = None):
                return get_advanced_player_stats(player_id, window, split=split, db=self.db, game_id=game_id, league=self.league)

            def _get_rotation_stats(team_id: str, window: str):
                return get_rotation_stats(team_id, window, game_id=game_id, db=self.db, league=self.league)

            class _LineupsArgs(BaseModel):
                team_id: str

            class _RotationStatsArgs(BaseModel):
                team_id: str
                window: str

            class _TeamStatsArgs(BaseModel):
                team_id: str
                window: str
                split: Optional[str] = None

            class _CompareTeamStatsArgs(BaseModel):
                team_a_id: str
                team_b_id: str
                window: str

            class _TeamGamesArgs(BaseModel):
                team_id: str
                window: str
                split: Optional[str] = None

            class _HeadToHeadArgs(BaseModel):
                team_a_id: str
                team_b_id: str
                window: str = "season"

            class _PlayerStatsArgs(BaseModel):
                player_id: str
                window: str
                split: Optional[str] = None

            class _RunCodeArgs(BaseModel):
                code: str

            tools = [
                build_tool(
                    "get_lineups",
                    "Get starters/bench/injured for a team_id (required).",
                    recording(_get_lineups, "get_lineups"),
                    args_schema=_LineupsArgs,
                ),
                build_tool(
                    "get_team_stats",
                    'Get pre-computed team aggregates (wins, losses, averages). USE THIS for team records instead of counting games. Args: team_id, window ("season", "games10", "games12"), optional split ("home"/"away").',
                    recording(_get_team_stats, "get_team_stats"),
                    args_schema=_TeamStatsArgs,
                ),
                build_tool(
                    "compare_team_stats",
                    "Compare two teams' stats side by side with deltas. Args: team_a_id, team_b_id, window.",
                    recording(_compare_team_stats, "compare_team_stats"),
                    args_schema=_CompareTeamStatsArgs,
                ),
                build_tool(
                    "get_rotation_stats",
                    "Get pre-computed rotation/talent aggregates (top-1 PER, top-3 avg, starter avg, MPG-weighted PER). USE THIS for player talent audits. Args: team_id, window.",
                    recording(_get_rotation_stats, "get_rotation_stats"),
                    args_schema=_RotationStatsArgs,
                ),
                build_tool(
                    "get_team_games",
                    'Get individual game results for trend analysis (NOT for counting records - use get_team_stats). NOT for head-to-head - use get_head_to_head_games. Args: team_id, window, optional split.',
                    recording(_get_team_games, "get_team_games"),
                    args_schema=_TeamGamesArgs,
                ),
                build_tool(
                    "get_head_to_head_games",
                    'Get games where two specific teams played EACH OTHER (head-to-head). This is the ONLY tool for H2H game lists. Args: team_a_id, team_b_id, window ("season", "games5", "seasons2").',
                    recording(_get_head_to_head_games, "get_head_to_head_games"),
                    args_schema=_HeadToHeadArgs,
                ),
                build_tool(
                    "get_head_to_head_stats",
                    'Get aggregated head-to-head record and stats between two teams. Use for H2H record verification. Args: team_a_id, team_b_id, window ("season", "games5", "seasons2").',
                    recording(_get_head_to_head_stats, "get_head_to_head_stats"),
                    args_schema=_HeadToHeadArgs,
                ),
                build_tool(
                    "get_player_stats",
                    "Get windowed raw player game stats.",
                    recording(_get_player_stats, "get_player_stats"),
                    args_schema=_PlayerStatsArgs,
                ),
                build_tool(
                    "get_advanced_player_stats",
                    "Get lightweight derived player stats over a window.",
                    recording(_get_adv_player_stats, "get_advanced_player_stats"),
                    args_schema=_PlayerStatsArgs,
                ),
                build_tool(
                    "run_code",
                    "Execute python for ad-hoc aggregation (optional).",
                    recording(code_exec.run_code, "run_code"),
                    args_schema=_RunCodeArgs,
                ),
            ]

        elif agent == "experimenter":
            from bball_app.core.services.lineup_service import get_lineups
            from bball_app.agents.tools.experimenter_tools import (
                predict_game_and_persist,
                set_player_lineup_bucket,
            )

            def _get_lineups(team_id: str):
                return get_lineups(team_id, game_id=game_id, db=self.db, league=self.league)

            def _set_player_lineup_bucket(player_id: str, bucket: str):
                return set_player_lineup_bucket(player_id, bucket, game_id=game_id, db=self.db, league=self.league)

            def _predict():
                return predict_game_and_persist(game_id=game_id, db=self.db, league=self.league)

            class _LineupsArgs(BaseModel):
                team_id: str

            class _SetBucketArgs(BaseModel):
                player_id: str
                bucket: str

            class _PredictArgs(BaseModel):
                pass

            tools = [
                build_tool(
                    "get_lineups",
                    "Get starters/bench/injured for a team_id (required).",
                    recording(_get_lineups, "get_lineups"),
                    args_schema=_LineupsArgs,
                ),
                build_tool(
                    "set_player_lineup_bucket",
                    'Move a player to a roster bucket (bucket: "injured" | "bench" | "starter"). Persists in nba_rosters.',
                    recording(_set_player_lineup_bucket, "set_player_lineup_bucket"),
                    args_schema=_SetBucketArgs,
                ),
                build_tool(
                    "predict",
                    "Run the standard core PredictionService for this game_id and persist to model_predictions.",
                    recording(_predict, "predict"),
                    args_schema=_PredictArgs,
                ),
            ]

        elif agent == "model_inspector":
            from bball_app.agents.tools.model_inspector_tools import (
                get_base_model_direction_table,
                get_ensemble_meta_model_params,
                get_prediction_base_outputs,
                get_prediction_doc,
                get_prediction_feature_values,
                get_prediction_snapshot_base_outputs,
                get_prediction_snapshot_doc,
                get_selected_configs,
            )

            def _get_base_model_direction_table(game_id: str = game_id):
                return get_base_model_direction_table(game_id, db=self.db, league=self.league)

            def _get_selected_configs():
                return get_selected_configs(db=self.db, league=self.league)

            def _get_prediction_doc(game_id: str = game_id):
                return get_prediction_doc(game_id, db=self.db, league=self.league)

            def _get_prediction_feature_values(game_id: str = game_id, keys: Optional[List[str]] = None):
                return get_prediction_feature_values(game_id, keys=keys, db=self.db, league=self.league)

            def _get_prediction_base_outputs(game_id: str = game_id):
                return get_prediction_base_outputs(game_id, db=self.db, league=self.league)

            def _get_ensemble_meta_model_params(game_id: str = game_id):
                return get_ensemble_meta_model_params(game_id, db=self.db, league=self.league)

            def _get_prediction_snapshot_doc(snapshot_id: str):
                return get_prediction_snapshot_doc(snapshot_id, db=self.db, league=self.league)

            def _get_prediction_snapshot_base_outputs(snapshot_id: str):
                return get_prediction_snapshot_base_outputs(snapshot_id, db=self.db, league=self.league)

            class _PredDocArgs(BaseModel):
                game_id: str

            class _PredFeatArgs(BaseModel):
                game_id: str
                keys: Optional[List[str]] = None

            class _SnapshotArgs(BaseModel):
                snapshot_id: str

            tools = [
                build_tool(
                    "get_base_model_direction_table",
                    "⭐ USE FIRST: Get pre-computed direction table showing which team each base model favors. Returns favors=HOME or AWAY for each base model. Do NOT override these directions.",
                    recording(_get_base_model_direction_table, "get_base_model_direction_table"),
                    args_schema=_PredDocArgs,
                ),
                build_tool("get_selected_configs", "Get currently selected classifier + points model configs.", recording(_get_selected_configs, "get_selected_configs")),
                build_tool(
                    "get_ensemble_meta_model_params",
                    "Get meta-model coefficients/intercept and meta_feature_cols for contribution analysis.",
                    recording(_get_ensemble_meta_model_params, "get_ensemble_meta_model_params"),
                    args_schema=_PredDocArgs,
                ),
                build_tool(
                    "get_prediction_snapshot_doc",
                    "Get ensemble base model breakdown for a scenario snapshot_id.",
                    recording(_get_prediction_snapshot_doc, "get_prediction_snapshot_doc"),
                    args_schema=_SnapshotArgs,
                ),
                build_tool(
                    "get_prediction_snapshot_base_outputs",
                    "Get ensemble meta_feature_values for a scenario snapshot_id.",
                    recording(_get_prediction_snapshot_base_outputs, "get_prediction_snapshot_base_outputs"),
                    args_schema=_SnapshotArgs,
                ),
                build_tool(
                    "get_prediction_doc",
                    "Get ensemble base model breakdown for this game_id.",
                    recording(_get_prediction_doc, "get_prediction_doc"),
                    args_schema=_PredDocArgs,
                ),
                build_tool(
                    "get_prediction_feature_values",
                    "Get features_dict for this game_id; optionally filter by keys.",
                    recording(_get_prediction_feature_values, "get_prediction_feature_values"),
                    args_schema=_PredFeatArgs,
                ),
                build_tool(
                    "get_prediction_base_outputs",
                    "Get ensemble meta_feature_values for this game_id.",
                    recording(_get_prediction_base_outputs, "get_prediction_base_outputs"),
                    args_schema=_PredDocArgs,
                ),
            ]

        else:
            # Unknown agent: fall back
            out, tools_used = self._run_stub_agent(agent=agent, instruction=instruction, shared=shared, options=options)
            self._record_agent_output(game_id=game_id, agent=agent, output=out, tools=tools_used, agent_actions=agent_actions)
            return out

        try:
            out, trace_msgs = run_agent_with_tools(
                system_prompt=system,
                user_prompt=user_prompt,
                tools=tools,
                temperature=0.2,
                return_messages=True,
            )
            self._record_tool_errors_from_trace(
                game_id=game_id,
                agent=agent,
                system_ref=system_ref,
                trace_msgs=trace_msgs,
                agent_actions=agent_actions,
            )
        except Exception as e:
            # Do not fail the entire request on a single tool/agent error.
            out = f"[ERROR] {agent} failed: {e}"
            self._record_agent_output(game_id=game_id, agent=agent, output=out, agent_actions=agent_actions, system=system_ref)
            return out

        self._record_agent_output(game_id=game_id, agent=agent, output=out, agent_actions=agent_actions, system=system_ref)
        return out

    def _record_tool_errors_from_trace(
        self,
        *,
        game_id: str,
        agent: str,
        system_ref: str,
        trace_msgs: List[Any],
        agent_actions: List[AgentAction],
    ) -> None:
        """
        Persist tool-call failures that happen before our python tool wrapper executes
        (e.g., schema validation errors -> ToolMessage(status='error')).
        """
        for m in trace_msgs or []:
            try:
                # LangChain ToolMessage has type='tool' and a string `content`
                m_type = getattr(m, "type", None) or (m.get("type") if isinstance(m, dict) else None)
                if m_type != "tool":
                    continue
                content = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
                if not content or not isinstance(content, str):
                    continue
                payload = None
                try:
                    payload = json.loads(content)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("error") != "tool_call_failed":
                    continue
                tool_name = payload.get("tool") or "tool"
                args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
                self._record_tool_event(
                    game_id=game_id,
                    agent=agent,
                    tool_name=str(tool_name),
                    args=args or {},
                    output=payload,
                    agent_actions=agent_actions,
                    system=system_ref,
                )
            except Exception:
                continue

    def _run_stub_agent(
        self,
        *,
        agent: str,
        instruction: str,
        shared: SharedContext,
        options: ControllerOptions,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Best-effort non-LLM implementation for early integration.

        Returns:
          (output_text, tool_calls_for_history)
        """
        tools: List[Dict[str, Any]] = []

        game = (shared.get("game") or {})
        home = (game.get("home") or {}).get("name", "")
        away = (game.get("away") or {}).get("name", "")
        full_home = (game.get("home") or {}).get("full_name", "") or home
        full_away = (game.get("away") or {}).get("full_name", "") or away
        p_home = ((shared.get("ensemble_model") or {}).get("p_home"))
        pred_info = {}

        if agent == "model_inspector":
            # Minimal stub: fetch prediction doc directly from SSoT when shared context doesn't contain it
            from bball_app.core.services.prediction import PredictionService

            svc = PredictionService(db=self.db, league=self.league)
            pred_info = svc.get_prediction_for_game(game_id) or {}

            hw = pred_info.get("home_win_prob")
            aw = pred_info.get("away_win_prob")
            winner = pred_info.get("predicted_winner")
            features = pred_info.get("features_dict") or {}
            top_feats = list(features.items())[:10]
            lines = [
                f"{full_away} @ {full_home}",
                f"Predicted winner: {winner}",
                f"Home win prob: {hw}%",
                f"Away win prob: {aw}%",
                "",
                "Feature snapshot (first 10):",
                *(f"- {k}: {v}" for k, v in top_feats),
            ]
            return "\n".join(lines), tools

        if agent == "stats_agent":
            from bball_app.agents.tools.game_tools import get_rosters

            # Use roster tool as best-effort lineup/injury view (sync with prediction inputs is handled by core prediction flow)
            home_roster = get_rosters(home, db=self.db) if home else {}
            away_roster = get_rosters(away, db=self.db) if away else {}
            tools.append({"name": "get_rosters", "args": {"team": home}, "output": home_roster})
            tools.append({"name": "get_rosters", "args": {"team": away}, "output": away_roster})

            def summarize_roster(r: Dict[str, Any]) -> str:
                players = r.get("roster") or []
                starters = [p for p in players if p.get("starter")]
                injured = [p for p in players if p.get("injured")]
                return f"{len(players)} players; starters={len(starters)}; injured={len(injured)}"

            return (
                "\n".join(
                    [
                        f"{full_away} @ {full_home}",
                        f"{away}: {summarize_roster(away_roster)}",
                        f"{home}: {summarize_roster(home_roster)}",
                        "",
                        "Note: stats narratives agent is not yet running deep windows/h2h/travel in this stub.",
                    ]
                ),
                tools,
            )

        if agent == "research_media_agent":
            from bball_app.core.services.news_tools import get_game_news, get_team_news

            game_id = shared.get("game_id", "")
            away_team_id = (game.get("away") or {}).get("team_id", "")
            home_team_id = (game.get("home") or {}).get("team_id", "")

            game_news = get_game_news(
                game_id,
                force_refresh=options.force_web_refresh,
                league=self.league,
                league_id=self.league_id,
                db=self.db,
            )
            tools.append(
                {
                    "name": "get_game_news",
                    "args": {"game_id": game_id, "force_refresh": options.force_web_refresh},
                    "output": game_news,
                }
            )

            home_news = get_team_news(
                home_team_id,
                force_refresh=options.force_web_refresh,
                league=self.league,
                league_id=self.league_id,
                db=self.db,
            ) if home_team_id else []
            away_news = get_team_news(
                away_team_id,
                force_refresh=options.force_web_refresh,
                league=self.league,
                league_id=self.league_id,
                db=self.db,
            ) if away_team_id else []

            if home_team_id:
                tools.append(
                    {
                        "name": "get_team_news",
                        "args": {"team_id": home_team_id, "force_refresh": options.force_web_refresh},
                        "output": home_news,
                    }
                )
            if away_team_id:
                tools.append(
                    {
                        "name": "get_team_news",
                        "args": {"team_id": away_team_id, "force_refresh": options.force_web_refresh},
                        "output": away_news,
                    }
                )

            def top_sources(items: List[Dict[str, Any]]) -> List[str]:
                srcs = []
                for it in items[:3]:
                    md = it.get("metadata") or {}
                    title = md.get("title") or ""
                    link = md.get("link") or ""
                    src = it.get("source") or "web"
                    srcs.append(f"- {src}: {title} ({link})")
                return srcs

            lines = [
                f"{full_away} @ {full_home}",
                "",
                "Top game news sources (first 3):",
                *(top_sources(game_news) or ["- (none)"]),
                "",
                f"Top {away} news sources (first 3):",
                *(top_sources(away_news) or ["- (none)"]),
                "",
                f"Top {home} news sources (first 3):",
                *(top_sources(home_news) or ["- (none)"]),
            ]
            return "\n".join(lines), tools

        # Fallback
        return (
            f"[{agent}] (stub)\n"
            f"Instruction: {instruction}\n"
            f"Game: {away} @ {home}\n"
            f"p_home: {p_home}\n",
            tools,
        )

    def _synthesize_final(
        self,
        *,
        user_message: str,
        shared: SharedContext,
        workflow_outputs: Dict[str, str],
    ) -> str:
        # Fallback synthesis (used when the LLM final synthesizer is unavailable or fails).
        # This must read like a normal user-facing answer (no internal stub labels).
        game = shared.get("game") or {}
        home = (game.get("home") or {}).get("full_name") or (game.get("home") or {}).get("name") or ""
        away = (game.get("away") or {}).get("full_name") or (game.get("away") or {}).get("name") or ""
        date = (game.get("date") or "")
        p_home = (shared.get("ensemble_model") or {}).get("p_home")

        def _american_odds(p: float) -> Optional[int]:
            try:
                p = float(p)
                if p <= 0 or p >= 1:
                    return None
                if p >= 0.5:
                    return int(round(-100.0 * (p / (1.0 - p))))
                return int(round(100.0 * ((1.0 - p) / p)))
            except Exception:
                return None

        if isinstance(p_home, (int, float)):
            p_home_f = float(p_home)
        else:
            p_home_f = None

        header = f"{away} @ {home}".strip()
        if date:
            header = f"{header} ({date})"

        lines: List[str] = [header]

        if p_home_f is None:
            # No probability available: provide best-effort from any workflow output
            lines.append("I don’t have a usable model win probability for the home team in context yet.")
        else:
            p_away = 1.0 - p_home_f
            favored = home if p_home_f > 0.5 else away
            home_odds = _american_odds(p_home_f)
            away_odds = _american_odds(p_away)

            lines.append(f"The model favors **{favored}**.")
            lines.append(
                f"- {home}: {p_home_f*100:.1f}% (implied {home_odds:+d} moneyline)"
                if home_odds is not None
                else f"- {home}: {p_home_f*100:.1f}%"
            )
            lines.append(
                f"- {away}: {p_away*100:.1f}% (implied {away_odds:+d} moneyline)"
                if away_odds is not None
                else f"- {away}: {p_away*100:.1f}%"
            )

        # Add brief context if specialist outputs exist
        if workflow_outputs:
            lines.append("")
            lines.append("Context:")
            for k, v in workflow_outputs.items():
                if not v:
                    continue
                # Include a small excerpt (first non-empty ~4 lines) to avoid losing useful content
                excerpt_lines = []
                for ln in v.splitlines():
                    s = ln.strip()
                    if not s:
                        continue
                    excerpt_lines.append(s)
                    if len(excerpt_lines) >= 4:
                        break
                if excerpt_lines:
                    if len(excerpt_lines) == 1:
                        lines.append(f"- {k}: {excerpt_lines[0]}")
                    else:
                        lines.append(f"- {k}:")
                        lines.extend([f"  - {ln}" for ln in excerpt_lines])

        return "\n".join(lines).strip()

