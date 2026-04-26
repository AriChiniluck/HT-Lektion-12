import json
import re
import threading
from collections import defaultdict
from contextvars import ContextVar
from functools import lru_cache
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain.agents.middleware.types import ToolCallRequest
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from config import build_chat_model, get_supervisor_system_prompt, settings
from tools import save_report
from agents import plan, research, critique

import logging

BEST_EFFORT_DISCLAIMER = (
    "> **⚠️ Best-effort draft:** this report was saved after reaching the maximum "
    "number of revise cycles and may still contain unresolved gaps noted by the Critic.\n\n"
)

def _make_counters(original_request: str | None = None) -> dict:
    return {
        "research_calls": 0,
        "revise_cycles": 0,
        "limit_reached": False,
        "awaiting_save": False,
        "force_research": False,
        "last_critique_payload": None,
        "last_findings": None,
        "last_plan": None,
        "last_reviewer_feedback": None,
        "original_request": original_request,
    }


_RUN_LIMITS: dict[str, dict[str, object]] = defaultdict(_make_counters)
_DEFAULT_THREAD_ID = "default-thread"

# ContextVar carries the real thread_id from wrap_tool_call (where configurable
# is available) into wrap_model_call (where it is not), within the same logical
# execution context. Unlike threading.local(), ContextVar is isolated per async
# task and per OS thread — safe for multi-user deployments.
_current_thread_id: ContextVar[str | None] = ContextVar("_current_thread_id", default=None)

# Thread-local store for the active graph thread_id and session_id.
# Set by main.py via set_active_thread_id() BEFORE each supervisor.stream() call,
# so wrap_model_call can resolve the correct thread_id even on the very first
# model invocation — before any tool call has had a chance to set the ContextVar.
# Each OS thread (= each concurrent user in a sync multi-user REPL) has its own
# isolated copy, making this safe without locks.
_thread_local_store = threading.local()


def set_active_thread_id(thread_id: str, session_id: str | None = None) -> None:
    """Register thread_id (and optional session_id) for this OS thread.

    Call from main.py immediately before supervisor.stream() so that
    wrap_model_call resolves the correct thread_id even when the LangGraph
    configurable is not available in the model request object.
    Safe for multi-user: threading.local() is isolated per OS thread.
    """
    _thread_local_store.thread_id = thread_id
    _thread_local_store.session_id = session_id


def clear_active_thread_id() -> None:
    """Clear the registration for this OS thread after the stream ends."""
    _thread_local_store.thread_id = None
    _thread_local_store.session_id = None


def _get_active_session_id() -> str | None:
    """Return the session_id registered for this OS thread (or None)."""
    return getattr(_thread_local_store, "session_id", None)


# Per-thread-id reentrant locks protect counter read-modify-write operations
# against race conditions in multi-user (multi-threaded) deployments.
_THREAD_LOCKS: dict[str, threading.RLock] = {}
_THREAD_LOCKS_META = threading.Lock()


def _get_thread_lock(thread_id: str) -> threading.RLock:
    """Return (and lazily create) the per-thread-id RLock for safe concurrent counter mutation."""
    with _THREAD_LOCKS_META:
        if thread_id not in _THREAD_LOCKS:
            _THREAD_LOCKS[thread_id] = threading.RLock()
        return _THREAD_LOCKS[thread_id]


def _extract_thread_id_from_mapping(mapping: object) -> str | None:
    if not isinstance(mapping, dict):
        return None
    configurable = mapping.get("configurable") or {}
    if isinstance(configurable, dict):
        thread_id = configurable.get("thread_id")
        if thread_id:
            return str(thread_id)
    thread_id = mapping.get("thread_id")
    if thread_id:
        return str(thread_id)
    return None


def _extract_thread_id_from_request(request) -> str | None:
    """Try to read configurable.thread_id from any request type (model or tool call)."""
    runtime = getattr(request, "runtime", None)
    candidates = [
        getattr(request, "config", None),
        getattr(runtime, "config", None),
        getattr(runtime, "context", None),
        getattr(runtime, "state", None),
    ]
    for candidate in candidates:
        thread_id = _extract_thread_id_from_mapping(candidate)
        if thread_id:
            return thread_id
    return None


def _get_thread_id(request) -> str:
    """Extract thread_id from request.

    Precedence:
      1. configurable.thread_id from the request (always present in tool calls).
      2. _current_thread_id ContextVar — set by a prior tool call in the same
         logical context (may not propagate across LangGraph node boundaries).
      3. _thread_local_store.thread_id — set by main.py via set_active_thread_id()
         BEFORE the supervisor.stream() call.  Reliable even for the very first
         model invocation before any tool call, and safe for multi-user because
         threading.local() is isolated per OS thread.
      4. A new auto-* fallback (last resort; emits a warning).
    """
    real = _extract_thread_id_from_request(request)
    if real:
        _current_thread_id.set(real)
        return real
    cached = _current_thread_id.get()
    if cached:
        return cached
    registered = getattr(_thread_local_store, "thread_id", None)
    if registered:
        _current_thread_id.set(registered)
        return registered
    fallback = f"auto-{uuid4().hex[:8]}"
    logging.warning(
        "No thread_id found in request config; assigned fallback ID %r. "
        "Pass config={'configurable': {'thread_id': <id>}} for deterministic state.",
        fallback,
    )
    _current_thread_id.set(fallback)
    return fallback


def _mirror_default_thread_state(thread_id: str) -> None:
    if thread_id == _DEFAULT_THREAD_ID:
        return
    counters = _RUN_LIMITS.get(thread_id)
    if counters is not None:
        _RUN_LIMITS[_DEFAULT_THREAD_ID] = counters


_INVALID_JSON_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _sanitize_text(text: str) -> str:
    """Strip control characters that break JSON serialisation of the OpenAI API payload."""
    return _INVALID_JSON_CHARS.sub("", text)


def _tool_content_to_text(content) -> str:
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return _sanitize_text("".join(parts))
    return _sanitize_text(str(content)) if content else ""


def _prepend_best_effort_disclaimer(content: str) -> str:
    text = content or ""
    if text.startswith(BEST_EFFORT_DISCLAIMER):
        return text
    return BEST_EFFORT_DISCLAIMER + text


def _suggest_report_filename(content: str, default_stem: str = "report") -> str:
    for line in str(content or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", title).strip("_").lower()
            if slug:
                return f"{slug[:80]}.md"
    return f"{default_stem}.md"


def reset_supervisor_limits(thread_id: str | None = None) -> None:
    if thread_id is None:
        _RUN_LIMITS.clear()
        return
    if _RUN_LIMITS.get(_DEFAULT_THREAD_ID) is _RUN_LIMITS.get(thread_id):
        _RUN_LIMITS.pop(_DEFAULT_THREAD_ID, None)
    _RUN_LIMITS.pop(thread_id, None)


def get_last_critique_payload(thread_id: str) -> dict | None:
    """Return the last critique payload for a given thread (used by main.py for debug display)."""
    return _RUN_LIMITS.get(thread_id, {}).get("last_critique_payload")


def reset_awaiting_save(thread_id: str) -> None:
    """Reset awaiting_save flag so auto-save does not trigger after a HITL rejection."""
    if thread_id in _RUN_LIMITS:
        _RUN_LIMITS[thread_id]["awaiting_save"] = False


def _build_research_followup_from_critique(
    payload: dict | None,
    last_findings: str | None = None,
    original_request: str | None = None,
) -> str:
    prefix = f"Original user request:\n{original_request}\n\n" if original_request else ""

    if not isinstance(payload, dict):
        base = (
            "Revise the previous research using the critic's feedback. "
            "Strengthen the answer, verify the sources, and improve the structure."
        )
        if last_findings:
            preview = last_findings[:1500] + ("..." if len(last_findings) > 1500 else "")
            return f"{prefix}{base}\n\nContext from previous findings (improve upon these):\n{preview}"
        return f"{prefix}{base}"

    revision_requests = [
        str(item).strip() for item in (payload.get("revision_requests") or []) if str(item).strip()
    ]
    gaps = [str(item).strip() for item in (payload.get("gaps") or []) if str(item).strip()]

    lines = [f"{prefix}Revise the previous research using the critic's feedback."]
    lines.append("Important: Do not remove or rewrite already found information unless it is incorrect. Only add new facts, examples, or clarifications required by the Critic. The final answer must include all previously found relevant information, plus the requested additions.")
    if revision_requests:
        lines.append("Address these revision requests:")
        lines.extend(f"- {item}" for item in revision_requests[:8])
    elif gaps:
        lines.append("Fix these gaps:")
        lines.extend(f"- {item}" for item in gaps[:8])

    if last_findings:
        preview = last_findings[:1500] + ("..." if len(last_findings) > 1500 else "")
        lines.append("\nContext from previous findings (improve upon these):")
        lines.append(preview)

    lines.append("Return updated findings with verified sources and a clearer structure.")
    return "\n".join(lines)


class RevisionLimitMiddleware(AgentMiddleware):
    """Hard-stop repeated revise loops after the configured maximum and force a save step when needed."""

    def wrap_model_call(self, request, handler):
        thread_id = _get_thread_id(request)
        if settings.debug:
            counters = _RUN_LIMITS.get(thread_id, {})
            short_state = {
                'research_calls': counters.get('research_calls'),
                'revise_cycles': counters.get('revise_cycles'),
                'awaiting_save': counters.get('awaiting_save'),
                'limit_reached': counters.get('limit_reached'),
            }
            session_id = _get_active_session_id()
            sid_tag = f" | session_id: {session_id}" if session_id else ""
            print(f">>>>> thread_id: {thread_id}{sid_tag}")
            print(f">>>>> _RUN_LIMITS[{thread_id}]: {short_state}")
        counters = _RUN_LIMITS[thread_id]

        # Inject research BEFORE calling the LLM to avoid wasted token spend
        if counters.get("force_research"):
            followup_plan = _build_research_followup_from_critique(
                counters.get("last_critique_payload"),
                counters.get("last_findings"),
                counters.get("original_request"),
            )
            counters["force_research"] = False
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "research",
                        "args": {"plan": followup_plan},
                        "id": f"call_auto_research_{uuid4().hex[:8]}",
                        "type": "tool_call",
                    }
                ],
            )

        # Fix A (Critical): when revision limit reached, inject save_report BEFORE the LLM call.
        # This prevents the LLM from seeing the final REVISE signal and outputting plain text
        # instead of calling save_report. Mirrors the same pre-call injection pattern as force_research.
        if counters.get("limit_reached") and counters.get("awaiting_save"):
            last_findings = str(counters.get("last_findings") or "").strip()
            if not last_findings:
                last_findings = "No research findings were collected before the revision limit was reached."
            if last_findings:
                content = _prepend_best_effort_disclaimer(last_findings)
                filename = _suggest_report_filename(content, default_stem="best_effort_report")
                counters["awaiting_save"] = False
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "save_report",
                            "args": {"filename": filename, "content": content},
                            "id": f"call_auto_save_{uuid4().hex[:8]}",
                            "type": "tool_call",
                        }
                    ],
                )

        response = handler(request)

        if not counters.get("awaiting_save"):
            return response

        # Fix B (Backup for APPROVE-flow): robust extraction of the last AIMessage regardless
        # of the wrapper type returned by handler() in langchain >= 1.2.0.
        last_message = None
        if isinstance(response, AIMessage):
            last_message = response
        elif hasattr(response, "message") and isinstance(response.message, AIMessage):
            last_message = response.message
        else:
            candidates = list(getattr(response, "result", None) or getattr(response, "messages", None) or [])
            for msg in reversed(candidates):
                if isinstance(msg, AIMessage):
                    last_message = msg
                    break

        if last_message is None:
            return response

        tool_calls = getattr(last_message, "tool_calls", []) or []
        if tool_calls:
            return response

        content = _tool_content_to_text(getattr(last_message, "content", "")).strip()
        if not content or len(content) < settings.auto_save_min_content_len:
            return response

        lowered = content.lower()
        if any(marker in lowered for marker in ["report saved to", "reviewer requested changes", "user rejected"]):
            counters["awaiting_save"] = False
            return response

        filename = _suggest_report_filename(
            content,
            default_stem="best_effort_report" if counters.get("limit_reached") else "report",
        )
        counters["awaiting_save"] = False
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "save_report",
                    "args": {"filename": filename, "content": content},
                    "id": f"call_auto_save_{uuid4().hex[:8]}",
                    "type": "tool_call",
                }
            ],
        )

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        tool_name = request.tool_call.get("name") or getattr(request.tool, "name", "")
        tool_call_id = request.tool_call.get("id") or getattr(
            request.runtime, "tool_call_id", "tool-call"
        )
        thread_id = _get_thread_id(request)
        _mirror_default_thread_state(thread_id)
        
        if settings.debug:
            counters = _RUN_LIMITS.get(thread_id, {})
            short_state = {
                'research_calls': counters.get('research_calls'),
                'revise_cycles': counters.get('revise_cycles'),
                'awaiting_save': counters.get('awaiting_save'),
                'limit_reached': counters.get('limit_reached'),
            }
            session_id = _get_active_session_id()
            sid_tag = f" | session_id: {session_id}" if session_id else ""
            print(f">>>>> TOOL: {tool_name}, thread_id: {thread_id}{sid_tag}")
            print(f">>>>> _RUN_LIMITS[{thread_id}]: {short_state}")

        if tool_name == "plan":
            incoming_request = str((request.tool_call.get("args") or {}).get("request", "") or "").strip()
            counters = _RUN_LIMITS.get(thread_id)

            if counters is None:
                _RUN_LIMITS[thread_id] = _make_counters(original_request=incoming_request or None)
                _mirror_default_thread_state(thread_id)
                counters = _RUN_LIMITS[thread_id]

            if incoming_request and not counters.get("original_request"):
                counters["original_request"] = incoming_request

            # If the supervisor tries to call Planner again after research already started,
            # keep the existing counters. This avoids wiping the revise-limit state and
            # prevents repeated critique→plan→research loops.
            if int(counters.get("research_calls", 0)) > 0:
                if counters.get("limit_reached") and counters.get("awaiting_save"):
                    return ToolMessage(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error",
                        content=(
                            "⛔ Revision limit already reached. Do not call `plan` again. "
                            "Immediately compose the best-effort draft from the latest findings and call `save_report`."
                        ),
                    )

                if counters.get("force_research"):
                    followup_plan = _build_research_followup_from_critique(
                        counters.get("last_critique_payload"),
                        counters.get("last_findings"),
                        counters.get("original_request"),
                    )
                    counters["force_research"] = False
                    return ToolMessage(
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="success",
                        content=(
                            "Skipping redundant mid-run re-planning to avoid a revise loop. "
                            "Use the critic-guided follow-up below as the next research instruction:\n\n"
                            f"{followup_plan}"
                        ),
                    )

            result = handler(request)
            plan_text = _tool_content_to_text(getattr(result, "content", "")).strip()
            if plan_text:
                counters["last_plan"] = plan_text
            return result

        if tool_name == "research":
            counters = _RUN_LIMITS[thread_id]
            args = request.tool_call.setdefault("args", {})
            incoming_plan = str(args.get("plan", "") or "").strip()
            if int(counters["research_calls"]) == 0:
                stored_plan = str(counters.get("last_plan", "") or "").strip()
                if stored_plan and incoming_plan and not incoming_plan.lstrip().startswith("{"):
                    args["plan"] = stored_plan
                    incoming_plan = stored_plan
            if counters.get("limit_reached") and counters.get("awaiting_save"):
                return ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    content=(
                        "⛔ Research is already blocked for this thread because the supervisor is waiting "
                        "to save or revise the best-effort draft. Do not call `research` again. "
                        "Immediately compose the best-effort draft from the latest findings and call `save_report`."
                    ),
                )
            if (
                int(counters["research_calls"]) > 0
                and int(counters["revise_cycles"]) >= settings.critique_max_rounds
            ):
                counters["limit_reached"] = True
                return ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                    content=(
                        "⛔ Hard revision limit reached. "
                        f"The Critic has already requested {counters['revise_cycles']} revise cycle(s), "
                        f"which matches the configured maximum of {settings.critique_max_rounds}. "
                        "Do not call `research` again. Immediately compose a best-effort draft report from the "
                        "evidence already collected, add a first-line warning that the draft was saved after "
                        "reaching the revision limit, and call `save_report` now."
                    ),
                )
            with _get_thread_lock(thread_id):
                counters["research_calls"] = int(counters["research_calls"]) + 1
            result = handler(request)
            findings_text = _tool_content_to_text(getattr(result, "content", ""))
            if findings_text.strip():
                counters["last_findings"] = findings_text

            # If research failed because of recursion or rate limit, stop retry loops early
            # and force the supervisor to save the best available draft instead of looping.
            lowered = findings_text.lower()
            if "research agent failed:" in lowered:
                counters["limit_reached"] = True
                counters["awaiting_save"] = True
                counters["force_research"] = False

            return result

        if tool_name == "save_report":
            counters = _RUN_LIMITS.setdefault(thread_id, _make_counters())
            counters["awaiting_save"] = True

            args = request.tool_call.setdefault("args", {})
            feedback_text = str(args.get("feedback", "") or "").strip()
            if counters.get("limit_reached"):
                content = str(args.get("content", ""))
                args["content"] = _prepend_best_effort_disclaimer(content)
                if not args.get("filename"):
                    args["filename"] = "best_effort_report.md"
            result = handler(request)
            result_text = _tool_content_to_text(getattr(result, "content", ""))
            lowered = result_text.lower()
            if "reviewer requested changes" in lowered or "report not saved" in lowered:
                counters["awaiting_save"] = False
                # INTENTIONAL: human reviewer authority overrides the automatic revise-cycle
                # limit. A human can always request more revisions even after the critic’s
                # automatic limit is reached. This is by design (human > automatic limit).
                # If you want to hard-cap HITL revisions too, check
                # counters["revise_cycles"] >= settings.critique_max_rounds here.
                counters["force_research"] = True
                counters["last_reviewer_feedback"] = feedback_text or result_text
                counters["last_critique_payload"] = {
                    "verdict": "REVISE",
                    "gaps": [
                        "The reviewer requested changes, so the previous draft was not approved for saving.",
                    ],
                    "revision_requests": [
                        feedback_text or "Revise the draft according to the review feedback before saving again.",
                        "Make sure the final answer clearly reflects the review feedback before the next save attempt.",
                    ],
                }
                return result
            if "report saved to" in lowered:
                # Remove thread state entirely so the next query starts fresh
                _RUN_LIMITS.pop(thread_id, None)
                # Return only a short confirmation, suppressing findings/excerpt after approve
                return ToolMessage(
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="success",
                    content="Report saved successfully."
                )
            if "user rejected" in lowered:
                counters["awaiting_save"] = False
            return result

        result = handler(request)

        if tool_name == "critique":
            counters = _RUN_LIMITS[thread_id]
            text = _tool_content_to_text(getattr(result, "content", ""))
            verdict = None
            payload = None

            try:
                payload = json.loads(text)
                verdict = payload.get("verdict")
            except Exception:
                if "REVISE" in text:
                    verdict = "REVISE"
                elif "APPROVE" in text:
                    verdict = "APPROVE"

            counters["last_critique_payload"] = payload

            # Do not count exception-fallback critiques as a real revise cycle —
            # transient errors (timeout, rate-limit) should not consume the revision budget.
            is_critique_error = isinstance(payload, dict) and payload.get("is_error", False)

            if verdict == "REVISE":
                counters["awaiting_save"] = False
                with _get_thread_lock(thread_id):
                    if not is_critique_error:
                        counters["revise_cycles"] = int(counters["revise_cycles"]) + 1
                counters["force_research"] = False
                if int(counters["revise_cycles"]) >= settings.critique_max_rounds:
                    counters["limit_reached"] = True
                    counters["awaiting_save"] = True
                else:
                    counters["limit_reached"] = False
                    counters["force_research"] = True
            elif verdict == "APPROVE":
                counters["limit_reached"] = False
                counters["awaiting_save"] = True
                counters["force_research"] = False

        return result


@lru_cache(maxsize=1)
def build_supervisor():
    return create_agent(
        model=build_chat_model(temperature=0.0, model=settings.supervisor_model),
        tools=[plan, research, critique, save_report],
        system_prompt=get_supervisor_system_prompt(),
        middleware=[
            RevisionLimitMiddleware(),
            HumanInTheLoopMiddleware(
                interrupt_on={"save_report": True},
            ),
        ],
        checkpointer=InMemorySaver(),
    )


supervisor = build_supervisor()
