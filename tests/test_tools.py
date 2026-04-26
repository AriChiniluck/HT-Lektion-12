"""
test_tools.py — Tool Correctness tests for the multi-agent system.

Verifies that each agent calls the correct tools for a given input:
    1. Planner receives a research request → must return a plan without doing retrieval itself
  2. Researcher receives a plan → must call at least one search tool
     (knowledge_search, web_search, or read_url)
  3. Supervisor completes a full pipeline → must call `save_report`

Metric: ToolCorrectnessMetric (deepeval)
  threshold=0.5 — lenient because tool order / extra tool calls are acceptable.

Run with debug output:
  deepeval test run tests/test_tools.py --debug
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import deepeval
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from config import settings
from tools import debug_print, save_report
from agents.planner import get_planner_agent
from agents.research import get_research_agent
from supervisor import (
    RevisionLimitMiddleware,
    _RUN_LIMITS,
    reset_supervisor_limits,
    set_active_thread_id,
    clear_active_thread_id,
    supervisor,
)

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)

TOOL_METRIC = ToolCorrectnessMetric(threshold=0.5, model=EVAL_MODEL)


def test_revision_state_survives_mid_run_replan() -> None:
    """A re-plan in the middle of a run must not erase the revise-limit counters."""
    thread_id = f"test-replan-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()
    _RUN_LIMITS[thread_id] = {
        "research_calls": 2,
        "revise_cycles": 2,
        "limit_reached": True,
        "awaiting_save": True,
        "force_research": False,
        "last_critique_payload": {"verdict": "REVISE"},
        "last_findings": "Best available findings.",
        "last_reviewer_feedback": None,
        "original_request": "Tell me about LangGraph in 2027.",
    }

    request = SimpleNamespace(
        tool_call={"name": "plan", "args": {"request": "Revise and continue."}, "id": "plan-1"},
        tool=SimpleNamespace(name="plan"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    result = middleware.wrap_tool_call(request, lambda req: "ok")

    assert isinstance(result, ToolMessage)
    assert "save_report" in str(result.content)
    counters = _RUN_LIMITS[thread_id]
    assert counters["revise_cycles"] == 2, "Mid-run plan() reset revise_cycles and re-opened the loop."
    assert counters["limit_reached"] is True
    assert counters["awaiting_save"] is True
    assert counters["original_request"] == "Tell me about LangGraph in 2027."

    reset_supervisor_limits(thread_id)


def test_save_report_with_feedback_does_not_save() -> None:
    """Reviewer feedback should block saving and ask for a revision."""
    result = save_report.invoke(
        {
            "filename": "draft.md",
            "content": "# Draft\n\nInitial content.",
            "feedback": "Update the answer to reflect 2026 and frame 2027 as a forecast.",
        }
    )

    lowered = result.lower()
    assert "not saved" in lowered
    assert "revise" in lowered


def test_research_is_blocked_after_fail_fast_state() -> None:
    """If research already failed and the supervisor is awaiting save, more research calls must be blocked."""
    thread_id = f"test-fail-fast-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()
    _RUN_LIMITS[thread_id] = {
        "research_calls": 1,
        "revise_cycles": 0,
        "limit_reached": True,
        "awaiting_save": True,
        "force_research": False,
        "last_critique_payload": None,
        "last_findings": "Research agent failed: model_not_found",
        "last_reviewer_feedback": None,
        "original_request": "Write a short report about RAG.",
    }

    request = SimpleNamespace(
        tool_call={"name": "research", "args": {"plan": "Continue research"}, "id": "research-1"},
        tool=SimpleNamespace(name="research"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    result = middleware.wrap_tool_call(request, lambda req: "should not run")

    assert isinstance(result, ToolMessage)
    assert "Do not call `research` again" in str(result.content)

    reset_supervisor_limits(thread_id)


def test_tool_calls_mirror_default_thread_state_for_model_phase() -> None:
    """Tool-phase thread state must be visible to model-phase middleware even if model calls resolve to default-thread."""
    thread_id = f"test-default-thread-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()

    plan_request = SimpleNamespace(
        tool_call={"name": "plan", "args": {"request": "Summarize RAG"}, "id": "plan-1"},
        tool=SimpleNamespace(name="plan"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    middleware.wrap_tool_call(plan_request, lambda req: ToolMessage(tool_call_id="plan-1", name="plan", content="ok"))

    assert _RUN_LIMITS.get("default-thread") is _RUN_LIMITS.get(thread_id)

    reset_supervisor_limits(thread_id)


def test_first_research_call_uses_stored_structured_plan() -> None:
    """If Supervisor passes only a short goal, middleware should restore the full Planner output on the first research call."""
    thread_id = f"test-stored-plan-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()
    full_plan = '{"goal": "Compare RAG types", "search_queries": ["naive rag", "agentic rag"], "sources_to_check": ["knowledge_base", "web"], "output_format": "report"}'
    _RUN_LIMITS[thread_id] = {
        "research_calls": 0,
        "revise_cycles": 0,
        "limit_reached": False,
        "awaiting_save": False,
        "force_research": False,
        "last_critique_payload": None,
        "last_findings": None,
        "last_plan": full_plan,
        "last_reviewer_feedback": None,
        "original_request": "Compare RAG types.",
    }

    request = SimpleNamespace(
        tool_call={"name": "research", "args": {"plan": "Compare RAG types"}, "id": "research-1"},
        tool=SimpleNamespace(name="research"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    captured = {}

    def _handler(req):
        captured["plan"] = req.tool_call["args"]["plan"]
        return ToolMessage(tool_call_id="research-1", name="research", content="ok")

    middleware.wrap_tool_call(request, _handler)

    assert captured["plan"] == full_plan

    reset_supervisor_limits(thread_id)


def test_successful_save_resets_thread_state() -> None:
    """A successful save_report should clear per-thread supervisor state so the next query starts fresh."""
    thread_id = f"test-save-reset-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()
    _RUN_LIMITS[thread_id] = {
        "research_calls": 2,
        "revise_cycles": 2,
        "limit_reached": True,
        "awaiting_save": True,
        "force_research": False,
        "last_critique_payload": {"verdict": "REVISE"},
        "last_findings": "Best effort draft",
        "last_plan": None,
        "last_reviewer_feedback": None,
        "original_request": "Compare RAG types.",
    }

    request = SimpleNamespace(
        tool_call={"name": "save_report", "args": {"filename": "draft.md", "content": "# Draft"}, "id": "save-1"},
        tool=SimpleNamespace(name="save_report"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    middleware.wrap_tool_call(
        request,
        lambda req: ToolMessage(tool_call_id="save-1", name="save_report", content="Report saved to C:/tmp/report.md"),
    )

    assert thread_id not in _RUN_LIMITS


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_ai_tool_calls(messages: list) -> list[ToolCall]:
    """Parse AIMessages from an agent result and return ToolCall objects."""
    captured: list[ToolCall] = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", []) or []):
                name = tc.get("name", "")
                args = tc.get("args", {}) or {}
                if name:
                    debug_print(f"  [test_tools] AI tool call: {name}({list(args.keys())})")
                    captured.append(ToolCall(name=name, input_parameters=args))
    return captured


def _stream_supervisor_collect(payload, config: dict) -> tuple[list[ToolCall], list, str]:
    """Stream the supervisor and collect tool calls, interrupts, and final text."""
    tool_calls: list[ToolCall] = []
    interrupts: list = []
    final_texts: list[str] = []
    thread_id = (config.get("configurable") or {}).get("thread_id", "")
    if thread_id:
        set_active_thread_id(thread_id)
    try:
        for chunk in supervisor.stream(
            payload,
            config=config,
            stream_mode=["updates"],
            version="v2",
        ):
            if chunk["type"] != "updates":
                continue

            data = chunk["data"]

            if "__interrupt__" in data:
                interrupts = list(data["__interrupt__"])
                continue

            model_payload = data.get("model") or {}
            for msg in model_payload.get("messages", []):
                for tc in (getattr(msg, "tool_calls", []) or []):
                    name = tc.get("name", "")
                    args = tc.get("args", {}) or {}
                    if name:
                        debug_print(f"  [test_tools] supervisor tool call: {name}")
                        tool_calls.append(ToolCall(name=name, input_parameters=args))

                text = ""
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str):
                    text = content.strip()
                elif isinstance(content, list):
                    text = "".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    ).strip()
                if text and not getattr(msg, "tool_calls", None):
                    final_texts.append(text)
    finally:
        if thread_id:
            clear_active_thread_id()

    return tool_calls, interrupts, "\n".join(final_texts).strip()


# ── Test 1: Planner returns a plan without retrieval ─────────────────────────


def test_tool_correctness_planner_returns_plan_without_tools() -> None:
    """Planner should produce a plan directly and leave retrieval to the Researcher."""
    request = "Explain naive RAG, sentence-window retrieval, and parent-child chunking from the course knowledge base"

    debug_print(f"\n[test_tools] Test 1 — Planner tool correctness")
    debug_print(f"  request: {request!r}")

    agent = get_planner_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    messages = result.get("messages", [])
    tools_called = _extract_ai_tool_calls(messages)
    tool_names_called = [tc.name for tc in tools_called]

    debug_print(f"  tools called: {tool_names_called}")

    # Build final text for LLMTestCase
    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                final_output = content[:500]
                break
    if not final_output:
        final_output = str(result.get("structured_response", "plan produced"))

    assert not tools_called, f"Planner should not call retrieval tools, but called: {tool_names_called}"
    assert final_output or result.get("structured_response")


# ── Test 2: Researcher → at least one search tool ────────────────────────────


def test_tool_correctness_researcher_uses_search_tools() -> None:
    """Researcher should call at least one search tool when executing a plan."""
    plan_text = (
        "Research goal: Describe main differences between naive RAG and agentic RAG.\n\n"
        "Search queries:\n"
        "- naive RAG chunk splitting\n"
        "- agentic RAG orchestration 2025 2026\n\n"
        "Sources to consult: knowledge_base, web\n\n"
        "Expected output: Comparative analysis with sources."
    )

    debug_print(f"\n[test_tools] Test 2 — Researcher tool correctness")

    agent = get_research_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": plan_text}]},
        config={"recursion_limit": settings.graph_recursion_limit},
    )
    messages = result.get("messages", [])
    tools_called = _extract_ai_tool_calls(messages)
    tool_names_called = [tc.name for tc in tools_called]

    debug_print(f"  tools called: {tool_names_called}")

    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                final_output = content[:600]
                break

    search_tools = {"knowledge_search", "web_search", "read_url"}
    assert search_tools & set(tool_names_called), (
        f"Researcher should call at least one of {search_tools}, "
        f"but only called: {tool_names_called}"
    )

    test_case = LLMTestCase(
        input=plan_text,
        actual_output=final_output or "researcher output",
        tools_called=tools_called,
        expected_tools=[
            ToolCall(name="knowledge_search"),
        ],
    )

    deepeval.assert_test(test_case, [TOOL_METRIC])


# ── Test 3: Supervisor → save_report after full pipeline ─────────────────────


def test_tool_correctness_supervisor_calls_save_report() -> None:
    """After the full Planner→Researcher→Critic pipeline, Supervisor must call `save_report`."""
    thread_id = f"test-tools-{uuid4().hex[:8]}"
    reset_supervisor_limits(thread_id)  # ensure clean state for new thread
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.graph_recursion_limit,
    }
    # Use a complex, multi-part query that forces the full research pipeline
    user_input = (
        "Research all major RAG approaches from the course materials: naive RAG, "
        "advanced RAG, and agentic RAG. Compare their architectures and create a report."
    )

    debug_print(f"\n[test_tools] Test 3 — Supervisor → save_report, thread={thread_id}")
    debug_print(f"  query: {user_input[:80]!r}...")

    payload: dict | Command = {"messages": [{"role": "user", "content": user_input}]}

    all_tool_calls: list[ToolCall] = []
    max_rounds = 6  # plan + research + critique + optional revision + save + approve

    for round_num in range(max_rounds):
        tool_calls, interrupts, _ = _stream_supervisor_collect(payload, config)
        all_tool_calls.extend(tool_calls)

        names_this_round = [tc.name for tc in tool_calls]
        debug_print(f"  round {round_num}: tool calls = {names_this_round}")

        if not interrupts:
            break

        # Auto-approve save_report interrupt
        debug_print("  [test_tools] auto-approving save_report interrupt")
        payload = Command(resume={"decisions": [{"type": "approve"}]})

    reset_supervisor_limits(thread_id)

    all_names = [tc.name for tc in all_tool_calls]
    debug_print(f"  all tools called across full run: {all_names}")

    assert any(t in all_names for t in ("plan", "research")), (
        f"Supervisor did not invoke any research tools — query may be too simple or pipeline was skipped. "
        f"Called: {all_names}"
    )
    assert "save_report" in all_names, (
        f"Supervisor should call `save_report` after completing the pipeline, "
        f"but only called: {all_names}"
    )

    test_case = LLMTestCase(
        input=user_input,
        actual_output="supervisor completed pipeline",
        tools_called=all_tool_calls,
        expected_tools=[
            ToolCall(name="plan"),
            ToolCall(name="research"),
            ToolCall(name="critique"),
            ToolCall(name="save_report"),
        ],
    )

    deepeval.assert_test(test_case, [TOOL_METRIC])


# ── Test: concurrent thread isolation ────────────────────────────────────────


def test_concurrent_thread_ids_do_not_share_state() -> None:
    """Two OS threads using different thread_ids must never contaminate each other's counters.

    This validates the threading fix: each thread_id has its own RLock and its own
    counter dict in _RUN_LIMITS. Mutations on thread A must be invisible on thread B.
    """
    thread_id_a = f"isolation-a-{uuid4().hex[:8]}"
    thread_id_b = f"isolation-b-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()

    errors: list[str] = []

    def run_thread_a():
        _RUN_LIMITS[thread_id_a] = {
            "research_calls": 0, "revise_cycles": 0, "limit_reached": False,
            "awaiting_save": False, "force_research": False,
            "last_critique_payload": None, "last_findings": None,
            "last_plan": None, "last_reviewer_feedback": None,
            "original_request": "Thread A request",
        }
        req = SimpleNamespace(
            tool_call={"name": "research", "args": {"plan": "thread A plan"}, "id": "ra-1"},
            tool=SimpleNamespace(name="research"),
            runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id_a}}),
        )
        middleware.wrap_tool_call(
            req,
            lambda r: ToolMessage(tool_call_id="ra-1", name="research", content="findings A"),
        )
        # After one research call, thread A's counter should be 1
        if _RUN_LIMITS[thread_id_a]["research_calls"] != 1:
            errors.append(
                f"Thread A research_calls expected 1, got {_RUN_LIMITS[thread_id_a]['research_calls']}"
            )

    def run_thread_b():
        _RUN_LIMITS[thread_id_b] = {
            "research_calls": 0, "revise_cycles": 0, "limit_reached": False,
            "awaiting_save": False, "force_research": False,
            "last_critique_payload": None, "last_findings": None,
            "last_plan": None, "last_reviewer_feedback": None,
            "original_request": "Thread B request",
        }
        req = SimpleNamespace(
            tool_call={"name": "research", "args": {"plan": "thread B plan"}, "id": "rb-1"},
            tool=SimpleNamespace(name="research"),
            runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id_b}}),
        )
        middleware.wrap_tool_call(
            req,
            lambda r: ToolMessage(tool_call_id="rb-1", name="research", content="findings B"),
        )
        # Thread B's counter must be independent of thread A
        if _RUN_LIMITS[thread_id_b]["research_calls"] != 1:
            errors.append(
                f"Thread B research_calls expected 1, got {_RUN_LIMITS[thread_id_b]['research_calls']}"
            )

    t_a = threading.Thread(target=run_thread_a)
    t_b = threading.Thread(target=run_thread_b)
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    reset_supervisor_limits(thread_id_a)
    reset_supervisor_limits(thread_id_b)

    assert not errors, "\n".join(errors)
    assert thread_id_a not in _RUN_LIMITS, "Thread A state was not cleaned up"
    assert thread_id_b not in _RUN_LIMITS, "Thread B state was not cleaned up"

    # Cross-contamination check: thread A's state should never appear under thread B's key
    # (we check this after cleanup — both keys must be gone independently)


# ── Test: HITL rejection does not save and blocks awaiting_save ──────────────


def test_hitl_rejection_does_not_save_and_resets_awaiting_save() -> None:
    """When a user rejects the save_report (type=reject via HITL), the file must NOT be saved
    and awaiting_save must be reset so the pipeline does not auto-save on the next LLM call.

    This tests the save_report tool directly (no real pipeline needed).
    """
    # Rejection path: feedback is provided as reviewer change request
    result = save_report.invoke({
        "filename": "test_rejection.md",
        "content": "# Draft\n\nSome research content here.",
        "feedback": "The answer is missing key details about agentic RAG. Please revise.",
    })

    lowered = result.lower()
    assert "not saved" in lowered or "report not saved" in lowered, (
        f"Expected save to be blocked when feedback is provided, got: {result!r}"
    )
    assert "revise" in lowered or "feedback" in lowered, (
        f"Response should reference the feedback or revision, got: {result!r}"
    )

    # Middleware level: after a HITL rejection, awaiting_save must be cleared
    thread_id = f"test-hitl-reject-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()
    _RUN_LIMITS[thread_id] = {
        "research_calls": 1, "revise_cycles": 0, "limit_reached": False,
        "awaiting_save": True, "force_research": False,
        "last_critique_payload": None, "last_findings": "some findings",
        "last_plan": None, "last_reviewer_feedback": None,
        "original_request": "Tell me about RAG.",
    }

    req = SimpleNamespace(
        tool_call={
            "name": "save_report",
            "args": {
                "filename": "draft.md",
                "content": "# Draft\n\nSome findings.",
                "feedback": "Please add more detail about sentence-window retrieval.",
            },
            "id": "save-reject-1",
        },
        tool=SimpleNamespace(name="save_report"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    middleware.wrap_tool_call(
        req,
        lambda r: ToolMessage(
            tool_call_id="save-reject-1",
            name="save_report",
            content="REPORT NOT SAVED.\nReviewer requested changes before saving.\nFeedback: Please add more detail.",
        ),
    )

    counters = _RUN_LIMITS.get(thread_id, {})
    assert not counters.get("awaiting_save"), (
        "awaiting_save must be False after HITL rejection so Fix-B does not auto-save on next LLM call"
    )
    assert counters.get("force_research"), (
        "force_research must be True after HITL rejection to trigger a revision cycle"
    )

    reset_supervisor_limits(thread_id)


def test_empty_researcher_output_does_not_loop() -> None:
    """When all search tools return empty results, the middleware must mark limit_reached=True
    and awaiting_save=True so the supervisor is forced to save a best-effort draft
    instead of retrying research indefinitely.

    Uses unittest.mock to patch search tools — no real LLM call needed.
    """
    from unittest.mock import patch

    thread_id = f"test-empty-research-{uuid4().hex[:8]}"
    middleware = RevisionLimitMiddleware()

    # Seed state: plan exists, research has NOT started yet
    _RUN_LIMITS[thread_id] = {
        "research_calls": 0,
        "revise_cycles": 0,
        "limit_reached": False,
        "awaiting_save": False,
        "force_research": False,
        "last_critique_payload": None,
        "last_findings": None,
        "last_plan": "Research the topic of agentic RAG.",
        "last_reviewer_feedback": None,
        "original_request": "Explain agentic RAG.",
    }

    empty_response = ToolMessage(
        tool_call_id="research-1",
        name="research",
        content="Research agent failed: no results found in knowledge base or web search.",
    )

    request = SimpleNamespace(
        tool_call={
            "name": "research",
            "args": {"plan": "Research agentic RAG."},
            "id": "research-1",
        },
        tool=SimpleNamespace(name="research"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    result = middleware.wrap_tool_call(request, lambda req: empty_response)

    counters = _RUN_LIMITS.get(thread_id, {})

    assert counters.get("limit_reached"), (
        "limit_reached must be True when research agent reports failure, "
        "to prevent the supervisor from retrying research in a loop"
    )
    assert counters.get("awaiting_save"), (
        "awaiting_save must be True after research failure so Fix-B triggers "
        "an automatic best-effort save on the next LLM call"
    )
    assert int(counters.get("research_calls", 0)) == 1, (
        "research_calls must be incremented even on failure so the supervisor "
        "knows at least one attempt was made"
    )

    # Simulate middleware blocking a second research attempt
    request2 = SimpleNamespace(
        tool_call={
            "name": "research",
            "args": {"plan": "Research agentic RAG again."},
            "id": "research-2",
        },
        tool=SimpleNamespace(name="research"),
        runtime=SimpleNamespace(config={"configurable": {"thread_id": thread_id}}),
    )

    blocked = middleware.wrap_tool_call(request2, lambda req: empty_response)
    blocked_text = str(getattr(blocked, "content", "")).lower()

    assert "blocked" in blocked_text or "⛔" in blocked_text or "do not call" in blocked_text, (
        f"Second research call must be blocked after limit_reached=True, got: {blocked_text!r}"
    )

    reset_supervisor_limits(thread_id)
