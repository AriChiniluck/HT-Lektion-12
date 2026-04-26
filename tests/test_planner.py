"""
test_planner.py — Component tests for the Planner agent.

Metric: GEval "Plan Quality"
  - plan contains specific, non-vague search queries
  - sources_to_check references relevant sources
  - output_format matches the user's request intent

Run with debug output:
  deepeval test run tests/test_planner.py --debug
  DEBUG=1 deepeval test run tests/test_planner.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import deepeval
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from config import settings
from tools import debug_print
from agents.planner import plan, get_planner_agent
from langchain_core.messages import AIMessage

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)

# ── Metric ────────────────────────────────────────────────────────────────────

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Check that the plan contains specific search queries (not vague or generic)",
        "Check that sources_to_check includes relevant sources for the topic (knowledge_base, web, or both)",
        "Check that output_format matches what the user would expect for this type of request",
        "Check that the goal clearly captures the user's intent without losing scope",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _invoke_planner(request: str) -> str:
    """Call the plan tool and return its JSON output string."""
    debug_print(f"\n[test_planner] → plan(request={request!r})")
    result = plan.invoke({"request": request})
    debug_print(f"[test_planner] ← plan returned ({len(result)} chars):")
    if settings.debug:
        try:
            parsed = json.loads(result)
            debug_print(json.dumps(parsed, ensure_ascii=False, indent=2)[:800])
        except Exception:
            debug_print(result[:800])
    return result


def _capture_planner_tool_calls(request: str) -> list[str]:
    """Return names of tools called internally by the Planner agent."""
    debug_print(f"[test_planner] Capturing internal tool calls for: {request!r}")
    agent = get_planner_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": request}]})
    messages = result.get("messages", [])
    names = []
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in (getattr(msg, "tool_calls", []) or []):
                name = tc.get("name", "")
                if name:
                    names.append(name)
                    debug_print(f"  [test_planner] tool called: {name}")
    return names


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "user_input",
    [
        "Compare naive RAG vs sentence-window retrieval",
        "What are the best practices for multi-agent systems in 2026?",
        "Explain how LangGraph orchestrates agent workflows",
    ],
)
def test_planner_plan_quality(user_input: str, eval_model: str) -> None:
    """Planner should produce a structured, specific research plan with actionable queries."""
    actual_output = _invoke_planner(user_input)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
    )

    debug_print(f"[test_planner] Running GEval 'Plan Quality' on: {user_input!r}")
    deepeval.assert_test(test_case, [plan_quality])


def test_planner_does_not_call_retrieval_tools() -> None:
    """Planner should return a structured plan directly and leave retrieval to the Researcher."""
    request = "Explain naive RAG, sentence-window retrieval, and parent-child chunking as covered in the course materials"
    tool_names = _capture_planner_tool_calls(request)

    debug_print(f"[test_planner] Tools used: {tool_names}")

    assert not tool_names, f"Planner should not call retrieval tools, but called: {tool_names}"

    result = _invoke_planner(request)
    assert result and len(result) > 50, (
        f"Planner returned an empty or too-short plan: {result!r}"
    )


def test_planner_preserves_user_language() -> None:
    """Plan fields should be in the same language as the user request (Ukrainian)."""
    request = "Поясни різницю між RAG і файн-тюнінгом"
    actual_output = _invoke_planner(request)

    debug_print(f"[test_planner] Checking language preservation for Ukrainian input")

    # At minimum the plan should not be empty and should contain some Cyrillic chars
    has_cyrillic = any("\u0400" <= ch <= "\u04FF" for ch in actual_output)
    assert has_cyrillic, (
        "Planner should preserve the user's Ukrainian language in plan fields, "
        f"but got: {actual_output[:200]}"
    )
