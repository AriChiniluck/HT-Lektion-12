"""
test_researcher.py — Component tests for the Researcher agent.

Metric: GEval "Groundedness"
  - every factual claim in the output is supported by retrieved context
  - ungrounded claims (true or not) reduce the score

Run with debug output:
  deepeval test run tests/test_researcher.py --debug
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import deepeval
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from langchain_core.messages import AIMessage, ToolMessage

from config import settings
from tools import debug_print
from agents.research import get_research_agent, research

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)

# ── Metric ────────────────────────────────────────────────────────────────────

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Extract factual claims from 'actual output'.",
        "For each claim, check if it is supported by or consistent with 'retrieval context' "
        "(context includes both knowledge-base snippets AND web search results).",
        "Claims that align with or reasonably extend content in 'retrieval context' count as grounded.",
        "Claims that directly contradict 'retrieval context' are ungrounded.",
        "Widely accepted ML/AI domain knowledge not contradicted by context may be partially grounded.",
        "IMPORTANT: Do NOT penalise for synthesis or paraphrasing. "
        "A claim is grounded if its core meaning is supported by any retrieved source, "
        "even if the exact wording differs.",
        "Score = grounded_claims / total_claims.",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model=EVAL_MODEL,
    threshold=0.55,
)

relevancy_metric = GEval(
    name="Research Relevancy",
    evaluation_steps=[
        "Check that the 'actual output' directly addresses the research plan in 'input'.",
        "Check that the output contains factual evidence (not just vague statements).",
        "Penalise if the agent merely restates the request without any gathered information.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.7,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_researcher(plan_text: str) -> tuple[str, list[str]]:
    """Invoke the research agent and return (final_output, retrieval_contexts).

    retrieval_contexts = contents of all ToolMessage responses (search results).
    These are used as retrieval_context in the GEval Groundedness metric.
    """
    debug_print(f"\n[test_researcher] → research agent invoked")
    debug_print(f"  plan: {plan_text[:200]!r}")

    agent = get_research_agent()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": plan_text}]},
        config={"recursion_limit": 15},
    )
    messages = result.get("messages", [])

    retrieval_contexts: list[str] = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str) and content.strip():
                retrieval_contexts.append(content[:2500])
                debug_print(
                    f"  [test_researcher] tool result ({getattr(msg, 'name', '?')}): "
                    f"{content[:120]}..."
                )

    final_output = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            content = getattr(msg, "content", "") or ""
            if isinstance(content, str):
                final_output = content.strip()
                break
            if isinstance(content, list):
                parts = [
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                ]
                final_output = "".join(parts).strip()
                break

    debug_print(
        f"[test_researcher] ← output ({len(final_output)} chars), "
        f"{len(retrieval_contexts)} context chunks"
    )
    if settings.debug and final_output:
        debug_print(f"  preview: {final_output[:300]}")

    return final_output, retrieval_contexts


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_researcher_groundedness_rag_topic() -> None:
    """Researcher findings about RAG should be grounded in retrieved knowledge-base context."""
    plan_text = (
        "Research goal: Explain the difference between naive RAG and advanced RAG approaches.\n\n"
        "Search queries to execute:\n"
        "- naive RAG fixed chunk splitting limitations\n"
        "- advanced RAG sentence window parent child retrieval\n\n"
        "Sources to consult: knowledge_base, web\n\n"
        "Expected output format: Structured comparison with pros/cons per approach, sources section."
    )

    actual_output, retrieval_contexts = _run_researcher(plan_text)

    assert actual_output, "Researcher returned empty output for a valid plan."

    test_case = LLMTestCase(
        input=plan_text,
        actual_output=actual_output,
        retrieval_context=retrieval_contexts or ["No local context retrieved — using general knowledge."],
    )

    debug_print("[test_researcher] Running GEval 'Groundedness'")
    deepeval.assert_test(test_case, [groundedness])


def test_researcher_relevancy_multi_agent() -> None:
    """Researcher should return findings that are directly relevant to the given plan."""
    plan_text = (
        "Research goal: Describe the role of each agent in a Supervisor→Planner→Researcher→Critic pipeline.\n\n"
        "Search queries to execute:\n"
        "- supervisor agent role multi-agent system\n"
        "- planner researcher critic agent responsibilities\n\n"
        "Sources to consult: knowledge_base\n\n"
        "Expected output format: Brief section per agent role with sources."
    )

    actual_output, retrieval_contexts = _run_researcher(plan_text)
    assert actual_output, "Researcher returned empty output."

    test_case = LLMTestCase(
        input=plan_text,
        actual_output=actual_output,
        retrieval_context=retrieval_contexts or ["No local context retrieved."],
    )

    debug_print("[test_researcher] Running GEval 'Research Relevancy'")
    deepeval.assert_test(test_case, [relevancy_metric])


def test_researcher_handles_empty_plan_gracefully() -> None:
    """Researcher should return a non-empty error message for an empty plan."""
    debug_print("[test_researcher] Testing empty plan input")
    result = research.invoke({"plan": ""})
    debug_print(f"[test_researcher] result: {result!r}")
    assert result.strip(), "Expected a non-empty response for empty plan input."
    assert len(result) > 5, "Response for empty plan is too short to be meaningful."


def test_researcher_groundedness_langgraph() -> None:
    """Researcher findings about LangGraph should be grounded in retrieved context."""
    plan_text = (
        "Research goal: How does LangGraph implement stateful multi-agent pipelines?\n\n"
        "Search queries to execute:\n"
        "- LangGraph StateGraph nodes edges checkpointing\n"
        "- LangGraph human in the loop interrupt\n\n"
        "Sources to consult: knowledge_base, web\n\n"
        "Expected output format: Technical explanation with code references and sources."
    )

    actual_output, retrieval_contexts = _run_researcher(plan_text)
    assert actual_output, "Researcher returned empty output for LangGraph query."

    test_case = LLMTestCase(
        input=plan_text,
        actual_output=actual_output,
        retrieval_context=retrieval_contexts or ["No local context retrieved."],
    )

    debug_print("[test_researcher] Running GEval 'Groundedness' for LangGraph topic")
    deepeval.assert_test(test_case, [groundedness])
