"""
test_critic.py — Component tests for the Critic agent.

Metric: GEval "Critique Quality"
  - critique identifies specific, not vague issues
  - revision_requests are actionable
  - APPROVE verdict → gaps should be empty or minor
  - REVISE verdict → must include at least one revision_request

Custom metric: GEval "Verdict Consistency"
  - checks internal consistency of the CritiqueResult structure

Run with debug output:
  deepeval test run tests/test_critic.py --debug
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
from agents.critic import critique

EVAL_MODEL = os.getenv("DEEPEVAL_MODEL", settings.eval_model)

# ── Metrics ───────────────────────────────────────────────────────────────────

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "IMPORTANT: 'actual output' is the CRITIQUE RESPONSE (a JSON CritiqueResult object), NOT the research findings. "
        "Evaluate only the quality of the critique response itself, regardless of how good or bad the findings are.",
        "Check that the critique identifies specific issues, not vague or generic complaints.",
        "Check that revision_requests are actionable — a researcher should be able to act on them directly.",
        "If verdict is APPROVE, the gaps list should be empty or contain only minor, non-blocking items.",
        "If verdict is REVISE, there must be at least one concrete revision_request.",
        "The CritiqueResult JSON contains three boolean fields: is_fresh, is_complete, is_well_structured. "
        "If all three fields are present in the JSON, the critique has evaluated each dimension separately — "
        "award full marks for this criterion regardless of whether they are True or False.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.5,
)

# Custom business-logic metric: verdict consistency with content
verdict_consistency = GEval(
    name="Verdict Consistency",
    evaluation_steps=[
        "Read the 'actual output' which is a JSON CritiqueResult.",
        "IMPORTANT: evaluate only the internal consistency of the CritiqueResult JSON, NOT the quality of the research findings.",
        "If verdict is APPROVE: is_complete, is_fresh, and is_well_structured should generally be True "
        "(or the gaps list should explain the exceptions).",
        "If verdict is REVISE: at least one of is_complete, is_fresh, is_well_structured should be False, "
        "and revision_requests should not be empty.",
        "Penalise contradictions, e.g. verdict APPROVE with is_complete=false and non-empty revision_requests.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model=EVAL_MODEL,
    threshold=0.5,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

GOOD_FINDINGS = """
# RAG Approaches: Naive vs Advanced vs Agentic

## Naive RAG
Fixed-size chunk splitting (400–800 tokens). Fast and simple but loses cross-sentence context.
Source: lecture_rag_basics.pdf / page 3 / Relevance: 0.91

## Advanced RAG
Sentence-window retrieval expands each matched sentence with surrounding context (±3 sentences).
Parent-child chunking allows fine-grained matching with coarser retrieval for context.
Source: rag_advanced_2024.pdf / page 7 / Relevance: 0.88

## Agentic RAG
Uses an orchestrating agent that decides whether to re-retrieve or refine the query.
Applied in production systems as of 2025–2026.
Source: agentic_rag_survey_2026.pdf / page 12 / Relevance: 0.85

## Sources
- lecture_rag_basics.pdf
- rag_advanced_2024.pdf
- agentic_rag_survey_2026.pdf
"""

WEAK_FINDINGS = """
RAG is good. There are different types. Some are better than others. 
You should use whatever works for your use case.
"""

RESEARCH_REQUEST = "Compare naive RAG vs sentence-window retrieval and explain when to prefer each."
RESEARCH_PLAN = (
    "Goal: Compare RAG approaches.\n"
    "Queries: naive RAG chunk splitting limitations, sentence-window context improvement.\n"
    "Sources: knowledge_base, web."
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _invoke_critique(original_request: str, findings: str, plan: str = "") -> str:
    debug_print(f"\n[test_critic] → critique(request={original_request[:80]!r})")
    debug_print(f"  findings snippet: {findings[:120]!r}")

    result = critique.invoke({
        "original_request": original_request,
        "findings": findings,
        "plan": plan,
    })

    debug_print(f"[test_critic] ← critique returned ({len(result)} chars)")
    if settings.debug:
        try:
            parsed = json.loads(result)
            debug_print(f"  verdict: {parsed.get('verdict')}")
            debug_print(f"  is_complete: {parsed.get('is_complete')}, is_fresh: {parsed.get('is_fresh')}")
            revision = parsed.get("revision_requests", [])
            if revision:
                debug_print(f"  revision_requests: {revision}")
        except Exception:
            debug_print(f"  (raw): {result[:300]}")

    return result


# ── Tests ─────────────────────────────────────────────────────────────────────


def test_critic_approves_high_quality_findings() -> None:
    """Critic should return APPROVE for well-structured, sourced, complete findings."""
    result = _invoke_critique(RESEARCH_REQUEST, GOOD_FINDINGS, RESEARCH_PLAN)

    try:
        payload = json.loads(result)
        verdict = payload.get("verdict", "REVISE")
    except Exception:
        verdict = "APPROVE" if "APPROVE" in result else "REVISE"

    debug_print(f"[test_critic] Verdict for good findings: {verdict}")

    test_case = LLMTestCase(
        input=f"Request: {RESEARCH_REQUEST}\nFindings: {GOOD_FINDINGS[:400]}",
        actual_output=result,
    )
    deepeval.assert_test(test_case, [critique_quality, verdict_consistency])


def test_critic_revises_weak_findings() -> None:
    """Critic should return REVISE for vague, unsourced, incomplete findings."""
    result = _invoke_critique(RESEARCH_REQUEST, WEAK_FINDINGS, RESEARCH_PLAN)

    try:
        payload = json.loads(result)
        verdict = payload.get("verdict", "REVISE")
        revision_requests = payload.get("revision_requests", [])
    except Exception:
        verdict = "REVISE" if "REVISE" in result else "APPROVE"
        revision_requests = []

    debug_print(f"[test_critic] Verdict for weak findings: {verdict}")

    assert verdict == "REVISE", (
        f"Critic should REVISE vague, unsourced findings but returned {verdict}.\n"
        f"Full critique: {result}"
    )
    assert revision_requests, (
        "Critic returned REVISE but provided no revision_requests — researchers cannot act on this."
    )

    test_case = LLMTestCase(
        input=f"Request: {RESEARCH_REQUEST}\nFindings: {WEAK_FINDINGS}",
        actual_output=result,
    )
    deepeval.assert_test(test_case, [critique_quality])


def test_critic_verdict_consistency() -> None:
    """CritiqueResult fields should be internally consistent (no contradictions)."""
    # Use moderately good findings to exercise the consistency check in both directions
    moderate_findings = (
        "RAG stands for Retrieval-Augmented Generation. "
        "Naive RAG uses fixed chunks. Sentence-window retrieval is better in some cases. "
        "Source: web search."
    )
    result = _invoke_critique(RESEARCH_REQUEST, moderate_findings)

    debug_print(f"[test_critic] Running GEval 'Verdict Consistency'")

    test_case = LLMTestCase(
        input=f"Request: {RESEARCH_REQUEST}\nFindings: {moderate_findings}",
        actual_output=result,
    )
    deepeval.assert_test(test_case, [verdict_consistency])


def test_critic_respects_plan_coverage() -> None:
    """If research plan included specific queries, critic must check they were all covered."""
    plan_with_specific_queries = (
        "Goal: Explain FAISS and BM25 hybrid retrieval.\n"
        "Queries: FAISS IVF index structure, BM25 term frequency scoring, hybrid fusion strategies.\n"
        "Sources: knowledge_base."
    )
    findings_missing_bm25 = (
        "# FAISS for Vector Search\n"
        "FAISS builds an IVF index for fast approximate nearest-neighbor search.\n"
        "Source: faiss_docs / page 1 / Relevance: 0.93\n\n"
        "## Sources\n- faiss_docs"
        # Deliberately missing BM25 and hybrid fusion
    )
    result = _invoke_critique(
        original_request="Explain FAISS and BM25 hybrid retrieval",
        findings=findings_missing_bm25,
        plan=plan_with_specific_queries,
    )

    try:
        payload = json.loads(result)
        verdict = payload.get("verdict", "APPROVE")
        gaps = payload.get("gaps", [])
    except Exception:
        verdict = "REVISE"
        gaps = []

    debug_print(f"[test_critic] Plan-coverage verdict: {verdict}, gaps: {gaps}")

    # Critic should notice BM25 / hybrid fusion is missing from findings
    assert verdict == "REVISE" or gaps, (
        "Critic should flag missing BM25 / hybrid coverage from the research plan."
    )

    test_case = LLMTestCase(
        input=f"Plan: {plan_with_specific_queries}\nFindings: {findings_missing_bm25}",
        actual_output=result,
    )
    deepeval.assert_test(test_case, [critique_quality])
