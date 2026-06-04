"""Tests for V2 AgentState and Pydantic sub-models."""

from __future__ import annotations

import operator

import pytest

from enterprise_rag.agents.state import (
    AgentAnswer,
    AgentState,
    ChunkResult,
    QueryPlan,
    RetrievalOutput,
    RetrievalPlan,
    SubQuery,
    TraceEntry,
    ValidationDecision,
    ValidationResult,
    make_initial_state,
)


# ── SubQuery ───────────────────────────────────────────────────────────────────


def test_sub_query_defaults():
    sq = SubQuery(sub_query_id="q1", text="What is the hotel limit?")
    assert sq.strategy_hint is None
    assert sq.entity_slots == {}
    assert sq.depends_on is None


def test_sub_query_with_entity_slots():
    sq = SubQuery(
        sub_query_id="q2",
        text="Which endpoints use {auth_method}?",
        entity_slots={"auth_method": "OAuth2"},
        depends_on="q1",
    )
    assert sq.entity_slots["auth_method"] == "OAuth2"
    assert sq.depends_on == "q1"


# ── QueryPlan ──────────────────────────────────────────────────────────────────


def test_query_plan_single_hop():
    plan = QueryPlan(
        query_id="q001",
        raw_query="What is the hotel limit?",
        reasoning_type="single_hop",
        is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="What is the hotel limit?")],
    )
    assert len(plan.sub_queries) == 1
    assert plan.classification_confidence == 1.0


def test_query_plan_multi_hop():
    plan = QueryPlan(
        query_id="q002",
        raw_query="Which endpoints use the same auth as data export?",
        reasoning_type="multi_hop",
        is_decomposed=True,
        sub_queries=[
            SubQuery(sub_query_id="q1", text="What auth does data export use?"),
            SubQuery(
                sub_query_id="q2",
                text="Which endpoints use {auth_method}?",
                entity_slots={},
                depends_on="q1",
            ),
        ],
    )
    assert plan.is_decomposed is True
    assert len(plan.sub_queries) == 2


# ── RetrievalPlan ──────────────────────────────────────────────────────────────


def test_retrieval_plan_fields():
    rp = RetrievalPlan(
        sub_query_id="q1",
        query_text="hotel daily expense limit",
        strategy="bm25",
        planner_source="rule",
    )
    assert rp.top_k == 10
    assert rp.strategy == "bm25"


# ── ChunkResult ────────────────────────────────────────────────────────────────


def test_chunk_result_fields():
    cr = ChunkResult(
        chunk_id="TRV-POL-001-C001",
        document_id="TRV-POL-001",
        rank=1,
        score=5.23,
        text="The maximum daily hotel expense is $200.",
        category="travel_policy",
    )
    assert cr.strategy_used == ""
    assert cr.section_path == []


# ── ValidationResult ───────────────────────────────────────────────────────────


def test_validation_result_passed():
    decisions = [
        ValidationDecision(chunk_id=f"c{i}", relevance_score=0.8, passed=True)
        for i in range(5)
    ]
    vr = ValidationResult(
        passed=True,
        decisions=decisions,
        passed_count=5,
        total_count=5,
        threshold=0.3,
        min_passed=3,
    )
    assert vr.passed is True
    assert vr.failure_reason == ""
    assert vr.suggested_strategy is None


def test_validation_result_failed_with_suggestion():
    decisions = [
        ValidationDecision(chunk_id="c1", relevance_score=0.1, passed=False)
    ]
    vr = ValidationResult(
        passed=False,
        decisions=decisions,
        passed_count=0,
        total_count=1,
        threshold=0.3,
        min_passed=3,
        failure_reason="too_few_relevant",
        suggested_strategy="hybrid",
    )
    assert vr.passed is False
    assert vr.suggested_strategy == "hybrid"


# ── AgentAnswer ────────────────────────────────────────────────────────────────


def test_agent_answer_fields():
    ans = AgentAnswer(
        query_id="q001",
        answer_text="The daily hotel limit is $200.",
        cited_chunk_ids=["TRV-POL-001-C001"],
        confidence=0.9,
    )
    assert ans.has_contradiction is False
    assert ans.evidence_coverage == 0.0


# ── TraceEntry ────────────────────────────────────────────────────────────────


def test_trace_entry_has_timestamp():
    entry = TraceEntry(agent="qa", event="classified", data={"type": "single_hop"})
    assert entry.timestamp_utc
    assert "T" in entry.timestamp_utc  # ISO 8601 format


# ── make_initial_state ────────────────────────────────────────────────────────


def test_make_initial_state_defaults():
    state = make_initial_state("q001", "What is the hotel limit?")
    assert state["query_id"] == "q001"
    assert state["raw_query"] == "What is the hotel limit?"
    assert state["query_plan"] is None
    assert state["retrieval_plans"] == []
    assert state["retrieval_outputs"] == []
    assert state["validation_result"] is None
    assert state["validated_evidence"] == []
    assert state["answer"] is None
    assert state["trace"] == []
    assert state["loop_count"] == 0
    assert state["error"] is None


def test_make_initial_state_with_metadata():
    meta = {"reasoning_type": "temporal", "category": "api_documentation"}
    state = make_initial_state("q002", "Which API version?", query_metadata=meta)
    assert state["query_metadata"]["reasoning_type"] == "temporal"


# ── Trace accumulation (operator.add reducer) ─────────────────────────────────


def test_trace_list_accumulates_via_add():
    e1 = TraceEntry(agent="qa", event="start", data={})
    e2 = TraceEntry(agent="plan", event="strategy_selected", data={"strategy": "bm25"})
    combined = operator.add([e1], [e2])
    assert len(combined) == 2
    assert combined[0].agent == "qa"
    assert combined[1].agent == "plan"
