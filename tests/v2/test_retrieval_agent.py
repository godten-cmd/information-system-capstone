"""Tests for Retrieval Agent: sequential multi-hop, aggregation merge, slot filling."""

from __future__ import annotations

import pytest

from enterprise_rag.agents.retrieval import (
    RetrievalAgent,
    _detect_slots,
    _fill_slots,
    _find_dependency,
    _topological_sort,
)
from enterprise_rag.agents.state import (
    ChunkResult,
    QueryPlan,
    RetrievalOutput,
    RetrievalPlan,
    SubQuery,
)


# ── Slot detection ─────────────────────────────────────────────────────────────


def test_detect_slots_found():
    slots = _detect_slots("Which endpoints use {auth_method} and {protocol}?")
    assert set(slots) == {"auth_method", "protocol"}


def test_detect_slots_none():
    slots = _detect_slots("What is the hotel limit?")
    assert slots == []


# ── _topological_sort ─────────────────────────────────────────────────────────


def _make_plans(specs: list[tuple[str, str]]) -> list[RetrievalPlan]:
    return [
        RetrievalPlan(
            sub_query_id=sqid,
            query_text=text,
            strategy="hybrid",
            planner_source="rule",
        )
        for sqid, text in specs
    ]


def test_topological_sort_no_deps():
    plans = _make_plans([("q1", "plain"), ("q2", "also plain")])
    sorted_plans = _topological_sort(plans)
    assert len(sorted_plans) == 2


def test_topological_sort_with_slot():
    plans = _make_plans([
        ("q1", "What auth does data export use?"),
        ("q2", "Which endpoints use {value_from_q1}?"),
    ])
    sorted_plans = _topological_sort(plans)
    # q1 (no slot) should come before q2 (has slot)
    ids = [p.sub_query_id for p in sorted_plans]
    assert ids.index("q1") < ids.index("q2")


# ── _find_dependency ──────────────────────────────────────────────────────────


def test_find_dependency_no_slot():
    plans = _make_plans([("q1", "plain"), ("q2", "also plain")])
    assert _find_dependency(plans[1], plans) is None


def test_find_dependency_slot_references_q1():
    plans = _make_plans([
        ("q1", "What auth does data export use?"),
        ("q2", "Which endpoints use {value_from_q1}?"),
    ])
    dep = _find_dependency(plans[1], plans)
    assert dep == "q1"


# ── _fill_slots ───────────────────────────────────────────────────────────────


def _make_chunks(texts: list[str]) -> list[ChunkResult]:
    return [
        ChunkResult(
            chunk_id=f"c{i}", document_id="d1", rank=i + 1,
            score=5.0 - i, text=text, category="api_documentation"
        )
        for i, text in enumerate(texts)
    ]


def test_fill_slots_no_slots_unchanged():
    plan = RetrievalPlan(
        sub_query_id="q2",
        query_text="plain query with no slots",
        strategy="bm25",
        planner_source="rule",
    )
    chunks = _make_chunks(["OAuth2 is used for authentication"])
    result = _fill_slots(plan, chunks, provider=None)
    assert result.query_text == "plain query with no slots"


def test_fill_slots_heuristic_fallback():
    plan = RetrievalPlan(
        sub_query_id="q2",
        query_text="Which endpoints use {auth_method}?",
        strategy="bm25",
        planner_source="rule",
    )
    chunks = _make_chunks(["The data export service uses OAuth2 for authentication."])
    result = _fill_slots(plan, chunks, provider=None)
    # Heuristic uses first 80 chars of top chunk
    assert "{auth_method}" not in result.query_text
    assert "The data export service" in result.query_text


def test_fill_slots_marks_filled_in_rationale():
    plan = RetrievalPlan(
        sub_query_id="q2",
        query_text="Which endpoints use {auth_method}?",
        strategy="bm25",
        planner_source="rule",
        planner_rationale="R01: bm25",
    )
    chunks = _make_chunks(["OAuth2 authentication method"])
    result = _fill_slots(plan, chunks, provider=None)
    assert "[slot-filled]" in result.planner_rationale


# ── RetrievalAgent — independent mode ─────────────────────────────────────────


def _make_single_hop_setup():
    sq = SubQuery(sub_query_id="q1", text="What is the hotel limit?")
    plan = QueryPlan(
        query_id="q_test", raw_query="What is the hotel limit?",
        reasoning_type="single_hop", is_decomposed=False,
        sub_queries=[sq], difficulty_signals=[],
    )
    ret_plan = RetrievalPlan(
        sub_query_id="q1", query_text="What is the hotel limit?",
        strategy="bm25", planner_source="rule",
    )
    return plan, [ret_plan]


@pytest.mark.integration
def test_retrieval_agent_single_hop_returns_output():
    agent = RetrievalAgent(provider=None)
    plan, ret_plans = _make_single_hop_setup()
    outputs = agent.execute("q_test", ret_plans, plan)
    assert len(outputs) == 1
    assert isinstance(outputs[0], RetrievalOutput)
    assert len(outputs[0].chunks) > 0
    assert outputs[0].chunks[0].rank == 1


@pytest.mark.integration
def test_retrieval_agent_temporal_uses_bm25():
    agent = RetrievalAgent(provider=None)
    sq = SubQuery(sub_query_id="q1", text="Which API version before Q3 2024?")
    plan = QueryPlan(
        query_id="q_temp", raw_query="Which API version before Q3 2024?",
        reasoning_type="temporal", is_decomposed=False,
        sub_queries=[sq], difficulty_signals=["temporal_reasoning"],
    )
    ret_plan = RetrievalPlan(
        sub_query_id="q1", query_text="Which API version before Q3 2024?",
        strategy="bm25", planner_source="rule",
    )
    outputs = agent.execute("q_temp", [ret_plan], plan)
    assert len(outputs) == 1
    assert outputs[0].strategy == "bm25"


@pytest.mark.integration
def test_retrieval_agent_multi_hop_sequential():
    agent = RetrievalAgent(provider=None)
    sq1 = SubQuery(sub_query_id="q1", text="What authentication method does the data export service use?")
    sq2 = SubQuery(sub_query_id="q2", text="Which API endpoints use {auth_method_q1}?",
                   depends_on="q1")
    plan = QueryPlan(
        query_id="q_mh", raw_query="multi-hop test",
        reasoning_type="multi_hop", is_decomposed=True,
        sub_queries=[sq1, sq2], difficulty_signals=["multi_document_dependency"],
    )
    ret_plans = [
        RetrievalPlan(sub_query_id="q1", query_text=sq1.text, strategy="hybrid", planner_source="rule"),
        RetrievalPlan(sub_query_id="q2", query_text=sq2.text, strategy="bm25", planner_source="rule"),
    ]
    outputs = agent.execute("q_mh", ret_plans, plan)
    assert len(outputs) == 2
    # Q2 text should have {auth_method_q1} replaced with actual content
    q2_output = next(o for o in outputs if o.sub_query_id == "q2")
    assert "{auth_method_q1}" not in q2_output.query_text


@pytest.mark.integration
def test_retrieval_agent_aggregation_merges():
    agent = RetrievalAgent(provider=None)
    sq1 = SubQuery(sub_query_id="q1", text="2FA security policy sections")
    sq2 = SubQuery(sub_query_id="q2", text="two-factor authentication requirements policy")
    plan = QueryPlan(
        query_id="q_agg", raw_query="list all 2FA policy sections",
        reasoning_type="aggregation", is_decomposed=True,
        sub_queries=[sq1, sq2], difficulty_signals=[],
    )
    ret_plans = [
        RetrievalPlan(sub_query_id="q1", query_text=sq1.text, strategy="hybrid", planner_source="rule"),
        RetrievalPlan(sub_query_id="q2", query_text=sq2.text, strategy="bm25", planner_source="rule"),
    ]
    outputs = agent.execute("q_agg", ret_plans, plan)
    assert len(outputs) == 1  # merged into single output
    assert outputs[0].sub_query_id == "merged"
    # No duplicate chunk_ids
    chunk_ids = [c.chunk_id for c in outputs[0].chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


# ── Full pipeline node test (integration, real indexes) ───────────────────────


@pytest.mark.integration
def test_retrieval_node_in_pipeline():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.builder import build_graph

    graph = build_graph()
    state = make_initial_state(
        "q_b_001",
        "List all security policy sections that mandate two-factor authentication.",
        {"reasoning_type": "aggregation", "category": "security_policy", "use_oracle": True},
    )
    result = graph.invoke(state)
    assert result["error"] is None
    assert result["retrieval_outputs"] is not None
    assert len(result["retrieval_outputs"]) > 0
    assert result["validated_evidence"] is not None
    # Aggregation → hybrid
    strategies = {o.strategy for o in result["retrieval_outputs"]}
    assert "hybrid" in strategies or "bm25" in strategies
