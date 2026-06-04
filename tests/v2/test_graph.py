"""Tests for LangGraph graph topology, edges, and stub node behavior.

Graph compilation and stub node tests do NOT require real indexes.
Integration tests (marked) run the full pipeline with real indexes.
"""

from __future__ import annotations

import pytest

from enterprise_rag.agents.state import (
    QueryPlan,
    RetrievalPlan,
    SubQuery,
    ValidationResult,
    ValidationDecision,
    make_initial_state,
)
from enterprise_rag.graph.edges import MAX_LOOPS, route_after_validation


# ── Graph compilation ──────────────────────────────────────────────────────────


def test_graph_compiles():
    from enterprise_rag.graph.builder import build_graph
    graph = build_graph()
    assert graph is not None


def test_get_graph_returns_same_instance():
    from enterprise_rag.graph.builder import get_graph
    g1 = get_graph()
    g2 = get_graph()
    assert g1 is g2


def test_graph_has_expected_nodes():
    from enterprise_rag.graph.builder import (
        NODE_ANSWER, NODE_LOOP_COUNTER, NODE_PLAN,
        NODE_QA, NODE_RETRIEVAL, NODE_VALIDATION,
        build_graph,
    )
    graph = build_graph()
    node_names = set(graph.nodes)
    for expected in [NODE_QA, NODE_PLAN, NODE_RETRIEVAL, NODE_VALIDATION,
                     NODE_ANSWER, NODE_LOOP_COUNTER]:
        assert expected in node_names, f"Missing node: {expected}"


# ── route_after_validation ────────────────────────────────────────────────────


def _make_val_result(passed: bool, suggestion: str | None = None) -> ValidationResult:
    decisions = [
        ValidationDecision(chunk_id="c1", relevance_score=0.9 if passed else 0.1, passed=passed)
    ]
    return ValidationResult(
        passed=passed,
        decisions=decisions,
        passed_count=1 if passed else 0,
        total_count=1,
        threshold=0.3,
        min_passed=3,
        suggested_strategy=suggestion,
    )


def test_route_passes_to_answer():
    state = make_initial_state("q1", "test")
    state["validation_result"] = _make_val_result(passed=True)
    assert route_after_validation(state) == "answer"


def test_route_failed_under_max_loops():
    state = make_initial_state("q1", "test")
    state["validation_result"] = _make_val_result(passed=False)
    state["loop_count"] = 0
    assert route_after_validation(state) == "re_retrieve"


def test_route_failed_at_max_loops_goes_to_answer():
    state = make_initial_state("q1", "test")
    state["validation_result"] = _make_val_result(passed=False)
    state["loop_count"] = MAX_LOOPS
    assert route_after_validation(state) == "answer"


def test_route_error_state_terminates():
    state = make_initial_state("q1", "test")
    state["error"] = "something went wrong"
    assert route_after_validation(state) == "end"


def test_route_no_validation_result_terminates():
    state = make_initial_state("q1", "test")
    state["validation_result"] = None
    assert route_after_validation(state) == "end"


def test_max_loops_constant():
    assert MAX_LOOPS == 3


# ── increment_loop_count node ─────────────────────────────────────────────────


def test_loop_counter_increments():
    from enterprise_rag.graph.edges import increment_loop_count
    state = make_initial_state("q1", "test")
    state["loop_count"] = 1
    state["trace"] = []
    result = increment_loop_count(state)
    assert result["loop_count"] == 2
    assert len(result["trace"]) == 1
    assert result["trace"][0].event == "re_retrieval_loop"


# ── Stub node unit tests ───────────────────────────────────────────────────────


def test_qa_agent_node_produces_query_plan():
    from enterprise_rag.graph.nodes import qa_agent_node
    state = make_initial_state("q1", "What is the hotel limit?",
                               {"reasoning_type": "single_hop", "category": "travel_policy"})
    state["trace"] = []
    result = qa_agent_node(state)
    assert "query_plan" in result
    plan = result["query_plan"]
    assert isinstance(plan, QueryPlan)
    assert plan.reasoning_type == "single_hop"
    assert len(plan.sub_queries) == 1
    assert result["trace"][0].agent == "qa"


def test_plan_agent_node_produces_retrieval_plans():
    from enterprise_rag.graph.nodes import plan_agent_node
    state = make_initial_state("q1", "What is the hotel limit?")
    state["trace"] = []
    state["loop_count"] = 0
    state["query_plan"] = QueryPlan(
        query_id="q1",
        raw_query="What is the hotel limit?",
        reasoning_type="temporal",
        is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="What is the hotel limit?")],
    )
    state["validation_result"] = None
    result = plan_agent_node(state)
    assert "retrieval_plans" in result
    plans = result["retrieval_plans"]
    assert len(plans) == 1
    assert isinstance(plans[0], RetrievalPlan)
    # temporal → bm25 per empirical oracle
    assert plans[0].strategy == "bm25"


def test_plan_agent_node_no_query_plan_returns_error():
    from enterprise_rag.graph.nodes import plan_agent_node
    state = make_initial_state("q1", "test")
    state["query_plan"] = None
    state["trace"] = []
    result = plan_agent_node(state)
    assert result.get("error") is not None


def test_validation_agent_node_produces_result():
    from enterprise_rag.graph.nodes import validation_agent_node
    from enterprise_rag.agents.state import ChunkResult, RetrievalOutput
    state = make_initial_state("q1", "hotel limit")
    state["trace"] = []
    state["query_plan"] = QueryPlan(
        query_id="q1", raw_query="hotel limit",
        reasoning_type="single_hop", is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="hotel limit")]
    )
    chunks = [
        ChunkResult(chunk_id=f"c{i}", document_id="d1", rank=i+1,
                    score=5.0 - i, text=f"text {i}", category="travel_policy",
                    sub_query_id="q1")
        for i in range(5)
    ]
    state["retrieval_outputs"] = [
        RetrievalOutput(sub_query_id="q1", query_text="hotel limit",
                        strategy="bm25", chunks=chunks)
    ]
    state["retrieval_plans"] = [
        RetrievalPlan(sub_query_id="q1", query_text="hotel limit",
                      strategy="bm25", planner_source="rule")
    ]
    result = validation_agent_node(state)
    assert "validation_result" in result
    vr = result["validation_result"]
    assert isinstance(vr, ValidationResult)
    assert vr.total_count == 5


def test_answer_agent_node_produces_answer():
    from enterprise_rag.graph.nodes import answer_agent_node
    from enterprise_rag.agents.state import ChunkResult, QueryPlan, SubQuery
    import enterprise_rag.agents.answer as ans_mod
    from enterprise_rag.agents.answer import AnswerAgent
    ans_mod._agent = AnswerAgent(provider=None)  # force extractive (no LLM needed)

    state = make_initial_state("q1", "hotel limit")
    state["trace"] = []
    state["query_plan"] = QueryPlan(
        query_id="q1", raw_query="hotel limit",
        reasoning_type="single_hop", is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="hotel limit")],
        difficulty_signals=[],
    )
    state["validated_evidence"] = [
        ChunkResult(chunk_id=f"c{i}", document_id="d1", rank=i+1,
                    score=5.0 - i, text=f"evidence text {i}", category="travel_policy")
        for i in range(3)
    ]
    result = answer_agent_node(state)
    assert "answer" in result
    from enterprise_rag.agents.state import AgentAnswer
    assert isinstance(result["answer"], AgentAnswer)
    assert len(result["answer"].cited_chunk_ids) > 0
    ans_mod._agent = None


# ── Full pipeline integration test (requires real indexes) ────────────────────


@pytest.mark.integration
def test_full_pipeline_single_hop():
    from enterprise_rag.graph.builder import build_graph
    graph = build_graph()
    state = make_initial_state(
        "q_int_001",
        "What is the maximum daily hotel expense?",
        {"reasoning_type": "single_hop", "category": "travel_policy"},
    )
    result = graph.invoke(state)
    assert result["answer"] is not None
    assert result["error"] is None
    assert len(result["trace"]) >= 5   # one entry per agent
    assert result["loop_count"] == 0    # should pass validation on first attempt


@pytest.mark.integration
def test_full_pipeline_trace_agents_covered():
    from enterprise_rag.graph.builder import build_graph
    graph = build_graph()
    state = make_initial_state(
        "q_int_002",
        "List all security policy sections that mandate two-factor authentication.",
        {"reasoning_type": "aggregation", "category": "security_policy"},
    )
    result = graph.invoke(state)
    agent_names = {e.agent for e in result["trace"]}
    assert "qa" in agent_names
    assert "plan" in agent_names
    assert "retrieval" in agent_names
    assert "validation" in agent_names
    assert "answer" in agent_names
