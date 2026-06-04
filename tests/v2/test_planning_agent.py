"""Tests for Planning Agent: rule dispatch, LLM fallback, re-retrieval override."""

from __future__ import annotations

import pytest

from enterprise_rag.agents.planning import PlanningAgent, _build_rule_query, _fill_entity_slots
from enterprise_rag.agents.state import (
    QueryPlan,
    RetrievalPlan,
    SubQuery,
    ValidationResult,
    ValidationDecision,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_plan(
    reasoning_type: str,
    query_text: str = "test query",
    difficulty_signals: list[str] | None = None,
    category: str | None = None,
    entity_slots: dict | None = None,
) -> tuple[QueryPlan, SubQuery]:
    sq = SubQuery(
        sub_query_id="q1",
        text=query_text,
        strategy_hint=None,
        target_category=category,
        entity_slots=entity_slots or {},
    )
    plan = QueryPlan(
        query_id="q_test",
        raw_query=query_text,
        reasoning_type=reasoning_type,
        is_decomposed=False,
        sub_queries=[sq],
        difficulty_signals=difficulty_signals or [],
    )
    return plan, sq


def _make_val_result(suggested_strategy: str | None = None) -> ValidationResult:
    return ValidationResult(
        passed=False,
        decisions=[],
        passed_count=0,
        total_count=0,
        threshold=0.3,
        min_passed=3,
        suggested_strategy=suggested_strategy,
    )


# ── _build_rule_query ─────────────────────────────────────────────────────────


def test_build_rule_query_maps_fields():
    query_plan, sq = _make_plan(
        "temporal", difficulty_signals=["temporal_reasoning"], category="API Documentation"
    )
    rq = _build_rule_query("q_test", sq, query_plan)
    assert rq["reasoning_type"] == "temporal"
    assert "temporal_reasoning" in rq["retrieval_difficulty_factors"]
    assert rq["category"] == "API Documentation"
    assert "q_test" in rq["query_id"]


def test_build_rule_query_neutral_oracle_fields():
    query_plan, sq = _make_plan("single_hop")
    rq = _build_rule_query("q1", sq, query_plan)
    assert rq["planner_oracle_strategy"] == "hybrid"
    assert rq["oracle_strategy_source"] == "v2_planning_agent"


# ── _fill_entity_slots ────────────────────────────────────────────────────────


def test_fill_entity_slots_no_slots():
    sq = SubQuery(sub_query_id="q1", text="plain query with no slots")
    assert _fill_entity_slots(sq) == "plain query with no slots"


def test_fill_entity_slots_with_prefilled():
    sq = SubQuery(
        sub_query_id="q2",
        text="Which endpoints use {auth_method}?",
        entity_slots={"auth_method": "OAuth2"},
    )
    result = _fill_entity_slots(sq)
    assert result == "Which endpoints use OAuth2?"


# ── PlanningAgent rule dispatch ───────────────────────────────────────────────


@pytest.mark.parametrize("reasoning_type,expected_strategy", [
    ("temporal", "bm25"),
    ("exception", "dense"),
    ("multi_hop", "hybrid"),
    ("aggregation", "hybrid"),
    ("comparison", "hybrid"),
])
def test_planning_agent_rule_dispatch(reasoning_type, expected_strategy):
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan(reasoning_type)
    result = agent.plan_sub_query("q_test", sq, plan)
    assert isinstance(result, RetrievalPlan)
    assert result.strategy == expected_strategy
    assert result.planner_source == "rule"


def test_planning_agent_single_hop_defaults_hybrid():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("single_hop")
    result = agent.plan_sub_query("q_test", sq, plan)
    # R07 (default) → hybrid or R06 (technical) → dense depending on category
    assert result.strategy in {"hybrid", "dense"}


def test_planning_agent_api_docs_single_hop_dense():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("single_hop", category="API Documentation")
    result = agent.plan_sub_query("q_test", sq, plan)
    # R06: technical single_hop → dense
    assert result.strategy == "dense"
    assert result.planner_source == "rule"


def test_planning_agent_table_dependency_hybrid():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("single_hop", difficulty_signals=["table_dependency"])
    result = agent.plan_sub_query("q_test", sq, plan)
    assert result.strategy == "hybrid"


def test_planning_agent_exception_handling_factor():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("single_hop", difficulty_signals=["exception_handling"])
    result = agent.plan_sub_query("q_test", sq, plan)
    # R02 fires on exception_handling factor
    assert result.strategy == "dense"


# ── Re-retrieval override ─────────────────────────────────────────────────────


def test_planning_agent_reretrieval_override():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("temporal")  # would normally → bm25
    val_result = _make_val_result(suggested_strategy="hybrid")
    result = agent.plan_sub_query("q_test", sq, plan, loop_count=1, validation_result=val_result)
    # Val Agent override takes precedence over rule planner
    assert result.strategy == "hybrid"
    assert result.planner_source == "validation_override"


def test_planning_agent_no_override_on_loop_0():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("temporal")
    val_result = _make_val_result(suggested_strategy="hybrid")
    result = agent.plan_sub_query("q_test", sq, plan, loop_count=0, validation_result=val_result)
    # loop_count=0: no override, rule fires normally
    assert result.strategy == "bm25"
    assert result.planner_source == "rule"


def test_planning_agent_no_override_when_val_has_no_suggestion():
    agent = PlanningAgent(provider=None)
    plan, sq = _make_plan("temporal")
    val_result = _make_val_result(suggested_strategy=None)
    result = agent.plan_sub_query("q_test", sq, plan, loop_count=1, validation_result=val_result)
    # No suggestion → rule fires
    assert result.strategy == "bm25"


# ── plan_all ──────────────────────────────────────────────────────────────────


def test_plan_all_single():
    agent = PlanningAgent(provider=None)
    plan, _ = _make_plan("temporal")
    results = agent.plan_all("q_test", plan)
    assert len(results) == 1
    assert results[0].strategy == "bm25"


def test_plan_all_multi_hop():
    agent = PlanningAgent(provider=None)
    sq1 = SubQuery(sub_query_id="q1", text="What auth does data export use?",
                   target_category="System Design Documents")
    sq2 = SubQuery(sub_query_id="q2", text="Which endpoints use {auth}?",
                   target_category="API Documentation", depends_on="q1")
    multi_plan = QueryPlan(
        query_id="q_mh", raw_query="multi-hop query",
        reasoning_type="multi_hop", is_decomposed=True,
        sub_queries=[sq1, sq2], difficulty_signals=["multi_document_dependency"],
    )
    results = agent.plan_all("q_mh", multi_plan)
    assert len(results) == 2
    assert all(isinstance(r, RetrievalPlan) for r in results)
    # multi_hop → hybrid for both
    assert all(r.strategy == "hybrid" for r in results)


# ── LLM fallback mock ─────────────────────────────────────────────────────────


class _MockPlanProvider:
    model_name = "mock-plan"

    def complete(self, user_prompt, system_prompt, **kwargs):
        import json
        from enterprise_rag.planning.providers.base import ProviderResponse
        return ProviderResponse(
            content=json.dumps({"strategy": "dense", "rationale": "semantic lookup"}),
            prompt_tokens=80,
            completion_tokens=20,
            model="mock-plan",
            latency_s=0.01,
        )


def test_planning_agent_llm_fallback_fires_for_low_confidence():
    """R07 (default, conf=0.60) should trigger LLM fallback when provider available."""
    agent = PlanningAgent(provider=_MockPlanProvider())
    # single_hop with no special factors → R07 fires (conf=0.60)
    plan, sq = _make_plan("single_hop")
    result = agent.plan_sub_query("q_test", sq, plan)
    # LLM mock returns "dense"
    assert result.strategy == "dense"
    assert result.planner_source == "llm"


def test_planning_agent_llm_fallback_not_fired_high_confidence():
    """R01 (temporal, conf=0.80 ≥ 0.75) should NOT trigger LLM fallback."""
    agent = PlanningAgent(provider=_MockPlanProvider())
    plan, sq = _make_plan("temporal")
    result = agent.plan_sub_query("q_test", sq, plan)
    # Rule fires directly without LLM
    assert result.strategy == "bm25"
    assert result.planner_source == "rule"


def test_planning_agent_llm_fallback_failure_uses_rule():
    class _FailProvider:
        model_name = "fail"
        def complete(self, *a, **k):
            raise RuntimeError("API error")

    agent = PlanningAgent(provider=_FailProvider())
    plan, sq = _make_plan("single_hop")  # R07: low confidence
    result = agent.plan_sub_query("q_test", sq, plan)
    # LLM fails → falls back to rule result (hybrid from R07)
    assert result.strategy == "hybrid"
    assert result.planner_source == "rule"


# ── Integration: full node round-trip (no indexes needed) ────────────────────


def test_plan_agent_node_round_trip():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import plan_agent_node

    state = make_initial_state("q1", "What is the hotel limit?",
                               {"reasoning_type": "single_hop", "use_oracle": True})
    state["trace"] = []
    state["loop_count"] = 0
    state["validation_result"] = None
    state["query_plan"] = QueryPlan(
        query_id="q1",
        raw_query="What is the hotel limit?",
        reasoning_type="single_hop",
        is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="What is the hotel limit?")],
        difficulty_signals=[],
    )

    result = plan_agent_node(state)
    assert "retrieval_plans" in result
    assert len(result["retrieval_plans"]) == 1
    assert result["retrieval_plans"][0].sub_query_id == "q1"
