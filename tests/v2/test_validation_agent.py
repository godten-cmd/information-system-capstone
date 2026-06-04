"""Tests for Validation Agent: scorers, failure diagnosis, re-retrieval trigger."""

from __future__ import annotations

import pytest

from enterprise_rag.agents.state import (
    ChunkResult,
    QueryPlan,
    RetrievalOutput,
    RetrievalPlan,
    SubQuery,
    ValidationDecision,
    ValidationResult,
)
from enterprise_rag.agents.validation import (
    AdaptiveScorer,
    BM25RelevanceScorer,
    DenseRelevanceScorer,
    LLMRelevanceScorer,
    RankProxyScorer,
    ValidationAgent,
    ValidationConfig,
    _diagnose_failure,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────


def _chunk(
    chunk_id: str = "c1",
    rank: int = 1,
    score: float = 0.9,
    strategy: str = "bm25",
    text: str = "dummy text",
    sub_query_id: str | None = "q1",
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc1",
        rank=rank,
        score=score,
        text=text,
        category="travel_policy",
        strategy_used=strategy,
        sub_query_id=sub_query_id,
    )


def _make_query_plan(reasoning_type: str = "single_hop") -> QueryPlan:
    return QueryPlan(
        query_id="q_test",
        raw_query="What is the hotel limit?",
        reasoning_type=reasoning_type,
        is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text="What is the hotel limit?")],
        difficulty_signals=[],
    )


def _make_retrieval_plan(strategy: str = "bm25") -> RetrievalPlan:
    return RetrievalPlan(
        sub_query_id="q1",
        query_text="What is the hotel limit?",
        strategy=strategy,
        planner_source="rule",
    )


def _make_retrieval_output(
    sub_query_id: str = "q1",
    query_text: str = "What is the hotel limit?",
    strategy: str = "bm25",
    chunks: list[ChunkResult] | None = None,
) -> RetrievalOutput:
    return RetrievalOutput(
        sub_query_id=sub_query_id,
        query_text=query_text,
        strategy=strategy,
        chunks=chunks or [],
    )


# ── ValidationConfig ───────────────────────────────────────────────────────────


def test_config_default_threshold():
    cfg = ValidationConfig()
    assert cfg.threshold_for("single_hop") == pytest.approx(0.30)


def test_config_threshold_overrides():
    cfg = ValidationConfig()
    assert cfg.threshold_for("temporal") == pytest.approx(0.20)
    assert cfg.threshold_for("exception") == pytest.approx(0.22)
    assert cfg.threshold_for("multi_hop") == pytest.approx(0.20)


def test_config_custom_override():
    cfg = ValidationConfig(threshold_overrides={"temporal": 0.15})
    assert cfg.threshold_for("temporal") == pytest.approx(0.15)
    assert cfg.threshold_for("single_hop") == pytest.approx(0.30)


# ── RankProxyScorer ────────────────────────────────────────────────────────────


def test_rank_proxy_rank1_is_one():
    scorer = RankProxyScorer()
    chunks = [_chunk(rank=1)]
    scores = scorer.score("test query", chunks)
    assert scores == [pytest.approx(1.0)]


def test_rank_proxy_rank10_is_zero_point_one():
    scorer = RankProxyScorer()
    chunks = [_chunk(rank=10)]
    scores = scorer.score("test query", chunks)
    assert scores == [pytest.approx(0.1)]


def test_rank_proxy_rank11_clamps_to_zero():
    scorer = RankProxyScorer()
    chunks = [_chunk(rank=11)]
    scores = scorer.score("test query", chunks)
    assert scores[0] == pytest.approx(0.0)


def test_rank_proxy_empty():
    scorer = RankProxyScorer()
    assert scorer.score("q", []) == []


def test_rank_proxy_multiple_chunks():
    scorer = RankProxyScorer()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i) for i in range(1, 6)]
    scores = scorer.score("q", chunks)
    assert len(scores) == 5
    assert scores[0] > scores[-1]  # rank 1 > rank 5


# ── DenseRelevanceScorer ───────────────────────────────────────────────────────


def test_dense_scorer_uses_chunk_score():
    scorer = DenseRelevanceScorer()
    chunks = [_chunk(score=0.75), _chunk(chunk_id="c2", score=0.42)]
    scores = scorer.score("test query", chunks)
    assert scores[0] == pytest.approx(0.75)
    assert scores[1] == pytest.approx(0.42)


def test_dense_scorer_clamps_above_one():
    scorer = DenseRelevanceScorer()
    chunks = [_chunk(score=1.5)]
    scores = scorer.score("q", chunks)
    assert scores[0] == pytest.approx(1.0)


def test_dense_scorer_clamps_below_zero():
    scorer = DenseRelevanceScorer()
    chunks = [_chunk(score=-0.1)]
    scores = scorer.score("q", chunks)
    assert scores[0] == pytest.approx(0.0)


def test_dense_scorer_empty():
    scorer = DenseRelevanceScorer()
    assert scorer.score("q", []) == []


# ── AdaptiveScorer ─────────────────────────────────────────────────────────────


def test_adaptive_empty():
    scorer = AdaptiveScorer()
    assert scorer.score("q", []) == []


def test_adaptive_dense_chunks_use_chunk_score():
    scorer = AdaptiveScorer()
    chunk = _chunk(score=0.8, strategy="dense")
    scores = scorer.score("q", [chunk])
    assert scores[0] == pytest.approx(0.8)


def test_adaptive_hybrid_chunks_average():
    scorer = AdaptiveScorer()
    # hybrid uses avg(bm25, dense); dense score is from chunk.score
    chunk = _chunk(score=0.6, strategy="hybrid", chunk_id="c1", rank=1)
    scores = scorer.score("q", [chunk])
    # Result should be between 0 and 1
    assert 0.0 <= scores[0] <= 1.0


def test_adaptive_unknown_strategy_uses_bm25():
    scorer = AdaptiveScorer()
    chunk = _chunk(strategy="colbert", score=0.5)
    # Unknown strategy routes to BM25 branch (won't crash, returns a float)
    scores = scorer.score("some query", [chunk])
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


def test_adaptive_preserves_order():
    scorer = AdaptiveScorer()
    chunks = [
        _chunk(chunk_id="a", score=0.9, strategy="dense"),
        _chunk(chunk_id="b", score=0.4, strategy="dense"),
        _chunk(chunk_id="c", score=0.7, strategy="dense"),
    ]
    scores = scorer.score("q", chunks)
    assert len(scores) == 3
    assert scores[0] == pytest.approx(0.9)
    assert scores[1] == pytest.approx(0.4)
    assert scores[2] == pytest.approx(0.7)


def test_adaptive_mixed_strategies():
    scorer = AdaptiveScorer()
    chunks = [
        _chunk(chunk_id="a", score=0.8, strategy="dense"),
        _chunk(chunk_id="b", score=0.3, strategy="bm25", rank=5),
    ]
    scores = scorer.score("query text", chunks)
    assert len(scores) == 2
    assert scores[0] == pytest.approx(0.8)  # dense: use chunk.score
    assert 0.0 <= scores[1] <= 1.0          # bm25: normalized BM25 or fallback


# ── LLMRelevanceScorer ─────────────────────────────────────────────────────────


class _MockProvider:
    model_name = "mock-llm"

    def complete(self, user_prompt, system_prompt, **kwargs):
        from enterprise_rag.planning.providers.base import ProviderResponse
        import json
        # Count passages by finding lines starting with digits
        n = user_prompt.count("\n1. ") + user_prompt.count("\n2. ") + user_prompt.count("\n3. ") + \
            user_prompt.count("\n4. ") + user_prompt.count("\n5. ")
        n = max(n, 1)
        scores = [0.8] * n
        return ProviderResponse(
            content=json.dumps(scores),
            prompt_tokens=50,
            completion_tokens=10,
            model="mock-llm",
            latency_s=0.01,
        )


class _FailingProvider:
    model_name = "fail"

    def complete(self, *a, **k):
        raise RuntimeError("API error")


def test_llm_scorer_no_provider_uses_adaptive():
    scorer = LLMRelevanceScorer(provider=None)
    chunks = [_chunk(score=0.7, strategy="dense")]
    scores = scorer.score("q", chunks)
    assert scores[0] == pytest.approx(0.7)


def test_llm_scorer_empty():
    scorer = LLMRelevanceScorer(provider=None)
    assert scorer.score("q", []) == []


def test_llm_scorer_with_mock_provider():
    scorer = LLMRelevanceScorer(provider=_MockProvider())
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.5, strategy="bm25")
              for i in range(3)]
    scores = scorer.score("hotel limit query", chunks)
    assert len(scores) == 3
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_llm_scorer_batches_correctly():
    scorer = LLMRelevanceScorer(provider=_MockProvider())
    # BATCH_SIZE = 5; send 7 chunks → 2 batches
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.5, strategy="dense")
              for i in range(7)]
    scores = scorer.score("q", chunks)
    assert len(scores) == 7


def test_llm_scorer_falls_back_on_error():
    scorer = LLMRelevanceScorer(provider=_FailingProvider())
    chunks = [_chunk(score=0.6, strategy="dense")]
    # Should not raise; falls back to adaptive
    scores = scorer.score("q", chunks)
    assert len(scores) == 1
    assert 0.0 <= scores[0] <= 1.0


# ── _diagnose_failure ──────────────────────────────────────────────────────────


def _make_decisions(
    scores: list[float],
    threshold: float = 0.30,
) -> list[ValidationDecision]:
    return [
        ValidationDecision(
            chunk_id=f"c{i}",
            relevance_score=s,
            passed=s >= threshold,
            reason="test",
        )
        for i, s in enumerate(scores)
    ]


def test_diagnose_no_chunks():
    plan = _make_query_plan("single_hop")
    reason, strategy, hint = _diagnose_failure([], [], plan, [], [])
    assert reason == "no_chunks"
    assert strategy == "hybrid"


def test_diagnose_zero_recall_wrong_strategy():
    plan = _make_query_plan("temporal")  # oracle = bm25
    decisions = _make_decisions([0.0, 0.0, 0.0])
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(3)]
    ret_plan = _make_retrieval_plan(strategy="dense")  # wrong for temporal
    reason, strategy, hint = _diagnose_failure(decisions, chunks, plan, [ret_plan], [])
    assert reason == "wrong_strategy"
    assert strategy == "bm25"


def test_diagnose_zero_recall_already_hybrid():
    plan = _make_query_plan("temporal")
    decisions = _make_decisions([0.0, 0.0, 0.0])
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(3)]
    ret_plan = _make_retrieval_plan(strategy="bm25")  # correct strategy, still zero
    reason, strategy, hint = _diagnose_failure(decisions, chunks, plan, [ret_plan], [])
    # BM25 is the oracle for temporal, so zero with correct strategy → broaden
    assert reason in {"too_narrow", "wrong_strategy"}


def test_diagnose_missing_entity_multi_hop():
    plan = _make_query_plan("multi_hop")
    decisions = _make_decisions([0.1, 0.2, 0.0])
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(3)]
    ret_plan = _make_retrieval_plan("hybrid")
    # Unfilled slot in query text
    output = _make_retrieval_output(query_text="Which endpoints use {auth_method}?")
    reason, strategy, hint = _diagnose_failure(decisions, chunks, plan, [ret_plan], [output])
    assert reason == "missing_entity"
    assert hint == "re_slot_fill"


def test_diagnose_insufficient_coverage():
    plan = _make_query_plan("single_hop")
    # 2 pass, 4 fail → insufficient coverage (< min_passed=3)
    decisions = _make_decisions([0.5, 0.4, 0.1, 0.1, 0.0, 0.0])
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(6)]
    ret_plan = _make_retrieval_plan("bm25")
    reason, strategy, hint = _diagnose_failure(decisions, chunks, plan, [ret_plan], [])
    assert reason == "insufficient_coverage"
    assert strategy == "hybrid"


def test_diagnose_low_quality_suggests_hybrid():
    plan = _make_query_plan("single_hop")
    # 1 pass, rest fail
    decisions = _make_decisions([0.6, 0.1, 0.0, 0.0])
    chunks = [_chunk(chunk_id=f"c{i}") for i in range(4)]
    ret_plan = _make_retrieval_plan("dense")
    reason, strategy, hint = _diagnose_failure(decisions, chunks, plan, [ret_plan], [])
    assert strategy == "hybrid"


# ── ValidationAgent ────────────────────────────────────────────────────────────


def test_validation_agent_passes_with_enough_chunks():
    agent = ValidationAgent(config=ValidationConfig(scorer="rank_proxy"))
    # 5 high-rank chunks → all pass
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.9) for i in range(5)]
    plan = _make_query_plan("single_hop")
    ret_plan = _make_retrieval_plan()
    output = _make_retrieval_output(chunks=chunks)
    result = agent.validate(chunks, "hotel limit", plan, [ret_plan], [output])
    assert result.passed is True
    assert result.passed_count >= 3


def test_validation_agent_fails_with_few_chunks():
    agent = ValidationAgent(config=ValidationConfig(scorer="rank_proxy"))
    # Only 2 chunks → can't reach min_passed=3
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.9) for i in range(2)]
    plan = _make_query_plan("single_hop")
    ret_plan = _make_retrieval_plan()
    output = _make_retrieval_output(chunks=chunks)
    result = agent.validate(chunks, "hotel limit", plan, [ret_plan], [output])
    assert result.passed is False
    assert result.failure_reason != ""


def test_validation_agent_empty_chunks():
    agent = ValidationAgent(config=ValidationConfig(scorer="rank_proxy"))
    plan = _make_query_plan("single_hop")
    result = agent.validate([], "hotel limit", plan, [], [])
    assert result.passed is False
    assert result.total_count == 0
    assert result.passed_count == 0


def test_validation_agent_threshold_from_config():
    cfg = ValidationConfig(scorer="rank_proxy", relevance_threshold=0.0, min_passed_chunks=1)
    agent = ValidationAgent(config=cfg)
    chunks = [_chunk(chunk_id="c1", rank=11, score=0.0)]  # rank_proxy score = 0.0
    plan = _make_query_plan()
    output = _make_retrieval_output(chunks=chunks)
    result = agent.validate(chunks, "q", plan, [_make_retrieval_plan()], [output])
    # threshold=0.0 → score 0.0 passes (0.0 >= 0.0)
    assert result.passed is True


def test_validation_agent_uses_temporal_threshold():
    cfg = ValidationConfig(scorer="dense")
    agent = ValidationAgent(config=cfg)
    # Dense scores all 0.18 — below default 0.30 but above temporal 0.20
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.18, strategy="dense")
              for i in range(5)]
    plan = _make_query_plan("temporal")
    ret_plan = _make_retrieval_plan("bm25")
    output = _make_retrieval_output(chunks=chunks)
    result = agent.validate(chunks, "temporal query", plan, [ret_plan], [output])
    # temporal threshold=0.20 → 0.18 < 0.20 → still fails
    assert result.threshold == pytest.approx(0.20)


def test_validation_agent_scores_by_sub_query():
    agent = ValidationAgent(config=ValidationConfig(scorer="dense"))
    # Two sub-queries; chunks scored against their own sub-query text
    q1_chunk = _chunk(chunk_id="q1c1", score=0.9, strategy="dense", sub_query_id="q1")
    q2_chunk = _chunk(chunk_id="q2c1", score=0.8, strategy="dense", sub_query_id="q2")
    chunks = [q1_chunk, q2_chunk]

    plan = QueryPlan(
        query_id="q_mh", raw_query="multi q",
        reasoning_type="multi_hop", is_decomposed=True,
        sub_queries=[
            SubQuery(sub_query_id="q1", text="first sub-query"),
            SubQuery(sub_query_id="q2", text="second sub-query"),
        ],
        difficulty_signals=[],
    )
    outputs = [
        _make_retrieval_output(sub_query_id="q1", query_text="first sub-query", chunks=[q1_chunk]),
        _make_retrieval_output(sub_query_id="q2", query_text="second sub-query", chunks=[q2_chunk]),
    ]
    result = agent.validate(chunks, "multi q", plan, [], outputs)
    assert len(result.decisions) == 2
    assert result.decisions[0].relevance_score == pytest.approx(0.9)
    assert result.decisions[1].relevance_score == pytest.approx(0.8)


def test_validation_agent_decision_count_matches_chunks():
    agent = ValidationAgent(config=ValidationConfig(scorer="rank_proxy"))
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1) for i in range(8)]
    plan = _make_query_plan()
    output = _make_retrieval_output(chunks=chunks)
    result = agent.validate(chunks, "q", plan, [_make_retrieval_plan()], [output])
    assert len(result.decisions) == 8


def test_validation_agent_failure_diagnosis_populated():
    agent = ValidationAgent(config=ValidationConfig(scorer="rank_proxy"))
    # Low-rank chunks → scores near 0 → should fail
    chunks = [_chunk(chunk_id=f"c{i}", rank=10 + i, score=0.05) for i in range(3)]
    plan = _make_query_plan("temporal")
    ret_plan = _make_retrieval_plan("dense")  # wrong for temporal
    output = _make_retrieval_output(strategy="dense", chunks=chunks)
    result = agent.validate(chunks, "old api version", plan, [ret_plan], [output])
    assert result.passed is False
    assert result.failure_reason != ""


# ── get_validation_agent singleton ────────────────────────────────────────────


def test_get_validation_agent_returns_instance():
    import enterprise_rag.agents.validation as val_mod
    val_mod._agent = None  # reset singleton
    from enterprise_rag.agents.validation import get_validation_agent
    agent = get_validation_agent(scorer="rank_proxy")
    assert isinstance(agent, ValidationAgent)
    assert agent.config.scorer == "rank_proxy"
    val_mod._agent = None  # clean up


def test_get_validation_agent_rebuilds_on_scorer_change():
    import enterprise_rag.agents.validation as val_mod
    val_mod._agent = None
    from enterprise_rag.agents.validation import get_validation_agent
    a1 = get_validation_agent(scorer="rank_proxy")
    a2 = get_validation_agent(scorer="dense")
    assert a2.config.scorer == "dense"
    assert a1 is not a2
    val_mod._agent = None


def test_get_validation_agent_reuses_same_scorer():
    import enterprise_rag.agents.validation as val_mod
    val_mod._agent = None
    from enterprise_rag.agents.validation import get_validation_agent
    a1 = get_validation_agent(scorer="rank_proxy")
    a2 = get_validation_agent(scorer="rank_proxy")
    assert a1 is a2
    val_mod._agent = None


# ── validation_agent_node integration ─────────────────────────────────────────


def test_validation_node_no_query_plan():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import validation_agent_node

    state = make_initial_state("q1", "test")
    state["query_plan"] = None
    state["retrieval_outputs"] = []

    result = validation_agent_node(state)
    assert result["validation_result"] is None
    assert result["validated_evidence"] == []


def test_validation_node_with_chunks():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import validation_agent_node

    state = make_initial_state("q1", "What is the hotel limit?")
    state["query_plan"] = _make_query_plan("single_hop")
    state["retrieval_plans"] = [_make_retrieval_plan()]
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.9, strategy="bm25")
              for i in range(5)]
    state["retrieval_outputs"] = [_make_retrieval_output(chunks=chunks)]

    result = validation_agent_node(state)
    assert "validation_result" in result
    assert "validated_evidence" in result
    assert "trace" in result
    assert result["validation_result"] is not None


def test_validation_node_trace_has_scorer():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import validation_agent_node

    state = make_initial_state("q1", "test")
    state["query_plan"] = _make_query_plan()
    state["retrieval_plans"] = [_make_retrieval_plan()]
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.9) for i in range(5)]
    state["retrieval_outputs"] = [_make_retrieval_output(chunks=chunks)]

    import enterprise_rag.agents.validation as val_mod
    val_mod._agent = None

    result = validation_agent_node(state)
    trace_entry = result["trace"][0]
    assert "scorer" in trace_entry.data
    val_mod._agent = None
