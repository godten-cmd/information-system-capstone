"""Tests for Answer Agent: synthesis, citations, confidence, coverage, node integration."""

from __future__ import annotations

import pytest

from enterprise_rag.agents.answer import (
    AnswerAgent,
    AnswerConfig,
    _extract_cited_ids,
    _mean_score,
    get_answer_agent,
)
from enterprise_rag.agents.state import (
    AgentAnswer,
    ChunkResult,
    QueryPlan,
    SubQuery,
)


# ── Fixtures / helpers ─────────────────────────────────────────────────────────


def _chunk(
    chunk_id: str = "c1",
    rank: int = 1,
    score: float = 0.8,
    text: str = "The maximum hotel rate is $200 per night.",
    category: str = "travel_policy",
) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc1",
        rank=rank,
        score=score,
        text=text,
        category=category,
    )


def _make_plan(
    query_id: str = "q1",
    reasoning_type: str = "single_hop",
    query: str = "What is the hotel limit?",
) -> QueryPlan:
    return QueryPlan(
        query_id=query_id,
        raw_query=query,
        reasoning_type=reasoning_type,
        is_decomposed=False,
        sub_queries=[SubQuery(sub_query_id="q1", text=query)],
        difficulty_signals=[],
    )


class _MockProvider:
    """Returns a canned answer with citation markers."""

    model_name = "mock"

    def complete(self, user_prompt: str, system_prompt: str, **kwargs):
        from enterprise_rag.planning.providers.base import ProviderResponse
        return ProviderResponse(
            content="The hotel limit is $200 per night [1]. Exceptions require approval [2].",
            prompt_tokens=80,
            completion_tokens=20,
            model="mock",
            latency_s=0.01,
        )


class _NoCiteProvider:
    """Returns an answer with no citation markers."""

    model_name = "mock-no-cite"

    def complete(self, user_prompt: str, system_prompt: str, **kwargs):
        from enterprise_rag.planning.providers.base import ProviderResponse
        return ProviderResponse(
            content="The limit is two hundred dollars.",
            prompt_tokens=50,
            completion_tokens=10,
            model="mock-no-cite",
            latency_s=0.01,
        )


class _FailingProvider:
    model_name = "fail"

    def complete(self, *a, **k):
        raise RuntimeError("API error")


# ── _mean_score ────────────────────────────────────────────────────────────────


def test_mean_score_empty():
    assert _mean_score([]) == pytest.approx(0.0)


def test_mean_score_single():
    assert _mean_score([_chunk(score=0.6)]) == pytest.approx(0.6)


def test_mean_score_multiple():
    chunks = [_chunk(chunk_id="a", score=0.8), _chunk(chunk_id="b", score=0.4)]
    assert _mean_score(chunks) == pytest.approx(0.6)


# ── _extract_cited_ids ─────────────────────────────────────────────────────────


def _idx_map(*chunk_ids: str) -> dict[int, ChunkResult]:
    return {i + 1: _chunk(chunk_id=cid) for i, cid in enumerate(chunk_ids)}


def test_extract_cited_ids_basic():
    idx_map = _idx_map("c1", "c2", "c3")
    text = "The limit is $200 [1]. Exceptions need approval [2]."
    result = _extract_cited_ids(text, idx_map)
    assert result == ["c1", "c2"]


def test_extract_cited_ids_out_of_range():
    idx_map = _idx_map("c1", "c2")
    text = "See [5] for details."  # index 5 not in map
    result = _extract_cited_ids(text, idx_map)
    assert result == []


def test_extract_cited_ids_deduplicates():
    idx_map = _idx_map("c1", "c2")
    text = "See [1] and also [1] again."
    result = _extract_cited_ids(text, idx_map)
    assert result == ["c1"]  # deduplicated


def test_extract_cited_ids_preserves_order():
    idx_map = _idx_map("c1", "c2", "c3")
    text = "First [2], then [1], then [3]."
    result = _extract_cited_ids(text, idx_map)
    assert result == ["c1", "c2", "c3"]  # sorted by index, not appearance


def test_extract_cited_ids_no_citations():
    idx_map = _idx_map("c1")
    assert _extract_cited_ids("No markers here.", idx_map) == []


# ── AnswerConfig ───────────────────────────────────────────────────────────────


def test_answer_config_defaults():
    cfg = AnswerConfig()
    assert cfg.max_evidence_chunks == 10
    assert cfg.max_tokens == 512
    assert cfg.temperature == pytest.approx(0.1)


# ── AnswerAgent — no evidence ──────────────────────────────────────────────────


def test_synthesize_empty_evidence_returns_no_evidence_message():
    agent = AnswerAgent(provider=None)
    plan = _make_plan()
    result = agent.synthesize("What is the limit?", [], plan)
    assert isinstance(result, AgentAnswer)
    assert result.cited_chunk_ids == []
    assert result.confidence == pytest.approx(0.0)
    assert result.evidence_coverage == pytest.approx(0.0)
    assert "no relevant evidence" in result.answer_text.lower()


# ── AnswerAgent — extractive fallback ─────────────────────────────────────────


def test_extractive_fallback_no_provider():
    agent = AnswerAgent(provider=None)
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.8) for i in range(5)]
    result = agent.synthesize("hotel limit", chunks, plan)
    assert isinstance(result, AgentAnswer)
    assert len(result.cited_chunk_ids) == 3  # top-3
    assert result.confidence > 0.0
    # Extractive answer has [1] [2] [3] markers
    assert "[1]" in result.answer_text


def test_extractive_fallback_single_chunk():
    agent = AnswerAgent(provider=None)
    plan = _make_plan()
    chunks = [_chunk(chunk_id="c1", rank=1, score=0.7)]
    result = agent.synthesize("q", chunks, plan)
    assert result.cited_chunk_ids == ["c1"]
    assert result.evidence_coverage == pytest.approx(1.0)


def test_extractive_uses_rank_order():
    agent = AnswerAgent(provider=None)
    plan = _make_plan()
    # Deliberately pass in reverse rank order
    chunks = [
        _chunk(chunk_id="c3", rank=3, score=0.5),
        _chunk(chunk_id="c1", rank=1, score=0.9),
        _chunk(chunk_id="c2", rank=2, score=0.7),
    ]
    result = agent.synthesize("q", chunks, plan)
    # c1 (rank 1) should appear first in cited list
    assert result.cited_chunk_ids[0] == "c1"


# ── AnswerAgent — LLM synthesis ────────────────────────────────────────────────


def test_llm_synthesis_populates_cited_ids():
    agent = AnswerAgent(provider=_MockProvider())
    plan = _make_plan()
    chunks = [
        _chunk(chunk_id="c1", rank=1, text="The hotel limit is $200."),
        _chunk(chunk_id="c2", rank=2, text="Exceptions require manager approval."),
        _chunk(chunk_id="c3", rank=3, text="Receipts must be submitted within 30 days."),
    ]
    result = agent.synthesize("What is the hotel limit?", chunks, plan)
    assert isinstance(result, AgentAnswer)
    assert "c1" in result.cited_chunk_ids
    assert "c2" in result.cited_chunk_ids
    assert result.confidence > 0.0


def test_llm_synthesis_answer_text_from_provider():
    agent = AnswerAgent(provider=_MockProvider())
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1) for i in range(3)]
    result = agent.synthesize("hotel limit", chunks, plan)
    assert "$200" in result.answer_text


def test_llm_synthesis_no_citations_defaults_to_top3():
    agent = AnswerAgent(provider=_NoCiteProvider())
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1) for i in range(5)]
    result = agent.synthesize("q", chunks, plan)
    # No [N] markers in response → default to top-3
    assert len(result.cited_chunk_ids) == 3
    assert result.cited_chunk_ids == ["c0", "c1", "c2"]


def test_llm_synthesis_falls_back_on_provider_error():
    agent = AnswerAgent(provider=_FailingProvider())
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1) for i in range(5)]
    result = agent.synthesize("q", chunks, plan)
    # Falls back to extractive — should not raise
    assert isinstance(result, AgentAnswer)
    assert len(result.cited_chunk_ids) == 3


# ── Confidence and coverage ────────────────────────────────────────────────────


def test_confidence_is_mean_of_cited_chunks():
    agent = AnswerAgent(provider=_MockProvider())
    plan = _make_plan()
    # [1] and [2] cited by mock provider
    chunks = [
        _chunk(chunk_id="c1", rank=1, score=0.8, text="hotel limit $200"),
        _chunk(chunk_id="c2", rank=2, score=0.6, text="exceptions need approval"),
        _chunk(chunk_id="c3", rank=3, score=0.2, text="unrelated policy"),
    ]
    result = agent.synthesize("hotel limit", chunks, plan)
    # Confidence should reflect scores of cited c1(0.8) and c2(0.6), not c3
    assert result.confidence == pytest.approx((0.8 + 0.6) / 2, abs=0.01)


def test_coverage_fraction_of_all_evidence():
    agent = AnswerAgent(provider=_MockProvider())
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.7) for i in range(6)]
    result = agent.synthesize("q", chunks, plan)
    # cited_count / total_evidence ∈ [0, 1]
    assert 0.0 <= result.evidence_coverage <= 1.0


# ── max_evidence_chunks cap ────────────────────────────────────────────────────


def test_max_evidence_chunks_limits_context():
    cfg = AnswerConfig(max_evidence_chunks=2)
    agent = AnswerAgent(config=cfg, provider=_NoCiteProvider())
    plan = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1) for i in range(10)]
    result = agent.synthesize("q", chunks, plan)
    # No citations → defaults to top-3, but top_evidence is only 2 chunks
    assert len(result.cited_chunk_ids) <= 2


# ── get_answer_agent singleton ─────────────────────────────────────────────────


def test_get_answer_agent_returns_instance():
    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = None
    agent = get_answer_agent()
    assert isinstance(agent, AnswerAgent)
    ans_mod._agent = None


def test_get_answer_agent_reuses_singleton():
    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = None
    a1 = get_answer_agent()
    a2 = get_answer_agent()
    assert a1 is a2
    ans_mod._agent = None


# ── answer_agent_node ──────────────────────────────────────────────────────────


def test_answer_node_no_query_plan():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import answer_agent_node

    state = make_initial_state("q1", "test")
    state["query_plan"] = None
    state["validated_evidence"] = []

    result = answer_agent_node(state)
    assert result["answer"] is None
    assert result["trace"][0].event == "skipped"


def test_answer_node_empty_evidence():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import answer_agent_node

    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = AnswerAgent(provider=None)

    state = make_initial_state("q1", "What is the hotel limit?")
    state["query_plan"] = _make_plan()
    state["validated_evidence"] = []

    result = answer_agent_node(state)
    assert result["answer"] is not None
    assert result["answer"].cited_chunk_ids == []
    assert "no relevant evidence" in result["answer"].answer_text.lower()
    ans_mod._agent = None


def test_answer_node_with_evidence():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import answer_agent_node

    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = AnswerAgent(provider=None)

    state = make_initial_state("q1", "What is the hotel limit?")
    state["query_plan"] = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.8) for i in range(5)]
    state["validated_evidence"] = chunks

    result = answer_agent_node(state)
    assert "answer" in result
    assert result["answer"].query_id == "q1"
    assert len(result["answer"].cited_chunk_ids) > 0
    assert "trace" in result
    ans_mod._agent = None


def test_answer_node_trace_has_expected_keys():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import answer_agent_node

    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = AnswerAgent(provider=None)

    state = make_initial_state("q1", "test")
    state["query_plan"] = _make_plan()
    chunks = [_chunk(chunk_id=f"c{i}", rank=i + 1, score=0.7) for i in range(3)]
    state["validated_evidence"] = chunks

    result = answer_agent_node(state)
    trace = result["trace"][0]
    assert trace.agent == "answer"
    assert trace.event == "answer_generated"
    assert "cited_count" in trace.data
    assert "confidence" in trace.data
    assert trace.data["stub"] is False
    ans_mod._agent = None


def test_answer_node_with_mock_llm_provider():
    from enterprise_rag.agents.state import make_initial_state
    from enterprise_rag.graph.nodes import answer_agent_node

    import enterprise_rag.agents.answer as ans_mod
    ans_mod._agent = AnswerAgent(provider=_MockProvider())

    state = make_initial_state("q1", "What is the hotel limit?")
    state["query_plan"] = _make_plan()
    chunks = [
        _chunk(chunk_id="c1", rank=1, text="Hotel limit is $200 per night."),
        _chunk(chunk_id="c2", rank=2, text="Exceptions require approval."),
        _chunk(chunk_id="c3", rank=3, text="Receipt submission deadline is 30 days."),
    ]
    state["validated_evidence"] = chunks

    result = answer_agent_node(state)
    assert "$200" in result["answer"].answer_text
    assert "c1" in result["answer"].cited_chunk_ids
    ans_mod._agent = None
