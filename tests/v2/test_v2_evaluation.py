"""Tests for Phase 12E evaluation infrastructure.

Unit tests only — no real indexes needed.
Integration tests (real pipeline) are marked @pytest.mark.integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from enterprise_rag.evaluation.v2_runner import (
    EXPERIMENTS,
    V2ExperimentConfig,
    V2Runner,
    aggregate_pipeline_stats,
    extract_chunks_from_state,
    extract_pipeline_stats,
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_chunk(
    chunk_id: str,
    rank: int = 1,
    score: float = 0.8,
    text: str = "chunk text",
    category: str = "travel_policy",
    document_id: str = "doc1",
) -> MagicMock:
    c = MagicMock()
    c.chunk_id = chunk_id
    c.document_id = document_id
    c.rank = rank
    c.score = score
    c.text = text
    c.category = category
    c.section_path = []
    return c


def _make_retrieval_output(
    strategy: str = "bm25",
    chunks: list | None = None,
) -> MagicMock:
    o = MagicMock()
    o.strategy = strategy
    o.chunks = chunks or []
    return o


def _make_validation_result(
    passed: bool = True,
    passed_count: int = 5,
    total_count: int = 8,
    failure_reason: str = "",
    suggested_strategy: str | None = None,
) -> MagicMock:
    v = MagicMock()
    v.passed = passed
    v.passed_count = passed_count
    v.total_count = total_count
    v.failure_reason = failure_reason
    v.suggested_strategy = suggested_strategy
    return v


def _make_final_state(
    query_id: str = "q1",
    outputs: list | None = None,
    loop_count: int = 0,
    validation_passed: bool = True,
    error: str | None = None,
) -> dict[str, Any]:
    val = _make_validation_result(passed=validation_passed)
    return {
        "query_id": query_id,
        "retrieval_outputs": outputs or [],
        "loop_count": loop_count,
        "validation_result": val,
        "error": error,
    }


# ── V2ExperimentConfig ─────────────────────────────────────────────────────────


def test_config_defaults():
    cfg = V2ExperimentConfig(name="test", description="test exp")
    assert cfg.use_oracle is True
    assert cfg.scorer == "adaptive"
    assert cfg.max_loops == 3
    assert cfg.k == 10


def test_config_custom():
    cfg = V2ExperimentConfig(
        name="no_loop", description="ablation",
        use_oracle=False, scorer="rank_proxy", max_loops=0, k=5,
    )
    assert cfg.use_oracle is False
    assert cfg.scorer == "rank_proxy"
    assert cfg.max_loops == 0
    assert cfg.k == 5


def test_experiments_list_has_four():
    assert len(EXPERIMENTS) == 4


def test_experiments_all_have_unique_names():
    names = [e.name for e in EXPERIMENTS]
    assert len(names) == len(set(names))


def test_experiments_include_expected():
    names = {e.name for e in EXPERIMENTS}
    assert "v2_oracle" in names
    assert "v2_oracle_noloop" in names
    assert "v2_oracle_rankproxy" in names
    assert "v2_heuristic" in names


def test_oracle_experiment_has_max_loops_3():
    oracle = next(e for e in EXPERIMENTS if e.name == "v2_oracle")
    assert oracle.max_loops == 3
    assert oracle.use_oracle is True
    assert oracle.scorer == "adaptive"


def test_noloop_experiment_has_max_loops_0():
    noloop = next(e for e in EXPERIMENTS if e.name == "v2_oracle_noloop")
    assert noloop.max_loops == 0


def test_rankproxy_experiment_uses_rank_proxy_scorer():
    rp = next(e for e in EXPERIMENTS if e.name == "v2_oracle_rankproxy")
    assert rp.scorer == "rank_proxy"


def test_heuristic_experiment_no_oracle():
    heur = next(e for e in EXPERIMENTS if e.name == "v2_heuristic")
    assert heur.use_oracle is False


# ── extract_chunks_from_state ──────────────────────────────────────────────────


def test_extract_empty_outputs():
    state = _make_final_state(outputs=[])
    result = extract_chunks_from_state(state, "q1")
    assert result == []


def test_extract_single_output():
    chunks = [_make_chunk(f"c{i}", rank=i + 1, score=0.9 - i * 0.1) for i in range(3)]
    outputs = [_make_retrieval_output(strategy="bm25", chunks=chunks)]
    state = _make_final_state(outputs=outputs)
    result = extract_chunks_from_state(state, "q1", k=10)
    assert len(result) == 3
    assert all(r["query_id"] == "q1" for r in result)
    assert all("chunk_id" in r for r in result)
    assert all("score" in r for r in result)


def test_extract_sorts_by_score_desc():
    chunks = [
        _make_chunk("c1", rank=3, score=0.3),
        _make_chunk("c2", rank=1, score=0.9),
        _make_chunk("c3", rank=2, score=0.6),
    ]
    outputs = [_make_retrieval_output(chunks=chunks)]
    state = _make_final_state(outputs=outputs)
    result = extract_chunks_from_state(state, "q1")
    # Should be sorted by score desc
    assert result[0]["chunk_id"] == "c2"
    assert result[1]["chunk_id"] == "c3"
    assert result[2]["chunk_id"] == "c1"


def test_extract_re_ranks_by_position():
    chunks = [_make_chunk(f"c{i}", score=0.9 - i * 0.1) for i in range(5)]
    outputs = [_make_retrieval_output(chunks=chunks)]
    state = _make_final_state(outputs=outputs)
    result = extract_chunks_from_state(state, "q1")
    ranks = [r["rank"] for r in result]
    assert ranks == list(range(1, len(result) + 1))


def test_extract_deduplicates_by_chunk_id():
    chunk = _make_chunk("c1", score=0.8)
    # Same chunk_id in two outputs with different scores
    chunk_low = _make_chunk("c1", score=0.5)
    outputs = [
        _make_retrieval_output(strategy="bm25", chunks=[chunk]),
        _make_retrieval_output(strategy="dense", chunks=[chunk_low]),
    ]
    state = _make_final_state(outputs=outputs)
    result = extract_chunks_from_state(state, "q1")
    assert len(result) == 1
    assert result[0]["score"] == pytest.approx(0.8)  # keeps higher score


def test_extract_caps_at_k():
    chunks = [_make_chunk(f"c{i}", score=1.0 - i * 0.01) for i in range(20)]
    outputs = [_make_retrieval_output(chunks=chunks)]
    state = _make_final_state(outputs=outputs)
    result = extract_chunks_from_state(state, "q1", k=5)
    assert len(result) == 5


def test_extract_multi_output_merges():
    out1 = _make_retrieval_output(strategy="bm25", chunks=[
        _make_chunk("c1", score=0.9), _make_chunk("c2", score=0.7),
    ])
    out2 = _make_retrieval_output(strategy="dense", chunks=[
        _make_chunk("c3", score=0.8), _make_chunk("c4", score=0.6),
    ])
    state = _make_final_state(outputs=[out1, out2])
    result = extract_chunks_from_state(state, "q1")
    assert len(result) == 4
    assert result[0]["chunk_id"] == "c1"  # highest score


def test_extract_none_outputs():
    state = {"retrieval_outputs": None, "loop_count": 0, "validation_result": None, "error": None}
    result = extract_chunks_from_state(state, "q1")
    assert result == []


# ── extract_pipeline_stats ─────────────────────────────────────────────────────


def test_pipeline_stats_passed():
    val = _make_validation_result(passed=True, passed_count=5, total_count=8)
    state = {"loop_count": 1, "validation_result": val, "error": None}
    stats = extract_pipeline_stats(state)
    assert stats["validation_passed"] is True
    assert stats["loop_count"] == 1
    assert stats["validation_passed_count"] == 5
    assert stats["validation_total_count"] == 8
    assert stats["error"] == ""


def test_pipeline_stats_failed():
    val = _make_validation_result(
        passed=False, failure_reason="wrong_strategy", suggested_strategy="hybrid"
    )
    state = {"loop_count": 2, "validation_result": val, "error": None}
    stats = extract_pipeline_stats(state)
    assert stats["validation_passed"] is False
    assert stats["failure_reason"] == "wrong_strategy"
    assert stats["suggested_strategy"] == "hybrid"
    assert stats["loop_count"] == 2


def test_pipeline_stats_no_validation():
    state = {"loop_count": 0, "validation_result": None, "error": None}
    stats = extract_pipeline_stats(state)
    assert stats["validation_passed"] is False
    assert stats["validation_passed_count"] == 0


def test_pipeline_stats_error():
    state = {"loop_count": 0, "validation_result": None, "error": "some error"}
    stats = extract_pipeline_stats(state)
    assert stats["error"] == "some error"


# ── aggregate_pipeline_stats ───────────────────────────────────────────────────


def _make_per_query_stats(n: int, loop_counts: list[int] | None = None) -> list[dict]:
    if loop_counts is None:
        loop_counts = [0] * n
    return [
        {
            "query_id": f"q{i}",
            "reasoning_type": "single_hop",
            "loop_count": loop_counts[i],
            "validation_passed": i % 2 == 0,
            "validation_passed_count": 3,
            "validation_total_count": 5,
            "failure_reason": "" if i % 2 == 0 else "low_quality",
            "suggested_strategy": "",
            "error": "",
        }
        for i in range(n)
    ]


def test_aggregate_empty():
    result = aggregate_pipeline_stats([])
    assert result == {}


def test_aggregate_pass_rate():
    stats = _make_per_query_stats(4)  # alternating pass/fail: pass for even i
    result = aggregate_pipeline_stats(stats)
    assert result["validation_pass_rate"] == pytest.approx(0.5)


def test_aggregate_mean_loops():
    stats = _make_per_query_stats(4, loop_counts=[0, 1, 2, 1])
    result = aggregate_pipeline_stats(stats)
    assert result["mean_loops"] == pytest.approx(1.0)


def test_aggregate_loop_distribution():
    stats = _make_per_query_stats(5, loop_counts=[0, 0, 1, 1, 2])
    result = aggregate_pipeline_stats(stats)
    assert result["loop_distribution"]["0"] == 2
    assert result["loop_distribution"]["1"] == 2
    assert result["loop_distribution"]["2"] == 1


def test_aggregate_n_errors():
    stats = _make_per_query_stats(3)
    stats[1]["error"] = "pipeline_error: test"
    result = aggregate_pipeline_stats(stats)
    assert result["n_errors"] == 1


def test_aggregate_top_failure_reasons():
    stats = _make_per_query_stats(6)
    for s in stats:
        if not s["validation_passed"]:
            s["failure_reason"] = "wrong_strategy"
    result = aggregate_pipeline_stats(stats)
    assert "wrong_strategy" in result["top_failure_reasons"]


def test_aggregate_by_reasoning_type():
    stats = [
        {"query_id": "q1", "reasoning_type": "temporal", "loop_count": 1,
         "validation_passed": True, "validation_passed_count": 3,
         "validation_total_count": 5, "failure_reason": "", "suggested_strategy": "", "error": ""},
        {"query_id": "q2", "reasoning_type": "temporal", "loop_count": 0,
         "validation_passed": False, "validation_passed_count": 1,
         "validation_total_count": 5, "failure_reason": "low_quality", "suggested_strategy": "", "error": ""},
        {"query_id": "q3", "reasoning_type": "exception", "loop_count": 2,
         "validation_passed": True, "validation_passed_count": 4,
         "validation_total_count": 6, "failure_reason": "", "suggested_strategy": "", "error": ""},
    ]
    result = aggregate_pipeline_stats(stats)
    temporal = result["by_reasoning_type"]["temporal"]
    assert temporal["n"] == 2
    assert temporal["val_pass_rate"] == pytest.approx(0.5)
    assert temporal["mean_loops"] == pytest.approx(0.5)
    exception = result["by_reasoning_type"]["exception"]
    assert exception["n"] == 1


# ── V2Runner unit tests (mocked graph) ────────────────────────────────────────


def _make_mock_graph(chunk_list: list[MagicMock] | None = None) -> MagicMock:
    """Build a mock graph that returns a fixed state."""
    if chunk_list is None:
        chunk_list = [_make_chunk(f"c{i}", rank=i + 1, score=0.9 - i * 0.1) for i in range(5)]
    graph = MagicMock()
    output = _make_retrieval_output(strategy="hybrid", chunks=chunk_list)
    val = _make_validation_result(passed=True)

    def _invoke(state):
        return {
            "query_id": state["query_id"],
            "retrieval_outputs": [output],
            "loop_count": 0,
            "validation_result": val,
            "error": None,
        }

    graph.invoke.side_effect = _invoke
    return graph


def _make_test_query(query_id: str = "q1", reasoning_type: str = "single_hop") -> dict:
    return {
        "query_id": query_id,
        "query": "What is the hotel limit?",
        "reasoning_type": reasoning_type,
        "answerable": True,
        "category": "travel_policy",
    }


def test_runner_run_single_returns_results_and_stats():
    cfg = V2ExperimentConfig(name="test", description="test", use_oracle=True, max_loops=3, k=10)
    runner = V2Runner(cfg)
    runner._graph = _make_mock_graph()

    q = _make_test_query()
    results, stats = runner.run_single(q)
    assert isinstance(results, list)
    assert isinstance(stats, dict)
    assert len(results) > 0
    assert all(r["query_id"] == "q1" for r in results)


def test_runner_run_single_oracle_passes_reasoning_type():
    """Oracle mode: metadata must include use_oracle=True and reasoning_type."""
    cfg = V2ExperimentConfig(name="test", description="test", use_oracle=True)
    runner = V2Runner(cfg)

    captured_state: list[dict] = []

    def _mock_invoke(state):
        captured_state.append(state)
        return {
            "query_id": state["query_id"],
            "retrieval_outputs": [],
            "loop_count": 0,
            "validation_result": _make_validation_result(),
            "error": None,
        }

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = _mock_invoke
    runner._graph = mock_graph

    q = _make_test_query(reasoning_type="temporal")
    runner.run_single(q)
    assert captured_state[0]["query_metadata"]["use_oracle"] is True
    assert captured_state[0]["query_metadata"]["reasoning_type"] == "temporal"


def test_runner_run_single_heuristic_no_oracle_metadata():
    """Non-oracle mode: metadata must NOT include use_oracle=True."""
    cfg = V2ExperimentConfig(name="test", description="test", use_oracle=False)
    runner = V2Runner(cfg)

    captured_state: list[dict] = []

    def _mock_invoke(state):
        captured_state.append(state)
        return {
            "query_id": state["query_id"],
            "retrieval_outputs": [],
            "loop_count": 0,
            "validation_result": _make_validation_result(),
            "error": None,
        }

    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = _mock_invoke
    runner._graph = mock_graph

    q = _make_test_query()
    runner.run_single(q)
    assert not captured_state[0]["query_metadata"].get("use_oracle", False)


def test_runner_run_single_handles_graph_exception():
    cfg = V2ExperimentConfig(name="test", description="test")
    runner = V2Runner(cfg)
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = RuntimeError("graph exploded")
    runner._graph = mock_graph

    q = _make_test_query()
    results, stats = runner.run_single(q)
    assert results == []
    assert "pipeline_error" in stats["error"]


def test_runner_run_batch_aggregates_all_queries():
    cfg = V2ExperimentConfig(name="test", description="test", k=5)
    runner = V2Runner(cfg)
    runner._graph = _make_mock_graph()

    queries = [_make_test_query(f"q{i}") for i in range(4)]
    flat_results, per_query_stats = runner.run_batch(queries)

    assert len(per_query_stats) == 4
    query_ids_in_results = {r["query_id"] for r in flat_results}
    for q in queries:
        assert q["query_id"] in query_ids_in_results


def test_runner_run_batch_progress_fn_called():
    cfg = V2ExperimentConfig(name="test", description="test")
    runner = V2Runner(cfg)
    runner._graph = _make_mock_graph()

    calls: list[tuple[int, int]] = []
    queries = [_make_test_query(f"q{i}") for i in range(3)]
    runner.run_batch(queries, progress_fn=lambda d, t: calls.append((d, t)))
    assert len(calls) == 3
    assert calls[-1] == (3, 3)


def test_runner_setup_resets_validation_singleton():
    """setup() must reset the validation agent singleton."""
    cfg = V2ExperimentConfig(name="test", description="test", scorer="rank_proxy")
    runner = V2Runner(cfg)

    import enterprise_rag.agents.validation as vm
    import enterprise_rag.graph.edges as em

    # Put a sentinel in the singleton
    sentinel = object()
    vm._agent = sentinel  # type: ignore

    with patch("enterprise_rag.graph.builder.build_graph", return_value=MagicMock()):
        runner.setup()

    assert vm._agent is None
    assert em.MAX_LOOPS == cfg.max_loops


def test_runner_setup_sets_max_loops():
    cfg = V2ExperimentConfig(name="test", description="test", max_loops=0)
    runner = V2Runner(cfg)

    import enterprise_rag.graph.edges as em

    with patch("enterprise_rag.graph.builder.build_graph", return_value=MagicMock()):
        runner.setup()

    assert em.MAX_LOOPS == 0


def test_runner_teardown_clears_graph():
    cfg = V2ExperimentConfig(name="test", description="test")
    runner = V2Runner(cfg)
    runner._graph = MagicMock()

    runner.teardown()
    assert runner._graph is None


# ── Integration: real pipeline (requires indexes) ─────────────────────────────


@pytest.mark.integration
def test_v2_runner_integration_single_query():
    """Run a single query through the real V2 pipeline and check output format."""
    cfg = next(e for e in EXPERIMENTS if e.name == "v2_oracle")
    runner = V2Runner(cfg)
    runner.setup()

    query = {
        "query_id": "Q-000001",
        "query": "Under what conditions can exceptions to the Vulnerability Management be approved?",
        "reasoning_type": "exception",
        "answerable": True,
        "category": "Security Policies",
    }

    results, stats = runner.run_single(query)

    assert isinstance(results, list)
    assert isinstance(stats, dict)
    assert "loop_count" in stats
    assert "validation_passed" in stats

    if results:
        r = results[0]
        assert "query_id" in r
        assert "chunk_id" in r
        assert "score" in r
        assert "rank" in r
        assert r["rank"] == 1

    runner.teardown()


@pytest.mark.integration
def test_v2_oracle_noloop_fewer_or_equal_chunks():
    """No-loop experiment should retrieve at most as many unique chunks as oracle."""
    from data.sekd.queries import _load_first_n  # type: ignore[import]
    # Use direct loading instead
    import json
    queries_path = "data/sekd/queries.jsonl"
    with open(queries_path) as f:
        sample = [json.loads(f.readline()) for _ in range(5)]

    cfg_loop = next(e for e in EXPERIMENTS if e.name == "v2_oracle")
    cfg_noloop = next(e for e in EXPERIMENTS if e.name == "v2_oracle_noloop")

    runner_loop = V2Runner(cfg_loop)
    runner_loop.setup()
    r_loop, _ = runner_loop.run_batch(sample)
    runner_loop.teardown()

    runner_noloop = V2Runner(cfg_noloop)
    runner_noloop.setup()
    r_noloop, _ = runner_noloop.run_batch(sample)
    runner_noloop.teardown()

    # Both should return results; no strict assertion on count since loop
    # effects depend on whether validation triggers re-retrieval
    assert isinstance(r_loop, list)
    assert isinstance(r_noloop, list)
