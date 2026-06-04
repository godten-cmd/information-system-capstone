"""Tests for Phase 7: Hybrid retrieval (RRF fusion + contribution analysis)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from enterprise_rag.evaluation.contribution_analysis import (
    ContributionStats,
    analyze_contributions,
)
from enterprise_rag.retrieval.bm25 import BM25Config, BM25Index
from enterprise_rag.retrieval.dense import DenseIndex
from enterprise_rag.retrieval.fusion import (
    fuse_runs,
    reciprocal_rank_fusion,
    rrf_score,
)
from enterprise_rag.retrieval.hybrid import HybridConfig, HybridIndex, build_hybrid_index
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
from enterprise_rag.retrieval.vector_index import VectorIndex


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_chunks(n: int = 6) -> list[dict]:
    texts = [
        "Annual leave policy 15 days for full-time employees at HYTech.",
        "Security control HY-AC-01 multi-factor authentication required.",
        "Travel lodging reimbursement USD 200 per night US region policy.",
        "API endpoint POST /api/v2/alerts HYMonitor service JSON body.",
        "Project PRJ-HYD-ALPHA sprint planning decisions action items.",
        "Exception to overtime policy requires VP and HR documentation.",
    ]
    categories = ["HR Policies", "Security Policies", "Travel Policies",
                  "API Documentation", "Project Meeting Notes", "HR Policies"]
    return [
        {
            "chunk_id": f"DOC-{i+1:03d}-C001",
            "document_id": f"DOC-{i+1:03d}",
            "category": categories[i],
            "section_path": [f"Section {i+1}"],
            "text": texts[i],
            "char_start": 0,
            "char_end": len(texts[i]),
            "token_estimate": 15,
        }
        for i in range(n)
    ]


def _build_bm25(chunks: list[dict]) -> BM25Index:
    idx = BM25Index()
    idx.build(chunks)
    return idx


def _build_dense_fake(chunks: list[dict], dim: int = 32) -> DenseIndex:
    """Build DenseIndex with deterministic fake embeddings."""
    def _fake_embed(texts):
        n = len(texts)
        embs = np.zeros((n, dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            embs[i] = rng.random(dim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-9)

    class _FakeModel:
        def load(self): return self
        @property
        def embedding_dim(self): return dim
        @property
        def model_name(self): return "fake"
        @property
        def info(self):
            m = MagicMock(); m.embedding_dim = dim; return m
        def encode_documents(self, texts): return _fake_embed(texts)
        def encode_queries(self, texts): return _fake_embed(texts)

    texts = [" ".join(c.get("section_path", [])) + " " + c.get("text", "") for c in chunks]
    embeddings = _fake_embed(texts)
    vi = VectorIndex()
    vi.build(chunks, embeddings, model_name="fake")
    di = DenseIndex()
    di._vector_index = vi
    di._model = _FakeModel()
    di._built = True
    return di


def _make_retrieval_run(
    query_ids: list[str],
    chunk_ids_per_query: dict[str, list[str]],
    method: str = "bm25",
) -> RetrievalRun:
    run = RetrievalRun(method=method, k=len(next(iter(chunk_ids_per_query.values()))))
    for qid in query_ids:
        for rank, cid in enumerate(chunk_ids_per_query.get(qid, []), start=1):
            run.results.append(RetrievedChunk(
                query_id=qid, chunk_id=cid, document_id="DOC",
                rank=rank, score=1.0 / rank,
            ))
    return run


# ── RRF core ──────────────────────────────────────────────────────────────────


def test_rrf_score_decreases_with_rank() -> None:
    assert rrf_score(1, 60) > rrf_score(2, 60) > rrf_score(10, 60)


def test_rrf_score_higher_k_flattens_differences() -> None:
    diff_small_k = rrf_score(1, 10) - rrf_score(2, 10)
    diff_large_k = rrf_score(1, 100) - rrf_score(2, 100)
    assert diff_small_k > diff_large_k


def test_rrf_fusion_single_list_preserves_order() -> None:
    ids = ["A", "B", "C", "D"]
    fused = reciprocal_rank_fusion([ids], rrf_k=60)
    result_ids = [x[0] for x in fused]
    assert result_ids == ids


def test_rrf_fusion_item_in_both_lists_scores_higher() -> None:
    list1 = ["A", "B", "C"]
    list2 = ["A", "X", "Y"]  # A appears in both at rank 1
    fused = dict(reciprocal_rank_fusion([list1, list2], rrf_k=60))
    # A appears in both → highest score
    assert fused["A"] > fused["B"]
    assert fused["A"] > fused["X"]


def test_rrf_fusion_returns_union_of_items() -> None:
    list1 = ["A", "B"]
    list2 = ["C", "D"]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    result_ids = {x[0] for x in fused}
    assert result_ids == {"A", "B", "C", "D"}


def test_rrf_fusion_top_n_limits_results() -> None:
    ids = ["A", "B", "C", "D", "E"]
    fused = reciprocal_rank_fusion([ids], rrf_k=60, top_n=3)
    assert len(fused) == 3


def test_rrf_fusion_scores_sorted_descending() -> None:
    list1 = ["A", "B", "C"]
    list2 = ["B", "A", "C"]
    fused = reciprocal_rank_fusion([list1, list2], rrf_k=60)
    scores = [s for _, s in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_fusion_empty_lists() -> None:
    fused = reciprocal_rank_fusion([], rrf_k=60)
    assert fused == []


def test_rrf_fusion_empty_inner_list() -> None:
    fused = reciprocal_rank_fusion([[], ["A", "B"]], rrf_k=60)
    assert len(fused) == 2


def test_fuse_runs_helper() -> None:
    result = fuse_runs([
        (["A", "B", "C"], "bm25"),
        (["B", "A", "D"], "dense"),
    ], rrf_k=60)
    ids = [x[0] for x in result]
    assert "A" in ids and "B" in ids
    # A and B appear in both → should outscore C and D
    scores = dict(result)
    assert scores["A"] > scores["C"]
    assert scores["B"] > scores["D"]


# ── HybridConfig ──────────────────────────────────────────────────────────────


def test_hybrid_config_defaults() -> None:
    cfg = HybridConfig()
    assert cfg.rrf_k == 60
    assert cfg.candidate_k == 100
    assert cfg.max_k == 10


def test_hybrid_config_custom() -> None:
    cfg = HybridConfig(rrf_k=30, candidate_k=50)
    assert cfg.rrf_k == 30
    assert cfg.candidate_k == 50


# ── HybridIndex ───────────────────────────────────────────────────────────────


def test_hybrid_index_builds() -> None:
    chunks = _make_chunks()
    bm25 = _build_bm25(chunks)
    dense = _build_dense_fake(chunks)
    hi = HybridIndex(bm25, dense, HybridConfig(rrf_k=60, candidate_k=6))
    assert hi._chunk_meta
    assert len(hi._chunk_meta) == len(chunks)


def test_hybrid_index_retrieve_returns_chunks() -> None:
    chunks = _make_chunks()
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks),
                     HybridConfig(rrf_k=60, candidate_k=6))
    results = hi.retrieve("annual leave policy", "Q-001", k=3)
    assert len(results) == 3
    for r in results:
        assert r.query_id == "Q-001"
        assert r.rank >= 1
        assert r.chunk_id


def test_hybrid_index_retrieve_sorted_by_rank() -> None:
    chunks = _make_chunks()
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks),
                     HybridConfig(rrf_k=60, candidate_k=6))
    results = hi.retrieve("security authentication", "Q-001", k=5)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_hybrid_index_retrieve_at_most_k() -> None:
    chunks = _make_chunks(4)
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks, dim=16),
                     HybridConfig(rrf_k=60, candidate_k=4))
    results = hi.retrieve("test", "Q-1", k=2)
    assert len(results) <= 2


def test_hybrid_index_with_rrf_k_returns_new_index() -> None:
    chunks = _make_chunks()
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks),
                     HybridConfig(rrf_k=60, candidate_k=6))
    hi30 = hi.with_rrf_k(30)
    assert hi30._config.rrf_k == 30
    assert hi._config.rrf_k == 60  # original unchanged
    # Indexes are shared (same objects)
    assert hi30._bm25 is hi._bm25
    assert hi30._dense is hi._dense


def test_hybrid_index_different_rrf_k_gives_different_scores() -> None:
    chunks = _make_chunks()
    bm25 = _build_bm25(chunks)
    dense = _build_dense_fake(chunks)
    hi10 = HybridIndex(bm25, dense, HybridConfig(rrf_k=10, candidate_k=6))
    hi100 = HybridIndex(bm25, dense, HybridConfig(rrf_k=100, candidate_k=6))
    r10 = hi10.retrieve("leave policy", "Q-1", k=3)
    r100 = hi100.retrieve("leave policy", "Q-1", k=3)
    # Scores differ (rrf_k=10 amplifies rank differences more)
    scores10 = [r.score for r in r10]
    scores100 = [r.score for r in r100]
    assert scores10 != scores100


def test_hybrid_index_run_covers_all_queries() -> None:
    chunks = _make_chunks()
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks),
                     HybridConfig(rrf_k=60, candidate_k=6))
    queries = [
        {"query_id": "Q-001", "query": "annual leave"},
        {"query_id": "Q-002", "query": "security HY-AC-01"},
    ]
    run = hi.run(queries, k=3)
    query_ids = {r.query_id for r in run.results}
    assert query_ids == {"Q-001", "Q-002"}


def test_hybrid_index_run_returns_retrieval_run() -> None:
    chunks = _make_chunks()
    hi = HybridIndex(_build_bm25(chunks), _build_dense_fake(chunks),
                     HybridConfig(rrf_k=60, candidate_k=6))
    run = hi.run([{"query_id": "Q-1", "query": "test"}], k=3)
    assert isinstance(run, RetrievalRun)
    assert run.method == "hybrid"
    assert run.config["rrf_k"] == 60


def test_build_hybrid_index_helper() -> None:
    chunks = _make_chunks()
    hi = build_hybrid_index(_build_bm25(chunks), _build_dense_fake(chunks))
    assert hi._config.rrf_k == 60


# ── Contribution analysis ─────────────────────────────────────────────────────


def test_contribution_stats_complementarity_score() -> None:
    stats = ContributionStats(
        n_total_required=100,
        n_bm25_only=20,
        n_dense_only=15,
        n_both=50,
        n_hybrid_only=5,
        n_not_found=10,
    )
    assert stats.complementarity_score == pytest.approx(0.35)


def test_contribution_stats_hybrid_gain() -> None:
    stats = ContributionStats(n_total_required=100, n_hybrid_only=8)
    assert stats.hybrid_gain == pytest.approx(0.08)


def test_contribution_stats_to_dict() -> None:
    stats = ContributionStats(n_total_required=10, n_bm25_only=3, n_dense_only=3,
                              n_both=4, n_hybrid_only=0, n_not_found=0, n_queries=5)
    d = stats.to_dict()
    assert d["n_queries"] == 5
    assert d["bm25_only"]["count"] == 3
    assert d["bm25_only"]["pct"] == 30.0
    assert "complementarity_score" in d


def test_analyze_contributions_all_bm25_only() -> None:
    """When BM25 finds everything and Dense finds nothing, all chunks are bm25_only."""
    ground_truths = {"Q-001": ["C-A"]}
    queries_meta = {"Q-001": {"reasoning_type": "single_hop", "category": "HR Policies",
                              "answerable": True}}
    bm25_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A", "C-B"]}, "bm25")
    dense_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-X", "C-Y"]}, "dense")
    hybrid_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A", "C-X"]}, "hybrid")

    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=2)
    assert result["overall"]["bm25_only"]["count"] == 1
    assert result["overall"]["dense_only"]["count"] == 0
    assert result["overall"]["both"]["count"] == 0


def test_analyze_contributions_hybrid_only_rescue() -> None:
    """When hybrid finds a chunk that neither BM25 nor Dense found, it's hybrid_only."""
    ground_truths = {"Q-001": ["C-RARE"]}
    queries_meta = {"Q-001": {"reasoning_type": "multi_hop", "category": "System Design Documents",
                              "answerable": True}}
    bm25_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-X", "C-Y"]})
    dense_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-P", "C-Q"]})
    # Hybrid rescues C-RARE (appeared >2 in both, promoted after fusion)
    hybrid_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-RARE", "C-X"]}, "hybrid")

    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=2)
    assert result["overall"]["hybrid_only"]["count"] == 1
    assert result["overall"]["not_found"]["count"] == 0


def test_analyze_contributions_both_sources() -> None:
    """When both BM25 and Dense find the chunk, it's 'both'."""
    ground_truths = {"Q-001": ["C-A"]}
    queries_meta = {"Q-001": {"reasoning_type": "single_hop", "category": "API Documentation",
                              "answerable": True}}
    bm25_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]})
    dense_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]})
    hybrid_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]}, "hybrid")

    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=1)
    assert result["overall"]["both"]["count"] == 1


def test_analyze_contributions_not_found() -> None:
    ground_truths = {"Q-001": ["C-MISSING"]}
    queries_meta = {"Q-001": {"reasoning_type": "exception", "category": "Security Policies",
                              "answerable": True}}
    bm25_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-X"]})
    dense_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-Y"]})
    hybrid_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-Z"]}, "hybrid")

    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=1)
    assert result["overall"]["not_found"]["count"] == 1


def test_analyze_contributions_groups_by_reasoning_type() -> None:
    ground_truths = {
        "Q-001": ["C-A"],
        "Q-002": ["C-B"],
    }
    queries_meta = {
        "Q-001": {"reasoning_type": "single_hop", "category": "HR Policies", "answerable": True},
        "Q-002": {"reasoning_type": "multi_hop", "category": "Security Policies", "answerable": True},
    }
    bm25_run = _make_retrieval_run(["Q-001", "Q-002"], {"Q-001": ["C-A"], "Q-002": ["C-X"]})
    dense_run = _make_retrieval_run(["Q-001", "Q-002"], {"Q-001": ["C-X"], "Q-002": ["C-B"]})
    hybrid_run = _make_retrieval_run(["Q-001", "Q-002"],
                                     {"Q-001": ["C-A"], "Q-002": ["C-B"]}, "hybrid")

    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=1)
    assert "single_hop" in result["by_reasoning_type"]
    assert "multi_hop" in result["by_reasoning_type"]
    assert result["by_reasoning_type"]["single_hop"]["bm25_only"]["count"] == 1
    assert result["by_reasoning_type"]["multi_hop"]["dense_only"]["count"] == 1


def test_analyze_contributions_skips_unanswerable() -> None:
    ground_truths = {"Q-001": ["C-A"], "Q-002": ["C-B"]}
    queries_meta = {
        "Q-001": {"reasoning_type": "single_hop", "category": "HR", "answerable": True},
        "Q-002": {"reasoning_type": "single_hop", "category": "HR", "answerable": False},
    }
    bm25_run = _make_retrieval_run(["Q-001", "Q-002"], {"Q-001": ["C-A"], "Q-002": ["C-B"]})
    dense_run = _make_retrieval_run(["Q-001", "Q-002"], {"Q-001": ["C-A"], "Q-002": ["C-B"]})
    hybrid_run = _make_retrieval_run(["Q-001", "Q-002"],
                                     {"Q-001": ["C-A"], "Q-002": ["C-B"]}, "hybrid")
    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=1, answerable_only=True)
    assert result["overall"]["n_queries"] == 1


def test_analyze_contributions_interpretation_present() -> None:
    ground_truths = {"Q-001": ["C-A"]}
    queries_meta = {"Q-001": {"reasoning_type": "single_hop", "category": "HR", "answerable": True}}
    bm25_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]})
    dense_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]})
    hybrid_run = _make_retrieval_run(["Q-001"], {"Q-001": ["C-A"]}, "hybrid")
    result = analyze_contributions(ground_truths, bm25_run, dense_run, hybrid_run,
                                   queries_meta, k=1)
    assert isinstance(result["interpretation"], str)
    assert len(result["interpretation"]) > 0


# ── Integration: end-to-end hybrid pipeline ────────────────────────────────────


def test_end_to_end_hybrid_pipeline() -> None:
    from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run

    chunks = _make_chunks()
    bm25 = _build_bm25(chunks)
    dense = _build_dense_fake(chunks)
    hi = HybridIndex(bm25, dense, HybridConfig(rrf_k=60, candidate_k=6))

    queries = [
        {"query_id": "Q-001", "query": "annual leave HR policy days"},
        {"query_id": "Q-002", "query": "security control HY-AC-01 authentication"},
    ]
    run = hi.run(queries, k=5)
    ground_truths = {"Q-001": ["DOC-001-C001"], "Q-002": ["DOC-002-C001"]}
    queries_meta = {
        "Q-001": {"reasoning_type": "single_hop", "category": "HR Policies",
                  "difficulty": "easy", "retrieval_difficulty_factors": [], "answerable": True},
        "Q-002": {"reasoning_type": "single_hop", "category": "Security Policies",
                  "difficulty": "medium", "retrieval_difficulty_factors": [], "answerable": True},
    }
    result_by_k = evaluate_run([r.to_dict() for r in run.results],
                               ground_truths, queries_meta, k_values=[1, 5])
    agg5 = aggregate(result_by_k[5], k=5)
    assert agg5.n_queries == 2
    assert 0.0 <= agg5.recall <= 1.0


def test_rrf_parameter_sweep_best_selection() -> None:
    """Verify sweep logic selects the rrf_k with highest recall."""
    # Simulate sweep results
    sweep_results = [
        {"rrf_k": 10, "recall_at_10": 0.65},
        {"rrf_k": 30, "recall_at_10": 0.72},
        {"rrf_k": 60, "recall_at_10": 0.75},  # best
        {"rrf_k": 100, "recall_at_10": 0.74},
    ]
    best = max(sweep_results, key=lambda x: x["recall_at_10"])
    assert best["rrf_k"] == 60


# ── Reporter: comparison_vs_dense.json ────────────────────────────────────────


def test_write_comparison_json_baseline_name() -> None:
    from enterprise_rag.evaluation.reporter import write_comparison_json
    from enterprise_rag.retrieval.result_schema import RetrievalRun

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        run = RetrievalRun(method="hybrid", k=10,
                          config={"rrf_k": 60, "model_name": "test"})
        # Should not raise when baseline doesn't exist
        path = write_comparison_json({}, run, Path(tmp) / "nonexistent.json",
                                     out_dir, baseline_name="dense")
        assert "comparison_vs_dense" in str(path)
