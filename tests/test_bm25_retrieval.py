"""Tests for Phase 5: BM25 retrieval baseline.

Covers tokenization, index build/save/load, retrieval correctness,
metric computation, grouping, and the evaluation pipeline.
"""

from __future__ import annotations

import math
import pickle
import tempfile
from pathlib import Path

import pytest

from enterprise_rag.evaluation.retrieval_metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate,
    evaluate_query,
    evaluate_run,
    group_by,
    group_by_difficulty_factor,
    hit_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from enterprise_rag.retrieval.bm25 import BM25Config, BM25Index, build_bm25_index
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
from enterprise_rag.retrieval.tokenization import tokenize, tokenize_document, tokenize_query


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chunks(n: int = 5) -> list[dict]:
    texts = [
        "Employee annual leave policy entitlement for full-time employees is 15 days per year.",
        "Security vulnerability assessment and patch management procedures for HY-AC-01.",
        "Travel expense reimbursement lodging limit USD 200 per night for US region employees.",
        "API endpoint /api/v2/alerts for HYMonitor service accepts POST with JSON body.",
        "Project PRJ-HYD-ALPHA sprint planning meeting notes action items due 2024-03-15.",
    ]
    return [
        {
            "chunk_id": f"DOC-{i+1:03d}-C001",
            "document_id": f"DOC-{i+1:03d}",
            "category": "HR Policies" if i == 0 else "Security Policies" if i == 1 else "Travel Policies" if i == 2 else "API Documentation" if i == 3 else "Project Meeting Notes",
            "section_path": [f"Section {i+1}"],
            "text": texts[i],
            "char_start": 0,
            "char_end": len(texts[i]),
            "token_estimate": 20,
        }
        for i in range(n)
    ]


def _make_query_meta(
    query_id: str = "Q-001",
    category: str = "HR Policies",
    reasoning_type: str = "single_hop",
    difficulty: str = "easy",
    difficulty_factors: list[str] | None = None,
    answerable: bool = True,
) -> dict:
    return {
        "query_id": query_id,
        "query": f"Test query for {query_id}",
        "category": category,
        "reasoning_type": reasoning_type,
        "difficulty": difficulty,
        "retrieval_difficulty_factors": difficulty_factors or [],
        "planner_oracle_strategy": "dense",
        "answerable": answerable,
        "required_chunk_ids": [],
        "required_document_ids": [],
    }


# ── Tokenization ──────────────────────────────────────────────────────────────


def test_tokenize_preserves_control_id() -> None:
    tokens = tokenize("Control HY-AC-01 must be reviewed annually.")
    assert "hy-ac-01" in tokens


def test_tokenize_preserves_document_id() -> None:
    tokens = tokenize("See HR-POL-001 for details.")
    assert "hr-pol-001" in tokens


def test_tokenize_preserves_project_code() -> None:
    tokens = tokenize("Project PRJ-HYD-ALPHA is in planning.")
    assert "prj-hyd-alpha" in tokens


def test_tokenize_preserves_api_path() -> None:
    tokens = tokenize("Call /api/v2/alerts to get results.")
    assert any("/api/v2/alerts" in t for t in tokens)


def test_tokenize_preserves_iso_date() -> None:
    tokens = tokenize("Due date is 2024-03-15.")
    assert "2024-03-15" in tokens


def test_tokenize_document_removes_stopwords() -> None:
    tokens = tokenize_document("The employee is responsible for the report.")
    assert "the" not in tokens
    assert "is" not in tokens
    assert "for" not in tokens


def test_tokenize_query_keeps_stopwords() -> None:
    tokens = tokenize_query("What is the leave policy?")
    assert "the" in tokens or "is" in tokens  # at least one stop word kept


def test_tokenize_short_tokens_filtered() -> None:
    tokens = tokenize("a b hello world")
    # single-char tokens filtered, stop words removed
    assert "a" not in tokens
    assert "b" not in tokens
    assert "hello" in tokens


def test_tokenize_empty_string() -> None:
    assert tokenize("") == []


def test_tokenize_preserves_service_name() -> None:
    tokens = tokenize("The HYDeploy service handles deployments.")
    assert "hydeploy" in tokens


# ── BM25Index build ───────────────────────────────────────────────────────────


def test_bm25_index_builds_from_chunks() -> None:
    chunks = _make_chunks()
    with tempfile.TemporaryDirectory() as tmp:
        idx = BM25Index()
        idx.build(chunks)
        assert idx._bm25 is not None
        assert idx._meta.corpus_size == len(chunks)
        assert len(idx._meta.chunk_ids) == len(chunks)


def test_bm25_index_stores_chunk_ids_in_order() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    expected_ids = [c["chunk_id"] for c in chunks]
    assert idx._meta.chunk_ids == expected_ids


def test_bm25_index_save_and_load() -> None:
    chunks = _make_chunks()
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = Path(tmp)
        idx = BM25Index()
        idx.build(chunks)
        idx.save(idx_dir)
        assert (idx_dir / "bm25.pkl").exists()
        assert (idx_dir / "meta.json").exists()

        loaded = BM25Index.load(idx_dir)
        assert loaded._meta.corpus_size == idx._meta.corpus_size
        assert loaded._meta.chunk_ids == idx._meta.chunk_ids


def test_build_bm25_index_helper() -> None:
    chunks = _make_chunks()
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = Path(tmp) / "idx"
        idx = build_bm25_index(chunks, index_dir=idx_dir)
        assert (idx_dir / "bm25.pkl").exists()
        assert idx._meta.corpus_size == len(chunks)


def test_bm25_retrieve_raises_before_build() -> None:
    idx = BM25Index()
    with pytest.raises(RuntimeError, match="Index not built"):
        idx.retrieve("test query", "Q-001", k=5)


# ── BM25 retrieval correctness ────────────────────────────────────────────────


def test_bm25_retrieves_relevant_chunk_for_exact_match() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("annual leave policy days", "Q-001", k=5)
    assert len(results) > 0
    # The first chunk (annual leave) should be top-ranked
    assert results[0].chunk_id == "DOC-001-C001"


def test_bm25_retrieves_security_chunk_for_control_id() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("HY-AC-01 vulnerability patch", "Q-002", k=5)
    assert len(results) > 0
    assert results[0].chunk_id == "DOC-002-C001"


def test_bm25_returns_at_most_k_results() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("policy", "Q-001", k=3)
    assert len(results) <= 3


def test_bm25_results_sorted_by_rank() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("leave policy", "Q-001", k=5)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_bm25_results_have_required_fields() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("travel lodging USD", "Q-001", k=3)
    for r in results:
        assert r.query_id == "Q-001"
        assert r.chunk_id
        assert r.document_id
        assert isinstance(r.score, float)
        assert r.rank >= 1


def test_bm25_empty_query_returns_empty() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    results = idx.retrieve("", "Q-001", k=5)
    assert results == []


def test_bm25_run_covers_all_queries() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    queries = [
        {"query_id": "Q-001", "query": "annual leave"},
        {"query_id": "Q-002", "query": "security vulnerability"},
        {"query_id": "Q-003", "query": "travel lodging"},
    ]
    run = idx.run(queries, k=5)
    query_ids_in_results = {r.query_id for r in run.results}
    assert query_ids_in_results == {"Q-001", "Q-002", "Q-003"}


def test_bm25_run_returns_retrieval_run() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)
    run = idx.run([{"query_id": "Q-1", "query": "leave"}], k=3)
    assert isinstance(run, RetrievalRun)
    assert run.method == "bm25"
    assert run.k == 3


# ── Metric functions ──────────────────────────────────────────────────────────


def test_recall_at_k_perfect() -> None:
    assert recall_at_k(["A", "B", "C"], {"A", "B"}, k=2) == 1.0


def test_recall_at_k_zero() -> None:
    assert recall_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0


def test_recall_at_k_partial() -> None:
    result = recall_at_k(["A", "X", "Y"], {"A", "B"}, k=3)
    assert result == pytest.approx(0.5)


def test_recall_at_k_empty_relevant() -> None:
    assert recall_at_k(["A", "B"], set(), k=2) == 0.0


def test_precision_at_k_perfect() -> None:
    assert precision_at_k(["A", "B"], {"A", "B"}, k=2) == 1.0


def test_precision_at_k_zero() -> None:
    assert precision_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0


def test_precision_at_k_partial() -> None:
    result = precision_at_k(["A", "X", "Y"], {"A", "B"}, k=3)
    assert result == pytest.approx(1 / 3)


def test_mrr_first_position() -> None:
    assert mrr(["A", "B", "C"], {"A"}) == pytest.approx(1.0)


def test_mrr_second_position() -> None:
    assert mrr(["X", "A", "B"], {"A"}) == pytest.approx(0.5)


def test_mrr_no_relevant() -> None:
    assert mrr(["X", "Y"], {"A"}) == 0.0


def test_ndcg_at_k_perfect() -> None:
    result = ndcg_at_k(["A", "B"], {"A", "B"}, k=2)
    assert result == pytest.approx(1.0)


def test_ndcg_at_k_zero() -> None:
    assert ndcg_at_k(["X", "Y"], {"A", "B"}, k=2) == 0.0


def test_ndcg_at_k_second_position_lower_than_first() -> None:
    ndcg_first = ndcg_at_k(["A", "X"], {"A"}, k=2)
    ndcg_second = ndcg_at_k(["X", "A"], {"A"}, k=2)
    assert ndcg_first > ndcg_second


def test_hit_at_k_hit() -> None:
    assert hit_at_k(["X", "A", "Y"], {"A"}, k=3) == 1.0


def test_hit_at_k_miss() -> None:
    assert hit_at_k(["X", "Y", "Z"], {"A"}, k=3) == 0.0


def test_hit_at_k_cutoff_respected() -> None:
    assert hit_at_k(["X", "Y", "A"], {"A"}, k=2) == 0.0


# ── evaluate_query ────────────────────────────────────────────────────────────


def test_evaluate_query_returns_query_metrics() -> None:
    meta = _make_query_meta(category="HR Policies", reasoning_type="single_hop")
    qm = evaluate_query("Q-001", ["A", "B", "C"], ["A"], k=3, query_meta=meta)
    assert isinstance(qm, QueryMetrics)
    assert qm.recall == 1.0
    assert qm.hit == 1.0
    assert qm.category == "HR Policies"
    assert qm.reasoning_type == "single_hop"


def test_evaluate_query_captures_difficulty_factors() -> None:
    meta = _make_query_meta(difficulty_factors=["keyword_ambiguity", "table_dependency"])
    qm = evaluate_query("Q-001", ["X"], ["A"], k=1, query_meta=meta)
    assert qm.difficulty_factors == ["keyword_ambiguity", "table_dependency"]


def test_evaluate_query_no_meta_uses_defaults() -> None:
    qm = evaluate_query("Q-001", ["A"], ["A"], k=1)
    assert qm.category == ""
    assert qm.reasoning_type == ""


# ── aggregate + group_by ──────────────────────────────────────────────────────


def test_aggregate_empty_list() -> None:
    agg = aggregate([], k=5)
    assert agg.n_queries == 0
    assert agg.recall == 0.0


def test_aggregate_computes_means() -> None:
    qms = [
        QueryMetrics("Q-1", 5, 1.0, 0.2, 1.0, 1.0, 1.0, 1, 1, "HR", "single_hop", "easy", "", [], True),
        QueryMetrics("Q-2", 5, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, "HR", "single_hop", "easy", "", [], True),
    ]
    agg = aggregate(qms, k=5)
    assert agg.recall == pytest.approx(0.5)
    assert agg.mrr == pytest.approx(0.5)
    assert agg.n_queries == 2


def test_group_by_category() -> None:
    qms = [
        QueryMetrics("Q-1", 5, 1.0, 0.2, 1.0, 1.0, 1.0, 1, 1, "HR Policies", "single_hop", "easy", "", [], True),
        QueryMetrics("Q-2", 5, 0.5, 0.1, 0.5, 0.5, 1.0, 2, 1, "HR Policies", "single_hop", "medium", "", [], True),
        QueryMetrics("Q-3", 5, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0, "API Documentation", "single_hop", "easy", "", [], True),
    ]
    groups = group_by(qms, attr="category", k=5)
    assert "HR Policies" in groups
    assert "API Documentation" in groups
    assert groups["HR Policies"].n_queries == 2
    assert groups["API Documentation"].n_queries == 1


def test_group_by_difficulty_factor_multi_membership() -> None:
    # A query with 2 factors should appear in both groups
    qms = [
        QueryMetrics("Q-1", 5, 1.0, 0.2, 1.0, 1.0, 1.0, 1, 1, "HR", "single_hop", "easy", "", ["keyword_ambiguity", "table_dependency"], True),
    ]
    groups = group_by_difficulty_factor(qms, k=5)
    assert "keyword_ambiguity" in groups
    assert "table_dependency" in groups
    assert groups["keyword_ambiguity"].n_queries == 1
    assert groups["table_dependency"].n_queries == 1


def test_group_by_difficulty_factor_no_factors_goes_to_none() -> None:
    qms = [
        QueryMetrics("Q-1", 5, 1.0, 0.2, 1.0, 1.0, 1.0, 1, 1, "HR", "single_hop", "easy", "", [], True),
    ]
    groups = group_by_difficulty_factor(qms, k=5)
    assert "none" in groups


# ── evaluate_run ──────────────────────────────────────────────────────────────


def test_evaluate_run_basic() -> None:
    retrieval_results = [
        {"query_id": "Q-001", "chunk_id": "C-001", "rank": 1, "score": 1.5},
        {"query_id": "Q-001", "chunk_id": "C-002", "rank": 2, "score": 1.0},
    ]
    ground_truths = {"Q-001": ["C-001"]}
    queries_meta = {
        "Q-001": _make_query_meta("Q-001", answerable=True)
    }
    result_by_k = evaluate_run(retrieval_results, ground_truths, queries_meta, k_values=[1, 5])
    assert 1 in result_by_k
    assert 5 in result_by_k
    assert result_by_k[1][0].recall == 1.0  # C-001 is rank 1


def test_evaluate_run_skips_unanswerable() -> None:
    retrieval_results = [
        {"query_id": "Q-001", "chunk_id": "C-001", "rank": 1, "score": 1.0},
        {"query_id": "Q-002", "chunk_id": "C-002", "rank": 1, "score": 1.0},
    ]
    ground_truths = {"Q-001": ["C-001"], "Q-002": ["C-002"]}
    queries_meta = {
        "Q-001": _make_query_meta("Q-001", answerable=True),
        "Q-002": _make_query_meta("Q-002", answerable=False),
    }
    result_by_k = evaluate_run(
        retrieval_results, ground_truths, queries_meta, k_values=[5], answerable_only=True
    )
    assert len(result_by_k[5]) == 1
    assert result_by_k[5][0].query_id == "Q-001"


def test_evaluate_run_includes_unanswerable_when_flag_false() -> None:
    retrieval_results = [
        {"query_id": "Q-001", "chunk_id": "C-001", "rank": 1, "score": 1.0},
        {"query_id": "Q-002", "chunk_id": "C-002", "rank": 1, "score": 1.0},
    ]
    ground_truths = {"Q-001": ["C-001"], "Q-002": ["C-002"]}
    queries_meta = {
        "Q-001": _make_query_meta("Q-001", answerable=True),
        "Q-002": _make_query_meta("Q-002", answerable=False),
    }
    result_by_k = evaluate_run(
        retrieval_results, ground_truths, queries_meta, k_values=[5], answerable_only=False
    )
    assert len(result_by_k[5]) == 2


# ── result_schema ─────────────────────────────────────────────────────────────


def test_retrieved_chunk_to_dict() -> None:
    chunk = RetrievedChunk(
        query_id="Q-001",
        chunk_id="C-001",
        document_id="DOC-001",
        rank=1,
        score=2.5,
        category="HR Policies",
        section_path=["1. Purpose"],
    )
    d = chunk.to_dict()
    assert d["query_id"] == "Q-001"
    assert d["chunk_id"] == "C-001"
    assert d["score"] == 2.5
    assert "text" not in d  # to_dict excludes text field


def test_retrieval_run_results_for() -> None:
    run = RetrievalRun(method="bm25", k=5)
    run.results = [
        RetrievedChunk("Q-001", "C-001", "D-001", rank=1, score=2.0),
        RetrievedChunk("Q-001", "C-002", "D-001", rank=2, score=1.5),
        RetrievedChunk("Q-002", "C-003", "D-002", rank=1, score=3.0),
    ]
    q1_results = run.results_for("Q-001")
    assert len(q1_results) == 2
    assert q1_results[0].rank == 1


def test_retrieval_run_top_chunk_ids_for() -> None:
    run = RetrievalRun(method="bm25", k=5)
    run.results = [
        RetrievedChunk("Q-001", "C-001", "D-001", rank=1, score=2.0),
        RetrievedChunk("Q-001", "C-002", "D-001", rank=2, score=1.5),
        RetrievedChunk("Q-001", "C-003", "D-001", rank=3, score=1.0),
    ]
    ids = run.top_chunk_ids_for("Q-001", k=2)
    assert ids == ["C-001", "C-002"]


# ── Integration: build index + retrieve + evaluate ────────────────────────────


def test_end_to_end_bm25_pipeline() -> None:
    chunks = _make_chunks()
    idx = BM25Index()
    idx.build(chunks)

    queries = [
        {"query_id": "Q-001", "query": "annual leave 15 days"},
        {"query_id": "Q-002", "query": "HY-AC-01 vulnerability patch management"},
    ]
    run = idx.run(queries, k=5)

    ground_truths = {
        "Q-001": ["DOC-001-C001"],
        "Q-002": ["DOC-002-C001"],
    }
    queries_meta = {
        "Q-001": _make_query_meta("Q-001", category="HR Policies", reasoning_type="single_hop"),
        "Q-002": _make_query_meta("Q-002", category="Security Policies", reasoning_type="single_hop"),
    }

    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(
        retrieval_results, ground_truths, queries_meta, k_values=[1, 5]
    )

    agg5 = aggregate(result_by_k[5], k=5)
    assert agg5.n_queries == 2
    # BM25 should find the relevant chunks within top-5 for these exact-match queries
    assert agg5.recall > 0.0
