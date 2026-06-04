"""Tests for Phase 4: Query and Ground Truth generation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from enterprise_rag.dataset.enums import (
    DocumentCategory,
    OracleStrategy,
    QueryDifficulty,
    ReasoningType,
    RetrievalDifficultyFactor,
)
from enterprise_rag.dataset.generate_queries import (
    QueryConfig,
    SEKDQueryGenerator,
    build_generation_report,
)
from enterprise_rag.dataset.ground_truth import (
    build_evidence_spans,
    find_chunk_for_offset,
    find_quote_in_body,
)
from enterprise_rag.dataset.schemas import (
    Chunk,
    EnterpriseDocument,
    GroundTruth,
    Query,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_DOCS_PATH = Path("data/sekd/raw/documents.jsonl")
_CHUNKS_PATH = Path("data/sekd/processed/chunks.jsonl")


@pytest.fixture(scope="module")
def corpus_docs() -> list[EnterpriseDocument]:
    import jsonlines

    if not _DOCS_PATH.exists():
        pytest.skip(f"documents.jsonl not found at {_DOCS_PATH}")
    with jsonlines.open(_DOCS_PATH) as r:
        return [EnterpriseDocument.model_validate(d) for d in r]


@pytest.fixture(scope="module")
def corpus_chunks() -> list[Chunk]:
    import jsonlines

    if not _CHUNKS_PATH.exists():
        pytest.skip(f"chunks.jsonl not found at {_CHUNKS_PATH}")
    with jsonlines.open(_CHUNKS_PATH) as r:
        return [Chunk.model_validate(c) for c in r]


@pytest.fixture(scope="module")
def generated_output(
    corpus_docs: list[EnterpriseDocument],
    corpus_chunks: list[Chunk],
) -> tuple[list[Query], list[GroundTruth]]:
    gen = SEKDQueryGenerator(QueryConfig(seed=42))
    return gen.generate(corpus_docs, corpus_chunks)


@pytest.fixture(scope="module")
def queries(generated_output: tuple) -> list[Query]:
    return generated_output[0]


@pytest.fixture(scope="module")
def ground_truths(generated_output: tuple) -> list[GroundTruth]:
    return generated_output[1]


# ── Determinism ───────────────────────────────────────────────────────────────


def test_same_seed_same_output(corpus_docs, corpus_chunks):
    gen1 = SEKDQueryGenerator(QueryConfig(seed=42))
    gen2 = SEKDQueryGenerator(QueryConfig(seed=42))
    q1, gt1 = gen1.generate(corpus_docs, corpus_chunks)
    q2, gt2 = gen2.generate(corpus_docs, corpus_chunks)
    assert len(q1) == len(q2)
    for a, b in zip(q1, q2):
        assert a.query_id == b.query_id
        assert a.query_text == b.query_text


def test_different_seed_different_output(corpus_docs, corpus_chunks):
    gen1 = SEKDQueryGenerator(QueryConfig(seed=42))
    gen2 = SEKDQueryGenerator(QueryConfig(seed=99))
    q1, _ = gen1.generate(corpus_docs, corpus_chunks)
    q2, _ = gen2.generate(corpus_docs, corpus_chunks)
    texts1 = [q.query_text for q in q1]
    texts2 = [q.query_text for q in q2]
    assert texts1 != texts2


# ── Count and distribution constraints ───────────────────────────────────────


def test_minimum_total_queries(queries):
    assert len(queries) >= 400, f"Expected ≥400 queries, got {len(queries)}"


def test_minimum_unanswerable(queries):
    unanswerable = [q for q in queries if not q.answerable]
    assert len(unanswerable) >= 25, f"Expected ≥25 unanswerable, got {len(unanswerable)}"


def test_minimum_multi_hop(queries):
    multi_hop = [q for q in queries if q.reasoning_type == ReasoningType.MULTI_HOP]
    assert len(multi_hop) >= 50, f"Expected ≥50 multi_hop, got {len(multi_hop)}"


def test_minimum_per_category(queries):
    per_cat = Counter(q.category for q in queries)
    for cat in DocumentCategory:
        count = per_cat.get(cat, 0)
        assert count >= 40, f"Category {cat.value} has only {count} queries (min 40)"


def test_no_duplicate_query_ids(queries):
    ids = [q.query_id for q in queries]
    assert len(ids) == len(set(ids)), "Duplicate query_ids found"


def test_query_gt_count_matches(queries, ground_truths):
    assert len(queries) == len(ground_truths), "Query and GroundTruth counts differ"


# ── Schema validity ───────────────────────────────────────────────────────────


def test_all_queries_valid_pydantic(queries):
    for q in queries:
        assert isinstance(q, Query)
        assert q.query_id.startswith("Q-")
        assert len(q.query_text) > 5


def test_all_ground_truths_valid_pydantic(ground_truths):
    for gt in ground_truths:
        assert isinstance(gt, GroundTruth)
        assert gt.query_id.startswith("Q-")


def test_answerable_queries_have_evidence(queries, ground_truths):
    gt_by_id = {gt.query_id: gt for gt in ground_truths}
    for q in queries:
        if q.answerable:
            gt = gt_by_id[q.query_id]
            assert len(gt.evidence_spans) > 0, f"{q.query_id}: answerable but no evidence spans"
            assert len(gt.required_document_ids) > 0, f"{q.query_id}: answerable but no required_doc_ids"
            assert len(gt.required_chunk_ids) > 0, f"{q.query_id}: answerable but no required_chunk_ids"


def test_unanswerable_queries_have_no_evidence(queries, ground_truths):
    gt_by_id = {gt.query_id: gt for gt in ground_truths}
    for q in queries:
        if not q.answerable:
            gt = gt_by_id[q.query_id]
            assert len(gt.evidence_spans) == 0, f"{q.query_id}: unanswerable but has evidence"
            assert len(gt.required_document_ids) == 0, f"{q.query_id}: unanswerable but has doc_ids"


# ── Multi-document queries ────────────────────────────────────────────────────


def test_multi_hop_queries_span_multiple_docs(queries):
    multi_hop = [q for q in queries if q.reasoning_type == ReasoningType.MULTI_HOP]
    for q in multi_hop:
        assert len(q.required_document_ids) >= 2, (
            f"{q.query_id}: multi_hop query has only {len(q.required_document_ids)} doc(s)"
        )


def test_multi_doc_queries_have_dependency_factor(queries):
    for q in queries:
        if len(q.required_document_ids) >= 2:
            assert RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY in q.retrieval_difficulty_factors, (
                f"{q.query_id}: spans {len(q.required_document_ids)} docs but missing MULTI_DOCUMENT_DEPENDENCY"
            )


# ── Category-specific constraints ─────────────────────────────────────────────


def test_security_queries_include_exception_reasoning(queries):
    sec = [q for q in queries if q.category == DocumentCategory.SECURITY_POLICY]
    exception_queries = [q for q in sec if q.reasoning_type == ReasoningType.EXCEPTION]
    assert len(exception_queries) > 0, "No exception-reasoning queries in Security category"


def test_travel_queries_include_table_dependency(queries):
    travel = [q for q in queries if q.category == DocumentCategory.TRAVEL_POLICY]
    table_q = [q for q in travel if RetrievalDifficultyFactor.TABLE_DEPENDENCY in q.retrieval_difficulty_factors]
    assert len(table_q) > 0, "No table-dependency queries in Travel category"


def test_api_queries_include_version_conflict(queries):
    api = [q for q in queries if q.category == DocumentCategory.API_DOCUMENTATION]
    version_q = [q for q in api if RetrievalDifficultyFactor.DOCUMENT_VERSION_CONFLICT in q.retrieval_difficulty_factors]
    assert len(version_q) > 0, "No version-conflict queries in API Documentation category"


def test_hr_comparison_queries_exist(queries):
    hr_comp = [
        q for q in queries
        if q.category == DocumentCategory.HR_POLICY
        and q.reasoning_type == ReasoningType.COMPARISON
    ]
    assert len(hr_comp) > 0, "No cross-region HR comparison queries"


def test_meeting_temporal_queries_exist(queries):
    temporal = [
        q for q in queries
        if q.category == DocumentCategory.MEETING_NOTES
        and q.reasoning_type == ReasoningType.TEMPORAL
    ]
    assert len(temporal) >= 5, f"Expected ≥5 temporal meeting queries, got {len(temporal)}"


# ── Evidence span integrity ───────────────────────────────────────────────────


def test_required_chunk_ids_covered_by_evidence(queries, ground_truths):
    """The Pydantic validator enforces: required_chunk_ids ⊆ evidence_spans chunk_ids."""
    gt_by_id = {gt.query_id: gt for gt in ground_truths}
    for q in queries:
        if not q.answerable:
            continue
        gt = gt_by_id[q.query_id]
        evidence_chunk_ids = {span.chunk_id for span in gt.evidence_spans}
        for chunk_id in gt.required_chunk_ids:
            assert chunk_id in evidence_chunk_ids, (
                f"{q.query_id}: required chunk {chunk_id} has no evidence span"
            )


def test_required_doc_ids_covered_by_evidence(queries, ground_truths):
    """The Pydantic validator enforces: required_document_ids ⊆ evidence_spans doc_ids."""
    gt_by_id = {gt.query_id: gt for gt in ground_truths}
    for q in queries:
        if not q.answerable:
            continue
        gt = gt_by_id[q.query_id]
        evidence_doc_ids = {span.document_id for span in gt.evidence_spans}
        for doc_id in gt.required_document_ids:
            assert doc_id in evidence_doc_ids, (
                f"{q.query_id}: required doc {doc_id} has no evidence span"
            )


# ── Ground truth helpers ──────────────────────────────────────────────────────


def test_find_quote_in_body_exact_match():
    body = "This is a test sentence. The entitlement is 15 days per year."
    quote = "The entitlement is 15 days per year."
    result = find_quote_in_body(body, quote)
    assert result is not None
    start, end = result
    assert body[start:end] == quote


def test_find_quote_in_body_missing():
    body = "This policy covers sick leave."
    assert find_quote_in_body(body, "annual leave") is None


def _make_chunk(chunk_id: str, doc_id: str, text: str, start: int, end: int, section: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=doc_id,
        category=DocumentCategory.HR_POLICY,
        text=text,
        start_offset=start,
        end_offset=end,
        section_path=[section],
        token_estimate=len(text.split()),
    )


def test_find_chunk_for_offset_exact():
    chunks = [
        _make_chunk("D1-C001", "D1", "hello world", 0, 11, "1. Title"),
        _make_chunk("D1-C002", "D1", "foo bar", 11, 18, "1. Title"),
    ]
    assert find_chunk_for_offset(0, chunks).chunk_id == "D1-C001"
    assert find_chunk_for_offset(10, chunks).chunk_id == "D1-C001"
    assert find_chunk_for_offset(11, chunks).chunk_id == "D1-C002"


def test_find_chunk_for_offset_before_all_chunks():
    chunks = [_make_chunk("D1-C001", "D1", "section text", 100, 200, "1. Title")]
    # Offset 50 is before the first chunk — should return first chunk
    result = find_chunk_for_offset(50, chunks)
    assert result is not None
    assert result.chunk_id == "D1-C001"


def test_find_chunk_for_offset_empty():
    assert find_chunk_for_offset(0, []) is None


def test_build_evidence_spans_basic():
    body = "Policy ID: POL-001\nEntitlement is 15 days per year."
    chunks = [_make_chunk("D1-C001", "D1", body, 0, len(body), "1. Policy Statement")]
    quote_specs = [("Entitlement is 15 days per year.", "entitlement value")]
    spans, chunk_ids, doc_ids = build_evidence_spans("D1", body, chunks, quote_specs)
    assert len(spans) == 1
    assert spans[0].chunk_id == "D1-C001"
    assert spans[0].document_id == "D1"
    assert "D1-C001" in chunk_ids
    assert "D1" in doc_ids


def test_build_evidence_spans_missing_quote():
    body = "Some policy text."
    chunks = [_make_chunk("D1-C001", "D1", body, 0, len(body), "1. Title")]
    spans, chunk_ids, doc_ids = build_evidence_spans("D1", body, chunks, [("not in body", "label")])
    assert len(spans) == 0
    assert len(chunk_ids) == 0
    assert len(doc_ids) == 0


# ── Generation report ─────────────────────────────────────────────────────────


def test_generation_report(queries):
    report = build_generation_report(queries, elapsed=0.5)
    assert report.total_queries == len(queries)
    assert report.answerable_count + report.unanswerable_count == report.total_queries
    assert report.duration_seconds == 0.5
    assert len(report.per_category) > 0
    assert len(report.per_reasoning_type) > 0


# ── Fast generation ───────────────────────────────────────────────────────────


def test_small_corpus_generates_fast(corpus_docs, corpus_chunks):
    import time

    small_docs = corpus_docs[:20]
    small_chunks = [c for c in corpus_chunks if c.document_id in {d.document_id for d in small_docs}]
    gen = SEKDQueryGenerator(QueryConfig(seed=42))
    t0 = time.monotonic()
    queries, _ = gen.generate(small_docs, small_chunks)
    elapsed = time.monotonic() - t0
    assert elapsed < 10, f"Generation too slow: {elapsed:.1f}s"
    assert len(queries) > 0
