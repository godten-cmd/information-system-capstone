"""Tests for the SEKD chunking pipeline (Phase 3)."""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from enterprise_rag.dataset.enums import DocumentCategory
from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator
from enterprise_rag.dataset.schemas import Chunk, EnterpriseDocument
from enterprise_rag.preprocessing.chunking import (
    ChunkingConfig,
    ChunkingPipeline,
    build_quality_report,
    compute_statistics,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_CHUNK_ID_RE = re.compile(r"^.+-C\d{3}$")


@pytest.fixture(scope="module")
def corpus() -> list[EnterpriseDocument]:
    config = DatasetConfig(seed=42, document_count=120)
    gen = SEKDGenerator(config=config, seed=42)
    return gen.generate()


@pytest.fixture(scope="module")
def doc_index(corpus: list[EnterpriseDocument]) -> dict[str, EnterpriseDocument]:
    return {d.document_id: d for d in corpus}


@pytest.fixture(scope="module")
def section_aware_chunks(corpus: list[EnterpriseDocument]) -> list[Chunk]:
    pipeline = ChunkingPipeline(ChunkingConfig(strategy="section_aware"))
    return pipeline.chunk_corpus(corpus)


@pytest.fixture(scope="module")
def fixed_chunks(corpus: list[EnterpriseDocument]) -> list[Chunk]:
    pipeline = ChunkingPipeline(ChunkingConfig(strategy="fixed_size"))
    return pipeline.chunk_corpus(corpus)


@pytest.fixture(scope="module")
def recursive_chunks(corpus: list[EnterpriseDocument]) -> list[Chunk]:
    pipeline = ChunkingPipeline(ChunkingConfig(strategy="recursive"))
    return pipeline.chunk_corpus(corpus)


# ── Chunk ID format ───────────────────────────────────────────────────────────


def test_chunk_id_format(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        assert _CHUNK_ID_RE.match(c.chunk_id), (
            f"chunk_id '{c.chunk_id}' does not match {{doc_id}}-C{{NNN}}"
        )


def test_chunk_ids_belong_to_documents(
    section_aware_chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in section_aware_chunks:
        assert c.document_id in doc_index, (
            f"chunk {c.chunk_id} references unknown document {c.document_id}"
        )


def test_chunk_id_prefix_matches_document(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        prefix = c.chunk_id.rsplit("-C", 1)[0]
        assert prefix == c.document_id, (
            f"chunk_id prefix '{prefix}' != document_id '{c.document_id}'"
        )


def test_chunk_ids_sequential_per_document(section_aware_chunks: list[Chunk]) -> None:
    """Within each document chunks must be numbered 001, 002, 003 … without gaps."""
    by_doc: dict[str, list[int]] = defaultdict(list)
    for c in section_aware_chunks:
        seq = int(c.chunk_id.rsplit("-C", 1)[1])
        by_doc[c.document_id].append(seq)
    for doc_id, seqs in by_doc.items():
        seqs.sort()
        expected = list(range(1, len(seqs) + 1))
        assert seqs == expected, (
            f"{doc_id}: chunk sequence {seqs} != expected {expected}"
        )


# ── Coverage ──────────────────────────────────────────────────────────────────


def test_all_docs_have_chunks(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    chunked_ids = {c.document_id for c in section_aware_chunks}
    for doc in corpus:
        assert doc.document_id in chunked_ids, (
            f"{doc.document_id} has no chunks"
        )


def test_no_orphan_chunks(
    section_aware_chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in section_aware_chunks:
        assert c.document_id in doc_index


def test_total_chunks_in_spec_range(section_aware_chunks: list[Chunk]) -> None:
    """DATASET_SPEC recommends 800–2,000 chunks for a 120-doc corpus."""
    assert 800 <= len(section_aware_chunks) <= 2000, (
        f"Total chunks {len(section_aware_chunks)} outside spec range 800–2000"
    )


def test_every_document_category_has_chunks(section_aware_chunks: list[Chunk]) -> None:
    cats = {c.category.value for c in section_aware_chunks}
    for expected in [
        "HR Policies",
        "Travel Policies",
        "Security Policies",
        "API Documentation",
        "System Design Documents",
        "Project Meeting Notes",
    ]:
        assert expected in cats


# ── Offsets ───────────────────────────────────────────────────────────────────


def test_char_offsets_valid(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        assert c.end_offset > c.start_offset, (
            f"{c.chunk_id}: end_offset {c.end_offset} <= start_offset {c.start_offset}"
        )


def test_char_offsets_within_document_body(
    section_aware_chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in section_aware_chunks:
        body_len = len(doc_index[c.document_id].content)
        assert c.start_offset >= 0
        assert c.end_offset <= body_len + 1, (
            f"{c.chunk_id}: end_offset {c.end_offset} > body length {body_len}"
        )


# ── Content integrity ─────────────────────────────────────────────────────────


def test_no_empty_chunk_text(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        assert c.text.strip(), f"{c.chunk_id} has empty text"


def test_token_estimate_positive(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        assert c.token_estimate >= 1, f"{c.chunk_id} has token_estimate < 1"


def test_section_path_not_empty(section_aware_chunks: list[Chunk]) -> None:
    for c in section_aware_chunks:
        assert len(c.section_path) >= 1, f"{c.chunk_id} has empty section_path"
        assert all(isinstance(p, str) and p for p in c.section_path), (
            f"{c.chunk_id} has non-string or empty path element"
        )


# ── Section-aware specific ────────────────────────────────────────────────────


def test_section_aware_covers_all_sections(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    """Every section heading in a document must appear in at least one chunk's section_path."""
    chunk_paths_by_doc: dict[str, set[str]] = defaultdict(set)
    for c in section_aware_chunks:
        for p in c.section_path:
            chunk_paths_by_doc[c.document_id].add(p)

    for doc in corpus:
        for sec in doc.sections:
            heading = sec.heading
            covered = any(
                heading in p
                for p in chunk_paths_by_doc[doc.document_id]
            )
            assert covered, (
                f"{doc.document_id}: section '{heading}' not covered by any chunk"
            )


def test_api_endpoint_blocks_not_split(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    """No API chunk should start with a partial endpoint block (orphaned parameters)."""
    api_chunks = [
        c for c in section_aware_chunks
        if c.category == DocumentCategory.API_DOCUMENTATION
    ]
    for c in api_chunks:
        # A chunk that starts mid-endpoint would begin with "Parameters:" or "- `"
        # without a preceding "### METHOD /path" in the same chunk
        if c.text.strip().startswith("Parameters:"):
            # Must also contain the endpoint header
            assert "###" in c.text, (
                f"{c.chunk_id}: starts with 'Parameters:' but has no endpoint header"
            )


def test_table_bearing_sections_produce_table_chunks(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    """Documents with tables must have at least one chunk with has_table=True."""
    chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for c in section_aware_chunks:
        chunks_by_doc[c.document_id].append(c)

    for doc in corpus:
        if doc.tables:
            has_table_chunk = any(
                c.inherited_metadata.get("has_table", False)
                for c in chunks_by_doc[doc.document_id]
            )
            assert has_table_chunk, (
                f"{doc.document_id} has tables but no chunk with has_table=True"
            )


# ── Fixed-size strategy ───────────────────────────────────────────────────────


def test_fixed_size_chunks_within_max(fixed_chunks: list[Chunk]) -> None:
    """Fixed-size chunks should not dramatically exceed max_tokens (allow 2× buffer)."""
    for c in fixed_chunks:
        assert c.token_estimate <= 800, (
            f"{c.chunk_id}: token_estimate {c.token_estimate} > 800 (2× limit)"
        )


def test_fixed_size_all_docs_covered(
    corpus: list[EnterpriseDocument],
    fixed_chunks: list[Chunk],
) -> None:
    chunked_ids = {c.document_id for c in fixed_chunks}
    for doc in corpus:
        assert doc.document_id in chunked_ids


# ── Recursive strategy ────────────────────────────────────────────────────────


def test_recursive_all_docs_covered(
    corpus: list[EnterpriseDocument],
    recursive_chunks: list[Chunk],
) -> None:
    chunked_ids = {c.document_id for c in recursive_chunks}
    for doc in corpus:
        assert doc.document_id in chunked_ids


# ── Schema validation ─────────────────────────────────────────────────────────


def test_all_chunks_validate_as_chunk_schema(section_aware_chunks: list[Chunk]) -> None:
    """All chunks should round-trip through Chunk validation."""
    for c in section_aware_chunks:
        dumped = c.model_dump(by_alias=True, mode="json")
        restored = Chunk.model_validate(dumped)
        assert restored.chunk_id == c.chunk_id


# ── Statistics ────────────────────────────────────────────────────────────────


def test_compute_statistics(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    stats = compute_statistics(section_aware_chunks, corpus)
    assert stats.total_chunks == len(section_aware_chunks)
    assert stats.total_docs == len(corpus)
    assert stats.docs_with_chunks == 120
    assert stats.avg_tokens > 0
    assert stats.p50_tokens >= stats.p25_tokens
    assert stats.p95_tokens >= stats.p75_tokens


def test_quality_report_no_metadata_errors(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    report = build_quality_report(section_aware_chunks, corpus)
    assert report.metadata_invalid_count == 0, (
        f"Metadata errors found: {report.validation_errors}"
    )


def test_quality_report_section_path_coverage(
    corpus: list[EnterpriseDocument],
    section_aware_chunks: list[Chunk],
) -> None:
    report = build_quality_report(section_aware_chunks, corpus)
    assert report.section_path_coverage >= 0.95, (
        f"Section path coverage {report.section_path_coverage:.1%} < 95%"
    )
