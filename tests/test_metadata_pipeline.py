"""Tests for the SEKD metadata inheritance pipeline (Phase 3)."""

from __future__ import annotations

import pytest

from enterprise_rag.dataset.enums import DocumentStatus
from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator
from enterprise_rag.dataset.schemas import Chunk, EnterpriseDocument
from enterprise_rag.preprocessing.chunking import ChunkingConfig, ChunkingPipeline
from enterprise_rag.preprocessing.metadata import (
    _INHERITED_KEYS,
    build_inherited_metadata,
    validate_inherited_metadata,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus() -> list[EnterpriseDocument]:
    config = DatasetConfig(seed=42, document_count=120)
    gen = SEKDGenerator(config=config, seed=42)
    return gen.generate()


@pytest.fixture(scope="module")
def chunks(corpus: list[EnterpriseDocument]) -> list[Chunk]:
    pipeline = ChunkingPipeline(ChunkingConfig(strategy="section_aware"))
    return pipeline.chunk_corpus(corpus)


@pytest.fixture(scope="module")
def doc_index(corpus: list[EnterpriseDocument]) -> dict[str, EnterpriseDocument]:
    return {d.document_id: d for d in corpus}


# ── build_inherited_metadata unit tests ──────────────────────────────────────


def test_all_required_keys_present_in_inherited_metadata(
    corpus: list[EnterpriseDocument],
) -> None:
    for doc in corpus:
        meta = build_inherited_metadata(doc)
        missing = validate_inherited_metadata(meta)
        assert not missing, (
            f"{doc.document_id}: missing metadata keys {missing}"
        )


def test_inherited_metadata_contains_all_spec_keys() -> None:
    """_INHERITED_KEYS must cover all 9 required global fields plus title, version, source_path."""
    required_global = {
        "region",
        "department",
        "document_owner",
        "confidentiality",
        "effective_date",
        "review_cycle",
        "status",
        "authority_level",
        "tags",
        "company",
    }
    for key in required_global:
        assert key in _INHERITED_KEYS, f"Required global key '{key}' missing from _INHERITED_KEYS"


# ── Per-field inheritance correctness ────────────────────────────────────────


def test_title_inherited(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.title
        actual = c.inherited_metadata.get("title")
        assert actual == expected, (
            f"{c.chunk_id}: title '{actual}' != '{expected}'"
        )


def test_region_inherited_as_string(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.region.value
        actual = c.inherited_metadata.get("region")
        assert actual == expected, (
            f"{c.chunk_id}: region '{actual}' != '{expected}'"
        )


def test_version_inherited(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.version
        actual = c.inherited_metadata.get("version")
        assert actual == expected


def test_status_inherited_as_string(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.status.value
        actual = c.inherited_metadata.get("status")
        assert actual == expected, (
            f"{c.chunk_id}: status '{actual}' != '{expected}'"
        )


def test_document_owner_inherited(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.author_role
        actual = c.inherited_metadata.get("document_owner")
        assert actual == expected


def test_company_always_hytech(chunks: list[Chunk]) -> None:
    for c in chunks:
        assert c.inherited_metadata.get("company") == "HYTech Solutions", (
            f"{c.chunk_id}: company != 'HYTech Solutions'"
        )


def test_confidentiality_inherited_as_string(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        expected = doc_index[c.document_id].metadata.confidentiality_level.value
        actual = c.inherited_metadata.get("confidentiality")
        assert actual == expected


def test_tags_inherited_as_list(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    for c in chunks:
        actual = c.inherited_metadata.get("tags")
        assert isinstance(actual, list), (
            f"{c.chunk_id}: tags is not a list"
        )
        expected = doc_index[c.document_id].metadata.tags
        assert actual == list(expected)


# ── Deprecated document propagation ──────────────────────────────────────────


def test_deprecated_docs_propagate_status_to_chunks(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    deprecated_doc_ids = {
        doc_id
        for doc_id, doc in doc_index.items()
        if doc.metadata.status == DocumentStatus.DEPRECATED
    }
    assert deprecated_doc_ids, "No deprecated docs in corpus — cannot validate propagation"

    for c in chunks:
        if c.document_id in deprecated_doc_ids:
            assert c.inherited_metadata.get("status") == "deprecated", (
                f"{c.chunk_id}: deprecated doc status not propagated"
            )


# ── Section path inheritance ──────────────────────────────────────────────────


def test_section_path_matches_document_sections(
    chunks: list[Chunk],
    doc_index: dict[str, EnterpriseDocument],
) -> None:
    """Every chunk's section_path[0] must match a known section heading in the parent doc."""
    for c in chunks:
        if c.section_path == ["Document"]:
            continue
        doc = doc_index[c.document_id]
        known_headings = {s.heading for s in doc.sections}
        path_root = c.section_path[0]
        # path_root is either "N. Heading" or "Heading"
        # strip leading "N. " if present
        import re
        stripped = re.sub(r"^\d+\.\s*", "", path_root)
        assert stripped in known_headings or path_root in known_headings, (
            f"{c.chunk_id}: section_path root '{path_root}' not in doc sections "
            f"(known: {sorted(known_headings)})"
        )


# ── Metadata integrity ────────────────────────────────────────────────────────


def test_no_none_values_in_required_metadata(chunks: list[Chunk]) -> None:
    for c in chunks:
        for key in _INHERITED_KEYS:
            value = c.inherited_metadata.get(key)
            assert value is not None, (
                f"{c.chunk_id}: inherited_metadata['{key}'] is None"
            )


def test_effective_date_is_iso_string(chunks: list[Chunk]) -> None:
    import re
    date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for c in chunks:
        val = c.inherited_metadata.get("effective_date", "")
        assert date_re.match(str(val)), (
            f"{c.chunk_id}: effective_date '{val}' is not ISO 8601 date"
        )


def test_metadata_round_trips_through_json(chunks: list[Chunk]) -> None:
    import json
    for c in chunks[:20]:  # sample first 20 for speed
        dumped = json.dumps(c.inherited_metadata)
        loaded = json.loads(dumped)
        assert loaded == c.inherited_metadata
