"""Tests for the SEKD synthetic document generator (Phase 2)."""

from __future__ import annotations

import time

import pytest

from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator
from enterprise_rag.dataset.schemas import EnterpriseDocument


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpus_120() -> list[EnterpriseDocument]:
    config = DatasetConfig(seed=42, document_count=120)
    gen = SEKDGenerator(config=config, seed=42)
    return gen.generate()


@pytest.fixture(scope="module")
def corpus_120_alt() -> list[EnterpriseDocument]:
    """Same size, different seed — used to test seed independence."""
    config = DatasetConfig(seed=99, document_count=120)
    gen = SEKDGenerator(config=config, seed=99)
    return gen.generate()


# ── Determinism ───────────────────────────────────────────────────────────────


def test_same_seed_same_output(corpus_120: list[EnterpriseDocument]) -> None:
    config = DatasetConfig(seed=42, document_count=120)
    gen = SEKDGenerator(config=config, seed=42)
    docs2 = gen.generate()
    assert len(corpus_120) == len(docs2)
    for a, b in zip(corpus_120, docs2):
        assert a.document_id == b.document_id
        assert a.content == b.content


def test_different_seed_different_output(
    corpus_120: list[EnterpriseDocument],
    corpus_120_alt: list[EnterpriseDocument],
) -> None:
    bodies_a = [d.content for d in corpus_120]
    bodies_b = [d.content for d in corpus_120_alt]
    assert bodies_a != bodies_b


# ── Count and categories ──────────────────────────────────────────────────────


def test_document_count(corpus_120: list[EnterpriseDocument]) -> None:
    assert len(corpus_120) == 120


def test_all_categories_present(corpus_120: list[EnterpriseDocument]) -> None:
    categories = {d.metadata.category.value for d in corpus_120}
    assert "HR Policies" in categories
    assert "Travel Policies" in categories
    assert "Security Policies" in categories
    assert "API Documentation" in categories
    assert "System Design Documents" in categories
    assert "Project Meeting Notes" in categories


def test_category_counts(corpus_120: list[EnterpriseDocument]) -> None:
    counts: dict[str, int] = {}
    for d in corpus_120:
        c = d.metadata.category.value
        counts[c] = counts.get(c, 0) + 1
    assert counts["HR Policies"] == 20
    assert counts["Travel Policies"] == 15
    assert counts["Security Policies"] == 20
    assert counts["API Documentation"] == 20
    assert counts["System Design Documents"] == 15
    assert counts["Project Meeting Notes"] == 30


# ── Schema validity ───────────────────────────────────────────────────────────


def test_schema_validity(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        assert isinstance(doc, EnterpriseDocument)
        assert doc.metadata.document_id
        assert doc.content


def test_no_duplicate_ids(corpus_120: list[EnterpriseDocument]) -> None:
    ids = [d.document_id for d in corpus_120]
    assert len(ids) == len(set(ids))


def test_category_metadata_present(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        assert doc.metadata.category_metadata is not None, (
            f"{doc.document_id} missing category_metadata"
        )


# ── HR Policy specific ────────────────────────────────────────────────────────


def test_hr_policies_have_supersedes_chain(corpus_120: list[EnterpriseDocument]) -> None:
    hr_docs = [d for d in corpus_120 if d.metadata.category.value == "HR Policies"]
    supersedes_list = [
        d.metadata.category_metadata.supersedes  # type: ignore[union-attr]
        for d in hr_docs
        if d.metadata.category_metadata is not None
    ]
    assert any(s is not None for s in supersedes_list), "At least one HR policy must supersede another"


def test_hr_policies_have_revision_history(corpus_120: list[EnterpriseDocument]) -> None:
    hr_docs = [d for d in corpus_120 if d.metadata.category.value == "HR Policies"]
    for doc in hr_docs:
        assert "Revision History" in doc.content


# ── Travel Policy specific ────────────────────────────────────────────────────


def test_travel_policies_have_tables(corpus_120: list[EnterpriseDocument]) -> None:
    travel_docs = [d for d in corpus_120 if d.metadata.category.value == "Travel Policies"]
    for doc in travel_docs:
        assert len(doc.tables) >= 1, f"{doc.document_id} has no tables"


def test_travel_policies_have_spending_limits(corpus_120: list[EnterpriseDocument]) -> None:
    travel_docs = [d for d in corpus_120 if d.metadata.category.value == "Travel Policies"]
    for doc in travel_docs:
        assert "Spending Limits" in doc.content or "spending limit" in doc.content.lower()


# ── Security Policy specific ──────────────────────────────────────────────────


def test_security_has_deprecated_docs(corpus_120: list[EnterpriseDocument]) -> None:
    from enterprise_rag.dataset.enums import DocumentStatus
    sec_docs = [d for d in corpus_120 if d.metadata.category.value == "Security Policies"]
    deprecated = [d for d in sec_docs if d.metadata.status == DocumentStatus.DEPRECATED]
    assert len(deprecated) >= 1, "At least one Security policy must be deprecated"


def test_security_policies_have_escalation_tables(corpus_120: list[EnterpriseDocument]) -> None:
    sec_docs = [d for d in corpus_120 if d.metadata.category.value == "Security Policies"]
    for doc in sec_docs:
        assert len(doc.tables) >= 1, f"{doc.document_id} has no tables"


# ── API Documentation specific ────────────────────────────────────────────────


def test_api_has_deprecated_endpoints(corpus_120: list[EnterpriseDocument]) -> None:
    api_docs = [d for d in corpus_120 if d.metadata.category.value == "API Documentation"]
    deprecated_bodies = [d for d in api_docs if "deprecated" in d.content.lower()]
    assert len(deprecated_bodies) >= 1, "At least one API doc body must mention 'deprecated'"


def test_api_docs_have_rate_limit_tables(corpus_120: list[EnterpriseDocument]) -> None:
    api_docs = [d for d in corpus_120 if d.metadata.category.value == "API Documentation"]
    for doc in api_docs:
        assert "Rate Limit" in doc.content or "rate limit" in doc.content.lower()


# ── Meeting Notes specific ────────────────────────────────────────────────────


def test_meeting_notes_reference_documents(corpus_120: list[EnterpriseDocument]) -> None:
    meeting_docs = [d for d in corpus_120 if d.metadata.category.value == "Project Meeting Notes"]
    docs_with_refs = [d for d in meeting_docs if d.references]
    pct = len(docs_with_refs) / len(meeting_docs)
    assert pct >= 0.50, f"Only {pct:.0%} of meeting notes reference documents; expected ≥50%"


def test_meeting_notes_have_action_items(corpus_120: list[EnterpriseDocument]) -> None:
    meeting_docs = [d for d in corpus_120 if d.metadata.category.value == "Project Meeting Notes"]
    for doc in meeting_docs:
        assert "Action Items" in doc.content or "action" in doc.content.lower()


# ── Section and table offsets ─────────────────────────────────────────────────


def test_section_offsets_valid(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        for sec in doc.sections:
            assert sec.char_end > sec.char_start, (
                f"{doc.document_id} section '{sec.heading}': char_end not > char_start"
            )


def test_table_offsets_valid(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        for tbl in doc.tables:
            assert tbl.char_end > tbl.char_start, (
                f"{doc.document_id} table: char_end not > char_start"
            )


def test_all_table_rows_consistent(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        for tbl in doc.tables:
            for i, row in enumerate(tbl.rows):
                assert len(row) == len(tbl.headers), (
                    f"{doc.document_id}: row {i} has {len(row)} cells, "
                    f"expected {len(tbl.headers)}"
                )


# ── Document body content ─────────────────────────────────────────────────────


def test_all_docs_have_sections(corpus_120: list[EnterpriseDocument]) -> None:
    for doc in corpus_120:
        assert len(doc.sections) >= 1, f"{doc.document_id} has no sections"


def test_section_headings_are_not_list_items(corpus_120: list[EnterpriseDocument]) -> None:
    """Section headings must be structural noun phrases, not procedural list items."""
    for doc in corpus_120:
        for sec in doc.sections:
            assert not sec.heading.endswith("."), (
                f"{doc.document_id}: section heading ends with period: '{sec.heading}'"
            )
            assert len(sec.heading.split()) <= 5, (
                f"{doc.document_id}: section heading too long "
                f"({len(sec.heading.split())} words): '{sec.heading}'"
            )


def test_section_counts_per_category(corpus_120: list[EnterpriseDocument]) -> None:
    """Structural section counts should match template design after list-item fix."""
    from collections import defaultdict
    by_cat: dict[str, list[EnterpriseDocument]] = defaultdict(list)
    for d in corpus_120:
        by_cat[d.metadata.category.value].append(d)

    for doc in by_cat["HR Policies"]:
        assert 9 <= len(doc.sections) <= 11, (
            f"{doc.document_id}: unexpected HR section count {len(doc.sections)}"
        )
    for doc in by_cat["Travel Policies"]:
        assert 9 <= len(doc.sections) <= 11, (
            f"{doc.document_id}: unexpected Travel section count {len(doc.sections)}"
        )
    for doc in by_cat["Security Policies"]:
        assert 9 <= len(doc.sections) <= 11, (
            f"{doc.document_id}: unexpected Security section count {len(doc.sections)}"
        )
    for doc in by_cat["API Documentation"]:
        assert 7 <= len(doc.sections) <= 9, (
            f"{doc.document_id}: unexpected API section count {len(doc.sections)}"
        )
    for doc in by_cat["System Design Documents"]:
        assert 9 <= len(doc.sections) <= 12, (
            f"{doc.document_id}: unexpected SysDesign section count {len(doc.sections)}"
        )
    for doc in by_cat["Project Meeting Notes"]:
        assert 7 <= len(doc.sections) <= 9, (
            f"{doc.document_id}: unexpected Meeting section count {len(doc.sections)}"
        )


def test_alias_fields_serialise_correctly(corpus_120: list[EnterpriseDocument]) -> None:
    doc = corpus_120[0]
    dumped = doc.model_dump(by_alias=True, mode="json")
    assert "body" in dumped
    assert "content" not in dumped
    assert "document_owner" in dumped["metadata"]
    assert "author_role" not in dumped["metadata"]
    assert "confidentiality" in dumped["metadata"]
    assert "confidentiality_level" not in dumped["metadata"]


# ── Performance ───────────────────────────────────────────────────────────────


def test_small_corpus_generates_fast() -> None:
    start = time.monotonic()
    config = DatasetConfig(seed=42, document_count=20)
    gen = SEKDGenerator(config=config, seed=42)
    docs = gen.generate()
    elapsed = time.monotonic() - start
    assert len(docs) > 0
    assert elapsed < 5.0, f"20-doc generation took {elapsed:.2f}s, expected < 5s"
