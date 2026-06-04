"""Pure validation helper functions for SEKD dataset integrity.

These functions operate on raw values and collections — no Pydantic dependency —
so they can be used in unit tests, CLI validation scripts, and generator code
independently of model instantiation.

Cross-model corpus integrity checks (e.g. verifying that every chunk referenced
in ground truth actually exists) belong here rather than inside individual models,
because those checks require sets of IDs from multiple artifact files.
"""

from __future__ import annotations

import re

_CHUNK_ID_RE = re.compile(r"^.+-C\d{3}$")
_QUERY_ID_RE = re.compile(r"^Q-\d{6}$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(Z|[+-]\d{2}:\d{2})?)?$")


# ── ID format checks ──────────────────────────────────────────────────────────


def is_valid_chunk_id(chunk_id: str, document_id: str) -> bool:
    """Return True when chunk_id follows ``{document_id}-C{NNN}`` format.

    Uses ``rsplit`` on the last ``-C`` occurrence so document IDs that
    themselves contain ``-C`` (e.g. ``SYS-DES-COMPLEX-C01``) are handled
    correctly.
    """
    if not _CHUNK_ID_RE.match(chunk_id):
        return False
    prefix = chunk_id.rsplit("-C", 1)[0]
    return prefix == document_id


def is_valid_query_id(query_id: str) -> bool:
    """Return True when query_id follows ``Q-{NNNNNN}`` format."""
    return bool(_QUERY_ID_RE.match(query_id))


def extract_chunk_sequence(chunk_id: str) -> int | None:
    """Return the 3-digit sequence number from a chunk ID, or None if malformed."""
    parts = chunk_id.rsplit("-C", 1)
    if len(parts) != 2 or not parts[1].isdigit() or len(parts[1]) != 3:
        return None
    return int(parts[1])


def is_valid_iso_date(value: str) -> bool:
    """Return True when value is a valid ISO 8601 date or datetime string."""
    return bool(_ISO_DATE_RE.match(value))


# ── Cross-model corpus integrity ──────────────────────────────────────────────


def find_orphan_chunk_ids(
    chunk_ids: list[str],
    available_document_ids: set[str],
) -> list[str]:
    """Return chunk_ids whose document-ID prefix is not in available_document_ids."""
    orphans: list[str] = []
    for cid in chunk_ids:
        if "-C" not in cid:
            orphans.append(cid)
            continue
        doc_prefix = cid.rsplit("-C", 1)[0]
        if doc_prefix not in available_document_ids:
            orphans.append(cid)
    return orphans


def find_duplicate_ids(ids: list[str]) -> list[str]:
    """Return a list of IDs that appear more than once."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for id_ in ids:
        if id_ in seen:
            duplicates.append(id_)
        else:
            seen.add(id_)
    return duplicates


def validate_evidence_corpus_integrity(
    ground_truth_records: list[object],
    available_chunk_ids: set[str],
) -> list[str]:
    """Check that every evidence span in ground truth references an existing chunk.

    Args:
        ground_truth_records: List of GroundTruth model instances (typed as
            ``object`` to avoid a circular import).
        available_chunk_ids: Set of chunk IDs present in the generated corpus.

    Returns:
        A list of human-readable error strings; empty when all spans are valid.
    """
    errors: list[str] = []
    for gt in ground_truth_records:
        query_id: str = getattr(gt, "query_id", "?")
        for span in getattr(gt, "evidence_spans", []):
            cid: str = getattr(span, "chunk_id", "")
            if cid not in available_chunk_ids:
                errors.append(
                    f"GroundTruth(query_id={query_id!r}): "
                    f"evidence chunk_id {cid!r} not found in corpus"
                )
    return errors


def validate_query_chunk_references(
    query_records: list[object],
    available_chunk_ids: set[str],
    available_document_ids: set[str],
) -> list[str]:
    """Check that required_chunk_ids and required_document_ids in queries exist.

    Args:
        query_records: List of Query model instances.
        available_chunk_ids: Chunk IDs from the generated corpus.
        available_document_ids: Document IDs from the generated corpus.

    Returns:
        A list of human-readable error strings; empty when all references are valid.
    """
    errors: list[str] = []
    for q in query_records:
        qid: str = getattr(q, "query_id", "?")
        for doc_id in getattr(q, "required_document_ids", []):
            if doc_id not in available_document_ids:
                errors.append(
                    f"Query(query_id={qid!r}): "
                    f"required_document_id {doc_id!r} not found in corpus"
                )
        for chunk_id in getattr(q, "required_chunk_ids", []):
            if chunk_id not in available_chunk_ids:
                errors.append(
                    f"Query(query_id={qid!r}): "
                    f"required_chunk_id {chunk_id!r} not found in corpus"
                )
    return errors


def report_query_distribution(
    query_records: list[object],
) -> dict[str, dict[str, int]]:
    """Summarise query counts by difficulty, reasoning_type, and oracle strategy.

    Returns a dict with keys ``'difficulty'``, ``'reasoning_type'``, and
    ``'oracle_strategy'``, each mapping label → count.
    """
    dist: dict[str, dict[str, int]] = {
        "difficulty": {},
        "reasoning_type": {},
        "oracle_strategy": {},
    }
    for q in query_records:
        for key in dist:
            val = str(getattr(getattr(q, key, None) or "", "value", "") or "")
            if val:
                dist[key][val] = dist[key].get(val, 0) + 1
    return dist
