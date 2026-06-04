"""Evidence span building helpers for ground truth generation."""

from __future__ import annotations

from enterprise_rag.dataset.schemas import Chunk, EvidenceSpan


def find_quote_in_body(body: str, quote: str) -> tuple[int, int] | None:
    """Return (char_start, char_end) of the first occurrence of quote in body."""
    idx = body.find(quote)
    if idx == -1:
        return None
    return idx, idx + len(quote)


def find_chunk_for_offset(offset: int, chunks: list[Chunk]) -> Chunk | None:
    """Return the chunk whose [start_offset, end_offset) contains offset."""
    for c in chunks:
        if c.start_offset <= offset < c.end_offset:
            return c
    # Backward fallback: chunk with start_offset closest to but not exceeding offset
    best = None
    for c in chunks:
        if c.start_offset <= offset:
            if best is None or c.start_offset > best.start_offset:
                best = c
    if best is not None:
        return best
    # Forward fallback: offset is before all chunks — use chunk with smallest start
    if chunks:
        return min(chunks, key=lambda c: c.start_offset)
    return None


def build_evidence_spans(
    doc_id: str,
    body: str,
    chunks: list[Chunk],
    quote_specs: list[tuple[str, str]],
) -> tuple[list[EvidenceSpan], list[str], list[str]]:
    """
    Build evidence spans from quote_specs.

    Args:
        doc_id: parent document identifier
        body: full document body text
        chunks: chunks belonging to this document
        quote_specs: list of (quote_text, supports_label) pairs

    Returns:
        (evidence_spans, required_chunk_ids, required_doc_ids)
    """
    spans: list[EvidenceSpan] = []
    chunk_ids: list[str] = []

    for quote_text, supports_label in quote_specs:
        if not quote_text.strip():
            continue
        offsets = find_quote_in_body(body, quote_text)
        if offsets is None:
            continue
        q_start, q_end = offsets
        if q_end <= q_start:
            continue
        chunk = find_chunk_for_offset(q_start, chunks)
        if chunk is None:
            continue
        try:
            span = EvidenceSpan(
                document_id=doc_id,
                chunk_id=chunk.chunk_id,
                section_path=chunk.section_path,
                quote=quote_text,
                char_start=q_start,
                char_end=q_end,
                supports=supports_label,
            )
            spans.append(span)
            if chunk.chunk_id not in chunk_ids:
                chunk_ids.append(chunk.chunk_id)
        except Exception:
            continue

    required_doc_ids = [doc_id] if chunk_ids else []
    return spans, chunk_ids, required_doc_ids
