"""Chunking pipeline for SEKD enterprise documents.

Three strategies
----------------
section_aware  — one chunk per structural section; long sections split by
                 paragraph or, for API docs, by endpoint block.  This is the
                 canonical strategy whose output feeds Phase 4 and beyond.

fixed_size     — sliding character window over the full document body.
                 section_path is assigned by the section containing the
                 window's midpoint.

recursive      — hierarchical paragraph → sentence → character split applied
                 section by section; produces finer-grained chunks than
                 section_aware while still respecting section boundaries.

All three strategies emit ``Chunk`` objects validated by the Pydantic schema.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from enterprise_rag.dataset.enums import DocumentCategory
from enterprise_rag.dataset.schemas import (
    Chunk,
    DocumentSection,
    EnterpriseDocument,
)

from .metadata import build_inherited_metadata

# ── Constants ─────────────────────────────────────────────────────────────────

_CHARS_PER_TOKEN = 4  # rough heuristic for English enterprise text
_ENDPOINT_BLOCK_RE = re.compile(r"\n(?=### )")


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class ChunkingConfig:
    """Configuration for the chunking pipeline."""

    strategy: Literal["section_aware", "fixed_size", "recursive"] = "section_aware"
    max_tokens: int = 400
    overlap_tokens: int = 50
    min_tokens: int = 20


# ── Low-level splitting helpers ───────────────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _max_chars(config: ChunkingConfig) -> int:
    return config.max_tokens * _CHARS_PER_TOKEN


def _split_with_offsets(
    text: str,
    base: int,
    sep: str,
) -> list[tuple[str, int, int]]:
    """Split *text* at *sep* and return (piece, abs_start, abs_end) for non-empty pieces."""
    result: list[tuple[str, int, int]] = []
    cursor = 0
    while True:
        idx = text.find(sep, cursor)
        if idx == -1:
            piece = text[cursor:]
            if piece.strip():
                result.append((piece, base + cursor, base + cursor + len(piece)))
            break
        piece = text[cursor:idx]
        if piece.strip():
            result.append((piece, base + cursor, base + cursor + len(piece)))
        cursor = idx + len(sep)
    return result


def _group_paragraphs(
    paragraphs: list[tuple[str, int, int]],
    max_chars: int,
    min_chars: int,
) -> list[tuple[str, int, int]]:
    """Greedily group (text, start, end) paragraph tuples into max_chars windows.

    Tiny last paragraphs (< min_chars) are merged into the previous window.
    """
    windows: list[tuple[str, int, int]] = []
    bucket: list[tuple[str, int, int]] = []
    bucket_chars = 0

    for para in paragraphs:
        para_chars = len(para[0])
        if bucket and bucket_chars + para_chars > max_chars:
            combined = "".join(p[0] for p in bucket)
            windows.append((combined, bucket[0][1], bucket[-1][2]))
            bucket = []
            bucket_chars = 0
        bucket.append(para)
        bucket_chars += para_chars

    if bucket:
        combined = "".join(p[0] for p in bucket)
        windows.append((combined, bucket[0][1], bucket[-1][2]))

    # Merge tiny tail window into predecessor
    if len(windows) >= 2:
        last_text, last_start, last_end = windows[-1]
        if len(last_text) < min_chars:
            prev_text, prev_start, prev_end = windows[-2]
            windows[-2] = (prev_text + last_text, prev_start, last_end)
            windows.pop()

    return windows


def _split_endpoint_blocks(
    content: str,
    base: int,
) -> list[tuple[str, int, int]]:
    """Split an API Endpoints section at ``\\n### `` boundaries.

    The first block retains the section heading.  Subsequent blocks start
    with ``### ``.
    """
    # Positions in content where a new block begins (right after a \\n before ###)
    block_starts = [0]
    for m in _ENDPOINT_BLOCK_RE.finditer(content):
        block_starts.append(m.end())
    block_starts.append(len(content))

    result: list[tuple[str, int, int]] = []
    for i in range(len(block_starts) - 1):
        start = block_starts[i]
        end = block_starts[i + 1]
        piece = content[start:end]
        if piece.strip():
            result.append((piece, base + start, base + end))
    return result


def _recursive_text_split(
    text: str,
    base: int,
    max_chars: int,
    min_chars: int,
    depth: int = 0,
) -> list[tuple[str, int, int]]:
    """Recursively split *text* using coarser-to-finer separators.

    Hierarchy: paragraph (\\n\\n) → sentence (. ) → hard character limit.
    """
    if len(text) <= max_chars:
        return [(text, base, base + len(text))] if text.strip() else []

    separators = ["\n\n", ". ", "\n", " "]
    if depth < len(separators):
        sep = separators[depth]
        parts = _split_with_offsets(text, base, sep)
        if len(parts) <= 1:
            return _recursive_text_split(text, base, max_chars, min_chars, depth + 1)

        result: list[tuple[str, int, int]] = []
        for piece, ps, pe in parts:
            if len(piece) <= max_chars:
                if piece.strip():
                    result.append((piece, ps, pe))
            else:
                result.extend(
                    _recursive_text_split(piece, ps, max_chars, min_chars, depth + 1)
                )
        return result

    # Hard character split as last resort
    result = []
    cursor = 0
    while cursor < len(text):
        end = min(cursor + max_chars, len(text))
        piece = text[cursor:end]
        if piece.strip():
            result.append((piece, base + cursor, base + end))
        cursor = end
    return result


def _section_path_at(offset: int, sections: list[DocumentSection]) -> list[str]:
    """Return the path of the first section whose range contains *offset*."""
    for s in sections:
        if s.char_start <= offset < s.char_end:
            return s.path
    return sections[0].path if sections else ["Document"]


# ── Section-aware strategy ────────────────────────────────────────────────────


def _chunks_from_section(
    section: DocumentSection,
    doc: EnterpriseDocument,
    counter_start: int,
    base_meta: dict,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Produce one or more chunks from a single *section*."""
    max_chars = _max_chars(config)
    min_chars = config.min_tokens * _CHARS_PER_TOKEN
    content = section.content
    abs_base = section.char_start

    # Decide split method
    is_api_endpoints = (
        doc.category == DocumentCategory.API_DOCUMENTATION
        and section.heading == "Endpoints"
    )

    if is_api_endpoints:
        pieces = _split_endpoint_blocks(content, abs_base)
    elif len(content) > max_chars:
        raw_paragraphs = _split_with_offsets(content, abs_base, "\n\n")
        if not raw_paragraphs:
            raw_paragraphs = [(content, abs_base, abs_base + len(content))]
        pieces = _group_paragraphs(raw_paragraphs, max_chars, min_chars)
    else:
        pieces = [(content, abs_base, abs_base + len(content))]

    chunks: list[Chunk] = []
    for i, (text, cs, ce) in enumerate(pieces):
        if not text.strip():
            continue
        # Ensure offsets are valid
        if ce <= cs:
            ce = cs + 1

        sub_path = list(section.path)
        if is_api_endpoints and i > 0:
            # Label sub-path with the endpoint method+path if detectable
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            if first_line.startswith("### "):
                sub_path = list(section.path) + [first_line[4:].strip()]

        chunk_meta = dict(base_meta)
        chunk_meta["section_heading"] = section.heading
        chunk_meta["chunk_strategy"] = "section_aware"
        chunk_meta["has_table"] = "|" in text and "| --- |" in text

        seq = counter_start + len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.document_id}-C{seq:03d}",
                document_id=doc.document_id,
                category=doc.category,
                section_path=sub_path,
                text=text,
                inherited_metadata=chunk_meta,
                char_start=cs,
                char_end=ce,
                token_estimate=_estimate_tokens(text),
            )
        )
    return chunks


def _section_aware_chunks(
    doc: EnterpriseDocument,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Produce chunks using section-aware strategy."""
    if not doc.sections:
        # Fallback: whole body as one chunk
        base_meta = build_inherited_metadata(doc)
        base_meta["chunk_strategy"] = "section_aware"
        base_meta["has_table"] = bool(doc.tables)
        body = doc.content
        return [
            Chunk(
                chunk_id=f"{doc.document_id}-C001",
                document_id=doc.document_id,
                category=doc.category,
                section_path=["Document"],
                text=body,
                inherited_metadata=base_meta,
                char_start=0,
                char_end=len(body),
                token_estimate=_estimate_tokens(body),
            )
        ]

    base_meta = build_inherited_metadata(doc)
    chunks: list[Chunk] = []
    for section in doc.sections:
        new_chunks = _chunks_from_section(
            section=section,
            doc=doc,
            counter_start=len(chunks) + 1,
            base_meta=base_meta,
            config=config,
        )
        chunks.extend(new_chunks)
    return chunks


# ── Fixed-size strategy ───────────────────────────────────────────────────────


def _fixed_size_chunks(
    doc: EnterpriseDocument,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Produce chunks using a sliding character window over the full body."""
    body = doc.content
    window = config.max_tokens * _CHARS_PER_TOKEN
    overlap = config.overlap_tokens * _CHARS_PER_TOKEN
    min_chars = config.min_tokens * _CHARS_PER_TOKEN

    base_meta = build_inherited_metadata(doc)

    # Find paragraph-aligned split points to avoid mid-paragraph cuts
    paragraph_ends: list[int] = []
    pos = 0
    while True:
        idx = body.find("\n\n", pos)
        if idx == -1:
            paragraph_ends.append(len(body))
            break
        paragraph_ends.append(idx + 2)
        pos = idx + 2

    chunks: list[Chunk] = []
    cursor = 0

    while cursor < len(body):
        target = cursor + window
        # Snap to nearest paragraph end <= target (or hard cut at target)
        end = target
        for pe in paragraph_ends:
            if cursor < pe <= target:
                end = pe
        end = min(end, len(body))
        if end <= cursor:
            end = min(cursor + window, len(body))

        text = body[cursor:end]
        if not text.strip():
            cursor = end
            continue

        if len(text) >= min_chars or not chunks:
            mid = cursor + len(text) // 2
            s_path = _section_path_at(mid, doc.sections)

            meta = dict(base_meta)
            meta["chunk_strategy"] = "fixed_size"
            meta["has_table"] = "|" in text and "| --- |" in text
            if doc.sections:
                for s in doc.sections:
                    if s.char_start <= mid < s.char_end:
                        meta["section_heading"] = s.heading
                        break

            seq = len(chunks) + 1
            cs, ce = cursor, end
            if ce <= cs:
                ce = cs + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-C{seq:03d}",
                    document_id=doc.document_id,
                    category=doc.category,
                    section_path=s_path,
                    text=text,
                    inherited_metadata=meta,
                    char_start=cs,
                    char_end=ce,
                    token_estimate=_estimate_tokens(text),
                )
            )

        # Next window starts at end - overlap (paragraph-snapped downward)
        next_start = max(cursor + 1, end - overlap)
        # Snap to paragraph boundary >= next_start to avoid partial rows
        for pe in paragraph_ends:
            if next_start <= pe < end:
                next_start = pe
                break
        cursor = next_start

    return chunks


# ── Recursive strategy ────────────────────────────────────────────────────────


def _recursive_chunks(
    doc: EnterpriseDocument,
    config: ChunkingConfig,
) -> list[Chunk]:
    """Produce chunks using hierarchical recursive splitting per section."""
    if not doc.sections:
        return _fixed_size_chunks(doc, config)

    max_chars = _max_chars(config)
    min_chars = config.min_tokens * _CHARS_PER_TOKEN
    base_meta = build_inherited_metadata(doc)
    chunks: list[Chunk] = []

    for section in doc.sections:
        pieces = _recursive_text_split(
            text=section.content,
            base=section.char_start,
            max_chars=max_chars,
            min_chars=min_chars,
        )

        for text, cs, ce in pieces:
            if not text.strip():
                continue
            if ce <= cs:
                ce = cs + 1

            meta = dict(base_meta)
            meta["section_heading"] = section.heading
            meta["chunk_strategy"] = "recursive"
            meta["has_table"] = "|" in text and "| --- |" in text

            seq = len(chunks) + 1
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.document_id}-C{seq:03d}",
                    document_id=doc.document_id,
                    category=doc.category,
                    section_path=list(section.path),
                    text=text,
                    inherited_metadata=meta,
                    char_start=cs,
                    char_end=ce,
                    token_estimate=_estimate_tokens(text),
                )
            )

    return chunks


# ── Public pipeline ────────────────────────────────────────────────────────────


class ChunkingPipeline:
    """Converts a corpus of EnterpriseDocuments into Chunk objects."""

    def __init__(self, config: ChunkingConfig | None = None) -> None:
        self._config = config or ChunkingConfig()

    def chunk_document(self, doc: EnterpriseDocument) -> list[Chunk]:
        """Chunk a single document using the configured strategy."""
        strategy = self._config.strategy
        if strategy == "fixed_size":
            return _fixed_size_chunks(doc, self._config)
        elif strategy == "recursive":
            return _recursive_chunks(doc, self._config)
        else:
            return _section_aware_chunks(doc, self._config)

    def chunk_corpus(self, docs: list[EnterpriseDocument]) -> list[Chunk]:
        """Chunk all documents and return a flat list of Chunk objects."""
        result: list[Chunk] = []
        for doc in docs:
            result.extend(self.chunk_document(doc))
        return result


# ── Statistics & quality report ───────────────────────────────────────────────


@dataclass
class ChunkStatistics:
    """Aggregate statistics over a chunk corpus."""

    total_chunks: int = 0
    total_docs: int = 0
    per_category: dict[str, int] = field(default_factory=dict)
    chunks_per_doc: dict[str, int] = field(default_factory=dict)
    avg_tokens: float = 0.0
    min_tokens: int = 0
    max_tokens: int = 0
    p25_tokens: float = 0.0
    p50_tokens: float = 0.0
    p75_tokens: float = 0.0
    p90_tokens: float = 0.0
    p95_tokens: float = 0.0
    docs_with_chunks: int = 0
    table_chunks: int = 0


@dataclass
class ChunkQualityReport:
    """Quality metrics for a chunk corpus."""

    statistics: ChunkStatistics = field(default_factory=ChunkStatistics)
    metadata_valid_count: int = 0
    metadata_invalid_count: int = 0
    section_path_coverage: float = 0.0
    validation_errors: list[str] = field(default_factory=list)


def compute_statistics(
    chunks: list[Chunk],
    docs: list[EnterpriseDocument],
) -> ChunkStatistics:
    """Compute aggregate statistics for a chunk corpus."""
    import statistics as _stats

    stats = ChunkStatistics()
    stats.total_chunks = len(chunks)
    stats.total_docs = len(docs)

    if not chunks:
        return stats

    tokens = [c.token_estimate for c in chunks]
    stats.avg_tokens = sum(tokens) / len(tokens)
    stats.min_tokens = min(tokens)
    stats.max_tokens = max(tokens)

    sorted_tokens = sorted(tokens)
    n = len(sorted_tokens)

    def _pct(p: float) -> float:
        idx = min(int(p * n / 100), n - 1)
        return float(sorted_tokens[idx])

    stats.p25_tokens = _pct(25)
    stats.p50_tokens = _pct(50)
    stats.p75_tokens = _pct(75)
    stats.p90_tokens = _pct(90)
    stats.p95_tokens = _pct(95)

    by_cat: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    for c in chunks:
        cat = c.category.value
        by_cat[cat] = by_cat.get(cat, 0) + 1
        by_doc[c.document_id] = by_doc.get(c.document_id, 0) + 1

    stats.per_category = by_cat
    stats.chunks_per_doc = by_doc
    stats.docs_with_chunks = len(by_doc)
    stats.table_chunks = sum(
        1 for c in chunks if c.inherited_metadata.get("has_table", False)
    )

    return stats


def build_quality_report(
    chunks: list[Chunk],
    docs: list[EnterpriseDocument],
) -> ChunkQualityReport:
    """Build a quality report for the chunk corpus."""
    from .metadata import validate_inherited_metadata

    report = ChunkQualityReport()
    report.statistics = compute_statistics(chunks, docs)

    valid = 0
    invalid = 0
    errors: list[str] = []

    for c in chunks:
        missing = validate_inherited_metadata(c.inherited_metadata)
        if missing:
            invalid += 1
            errors.append(f"{c.chunk_id}: missing metadata keys {missing}")
        else:
            valid += 1

    report.metadata_valid_count = valid
    report.metadata_invalid_count = invalid
    report.validation_errors = errors[:20]  # cap for readability

    # Section path coverage: fraction of chunks with a meaningful path
    non_default = sum(
        1 for c in chunks if c.section_path != ["Document"]
    )
    report.section_path_coverage = non_default / len(chunks) if chunks else 0.0

    return report
