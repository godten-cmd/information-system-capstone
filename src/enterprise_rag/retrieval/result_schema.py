"""Standardized retrieval result schemas shared across all retrieval methods."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class RetrievedChunk:
    """A single ranked chunk returned by a retrieval system."""

    query_id: str
    chunk_id: str
    document_id: str
    rank: int
    score: float
    text: str = ""
    category: str = ""
    section_path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "rank": self.rank,
            "score": self.score,
            "category": self.category,
            "section_path": self.section_path,
        }


@dataclass
class RetrievalRun:
    """Container for one complete retrieval run over all queries."""

    method: str
    k: int
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    results: list[RetrievedChunk] = field(default_factory=list)
    config: dict = field(default_factory=dict)

    def results_for(self, query_id: str) -> list[RetrievedChunk]:
        return sorted(
            [r for r in self.results if r.query_id == query_id],
            key=lambda r: r.rank,
        )

    def top_chunk_ids_for(self, query_id: str, k: int | None = None) -> list[str]:
        results = self.results_for(query_id)
        if k is not None:
            results = results[:k]
        return [r.chunk_id for r in results]
