"""Schema placeholders for retrieval outputs."""

from __future__ import annotations

from pydantic import BaseModel


class RetrievalResult(BaseModel):
    """Placeholder schema for one ranked retrieval result."""

    query_id: str
    chunk_id: str
    document_id: str
    rank: int
    score: float

