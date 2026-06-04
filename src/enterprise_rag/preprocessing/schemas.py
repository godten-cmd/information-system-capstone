"""Schema placeholders for chunking and preprocessing artifacts."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChunkRecord(BaseModel):
    """Placeholder schema for a retrievable document chunk."""

    chunk_id: str
    document_id: str
    category: str
    section_path: list[str] = Field(default_factory=list)
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)

