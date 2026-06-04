"""Schema placeholders for experiment manifests."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RunManifest(BaseModel):
    """Placeholder schema for reproducible experiment metadata."""

    run_id: str
    seed: int
    dataset_version: str
    methods: list[str] = Field(default_factory=list)

