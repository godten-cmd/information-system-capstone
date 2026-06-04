"""Schema placeholders for evaluation outputs."""

from __future__ import annotations

from pydantic import BaseModel


class MetricRecord(BaseModel):
    """Placeholder schema for an evaluation metric."""

    method: str
    metric: str
    value: float
    group: str | None = None

