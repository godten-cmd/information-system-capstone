"""Schema placeholders for retrieval planner outputs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

SelectedStrategy = Literal[
    "keyword",
    "dense",
    "hybrid",
    "metadata_filtered",
    "hierarchical",
    "table_aware",
    "multi_hop",
]

PlannerType = Literal["rule_based", "llm_based"]


class PlannerDecision(BaseModel):
    """Placeholder schema for a planner strategy decision."""

    query_id: str
    selected_strategy: SelectedStrategy
    oracle_strategy: SelectedStrategy
    strategy_correct: bool
    planner_type: PlannerType

