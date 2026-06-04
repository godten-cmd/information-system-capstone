"""V2 shared agent state: TypedDict for LangGraph + Pydantic sub-models.

All Pydantic models in this module represent structured sub-objects stored
inside AgentState. Nodes update state by returning a partial dict; LangGraph
merges the partial dict into the shared state via its reducer logic.

The `trace` field uses operator.add as its reducer so each node can append
entries without overwriting prior entries from other nodes.
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, TypedDict

from pydantic import BaseModel, Field


# ── Sub-query and Query Plan ───────────────────────────────────────────────────


class SubQuery(BaseModel):
    """One sub-query within a decomposed QueryPlan."""

    sub_query_id: str
    text: str
    strategy_hint: str | None = None       # bm25 | dense | hybrid | None
    target_category: str | None = None     # restrict search to this doc category
    entity_slots: dict[str, str] = Field(default_factory=dict)
    depends_on: str | None = None          # sub_query_id this must run after


class QueryPlan(BaseModel):
    """Output of the Query Analysis Agent."""

    query_id: str
    raw_query: str
    reasoning_type: str                    # single_hop | multi_hop | aggregation |
    #                                        comparison | exception | temporal
    is_decomposed: bool
    sub_queries: list[SubQuery]
    difficulty_signals: list[str] = Field(default_factory=list)
    classification_confidence: float = 1.0


# ── Retrieval Plan ─────────────────────────────────────────────────────────────


class RetrievalPlan(BaseModel):
    """One retrieval plan for one sub-query, from the Planning Agent."""

    sub_query_id: str
    query_text: str                        # enriched with entity slot values if filled
    strategy: str                          # bm25 | dense | hybrid
    top_k: int = 10
    planner_source: str                    # rule | llm
    planner_rationale: str = ""


# ── Retrieval Output ───────────────────────────────────────────────────────────


class ChunkResult(BaseModel):
    """A single retrieved chunk with provenance metadata."""

    chunk_id: str
    document_id: str
    rank: int
    score: float
    text: str
    category: str
    section_path: list[str] = Field(default_factory=list)
    strategy_used: str = ""
    sub_query_id: str = ""


class RetrievalOutput(BaseModel):
    """All chunks returned for one sub-query by the Retrieval Agent."""

    sub_query_id: str
    query_text: str
    strategy: str
    chunks: list[ChunkResult]
    latency_ms: float = 0.0


# ── Validation ────────────────────────────────────────────────────────────────


class ValidationDecision(BaseModel):
    """Relevance verdict for one chunk from the Validation Agent."""

    chunk_id: str
    relevance_score: float
    passed: bool
    reason: str = ""


class ValidationResult(BaseModel):
    """Aggregate output of the Validation Agent for one loop iteration."""

    passed: bool                           # True ⟺ enough chunks cleared threshold
    decisions: list[ValidationDecision]
    passed_count: int
    total_count: int
    threshold: float
    min_passed: int
    failure_reason: str = ""              # too_narrow | wrong_strategy | missing_entity
    suggested_strategy: str | None = None
    suggested_query_expansion: str | None = None


# ── Answer ────────────────────────────────────────────────────────────────────


class AgentAnswer(BaseModel):
    """Output of the Answer Agent."""

    query_id: str
    answer_text: str
    cited_chunk_ids: list[str]
    confidence: float                      # 0.0–1.0
    has_contradiction: bool = False
    contradiction_note: str = ""
    evidence_coverage: float = 0.0        # fraction of validated evidence used


# ── Trace ─────────────────────────────────────────────────────────────────────


class TraceEntry(BaseModel):
    """One timestamped decision recorded by any agent."""

    agent: str                             # qa | plan | retrieval | validation | answer
    event: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Shared Graph State ────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """LangGraph shared state threaded through all V2 agent nodes.

    Nodes return a partial dict; LangGraph merges it into the current state.
    The `trace` field accumulates across all nodes via operator.add.
    """

    query_id: str
    raw_query: str
    query_metadata: dict[str, Any]         # SEKD metadata (reasoning_type, category, …)

    # Set by QA Agent
    query_plan: QueryPlan | None

    # Set by Planning Agent
    retrieval_plans: list[RetrievalPlan]

    # Set by Retrieval Agent
    retrieval_outputs: list[RetrievalOutput]

    # Set by Validation Agent
    validation_result: ValidationResult | None
    validated_evidence: list[ChunkResult]

    # Set by Answer Agent
    answer: AgentAnswer | None

    # Accumulated by all agents (operator.add reducer keeps history)
    trace: Annotated[list[TraceEntry], operator.add]

    # Loop control: incremented each time Validation Agent triggers re-retrieval
    loop_count: int

    # First error message if any agent raises; graph routes to END immediately
    error: str | None


# ── State factory ──────────────────────────────────────────────────────────────


def make_initial_state(
    query_id: str,
    raw_query: str,
    query_metadata: dict[str, Any] | None = None,
) -> AgentState:
    """Return a fully initialized AgentState for a single query."""
    return AgentState(
        query_id=query_id,
        raw_query=raw_query,
        query_metadata=query_metadata or {},
        query_plan=None,
        retrieval_plans=[],
        retrieval_outputs=[],
        validation_result=None,
        validated_evidence=[],
        answer=None,
        trace=[],
        loop_count=0,
        error=None,
    )
