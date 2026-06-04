"""LangGraph conditional edge logic for the V2 agentic pipeline.

The only branch point is after the Validation Agent:
  - "answer"      : validation passed → proceed to Answer Agent
  - "re_retrieve" : validation failed, loop_count < MAX_LOOPS → back to Plan Agent
  - "end"         : loop limit reached or unrecoverable error → terminate
"""

from __future__ import annotations

from enterprise_rag.agents.state import AgentState

MAX_LOOPS: int = 3  # maximum re-retrieval attempts before forced termination


def route_after_validation(state: AgentState) -> str:
    """Routing function attached to the validation_agent conditional edge.

    Returns one of: "answer" | "re_retrieve" | "end"
    """
    if state.get("error"):
        return "end"

    val = state.get("validation_result")
    if val is None:
        return "end"

    if val.passed:
        return "answer"

    loop_count = state.get("loop_count", 0)
    if loop_count < MAX_LOOPS:
        return "re_retrieve"

    # Loop limit exhausted — proceed with whatever evidence we have
    return "answer"


def increment_loop_count(state: AgentState) -> dict:
    """Node inserted on the re-retrieval back-edge to increment loop_count.

    LangGraph requires that back-edges pass through a node (not a raw edge)
    to allow state mutation. This node does exactly one thing.
    """
    from datetime import datetime, timezone

    from enterprise_rag.agents.state import TraceEntry

    new_count = state.get("loop_count", 0) + 1
    return {
        "loop_count": new_count,
        "trace": [
            TraceEntry(
                agent="loop_counter",
                event="re_retrieval_loop",
                data={"loop_count": new_count},
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        ],
    }
