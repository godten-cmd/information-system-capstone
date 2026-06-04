"""LangGraph StateGraph construction and compilation for the V2 pipeline.

Usage:
    from enterprise_rag.graph.builder import build_graph
    graph = build_graph()
    result = graph.invoke(make_initial_state("q001", "What is the hotel limit?"))
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from enterprise_rag.agents.state import AgentState
from enterprise_rag.graph.edges import (
    increment_loop_count,
    route_after_validation,
)
from enterprise_rag.graph.nodes import (
    answer_agent_node,
    plan_agent_node,
    qa_agent_node,
    retrieval_agent_node,
    validation_agent_node,
)

# Node name constants (avoid magic strings across the codebase)
NODE_QA = "qa_agent"
NODE_PLAN = "plan_agent"
NODE_RETRIEVAL = "retrieval_agent"
NODE_VALIDATION = "validation_agent"
NODE_ANSWER = "answer_agent"
NODE_LOOP_COUNTER = "loop_counter"


def build_graph() -> "CompiledStateGraph":  # type: ignore[name-defined]
    """Build and compile the V2 agentic RAG graph.

    Graph topology:
        START → qa_agent → plan_agent → retrieval_agent → validation_agent
                                ↑                                 │
                         loop_counter ←── re_retrieve ───────────┤
                                                                  │ answer
                                                        answer_agent → END
                                                        (also END on loop limit)

    Returns:
        A compiled LangGraph graph ready to invoke.
    """
    builder: StateGraph = StateGraph(AgentState)

    # ── Register nodes ─────────────────────────────────────────────────────────
    builder.add_node(NODE_QA, qa_agent_node)
    builder.add_node(NODE_PLAN, plan_agent_node)
    builder.add_node(NODE_RETRIEVAL, retrieval_agent_node)
    builder.add_node(NODE_VALIDATION, validation_agent_node)
    builder.add_node(NODE_ANSWER, answer_agent_node)
    builder.add_node(NODE_LOOP_COUNTER, increment_loop_count)

    # ── Main path edges ────────────────────────────────────────────────────────
    builder.add_edge(START, NODE_QA)
    builder.add_edge(NODE_QA, NODE_PLAN)
    builder.add_edge(NODE_PLAN, NODE_RETRIEVAL)
    builder.add_edge(NODE_RETRIEVAL, NODE_VALIDATION)
    builder.add_edge(NODE_ANSWER, END)

    # ── Conditional edge after Validation ─────────────────────────────────────
    builder.add_conditional_edges(
        NODE_VALIDATION,
        route_after_validation,
        {
            "answer": NODE_ANSWER,
            "re_retrieve": NODE_LOOP_COUNTER,
            "end": END,
        },
    )

    # ── Re-retrieval back-edge ─────────────────────────────────────────────────
    # loop_counter increments loop_count, then returns to plan_agent
    builder.add_edge(NODE_LOOP_COUNTER, NODE_PLAN)

    return builder.compile()


# Module-level compiled graph (lazy singleton — import triggers compilation)
_graph = None


def get_graph() -> "CompiledStateGraph":  # type: ignore[name-defined]
    """Return the compiled graph, building it once on first call."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
