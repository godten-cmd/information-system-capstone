"""LangGraph node functions for the V2 agentic pipeline.

Each function receives the full AgentState and returns a partial dict
that LangGraph merges into the state. Returning only changed keys avoids
accidental overwrites of fields owned by other nodes.

Phase 12A: stub implementations.
Phase 12B: QA Agent, Planning Agent, Retrieval Agent use real logic.
Phase 12C: Validation Agent uses real ValidationAgent (adaptive scorer + failure diagnosis).
Phase 12D: Answer Agent uses real AnswerAgent (LLM synthesis + inline [N] citations).
"""

from __future__ import annotations

from datetime import datetime, timezone

from enterprise_rag.agents.state import (
    AgentAnswer,
    AgentState,
    QueryPlan,
    RetrievalPlan,
    SubQuery,
    TraceEntry,
    ValidationResult,
)

# ── Helper ─────────────────────────────────────────────────────────────────────


def _trace(agent: str, event: str, **data: object) -> list[TraceEntry]:
    return [
        TraceEntry(
            agent=agent,
            event=event,
            data=dict(data),
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
        )
    ]


# ── Node 1: Query Analysis Agent ───────────────────────────────────────────────


def qa_agent_node(state: AgentState) -> dict:
    """Classify reasoning type and decompose multi-hop queries into sub-queries.

    Phase 12B: uses QAAgent with LLM provider. Supports oracle-override mode
    via query_metadata["reasoning_type"] only when explicitly flagged
    query_metadata["use_oracle"] == True (for ablation experiments).
    """
    from enterprise_rag.agents.query_analysis import get_qa_agent

    query_id = state["query_id"]
    raw_query = state["raw_query"]
    meta = state["query_metadata"]

    # Ablation mode: bypass LLM if oracle override is requested
    oracle_type = None
    if meta.get("use_oracle") is True:
        oracle_type = meta.get("reasoning_type")

    try:
        agent = get_qa_agent()
        plan = agent.analyze(
            query_id=query_id,
            raw_query=raw_query,
            oracle_reasoning_type=oracle_type,
        )
    except Exception as exc:
        # Fallback to single-hop if QA Agent fails entirely
        plan = QueryPlan(
            query_id=query_id,
            raw_query=raw_query,
            reasoning_type="single_hop",
            is_decomposed=False,
            sub_queries=[SubQuery(sub_query_id="q1", text=raw_query)],
            difficulty_signals=[],
            classification_confidence=0.0,
        )
        return {
            "query_plan": plan,
            "error": f"qa_agent_node: {exc}",
            "trace": _trace("qa", "error", reason=str(exc)),
        }

    return {
        "query_plan": plan,
        "trace": _trace(
            "qa",
            "query_classified",
            reasoning_type=plan.reasoning_type,
            is_decomposed=plan.is_decomposed,
            num_sub_queries=len(plan.sub_queries),
            confidence=plan.classification_confidence,
            difficulty_signals=plan.difficulty_signals,
            oracle_override=oracle_type is not None,
        ),
    }


# ── Node 2: Planning Agent ─────────────────────────────────────────────────────


def plan_agent_node(state: AgentState) -> dict:
    """Select retrieval strategy for each sub-query using V1 rule planner + LLM fallback."""
    from enterprise_rag.agents.planning import get_planning_agent

    query_plan = state["query_plan"]
    if query_plan is None:
        return {
            "error": "plan_agent_node: query_plan is None",
            "trace": _trace("plan", "error", reason="query_plan_missing"),
        }

    loop_count = state["loop_count"]
    prior_val = state.get("validation_result")

    try:
        agent = get_planning_agent()
        plans = agent.plan_all(
            query_id=state["query_id"],
            query_plan=query_plan,
            loop_count=loop_count,
            validation_result=prior_val,
        )
    except Exception as exc:
        # Fallback: one hybrid plan per sub-query
        plans = [
            RetrievalPlan(
                sub_query_id=sq.sub_query_id,
                query_text=sq.text,
                strategy="hybrid",
                top_k=10,
                planner_source="fallback",
                planner_rationale=f"error fallback: {exc}",
            )
            for sq in query_plan.sub_queries
        ]
        return {
            "retrieval_plans": plans,
            "error": f"plan_agent_node: {exc}",
            "trace": _trace("plan", "error", reason=str(exc)),
        }

    return {
        "retrieval_plans": plans,
        "trace": _trace(
            "plan",
            "strategies_selected",
            strategies={p.sub_query_id: p.strategy for p in plans},
            sources={p.sub_query_id: p.planner_source for p in plans},
            loop_count=loop_count,
        ),
    }


# ── Node 3: Retrieval Agent ────────────────────────────────────────────────────


def retrieval_agent_node(state: AgentState) -> dict:
    """Execute retrieval plans via MCP tools with multi-hop slot filling."""
    from enterprise_rag.agents.retrieval import get_retrieval_agent

    query_plan = state["query_plan"]
    if query_plan is None or not state["retrieval_plans"]:
        return {
            "retrieval_outputs": [],
            "trace": _trace("retrieval", "skipped", reason="no_plans"),
        }

    try:
        agent = get_retrieval_agent()
        outputs = agent.execute(
            query_id=state["query_id"],
            retrieval_plans=state["retrieval_plans"],
            query_plan=query_plan,
        )
    except Exception as exc:
        return {
            "retrieval_outputs": [],
            "error": f"retrieval_agent_node: {exc}",
            "trace": _trace("retrieval", "error", reason=str(exc)),
        }

    total_chunks = sum(len(o.chunks) for o in outputs)
    return {
        "retrieval_outputs": outputs,
        "trace": _trace(
            "retrieval",
            "chunks_retrieved",
            num_sub_queries=len(state["retrieval_plans"]),
            total_chunks=total_chunks,
            strategies=[o.strategy for o in outputs],
        ),
    }


# ── Node 4: Validation Agent ───────────────────────────────────────────────────


def validation_agent_node(state: AgentState) -> dict:
    """Score retrieved chunks for relevance; trigger re-retrieval if below threshold.

    Phase 12C: uses ValidationAgent with configurable scorer (adaptive by default).
    Scorer hierarchy: rank_proxy → bm25 → dense → adaptive → llm.
    """
    from enterprise_rag.agents.validation import get_validation_agent

    outputs = state["retrieval_outputs"]
    all_chunks: list[ChunkResult] = [c for o in outputs for c in o.chunks]
    query_plan = state["query_plan"]

    if query_plan is None:
        return {
            "validation_result": None,
            "validated_evidence": [],
            "trace": _trace("validation", "skipped", reason="no_query_plan"),
        }

    try:
        agent = get_validation_agent()
        result = agent.validate(
            chunks=all_chunks,
            query=state["raw_query"],
            query_plan=query_plan,
            retrieval_plans=state["retrieval_plans"],
            retrieval_outputs=outputs,
        )
    except Exception as exc:
        # Fallback: pass everything through so the pipeline can still answer
        result = ValidationResult(
            passed=True,
            decisions=[],
            passed_count=len(all_chunks),
            total_count=len(all_chunks),
            threshold=0.30,
            min_passed=3,
            failure_reason=f"validation_error: {exc}",
        )
        return {
            "validation_result": result,
            "validated_evidence": all_chunks,
            "error": f"validation_agent_node: {exc}",
            "trace": _trace("validation", "error", reason=str(exc)),
        }

    passed_chunks = [c for c, d in zip(all_chunks, result.decisions) if d.passed]

    return {
        "validation_result": result,
        "validated_evidence": passed_chunks,
        "trace": _trace(
            "validation",
            "validation_complete",
            passed=result.passed,
            passed_count=result.passed_count,
            total_count=result.total_count,
            threshold=result.threshold,
            failure_reason=result.failure_reason,
            suggested_strategy=result.suggested_strategy,
            scorer=agent.config.scorer,
        ),
    }


# ── Node 5: Answer Agent ───────────────────────────────────────────────────────


def answer_agent_node(state: AgentState) -> dict:
    """Synthesize a grounded answer from validated evidence using LLM + citations.

    Phase 12D: uses AnswerAgent with inline [N] citation markers.
    Falls back to extractive top-3 concatenation when no LLM provider is available.
    """
    from enterprise_rag.agents.answer import get_answer_agent

    query_plan = state["query_plan"]
    evidence = state["validated_evidence"]

    if query_plan is None:
        return {
            "answer": None,
            "trace": _trace("answer", "skipped", reason="no_query_plan"),
        }

    try:
        agent = get_answer_agent()
        answer = agent.synthesize(
            query=state["raw_query"],
            evidence=evidence,
            query_plan=query_plan,
        )
    except Exception as exc:
        answer = AgentAnswer(
            query_id=state["query_id"],
            answer_text="",
            cited_chunk_ids=[],
            confidence=0.0,
            evidence_coverage=0.0,
        )
        return {
            "answer": answer,
            "error": f"answer_agent_node: {exc}",
            "trace": _trace("answer", "error", reason=str(exc)),
        }

    return {
        "answer": answer,
        "trace": _trace(
            "answer",
            "answer_generated",
            cited_count=len(answer.cited_chunk_ids),
            confidence=answer.confidence,
            evidence_coverage=answer.evidence_coverage,
            evidence_count=len(evidence),
            stub=False,
        ),
    }
