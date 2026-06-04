"""Retrieval Agent: multi-hop sequential execution with entity slot filling.

Execution modes:
  single_hop / exception / temporal / comparison:
      Execute all sub-queries independently (parallel semantics, sequential impl).

  multi_hop:
      Execute sub-queries in dependency order.
      After each Q_n retrieval, extract the entity value needed by Q_{n+1}
      and fill the slot before running the next retrieval.

  aggregation:
      Execute all sub-queries independently; merge and deduplicate results.

Entity slot filling (multi-hop):
  When Q2.text contains {variable}, the Retrieval Agent:
  1. Retrieves Q1 results
  2. Calls _extract_entity_value() to find the slot value in Q1's top chunks
  3. Fills Q2.entity_slots and rewrites Q2.query_text with the actual value
  4. Retrieves Q2 with the enriched query
"""

from __future__ import annotations

import logging
import re
from typing import Any

from enterprise_rag.agents.state import ChunkResult, QueryPlan, RetrievalOutput, RetrievalPlan

logger = logging.getLogger(__name__)

# ── Tool dispatch ──────────────────────────────────────────────────────────────

def _get_tool(strategy: str):
    """Return the appropriate retrieval function for a strategy string."""
    if strategy == "bm25":
        from enterprise_rag.mcp.tools.bm25_tool import bm25_search
        return bm25_search
    if strategy == "dense":
        from enterprise_rag.mcp.tools.dense_tool import dense_search
        return dense_search
    # default: hybrid
    from enterprise_rag.mcp.tools.hybrid_tool import hybrid_search
    return hybrid_search


def _retrieve(
    query_text: str,
    strategy: str,
    query_id: str,
    top_k: int = 10,
) -> list[dict]:
    tool = _get_tool(strategy)
    return tool(query=query_text, top_k=top_k, query_id=query_id)


def _raw_to_chunks(raw: list[dict], strategy: str, sub_query_id: str) -> list[ChunkResult]:
    return [
        ChunkResult(
            chunk_id=r["chunk_id"],
            document_id=r["document_id"],
            rank=r["rank"],
            score=r["score"],
            text=r["text"],
            category=r["category"],
            section_path=r.get("section_path", []),
            strategy_used=strategy,
            sub_query_id=sub_query_id,
        )
        for r in raw
    ]


# ── Entity extraction for multi-hop slot filling ───────────────────────────────

_SLOT_PATTERN = re.compile(r"\{(\w+)\}")


def _detect_slots(text: str) -> list[str]:
    """Return list of slot names found in text, e.g. ['auth_method']."""
    return _SLOT_PATTERN.findall(text)


def _extract_entity_value(
    slot_name: str,
    q1_chunks: list[ChunkResult],
    provider: Any | None = None,
) -> str:
    """Extract the value for `slot_name` from Q1 retrieval results.

    Strategy:
    1. If LLM provider available: ask LLM to extract the value from top-3 chunks.
    2. Fallback: return the first 80 chars of the top-ranked chunk text
       (gives Q2 enough context to work as a semantic search).
    """
    top_chunks = sorted(q1_chunks, key=lambda c: c.rank)[:3]
    context = " | ".join(c.text[:200] for c in top_chunks)

    if provider is not None:
        import json as _json
        system = (
            "You are an entity extractor. Given context text and a slot name, "
            "extract the exact value that fills the slot. "
            "Respond with ONLY a JSON object: {\"value\": \"extracted value\"}"
        )
        user = f"Context: {context}\n\nSlot to fill: {slot_name}"
        try:
            resp = provider.complete(
                user_prompt=user,
                system_prompt=system,
                max_tokens=80,
                temperature=0.0,
            )
            raw = resp.content.strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                data = _json.loads(raw[start : end + 1])
                value = str(data.get("value", "")).strip()
                if value:
                    logger.debug("Entity extracted: %s = %r", slot_name, value)
                    return value
        except Exception as exc:
            logger.warning("Entity extraction LLM failed: %s", exc)

    # Fallback: use top chunk text as enrichment context (semantic search still works)
    if top_chunks:
        return top_chunks[0].text[:80].strip()
    return slot_name  # last resort: keep slot name as-is


def _fill_slots(
    plan: RetrievalPlan,
    q1_chunks: list[ChunkResult],
    provider: Any | None = None,
) -> RetrievalPlan:
    """Fill all {slot} variables in plan.query_text from Q1 results.

    Returns a new RetrievalPlan with enriched query_text.
    """
    slots = _detect_slots(plan.query_text)
    if not slots:
        return plan

    enriched_text = plan.query_text
    for slot in slots:
        value = _extract_entity_value(slot, q1_chunks, provider=provider)
        enriched_text = enriched_text.replace(f"{{{slot}}}", value)
        logger.debug("Slot filled: {%s} → %r", slot, value)

    return RetrievalPlan(
        sub_query_id=plan.sub_query_id,
        query_text=enriched_text,
        strategy=plan.strategy,
        top_k=plan.top_k,
        planner_source=plan.planner_source,
        planner_rationale=f"{plan.planner_rationale} [slot-filled]",
    )


# ── Retrieval Agent class ──────────────────────────────────────────────────────

class RetrievalAgent:
    """Executes retrieval plans with multi-hop slot filling and aggregation merging.

    Args:
        provider: LLM provider for entity extraction. None = heuristic fallback.
    """

    def __init__(self, provider: Any | None = None) -> None:
        self._provider = provider

    def execute(
        self,
        query_id: str,
        retrieval_plans: list[RetrievalPlan],
        query_plan: QueryPlan,
    ) -> list[RetrievalOutput]:
        """Execute all retrieval plans, respecting multi-hop dependencies.

        Returns:
            List of RetrievalOutput, one per sub-query.
        """
        reasoning_type = query_plan.reasoning_type

        if reasoning_type == "multi_hop" and len(retrieval_plans) > 1:
            return self._execute_sequential(query_id, retrieval_plans)
        if reasoning_type == "aggregation" and len(retrieval_plans) > 1:
            return self._execute_parallel_merged(query_id, retrieval_plans)
        return self._execute_independent(query_id, retrieval_plans)

    def _execute_independent(
        self,
        query_id: str,
        plans: list[RetrievalPlan],
    ) -> list[RetrievalOutput]:
        """Execute each plan independently (single_hop, exception, temporal, comparison)."""
        outputs = []
        for plan in plans:
            raw = _retrieve(
                plan.query_text,
                plan.strategy,
                f"{query_id}_{plan.sub_query_id}",
                plan.top_k,
            )
            chunks = _raw_to_chunks(raw, plan.strategy, plan.sub_query_id)
            outputs.append(
                RetrievalOutput(
                    sub_query_id=plan.sub_query_id,
                    query_text=plan.query_text,
                    strategy=plan.strategy,
                    chunks=chunks,
                    latency_ms=raw[0]["latency_ms"] if raw else 0.0,
                )
            )
        return outputs

    def _execute_sequential(
        self,
        query_id: str,
        plans: list[RetrievalPlan],
    ) -> list[RetrievalOutput]:
        """Execute sub-queries in dependency order; fill slots from prior results."""
        # Build lookup: sub_query_id → plan (mutable via slot filling)
        plan_map: dict[str, RetrievalPlan] = {p.sub_query_id: p for p in plans}
        # Determine execution order: plans without depends_on first, then dependents
        ordered = _topological_sort(plans)

        outputs: list[RetrievalOutput] = []
        results_by_id: dict[str, list[ChunkResult]] = {}

        for plan in ordered:
            # Fill any {slots} using the resolved results of the dependency
            filled_plan = plan
            if plan.sub_query_id in plan_map:
                slots = _detect_slots(plan.query_text)
                if slots:
                    # Find the dependency's results
                    dep_id = _find_dependency(plan, plans)
                    if dep_id and dep_id in results_by_id:
                        filled_plan = _fill_slots(plan, results_by_id[dep_id], self._provider)
                    else:
                        logger.warning(
                            "Slot fill: dependency %r not yet resolved for %s",
                            dep_id, plan.sub_query_id
                        )

            raw = _retrieve(
                filled_plan.query_text,
                filled_plan.strategy,
                f"{query_id}_{filled_plan.sub_query_id}",
                filled_plan.top_k,
            )
            chunks = _raw_to_chunks(raw, filled_plan.strategy, filled_plan.sub_query_id)
            results_by_id[filled_plan.sub_query_id] = chunks
            outputs.append(
                RetrievalOutput(
                    sub_query_id=filled_plan.sub_query_id,
                    query_text=filled_plan.query_text,
                    strategy=filled_plan.strategy,
                    chunks=chunks,
                    latency_ms=raw[0]["latency_ms"] if raw else 0.0,
                )
            )

        return outputs

    def _execute_parallel_merged(
        self,
        query_id: str,
        plans: list[RetrievalPlan],
    ) -> list[RetrievalOutput]:
        """Execute all plans independently, then emit one merged output.

        Used for aggregation queries where multiple sub-queries cover different
        aspects of the same aggregation and results should be unified.
        """
        all_chunks: list[ChunkResult] = []
        latency_total = 0.0

        for plan in plans:
            raw = _retrieve(
                plan.query_text,
                plan.strategy,
                f"{query_id}_{plan.sub_query_id}",
                plan.top_k,
            )
            chunks = _raw_to_chunks(raw, plan.strategy, plan.sub_query_id)
            all_chunks.extend(chunks)
            if raw:
                latency_total += raw[0]["latency_ms"]

        # Deduplicate by chunk_id, keeping the lowest rank (best) occurrence
        seen: dict[str, ChunkResult] = {}
        for chunk in all_chunks:
            if chunk.chunk_id not in seen or chunk.rank < seen[chunk.chunk_id].rank:
                seen[chunk.chunk_id] = chunk

        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)
        # Re-rank after merge
        for new_rank, chunk in enumerate(merged, start=1):
            object.__setattr__(chunk, "rank", new_rank) if hasattr(chunk, "__setattr__") else None

        return [
            RetrievalOutput(
                sub_query_id="merged",
                query_text=" | ".join(p.query_text for p in plans),
                strategy="merged",
                chunks=merged,
                latency_ms=latency_total,
            )
        ]


def _topological_sort(plans: list[RetrievalPlan]) -> list[RetrievalPlan]:
    """Sort plans so that each plan comes after its dependency.

    Simple: plans with no depends_on first, then dependents.
    Does not handle circular dependencies (which the QA Agent should not produce).
    """
    no_dep = [p for p in plans if not _find_dependency(p, plans)]
    has_dep = [p for p in plans if _find_dependency(p, plans)]
    return no_dep + has_dep


def _find_dependency(plan: RetrievalPlan, all_plans: list[RetrievalPlan]) -> str | None:
    """Return the sub_query_id this plan depends on, if any slots exist."""
    slots = _detect_slots(plan.query_text)
    if not slots:
        return None
    # Match slot names to sub_query_ids of other plans (e.g. "q1" in "{value_from_q1}")
    for other in all_plans:
        if other.sub_query_id == plan.sub_query_id:
            continue
        for slot in slots:
            if other.sub_query_id in slot:
                return other.sub_query_id
    # If no explicit match, return the first other plan's id (linear dependency assumption)
    others = [p for p in all_plans if p.sub_query_id != plan.sub_query_id]
    return others[0].sub_query_id if others else None


# ── Module-level singleton ─────────────────────────────────────────────────────

_agent: RetrievalAgent | None = None


def get_retrieval_agent() -> RetrievalAgent:
    global _agent
    if _agent is None:
        try:
            from enterprise_rag.planning.providers.factory import create_provider
            provider = create_provider(prompt_version="structured")
        except Exception:
            provider = None
        _agent = RetrievalAgent(provider=provider)
    return _agent
