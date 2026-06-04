"""Planning Agent: wraps V1 RuleBasedPlanner with LLM fallback.

Dispatch logic:
  Rule confidence ≥ 0.75  (R01–R05) → use rule result directly
  Rule confidence < 0.75  (R06–R07) → try LLM; fall back to rule on failure
  Re-retrieval loop       (loop_count > 0) → honour Val Agent strategy override first

V1 integration contract:
  RuleBasedPlanner.plan() expects a dict with:
    query_id, reasoning_type, retrieval_difficulty_factors, category,
    planner_oracle_strategy, oracle_strategy_source
  All oracle fields are set to neutral values in V2 (not used for SSA scoring here).
"""

from __future__ import annotations

import logging
from typing import Any

from enterprise_rag.agents.state import QueryPlan, RetrievalPlan, SubQuery, ValidationResult
from enterprise_rag.planning.rule_based import RuleBasedPlanner
from enterprise_rag.planning.providers.base import BasePlannerProvider

logger = logging.getLogger(__name__)

# Rule confidence threshold: rules below this fire the LLM fallback
_LLM_FALLBACK_THRESHOLD = 0.75

# ── LLM fallback prompt ────────────────────────────────────────────────────────

_PLAN_SYSTEM_PROMPT = """\
You are a retrieval strategy selector for an enterprise knowledge retrieval system.

Select ONE retrieval strategy for the given sub-query.

STRATEGIES:
- bm25    Exact keyword matching; best for dates, version numbers, policy codes, names
- dense   Semantic similarity; best for exception clauses, concept questions, paraphrases
- hybrid  BM25 + Dense fusion; best for multi-hop, aggregation, or mixed signals

Respond with ONLY valid JSON — no markdown, no extra text:
{"strategy": "bm25|dense|hybrid", "rationale": "one sentence"}"""

_PLAN_USER_TEMPLATE = """\
Sub-query: {query_text}
Reasoning type: {reasoning_type}
Difficulty signals: {difficulty_signals}
Document category: {category}"""


def _build_rule_query(
    query_id: str,
    sub_query: SubQuery,
    query_plan: QueryPlan,
) -> dict[str, Any]:
    """Convert V2 types to the dict format expected by V1 RuleBasedPlanner."""
    return {
        "query_id": f"{query_id}_{sub_query.sub_query_id}",
        "reasoning_type": query_plan.reasoning_type,
        "retrieval_difficulty_factors": query_plan.difficulty_signals,
        "category": sub_query.target_category or "",
        # Oracle fields not meaningful in V2 — set to neutral defaults
        "planner_oracle_strategy": "hybrid",
        "oracle_strategy_source": "v2_planning_agent",
    }


def _llm_select_strategy(
    provider: BasePlannerProvider,
    sub_query: SubQuery,
    query_plan: QueryPlan,
) -> str | None:
    """Ask LLM to select a strategy. Returns strategy string or None on failure."""
    import json

    user_prompt = _PLAN_USER_TEMPLATE.format(
        query_text=sub_query.text,
        reasoning_type=query_plan.reasoning_type,
        difficulty_signals=", ".join(query_plan.difficulty_signals) or "none",
        category=sub_query.target_category or "unknown",
    )
    try:
        resp = provider.complete(
            user_prompt=user_prompt,
            system_prompt=_PLAN_SYSTEM_PROMPT,
            max_tokens=120,
            temperature=0.0,
        )
        raw = resp.content.strip()
        # Extract JSON
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start : end + 1])
            strategy = data.get("strategy", "")
            if strategy in {"bm25", "dense", "hybrid"}:
                logger.debug("LLM fallback strategy: %s (rationale: %s)", strategy, data.get("rationale", ""))
                return strategy
    except Exception as exc:
        logger.warning("Planning Agent LLM fallback failed: %s", exc)
    return None


# ── PlanningAgent class ────────────────────────────────────────────────────────

class PlanningAgent:
    """Hybrid rule + LLM planning agent.

    Args:
        provider: LLM provider for low-confidence fallback. None = rule-only.
    """

    def __init__(self, provider: BasePlannerProvider | None = None) -> None:
        self._rule_planner = RuleBasedPlanner()
        self._provider = provider

    def plan_sub_query(
        self,
        query_id: str,
        sub_query: SubQuery,
        query_plan: QueryPlan,
        loop_count: int = 0,
        validation_result: ValidationResult | None = None,
    ) -> RetrievalPlan:
        """Produce a RetrievalPlan for a single sub-query.

        On re-retrieval loops, Val Agent's suggested_strategy overrides the
        normal rule/LLM dispatch.
        """
        # ── Re-retrieval override ──────────────────────────────────────────────
        if loop_count > 0 and validation_result and validation_result.suggested_strategy:
            strategy = validation_result.suggested_strategy
            rationale = f"Loop {loop_count}: Val Agent override → {strategy}"
            return RetrievalPlan(
                sub_query_id=sub_query.sub_query_id,
                query_text=_fill_entity_slots(sub_query),
                strategy=strategy,
                top_k=10,
                planner_source="validation_override",
                planner_rationale=rationale,
            )

        # ── Rule planner ───────────────────────────────────────────────────────
        rule_query = _build_rule_query(query_id, sub_query, query_plan)
        decision = self._rule_planner.plan(rule_query)
        strategy = decision.selected_strategy
        source = "rule"
        rationale = f"{decision.matched_rules[0]}: {strategy}"

        # ── LLM fallback for low-confidence rules ──────────────────────────────
        if decision.planner_confidence < _LLM_FALLBACK_THRESHOLD and self._provider is not None:
            llm_strategy = _llm_select_strategy(self._provider, sub_query, query_plan)
            if llm_strategy is not None:
                strategy = llm_strategy
                source = "llm"
                rationale = f"LLM fallback (rule={decision.matched_rules[0]}, conf={decision.planner_confidence:.2f}): {strategy}"

        return RetrievalPlan(
            sub_query_id=sub_query.sub_query_id,
            query_text=_fill_entity_slots(sub_query),
            strategy=strategy,
            top_k=10,
            planner_source=source,
            planner_rationale=rationale,
        )

    def plan_all(
        self,
        query_id: str,
        query_plan: QueryPlan,
        loop_count: int = 0,
        validation_result: ValidationResult | None = None,
    ) -> list[RetrievalPlan]:
        """Plan all sub-queries in a QueryPlan."""
        return [
            self.plan_sub_query(query_id, sq, query_plan, loop_count, validation_result)
            for sq in query_plan.sub_queries
        ]


def _fill_entity_slots(sub_query: SubQuery) -> str:
    """Substitute any pre-filled entity slots into the sub-query text."""
    text = sub_query.text
    for slot, value in sub_query.entity_slots.items():
        text = text.replace(f"{{{slot}}}", value)
    return text


# ── Module-level singleton ─────────────────────────────────────────────────────

_agent: PlanningAgent | None = None


def get_planning_agent() -> PlanningAgent:
    global _agent
    if _agent is None:
        try:
            from enterprise_rag.planning.providers.factory import create_provider
            provider = create_provider(prompt_version="structured")
        except Exception:
            provider = None
        _agent = PlanningAgent(provider=provider)
    return _agent
