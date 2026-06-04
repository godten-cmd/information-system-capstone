"""QA Agent: LLM-based query classification and multi-hop decomposition.

Operates WITHOUT oracle metadata — classifies reasoning type and difficulty
from query text alone. This is the key difference from V1, which was given
the oracle reasoning_type as input to the planner.

Oracle-override mode (for ablation experiment E6):
    Pass oracle_reasoning_type to QAAgent.analyze() to bypass LLM classification.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from enterprise_rag.agents.state import QueryPlan, SubQuery
from enterprise_rag.planning.providers.base import BasePlannerProvider

logger = logging.getLogger(__name__)

# ── System prompt ──────────────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
You are a Query Analysis Agent for an enterprise knowledge retrieval system.

Given a query, output a JSON object with:
1. reasoning_type — how to answer this query
2. difficulty_signals — what makes it hard
3. sub_queries — decomposed steps (multi_hop only gets multiple; all others get one)

REASONING TYPES:
- single_hop   One document/chunk contains the complete answer
- multi_hop    Requires chaining across multiple documents (A→B→C pattern)
- aggregation  Collect and list ALL instances matching a criterion
- comparison   Contrast two or more entities on the same dimension
- exception    Find conditions that exempt from a general rule
- temporal     Involves dates, versions, time ordering, before/after

DIFFICULTY SIGNALS (include all that apply):
- temporal_reasoning         date/version references, "before/after/since/during"
- exception_handling         "except", "unless", "exempt", "waiver", "special case"
- multi_document_dependency  answer spans multiple documents
- table_dependency           requires reading numeric values from a table
- ambiguous_scope            could apply to multiple policies/categories
- cross_category             spans multiple document categories

DECOMPOSITION (multi_hop queries only):
- Create 2-3 ordered sub-queries; each can be retrieved independently
- When Q2 depends on a value found by Q1, write it as a slot: {variable_name}
- Set depends_on to the sub_query_id of the prior step

STRATEGY HINTS (optional guidance for the Planning Agent):
- bm25    exact keywords, dates, policy codes, version numbers
- dense   semantic concepts, exception clauses, paraphrased lookup
- hybrid  multi-step, aggregation, or mixed signals
- null    let the Planning Agent decide

OUTPUT: valid JSON only — no markdown, no explanation, nothing outside the JSON:
{
  "reasoning_type": "single_hop",
  "difficulty_signals": [],
  "is_decomposed": false,
  "sub_queries": [
    {
      "sub_query_id": "q1",
      "text": "<query text>",
      "strategy_hint": null,
      "target_category": null,
      "depends_on": null
    }
  ],
  "classification_confidence": 0.90
}"""

_QA_USER_TEMPLATE = "Query: {query}"

# ── Valid values ───────────────────────────────────────────────────────────────

_VALID_REASONING_TYPES = frozenset(
    {"single_hop", "multi_hop", "aggregation", "comparison", "exception", "temporal"}
)
_VALID_STRATEGY_HINTS = frozenset({"bm25", "dense", "hybrid", None})
_VALID_DIFFICULTY_SIGNALS = frozenset(
    {
        "temporal_reasoning",
        "exception_handling",
        "multi_document_dependency",
        "table_dependency",
        "ambiguous_scope",
        "cross_category",
    }
)


# ── Fallback heuristic (no LLM available) ────────────────────────────────────

_TEMPORAL_PATTERNS = re.compile(
    r"\b(before|after|since|during|when|version|v\d|q[1-4]\s*20\d\d|20\d\d|overhaul|release)\b",
    re.IGNORECASE,
)
_EXCEPTION_PATTERNS = re.compile(
    r"\b(except|unless|exempt|waiver|special case|override|exempt)\b", re.IGNORECASE
)
_AGGREGATION_PATTERNS = re.compile(
    r"\b(all|every|each|list|count|how many|enumerate)\b", re.IGNORECASE
)
_COMPARISON_PATTERNS = re.compile(
    r"\b(compare|versus|vs\.?|difference|contrast|senior|junior|between)\b", re.IGNORECASE
)
_MULTI_HOP_PATTERNS = re.compile(
    r"\b(same .{0,30} (as|used by)|which .{0,30} (use|require|follow) .{0,30} (same|documented|described))\b",
    re.IGNORECASE,
)


def _heuristic_classify(query: str) -> str:
    """Fast regex-based fallback classification when no LLM is available."""
    if _TEMPORAL_PATTERNS.search(query):
        return "temporal"
    if _EXCEPTION_PATTERNS.search(query):
        return "exception"
    if _MULTI_HOP_PATTERNS.search(query):
        return "multi_hop"
    if _AGGREGATION_PATTERNS.search(query):
        return "aggregation"
    if _COMPARISON_PATTERNS.search(query):
        return "comparison"
    return "single_hop"


def _heuristic_signals(query: str) -> list[str]:
    signals: list[str] = []
    if _TEMPORAL_PATTERNS.search(query):
        signals.append("temporal_reasoning")
    if _EXCEPTION_PATTERNS.search(query):
        signals.append("exception_handling")
    return signals


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract the first JSON object from LLM output, tolerating preamble/postamble."""
    text = text.strip()
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No valid JSON object found in: {text[:200]!r}")


def _parse_qa_response(raw: str, query_id: str, raw_query: str) -> QueryPlan:
    """Parse LLM JSON response into a QueryPlan, validating all fields."""
    data = _extract_json(raw)

    reasoning_type = str(data.get("reasoning_type", "single_hop"))
    if reasoning_type not in _VALID_REASONING_TYPES:
        logger.warning("QA Agent: unknown reasoning_type %r, defaulting to single_hop", reasoning_type)
        reasoning_type = "single_hop"

    raw_signals = data.get("difficulty_signals", [])
    difficulty_signals = [s for s in raw_signals if s in _VALID_DIFFICULTY_SIGNALS]

    is_decomposed = bool(data.get("is_decomposed", False))
    confidence = float(data.get("classification_confidence", 0.8))
    confidence = max(0.0, min(1.0, confidence))

    raw_sqs = data.get("sub_queries", [])
    if not raw_sqs:
        # LLM forgot to include sub_queries — create single fallback
        raw_sqs = [{"sub_query_id": "q1", "text": raw_query}]

    sub_queries: list[SubQuery] = []
    for sq in raw_sqs:
        hint = sq.get("strategy_hint")
        if hint not in _VALID_STRATEGY_HINTS:
            hint = None
        sub_queries.append(
            SubQuery(
                sub_query_id=str(sq.get("sub_query_id", f"q{len(sub_queries)+1}")),
                text=str(sq.get("text", raw_query)),
                strategy_hint=hint,
                target_category=sq.get("target_category") or None,
                entity_slots={},
                depends_on=sq.get("depends_on") or None,
            )
        )

    # Safety: if multi_hop but only one sub-query, mark as not decomposed
    if is_decomposed and len(sub_queries) < 2:
        is_decomposed = False

    return QueryPlan(
        query_id=query_id,
        raw_query=raw_query,
        reasoning_type=reasoning_type,
        is_decomposed=is_decomposed,
        sub_queries=sub_queries,
        difficulty_signals=difficulty_signals,
        classification_confidence=confidence,
    )


# ── QA Agent class ─────────────────────────────────────────────────────────────

class QAAgent:
    """LLM-based query classifier and decomposer.

    Args:
        provider: LLM provider (OpenAI, Anthropic, or Mock).
            If None, falls back to heuristic classification.
    """

    def __init__(self, provider: BasePlannerProvider | None = None) -> None:
        self._provider = provider

    def analyze(
        self,
        query_id: str,
        raw_query: str,
        oracle_reasoning_type: str | None = None,
    ) -> QueryPlan:
        """Classify and decompose a query.

        Args:
            query_id: Unique identifier.
            raw_query: The query text.
            oracle_reasoning_type: If provided (ablation mode), skip LLM
                classification and use this type directly. Sub-query
                decomposition still runs.

        Returns:
            QueryPlan with reasoning_type, difficulty_signals, sub_queries.
        """
        if oracle_reasoning_type is not None:
            # Ablation / evaluation mode: bypass LLM classification
            return self._make_oracle_plan(query_id, raw_query, oracle_reasoning_type)

        if self._provider is None:
            return self._heuristic_plan(query_id, raw_query)

        return self._llm_plan(query_id, raw_query)

    def _llm_plan(self, query_id: str, raw_query: str) -> QueryPlan:
        user_prompt = _QA_USER_TEMPLATE.format(query=raw_query)
        try:
            response = self._provider.complete(
                user_prompt=user_prompt,
                system_prompt=_QA_SYSTEM_PROMPT,
                max_tokens=600,
                temperature=0.0,
            )
            plan = _parse_qa_response(response.content, query_id, raw_query)
            logger.debug(
                "QA Agent [%s]: reasoning_type=%s confidence=%.2f decomposed=%s",
                query_id,
                plan.reasoning_type,
                plan.classification_confidence,
                plan.is_decomposed,
            )
            return plan
        except Exception as exc:
            logger.warning("QA Agent LLM failed for %s: %s — using heuristic", query_id, exc)
            return self._heuristic_plan(query_id, raw_query)

    def _heuristic_plan(self, query_id: str, raw_query: str) -> QueryPlan:
        reasoning_type = _heuristic_classify(raw_query)
        signals = _heuristic_signals(raw_query)
        return QueryPlan(
            query_id=query_id,
            raw_query=raw_query,
            reasoning_type=reasoning_type,
            is_decomposed=False,
            sub_queries=[
                SubQuery(sub_query_id="q1", text=raw_query, strategy_hint=None)
            ],
            difficulty_signals=signals,
            classification_confidence=0.6,  # heuristic is less confident
        )

    def _make_oracle_plan(
        self, query_id: str, raw_query: str, oracle_reasoning_type: str
    ) -> QueryPlan:
        """Build a minimal QueryPlan from oracle type (no LLM)."""
        return QueryPlan(
            query_id=query_id,
            raw_query=raw_query,
            reasoning_type=oracle_reasoning_type,
            is_decomposed=False,
            sub_queries=[
                SubQuery(sub_query_id="q1", text=raw_query, strategy_hint=None)
            ],
            difficulty_signals=[],
            classification_confidence=1.0,
        )


# ── Module-level singleton (lazy) ──────────────────────────────────────────────

_agent: QAAgent | None = None


def get_qa_agent() -> QAAgent:
    """Return a module-level QAAgent, constructing once via auto-detected provider."""
    global _agent
    if _agent is None:
        try:
            from enterprise_rag.planning.providers.factory import create_provider
            provider = create_provider(prompt_version="structured")
        except Exception:
            provider = None
        _agent = QAAgent(provider=provider)
    return _agent
