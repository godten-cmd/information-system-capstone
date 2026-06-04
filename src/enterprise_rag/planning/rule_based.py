"""Rule-Based Retrieval Planner for enterprise RAG.

Selects one of {bm25, dense, hybrid} for each query using seven interpretable
rules based on reasoning_type, retrieval_difficulty_factors, and category.

Rules are evaluated in priority order; the first matching rule wins.

Rule table (for academic paper):
┌─────┬───────────────────────────────────────────────────────────────────────────────────┬──────────┬────────────┐
│ ID  │ Condition                                                                         │ Strategy │ Confidence │
├─────┼───────────────────────────────────────────────────────────────────────────────────┼──────────┼────────────┤
│ R01 │ reasoning_type == 'temporal' OR 'temporal_reasoning' ∈ factors                   │ bm25     │ 0.80       │
│ R02 │ reasoning_type == 'exception' OR 'exception_handling' ∈ factors                  │ dense    │ 0.85       │
│ R03 │ reasoning_type == 'multi_hop' OR 'multi_document_dependency' ∈ factors           │ hybrid   │ 0.90       │
│ R04 │ 'table_dependency' ∈ factors                                                     │ hybrid   │ 0.75       │
│ R05 │ reasoning_type ∈ {'comparison', 'aggregation'}                                   │ hybrid   │ 0.75       │
│ R06 │ category ∈ {SysDesign, API Docs} AND reasoning_type == 'single_hop'              │ dense    │ 0.70       │
│ R07 │ (default)                                                                         │ hybrid   │ 0.60       │
└─────┴───────────────────────────────────────────────────────────────────────────────────┴──────────┴────────────┘

Empirical rationale:
  R01: BM25 Recall@10=0.200 vs Hybrid=0.100 vs Dense=0.000 for temporal queries.
       Oracle labels temporal as 'multi_hop' (→hybrid) but empirical evidence contradicts this.
  R02: Oracle=dense for 100% of exception queries; semantic matching captures clause nuance.
  R03: Multi-hop evidence requires lexical IDs (BM25) + semantic context (Dense) together.
  R04: Hybrid Recall@10=0.927 for table_dependency; exact values + context both needed.
  R05: Hybrid Recall@10=0.849 (comparison), 0.914 (aggregation) — coverage is critical.
  R06: Dense outperforms BM25 by +18.9% on System Design; +2.9% on API Documentation.
  R07: Hybrid is the best fixed baseline: Recall@10=0.745, MRR=0.530, nDCG=0.566.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from enterprise_rag.planning.schema import (
    AVAILABLE_STRATEGIES,
    ORACLE_DISCREPANCY_NOTES,
    ORACLE_STRATEGY_MAP,
    PlannerDecision,
)

_SYS_DESIGN_CATEGORIES = {"System Design Documents", "API Documentation"}


@dataclass(frozen=True)
class Rule:
    """A single deterministic rule in the rule-based planner."""

    rule_id: str
    name: str
    description: str
    strategy: str   # bm25 | dense | hybrid
    confidence: float


# Ordered rule list — evaluated top-to-bottom; first match wins.
RULES: list[Rule] = [
    Rule(
        rule_id="R01",
        name="temporal_to_bm25",
        description=(
            "Temporal reasoning: queries about dates, time periods, or event sequences. "
            "BM25 matches date tokens and temporal keywords more reliably than dense retrieval."
        ),
        strategy="bm25",
        confidence=0.80,
    ),
    Rule(
        rule_id="R02",
        name="exception_to_dense",
        description=(
            "Exception clause interpretation: queries about conditions or exceptions to rules. "
            "Exception clauses are phrased differently from policy headings; "
            "semantic matching is more robust than keyword matching."
        ),
        strategy="dense",
        confidence=0.85,
    ),
    Rule(
        rule_id="R03",
        name="multi_hop_to_hybrid",
        description=(
            "Multi-hop or multi-document queries: evidence spans multiple documents. "
            "Requires lexical matching (BM25) for document IDs and codes plus "
            "semantic matching (Dense) for content; hybrid RRF fusion covers both."
        ),
        strategy="hybrid",
        confidence=0.90,
    ),
    Rule(
        rule_id="R04",
        name="table_to_hybrid",
        description=(
            "Table value lookup: queries requiring exact numeric values from tables. "
            "Hybrid retrieval achieves Recall@10=0.927 on table_dependency queries; "
            "exact-value BM25 component plus Dense context improves coverage."
        ),
        strategy="hybrid",
        confidence=0.75,
    ),
    Rule(
        rule_id="R05",
        name="comparison_aggregation_to_hybrid",
        description=(
            "Comparison or aggregation queries: require coverage across related chunks. "
            "Hybrid fusion maximizes recall for spanning queries "
            "(Recall@10: comparison=0.849, aggregation=0.914)."
        ),
        strategy="hybrid",
        confidence=0.75,
    ),
    Rule(
        rule_id="R06",
        name="technical_single_hop_to_dense",
        description=(
            "Single-hop technical queries in System Design or API Documentation. "
            "Dense retrieval outperforms BM25 by +18.9% on System Design and "
            "+2.9% on API Documentation; architecture and API queries benefit "
            "from semantic similarity."
        ),
        strategy="dense",
        confidence=0.70,
    ),
    Rule(
        rule_id="R07",
        name="default_to_hybrid",
        description=(
            "Default fallback: no more specific rule matched. "
            "Hybrid is the best fixed baseline overall "
            "(Recall@10=0.745, MRR=0.530, nDCG=0.566)."
        ),
        strategy="hybrid",
        confidence=0.60,
    ),
]

# Lookup by rule_id
RULES_BY_ID: dict[str, Rule] = {r.rule_id: r for r in RULES}


def _matches_r01(reasoning_type: str, factors: list[str]) -> bool:
    return reasoning_type == "temporal" or "temporal_reasoning" in factors


def _matches_r02(reasoning_type: str, factors: list[str]) -> bool:
    return reasoning_type == "exception" or "exception_handling" in factors


def _matches_r03(reasoning_type: str, factors: list[str]) -> bool:
    return reasoning_type == "multi_hop" or "multi_document_dependency" in factors


def _matches_r04(factors: list[str]) -> bool:
    return "table_dependency" in factors


def _matches_r05(reasoning_type: str) -> bool:
    return reasoning_type in {"comparison", "aggregation"}


def _matches_r06(category: str, reasoning_type: str) -> bool:
    return category in _SYS_DESIGN_CATEGORIES and reasoning_type == "single_hop"


def _apply_rules(
    reasoning_type: str,
    factors: list[str],
    category: str,
) -> tuple[Rule, list[str]]:
    """Apply rules in priority order. Returns (winning_rule, all_matched_rule_ids)."""
    matched: list[str] = []

    if _matches_r01(reasoning_type, factors):
        matched.append("R01")
    if _matches_r02(reasoning_type, factors):
        matched.append("R02")
    if _matches_r03(reasoning_type, factors):
        matched.append("R03")
    if _matches_r04(factors):
        matched.append("R04")
    if _matches_r05(reasoning_type):
        matched.append("R05")
    if _matches_r06(category, reasoning_type):
        matched.append("R06")

    if matched:
        winning_rule = RULES_BY_ID[matched[0]]
        return winning_rule, matched
    else:
        return RULES_BY_ID["R07"], ["R07"]


class RuleBasedPlanner:
    """Deterministic retrieval strategy planner using interpretable rules.

    Routes each query to exactly one of {bm25, dense, hybrid} based on
    reasoning_type, retrieval_difficulty_factors, and document category.
    """

    planner_type: str = "rule_based"

    def plan(self, query: dict[str, Any]) -> PlannerDecision:
        """Select a retrieval strategy for a single query."""
        query_id = query["query_id"]
        reasoning_type = query.get("reasoning_type", "") or ""
        factors: list[str] = query.get("retrieval_difficulty_factors", []) or []
        category = query.get("category", "") or ""
        oracle_raw = query.get("planner_oracle_strategy", "hybrid") or "hybrid"
        oracle_source = query.get("oracle_strategy_source", "") or ""

        winning_rule, matched = _apply_rules(reasoning_type, factors, category)
        oracle_mapped = ORACLE_STRATEGY_MAP.get(oracle_raw, oracle_raw)

        discrepancy = (
            reasoning_type in ORACLE_DISCREPANCY_NOTES
            and oracle_source != "empirical_validation"
        )
        note = ORACLE_DISCREPANCY_NOTES.get(reasoning_type, "")

        return PlannerDecision(
            query_id=query_id,
            selected_strategy=winning_rule.strategy,
            matched_rules=matched,
            planner_confidence=winning_rule.confidence,
            oracle_strategy=oracle_raw,
            oracle_strategy_mapped=oracle_mapped,
            oracle_strategy_source=oracle_source,
            is_correct=winning_rule.strategy == oracle_mapped,
            reasoning_type=reasoning_type,
            category=category,
            difficulty_factors=factors,
            oracle_discrepancy=discrepancy,
            discrepancy_note=note,
        )

    def plan_batch(
        self,
        queries: list[dict[str, Any]],
    ) -> tuple[list[PlannerDecision], float]:
        """Plan all queries and return (decisions, total_seconds)."""
        t0 = time.monotonic()
        decisions = [self.plan(q) for q in queries]
        elapsed = time.monotonic() - t0
        return decisions, elapsed

    def describe_rules(self) -> list[dict[str, Any]]:
        """Return human-readable rule descriptions for reporting."""
        return [
            {
                "rule_id": r.rule_id,
                "name": r.name,
                "strategy": r.strategy,
                "confidence": r.confidence,
                "description": r.description,
            }
            for r in RULES
        ]
