"""Schemas for retrieval planner decisions (Rule-Based and LLM-Based).

Available retrieval backends: bm25 | dense | hybrid
Oracle vocabulary (from dataset): keyword | dense | hybrid | table_aware |
    multi_hop | metadata_filtered | hierarchical
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The three retrieval backends we actually have.
AVAILABLE_STRATEGIES: frozenset[str] = frozenset({"bm25", "dense", "hybrid"})

# Extended oracle vocabulary → available backend mapping.
# Derived from:
#   keyword → BM25 (exact lexical matching)
#   hierarchical → Dense (section-level semantic retrieval)
#   table_aware → Hybrid (exact values + context; Hybrid Recall@10=0.927 on table queries)
#   multi_hop → Hybrid (lexical IDs + semantic context across documents)
#   metadata_filtered → Hybrid (best approximation without metadata filters)
ORACLE_STRATEGY_MAP: dict[str, str] = {
    "keyword": "bm25",
    "dense": "dense",
    "hybrid": "hybrid",
    "table_aware": "hybrid",
    "multi_hop": "hybrid",
    "metadata_filtered": "hybrid",
    "hierarchical": "dense",
}

# Queries where empirical retrieval results contradict the mapped oracle label.
# Source: Phase 5–7 evaluation. These are flagged as oracle discrepancies.
ORACLE_DISCREPANCY_NOTES: dict[str, str] = {
    "temporal": (
        "Oracle=multi_hop→hybrid, but BM25 outperforms Hybrid for temporal queries "
        "(Recall@10: BM25=0.200 > Hybrid=0.100 > Dense=0.000). "
        "Empirical evidence supports bm25 for this reasoning type."
    ),
}


@dataclass
class PlannerDecision:
    """A single planner decision for one query.

    Fields required for Phase 8 evaluation and oracle validation framework.
    """

    query_id: str
    selected_strategy: str                  # bm25 | dense | hybrid
    matched_rules: list[str]                # rule IDs that fired, in priority order
    planner_confidence: float               # confidence of the selected strategy
    oracle_strategy: str                    # original extended strategy from query metadata
    oracle_strategy_mapped: str             # mapped to bm25 | dense | hybrid
    oracle_strategy_source: str             # heuristic_assignment | manual_annotation | empirical_validation
    is_correct: bool                        # selected_strategy == oracle_strategy_mapped
    reasoning_type: str = ""
    category: str = ""
    difficulty_factors: list[str] = field(default_factory=list)
    oracle_discrepancy: bool = False        # empirical evidence contradicts oracle
    discrepancy_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "selected_strategy": self.selected_strategy,
            "matched_rules": self.matched_rules,
            "planner_confidence": round(self.planner_confidence, 4),
            "oracle_strategy": self.oracle_strategy,
            "oracle_strategy_mapped": self.oracle_strategy_mapped,
            "oracle_strategy_source": self.oracle_strategy_source,
            "is_correct": self.is_correct,
            "reasoning_type": self.reasoning_type,
            "category": self.category,
            "difficulty_factors": self.difficulty_factors,
            "oracle_discrepancy": self.oracle_discrepancy,
            "discrepancy_note": self.discrepancy_note,
        }


# ── Cost model ────────────────────────────────────────────────────────────────

# USD per 1 M tokens (input / output)
MODEL_COST_PER_M: dict[str, dict[str, float]] = {
    "gpt-5": {"input": 10.00, "output": 40.00},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "mock": {"input": 0.00, "output": 0.00},
}

_FALLBACK_COST = {"input": 2.00, "output": 8.00}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Return estimated USD cost for a single call."""
    rates = MODEL_COST_PER_M.get(model, _FALLBACK_COST)
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


# ── LLMDecision ───────────────────────────────────────────────────────────────


@dataclass
class LLMDecision:
    """A retrieval strategy decision produced by the LLM-based planner.

    Shares the same interface as PlannerDecision so it can be used with
    compute_planner_metrics() and group_decisions_by() without changes.
    """

    # ── Core (same interface as PlannerDecision) ──────────────────────────────
    query_id: str
    selected_strategy: str
    matched_rules: list[str]          # always [] for LLM planner
    planner_confidence: float
    oracle_strategy: str
    oracle_strategy_mapped: str
    oracle_strategy_source: str
    is_correct: bool
    reasoning_type: str = ""
    category: str = ""
    difficulty_factors: list[str] = field(default_factory=list)
    oracle_discrepancy: bool = False
    discrepancy_note: str = ""

    # ── LLM-specific ──────────────────────────────────────────────────────────
    prompt_version: str = ""
    model: str = ""
    raw_response: str = ""
    parsed_reasoning: str = ""
    parse_attempts: int = 1
    parse_success: bool = True
    fallback_used: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    estimated_cost_usd: float = 0.0
    error_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "selected_strategy": self.selected_strategy,
            "matched_rules": self.matched_rules,
            "planner_confidence": round(self.planner_confidence, 4),
            "oracle_strategy": self.oracle_strategy,
            "oracle_strategy_mapped": self.oracle_strategy_mapped,
            "oracle_strategy_source": self.oracle_strategy_source,
            "is_correct": self.is_correct,
            "reasoning_type": self.reasoning_type,
            "category": self.category,
            "difficulty_factors": self.difficulty_factors,
            "oracle_discrepancy": self.oracle_discrepancy,
            "discrepancy_note": self.discrepancy_note,
            "prompt_version": self.prompt_version,
            "model": self.model,
            "raw_response": self.raw_response,
            "parsed_reasoning": self.parsed_reasoning,
            "parse_attempts": self.parse_attempts,
            "parse_success": self.parse_success,
            "fallback_used": self.fallback_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": round(self.latency_s, 4),
            "estimated_cost_usd": round(self.estimated_cost_usd, 8),
            "error_category": self.error_category,
        }
