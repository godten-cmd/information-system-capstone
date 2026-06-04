"""Distribution configuration and assignment for query label targets."""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class DistributionConfig:
    """Target distribution percentages and minimum counts for query generation."""

    total_queries: int = 400
    min_per_category: int = 40
    min_unanswerable: int = 25
    min_multi_hop: int = 50

    difficulty_targets: dict[str, float] = field(
        default_factory=lambda: {
            "easy": 0.30,
            "medium": 0.35,
            "hard": 0.25,
            "adversarial": 0.10,
        }
    )
    reasoning_targets: dict[str, float] = field(
        default_factory=lambda: {
            "single_hop": 0.40,
            "multi_hop": 0.16,
            "comparison": 0.13,
            "temporal": 0.06,
            "exception": 0.13,
            "aggregation": 0.12,
        }
    )
    oracle_strategy_targets: dict[str, float] = field(
        default_factory=lambda: {
            "keyword": 0.15,
            "dense": 0.15,
            "hybrid": 0.30,
            "metadata_filtered": 0.15,
            "hierarchical": 0.10,
            "table_aware": 0.10,
            "multi_hop": 0.05,
        }
    )


def adjust_oracle_strategy_distribution(
    raw_queries: list,
    config: DistributionConfig,
    rng: random.Random,
) -> None:
    """
    Reassign oracle_strategy labels on flexible queries to approximate targets.

    Modifies raw_queries in-place. Only changes labels when reassignment is
    defensible (keyword → hybrid, table_aware → metadata_filtered, etc.).
    Sets oracle_strategy_source to "heuristic_assignment" for changed queries.
    """
    answerable = [q for q in raw_queries if q.answerable]
    if not answerable:
        return

    total = len(answerable)
    targets = {k: int(v * total) for k, v in config.oracle_strategy_targets.items()}

    # Compute current counts
    current = Counter(q.oracle_strategy.value for q in answerable)

    # Map from strategy name → list of indices in answerable that could accept it
    # Flexibility rules:
    #   keyword → hybrid: any policy/factual query (not table-specific)
    #   table_aware → metadata_filtered: any table query that also needs category filter
    #   dense → hybrid: any semantic query with specific terminology
    _KEYWORD_TO_HYBRID = {"keyword"}
    _TABLE_TO_META = {"table_aware"}
    _DENSE_TO_HYBRID = {"dense"}

    overrepresented = [k for k, v in current.items() if v > targets.get(k, 0)]
    underrepresented = [k for k, v in targets.items() if current.get(k, 0) < v]

    for src in overrepresented:
        excess = current[src] - targets.get(src, 0)
        if excess <= 0:
            continue

        # Find candidate queries with this strategy that can be reassigned
        candidates = [q for q in answerable if q.oracle_strategy.value == src]
        rng.shuffle(candidates)

        for dst in underrepresented:
            if excess <= 0:
                break
            deficit = targets.get(dst, 0) - current.get(dst, 0)
            if deficit <= 0:
                continue

            # Check compatibility
            compatible = False
            if src in _KEYWORD_TO_HYBRID and dst == "hybrid":
                compatible = True
            elif src in _TABLE_TO_META and dst == "metadata_filtered":
                compatible = True
            elif src in _DENSE_TO_HYBRID and dst == "hybrid":
                compatible = True

            if not compatible:
                continue

            from .enums import OracleStrategy, OracleStrategySource  # noqa: PLC0415

            for q in candidates[:min(deficit, excess)]:
                try:
                    q.oracle_strategy = OracleStrategy(dst)
                    q.oracle_strategy_source = OracleStrategySource.HEURISTIC_ASSIGNMENT
                    current[src] -= 1
                    current[dst] = current.get(dst, 0) + 1
                    excess -= 1
                    deficit -= 1
                except ValueError:
                    pass
