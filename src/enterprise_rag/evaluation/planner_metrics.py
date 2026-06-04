"""Planner evaluation metrics: Strategy Selection Accuracy, Precision, Recall.

Computes per-strategy, per-reasoning-type, and per-category planner metrics.
Also computes per-rule contribution analysis.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from enterprise_rag.planning.schema import AVAILABLE_STRATEGIES, PlannerDecision


@dataclass
class PlannerMetrics:
    """Aggregate planner accuracy metrics."""

    n_queries: int
    n_correct: int
    n_discrepancy: int          # oracle label flagged as empirically contradicted
    accuracy: float             # n_correct / n_queries
    accuracy_ex_discrepancy: float  # accuracy excluding discrepancy queries

    # Per-strategy: Precision = queries correctly routed / queries routed to strategy
    #               Recall    = queries correctly routed / total queries with that oracle
    precision_by_strategy: dict[str, float]
    recall_by_strategy: dict[str, float]
    f1_by_strategy: dict[str, float]
    support_by_strategy: dict[str, int]     # oracle count per strategy
    predicted_by_strategy: dict[str, int]   # predicted count per strategy

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_queries": self.n_queries,
            "n_correct": self.n_correct,
            "n_discrepancy": self.n_discrepancy,
            "accuracy": round(self.accuracy, 4),
            "accuracy_excluding_discrepancy": round(self.accuracy_ex_discrepancy, 4),
            "per_strategy": {
                s: {
                    "precision": round(self.precision_by_strategy.get(s, 0.0), 4),
                    "recall": round(self.recall_by_strategy.get(s, 0.0), 4),
                    "f1": round(self.f1_by_strategy.get(s, 0.0), 4),
                    "oracle_support": self.support_by_strategy.get(s, 0),
                    "predicted": self.predicted_by_strategy.get(s, 0),
                }
                for s in sorted(AVAILABLE_STRATEGIES)
            },
        }


def compute_planner_metrics(decisions: list[PlannerDecision]) -> PlannerMetrics:
    """Compute overall planner accuracy and per-strategy precision/recall."""
    n = len(decisions)
    if n == 0:
        empty: dict[str, float] = {s: 0.0 for s in sorted(AVAILABLE_STRATEGIES)}
        empty_int: dict[str, int] = {s: 0 for s in sorted(AVAILABLE_STRATEGIES)}
        return PlannerMetrics(
            n_queries=0, n_correct=0, n_discrepancy=0,
            accuracy=0.0, accuracy_ex_discrepancy=0.0,
            precision_by_strategy=empty, recall_by_strategy=empty,
            f1_by_strategy=empty, support_by_strategy=empty_int,
            predicted_by_strategy=empty_int,
        )

    n_correct = sum(1 for d in decisions if d.is_correct)
    n_discrepancy = sum(1 for d in decisions if d.oracle_discrepancy)

    # Decisions excluding oracle-discrepancy queries
    non_disc = [d for d in decisions if not d.oracle_discrepancy]
    n_nd_correct = sum(1 for d in non_disc if d.is_correct)
    acc_excl = n_nd_correct / len(non_disc) if non_disc else 0.0

    # True positives, false positives, false negatives per strategy
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    support: dict[str, int] = defaultdict(int)
    predicted: dict[str, int] = defaultdict(int)

    for d in decisions:
        support[d.oracle_strategy_mapped] += 1
        predicted[d.selected_strategy] += 1
        if d.is_correct:
            tp[d.selected_strategy] += 1
        else:
            fp[d.selected_strategy] += 1
            fn[d.oracle_strategy_mapped] += 1

    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}

    for s in AVAILABLE_STRATEGIES:
        p = tp[s] / (tp[s] + fp[s]) if (tp[s] + fp[s]) > 0 else 0.0
        r = tp[s] / (tp[s] + fn[s]) if (tp[s] + fn[s]) > 0 else 0.0
        precision[s] = p
        recall[s] = r
        f1[s] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    return PlannerMetrics(
        n_queries=n,
        n_correct=n_correct,
        n_discrepancy=n_discrepancy,
        accuracy=n_correct / n,
        accuracy_ex_discrepancy=acc_excl,
        precision_by_strategy=precision,
        recall_by_strategy=recall,
        f1_by_strategy=f1,
        support_by_strategy=dict(support),
        predicted_by_strategy=dict(predicted),
    )


def group_decisions_by(
    decisions: list[PlannerDecision],
    attr: str,
) -> dict[str, "GroupedPlannerMetrics"]:
    """Compute planner metrics grouped by an attribute (reasoning_type or category)."""
    groups: dict[str, list[PlannerDecision]] = defaultdict(list)
    for d in decisions:
        key = getattr(d, attr, "unknown") or "unknown"
        groups[key].append(d)
    return {
        k: _group_metrics(k, v)
        for k, v in sorted(groups.items())
    }


@dataclass
class GroupedPlannerMetrics:
    group_value: str
    n_queries: int
    n_correct: int
    accuracy: float
    strategy_distribution: dict[str, int]  # selected strategy counts


def _group_metrics(group_value: str, decisions: list[PlannerDecision]) -> GroupedPlannerMetrics:
    n = len(decisions)
    n_correct = sum(1 for d in decisions if d.is_correct)
    dist: dict[str, int] = defaultdict(int)
    for d in decisions:
        dist[d.selected_strategy] += 1
    return GroupedPlannerMetrics(
        group_value=group_value,
        n_queries=n,
        n_correct=n_correct,
        accuracy=n_correct / n if n > 0 else 0.0,
        strategy_distribution=dict(dist),
    )


@dataclass
class RuleContribution:
    """Per-rule usage and accuracy statistics."""
    rule_id: str
    rule_name: str
    strategy: str
    usage_count: int           # queries where this rule was the winning rule
    accuracy: float            # fraction where selected == oracle_mapped
    n_correct: int


def compute_rule_contributions(decisions: list[PlannerDecision]) -> list[RuleContribution]:
    """Compute per-rule usage count and accuracy.

    'Winning rule' = matched_rules[0] (the highest-priority rule that fired).
    """
    from enterprise_rag.planning.rule_based import RULES_BY_ID

    usage: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)

    for d in decisions:
        if d.matched_rules:
            winning = d.matched_rules[0]
            usage[winning] += 1
            if d.is_correct:
                correct[winning] += 1

    from enterprise_rag.planning.rule_based import RULES
    return [
        RuleContribution(
            rule_id=r.rule_id,
            rule_name=r.name,
            strategy=r.strategy,
            usage_count=usage.get(r.rule_id, 0),
            accuracy=correct.get(r.rule_id, 0) / usage[r.rule_id]
            if usage.get(r.rule_id, 0) > 0 else 0.0,
            n_correct=correct.get(r.rule_id, 0),
        )
        for r in RULES
    ]


def strategy_distribution_report(decisions: list[PlannerDecision]) -> list[dict[str, Any]]:
    """Compute strategy distribution: count, %, and accuracy per selected strategy."""
    n = len(decisions)
    buckets: dict[str, list[PlannerDecision]] = defaultdict(list)
    for d in decisions:
        buckets[d.selected_strategy].append(d)

    rows = []
    for strategy in sorted(AVAILABLE_STRATEGIES):
        ds = buckets.get(strategy, [])
        count = len(ds)
        n_correct = sum(1 for d in ds if d.is_correct)
        rows.append({
            "selected_strategy": strategy,
            "count": count,
            "pct": round(count / n * 100, 1) if n > 0 else 0.0,
            "n_correct": n_correct,
            "accuracy": round(n_correct / count, 4) if count > 0 else 0.0,
        })
    return rows
