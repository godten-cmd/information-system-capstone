"""Retrieval evaluation metrics: Recall@K, Precision@K, MRR, nDCG@K.

All metrics are computed from saved retrieval result lists, not in-memory objects,
to ensure reproducibility and allow re-evaluation without re-running retrieval.

Metric definitions:
- Recall@K:    |relevant ∩ retrieved@K| / |relevant|
- Precision@K: |relevant ∩ retrieved@K| / K
- MRR:         1 / rank_of_first_relevant  (0 if no relevant in top-K)
- nDCG@K:      DCG@K / IDCG@K  (graded relevance: 1 if relevant, 0 otherwise)
- Hit@K:       1 if at least one relevant in top-K, else 0
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """Per-query evaluation metrics for a single K value."""

    query_id: str
    k: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit: float
    n_relevant: int
    n_retrieved_relevant: int
    # context
    category: str = ""
    reasoning_type: str = ""
    difficulty: str = ""
    oracle_strategy: str = ""
    difficulty_factors: list[str] = field(default_factory=list)
    answerable: bool = True


@dataclass
class AggregateMetrics:
    """Aggregate metrics over a set of queries at a specific K."""

    k: int
    n_queries: int
    recall: float
    precision: float
    mrr: float
    ndcg: float
    hit_rate: float
    group_key: str = "overall"
    group_value: str = "all"


# ── Core metric functions ─────────────────────────────────────────────────────


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Recall@K: fraction of relevant items retrieved in top K."""
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Precision@K: fraction of top-K retrieved items that are relevant."""
    if k == 0:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank contribution: 1/rank of first relevant result (0 if none)."""
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """nDCG@K with binary relevance (1 if relevant, 0 otherwise)."""
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, r in enumerate(retrieved[:k], start=1)
        if r in relevant
    )

    # Ideal DCG: best possible ranking — all relevant items at top positions
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, n_ideal + 1))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Hit@K: 1.0 if any relevant item is in top K, else 0.0."""
    return 1.0 if any(r in relevant for r in retrieved[:k]) else 0.0


# ── Per-query evaluation ──────────────────────────────────────────────────────


def evaluate_query(
    query_id: str,
    retrieved_chunk_ids: list[str],
    required_chunk_ids: list[str],
    k: int,
    query_meta: dict | None = None,
) -> QueryMetrics:
    """Compute all metrics for one query at a given K.

    Args:
        query_id: Query identifier.
        retrieved_chunk_ids: Ordered list of retrieved chunk IDs (best first).
        required_chunk_ids: Ground-truth relevant chunk IDs.
        k: Cutoff for @K metrics.
        query_meta: Optional dict with category, reasoning_type, etc.

    Returns:
        QueryMetrics with all computed values.
    """
    relevant = set(required_chunk_ids)
    meta = query_meta or {}

    return QueryMetrics(
        query_id=query_id,
        k=k,
        recall=recall_at_k(retrieved_chunk_ids, relevant, k),
        precision=precision_at_k(retrieved_chunk_ids, relevant, k),
        mrr=mrr(retrieved_chunk_ids, relevant),
        ndcg=ndcg_at_k(retrieved_chunk_ids, relevant, k),
        hit=hit_at_k(retrieved_chunk_ids, relevant, k),
        n_relevant=len(relevant),
        n_retrieved_relevant=sum(1 for r in retrieved_chunk_ids[:k] if r in relevant),
        category=meta.get("category", ""),
        reasoning_type=meta.get("reasoning_type", ""),
        difficulty=meta.get("difficulty", ""),
        oracle_strategy=meta.get("planner_oracle_strategy", ""),
        difficulty_factors=meta.get("retrieval_difficulty_factors", []),
        answerable=meta.get("answerable", True),
    )


# ── Aggregate helpers ─────────────────────────────────────────────────────────


def aggregate(
    query_metrics: list[QueryMetrics],
    k: int,
    group_key: str = "overall",
    group_value: str = "all",
) -> AggregateMetrics:
    """Compute mean metrics over a list of per-query results."""
    if not query_metrics:
        return AggregateMetrics(k=k, n_queries=0, recall=0.0, precision=0.0, mrr=0.0, ndcg=0.0, hit_rate=0.0, group_key=group_key, group_value=group_value)
    n = len(query_metrics)
    return AggregateMetrics(
        k=k,
        n_queries=n,
        recall=sum(m.recall for m in query_metrics) / n,
        precision=sum(m.precision for m in query_metrics) / n,
        mrr=sum(m.mrr for m in query_metrics) / n,
        ndcg=sum(m.ndcg for m in query_metrics) / n,
        hit_rate=sum(m.hit for m in query_metrics) / n,
        group_key=group_key,
        group_value=group_value,
    )


def group_by(
    query_metrics: list[QueryMetrics],
    attr: str,
    k: int,
) -> dict[str, AggregateMetrics]:
    """Group metrics by an attribute and aggregate each group."""
    groups: dict[str, list[QueryMetrics]] = defaultdict(list)
    for m in query_metrics:
        value = getattr(m, attr, "unknown") or "unknown"
        groups[str(value)].append(m)
    return {v: aggregate(ms, k=k, group_key=attr, group_value=v) for v, ms in groups.items()}


def group_by_difficulty_factor(
    query_metrics: list[QueryMetrics],
    k: int,
) -> dict[str, AggregateMetrics]:
    """Group by each retrieval difficulty factor (one query can appear in multiple groups)."""
    factor_map: dict[str, list[QueryMetrics]] = defaultdict(list)
    for m in query_metrics:
        if m.difficulty_factors:
            for factor in m.difficulty_factors:
                factor_map[factor].append(m)
        else:
            factor_map["none"].append(m)
    return {f: aggregate(ms, k=k, group_key="difficulty_factor", group_value=f) for f, ms in factor_map.items()}


# ── Full evaluation pipeline ──────────────────────────────────────────────────


def evaluate_run(
    retrieval_results: list[dict],
    ground_truths: dict[str, list[str]],
    queries_meta: dict[str, dict],
    k_values: list[int] = (1, 3, 5, 10),
    answerable_only: bool = True,
) -> dict[int, list[QueryMetrics]]:
    """Evaluate a full retrieval run at multiple K values.

    Args:
        retrieval_results: List of dicts with query_id, chunk_id, rank, score.
        ground_truths: Map from query_id to list of required_chunk_ids.
        queries_meta: Map from query_id to query metadata dict.
        k_values: K cutoffs to evaluate at.
        answerable_only: If True, skip unanswerable queries (no required chunks).

    Returns:
        Dict mapping K → list of QueryMetrics.
    """
    # Group retrieval results by query_id, sorted by rank
    by_query: dict[str, list[dict]] = defaultdict(list)
    for r in retrieval_results:
        by_query[r["query_id"]].append(r)
    for qid in by_query:
        by_query[qid].sort(key=lambda r: r["rank"])

    result_by_k: dict[int, list[QueryMetrics]] = {k: [] for k in k_values}

    for query_id, required_chunks in ground_truths.items():
        meta = queries_meta.get(query_id, {})
        answerable = meta.get("answerable", True)

        if answerable_only and not answerable:
            continue
        if not required_chunks:
            continue

        retrieved = [r["chunk_id"] for r in by_query.get(query_id, [])]

        for k in k_values:
            qm = evaluate_query(
                query_id=query_id,
                retrieved_chunk_ids=retrieved,
                required_chunk_ids=required_chunks,
                k=k,
                query_meta=meta,
            )
            result_by_k[k].append(qm)

    return result_by_k
