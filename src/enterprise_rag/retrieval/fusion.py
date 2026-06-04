"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists.

RRF score for item at rank r across a list:  1 / (rrf_k + r)
Final score = sum of individual list scores.

Reference: Cormack et al. (2009) "Reciprocal Rank Fusion Outperforms
Condorcet and Individual Rank Learning Methods", SIGIR 2009.
"""

from __future__ import annotations

from collections import defaultdict


def rrf_score(rank: int, rrf_k: int) -> float:
    """Contribution of one ranked list item at position `rank` (1-based)."""
    return 1.0 / (rrf_k + rank)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists of chunk IDs using RRF.

    Args:
        ranked_lists: Each inner list is an ordered sequence of chunk_ids,
                      best-first (index 0 = rank 1).
        rrf_k: RRF smoothing constant. Higher values make scoring more
               uniform (less sensitive to exact rank position).
               Common choices: 10, 30, 60, 100.
        top_n: If provided, return only the top-n results. Otherwise
               return all unique items.

    Returns:
        List of (chunk_id, rrf_score) sorted by score descending.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, chunk_id in enumerate(ranked, start=1):
            scores[chunk_id] += rrf_score(rank, rrf_k)

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    if top_n is not None:
        fused = fused[:top_n]
    return fused


def fuse_runs(
    runs_and_ids: list[tuple[list[str], str]],
    rrf_k: int = 60,
    top_n: int | None = None,
) -> list[tuple[str, float]]:
    """Fuse multiple per-method ranked lists.

    Args:
        runs_and_ids: List of (ranked_chunk_ids, method_name) pairs.
        rrf_k: RRF smoothing constant.
        top_n: Maximum results to return.

    Returns:
        List of (chunk_id, rrf_score) sorted descending.
    """
    return reciprocal_rank_fusion(
        ranked_lists=[r for r, _ in runs_and_ids],
        rrf_k=rrf_k,
        top_n=top_n,
    )
