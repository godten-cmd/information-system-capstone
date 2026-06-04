"""Contribution analysis: quantify how much BM25 and Dense each contribute to Hybrid.

For each required chunk, classify it into:
  bm25_only   — found by BM25@k but not Dense@k
  dense_only  — found by Dense@k but not BM25@k
  both        — found by both BM25@k and Dense@k
  hybrid_only — found by Hybrid@k but by neither BM25@k nor Dense@k
                (RRF promotes a chunk ranked >k in both to top-k)
  not_found   — not found by any method at @k

Complementarity score = (bm25_only + dense_only) / total_required_chunks
  High score → methods are highly complementary (each covers unique ground).
  Low score  → both methods find the same chunks (diminishing returns from fusion).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from enterprise_rag.retrieval.result_schema import RetrievalRun


@dataclass
class ChunkContribution:
    """Per-chunk classification."""

    chunk_id: str
    query_id: str
    source: str  # bm25_only | dense_only | both | hybrid_only | not_found


@dataclass
class ContributionStats:
    """Aggregate contribution counts for a group of queries."""

    n_total_required: int = 0
    n_bm25_only: int = 0
    n_dense_only: int = 0
    n_both: int = 0
    n_hybrid_only: int = 0
    n_not_found: int = 0
    n_queries: int = 0

    @property
    def complementarity_score(self) -> float:
        if self.n_total_required == 0:
            return 0.0
        return (self.n_bm25_only + self.n_dense_only) / self.n_total_required

    @property
    def hybrid_gain(self) -> float:
        """Fraction of required chunks found only after fusion (not by either alone at @k)."""
        if self.n_total_required == 0:
            return 0.0
        return self.n_hybrid_only / self.n_total_required

    def to_dict(self) -> dict[str, Any]:
        total = self.n_total_required or 1
        return {
            "n_queries": self.n_queries,
            "n_total_required_chunks": self.n_total_required,
            "bm25_only": {"count": self.n_bm25_only, "pct": round(self.n_bm25_only / total * 100, 1)},
            "dense_only": {"count": self.n_dense_only, "pct": round(self.n_dense_only / total * 100, 1)},
            "both": {"count": self.n_both, "pct": round(self.n_both / total * 100, 1)},
            "hybrid_only": {"count": self.n_hybrid_only, "pct": round(self.n_hybrid_only / total * 100, 1)},
            "not_found": {"count": self.n_not_found, "pct": round(self.n_not_found / total * 100, 1)},
            "complementarity_score": round(self.complementarity_score, 4),
            "hybrid_gain": round(self.hybrid_gain, 4),
        }


def _found_set(run: RetrievalRun, query_id: str, k: int) -> set[str]:
    """Set of chunk_ids in the top-k results for a given query."""
    results = run.results_for(query_id)[:k]
    return {r.chunk_id for r in results}


def analyze_contributions(
    ground_truths: dict[str, list[str]],
    bm25_run: RetrievalRun,
    dense_run: RetrievalRun,
    hybrid_run: RetrievalRun,
    queries_meta: dict[str, dict],
    k: int = 10,
    answerable_only: bool = True,
) -> dict[str, Any]:
    """Classify every required chunk by which method(s) found it.

    Returns a dict with:
      - 'overall': ContributionStats for all queries
      - 'by_reasoning_type': dict[reasoning_type → ContributionStats]
      - 'by_category': dict[category → ContributionStats]
      - 'per_query': list of per-query contribution dicts
    """
    overall = ContributionStats()
    by_rt: dict[str, ContributionStats] = defaultdict(ContributionStats)
    by_cat: dict[str, ContributionStats] = defaultdict(ContributionStats)
    per_query: list[dict[str, Any]] = []

    for query_id, required_ids in ground_truths.items():
        meta = queries_meta.get(query_id, {})
        answerable = meta.get("answerable", True)
        if answerable_only and not answerable:
            continue
        if not required_ids:
            continue

        bm25_found = _found_set(bm25_run, query_id, k)
        dense_found = _found_set(dense_run, query_id, k)
        hybrid_found = _found_set(hybrid_run, query_id, k)

        rt = meta.get("reasoning_type", "unknown") or "unknown"
        cat = meta.get("category", "unknown") or "unknown"

        q_stats = {
            "bm25_only": 0, "dense_only": 0, "both": 0,
            "hybrid_only": 0, "not_found": 0,
        }

        for cid in required_ids:
            in_bm25 = cid in bm25_found
            in_dense = cid in dense_found
            in_hybrid = cid in hybrid_found

            if in_bm25 and in_dense:
                source = "both"
            elif in_bm25 and not in_dense:
                source = "bm25_only"
            elif in_dense and not in_bm25:
                source = "dense_only"
            elif in_hybrid and not in_bm25 and not in_dense:
                source = "hybrid_only"
            else:
                source = "not_found"

            q_stats[source] += 1
            # accumulate
            setattr(overall, f"n_{source}", getattr(overall, f"n_{source}") + 1)
            setattr(by_rt[rt], f"n_{source}", getattr(by_rt[rt], f"n_{source}") + 1)
            setattr(by_cat[cat], f"n_{source}", getattr(by_cat[cat], f"n_{source}") + 1)
            overall.n_total_required += 1
            by_rt[rt].n_total_required += 1
            by_cat[cat].n_total_required += 1

        overall.n_queries += 1
        by_rt[rt].n_queries += 1
        by_cat[cat].n_queries += 1

        per_query.append({
            "query_id": query_id,
            "n_required": len(required_ids),
            **q_stats,
            "reasoning_type": rt,
            "category": cat,
        })

    return {
        "k": k,
        "overall": overall.to_dict(),
        "by_reasoning_type": {rt: stats.to_dict() for rt, stats in sorted(by_rt.items())},
        "by_category": {cat: stats.to_dict() for cat, stats in sorted(by_cat.items())},
        "per_query": per_query,
        "interpretation": _interpret(overall),
    }


def _interpret(stats: ContributionStats) -> str:
    cs = stats.complementarity_score
    hg = stats.hybrid_gain
    if cs > 0.30:
        comp = "high complementarity — BM25 and Dense cover substantially different queries"
    elif cs > 0.15:
        comp = "moderate complementarity — methods partially overlap in what they retrieve"
    else:
        comp = "low complementarity — both methods largely retrieve the same chunks"

    if hg > 0.05:
        rescue = f"; hybrid fusion rescues {hg*100:.1f}% of chunks that neither method found alone"
    else:
        rescue = f"; hybrid fusion provides minimal additional coverage ({hg*100:.1f}% hybrid-only)"

    return comp + rescue
