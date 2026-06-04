"""V2 pipeline evaluation runner.

Wraps the LangGraph pipeline so it can be driven over a query batch and produce
retrieval result dicts compatible with the existing evaluate_run() infrastructure.

Experiment configs:
  v2_oracle          oracle QA + adaptive scorer + re-retrieval (max 3)
  v2_oracle_noloop   oracle QA + adaptive scorer, no re-retrieval (max 0)
  v2_oracle_rankproxy oracle QA + rank_proxy scorer + re-retrieval
  v2_heuristic       heuristic QA + adaptive scorer + re-retrieval
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Experiment configuration ───────────────────────────────────────────────────


@dataclass
class V2ExperimentConfig:
    """Configuration for one V2 evaluation experiment."""

    name: str
    description: str
    use_oracle: bool = True     # pass ground-truth reasoning_type to bypass LLM QA
    scorer: str = "adaptive"    # validation scorer: rank_proxy | bm25 | dense | adaptive | llm
    max_loops: int = 3          # max re-retrieval attempts (0 = no re-retrieval)
    k: int = 10                 # top-K chunks to keep per query


EXPERIMENTS: list[V2ExperimentConfig] = [
    V2ExperimentConfig(
        name="v2_oracle",
        description="V2 oracle QA + adaptive scorer + re-retrieval (max 3)",
        use_oracle=True,
        scorer="adaptive",
        max_loops=3,
    ),
    V2ExperimentConfig(
        name="v2_oracle_noloop",
        description="V2 oracle QA + adaptive scorer, no re-retrieval (ablation)",
        use_oracle=True,
        scorer="adaptive",
        max_loops=0,
    ),
    V2ExperimentConfig(
        name="v2_oracle_rankproxy",
        description="V2 oracle QA + rank_proxy scorer + re-retrieval (ablation)",
        use_oracle=True,
        scorer="rank_proxy",
        max_loops=3,
    ),
    V2ExperimentConfig(
        name="v2_heuristic",
        description="V2 heuristic QA + adaptive scorer + re-retrieval (ablation)",
        use_oracle=False,
        scorer="adaptive",
        max_loops=3,
    ),
]


# ── Chunk extraction helpers ───────────────────────────────────────────────────


def extract_chunks_from_state(
    final_state: dict[str, Any],
    query_id: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """Extract and rank retrieved chunks from the final pipeline state.

    Takes all chunks from all sub-query RetrievalOutputs in the final state,
    deduplicates by chunk_id (keeping highest score), sorts descending by score,
    and returns the top-k as dicts compatible with evaluate_run().

    Args:
        final_state: The AgentState dict returned by graph.invoke().
        query_id:    Used to tag each result dict.
        k:           Max chunks to return.

    Returns:
        List of dicts: {query_id, chunk_id, document_id, rank, score, category,
                        section_path, text, strategy}.
    """
    outputs = final_state.get("retrieval_outputs") or []
    seen: dict[str, dict[str, Any]] = {}

    for output in outputs:
        strategy = getattr(output, "strategy", "unknown")
        for chunk in getattr(output, "chunks", []):
            cid = chunk.chunk_id
            score = float(chunk.score)
            if cid not in seen or score > seen[cid]["score"]:
                seen[cid] = {
                    "query_id": query_id,
                    "chunk_id": cid,
                    "document_id": chunk.document_id or "",
                    "rank": chunk.rank,
                    "score": score,
                    "category": chunk.category or "",
                    "section_path": getattr(chunk, "section_path", []) or [],
                    "text": chunk.text or "",
                    "strategy": strategy,
                }

    ranked = sorted(seen.values(), key=lambda c: -c["score"])[:k]
    for i, c in enumerate(ranked, start=1):
        c["rank"] = i
    return ranked


def extract_pipeline_stats(final_state: dict[str, Any]) -> dict[str, Any]:
    """Extract V2-specific pipeline statistics from the final state."""
    val = final_state.get("validation_result")
    return {
        "loop_count": final_state.get("loop_count", 0),
        "validation_passed": bool(val and val.passed),
        "validation_passed_count": val.passed_count if val else 0,
        "validation_total_count": val.total_count if val else 0,
        "failure_reason": (val.failure_reason if val else "") or "",
        "suggested_strategy": (val.suggested_strategy if val else "") or "",
        "error": final_state.get("error") or "",
    }


# ── V2Runner ───────────────────────────────────────────────────────────────────


class V2Runner:
    """Run the V2 LangGraph pipeline over a query batch for evaluation.

    Manages module-level singletons and the MAX_LOOPS setting so that different
    experiment configs can be evaluated cleanly in the same process.

    Usage:
        runner = V2Runner(config)
        results, stats = runner.run_batch(queries)
        # results → list[dict] for evaluate_run()
        # stats   → per-query pipeline statistics
    """

    def __init__(self, config: V2ExperimentConfig) -> None:
        self.config = config
        self._graph: Any = None

    def setup(self) -> None:
        """Apply experiment configuration to module-level settings."""
        # Scorer for ValidationAgent
        os.environ["V2_VALIDATION_SCORER"] = self.config.scorer

        # Reset agent singletons so they rebuild with new settings
        import enterprise_rag.agents.validation as vm
        vm._agent = None

        import enterprise_rag.agents.answer as am
        am._agent = None

        import enterprise_rag.agents.planning as pm
        pm._agent = None

        import enterprise_rag.agents.query_analysis as qm
        qm._agent = None

        # Override re-retrieval loop limit
        import enterprise_rag.graph.edges as em
        em.MAX_LOOPS = self.config.max_loops

        # Build a fresh graph
        from enterprise_rag.graph.builder import build_graph
        self._graph = build_graph()

    def run_single(self, query: dict[str, Any]) -> tuple[list[dict], dict]:
        """Run one query through the V2 pipeline.

        Returns:
            (retrieval_results, pipeline_stats) where retrieval_results is
            a list of dicts ready for evaluate_run().
        """
        if self._graph is None:
            self.setup()

        from enterprise_rag.agents.state import make_initial_state

        metadata: dict[str, Any] = {}
        if self.config.use_oracle:
            metadata["use_oracle"] = True
            metadata["reasoning_type"] = query.get("reasoning_type", "single_hop")

        state = make_initial_state(
            query_id=query["query_id"],
            raw_query=query["query"],
            query_metadata=metadata,
        )

        try:
            final_state = self._graph.invoke(state)
        except Exception as exc:
            logger.warning("Query %s failed: %s", query["query_id"], exc)
            return [], {
                "loop_count": 0,
                "validation_passed": False,
                "validation_passed_count": 0,
                "validation_total_count": 0,
                "failure_reason": "pipeline_error",
                "suggested_strategy": "",
                "error": f"pipeline_error: {exc}",
            }

        results = extract_chunks_from_state(final_state, query["query_id"], k=self.config.k)
        stats = extract_pipeline_stats(final_state)
        return results, stats

    def run_batch(
        self,
        queries: list[dict[str, Any]],
        progress_fn: Any = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Run all queries. Returns (flat_results, per_query_stats).

        Args:
            queries:     List of query dicts from queries.jsonl.
            progress_fn: Optional callable(done, total) for progress updates.
        """
        if self._graph is None:
            self.setup()

        flat_results: list[dict[str, Any]] = []
        per_query_stats: list[dict[str, Any]] = []
        total = len(queries)

        for i, q in enumerate(queries):
            results, stats = self.run_single(q)
            flat_results.extend(results)
            per_query_stats.append({
                "query_id": q["query_id"],
                "reasoning_type": q.get("reasoning_type", ""),
                **stats,
            })
            if progress_fn:
                progress_fn(i + 1, total)

        return flat_results, per_query_stats

    def teardown(self) -> None:
        """Reset module state after experiment (allows clean re-runs)."""
        import enterprise_rag.agents.validation as vm
        vm._agent = None
        import enterprise_rag.agents.answer as am
        am._agent = None
        import enterprise_rag.agents.planning as pm
        pm._agent = None
        import enterprise_rag.agents.query_analysis as qm
        qm._agent = None
        self._graph = None


# ── Pipeline statistics aggregation ───────────────────────────────────────────


def aggregate_pipeline_stats(per_query_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate pipeline statistics over all queries."""
    if not per_query_stats:
        return {}

    n = len(per_query_stats)
    loop_counts = [s["loop_count"] for s in per_query_stats]
    val_passed = [s["validation_passed"] for s in per_query_stats]
    errors = [s for s in per_query_stats if s.get("error")]

    from collections import Counter
    loop_dist = Counter(loop_counts)
    failure_reasons = Counter(
        s["failure_reason"]
        for s in per_query_stats
        if not s["validation_passed"] and s["failure_reason"]
    )

    rt_groups: dict[str, list[dict]] = {}
    for s in per_query_stats:
        rt = s.get("reasoning_type", "unknown")
        rt_groups.setdefault(rt, []).append(s)

    rt_stats = {}
    for rt, group in rt_groups.items():
        rt_stats[rt] = {
            "n": len(group),
            "val_pass_rate": round(sum(1 for s in group if s["validation_passed"]) / len(group), 4),
            "mean_loops": round(sum(s["loop_count"] for s in group) / len(group), 3),
        }

    return {
        "n_queries": n,
        "validation_pass_rate": round(sum(val_passed) / n, 4),
        "mean_loops": round(sum(loop_counts) / n, 3),
        "n_errors": len(errors),
        "loop_distribution": {str(k): v for k, v in sorted(loop_dist.items())},
        "top_failure_reasons": dict(failure_reasons.most_common(5)),
        "by_reasoning_type": rt_stats,
    }
