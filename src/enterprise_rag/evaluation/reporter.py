"""Output writers for retrieval evaluation runs (BM25, dense, hybrid).

Writes: metrics.json, per_category.csv, per_reasoning_type.csv,
retrieval_results.jsonl, failure_analysis.json, comparison_vs_bm25.json.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from enterprise_rag.evaluation.retrieval_metrics import (
    AggregateMetrics,
    QueryMetrics,
    aggregate,
    group_by,
    group_by_difficulty_factor,
)
from enterprise_rag.retrieval.result_schema import RetrievalRun


# ── metrics.json ──────────────────────────────────────────────────────────────


def _agg_to_dict(agg: AggregateMetrics) -> dict[str, Any]:
    return {
        "n_queries": agg.n_queries,
        "recall": round(agg.recall, 4),
        "precision": round(agg.precision, 4),
        "mrr": round(agg.mrr, 4),
        "ndcg": round(agg.ndcg, 4),
        "hit_rate": round(agg.hit_rate, 4),
    }


def write_metrics_json(
    result_by_k: dict[int, list[QueryMetrics]],
    run: RetrievalRun,
    output_dir: Path,
) -> Path:
    """Write overall and grouped aggregate metrics to metrics.json."""
    k_values = sorted(result_by_k.keys())

    overall: dict[str, Any] = {}
    by_difficulty_factor: dict[str, Any] = {}

    for k in k_values:
        qms = result_by_k[k]
        overall[f"k{k}"] = _agg_to_dict(aggregate(qms, k=k))
        factor_groups = group_by_difficulty_factor(qms, k=k)
        by_difficulty_factor[f"k{k}"] = {
            factor: _agg_to_dict(agg)
            for factor, agg in sorted(factor_groups.items())
        }

    output: dict[str, Any] = {
        "method": run.method,
        "config": run.config,
        "timestamp": run.timestamp,
        "k_values": k_values,
        "overall": overall,
        "by_difficulty_factor": by_difficulty_factor,
    }

    path = output_dir / "metrics.json"
    path.write_text(json.dumps(output, indent=2))
    return path


# ── CSV helpers ───────────────────────────────────────────────────────────────


_CSV_FIELDS = ["group_value", "k", "n_queries", "recall", "precision", "mrr", "ndcg", "hit_rate"]


def _write_grouped_csv(
    result_by_k: dict[int, list[QueryMetrics]],
    attr: str,
    path: Path,
) -> Path:
    rows: list[dict[str, Any]] = []
    for k in sorted(result_by_k.keys()):
        groups = group_by(result_by_k[k], attr=attr, k=k)
        for group_value, agg in sorted(groups.items()):
            rows.append(
                {
                    "group_value": group_value,
                    "k": k,
                    "n_queries": agg.n_queries,
                    "recall": round(agg.recall, 4),
                    "precision": round(agg.precision, 4),
                    "mrr": round(agg.mrr, 4),
                    "ndcg": round(agg.ndcg, 4),
                    "hit_rate": round(agg.hit_rate, 4),
                }
            )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_per_category_csv(
    result_by_k: dict[int, list[QueryMetrics]],
    output_dir: Path,
) -> Path:
    return _write_grouped_csv(result_by_k, attr="category", path=output_dir / "per_category.csv")


def write_per_reasoning_type_csv(
    result_by_k: dict[int, list[QueryMetrics]],
    output_dir: Path,
) -> Path:
    return _write_grouped_csv(result_by_k, attr="reasoning_type", path=output_dir / "per_reasoning_type.csv")


# ── retrieval_results.jsonl ───────────────────────────────────────────────────


def write_retrieval_results_jsonl(run: RetrievalRun, output_dir: Path) -> Path:
    path = output_dir / "retrieval_results.jsonl"
    with open(path, "w") as f:
        for chunk in run.results:
            f.write(json.dumps(chunk.to_dict()) + "\n")
    return path


# ── failure_analysis.json ─────────────────────────────────────────────────────


def write_failure_analysis(
    result_by_k: dict[int, list[QueryMetrics]],
    queries_by_id: dict[str, dict],
    run: RetrievalRun,
    output_dir: Path,
    top_n: int = 20,
    analysis_k: int = 10,
) -> Path:
    """Identify worst-performing queries and compute failure patterns."""
    qms = result_by_k.get(analysis_k, [])

    # Sort by recall ASC, then mrr ASC, then n_relevant DESC (most severe failures first)
    sorted_qms = sorted(qms, key=lambda m: (m.recall, m.mrr, -m.n_relevant))

    # Build retrieved chunk ID lookup from the run
    retrieved_by_query: dict[str, list[str]] = {}
    for chunk in run.results:
        retrieved_by_query.setdefault(chunk.query_id, [])
        retrieved_by_query[chunk.query_id].append(chunk.chunk_id)

    top_failures = []
    for m in sorted_qms[:top_n]:
        q_meta = queries_by_id.get(m.query_id, {})
        top_failures.append(
            {
                "query_id": m.query_id,
                "query": q_meta.get("query", ""),
                "recall_at_k": round(m.recall, 4),
                "mrr": round(m.mrr, 4),
                "ndcg": round(m.ndcg, 4),
                "n_relevant": m.n_relevant,
                "n_retrieved_relevant": m.n_retrieved_relevant,
                "category": m.category,
                "reasoning_type": m.reasoning_type,
                "difficulty": m.difficulty,
                "oracle_strategy": m.oracle_strategy,
                "difficulty_factors": m.difficulty_factors,
                "required_chunk_ids": q_meta.get("required_chunk_ids", []),
                "retrieved_chunk_ids": retrieved_by_query.get(m.query_id, [])[:analysis_k],
            }
        )

    # Pattern analysis
    def _failure_stats(group: list[QueryMetrics]) -> dict[str, Any]:
        failed = [m for m in group if m.recall == 0.0]
        partial = [m for m in group if 0.0 < m.recall < 1.0]
        perfect = [m for m in group if m.recall == 1.0]
        avg_recall = sum(m.recall for m in group) / len(group) if group else 0.0
        return {
            "total": len(group),
            "zero_recall": len(failed),
            "partial_recall": len(partial),
            "perfect_recall": len(perfect),
            "zero_recall_rate": round(len(failed) / len(group), 4) if group else 0.0,
            "avg_recall": round(avg_recall, 4),
        }

    by_reasoning: dict[str, list[QueryMetrics]] = {}
    by_difficulty: dict[str, list[QueryMetrics]] = {}
    by_factor: dict[str, list[QueryMetrics]] = {}
    by_category: dict[str, list[QueryMetrics]] = {}

    for m in qms:
        by_reasoning.setdefault(m.reasoning_type or "unknown", []).append(m)
        by_difficulty.setdefault(m.difficulty or "unknown", []).append(m)
        by_category.setdefault(m.category or "unknown", []).append(m)
        if m.difficulty_factors:
            for f in m.difficulty_factors:
                by_factor.setdefault(f, []).append(m)
        else:
            by_factor.setdefault("none", []).append(m)

    output = {
        "k": analysis_k,
        "total_queries": len(qms),
        "total_zero_recall": sum(1 for m in qms if m.recall == 0.0),
        "total_perfect_recall": sum(1 for m in qms if m.recall == 1.0),
        "overall_avg_recall": round(sum(m.recall for m in qms) / len(qms), 4) if qms else 0.0,
        "top_failures": top_failures,
        "failure_patterns": {
            "by_reasoning_type": {k: _failure_stats(v) for k, v in sorted(by_reasoning.items())},
            "by_difficulty": {k: _failure_stats(v) for k, v in sorted(by_difficulty.items())},
            "by_category": {k: _failure_stats(v) for k, v in sorted(by_category.items())},
            "by_difficulty_factor": {k: _failure_stats(v) for k, v in sorted(by_factor.items())},
        },
    }

    path = output_dir / "failure_analysis.json"
    path.write_text(json.dumps(output, indent=2))
    return path


# ── comparison_vs_baseline.json ───────────────────────────────────────────────


def write_comparison_json(
    result_by_k: dict[int, list[QueryMetrics]],
    run: RetrievalRun,
    baseline_metrics_path: Path,
    output_dir: Path,
    baseline_name: str = "bm25",
) -> Path:
    """Write side-by-side comparison of the current run vs a baseline.

    Args:
        result_by_k: Current run's per-K QueryMetrics.
        run: Current RetrievalRun (for method name / config).
        baseline_metrics_path: Path to the baseline's metrics.json.
        output_dir: Where to write the output file.
        baseline_name: Label for the baseline (used in field names and filename).
    """
    out_path = output_dir / f"comparison_vs_{baseline_name}.json"
    if not baseline_metrics_path.exists():
        return out_path  # skip silently if baseline not present

    with open(baseline_metrics_path) as f:
        baseline = json.load(f)

    k_values = sorted(result_by_k.keys())
    metric_keys = ["recall", "precision", "mrr", "ndcg", "hit_rate"]
    current_name = run.method

    def _delta(current_val: float, base_val: float) -> float:
        return round(current_val - base_val, 4)

    # Overall comparison
    overall_cmp: dict[str, Any] = {}
    for k in k_values:
        qms = result_by_k[k]
        agg = aggregate(qms, k=k)
        base_k = baseline.get("overall", {}).get(f"k{k}", {})
        overall_cmp[f"k{k}"] = {
            "n_queries": agg.n_queries,
            **{
                m: {
                    current_name: round(getattr(agg, m), 4),
                    baseline_name: round(base_k.get(m, 0.0), 4),
                    "delta": _delta(getattr(agg, m), base_k.get(m, 0.0)),
                }
                for m in metric_keys
            },
        }

    max_k = max(k_values)
    cat_groups = group_by(result_by_k[max_k], attr="category", k=max_k)
    rt_groups = group_by(result_by_k[max_k], attr="reasoning_type", k=max_k)

    def _group_cmp(current_groups: dict[str, AggregateMetrics], base_csv: Path) -> dict[str, Any]:
        base_by_group: dict[str, dict] = {}
        if base_csv.exists():
            import csv as csv_mod
            with open(base_csv) as f:
                for row in csv_mod.DictReader(f):
                    if int(row["k"]) == max_k:
                        base_by_group[row["group_value"]] = {
                            m: float(row[m]) for m in metric_keys
                        }
        result = {}
        for gv, agg in sorted(current_groups.items()):
            base_g = base_by_group.get(gv, {})
            result[gv] = {
                "n_queries": agg.n_queries,
                **{
                    m: {
                        current_name: round(getattr(agg, m), 4),
                        baseline_name: round(base_g.get(m, 0.0), 4),
                        "delta": _delta(getattr(agg, m), base_g.get(m, 0.0)),
                    }
                    for m in metric_keys
                },
            }
        return result

    base_dir = baseline_metrics_path.parent
    cat_cmp = _group_cmp(cat_groups, base_dir / "per_category.csv")
    rt_cmp = _group_cmp(rt_groups, base_dir / "per_reasoning_type.csv")

    wins = [
        {"group": gv, "type": "category",
         "recall_delta": cat_cmp[gv]["recall"]["delta"]}
        for gv in cat_cmp if cat_cmp[gv]["recall"]["delta"] > 0.05
    ]
    losses = [
        {"group": gv, "type": "category",
         "recall_delta": cat_cmp[gv]["recall"]["delta"]}
        for gv in cat_cmp if cat_cmp[gv]["recall"]["delta"] < -0.03
    ]
    rt_wins = [
        {"group": gv, "type": "reasoning_type",
         "recall_delta": rt_cmp[gv]["recall"]["delta"]}
        for gv in rt_cmp if rt_cmp[gv]["recall"]["delta"] > 0.05
    ]
    rt_losses = [
        {"group": gv, "type": "reasoning_type",
         "recall_delta": rt_cmp[gv]["recall"]["delta"]}
        for gv in rt_cmp if rt_cmp[gv]["recall"]["delta"] < -0.03
    ]

    output: dict[str, Any] = {
        "current_method": current_name,
        "current_config": run.config,
        "current_timestamp": run.timestamp,
        "baseline_name": baseline_name,
        "baseline_timestamp": baseline.get("timestamp", ""),
        "k_values": k_values,
        "overall": overall_cmp,
        f"by_category_k{max_k}": cat_cmp,
        f"by_reasoning_type_k{max_k}": rt_cmp,
        "notable_improvements": sorted(wins + rt_wins, key=lambda x: -x["recall_delta"]),
        "notable_regressions": sorted(losses + rt_losses, key=lambda x: x["recall_delta"]),
    }

    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# ── Convenience: write all outputs at once ────────────────────────────────────


def write_all_outputs(
    result_by_k: dict[int, list[QueryMetrics]],
    run: RetrievalRun,
    queries_by_id: dict[str, dict],
    output_dir: Path,
    bm25_metrics_path: Path | None = None,
    dense_metrics_path: Path | None = None,
) -> dict[str, Path]:
    """Write all evaluation output files and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_json": write_metrics_json(result_by_k, run, output_dir),
        "per_category_csv": write_per_category_csv(result_by_k, output_dir),
        "per_reasoning_type_csv": write_per_reasoning_type_csv(result_by_k, output_dir),
        "retrieval_results_jsonl": write_retrieval_results_jsonl(run, output_dir),
        "failure_analysis_json": write_failure_analysis(result_by_k, queries_by_id, run, output_dir),
    }
    if bm25_metrics_path is not None:
        paths["comparison_vs_bm25_json"] = write_comparison_json(
            result_by_k, run, bm25_metrics_path, output_dir, baseline_name="bm25"
        )
    if dense_metrics_path is not None:
        paths["comparison_vs_dense_json"] = write_comparison_json(
            result_by_k, run, dense_metrics_path, output_dir, baseline_name="dense"
        )
    return paths
