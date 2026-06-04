"""LLM-Based Retrieval Planner evaluation runner (Phase 9).

Loads pre-computed BM25, Dense, and Hybrid retrieval results, runs the
LLM-based planner for one or all three prompt versions, assembles planner
retrieval runs, evaluates retrieval metrics, and writes all outputs to
outputs/evaluation/llm_planner/.

Usage:
    # Default: mock provider, structured prompt
    python scripts/run_llm_planner.py

    # OpenAI (requires OPENAI_API_KEY env var)
    python scripts/run_llm_planner.py --provider openai --prompt-version structured

    # Anthropic
    python scripts/run_llm_planner.py --provider anthropic --model claude-haiku-4-5-20251001

    # Full prompt comparison study (runs all three prompt versions)
    python scripts/run_llm_planner.py --run-prompt-comparison

Reads:
    data/sekd/queries.jsonl
    data/sekd/ground_truth.jsonl
    outputs/evaluation/bm25/retrieval_results.jsonl
    outputs/evaluation/dense/retrieval_results.jsonl
    outputs/evaluation/hybrid/retrieval_results.jsonl
    outputs/evaluation/rule_planner/planner_decisions.jsonl  (for comparison)

Writes (outputs/evaluation/llm_planner/):
    planner_metrics.json
    retrieval_metrics.json
    strategy_distribution.csv
    prompt_comparison.csv
    failure_analysis.json
    cost_analysis.json
    comparison_vs_rule_planner.json
    llm_decisions.jsonl
    run_manifest.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Annotated, Any

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from enterprise_rag.evaluation.planner_metrics import (
    compute_planner_metrics,
    group_decisions_by,
    strategy_distribution_report,
)
from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
from enterprise_rag.evaluation.reporter import write_failure_analysis, write_metrics_json
from enterprise_rag.planning.llm_based import LLMBasedPlanner, run_prompt_comparison
from enterprise_rag.planning.providers.factory import create_provider
from enterprise_rag.planning.schema import LLMDecision, ORACLE_STRATEGY_MAP, estimate_cost
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk

app = typer.Typer(add_completion=False)

_RULE_PLANNER_SSA = 0.7842  # Phase 8 reference value (post-oracle-fix)
_RULE_PLANNER_RECALL10 = 0.7553


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_retrieval_run(path: Path, method: str) -> dict[str, list[dict]]:
    by_query: dict[str, list[dict]] = defaultdict(list)
    for rec in _load_jsonl(path):
        by_query[rec["query_id"]].append(rec)
    for qid in by_query:
        by_query[qid].sort(key=lambda r: r["rank"])
    return dict(by_query)


def _assemble_planner_run(
    decisions: list[LLMDecision],
    queries: list[dict],
    retrieval_results: dict[str, dict[str, list[dict]]],
    k: int,
    method_label: str = "llm_planner",
) -> RetrievalRun:
    decision_map = {d.query_id: d for d in decisions}
    run = RetrievalRun(
        method=method_label,
        k=k,
        config={"planner_type": "llm_based", "k": k},
    )
    for q in queries:
        qid = q["query_id"]
        dec = decision_map.get(qid)
        if dec is None:
            continue
        strat = dec.selected_strategy
        results_for_q = retrieval_results.get(strat, {}).get(qid, [])[:k]
        for rec in results_for_q:
            run.results.append(RetrievedChunk(
                query_id=qid,
                chunk_id=rec["chunk_id"],
                document_id=rec.get("document_id", ""),
                rank=rec["rank"],
                score=rec.get("score", 0.0),
                category=rec.get("category", ""),
                section_path=rec.get("section_path", []),
                text="",
            ))
    return run


# ── Metric writers ────────────────────────────────────────────────────────────


def _write_strategy_distribution(decisions: list[LLMDecision], path: Path) -> None:
    rows = strategy_distribution_report(decisions)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selected_strategy", "count", "pct", "n_correct", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)


def _write_prompt_comparison(
    all_decisions: dict[str, list[LLMDecision]],
    path: Path,
) -> None:
    rows = []
    for version, decisions in all_decisions.items():
        if not decisions:
            continue
        m = compute_planner_metrics(decisions)
        parse_fails = sum(1 for d in decisions if not d.parse_success)
        avg_prompt_tokens = mean(d.prompt_tokens for d in decisions) if decisions else 0
        avg_completion_tokens = mean(d.completion_tokens for d in decisions) if decisions else 0
        total_cost = sum(d.estimated_cost_usd for d in decisions)
        rows.append({
            "prompt_version": version,
            "n_queries": m.n_queries,
            "n_correct": m.n_correct,
            "accuracy": round(m.accuracy, 4),
            "parse_failures": parse_fails,
            "parse_failure_rate": round(parse_fails / m.n_queries, 4) if m.n_queries else 0,
            "bm25_precision": round(m.precision_by_strategy.get("bm25", 0.0), 4),
            "bm25_recall": round(m.recall_by_strategy.get("bm25", 0.0), 4),
            "bm25_f1": round(m.f1_by_strategy.get("bm25", 0.0), 4),
            "dense_precision": round(m.precision_by_strategy.get("dense", 0.0), 4),
            "dense_recall": round(m.recall_by_strategy.get("dense", 0.0), 4),
            "dense_f1": round(m.f1_by_strategy.get("dense", 0.0), 4),
            "hybrid_precision": round(m.precision_by_strategy.get("hybrid", 0.0), 4),
            "hybrid_recall": round(m.recall_by_strategy.get("hybrid", 0.0), 4),
            "hybrid_f1": round(m.f1_by_strategy.get("hybrid", 0.0), 4),
            "avg_prompt_tokens": round(avg_prompt_tokens, 1),
            "avg_completion_tokens": round(avg_completion_tokens, 1),
            "total_cost_usd": round(total_cost, 6),
        })
    fields = [
        "prompt_version", "n_queries", "n_correct", "accuracy",
        "parse_failures", "parse_failure_rate",
        "bm25_precision", "bm25_recall", "bm25_f1",
        "dense_precision", "dense_recall", "dense_f1",
        "hybrid_precision", "hybrid_recall", "hybrid_f1",
        "avg_prompt_tokens", "avg_completion_tokens", "total_cost_usd",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_planner_metrics(
    decisions: list[LLMDecision],
    result_by_k: dict[int, list],
    path: Path,
    prompt_version: str,
    model: str,
) -> None:
    m = compute_planner_metrics(decisions)
    by_rt = group_decisions_by(decisions, "reasoning_type")
    by_cat = group_decisions_by(decisions, "category")

    def _grp(g: Any) -> dict:
        return {
            "n": g.n_queries,
            "accuracy": round(g.accuracy, 4),
            "distribution": g.strategy_distribution,
        }

    parse_fails = sum(1 for d in decisions if not d.parse_success)
    fallbacks = sum(1 for d in decisions if d.fallback_used)

    output = {
        "planner_type": "llm_based",
        "prompt_version": prompt_version,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "parse_failures": parse_fails,
        "fallback_count": fallbacks,
        "overall": m.to_dict(),
        "by_reasoning_type": {k: _grp(v) for k, v in sorted(by_rt.items())},
        "by_category": {k: _grp(v) for k, v in sorted(by_cat.items())},
        "oracle_discrepancies": [d.to_dict() for d in decisions if d.oracle_discrepancy],
    }
    path.write_text(json.dumps(output, indent=2))


def _write_cost_analysis(
    all_decisions: dict[str, list[LLMDecision]],
    path: Path,
    rule_planner_latency_ms: float = 0.001,
) -> None:
    versions_data: dict[str, Any] = {}
    for version, decisions in all_decisions.items():
        if not decisions:
            continue
        latencies = [d.latency_s * 1000 for d in decisions]  # to ms
        costs = [d.estimated_cost_usd for d in decisions]
        prompt_toks = [d.prompt_tokens for d in decisions]
        completion_toks = [d.completion_tokens for d in decisions]
        model = decisions[0].model if decisions else "unknown"
        total_cost = sum(costs)

        versions_data[version] = {
            "model": model,
            "n_queries": len(decisions),
            "latency_ms": {
                "mean": round(mean(latencies), 3),
                "median": round(median(latencies), 3),
                "p95": round(sorted(latencies)[int(0.95 * len(latencies))], 3),
                "stdev": round(stdev(latencies) if len(latencies) > 1 else 0.0, 3),
            },
            "tokens": {
                "avg_prompt": round(mean(prompt_toks), 1),
                "avg_completion": round(mean(completion_toks), 1),
                "total_prompt": sum(prompt_toks),
                "total_completion": sum(completion_toks),
            },
            "cost_usd": {
                "total": round(total_cost, 6),
                "per_query_avg": round(total_cost / len(decisions), 8) if decisions else 0,
                "per_1000_queries": round(total_cost / len(decisions) * 1000, 4) if decisions else 0,
            },
        }

    # Rule planner reference
    rule_ref: dict[str, Any] = {
        "model": "rule_based",
        "latency_ms": {"mean": rule_planner_latency_ms, "note": "CPU-only deterministic rules"},
        "cost_usd": {"total": 0.0, "per_query_avg": 0.0},
    }

    # Compute latency overhead ratio
    if versions_data:
        best_version = max(versions_data, key=lambda v: versions_data[v]["n_queries"])
        llm_mean_ms = versions_data[best_version]["latency_ms"]["mean"]
        overhead_ratio = llm_mean_ms / rule_planner_latency_ms if rule_planner_latency_ms > 0 else float("inf")
    else:
        overhead_ratio = 0.0

    output: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rule_planner_reference": rule_ref,
        "llm_planner_by_prompt": versions_data,
        "latency_overhead_vs_rule_planner": round(overhead_ratio, 0),
        "note": (
            "Latency overhead shows how many times slower the LLM planner is vs the rule planner. "
            "Mock provider latency is ~0ms; real API latency is 200-2000ms per query."
        ),
    }
    path.write_text(json.dumps(output, indent=2))


def _write_comparison_vs_rule_planner(
    decisions: list[LLMDecision],
    result_by_k: dict[int, list],
    rule_planner_metrics: dict[str, Any] | None,
    rule_planner_retrieval: dict[str, Any] | None,
    output_dir: Path,
    k_values: list[int],
) -> Path:
    max_k = max(k_values)
    m = compute_planner_metrics(decisions)
    planner_ssa = m.accuracy
    rule_ssa = rule_planner_metrics.get("overall", {}).get("accuracy", _RULE_PLANNER_SSA) if rule_planner_metrics else _RULE_PLANNER_SSA

    overall_retr: dict[str, Any] = {}
    for k in k_values:
        qms = result_by_k[k]
        agg = aggregate(qms, k=k)
        rule_k = rule_planner_retrieval.get("overall", {}).get(f"k{k}", {}) if rule_planner_retrieval else {}
        overall_retr[f"k{k}"] = {
            "n_queries": agg.n_queries,
            "recall": {"llm_planner": round(agg.recall, 4), "rule_planner": round(rule_k.get("recall", 0.0), 4),
                       "delta": round(agg.recall - rule_k.get("recall", 0.0), 4)},
            "precision": {"llm_planner": round(agg.precision, 4), "rule_planner": round(rule_k.get("precision", 0.0), 4),
                          "delta": round(agg.precision - rule_k.get("precision", 0.0), 4)},
            "mrr": {"llm_planner": round(agg.mrr, 4), "rule_planner": round(rule_k.get("mrr", 0.0), 4),
                    "delta": round(agg.mrr - rule_k.get("mrr", 0.0), 4)},
            "ndcg": {"llm_planner": round(agg.ndcg, 4), "rule_planner": round(rule_k.get("ndcg", 0.0), 4),
                     "delta": round(agg.ndcg - rule_k.get("ndcg", 0.0), 4)},
            "hit_rate": {"llm_planner": round(agg.hit_rate, 4), "rule_planner": round(rule_k.get("hit_rate", 0.0), 4),
                         "delta": round(agg.hit_rate - rule_k.get("hit_rate", 0.0), 4)},
        }

    from enterprise_rag.evaluation.retrieval_metrics import group_by
    cat_groups = group_by(result_by_k[max_k], attr="category", k=max_k)
    rt_groups = group_by(result_by_k[max_k], attr="reasoning_type", k=max_k)

    def _grp_block(current_groups: dict, rule_csv: Path, group_key: str) -> dict:
        # Pre-load rule planner values from CSV if it exists.
        rule_vals: dict[str, dict[str, float]] = {}
        if rule_csv.exists():
            import csv as csv_mod
            with open(rule_csv) as f:
                for r in csv_mod.DictReader(f):
                    if int(r.get("k", 0)) == max_k:
                        gv_key = r.get("group_value", "")
                        rule_vals[gv_key] = {
                            m: float(r.get(m, 0.0))
                            for m in ["recall", "precision", "mrr", "ndcg", "hit_rate"]
                        }

        result: dict[str, Any] = {}
        for gv, agg in sorted(current_groups.items()):
            row: dict[str, Any] = {"n_queries": agg.n_queries}
            rule_row = rule_vals.get(gv)
            for metric in ["recall", "precision", "mrr", "ndcg", "hit_rate"]:
                curr = round(getattr(agg, metric), 4)
                if rule_row is not None:
                    rv = round(rule_row[metric], 4)
                    row[metric] = {"llm_planner": curr, "rule_planner": rv,
                                   "delta": round(curr - rv, 4)}
                else:
                    row[metric] = {"llm_planner": curr, "rule_planner": None, "delta": None}
            result[gv] = row
        return result

    rule_dir = output_dir.parent / "rule_planner"
    cat_cmp = _grp_block(cat_groups, rule_dir / "per_category.csv", "category")
    rt_cmp = _grp_block(rt_groups, rule_dir / "per_reasoning_type.csv", "reasoning_type")

    # Notable wins/regressions
    cat_wins = [{"group": g, "type": "category", "recall_delta": cat_cmp[g]["recall"]["delta"]}
                for g in cat_cmp if cat_cmp[g]["recall"]["delta"] is not None and cat_cmp[g]["recall"]["delta"] > 0.03]
    cat_losses = [{"group": g, "type": "category", "recall_delta": cat_cmp[g]["recall"]["delta"]}
                  for g in cat_cmp if cat_cmp[g]["recall"]["delta"] is not None and cat_cmp[g]["recall"]["delta"] < -0.03]
    rt_wins = [{"group": g, "type": "reasoning_type", "recall_delta": rt_cmp[g]["recall"]["delta"]}
               for g in rt_cmp if rt_cmp[g]["recall"]["delta"] is not None and rt_cmp[g]["recall"]["delta"] > 0.03]
    rt_losses = [{"group": g, "type": "reasoning_type", "recall_delta": rt_cmp[g]["recall"]["delta"]}
                 for g in rt_cmp if rt_cmp[g]["recall"]["delta"] is not None and rt_cmp[g]["recall"]["delta"] < -0.03]

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "planner_accuracy": {
            "llm_planner": round(planner_ssa, 4),
            "rule_planner": round(rule_ssa, 4),
            "delta": round(planner_ssa - rule_ssa, 4),
        },
        "overall_retrieval": overall_retr,
        f"by_category_k{max_k}": cat_cmp,
        f"by_reasoning_type_k{max_k}": rt_cmp,
        "notable_improvements": sorted(cat_wins + rt_wins, key=lambda x: -x["recall_delta"]),
        "notable_regressions": sorted(cat_losses + rt_losses, key=lambda x: x["recall_delta"]),
    }
    path = output_dir / "comparison_vs_rule_planner.json"
    path.write_text(json.dumps(out, indent=2))
    return path


def _write_llm_failure_analysis(
    decisions: list[LLMDecision],
    path: Path,
    top_n: int = 20,
) -> None:
    """Classify and summarise incorrect LLM planner decisions."""
    incorrect = [d for d in decisions if not d.is_correct]
    total = len(decisions)
    n_incorrect = len(incorrect)

    # Error category distribution
    by_cat: dict[str, list[LLMDecision]] = defaultdict(list)
    for d in incorrect:
        by_cat[d.error_category or "unknown"].append(d)

    by_rt: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for d in incorrect:
        by_rt[d.reasoning_type]["total"] += 1
        by_rt[d.reasoning_type][d.error_category or "unknown"] += 1

    by_category: dict[str, int] = defaultdict(int)
    for d in incorrect:
        by_category[d.category] += 1

    # Representative examples (top_n most common error patterns)
    examples = []
    for d in incorrect[:top_n]:
        examples.append({
            "query_id": d.query_id,
            "prompt_version": d.prompt_version,
            "selected_strategy": d.selected_strategy,
            "oracle_strategy_mapped": d.oracle_strategy_mapped,
            "oracle_strategy_source": d.oracle_strategy_source,
            "reasoning_type": d.reasoning_type,
            "category": d.category,
            "difficulty_factors": d.difficulty_factors,
            "error_category": d.error_category,
            "parsed_reasoning": d.parsed_reasoning,
            "parse_success": d.parse_success,
        })

    output: dict[str, Any] = {
        "total_queries": total,
        "total_incorrect": n_incorrect,
        "error_rate": round(n_incorrect / total, 4) if total else 0.0,
        "parse_failures": sum(1 for d in decisions if not d.parse_success),
        "fallback_used": sum(1 for d in decisions if d.fallback_used),
        "error_categories": {
            cat: {
                "count": len(ds),
                "pct": round(len(ds) / n_incorrect * 100, 1) if n_incorrect else 0.0,
                "description": _ERROR_DESCRIPTIONS.get(cat, cat),
            }
            for cat, ds in sorted(by_cat.items(), key=lambda x: -len(x[1]))
        },
        "errors_by_reasoning_type": {
            rt: dict(counts) for rt, counts in sorted(by_rt.items())
        },
        "errors_by_category": dict(sorted(by_category.items(), key=lambda x: -x[1])),
        "representative_examples": examples,
    }
    path.write_text(json.dumps(output, indent=2))


_ERROR_DESCRIPTIONS = {
    "prompt_ambiguity": "Zero-shot prompt lacks metadata; LLM could not infer strategy from text alone",
    "reasoning_confusion": "LLM misidentified the primary reasoning type (e.g., temporal or exception)",
    "difficulty_factor_confusion": "LLM missed a key difficulty factor (table_dependency or multi_document_dependency)",
    "category_confusion": "LLM did not use the document category signal correctly",
    "oracle_disagreement": "Both strategies are plausible; oracle label may be ambiguous for this query",
}


# ── Print helpers ─────────────────────────────────────────────────────────────


def _print_summary(
    decisions: list[LLMDecision],
    result_by_k: dict[int, list],
    prompt_version: str,
    model: str,
) -> None:
    m = compute_planner_metrics(decisions)
    agg10 = aggregate(result_by_k[10], k=10)
    parse_fails = sum(1 for d in decisions if not d.parse_success)
    total_cost = sum(d.estimated_cost_usd for d in decisions)

    width = 72
    sep = "─" * width
    typer.echo(f"\n{sep}")
    typer.echo(f"LLM Planner — Retrieval Summary  [{prompt_version} / {model}]")
    typer.echo(sep)
    header = f"{'k':>4}  {'Recall':>8}  {'Prec':>8}  {'MRR':>8}  {'nDCG':>8}  {'Hit':>8}  {'Queries':>8}"
    typer.echo(header)
    typer.echo(sep)
    for k in sorted(result_by_k.keys()):
        agg = aggregate(result_by_k[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:8.4f}  {agg.precision:8.4f}  {agg.mrr:8.4f}  {agg.ndcg:8.4f}  {agg.hit_rate:8.4f}  {agg.n_queries:>8}")
    typer.echo(sep)

    typer.echo(f"\n{sep}")
    typer.echo("Planner Strategy Selection Accuracy")
    typer.echo(sep)
    typer.echo(f"  Overall accuracy:              {m.accuracy:.4f} ({m.n_correct}/{m.n_queries})")
    typer.echo(f"  Parse failures:                {parse_fails}")
    typer.echo(f"  Total estimated cost (USD):    ${total_cost:.6f}")
    typer.echo(f"\n  {'Strategy':>8}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'Oracle':>8}  {'Predicted':>9}")
    for s in sorted(["bm25", "dense", "hybrid"]):
        typer.echo(
            f"  {s:>8}  {m.precision_by_strategy.get(s, 0):.4f}  "
            f"{m.recall_by_strategy.get(s, 0):.4f}  "
            f"{m.f1_by_strategy.get(s, 0):.4f}  "
            f"{m.support_by_strategy.get(s, 0):>8}  "
            f"{m.predicted_by_strategy.get(s, 0):>9}"
        )

    typer.echo(f"\n  vs. Rule Planner:")
    typer.echo(f"    SSA delta:      {m.accuracy - _RULE_PLANNER_SSA:+.4f}")
    typer.echo(f"    Recall@10 delta: {agg10.recall - _RULE_PLANNER_RECALL10:+.4f}")
    typer.echo(sep)


# ── Main command ──────────────────────────────────────────────────────────────


@app.command()
def main(
    queries_path: Annotated[Path, typer.Option()] = Path("data/sekd/queries.jsonl"),
    ground_truth_path: Annotated[Path, typer.Option()] = Path("data/sekd/ground_truth.jsonl"),
    bm25_results: Annotated[Path, typer.Option()] = Path("outputs/evaluation/bm25/retrieval_results.jsonl"),
    dense_results: Annotated[Path, typer.Option()] = Path("outputs/evaluation/dense/retrieval_results.jsonl"),
    hybrid_results: Annotated[Path, typer.Option()] = Path("outputs/evaluation/hybrid/retrieval_results.jsonl"),
    rule_planner_decisions: Annotated[Path, typer.Option()] = Path("outputs/evaluation/rule_planner/planner_decisions.jsonl"),
    rule_planner_metrics: Annotated[Path, typer.Option()] = Path("outputs/evaluation/rule_planner/planner_metrics.json"),
    rule_planner_retrieval: Annotated[Path, typer.Option()] = Path("outputs/evaluation/rule_planner/metrics.json"),
    output_dir: Annotated[Path, typer.Option()] = Path("outputs/evaluation/llm_planner"),
    cache_dir: Annotated[Path, typer.Option()] = Path("outputs/cache/planner"),
    provider: Annotated[str | None, typer.Option(help="openai | anthropic | mock")] = None,
    model: Annotated[str | None, typer.Option()] = None,
    prompt_version: Annotated[str, typer.Option(help="zero_shot | structured | few_shot")] = "structured",
    run_prompt_comparison: Annotated[bool, typer.Option(help="Run all 3 prompt versions.")] = False,
    k_values: Annotated[list[int], typer.Option()] = [1, 3, 5, 10],  # noqa: B006
    answerable_only: Annotated[bool, typer.Option()] = True,
) -> None:
    """Run LLM-Based Retrieval Planner, evaluate, and write all outputs."""
    t_start = time.monotonic()
    max_k = max(k_values)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    typer.echo(f"Loading queries from {queries_path} ...")
    queries = _load_jsonl(queries_path)
    typer.echo(f"  {len(queries)} queries")

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    for gt in _load_jsonl(ground_truth_path):
        ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])

    if answerable_only:
        queries = [q for q in queries if q.get("answerable", True) and ground_truths.get(q["query_id"])]
        typer.echo(f"  {len(queries)} answerable queries after filtering")

    queries_meta = {q["query_id"]: q for q in queries}

    # ── 2. Load pre-computed retrieval results ────────────────────────────────
    typer.echo("Loading pre-computed retrieval results ...")
    retrieval_runs: dict[str, dict[str, list[dict]]] = {}
    for method, path in [("bm25", bm25_results), ("dense", dense_results), ("hybrid", hybrid_results)]:
        if path.exists():
            retrieval_runs[method] = _load_retrieval_run(path, method)
            typer.echo(f"  {method}: {len(retrieval_runs[method])} queries from {path}")
        else:
            typer.echo(f"  WARNING: {path} not found")
    if len(retrieval_runs) < 3:
        typer.echo("ERROR: All three retrieval results required.", err=True)
        raise typer.Exit(1)

    # ── 3. Create provider ────────────────────────────────────────────────────
    typer.echo("Creating LLM provider ...")
    prov = create_provider(provider=provider, model=model, prompt_version=prompt_version)
    typer.echo(f"  Provider: {prov.provider_name} / Model: {prov.model_name}")

    # ── 4. Run planner (single or all prompt versions) ────────────────────────
    all_decisions: dict[str, list[LLMDecision]] = {}

    if run_prompt_comparison:
        typer.echo("Running prompt comparison (all 3 versions) ...")
        from enterprise_rag.planning.llm_based import run_prompt_comparison as _rpc
        all_decisions = _rpc(queries, prov, cache_dir=cache_dir)
        for v, ds in all_decisions.items():
            m = compute_planner_metrics(ds)
            typer.echo(f"  {v}: SSA={m.accuracy:.4f} ({m.n_correct}/{m.n_queries})")
        # Best = highest SSA
        best_version = max(all_decisions, key=lambda v: compute_planner_metrics(all_decisions[v]).accuracy)
        typer.echo(f"  Best prompt version: {best_version}")
    else:
        typer.echo(f"Running LLM planner [{prompt_version}] ...")
        planner = LLMBasedPlanner(
            provider=prov,
            prompt_version=prompt_version,
            cache_dir=cache_dir,
        )
        decisions, elapsed = planner.plan_batch(queries, verbose=True)
        typer.echo(f"  {len(decisions)} decisions in {elapsed*1000:.1f}ms")
        all_decisions = {prompt_version: decisions}
        best_version = prompt_version

    best_decisions = all_decisions[best_version]

    # ── 5. Assemble and evaluate retrieval run ────────────────────────────────
    typer.echo(f"Assembling retrieval run from {best_version} decisions ...")
    planner_run = _assemble_planner_run(best_decisions, queries, retrieval_runs, k=max_k)
    typer.echo(f"  {len(planner_run.results)} results assembled")

    typer.echo("Evaluating retrieval metrics ...")
    result_by_k = evaluate_run(
        retrieval_results=[r.to_dict() for r in planner_run.results],
        ground_truths=ground_truths,
        queries_meta=queries_meta,
        k_values=k_values,
        answerable_only=True,
    )

    # ── 6. Load rule planner reference for comparison ─────────────────────────
    rule_m_data = json.loads(rule_planner_metrics.read_text()) if rule_planner_metrics.exists() else None
    rule_r_data = json.loads(rule_planner_retrieval.read_text()) if rule_planner_retrieval.exists() else None

    # ── 7. Write all outputs ──────────────────────────────────────────────────
    typer.echo(f"Writing outputs to {output_dir} ...")

    # planner_metrics.json
    pm_path = output_dir / "planner_metrics.json"
    _write_planner_metrics(best_decisions, result_by_k, pm_path, best_version, prov.model_name)
    typer.echo(f"  planner_metrics: {pm_path}")

    # retrieval_metrics.json  (reuse the reporter's write_metrics_json)
    rm_path = write_metrics_json(result_by_k, planner_run, output_dir)
    typer.echo(f"  retrieval_metrics: {output_dir / 'metrics.json'}")

    # strategy_distribution.csv
    sd_path = output_dir / "strategy_distribution.csv"
    _write_strategy_distribution(best_decisions, sd_path)
    typer.echo(f"  strategy_distribution: {sd_path}")

    # prompt_comparison.csv
    pc_path = output_dir / "prompt_comparison.csv"
    _write_prompt_comparison(all_decisions, pc_path)
    typer.echo(f"  prompt_comparison: {pc_path}")

    # failure_analysis.json  (retrieval failures from the reporter)
    fa_path = write_failure_analysis(result_by_k, queries_meta, planner_run, output_dir)
    typer.echo(f"  retrieval_failure_analysis: {fa_path}")

    # llm_failure_analysis.json  (LLM decision errors)
    lfa_path = output_dir / "llm_failure_analysis.json"
    _write_llm_failure_analysis(best_decisions, lfa_path)
    typer.echo(f"  llm_failure_analysis: {lfa_path}")

    # cost_analysis.json
    ca_path = output_dir / "cost_analysis.json"
    _write_cost_analysis(all_decisions, ca_path)
    typer.echo(f"  cost_analysis: {ca_path}")

    # comparison_vs_rule_planner.json
    cmp_path = _write_comparison_vs_rule_planner(
        best_decisions, result_by_k, rule_m_data, rule_r_data,
        output_dir, k_values,
    )
    typer.echo(f"  comparison_vs_rule_planner: {cmp_path}")

    # llm_decisions.jsonl
    dec_path = output_dir / "llm_decisions.jsonl"
    with open(dec_path, "w") as f:
        for d in best_decisions:
            f.write(json.dumps(d.to_dict()) + "\n")
    typer.echo(f"  llm_decisions: {dec_path}")

    # per_category / per_reasoning_type CSVs (for downstream comparisons)
    from enterprise_rag.evaluation.reporter import write_per_category_csv, write_per_reasoning_type_csv
    write_per_category_csv(result_by_k, output_dir)
    write_per_reasoning_type_csv(result_by_k, output_dir)

    # run_manifest.json
    total_cost = sum(d.estimated_cost_usd for d in best_decisions)
    parse_fails = sum(1 for d in best_decisions if not d.parse_success)
    mf_path = output_dir / "run_manifest.json"
    manifest: dict[str, Any] = {
        "phase": "Phase 9 — LLM-Based Retrieval Planner",
        "planner_type": "llm_based",
        "provider": prov.provider_name,
        "model": prov.model_name,
        "prompt_version": best_version,
        "run_prompt_comparison": run_prompt_comparison,
        "prompt_versions_run": list(all_decisions.keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_wall_clock_s": round(time.monotonic() - t_start, 2),
        "n_queries": len(best_decisions),
        "parse_failures": parse_fails,
        "parse_failure_rate": round(parse_fails / len(best_decisions), 4) if best_decisions else 0,
        "total_cost_usd": round(total_cost, 6),
        "cache_dir": str(cache_dir),
        "output_files": [
            str(pm_path), str(output_dir / "metrics.json"), str(sd_path),
            str(pc_path), str(fa_path), str(lfa_path), str(ca_path),
            str(cmp_path), str(dec_path),
        ],
    }
    mf_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"  run_manifest: {mf_path}")

    # ── 8. Print summary ──────────────────────────────────────────────────────
    _print_summary(best_decisions, result_by_k, best_version, prov.model_name)
    typer.echo(f"\nTotal time: {time.monotonic() - t_start:.2f}s")


if __name__ == "__main__":
    app()
