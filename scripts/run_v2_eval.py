"""V2 Agentic RAG Evaluation Suite.

Runs 4 experiments comparing V2 pipeline configurations against each other
and against V1 baselines (BM25, Dense, Hybrid, Rule Planner).

Experiments:
  E1  v2_oracle          oracle QA + adaptive scorer + re-retrieval (main result)
  E2  v2_oracle_noloop   oracle QA + adaptive scorer, no re-retrieval (ablation)
  E3  v2_oracle_rankproxy oracle QA + rank_proxy scorer (ablation)
  E4  v2_heuristic       heuristic QA + adaptive scorer (ablation)

Usage:
    python scripts/run_v2_eval.py
    python scripts/run_v2_eval.py --experiment v2_oracle
    python scripts/run_v2_eval.py --answerable-only

Reads:
    data/sekd/queries.jsonl
    data/sekd/ground_truth.jsonl
    outputs/evaluation/bm25/metrics.json        (for comparison)
    outputs/evaluation/dense/metrics.json
    outputs/evaluation/hybrid/metrics.json
    outputs/evaluation/rule_planner/metrics.json

Writes:
    outputs/evaluation/v2/{experiment_name}/metrics.json
    outputs/evaluation/v2/{experiment_name}/retrieval_results.jsonl
    outputs/evaluation/v2/{experiment_name}/pipeline_stats.json
    outputs/evaluation/v2/{experiment_name}/run_manifest.json
    outputs/evaluation/v2/comparison.json       (all experiments + V1 baselines)
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run, group_by
from enterprise_rag.evaluation.v2_runner import (
    EXPERIMENTS,
    V2ExperimentConfig,
    V2Runner,
    aggregate_pipeline_stats,
)

app = typer.Typer(add_completion=False)

K_VALUES = [1, 3, 5, 10]
MAX_K = 10

V1_BASELINE_NAMES = ["bm25", "dense", "hybrid", "rule_planner"]


# ── I/O helpers ───────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_v1_metrics(
    base_dir: Path,
    names: list[str] = V1_BASELINE_NAMES,
) -> dict[str, dict]:
    baselines = {}
    for name in names:
        path = base_dir / name / "metrics.json"
        if path.exists():
            with open(path) as f:
                baselines[name] = json.load(f)
        else:
            typer.echo(f"  WARNING: V1 baseline {name} not found at {path}", err=True)
    return baselines


def _agg_dict(agg) -> dict:
    return {
        "n_queries": agg.n_queries,
        "recall": round(agg.recall, 4),
        "precision": round(agg.precision, 4),
        "mrr": round(agg.mrr, 4),
        "ndcg": round(agg.ndcg, 4),
        "hit_rate": round(agg.hit_rate, 4),
    }


# ── Per-experiment output writers ─────────────────────────────────────────────


def _write_experiment_outputs(
    config: V2ExperimentConfig,
    result_by_k: dict[int, list],
    flat_results: list[dict],
    per_query_stats: list[dict],
    output_dir: Path,
    elapsed: float,
    queries: list[dict],
    answerable_only: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    overall: dict = {}
    by_reasoning_type: dict = {}
    by_category: dict = {}

    for k in K_VALUES:
        qms = result_by_k[k]
        overall[f"k{k}"] = _agg_dict(aggregate(qms, k=k))
        rt_groups = group_by(qms, attr="reasoning_type", k=k)
        by_reasoning_type[f"k{k}"] = {
            v: _agg_dict(agg) for v, agg in sorted(rt_groups.items())
        }
        cat_groups = group_by(qms, attr="category", k=k)
        by_category[f"k{k}"] = {
            v: _agg_dict(agg) for v, agg in sorted(cat_groups.items())
        }

    metrics_doc = {
        "method": config.name,
        "description": config.description,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "use_oracle": config.use_oracle,
            "scorer": config.scorer,
            "max_loops": config.max_loops,
            "k": config.k,
        },
        "overall": overall,
        "by_reasoning_type": by_reasoning_type,
        "by_category": by_category,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_doc, indent=2))

    # retrieval_results.jsonl
    with open(output_dir / "retrieval_results.jsonl", "w") as f:
        for r in flat_results:
            f.write(json.dumps(r) + "\n")

    # pipeline_stats.json
    agg_stats = aggregate_pipeline_stats(per_query_stats)
    pipeline_doc = {
        "method": config.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregate": agg_stats,
        "per_query": per_query_stats,
    }
    (output_dir / "pipeline_stats.json").write_text(json.dumps(pipeline_doc, indent=2))

    # run_manifest.json
    agg10 = aggregate(result_by_k[MAX_K], k=MAX_K)
    n_answerable = sum(1 for q in queries if q.get("answerable", True))
    manifest = {
        "method": config.name,
        "description": config.description,
        "config": {
            "use_oracle": config.use_oracle,
            "scorer": config.scorer,
            "max_loops": config.max_loops,
        },
        "n_queries_total": len(queries),
        "n_answerable": n_answerable,
        "answerable_only": answerable_only,
        "k_values": K_VALUES,
        "recall_at_10": round(agg10.recall, 4),
        "mrr": round(agg10.mrr, 4),
        "ndcg_at_10": round(agg10.ndcg, 4),
        "hit_rate_at_10": round(agg10.hit_rate, 4),
        "validation_pass_rate": agg_stats.get("validation_pass_rate", 0.0),
        "mean_loops": agg_stats.get("mean_loops", 0.0),
        "n_errors": agg_stats.get("n_errors", 0),
        "duration_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


# ── Comparison table builder ──────────────────────────────────────────────────


def _build_comparison(
    exp_results: dict[str, dict],  # {exp_name: metrics_doc}
    v1_baselines: dict[str, dict],
    output_dir: Path,
) -> Path:
    """Write comparison.json with all V2 experiments and V1 baselines at k=10."""
    metric_keys = ["recall", "precision", "mrr", "ndcg", "hit_rate"]
    all_methods: dict[str, dict] = {}

    # V1 baselines
    for name, bdata in v1_baselines.items():
        k10 = bdata.get("overall", {}).get(f"k{MAX_K}", {})
        all_methods[name] = {
            "type": "v1_baseline",
            "k10": {m: round(k10.get(m, 0.0), 4) for m in metric_keys},
        }

    # V2 experiments
    v2_oracle_k10 = None
    for name, mdata in exp_results.items():
        k10 = mdata.get("overall", {}).get(f"k{MAX_K}", {})
        all_methods[name] = {
            "type": "v2_experiment",
            "description": mdata.get("description", ""),
            "config": mdata.get("config", {}),
            "k10": {m: round(k10.get(m, 0.0), 4) for m in metric_keys},
        }
        if name == "v2_oracle":
            v2_oracle_k10 = all_methods[name]["k10"]

    # Compute deltas of v2_oracle vs each V1 baseline
    deltas: dict[str, dict] = {}
    if v2_oracle_k10:
        for bname in v1_baselines:
            bk10 = all_methods.get(bname, {}).get("k10", {})
            deltas[f"v2_oracle_vs_{bname}"] = {
                m: round(v2_oracle_k10.get(m, 0.0) - bk10.get(m, 0.0), 4)
                for m in metric_keys
            }

    # Per-reasoning-type breakdown for v2_oracle vs rule_planner (best V1)
    rt_comparison: dict[str, Any] = {}
    if "v2_oracle" in exp_results and "rule_planner" in v1_baselines:
        v2_rt = exp_results["v2_oracle"].get("by_reasoning_type", {}).get(f"k{MAX_K}", {})
        v1_rt = v1_baselines["rule_planner"].get("by_reasoning_type", {}).get(f"k{MAX_K}", {})
        for rt in set(v2_rt) | set(v1_rt):
            v2m = v2_rt.get(rt, {})
            v1m = v1_rt.get(rt, {})
            rt_comparison[rt] = {
                "v2_oracle": {m: round(v2m.get(m, 0.0), 4) for m in metric_keys},
                "rule_planner": {m: round(v1m.get(m, 0.0), 4) for m in metric_keys},
                "delta": {
                    m: round(v2m.get(m, 0.0) - v1m.get(m, 0.0), 4)
                    for m in metric_keys
                },
            }

    comparison_doc = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": MAX_K,
        "metric_keys": metric_keys,
        "methods": all_methods,
        "deltas_v2_oracle": deltas,
        f"by_reasoning_type_k{MAX_K}_v2oracle_vs_rule_planner": rt_comparison,
    }

    path = output_dir / "comparison.json"
    path.write_text(json.dumps(comparison_doc, indent=2))
    return path


# ── Progress helper ───────────────────────────────────────────────────────────


def _make_progress(name: str, total: int):
    bar_width = 40
    _t = [time.monotonic()]

    def _progress(done: int, _total: int) -> None:
        filled = int(bar_width * done / _total)
        bar = "█" * filled + "░" * (bar_width - filled)
        elapsed = time.monotonic() - _t[0]
        rate = done / elapsed if elapsed > 0 else 0
        eta = (_total - done) / rate if rate > 0 else 0
        typer.echo(
            f"\r  [{bar}] {done:>4}/{_total}  {rate:4.1f} q/s  ETA {eta:.0f}s  ",
            nl=False,
        )

    return _progress


# ── Print helpers ─────────────────────────────────────────────────────────────


def _print_summary(
    exp_results: dict[str, dict],
    v1_baselines: dict[str, dict],
    pipeline_stats_by_exp: dict[str, dict],
) -> None:
    sep = "─" * 88
    typer.echo(f"\n{sep}")
    typer.echo("V2 Evaluation Results — Retrieval Metrics at k=10")
    typer.echo(sep)
    typer.echo(f"  {'Method':<28}  {'Recall':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}  {'Type'}")
    typer.echo(sep)

    def _row(name: str, k10: dict, method_type: str) -> None:
        typer.echo(
            f"  {name:<28}  "
            f"{k10.get('recall', 0):.4f}   "
            f"{k10.get('mrr', 0):.4f}   "
            f"{k10.get('ndcg', 0):.4f}   "
            f"{k10.get('hit_rate', 0):.4f}   "
            f"{method_type}"
        )

    # V1 baselines first
    for bname in V1_BASELINE_NAMES:
        if bname in v1_baselines:
            k10 = v1_baselines[bname].get("overall", {}).get(f"k{MAX_K}", {})
            _row(bname, k10, "v1")

    typer.echo(f"  {'─'*85}")

    # V2 experiments
    for exp_name, mdata in exp_results.items():
        k10 = mdata.get("overall", {}).get(f"k{MAX_K}", {})
        _row(exp_name, k10, "v2")

    typer.echo(sep)

    # Deltas: v2_oracle vs best V1
    if "v2_oracle" in exp_results and "rule_planner" in v1_baselines:
        v2k10 = exp_results["v2_oracle"].get("overall", {}).get(f"k{MAX_K}", {})
        v1k10 = v1_baselines["rule_planner"].get("overall", {}).get(f"k{MAX_K}", {})
        r_delta = v2k10.get("recall", 0) - v1k10.get("recall", 0)
        m_delta = v2k10.get("mrr", 0) - v1k10.get("mrr", 0)
        n_delta = v2k10.get("ndcg", 0) - v1k10.get("ndcg", 0)
        sign = lambda x: "+" if x >= 0 else ""
        typer.echo(f"\n  v2_oracle vs rule_planner:  "
                   f"Recall {sign(r_delta)}{r_delta:.4f}  "
                   f"MRR {sign(m_delta)}{m_delta:.4f}  "
                   f"nDCG {sign(n_delta)}{n_delta:.4f}")

    # Pipeline stats
    if pipeline_stats_by_exp:
        typer.echo(f"\n{sep}")
        typer.echo("V2 Pipeline Statistics")
        typer.echo(sep)
        typer.echo(f"  {'Experiment':<28}  {'ValPass':>7}  {'Loops':>6}  {'Errors':>7}")
        typer.echo(sep)
        for exp_name, stats in pipeline_stats_by_exp.items():
            typer.echo(
                f"  {exp_name:<28}  "
                f"{stats.get('validation_pass_rate', 0):.4f}   "
                f"{stats.get('mean_loops', 0):.3f}   "
                f"{stats.get('n_errors', 0):>7}"
            )
        typer.echo(sep)

    # Per reasoning type
    if "v2_oracle" in exp_results:
        rt_data = exp_results["v2_oracle"].get("by_reasoning_type", {}).get(f"k{MAX_K}", {})
        if rt_data:
            typer.echo(f"\n{sep}")
            typer.echo(f"v2_oracle — Per Reasoning Type at k={MAX_K}")
            typer.echo(sep)
            typer.echo(f"  {'Type':<16}  {'Recall':>7}  {'MRR':>7}  {'nDCG':>7}  {'N':>5}")
            typer.echo(sep)
            for rt, metrics in sorted(rt_data.items()):
                typer.echo(
                    f"  {rt:<16}  "
                    f"{metrics.get('recall', 0):.4f}   "
                    f"{metrics.get('mrr', 0):.4f}   "
                    f"{metrics.get('ndcg', 0):.4f}   "
                    f"{metrics.get('n_queries', 0):>5}"
                )
            typer.echo(sep)


# ── Main command ──────────────────────────────────────────────────────────────

from typing import Any  # noqa: E402


@app.command()
def main(
    queries_path: Annotated[
        Path, typer.Option(help="Path to queries.jsonl.")
    ] = Path("data/sekd/queries.jsonl"),
    ground_truth_path: Annotated[
        Path, typer.Option(help="Path to ground_truth.jsonl.")
    ] = Path("data/sekd/ground_truth.jsonl"),
    v1_eval_dir: Annotated[
        Path, typer.Option(help="Directory containing V1 baseline outputs.")
    ] = Path("outputs/evaluation"),
    output_dir: Annotated[
        Path, typer.Option(help="Root output directory for V2 evaluation.")
    ] = Path("outputs/evaluation/v2"),
    experiment: Annotated[
        str | None,
        typer.Option(help="Run only this experiment name (default: all).")
    ] = None,
    answerable_only: Annotated[
        bool, typer.Option(help="Skip unanswerable queries in evaluation.")
    ] = True,
    k: Annotated[
        int, typer.Option(help="Top-K chunks to retrieve per query.")
    ] = 10,
) -> None:
    """Run V2 evaluation suite — 4 experiments vs V1 baselines."""
    t_total = time.monotonic()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    typer.echo(f"Loading queries from {queries_path} ...")
    queries = _load_jsonl(queries_path)
    answerable = [q for q in queries if q.get("answerable", True)] if answerable_only else queries
    typer.echo(f"  {len(queries)} total, {len(answerable)} answerable")

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {
        gt["query_id"]: gt.get("required_chunk_ids", [])
        for gt in _load_jsonl(ground_truth_path)
    }
    queries_meta = {q["query_id"]: q for q in queries}

    # ── 2. Load V1 baselines ──────────────────────────────────────────────────
    typer.echo("Loading V1 baseline metrics ...")
    v1_baselines = _load_v1_metrics(v1_eval_dir)
    typer.echo(f"  Found: {list(v1_baselines.keys())}")

    # ── 3. Select experiments ─────────────────────────────────────────────────
    to_run = [e for e in EXPERIMENTS if experiment is None or e.name == experiment]
    if not to_run:
        typer.echo(f"ERROR: No experiment named '{experiment}'. "
                   f"Valid: {[e.name for e in EXPERIMENTS]}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Running {len(to_run)} experiment(s): {[e.name for e in to_run]}")

    # ── 4. Run experiments ────────────────────────────────────────────────────
    exp_results: dict[str, dict] = {}
    pipeline_stats_by_exp: dict[str, dict] = {}
    output_dir.mkdir(parents=True, exist_ok=True)

    for cfg in to_run:
        cfg.k = k
        typer.echo(f"\n{'═' * 72}")
        typer.echo(f"Experiment: {cfg.name}")
        typer.echo(f"  {cfg.description}")
        typer.echo(f"  oracle={cfg.use_oracle}  scorer={cfg.scorer}  max_loops={cfg.max_loops}  k={cfg.k}")
        typer.echo(f"{'─' * 72}")

        runner = V2Runner(cfg)
        runner.setup()

        t_exp = time.monotonic()
        progress = _make_progress(cfg.name, len(answerable))

        flat_results, per_query_stats = runner.run_batch(answerable, progress_fn=progress)
        typer.echo()  # newline after progress bar

        elapsed = time.monotonic() - t_exp
        typer.echo(f"  Done: {len(flat_results)} results from {len(answerable)} queries in {elapsed:.1f}s")

        # Evaluate metrics
        result_by_k = evaluate_run(
            flat_results, ground_truths, queries_meta,
            k_values=K_VALUES, answerable_only=answerable_only,
        )

        # Aggregate pipeline stats
        agg_stats = aggregate_pipeline_stats(per_query_stats)
        pipeline_stats_by_exp[cfg.name] = agg_stats
        typer.echo(f"  Val pass rate: {agg_stats.get('validation_pass_rate', 0):.3f}  "
                   f"Mean loops: {agg_stats.get('mean_loops', 0):.3f}  "
                   f"Errors: {agg_stats.get('n_errors', 0)}")

        # Print per-k table
        agg10 = aggregate(result_by_k[MAX_K], k=MAX_K)
        typer.echo(f"  Recall@10={agg10.recall:.4f}  MRR={agg10.mrr:.4f}  "
                   f"nDCG@10={agg10.ndcg:.4f}  Hit@10={agg10.hit_rate:.4f}")

        # Write outputs
        exp_output_dir = output_dir / cfg.name
        _write_experiment_outputs(
            cfg, result_by_k, flat_results, per_query_stats,
            exp_output_dir, elapsed, queries, answerable_only,
        )

        # Load back the metrics doc for comparison
        with open(exp_output_dir / "metrics.json") as f:
            exp_results[cfg.name] = json.load(f)

        runner.teardown()
        typer.echo(f"  Outputs written to {exp_output_dir}")

    # ── 5. Write comparison.json ──────────────────────────────────────────────
    cmp_path = _build_comparison(exp_results, v1_baselines, output_dir)
    typer.echo(f"\nComparison written to {cmp_path}")

    # ── 6. Print summary ──────────────────────────────────────────────────────
    _print_summary(exp_results, v1_baselines, pipeline_stats_by_exp)

    typer.echo(f"\nTotal time: {time.monotonic() - t_total:.1f}s")


if __name__ == "__main__":
    app()
