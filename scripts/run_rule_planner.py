"""Rule-Based Retrieval Planner runner.

Loads pre-computed BM25, Dense, and Hybrid retrieval results, applies the
rule-based planner to select a strategy per query, assembles a planner run
from the selected method's results, evaluates retrieval metrics, and writes
all outputs to outputs/evaluation/rule_planner/.

Usage:
    python scripts/run_rule_planner.py
    python scripts/run_rule_planner.py --answerable-only

Reads:
    data/sekd/queries.jsonl
    data/sekd/ground_truth.jsonl
    outputs/evaluation/bm25/retrieval_results.jsonl
    outputs/evaluation/dense/retrieval_results.jsonl
    outputs/evaluation/hybrid/retrieval_results.jsonl

Writes:
    outputs/evaluation/rule_planner/planner_metrics.json
    outputs/evaluation/rule_planner/retrieval_metrics.json
    outputs/evaluation/rule_planner/strategy_distribution.csv
    outputs/evaluation/rule_planner/rule_analysis.csv
    outputs/evaluation/rule_planner/comparison_vs_fixed_methods.json
    outputs/evaluation/rule_planner/failure_analysis.json
    outputs/evaluation/rule_planner/planner_decisions.jsonl
    outputs/evaluation/rule_planner/run_manifest.json
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
from typing import Annotated, Any

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("OMP_NUM_THREADS", "1")

from enterprise_rag.evaluation.planner_metrics import (
    compute_planner_metrics,
    compute_rule_contributions,
    group_decisions_by,
    strategy_distribution_report,
)
from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
from enterprise_rag.evaluation.reporter import write_failure_analysis, write_metrics_json
from enterprise_rag.planning.rule_based import RuleBasedPlanner
from enterprise_rag.planning.schema import ORACLE_STRATEGY_MAP
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk

app = typer.Typer(add_completion=False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_retrieval_run(path: Path, method: str) -> dict[str, list[dict]]:
    """Load retrieval_results.jsonl and index by query_id → sorted-by-rank list."""
    by_query: dict[str, list[dict]] = defaultdict(list)
    for rec in _load_jsonl(path):
        by_query[rec["query_id"]].append(rec)
    for qid in by_query:
        by_query[qid].sort(key=lambda r: r["rank"])
    return dict(by_query)


def _assemble_planner_run(
    decisions: list,
    queries: list[dict],
    retrieval_results: dict[str, dict[str, list[dict]]],
    k: int,
) -> RetrievalRun:
    """Build a RetrievalRun by selecting results from the planner-chosen method per query."""
    strategy_to_results = retrieval_results
    decision_map = {d.query_id: d for d in decisions}

    run = RetrievalRun(
        method="rule_planner",
        k=k,
        config={"planner_type": "rule_based", "k": k},
    )

    for q in queries:
        qid = q["query_id"]
        dec = decision_map.get(qid)
        if dec is None:
            continue
        strat = dec.selected_strategy
        results_for_q = strategy_to_results.get(strat, {}).get(qid, [])[:k]
        for rec in results_for_q:
            run.results.append(
                RetrievedChunk(
                    query_id=qid,
                    chunk_id=rec["chunk_id"],
                    document_id=rec.get("document_id", ""),
                    rank=rec["rank"],
                    score=rec.get("score", 0.0),
                    category=rec.get("category", ""),
                    section_path=rec.get("section_path", []),
                    text="",
                )
            )
    return run


def _comparison_json(
    result_by_k: dict[int, list],
    planner_run: RetrievalRun,
    baselines: dict[str, dict],
    output_dir: Path,
    k_values: list[int],
    queries_meta: dict[str, dict],
) -> Path:
    """Write comparison_vs_fixed_methods.json with deltas for all k-values."""
    metric_keys = ["recall", "precision", "mrr", "ndcg", "hit_rate"]
    max_k = max(k_values)

    # Per-k overall comparison
    overall: dict[str, Any] = {}
    for k in k_values:
        qms = result_by_k[k]
        agg = aggregate(qms, k=k)
        row: dict[str, Any] = {"n_queries": agg.n_queries}
        for m in metric_keys:
            curr_val = round(getattr(agg, m), 4)
            row[m] = {"rule_planner": curr_val}
            for bname, bdata in baselines.items():
                bval = round(bdata.get("overall", {}).get(f"k{k}", {}).get(m, 0.0), 4)
                row[m][bname] = bval
                row[m][f"delta_vs_{bname}"] = round(curr_val - bval, 4)
        overall[f"k{k}"] = row

    # Category breakdown at max_k
    from enterprise_rag.evaluation.retrieval_metrics import group_by
    cat_groups = group_by(result_by_k[max_k], attr="category", k=max_k)
    rt_groups = group_by(result_by_k[max_k], attr="reasoning_type", k=max_k)

    def _group_block(current_groups: dict, base_csv_dir: Path, group_key: str) -> dict:
        result: dict[str, Any] = {}
        for gv, agg in sorted(current_groups.items()):
            row = {"n_queries": agg.n_queries}
            for m in metric_keys:
                curr_val = round(getattr(agg, m), 4)
                row[m] = {"rule_planner": curr_val}
                for bname in baselines:
                    base_csv = base_csv_dir / bname / f"per_{group_key}.csv"
                    bval = 0.0
                    if base_csv.exists():
                        import csv as csv_mod
                        with open(base_csv) as f:
                            for r in csv_mod.DictReader(f):
                                if r["group_value"] == gv and int(r["k"]) == max_k:
                                    bval = float(r.get(m, 0.0))
                    row[m][bname] = round(bval, 4)
                    row[m][f"delta_vs_{bname}"] = round(curr_val - bval, 4)
            result[gv] = row
        return result

    base_dir = output_dir.parent
    cat_cmp = _group_block(cat_groups, base_dir, "category")
    rt_cmp = _group_block(rt_groups, base_dir, "reasoning_type")

    output: dict[str, Any] = {
        "current_method": "rule_planner",
        "planner_type": "rule_based",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k_values": k_values,
        "baselines": list(baselines.keys()),
        "overall": overall,
        f"by_category_k{max_k}": cat_cmp,
        f"by_reasoning_type_k{max_k}": rt_cmp,
    }

    path = output_dir / "comparison_vs_fixed_methods.json"
    path.write_text(json.dumps(output, indent=2))
    return path


# ── Main command ──────────────────────────────────────────────────────────────


@app.command()
def main(
    queries_path: Annotated[
        Path, typer.Option(help="Path to queries.jsonl.")
    ] = Path("data/sekd/queries.jsonl"),
    ground_truth_path: Annotated[
        Path, typer.Option(help="Path to ground_truth.jsonl.")
    ] = Path("data/sekd/ground_truth.jsonl"),
    bm25_results: Annotated[
        Path, typer.Option(help="Path to BM25 retrieval_results.jsonl.")
    ] = Path("outputs/evaluation/bm25/retrieval_results.jsonl"),
    dense_results: Annotated[
        Path, typer.Option(help="Path to Dense retrieval_results.jsonl.")
    ] = Path("outputs/evaluation/dense/retrieval_results.jsonl"),
    hybrid_results: Annotated[
        Path, typer.Option(help="Path to Hybrid retrieval_results.jsonl.")
    ] = Path("outputs/evaluation/hybrid/retrieval_results.jsonl"),
    bm25_metrics: Annotated[
        Path, typer.Option(help="BM25 metrics.json for comparison.")
    ] = Path("outputs/evaluation/bm25/metrics.json"),
    dense_metrics: Annotated[
        Path, typer.Option(help="Dense metrics.json for comparison.")
    ] = Path("outputs/evaluation/dense/metrics.json"),
    hybrid_metrics: Annotated[
        Path, typer.Option(help="Hybrid metrics.json for comparison.")
    ] = Path("outputs/evaluation/hybrid/metrics.json"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write evaluation outputs.")
    ] = Path("outputs/evaluation/rule_planner"),
    k_values: Annotated[
        list[int], typer.Option(help="K cutoffs to evaluate.")
    ] = [1, 3, 5, 10],  # noqa: B006
    answerable_only: Annotated[
        bool, typer.Option(help="Skip unanswerable queries.")
    ] = True,
) -> None:
    """Run Rule-Based Retrieval Planner, evaluate, and write all outputs."""
    t_start = time.monotonic()
    max_k = max(k_values)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    typer.echo(f"Loading queries from {queries_path} ...")
    queries = _load_jsonl(queries_path)
    typer.echo(f"  {len(queries)} queries")

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    for gt in _load_jsonl(ground_truth_path):
        ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])
    queries_meta = {q["query_id"]: q for q in queries}

    # ── 2. Load pre-computed retrieval results ────────────────────────────────
    typer.echo("Loading pre-computed retrieval results ...")
    retrieval_runs: dict[str, dict[str, list[dict]]] = {}
    for method, path in [("bm25", bm25_results), ("dense", dense_results), ("hybrid", hybrid_results)]:
        if not path.exists():
            typer.echo(f"  WARNING: {path} not found — {method} results unavailable")
        else:
            retrieval_runs[method] = _load_retrieval_run(path, method)
            n_queries = len(retrieval_runs[method])
            typer.echo(f"  {method}: {n_queries} queries loaded from {path}")

    if len(retrieval_runs) < 3:
        typer.echo("ERROR: All three retrieval results required.", err=True)
        raise typer.Exit(1)

    # ── 3. Run the rule-based planner ─────────────────────────────────────────
    typer.echo("Running rule-based planner ...")
    planner = RuleBasedPlanner()
    decisions, planner_elapsed = planner.plan_batch(queries)
    typer.echo(f"  {len(decisions)} decisions in {planner_elapsed*1000:.1f}ms")

    # ── 4. Assemble planner retrieval run ─────────────────────────────────────
    typer.echo(f"Assembling planner retrieval run (max_k={max_k}) ...")
    planner_run = _assemble_planner_run(decisions, queries, retrieval_runs, k=max_k)
    typer.echo(f"  {len(planner_run.results)} results in planner run")

    # ── 5. Evaluate retrieval metrics ─────────────────────────────────────────
    typer.echo("Evaluating retrieval metrics ...")
    ret_results = [r.to_dict() for r in planner_run.results]
    result_by_k = evaluate_run(ret_results, ground_truths, queries_meta,
                               k_values=k_values, answerable_only=answerable_only)

    # ── 6. Compute planner accuracy metrics ───────────────────────────────────
    typer.echo("Computing planner accuracy metrics ...")
    answerable_decisions = [
        d for d in decisions
        if not answerable_only or queries_meta.get(d.query_id, {}).get("answerable", True)
    ]
    planner_metrics = compute_planner_metrics(answerable_decisions)
    typer.echo(f"  Strategy Selection Accuracy: {planner_metrics.accuracy:.4f} "
               f"({planner_metrics.n_correct}/{planner_metrics.n_queries})")
    typer.echo(f"  Accuracy excl. discrepancy: {planner_metrics.accuracy_ex_discrepancy:.4f}")

    # Rule contributions
    rule_contribs = compute_rule_contributions(answerable_decisions)
    for rc in rule_contribs:
        if rc.usage_count > 0:
            typer.echo(f"  {rc.rule_id}: {rc.usage_count} queries → {rc.strategy} "
                       f"(accuracy={rc.accuracy:.3f})")

    # Grouped accuracy
    by_rt = group_decisions_by(answerable_decisions, "reasoning_type")
    by_cat = group_decisions_by(answerable_decisions, "category")
    dist = strategy_distribution_report(answerable_decisions)

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writing outputs to {output_dir} ...")

    # planner_metrics.json
    planner_acc_by_rt = {g: {"n": m.n_queries, "accuracy": round(m.accuracy, 4),
                              "distribution": m.strategy_distribution}
                         for g, m in by_rt.items()}
    planner_acc_by_cat = {g: {"n": m.n_queries, "accuracy": round(m.accuracy, 4),
                               "distribution": m.strategy_distribution}
                          for g, m in by_cat.items()}

    planner_metrics_doc = {
        "planner_type": "rule_based",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_rules": len(planner.describe_rules()),
        "rules": planner.describe_rules(),
        "overall": planner_metrics.to_dict(),
        "by_reasoning_type": planner_acc_by_rt,
        "by_category": planner_acc_by_cat,
        "oracle_discrepancies": [
            d.to_dict() for d in answerable_decisions if d.oracle_discrepancy
        ][:10],
    }
    pm_path = output_dir / "planner_metrics.json"
    pm_path.write_text(json.dumps(planner_metrics_doc, indent=2))
    typer.echo(f"  planner_metrics: {pm_path}")

    # retrieval_metrics.json
    rm_path = write_metrics_json(result_by_k, planner_run, output_dir)
    typer.echo(f"  retrieval_metrics: {rm_path}")

    # strategy_distribution.csv
    sd_path = output_dir / "strategy_distribution.csv"
    with open(sd_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["selected_strategy", "count", "pct", "n_correct", "accuracy"])
        writer.writeheader()
        writer.writerows(dist)
    typer.echo(f"  strategy_distribution: {sd_path}")

    # rule_analysis.csv
    ra_path = output_dir / "rule_analysis.csv"
    with open(ra_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["rule_id", "rule_name", "strategy",
                                                "usage_count", "accuracy", "n_correct"])
        writer.writeheader()
        for rc in rule_contribs:
            writer.writerow({
                "rule_id": rc.rule_id,
                "rule_name": rc.rule_name,
                "strategy": rc.strategy,
                "usage_count": rc.usage_count,
                "accuracy": round(rc.accuracy, 4),
                "n_correct": rc.n_correct,
            })
    typer.echo(f"  rule_analysis: {ra_path}")

    # failure_analysis.json
    fa_path = write_failure_analysis(result_by_k, queries_meta, planner_run, output_dir)
    typer.echo(f"  failure_analysis: {fa_path}")

    # comparison_vs_fixed_methods.json
    baselines: dict[str, dict] = {}
    for bname, bpath in [("bm25", bm25_metrics), ("dense", dense_metrics), ("hybrid", hybrid_metrics)]:
        if bpath.exists():
            with open(bpath) as f:
                baselines[bname] = json.load(f)
        else:
            typer.echo(f"  WARNING: {bpath} not found, {bname} omitted from comparison")

    cmp_path = _comparison_json(result_by_k, planner_run, baselines, output_dir, k_values, queries_meta)
    typer.echo(f"  comparison_vs_fixed_methods: {cmp_path}")

    # planner_decisions.jsonl
    pd_path = output_dir / "planner_decisions.jsonl"
    with open(pd_path, "w") as f:
        for d in decisions:
            f.write(json.dumps(d.to_dict()) + "\n")
    typer.echo(f"  planner_decisions: {pd_path}")

    # run_manifest.json
    elapsed = time.monotonic() - t_start
    manifest = {
        "method": "rule_planner",
        "planner_type": "rule_based",
        "n_rules": len(planner.describe_rules()),
        "n_queries": len(queries),
        "n_answerable": sum(1 for q in queries if q.get("answerable", True)),
        "answerable_only": answerable_only,
        "k_values": sorted(k_values),
        "strategy_selection_accuracy": round(planner_metrics.accuracy, 4),
        "accuracy_ex_discrepancy": round(planner_metrics.accuracy_ex_discrepancy, 4),
        "strategy_distribution": {d["selected_strategy"]: d["count"] for d in dist},
        "planner_latency_ms": round(planner_elapsed * 1000, 3),
        "total_duration_seconds": round(elapsed, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "oracle_discrepancy_count": planner_metrics.n_discrepancy,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"  run_manifest: {manifest_path}")

    # ── 8. Summary tables ─────────────────────────────────────────────────────
    typer.echo(f"\n{'─' * 72}")
    typer.echo("Rule-Based Planner — Retrieval Summary")
    typer.echo(f"{'─' * 72}")
    typer.echo(f"{'k':>4}  {'Recall':>7}  {'Prec':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}  {'Queries':>7}")
    typer.echo(f"{'─' * 72}")
    for k in sorted(k_values):
        agg = aggregate(result_by_k[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:>7.4f}  {agg.precision:>7.4f}  "
                   f"{agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}  {agg.n_queries:>7}")
    typer.echo(f"{'─' * 72}")

    typer.echo(f"\n{'─' * 72}")
    typer.echo("Planner Strategy Selection Accuracy")
    typer.echo(f"{'─' * 72}")
    typer.echo(f"  Overall accuracy:              {planner_metrics.accuracy:.4f} "
               f"({planner_metrics.n_correct}/{planner_metrics.n_queries})")
    typer.echo(f"  Excl. oracle discrepancies:    {planner_metrics.accuracy_ex_discrepancy:.4f}")
    typer.echo(f"  Oracle discrepancy queries:    {planner_metrics.n_discrepancy}")
    typer.echo()
    typer.echo(f"  {'Strategy':<10}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'Oracle':>7}  {'Predicted':>9}")
    for s in sorted(AVAILABLE_STRATEGIES):
        typer.echo(f"  {s:<10}  "
                   f"{planner_metrics.precision_by_strategy.get(s, 0):.4f}  "
                   f"  {planner_metrics.recall_by_strategy.get(s, 0):.4f}  "
                   f"  {planner_metrics.f1_by_strategy.get(s, 0):.4f}  "
                   f"  {planner_metrics.support_by_strategy.get(s, 0):>7}  "
                   f"  {planner_metrics.predicted_by_strategy.get(s, 0):>9}")
    typer.echo(f"{'─' * 72}")

    typer.echo(f"\n{'─' * 72}")
    typer.echo("Strategy Distribution and Rule Usage")
    typer.echo(f"{'─' * 72}")
    typer.echo(f"  {'Strategy':<8}  {'Count':>6}  {'Pct':>6}")
    for row in dist:
        typer.echo(f"  {row['selected_strategy']:<8}  {row['count']:>6}  {row['pct']:>5.1f}%")
    typer.echo()
    typer.echo(f"  {'Rule':<6}  {'Strategy':<8}  {'Usage':>6}  {'Correct':>8}  {'Accuracy':>9}")
    for rc in rule_contribs:
        if rc.usage_count > 0:
            typer.echo(f"  {rc.rule_id:<6}  {rc.strategy:<8}  {rc.usage_count:>6}  "
                       f"{rc.n_correct:>8}  {rc.accuracy:>9.4f}")
    typer.echo(f"{'─' * 72}")

    if baselines:
        typer.echo(f"\n{'─' * 80}")
        typer.echo("4-Way Comparison at k=10  (Rule Planner vs BM25 vs Dense vs Hybrid)")
        typer.echo(f"{'─' * 80}")
        agg10 = aggregate(result_by_k[max_k], k=max_k)
        names = ["rule_planner"] + list(baselines.keys())
        typer.echo(f"  {'Method':<16}  {'Recall':>8}  {'MRR':>8}  {'nDCG':>8}  {'Hit':>8}")
        typer.echo(f"  {'rule_planner':<16}  {agg10.recall:>8.4f}  {agg10.mrr:>8.4f}  "
                   f"{agg10.ndcg:>8.4f}  {agg10.hit_rate:>8.4f}")
        for bname, bdata in baselines.items():
            bk = bdata.get("overall", {}).get(f"k{max_k}", {})
            typer.echo(f"  {bname:<16}  "
                       f"{bk.get('recall', 0):.4f}  "
                       f"    {bk.get('mrr', 0):.4f}  "
                       f"    {bk.get('ndcg', 0):.4f}  "
                       f"    {bk.get('hit_rate', 0):.4f}")
        typer.echo(f"{'─' * 80}")

    typer.echo(f"\nTotal time: {time.monotonic() - t_start:.2f}s")


from enterprise_rag.planning.schema import AVAILABLE_STRATEGIES  # noqa: E402

if __name__ == "__main__":
    app()
