"""BM25 retrieval runner: build index, retrieve, evaluate, write outputs.

Usage:
    python scripts/run_bm25.py
    python scripts/run_bm25.py --k-values 1 3 5 10 --output-dir outputs/evaluation/bm25

Reads:
    data/sekd/processed/chunks.jsonl
    data/sekd/queries.jsonl
    data/sekd/ground_truth.jsonl

Writes:
    outputs/evaluation/bm25/metrics.json
    outputs/evaluation/bm25/per_category.csv
    outputs/evaluation/bm25/per_reasoning_type.csv
    outputs/evaluation/bm25/retrieval_results.jsonl
    outputs/evaluation/bm25/failure_analysis.json
    outputs/evaluation/bm25/index/  (serialized BM25 index)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Annotated

import typer

# Ensure src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enterprise_rag.evaluation.reporter import write_all_outputs
from enterprise_rag.evaluation.retrieval_metrics import evaluate_run
from enterprise_rag.retrieval.bm25 import BM25Config, BM25Index, build_bm25_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    chunks_path: Annotated[
        Path,
        typer.Option(help="Path to chunks.jsonl."),
    ] = Path("data/sekd/processed/chunks.jsonl"),
    queries_path: Annotated[
        Path,
        typer.Option(help="Path to queries.jsonl."),
    ] = Path("data/sekd/queries.jsonl"),
    ground_truth_path: Annotated[
        Path,
        typer.Option(help="Path to ground_truth.jsonl."),
    ] = Path("data/sekd/ground_truth.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write evaluation outputs."),
    ] = Path("outputs/evaluation/bm25"),
    k_values: Annotated[
        list[int],
        typer.Option(help="K cutoffs to evaluate (repeatable: --k-values 1 --k-values 3 ...)."),
    ] = [1, 3, 5, 10],  # noqa: B006
    rebuild_index: Annotated[
        bool,
        typer.Option(help="Force rebuild even if serialized index exists."),
    ] = False,
    answerable_only: Annotated[
        bool,
        typer.Option(help="Skip unanswerable queries during evaluation."),
    ] = True,
    k1: Annotated[float, typer.Option(help="BM25 k1 parameter.")] = 1.5,
    b: Annotated[float, typer.Option(help="BM25 b parameter.")] = 0.75,
) -> None:
    """Run BM25 retrieval and evaluation over the SEKD corpus."""
    t_start = time.monotonic()
    max_k = max(k_values)
    index_dir = output_dir / "index"

    # ── 1. Load chunks ────────────────────────────────────────────────────────
    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[dict] = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    typer.echo(f"  {len(chunks)} chunks loaded")

    # ── 2. Build or load BM25 index ───────────────────────────────────────────
    config = BM25Config(k1=k1, b=b, max_k=max_k)
    if not rebuild_index and (index_dir / "bm25.pkl").exists():
        typer.echo(f"Loading BM25 index from {index_dir} ...")
        index = BM25Index.load(index_dir, config=config)
    else:
        typer.echo(f"Building BM25 index ({len(chunks)} chunks) ...")
        index = build_bm25_index(chunks, index_dir=index_dir, config=config)

    # ── 3. Load queries ───────────────────────────────────────────────────────
    typer.echo(f"Loading queries from {queries_path} ...")
    queries: list[dict] = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    typer.echo(f"  {len(queries)} queries loaded")

    # ── 4. Load ground truths ─────────────────────────────────────────────────
    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = json.loads(line)
                ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])
    typer.echo(f"  {len(ground_truths)} ground truths loaded")

    # ── 5. Build query metadata dict ──────────────────────────────────────────
    queries_meta: dict[str, dict] = {q["query_id"]: q for q in queries}
    queries_by_id: dict[str, dict] = queries_meta  # same reference

    # ── 6. Run BM25 retrieval ─────────────────────────────────────────────────
    typer.echo(f"Running BM25 retrieval (k={max_k}, {len(queries)} queries) ...")
    t_ret = time.monotonic()
    run = index.run(queries, k=max_k, method="bm25")
    elapsed_ret = time.monotonic() - t_ret
    typer.echo(f"  Retrieved {len(run.results)} results in {elapsed_ret:.2f}s")

    # ── 7. Evaluate ───────────────────────────────────────────────────────────
    typer.echo(f"Evaluating at k ∈ {sorted(k_values)} ...")
    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(
        retrieval_results=retrieval_results,
        ground_truths=ground_truths,
        queries_meta=queries_meta,
        k_values=k_values,
        answerable_only=answerable_only,
    )

    # ── 8. Write outputs ──────────────────────────────────────────────────────
    typer.echo(f"Writing outputs to {output_dir} ...")
    paths = write_all_outputs(
        result_by_k=result_by_k,
        run=run,
        queries_by_id=queries_by_id,
        output_dir=output_dir,
    )
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")

    # ── 9. Print evaluation summary ───────────────────────────────────────────
    elapsed_total = time.monotonic() - t_start
    typer.echo(f"\n{'─' * 60}")
    typer.echo(f"BM25 Evaluation Summary  (total: {elapsed_total:.1f}s)")
    typer.echo(f"{'─' * 60}")
    typer.echo(f"{'k':>4}  {'Recall':>7}  {'Prec':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}  {'Queries':>7}")
    typer.echo(f"{'─' * 60}")

    from enterprise_rag.evaluation.retrieval_metrics import aggregate

    for k in sorted(k_values):
        qms = result_by_k[k]
        agg = aggregate(qms, k=k)
        typer.echo(
            f"{k:>4}  {agg.recall:>7.4f}  {agg.precision:>7.4f}  "
            f"{agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}  {agg.n_queries:>7}"
        )

    typer.echo(f"{'─' * 60}")

    # Quick failure summary at max K
    qms_max = result_by_k[max_k]
    n_zero = sum(1 for m in qms_max if m.recall == 0.0)
    n_perfect = sum(1 for m in qms_max if m.recall == 1.0)
    typer.echo(
        f"\nAt k={max_k}: {n_zero}/{len(qms_max)} zero-recall, "
        f"{n_perfect}/{len(qms_max)} perfect-recall"
    )
    typer.echo(f"See {output_dir / 'failure_analysis.json'} for failure details.")


if __name__ == "__main__":
    app()
