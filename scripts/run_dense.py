"""Dense retrieval runner: build embedding index, retrieve, evaluate, write outputs.

Usage:
    python scripts/run_dense.py
    python scripts/run_dense.py --model-name BAAI/bge-small-en-v1.5
    python scripts/run_dense.py --model-name sentence-transformers/all-MiniLM-L6-v2

Reads:
    data/sekd/processed/chunks.jsonl
    data/sekd/queries.jsonl
    data/sekd/ground_truth.jsonl

Writes:
    outputs/evaluation/dense/metrics.json
    outputs/evaluation/dense/per_category.csv
    outputs/evaluation/dense/per_reasoning_type.csv
    outputs/evaluation/dense/retrieval_results.jsonl
    outputs/evaluation/dense/failure_analysis.json
    outputs/evaluation/dense/comparison_vs_bm25.json
    outputs/evaluation/dense/run_manifest.json
    outputs/evaluation/dense/index/  (serialized FAISS index + embeddings)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enterprise_rag.evaluation.reporter import write_all_outputs
from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
from enterprise_rag.retrieval.dense import DenseConfig, DenseIndex, build_dense_index

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
    ] = Path("outputs/evaluation/dense"),
    bm25_output_dir: Annotated[
        Path,
        typer.Option(help="BM25 output directory (for comparison)."),
    ] = Path("outputs/evaluation/bm25"),
    model_name: Annotated[
        str,
        typer.Option(help="Sentence-transformers model name."),
    ] = "BAAI/bge-small-en-v1.5",
    k_values: Annotated[
        list[int],
        typer.Option(help="K cutoffs to evaluate."),
    ] = [1, 3, 5, 10],  # noqa: B006
    rebuild_index: Annotated[
        bool,
        typer.Option(help="Force rebuild even if serialized index exists."),
    ] = False,
    answerable_only: Annotated[
        bool,
        typer.Option(help="Skip unanswerable queries."),
    ] = True,
    batch_size: Annotated[
        int,
        typer.Option(help="Encoding batch size."),
    ] = 64,
) -> None:
    """Run dense retrieval and evaluation over the SEKD corpus."""
    t_start = time.monotonic()
    max_k = max(k_values)
    index_dir = output_dir / "index"

    # ── 1. Load chunks ─────────────────────────────────────────────────────
    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[dict] = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    typer.echo(f"  {len(chunks)} chunks loaded")

    # ── 2. Build or load dense index ───────────────────────────────────────
    config = DenseConfig(model_name=model_name, batch_size=batch_size, max_k=max_k)
    if not rebuild_index and (index_dir / "faiss.index").exists():
        typer.echo(f"Loading dense index from {index_dir} ...")
        t_load = time.monotonic()
        index = DenseIndex.load(index_dir, config=config)
        typer.echo(f"  Loaded in {time.monotonic() - t_load:.2f}s")
    else:
        typer.echo(f"Building dense index with {model_name} ...")
        index = build_dense_index(chunks, index_dir=index_dir, config=config)
        info = index.model_info
        typer.echo(
            f"  dim={info['embedding_dim']}, "
            f"build_time={info['build_duration_seconds']:.1f}s"
        )

    model_info = index.model_info
    typer.echo(
        f"  model={model_info['model_name']}, "
        f"dim={model_info['embedding_dim']}, "
        f"corpus={model_info['corpus_size']}"
    )

    # ── 3. Load queries ────────────────────────────────────────────────────
    typer.echo(f"Loading queries from {queries_path} ...")
    queries: list[dict] = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    typer.echo(f"  {len(queries)} queries loaded")

    # ── 4. Load ground truths ──────────────────────────────────────────────
    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = json.loads(line)
                ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])

    queries_meta: dict[str, dict] = {q["query_id"]: q for q in queries}

    # ── 5. Run dense retrieval ─────────────────────────────────────────────
    typer.echo(f"Running dense retrieval ({len(queries)} queries, k={max_k}) ...")
    t_ret = time.monotonic()
    run = index.run(queries, k=max_k, method="dense")
    elapsed_ret = time.monotonic() - t_ret
    typer.echo(f"  {len(run.results)} results in {elapsed_ret:.2f}s")
    typer.echo(f"  Avg latency per query: {elapsed_ret / len(queries) * 1000:.1f}ms")

    # ── 6. Evaluate ────────────────────────────────────────────────────────
    typer.echo(f"Evaluating at k ∈ {sorted(k_values)} ...")
    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(
        retrieval_results=retrieval_results,
        ground_truths=ground_truths,
        queries_meta=queries_meta,
        k_values=k_values,
        answerable_only=answerable_only,
    )

    # ── 7. Write outputs ───────────────────────────────────────────────────
    typer.echo(f"Writing outputs to {output_dir} ...")
    bm25_metrics = bm25_output_dir / "metrics.json"
    paths = write_all_outputs(
        result_by_k=result_by_k,
        run=run,
        queries_by_id=queries_meta,
        output_dir=output_dir,
        bm25_metrics_path=bm25_metrics if bm25_metrics.exists() else None,
    )
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")

    # ── 8. Write run manifest ──────────────────────────────────────────────
    elapsed_total = time.monotonic() - t_start
    manifest = {
        "method": "dense",
        "model_name": model_info["model_name"],
        "embedding_dim": model_info["embedding_dim"],
        "corpus_size": model_info["corpus_size"],
        "build_duration_seconds": model_info["build_duration_seconds"],
        "retrieval_duration_seconds": round(elapsed_ret, 3),
        "total_duration_seconds": round(elapsed_total, 3),
        "avg_latency_ms": round(elapsed_ret / len(queries) * 1000, 2),
        "n_queries": len(queries),
        "k_values": sorted(k_values),
        "answerable_only": answerable_only,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"  run_manifest: {manifest_path}")

    # ── 9. Print evaluation summary ────────────────────────────────────────
    typer.echo(f"\n{'─' * 68}")
    typer.echo(f"Dense Retrieval Summary  (model: {model_name}, total: {elapsed_total:.1f}s)")
    typer.echo(f"{'─' * 68}")
    typer.echo(f"{'k':>4}  {'Recall':>7}  {'Prec':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}  {'Queries':>7}")
    typer.echo(f"{'─' * 68}")
    for k in sorted(k_values):
        qms = result_by_k[k]
        agg = aggregate(qms, k=k)
        typer.echo(
            f"{k:>4}  {agg.recall:>7.4f}  {agg.precision:>7.4f}  "
            f"{agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}  {agg.n_queries:>7}"
        )
    typer.echo(f"{'─' * 68}")

    qms_max = result_by_k[max_k]
    n_zero = sum(1 for m in qms_max if m.recall == 0.0)
    n_perfect = sum(1 for m in qms_max if m.recall == 1.0)
    typer.echo(
        f"\nAt k={max_k}: {n_zero}/{len(qms_max)} zero-recall, "
        f"{n_perfect}/{len(qms_max)} perfect-recall"
    )

    if "comparison_vs_bm25_json" in paths:
        typer.echo(f"BM25 comparison: {paths['comparison_vs_bm25_json']}")


if __name__ == "__main__":
    app()
