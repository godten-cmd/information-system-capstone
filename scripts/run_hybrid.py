"""Hybrid retrieval runner: BM25 + Dense fused via RRF, with rrf_k sweep.

Usage:
    python scripts/run_hybrid.py
    python scripts/run_hybrid.py --candidate-k 100

Loads existing indexes from:
    outputs/evaluation/bm25/index/
    outputs/evaluation/dense/index/

Writes:
    outputs/evaluation/hybrid/metrics.json
    outputs/evaluation/hybrid/per_category.csv
    outputs/evaluation/hybrid/per_reasoning_type.csv
    outputs/evaluation/hybrid/retrieval_results.jsonl
    outputs/evaluation/hybrid/failure_analysis.json
    outputs/evaluation/hybrid/comparison_vs_bm25.json
    outputs/evaluation/hybrid/comparison_vs_dense.json
    outputs/evaluation/hybrid/rrf_sweep.json
    outputs/evaluation/hybrid/contribution_analysis.json
    outputs/evaluation/hybrid/run_manifest.json
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

# Limit OpenMP threads to avoid SIGSEGV in torch on Python 3.14 when loading
# the dense model after BM25 retrieval has run (memory/threading interaction).
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enterprise_rag.evaluation.contribution_analysis import analyze_contributions
from enterprise_rag.evaluation.reporter import write_all_outputs
from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
from enterprise_rag.retrieval.bm25 import BM25Config, BM25Index
from enterprise_rag.retrieval.dense import DenseConfig, DenseIndex
from enterprise_rag.retrieval.hybrid import HybridConfig, HybridIndex

app = typer.Typer(add_completion=False)

_RRF_K_VALUES = [10, 30, 60, 100]


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@app.command()
def main(
    chunks_path: Annotated[
        Path, typer.Option(help="Path to chunks.jsonl.")
    ] = Path("data/sekd/processed/chunks.jsonl"),
    queries_path: Annotated[
        Path, typer.Option(help="Path to queries.jsonl.")
    ] = Path("data/sekd/queries.jsonl"),
    ground_truth_path: Annotated[
        Path, typer.Option(help="Path to ground_truth.jsonl.")
    ] = Path("data/sekd/ground_truth.jsonl"),
    bm25_index_dir: Annotated[
        Path, typer.Option(help="Directory containing BM25 index.")
    ] = Path("outputs/evaluation/bm25/index"),
    dense_index_dir: Annotated[
        Path, typer.Option(help="Directory containing Dense index.")
    ] = Path("outputs/evaluation/dense/index"),
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write evaluation outputs.")
    ] = Path("outputs/evaluation/hybrid"),
    k_values: Annotated[
        list[int], typer.Option(help="K cutoffs to evaluate.")
    ] = [1, 3, 5, 10],  # noqa: B006
    candidate_k: Annotated[
        int, typer.Option(help="Candidates retrieved from each method before fusion.")
    ] = 100,
    answerable_only: Annotated[
        bool, typer.Option(help="Skip unanswerable queries.")
    ] = True,
) -> None:
    """Run hybrid BM25+Dense retrieval via RRF, sweep rrf_k, write all outputs."""
    t_start = time.monotonic()
    max_k = max(k_values)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks = _load_jsonl(chunks_path)
    typer.echo(f"  {len(chunks)} chunks")

    typer.echo(f"Loading queries from {queries_path} ...")
    queries = _load_jsonl(queries_path)
    typer.echo(f"  {len(queries)} queries")

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    for gt in _load_jsonl(ground_truth_path):
        ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])

    queries_meta = {q["query_id"]: q for q in queries}

    # ── 2. Load indexes ───────────────────────────────────────────────────────
    typer.echo(f"Loading BM25 index from {bm25_index_dir} ...")
    bm25_index = BM25Index.load(bm25_index_dir)
    typer.echo(f"  {bm25_index._meta.corpus_size} chunks")

    typer.echo(f"Loading dense index from {dense_index_dir} ...")
    dense_index = DenseIndex.load(dense_index_dir)
    typer.echo(f"  model={dense_index._vector_index._meta.model_name}, "
               f"dim={dense_index._vector_index._meta.embedding_dim}")

    # ── 3. Run candidate retrieval once (for all rrf_k values) ───────────────
    typer.echo(f"Running BM25 candidate retrieval (k={candidate_k}) ...")
    t_bm25 = time.monotonic()
    bm25_run_full = bm25_index.run(queries, k=candidate_k, method="bm25")
    bm25_elapsed = time.monotonic() - t_bm25
    typer.echo(f"  {len(bm25_run_full.results)} results in {bm25_elapsed:.2f}s")

    # Extract ranked lists immediately and free the large result object before
    # Dense model loading to avoid memory pressure causing SIGSEGV on Python 3.14.
    bm25_ranked: dict[str, list[str]] = {}
    for r in bm25_run_full.results:
        bm25_ranked.setdefault(r.query_id, []).append(r.chunk_id)
    del bm25_run_full
    import gc; gc.collect()

    # Reduce batch size to avoid native memory issues (SIGSEGV) on Python 3.14 + torch.
    dense_index._config.batch_size = 1

    typer.echo(f"Running Dense candidate retrieval (k={candidate_k}) ...")
    t_dense = time.monotonic()
    dense_run_full = dense_index.run(queries, k=candidate_k, method="dense")
    dense_elapsed = time.monotonic() - t_dense
    typer.echo(f"  {len(dense_run_full.results)} results in {dense_elapsed:.2f}s")

    dense_ranked: dict[str, list[str]] = {}
    for r in dense_run_full.results:
        dense_ranked.setdefault(r.query_id, []).append(r.chunk_id)
    del dense_run_full
    gc.collect()

    # ── 4. RRF parameter sweep ────────────────────────────────────────────────
    # Build a HybridIndex just for the metadata lookup; we'll do fusion inline
    hybrid_base = HybridIndex(bm25_index, dense_index, HybridConfig(rrf_k=60, candidate_k=candidate_k))

    typer.echo(f"\nRRF sweep over rrf_k ∈ {_RRF_K_VALUES} ...")
    sweep_results: list[dict] = []
    best_rrf_k = 60
    best_recall10 = 0.0

    for rrf_k in _RRF_K_VALUES:
        # Apply fusion on pre-computed ranked lists
        hi = hybrid_base.with_rrf_k(rrf_k)
        hybrid_run_k = _apply_fusion_from_ranked(
            queries, bm25_ranked, dense_ranked, hi, k=max_k
        )
        ret_results = [r.to_dict() for r in hybrid_run_k.results]
        res_by_k = evaluate_run(ret_results, ground_truths, queries_meta,
                                k_values=k_values, answerable_only=answerable_only)
        agg10 = aggregate(res_by_k[max_k], k=max_k)
        recall10 = agg10.recall
        ndcg10 = agg10.ndcg

        sweep_entry = {
            "rrf_k": rrf_k,
            "recall_at_10": round(recall10, 4),
            "ndcg_at_10": round(ndcg10, 4),
            "mrr": round(agg10.mrr, 4),
            "hit_rate_at_10": round(agg10.hit_rate, 4),
        }
        sweep_results.append(sweep_entry)
        typer.echo(f"  rrf_k={rrf_k:>3}  Recall@10={recall10:.4f}  nDCG@10={ndcg10:.4f}  MRR={agg10.mrr:.4f}")

        if recall10 > best_recall10:
            best_recall10 = recall10
            best_rrf_k = rrf_k

    typer.echo(f"\nBest rrf_k = {best_rrf_k}  (Recall@10 = {best_recall10:.4f})")

    # ── 5. Final run with best rrf_k ──────────────────────────────────────────
    hi_best = hybrid_base.with_rrf_k(best_rrf_k)
    typer.echo(f"Running final hybrid retrieval (rrf_k={best_rrf_k}, k={max_k}) ...")
    hybrid_run_best = _apply_fusion_from_ranked(
        queries, bm25_ranked, dense_ranked, hi_best, k=max_k
    )
    ret_results_best = [r.to_dict() for r in hybrid_run_best.results]
    result_by_k_best = evaluate_run(
        ret_results_best, ground_truths, queries_meta,
        k_values=k_values, answerable_only=answerable_only,
    )

    # ── 6. Contribution analysis ──────────────────────────────────────────────
    typer.echo("Running contribution analysis ...")
    # Rebuild lightweight runs from ranked dicts (full run objects freed earlier)
    bm25_run_10 = _run_from_ranked(bm25_ranked, queries, method="bm25", k=max_k)
    dense_run_10 = _run_from_ranked(dense_ranked, queries, method="dense", k=max_k)
    contrib = analyze_contributions(
        ground_truths=ground_truths,
        bm25_run=bm25_run_10,
        dense_run=dense_run_10,
        hybrid_run=hybrid_run_best,
        queries_meta=queries_meta,
        k=max_k,
        answerable_only=answerable_only,
    )

    # ── 7. Write all outputs ──────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Writing outputs to {output_dir} ...")

    bm25_metrics = Path("outputs/evaluation/bm25/metrics.json")
    dense_metrics = Path("outputs/evaluation/dense/metrics.json")
    paths = write_all_outputs(
        result_by_k=result_by_k_best,
        run=hybrid_run_best,
        queries_by_id=queries_meta,
        output_dir=output_dir,
        bm25_metrics_path=bm25_metrics if bm25_metrics.exists() else None,
        dense_metrics_path=dense_metrics if dense_metrics.exists() else None,
    )
    for name, path in paths.items():
        typer.echo(f"  {name}: {path}")

    # Sweep results
    sweep_path = output_dir / "rrf_sweep.json"
    sweep_path.write_text(json.dumps({
        "candidate_k": candidate_k,
        "rrf_k_values_tested": _RRF_K_VALUES,
        "best_rrf_k": best_rrf_k,
        "results": sweep_results,
    }, indent=2))
    typer.echo(f"  rrf_sweep: {sweep_path}")

    # Contribution analysis
    contrib_path = output_dir / "contribution_analysis.json"
    contrib_path.write_text(json.dumps(contrib, indent=2, default=str))
    typer.echo(f"  contribution_analysis: {contrib_path}")

    # Run manifest
    elapsed_total = time.monotonic() - t_start
    manifest = {
        "method": "hybrid",
        "rrf_k": best_rrf_k,
        "candidate_k": candidate_k,
        "bm25_corpus_size": bm25_index._meta.corpus_size,
        "dense_model": dense_index._vector_index._meta.model_name,
        "dense_dim": dense_index._vector_index._meta.embedding_dim,
        "n_queries": len(queries),
        "k_values": sorted(k_values),
        "answerable_only": answerable_only,
        "total_duration_seconds": round(elapsed_total, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rrf_sweep_best": sweep_results[[r["rrf_k"] for r in sweep_results].index(best_rrf_k)],
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    typer.echo(f"  run_manifest: {manifest_path}")

    # ── 8. Summary table ──────────────────────────────────────────────────────
    typer.echo(f"\n{'─' * 68}")
    typer.echo(f"Hybrid Retrieval Summary  (rrf_k={best_rrf_k}, candidate_k={candidate_k})")
    typer.echo(f"{'─' * 68}")
    typer.echo(f"{'k':>4}  {'Recall':>7}  {'Prec':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}  {'Queries':>7}")
    typer.echo(f"{'─' * 68}")
    for k in sorted(k_values):
        agg = aggregate(result_by_k_best[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:>7.4f}  {agg.precision:>7.4f}  "
                   f"{agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}  {agg.n_queries:>7}")
    typer.echo(f"{'─' * 68}")

    qms_max = result_by_k_best[max_k]
    n_zero = sum(1 for m in qms_max if m.recall == 0.0)
    n_perfect = sum(1 for m in qms_max if m.recall == 1.0)
    typer.echo(f"\nAt k={max_k}: {n_zero}/{len(qms_max)} zero-recall, {n_perfect}/{len(qms_max)} perfect-recall")

    # Contribution summary
    ov = contrib["overall"]
    typer.echo(f"\nContribution analysis (k={max_k}):")
    typer.echo(f"  BM25-only:    {ov['bm25_only']['count']} chunks ({ov['bm25_only']['pct']}%)")
    typer.echo(f"  Dense-only:   {ov['dense_only']['count']} chunks ({ov['dense_only']['pct']}%)")
    typer.echo(f"  Both:         {ov['both']['count']} chunks ({ov['both']['pct']}%)")
    typer.echo(f"  Hybrid-only:  {ov['hybrid_only']['count']} chunks ({ov['hybrid_only']['pct']}%)")
    typer.echo(f"  Not found:    {ov['not_found']['count']} chunks ({ov['not_found']['pct']}%)")
    typer.echo(f"  Complementarity score: {ov['complementarity_score']}")
    typer.echo(f"  {contrib['interpretation']}")


def _apply_fusion_from_ranked(
    queries: list[dict],
    bm25_ranked: dict[str, list[str]],
    dense_ranked: dict[str, list[str]],
    hybrid_index: HybridIndex,
    k: int,
) -> "RetrievalRun":
    """Apply RRF fusion on pre-computed ranked lists without re-running retrieval."""
    from enterprise_rag.retrieval.fusion import reciprocal_rank_fusion
    from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
    from dataclasses import asdict

    rrf_k = hybrid_index._config.rrf_k
    run = RetrievalRun(
        method="hybrid",
        k=k,
        config={
            "rrf_k": rrf_k,
            "candidate_k": hybrid_index._config.candidate_k,
        },
    )
    chunk_meta = hybrid_index._chunk_meta

    for q in queries:
        qid = q["query_id"]
        bm25_ids = bm25_ranked.get(qid, [])
        dense_ids = dense_ranked.get(qid, [])
        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], rrf_k=rrf_k, top_n=k)
        for rank, (chunk_id, score) in enumerate(fused, start=1):
            m = chunk_meta.get(chunk_id, {})
            run.results.append(RetrievedChunk(
                query_id=qid,
                chunk_id=chunk_id,
                document_id=m.get("document_id", ""),
                rank=rank,
                score=round(score, 6),
                category=m.get("category", ""),
                section_path=m.get("section_path", []),
                text=m.get("text", "")[:200],
            ))
    return run


def _apply_run_at_k(full_run: "RetrievalRun", queries: list[dict], k: int) -> "RetrievalRun":
    """Slice a full run to only the top-k results per query (for contribution analysis)."""
    from enterprise_rag.retrieval.result_schema import RetrievalRun

    sliced = RetrievalRun(method=full_run.method, k=k, config=full_run.config)
    for q in queries:
        qid = q["query_id"]
        sliced.results.extend(full_run.results_for(qid)[:k])
    return sliced


def _run_from_ranked(
    ranked: dict[str, list[str]],
    queries: list[dict],
    method: str,
    k: int,
) -> "RetrievalRun":
    """Build a lightweight RetrievalRun from pre-computed ranked lists (chunk_id only)."""
    from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk

    run = RetrievalRun(method=method, k=k, config={})
    for q in queries:
        qid = q["query_id"]
        for rank, chunk_id in enumerate(ranked.get(qid, [])[:k], start=1):
            run.results.append(RetrievedChunk(
                query_id=qid,
                chunk_id=chunk_id,
                document_id="",
                rank=rank,
                score=0.0,
                category="",
                section_path=[],
                text="",
            ))
    return run


if __name__ == "__main__":
    app()
