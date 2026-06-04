"""Command-line interface for the Enterprise RAG research prototype."""

from pathlib import Path
from typing import Annotated

import typer

from enterprise_rag.version import __version__

app = typer.Typer(
    name="enterprise-rag",
    help="Research prototype CLI for Agentic RAG enterprise experiments.",
    no_args_is_help=True,
)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def status(
    phase: Annotated[
        str,
        typer.Option(help="Implementation phase to report."),
    ] = "phase1",
) -> None:
    """Print the current implementation status."""
    typer.echo(f"{phase}: repository foundation ready")


@app.command(name="generate-documents")
def generate_documents(
    seed: Annotated[int, typer.Option(help="Random seed for deterministic generation.")] = 42,
    document_count: Annotated[int, typer.Option(help="Total number of documents to generate.")] = 120,
    output_dir: Annotated[Path, typer.Option(help="Directory to write output files.")] = Path("data/sekd/raw"),
) -> None:
    """Generate the SEKD synthetic enterprise document corpus."""
    from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator

    config = DatasetConfig(seed=seed, document_count=document_count)
    generator = SEKDGenerator(config=config, seed=seed)

    typer.echo(f"Generating {document_count} documents with seed={seed} ...")
    documents = generator.generate()

    typer.echo(f"Writing to {output_dir} ...")
    report = generator.write(documents, output_dir)

    typer.echo(f"Done: {report.total_documents} documents in {report.duration_seconds:.1f}s")
    for category, count in sorted(report.per_category.items()):
        typer.echo(f"  {category}: {count}")
    if report.validation_errors:
        typer.echo("Validation errors:")
        for err in report.validation_errors:
            typer.echo(f"  [!] {err}")


@app.command(name="build-chunks")
def build_chunks(
    input_dir: Annotated[Path, typer.Option(help="Directory containing documents.jsonl.")] = Path("data/sekd/raw"),
    output_dir: Annotated[Path, typer.Option(help="Directory to write chunks.jsonl.")] = Path("data/sekd/processed"),
    strategy: Annotated[str, typer.Option(help="Chunking strategy: section_aware, fixed_size, recursive.")] = "section_aware",
    max_tokens: Annotated[int, typer.Option(help="Maximum tokens per chunk.")] = 400,
) -> None:
    """Chunk SEKD documents into retrievable pieces."""
    import json
    import time

    import jsonlines

    from enterprise_rag.dataset.schemas import EnterpriseDocument
    from enterprise_rag.preprocessing.chunking import (
        ChunkingConfig,
        ChunkingPipeline,
        build_quality_report,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    documents_path = input_dir / "documents.jsonl"

    typer.echo(f"Loading documents from {documents_path} ...")
    docs: list[EnterpriseDocument] = []
    with jsonlines.open(documents_path) as reader:
        for record in reader:
            docs.append(EnterpriseDocument.model_validate(record))

    typer.echo(f"Chunking {len(docs)} documents (strategy={strategy}) ...")
    cfg = ChunkingConfig(strategy=strategy, max_tokens=max_tokens)  # type: ignore[arg-type]
    pipeline = ChunkingPipeline(config=cfg)

    t0 = time.monotonic()
    chunks = pipeline.chunk_corpus(docs)
    elapsed = time.monotonic() - t0

    chunks_path = output_dir / "chunks.jsonl"
    with jsonlines.open(chunks_path, mode="w") as writer:
        for chunk in chunks:
            writer.write(chunk.model_dump(by_alias=True, mode="json"))

    report = build_quality_report(chunks, docs)
    stats = report.statistics
    typer.echo(
        f"Done: {stats.total_chunks} chunks from {stats.docs_with_chunks} docs "
        f"in {elapsed:.1f}s"
    )
    typer.echo(f"Avg tokens/chunk: {stats.avg_tokens:.1f}")
    for cat, cnt in sorted(stats.per_category.items()):
        typer.echo(f"  {cat}: {cnt}")
    if report.validation_errors:
        typer.echo("Metadata errors:")
        for e in report.validation_errors:
            typer.echo(f"  [!] {e}")


@app.command(name="generate-queries")
def generate_queries(
    documents_path: Annotated[
        Path,
        typer.Option(help="Path to documents.jsonl."),
    ] = Path("data/sekd/raw/documents.jsonl"),
    chunks_path: Annotated[
        Path,
        typer.Option(help="Path to chunks.jsonl."),
    ] = Path("data/sekd/processed/chunks.jsonl"),
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write queries.jsonl and ground_truth.jsonl."),
    ] = Path("data/sekd"),
    seed: Annotated[int, typer.Option(help="Random seed for deterministic generation.")] = 42,
) -> None:
    """Generate SEKD queries and ground truths from the document corpus."""
    import time

    import jsonlines

    from enterprise_rag.dataset.generate_queries import (
        QueryConfig,
        SEKDQueryGenerator,
        build_generation_report,
    )
    from enterprise_rag.dataset.schemas import Chunk, EnterpriseDocument

    typer.echo(f"Loading documents from {documents_path} ...")
    docs: list[EnterpriseDocument] = []
    with jsonlines.open(documents_path) as reader:
        for record in reader:
            docs.append(EnterpriseDocument.model_validate(record))

    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[Chunk] = []
    with jsonlines.open(chunks_path) as reader:
        for record in reader:
            chunks.append(Chunk.model_validate(record))

    typer.echo(f"Generating queries with seed={seed} ...")
    config = QueryConfig(seed=seed)
    generator = SEKDQueryGenerator(config=config)

    t0 = time.monotonic()
    queries, ground_truths = generator.generate(docs, chunks)
    elapsed = time.monotonic() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    gt_path = output_dir / "ground_truth.jsonl"

    with jsonlines.open(queries_path, mode="w") as writer:
        for q in queries:
            writer.write(q.model_dump(by_alias=True, mode="json"))

    with jsonlines.open(gt_path, mode="w") as writer:
        for gt in ground_truths:
            writer.write(gt.model_dump(by_alias=True, mode="json"))

    report = build_generation_report(queries, elapsed)
    typer.echo(f"Done: {report.total_queries} queries in {elapsed:.1f}s")
    typer.echo(f"  Answerable: {report.answerable_count}  Unanswerable: {report.unanswerable_count}")
    for cat, cnt in sorted(report.per_category.items()):
        typer.echo(f"  {cat}: {cnt}")


@app.command(name="run-hybrid")
def run_hybrid(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write evaluation outputs."),
    ] = Path("outputs/evaluation/hybrid"),
    bm25_index_dir: Annotated[
        Path,
        typer.Option(help="BM25 index directory."),
    ] = Path("outputs/evaluation/bm25/index"),
    dense_index_dir: Annotated[
        Path,
        typer.Option(help="Dense index directory."),
    ] = Path("outputs/evaluation/dense/index"),
    candidate_k: Annotated[
        int,
        typer.Option(help="Candidates retrieved from each method before fusion."),
    ] = 100,
) -> None:
    """Run hybrid BM25+Dense retrieval via RRF and evaluate."""
    import json
    import time

    from enterprise_rag.evaluation.contribution_analysis import analyze_contributions
    from enterprise_rag.evaluation.reporter import write_all_outputs
    from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
    from enterprise_rag.retrieval.bm25 import BM25Index
    from enterprise_rag.retrieval.dense import DenseIndex
    from enterprise_rag.retrieval.fusion import reciprocal_rank_fusion
    from enterprise_rag.retrieval.hybrid import HybridConfig, HybridIndex
    from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk

    chunks_path = Path("data/sekd/processed/chunks.jsonl")
    queries_path = Path("data/sekd/queries.jsonl")
    ground_truth_path = Path("data/sekd/ground_truth.jsonl")
    k_values = [1, 3, 5, 10]
    max_k = 10

    def _load_jsonl(p):
        records = []
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    queries = _load_jsonl(queries_path)
    ground_truths = {gt["query_id"]: gt.get("required_chunk_ids", [])
                     for gt in _load_jsonl(ground_truth_path)}
    queries_meta = {q["query_id"]: q for q in queries}

    typer.echo("Loading indexes ...")
    bm25_index = BM25Index.load(bm25_index_dir)
    dense_index = DenseIndex.load(dense_index_dir)

    typer.echo(f"Running candidate retrieval (k={candidate_k}) ...")
    bm25_run_full = bm25_index.run(queries, k=candidate_k)
    dense_run_full = dense_index.run(queries, k=candidate_k)

    bm25_ranked = {}
    for r in bm25_run_full.results:
        bm25_ranked.setdefault(r.query_id, []).append(r.chunk_id)
    dense_ranked = {}
    for r in dense_run_full.results:
        dense_ranked.setdefault(r.query_id, []).append(r.chunk_id)

    hybrid_base = HybridIndex(bm25_index, dense_index, HybridConfig(rrf_k=60, candidate_k=candidate_k))

    typer.echo("RRF parameter sweep ...")
    best_rrf_k, best_recall = 60, 0.0
    for rrf_k in [10, 30, 60, 100]:
        hi = hybrid_base.with_rrf_k(rrf_k)
        h_run = RetrievalRun(method="hybrid", k=max_k, config={"rrf_k": rrf_k})
        chunk_meta = hybrid_base._chunk_meta
        for q in queries:
            qid = q["query_id"]
            fused = reciprocal_rank_fusion(
                [bm25_ranked.get(qid, []), dense_ranked.get(qid, [])],
                rrf_k=rrf_k, top_n=max_k,
            )
            for rank, (cid, score) in enumerate(fused, start=1):
                m = chunk_meta.get(cid, {})
                h_run.results.append(RetrievedChunk(
                    query_id=qid, chunk_id=cid, document_id=m.get("document_id", ""),
                    rank=rank, score=round(score, 6),
                    category=m.get("category", ""), section_path=m.get("section_path", []),
                ))
        res = evaluate_run([r.to_dict() for r in h_run.results], ground_truths,
                           queries_meta, k_values=k_values, answerable_only=True)
        recall = aggregate(res[max_k], k=max_k).recall
        typer.echo(f"  rrf_k={rrf_k}  Recall@10={recall:.4f}")
        if recall > best_recall:
            best_recall = recall
            best_rrf_k = rrf_k

    typer.echo(f"Best rrf_k={best_rrf_k} (Recall@10={best_recall:.4f})")
    hi_best = hybrid_base.with_rrf_k(best_rrf_k)
    best_run = RetrievalRun(method="hybrid", k=max_k, config={"rrf_k": best_rrf_k, "candidate_k": candidate_k})
    for q in queries:
        qid = q["query_id"]
        fused = reciprocal_rank_fusion(
            [bm25_ranked.get(qid, []), dense_ranked.get(qid, [])],
            rrf_k=best_rrf_k, top_n=max_k,
        )
        chunk_meta = hi_best._chunk_meta
        for rank, (cid, score) in enumerate(fused, start=1):
            m = chunk_meta.get(cid, {})
            best_run.results.append(RetrievedChunk(
                query_id=qid, chunk_id=cid, document_id=m.get("document_id", ""),
                rank=rank, score=round(score, 6),
                category=m.get("category", ""), section_path=m.get("section_path", []),
            ))
    result_by_k = evaluate_run([r.to_dict() for r in best_run.results], ground_truths,
                                queries_meta, k_values=k_values, answerable_only=True)

    bm25_metrics = Path("outputs/evaluation/bm25/metrics.json")
    dense_metrics = Path("outputs/evaluation/dense/metrics.json")
    write_all_outputs(
        result_by_k=result_by_k, run=best_run, queries_by_id=queries_meta,
        output_dir=output_dir,
        bm25_metrics_path=bm25_metrics if bm25_metrics.exists() else None,
        dense_metrics_path=dense_metrics if dense_metrics.exists() else None,
    )

    typer.echo(f"\n{'k':>4}  {'Recall':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}")
    for k in k_values:
        agg = aggregate(result_by_k[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:>7.4f}  {agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}")


@app.command(name="run-dense")
def run_dense(
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
    model_name: Annotated[
        str,
        typer.Option(help="Sentence-transformers model name."),
    ] = "BAAI/bge-small-en-v1.5",
    rebuild_index: Annotated[
        bool,
        typer.Option(help="Force rebuild even if serialized index exists."),
    ] = False,
) -> None:
    """Run dense retrieval and evaluation over the SEKD corpus."""
    import json
    import time

    from enterprise_rag.evaluation.reporter import write_all_outputs
    from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
    from enterprise_rag.retrieval.dense import DenseConfig, DenseIndex, build_dense_index

    k_values = [1, 3, 5, 10]
    max_k = 10
    index_dir = output_dir / "index"
    bm25_metrics = Path("outputs/evaluation/bm25/metrics.json")

    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[dict] = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    config = DenseConfig(model_name=model_name, max_k=max_k)
    if not rebuild_index and (index_dir / "faiss.index").exists():
        typer.echo(f"Loading dense index from {index_dir} ...")
        index = DenseIndex.load(index_dir, config=config)
    else:
        typer.echo(f"Building dense index with {model_name} ...")
        index = build_dense_index(chunks, index_dir=index_dir, config=config)

    typer.echo(f"Loading queries from {queries_path} ...")
    queries: list[dict] = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = json.loads(line)
                ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])

    queries_meta: dict[str, dict] = {q["query_id"]: q for q in queries}

    typer.echo(f"Running dense retrieval ({len(queries)} queries, k={max_k}) ...")
    t0 = time.monotonic()
    run = index.run(queries, k=max_k, method="dense")
    elapsed = time.monotonic() - t0
    typer.echo(f"  {len(run.results)} results in {elapsed:.2f}s")

    typer.echo(f"Evaluating at k ∈ {k_values} ...")
    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(
        retrieval_results=retrieval_results,
        ground_truths=ground_truths,
        queries_meta=queries_meta,
        k_values=k_values,
        answerable_only=True,
    )

    typer.echo(f"Writing outputs to {output_dir} ...")
    write_all_outputs(
        result_by_k=result_by_k,
        run=run,
        queries_by_id=queries_meta,
        output_dir=output_dir,
        bm25_metrics_path=bm25_metrics if bm25_metrics.exists() else None,
    )

    typer.echo(f"\n{'k':>4}  {'Recall':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}")
    for k in k_values:
        agg = aggregate(result_by_k[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:>7.4f}  {agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}")


@app.command(name="run-bm25")
def run_bm25(
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
    rebuild_index: Annotated[
        bool,
        typer.Option(help="Force rebuild even if serialized index exists."),
    ] = False,
) -> None:
    """Run BM25 retrieval and evaluation over the SEKD corpus."""
    import json
    import time

    from enterprise_rag.evaluation.reporter import write_all_outputs
    from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run
    from enterprise_rag.retrieval.bm25 import BM25Config, BM25Index, build_bm25_index

    k_values = [1, 3, 5, 10]
    max_k = 10
    index_dir = output_dir / "index"

    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[dict] = []
    with open(chunks_path) as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    config = BM25Config(max_k=max_k)
    if not rebuild_index and (index_dir / "bm25.pkl").exists():
        typer.echo(f"Loading BM25 index from {index_dir} ...")
        index = BM25Index.load(index_dir, config=config)
    else:
        typer.echo(f"Building BM25 index ({len(chunks)} chunks) ...")
        index = build_bm25_index(chunks, index_dir=index_dir, config=config)

    typer.echo(f"Loading queries from {queries_path} ...")
    queries: list[dict] = []
    with open(queries_path) as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))

    typer.echo(f"Loading ground truths from {ground_truth_path} ...")
    ground_truths: dict[str, list[str]] = {}
    with open(ground_truth_path) as f:
        for line in f:
            line = line.strip()
            if line:
                gt = json.loads(line)
                ground_truths[gt["query_id"]] = gt.get("required_chunk_ids", [])

    queries_meta: dict[str, dict] = {q["query_id"]: q for q in queries}

    typer.echo(f"Running retrieval ({len(queries)} queries, k={max_k}) ...")
    t0 = time.monotonic()
    run = index.run(queries, k=max_k, method="bm25")
    elapsed = time.monotonic() - t0
    typer.echo(f"  {len(run.results)} results in {elapsed:.2f}s")

    typer.echo(f"Evaluating at k ∈ {k_values} ...")
    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(
        retrieval_results=retrieval_results,
        ground_truths=ground_truths,
        queries_meta=queries_meta,
        k_values=k_values,
        answerable_only=True,
    )

    typer.echo(f"Writing outputs to {output_dir} ...")
    write_all_outputs(
        result_by_k=result_by_k,
        run=run,
        queries_by_id=queries_meta,
        output_dir=output_dir,
    )

    typer.echo(f"\n{'k':>4}  {'Recall':>7}  {'MRR':>7}  {'nDCG':>7}  {'Hit':>7}")
    for k in k_values:
        agg = aggregate(result_by_k[k], k=k)
        typer.echo(f"{k:>4}  {agg.recall:>7.4f}  {agg.mrr:>7.4f}  {agg.ndcg:>7.4f}  {agg.hit_rate:>7.4f}")


@app.command(name="run-rule-planner")
def run_rule_planner(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write evaluation outputs."),
    ] = Path("outputs/evaluation/rule_planner"),
    answerable_only: Annotated[
        bool,
        typer.Option(help="Skip unanswerable queries."),
    ] = True,
) -> None:
    """Run the Rule-Based Retrieval Planner and evaluate against fixed baselines."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/run_rule_planner.py",
        f"--output-dir={output_dir}",
    ]
    if not answerable_only:
        cmd.append("--no-answerable-only")
    subprocess.run(cmd, check=True)


@app.command(name="run-llm-planner")
def run_llm_planner(
    output_dir: Annotated[
        Path,
        typer.Option(help="Directory to write evaluation outputs."),
    ] = Path("outputs/evaluation/llm_planner"),
    provider: Annotated[
        str,
        typer.Option(help="LLM provider: openai | anthropic | mock"),
    ] = "mock",
    prompt_version: Annotated[
        str,
        typer.Option(help="Prompt version: zero_shot | structured | few_shot"),
    ] = "structured",
    run_prompt_comparison: Annotated[
        bool,
        typer.Option(help="Run all three prompt versions for comparison."),
    ] = False,
    model: Annotated[
        str | None,
        typer.Option(help="Model name override."),
    ] = None,
) -> None:
    """Run the LLM-Based Retrieval Planner and evaluate against fixed baselines."""
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/run_llm_planner.py",
        f"--output-dir={output_dir}",
        f"--provider={provider}",
        f"--prompt-version={prompt_version}",
    ]
    if run_prompt_comparison:
        cmd.append("--run-prompt-comparison")
    if model:
        cmd.append(f"--model={model}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()

