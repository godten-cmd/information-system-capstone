"""Standalone CLI script: generate SEKD queries and ground truths."""

from __future__ import annotations

import time
from pathlib import Path

import jsonlines
import typer

app = typer.Typer(add_completion=False)


@app.command()
def main(
    documents_path: Path = typer.Option(
        Path("data/sekd/raw/documents.jsonl"),
        help="Path to the documents.jsonl file.",
    ),
    chunks_path: Path = typer.Option(
        Path("data/sekd/processed/chunks.jsonl"),
        help="Path to the chunks.jsonl file.",
    ),
    output_dir: Path = typer.Option(
        Path("data/sekd"),
        help="Directory to write queries.jsonl and ground_truth.jsonl.",
    ),
    seed: int = typer.Option(42, help="Random seed for deterministic generation."),
) -> None:
    """Generate queries and ground truths for the SEKD corpus."""
    from enterprise_rag.dataset.generate_queries import (
        QueryConfig,
        QueryGenerationReport,
        SEKDQueryGenerator,
        build_generation_report,
    )
    from enterprise_rag.dataset.schemas import Chunk, EnterpriseDocument

    typer.echo(f"Loading documents from {documents_path} ...")
    docs: list[EnterpriseDocument] = []
    with jsonlines.open(documents_path) as reader:
        for record in reader:
            docs.append(EnterpriseDocument.model_validate(record))
    typer.echo(f"  Loaded {len(docs)} documents.")

    typer.echo(f"Loading chunks from {chunks_path} ...")
    chunks: list[Chunk] = []
    with jsonlines.open(chunks_path) as reader:
        for record in reader:
            chunks.append(Chunk.model_validate(record))
    typer.echo(f"  Loaded {len(chunks)} chunks.")

    typer.echo(f"Generating queries (seed={seed}) ...")
    config = QueryConfig(seed=seed)
    generator = SEKDQueryGenerator(config=config)

    t0 = time.monotonic()
    queries, ground_truths = generator.generate(docs, chunks)
    elapsed = time.monotonic() - t0

    output_dir.mkdir(parents=True, exist_ok=True)
    queries_path = output_dir / "queries.jsonl"
    gt_path = output_dir / "ground_truth.jsonl"

    typer.echo(f"Writing {len(queries)} queries to {queries_path} ...")
    with jsonlines.open(queries_path, mode="w") as writer:
        for q in queries:
            writer.write(q.model_dump(by_alias=True, mode="json"))

    typer.echo(f"Writing {len(ground_truths)} ground truths to {gt_path} ...")
    with jsonlines.open(gt_path, mode="w") as writer:
        for gt in ground_truths:
            writer.write(gt.model_dump(by_alias=True, mode="json"))

    report: QueryGenerationReport = build_generation_report(queries, elapsed)

    typer.echo(f"\nDone in {elapsed:.1f}s:")
    typer.echo(f"  Total queries    : {report.total_queries}")
    typer.echo(f"  Answerable       : {report.answerable_count}")
    typer.echo(f"  Unanswerable     : {report.unanswerable_count}")
    typer.echo("\nPer category:")
    for cat, cnt in sorted(report.per_category.items()):
        typer.echo(f"  {cat}: {cnt}")
    typer.echo("\nPer reasoning type:")
    for rt, cnt in sorted(report.per_reasoning_type.items()):
        typer.echo(f"  {rt}: {cnt}")
    typer.echo("\nPer difficulty:")
    for diff, cnt in sorted(report.per_difficulty.items()):
        typer.echo(f"  {diff}: {cnt}")
    typer.echo("\nPer oracle strategy:")
    for strat, cnt in sorted(report.per_oracle_strategy.items()):
        typer.echo(f"  {strat}: {cnt}")


if __name__ == "__main__":
    app()
