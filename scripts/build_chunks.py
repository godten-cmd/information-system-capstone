#!/usr/bin/env python
"""Build chunks.jsonl from documents.jsonl.

Usage:
    python scripts/build_chunks.py
    python scripts/build_chunks.py --input-dir data/sekd/raw \\
        --output-dir data/sekd/processed --strategy section_aware
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Allow running from repository root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jsonlines

from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator
from enterprise_rag.dataset.schemas import EnterpriseDocument
from enterprise_rag.preprocessing.chunking import (
    ChunkingConfig,
    ChunkingPipeline,
    build_quality_report,
)


def load_documents(documents_path: Path) -> list[EnterpriseDocument]:
    docs = []
    with jsonlines.open(documents_path) as reader:
        for record in reader:
            docs.append(EnterpriseDocument.model_validate(record))
    return docs


def main(
    input_dir: Path = Path("data/sekd/raw"),
    output_dir: Path = Path("data/sekd/processed"),
    strategy: str = "section_aware",
    max_tokens: int = 400,
    seed: int = 42,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    documents_path = input_dir / "documents.jsonl"
    if not documents_path.exists():
        # Re-generate if needed
        print(f"documents.jsonl not found at {documents_path}; regenerating ...")
        config = DatasetConfig(seed=seed, document_count=120)
        gen = SEKDGenerator(config=config, seed=seed)
        docs = gen.generate()
        gen.write(docs, input_dir)
    else:
        docs = load_documents(documents_path)

    print(f"Loaded {len(docs)} documents from {documents_path}")
    print(f"Strategy: {strategy}  max_tokens: {max_tokens}")

    cfg = ChunkingConfig(strategy=strategy, max_tokens=max_tokens)  # type: ignore[arg-type]
    pipeline = ChunkingPipeline(config=cfg)

    t0 = time.monotonic()
    chunks = pipeline.chunk_corpus(docs)
    elapsed = time.monotonic() - t0

    # Write chunks.jsonl
    chunks_path = output_dir / "chunks.jsonl"
    with jsonlines.open(chunks_path, mode="w") as writer:
        for chunk in chunks:
            writer.write(chunk.model_dump(by_alias=True, mode="json"))

    # Build and write quality report
    report = build_quality_report(chunks, docs)
    stats = report.statistics

    report_dict = {
        "strategy": strategy,
        "max_tokens": max_tokens,
        "total_chunks": stats.total_chunks,
        "total_docs": stats.total_docs,
        "docs_with_chunks": stats.docs_with_chunks,
        "avg_tokens": round(stats.avg_tokens, 1),
        "min_tokens": stats.min_tokens,
        "max_tokens_observed": stats.max_tokens,
        "percentiles": {
            "p25": stats.p25_tokens,
            "p50": stats.p50_tokens,
            "p75": stats.p75_tokens,
            "p90": stats.p90_tokens,
            "p95": stats.p95_tokens,
        },
        "per_category": stats.per_category,
        "chunks_per_doc": {
            "min": min(stats.chunks_per_doc.values()) if stats.chunks_per_doc else 0,
            "max": max(stats.chunks_per_doc.values()) if stats.chunks_per_doc else 0,
            "avg": round(
                sum(stats.chunks_per_doc.values()) / len(stats.chunks_per_doc), 1
            ) if stats.chunks_per_doc else 0,
        },
        "table_chunks": stats.table_chunks,
        "section_path_coverage": round(report.section_path_coverage, 4),
        "metadata_valid": report.metadata_valid_count,
        "metadata_invalid": report.metadata_invalid_count,
        "validation_errors": report.validation_errors[:5],
        "duration_seconds": round(elapsed, 3),
    }

    report_path = output_dir / "chunk_quality_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2))

    # Print summary
    print(f"\nCompleted in {elapsed:.3f}s")
    print(f"Total chunks: {stats.total_chunks}")
    print(f"Docs with chunks: {stats.docs_with_chunks}/{stats.total_docs}")
    print(f"Avg tokens/chunk: {stats.avg_tokens:.1f}  "
          f"(p50={stats.p50_tokens:.0f}, p95={stats.p95_tokens:.0f})")
    print(f"Chunks per doc: min={min(stats.chunks_per_doc.values())}, "
          f"max={max(stats.chunks_per_doc.values())}, "
          f"avg={sum(stats.chunks_per_doc.values())/len(stats.chunks_per_doc):.1f}")
    print(f"Table-containing chunks: {stats.table_chunks}")
    print(f"Section path coverage: {report.section_path_coverage:.1%}")
    print(f"Metadata valid: {report.metadata_valid_count}/{stats.total_chunks}")
    if report.validation_errors:
        print("Metadata errors (first 5):")
        for e in report.validation_errors[:5]:
            print(f"  [!] {e}")
    print()
    print("Per-category chunk distribution:")
    for cat, cnt in sorted(report_dict["per_category"].items()):
        n_docs = sum(1 for d in docs if d.metadata.category.value == cat)
        print(f"  {cat}: {cnt} chunks ({cnt/n_docs:.1f}/doc)")
    print()
    print(f"Output: {chunks_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build SEKD chunks from documents")
    parser.add_argument("--input-dir", default="data/sekd/raw", type=Path)
    parser.add_argument("--output-dir", default="data/sekd/processed", type=Path)
    parser.add_argument(
        "--strategy",
        default="section_aware",
        choices=["section_aware", "fixed_size", "recursive"],
    )
    parser.add_argument("--max-tokens", default=400, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
