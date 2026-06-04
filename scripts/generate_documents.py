#!/usr/bin/env python
"""Standalone script to generate the SEKD document corpus.

Usage:
    python scripts/generate_documents.py --seed 42 --document-count 120 --output-dir data/sekd/raw
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SEKD synthetic documents.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--document-count", type=int, default=120, help="Total document count")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sekd/raw"), help="Output directory")
    args = parser.parse_args()

    from enterprise_rag.dataset.generate_documents import DatasetConfig, SEKDGenerator

    config = DatasetConfig(seed=args.seed, document_count=args.document_count)
    generator = SEKDGenerator(config=config, seed=args.seed)

    print(f"Generating {args.document_count} documents (seed={args.seed}) ...")
    documents = generator.generate()

    print(f"Writing output to {args.output_dir} ...")
    report = generator.write(documents, args.output_dir)

    print(f"\nCompleted in {report.duration_seconds:.2f}s")
    print(f"Total documents: {report.total_documents}")
    for cat, cnt in sorted(report.per_category.items()):
        print(f"  {cat}: {cnt}")

    if report.validation_errors:
        print("\nValidation errors:")
        for err in report.validation_errors:
            print(f"  [!] {err}")


if __name__ == "__main__":
    main()
