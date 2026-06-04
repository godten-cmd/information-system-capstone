"""BM25 retrieval baseline for the SEKD corpus.

Uses rank-bm25 (BM25Okapi) with enterprise-aware tokenization.
Supports index serialization and multi-k retrieval.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
from enterprise_rag.retrieval.tokenization import tokenize_document, tokenize_query


@dataclass
class BM25Config:
    """Configuration for the BM25 index and retrieval."""

    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25
    max_k: int = 10


@dataclass
class BM25IndexMeta:
    """Metadata stored alongside the serialized BM25 index."""

    method: str = "bm25"
    corpus_size: int = 0
    chunk_ids: list[str] = None  # type: ignore[assignment]
    document_ids: list[str] = None  # type: ignore[assignment]
    categories: list[str] = None  # type: ignore[assignment]
    section_paths: list[list[str]] = None  # type: ignore[assignment]
    texts: list[str] = None  # type: ignore[assignment]
    config: dict[str, Any] = None  # type: ignore[assignment]
    build_timestamp: str = ""

    def __post_init__(self) -> None:
        if self.chunk_ids is None:
            self.chunk_ids = []
        if self.document_ids is None:
            self.document_ids = []
        if self.categories is None:
            self.categories = []
        if self.section_paths is None:
            self.section_paths = []
        if self.texts is None:
            self.texts = []
        if self.config is None:
            self.config = {}


class BM25Index:
    """BM25 sparse index over SEKD chunks."""

    def __init__(self, config: BM25Config | None = None) -> None:
        self._config = config or BM25Config()
        self._bm25: BM25Okapi | None = None
        self._meta = BM25IndexMeta()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """Build the BM25 index from a list of chunk dicts."""
        from datetime import datetime, timezone

        corpus_texts: list[list[str]] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            section = " ".join(chunk.get("section_path", []))
            combined = f"{section} {text}"
            corpus_texts.append(tokenize_document(combined))

        self._bm25 = BM25Okapi(
            corpus_texts,
            k1=self._config.k1,
            b=self._config.b,
            epsilon=self._config.epsilon,
        )
        self._meta = BM25IndexMeta(
            corpus_size=len(chunks),
            chunk_ids=[c["chunk_id"] for c in chunks],
            document_ids=[c["document_id"] for c in chunks],
            categories=[c.get("category", "") for c in chunks],
            section_paths=[c.get("section_path", []) for c in chunks],
            texts=[c.get("text", "") for c in chunks],
            config=asdict(self._config),
            build_timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """Save BM25 index and metadata to disk."""
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self._bm25, f)
        with open(index_dir / "meta.json", "w") as f:
            json.dump(asdict(self._meta), f, indent=2)

    @classmethod
    def load(cls, index_dir: Path, config: BM25Config | None = None) -> "BM25Index":
        """Load a previously saved BM25 index from disk."""
        index = cls(config=config)
        with open(index_dir / "bm25.pkl", "rb") as f:
            index._bm25 = pickle.load(f)
        with open(index_dir / "meta.json") as f:
            raw = json.load(f)
        # Reconstruct meta (ignore extra fields for forward compat)
        index._meta = BM25IndexMeta(
            corpus_size=raw.get("corpus_size", 0),
            chunk_ids=raw.get("chunk_ids", []),
            document_ids=raw.get("document_ids", []),
            categories=raw.get("categories", []),
            section_paths=raw.get("section_paths", []),
            texts=raw.get("texts", []),
            config=raw.get("config", {}),
            build_timestamp=raw.get("build_timestamp", ""),
        )
        return index

    # ── Query ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        query_id: str,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a single query.

        Args:
            query_text: Raw query text.
            query_id: Identifier for the query.
            k: Number of results to return.

        Returns:
            List of RetrievedChunk objects sorted by rank (1 = best).
        """
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        query_tokens = tokenize_query(query_text)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices (descending score)
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:k]

        results: list[RetrievedChunk] = []
        for rank, idx in enumerate(top_indices, start=1):
            score = float(scores[idx])
            if score <= 0:
                # BM25 returns 0 for no match — skip zero-score results beyond rank 1
                # (keep at least 1 result even if score=0)
                if rank > 1:
                    continue
            results.append(
                RetrievedChunk(
                    query_id=query_id,
                    chunk_id=self._meta.chunk_ids[idx],
                    document_id=self._meta.document_ids[idx],
                    rank=rank,
                    score=score,
                    category=self._meta.categories[idx],
                    section_path=self._meta.section_paths[idx],
                    text=self._meta.texts[idx][:200],  # truncate for output
                )
            )

        return results

    def run(
        self,
        queries: list[dict],
        k: int = 10,
        method: str = "bm25",
    ) -> RetrievalRun:
        """Run BM25 retrieval for all queries.

        Args:
            queries: List of query dicts with 'query_id' and 'query' fields.
            k: Number of results per query.
            method: Method name to record in the run.

        Returns:
            RetrievalRun containing all results.
        """
        run = RetrievalRun(method=method, k=k, config=asdict(self._config))
        for q in queries:
            query_id = q["query_id"]
            query_text = q.get("query", q.get("query_text", ""))
            results = self.retrieve(query_text, query_id, k=k)
            run.results.extend(results)
        return run


def build_bm25_index(
    chunks: list[dict],
    index_dir: Path,
    config: BM25Config | None = None,
) -> BM25Index:
    """Build and save a BM25 index from a list of chunk dicts."""
    t0 = time.monotonic()
    index = BM25Index(config=config)
    index.build(chunks)
    elapsed = time.monotonic() - t0
    index.save(index_dir)
    print(f"BM25 index built in {elapsed:.2f}s ({len(chunks)} chunks → {index_dir})")
    return index
