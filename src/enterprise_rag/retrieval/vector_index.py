"""FAISS vector index for dense retrieval over SEKD chunks."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class VectorIndexMeta:
    """Metadata stored alongside the serialized FAISS index."""

    model_name: str = ""
    embedding_dim: int = 0
    corpus_size: int = 0
    distance: str = "cosine"  # cosine (via inner product after L2 normalization)
    chunk_ids: list[str] = field(default_factory=list)
    document_ids: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    section_paths: list[list[str]] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    build_timestamp: str = ""
    build_duration_seconds: float = 0.0


class VectorIndex:
    """FAISS inner-product index over L2-normalized chunk embeddings.

    After L2 normalization, inner product is equivalent to cosine similarity.
    Uses IndexFlatIP for exact (brute-force) search, which is appropriate
    for a corpus of ~1200 chunks.
    """

    def __init__(self) -> None:
        self._index = None  # faiss.IndexFlatIP
        self._embeddings: np.ndarray | None = None
        self._meta = VectorIndexMeta()

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        model_name: str,
        config: dict | None = None,
    ) -> None:
        """Build the FAISS index from precomputed embeddings.

        Args:
            chunks: List of chunk dicts (for metadata extraction).
            embeddings: Float32 array of shape (n_chunks, dim), L2-normalized.
            model_name: Name of the embedding model used.
            config: Optional config dict to record in metadata.
        """
        import faiss
        from datetime import datetime, timezone

        t0 = time.monotonic()
        assert embeddings.shape[0] == len(chunks), "Embedding count must match chunk count"
        dim = embeddings.shape[1]

        self._embeddings = embeddings.astype(np.float32)
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(self._embeddings)

        elapsed = time.monotonic() - t0
        self._meta = VectorIndexMeta(
            model_name=model_name,
            embedding_dim=dim,
            corpus_size=len(chunks),
            chunk_ids=[c["chunk_id"] for c in chunks],
            document_ids=[c["document_id"] for c in chunks],
            categories=[c.get("category", "") for c in chunks],
            section_paths=[c.get("section_path", []) for c in chunks],
            texts=[c.get("text", "") for c in chunks],
            config=config or {},
            build_timestamp=datetime.now(timezone.utc).isoformat(),
            build_duration_seconds=round(elapsed, 3),
        )

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """Save FAISS index, embeddings, and metadata to disk."""
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_dir / "faiss.index"))
        np.save(str(index_dir / "embeddings.npy"), self._embeddings)
        with open(index_dir / "meta.json", "w") as f:
            json.dump(asdict(self._meta), f, indent=2)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorIndex":
        """Load a previously saved index from disk."""
        import faiss

        vi = cls()
        vi._index = faiss.read_index(str(index_dir / "faiss.index"))
        vi._embeddings = np.load(str(index_dir / "embeddings.npy"))
        with open(index_dir / "meta.json") as f:
            raw = json.load(f)
        vi._meta = VectorIndexMeta(**{k: v for k, v in raw.items() if k in VectorIndexMeta.__dataclass_fields__})
        return vi

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query_embedding: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices) for top-k nearest chunks.

        Args:
            query_embedding: Float32 array of shape (dim,) or (1, dim), L2-normalized.
            k: Number of results.

        Returns:
            scores: Float32 array of shape (k,) — cosine similarities.
            indices: Int64 array of shape (k,) — positions in the corpus.
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        q = query_embedding.astype(np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        actual_k = min(k, self._meta.corpus_size)
        scores, indices = self._index.search(q, actual_k)
        return scores[0], indices[0]
