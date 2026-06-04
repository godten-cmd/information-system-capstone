"""Dense retrieval baseline for the SEKD corpus.

Wraps EmbeddingModel + VectorIndex with the same interface as BM25Index:
build → save/load → retrieve/run → RetrievalRun.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from enterprise_rag.retrieval.embeddings import EmbeddingConfig, EmbeddingModel
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
from enterprise_rag.retrieval.vector_index import VectorIndex


@dataclass
class DenseConfig:
    """Configuration for dense retrieval."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"
    use_query_instruction: bool = True
    query_instruction: str = "Represent this sentence: "
    max_k: int = 10


class DenseIndex:
    """Dense bi-encoder index over SEKD chunks."""

    def __init__(self, config: DenseConfig | None = None) -> None:
        self._config = config or DenseConfig()
        self._vector_index = VectorIndex()
        self._model: EmbeddingModel | None = None
        self._built = False

    def _make_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            model_name=self._config.model_name,
            batch_size=self._config.batch_size,
            normalize=self._config.normalize,
            device=self._config.device,
            use_query_instruction=self._config.use_query_instruction,
            query_instruction=self._config.query_instruction,
            show_progress=True,
        )

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, chunks: list[dict]) -> None:
        """Build the dense index by encoding all chunks."""
        t0 = time.monotonic()
        emb_config = self._make_embedding_config()
        self._model = EmbeddingModel(emb_config)
        self._model.load()

        texts = [
            " ".join(c.get("section_path", [])) + " " + c.get("text", "")
            for c in chunks
        ]
        embeddings = self._model.encode_documents(texts)
        total_elapsed = time.monotonic() - t0

        self._vector_index.build(
            chunks=chunks,
            embeddings=embeddings,
            model_name=self._config.model_name,
            config=asdict(self._config),
        )
        # Override with full build time (model load + encoding + FAISS add)
        self._vector_index._meta.build_duration_seconds = round(total_elapsed, 3)
        self._built = True

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """Save vector index and config to disk."""
        self._vector_index.save(index_dir)
        with open(index_dir / "dense_config.json", "w") as f:
            json.dump(asdict(self._config), f, indent=2)

    @classmethod
    def load(cls, index_dir: Path, config: DenseConfig | None = None) -> "DenseIndex":
        """Load a previously saved dense index."""
        di = cls(config=config)
        di._vector_index = VectorIndex.load(index_dir)

        # Restore config from disk if not provided
        config_path = index_dir / "dense_config.json"
        if config is None and config_path.exists():
            with open(config_path) as f:
                raw = json.load(f)
            di._config = DenseConfig(**{k: v for k, v in raw.items() if k in DenseConfig.__dataclass_fields__})

        di._built = True
        return di

    def _ensure_model(self) -> EmbeddingModel:
        """Load the embedding model on demand (lazy for load() path)."""
        if self._model is None:
            emb_config = self._make_embedding_config()
            emb_config.show_progress = False
            self._model = EmbeddingModel(emb_config)
            self._model.load()
        return self._model

    # ── Query ─────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        query_id: str,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks for a single query.

        Returns:
            List of RetrievedChunk objects sorted by rank (1 = best).
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() or load() first.")

        model = self._ensure_model()
        q_emb = model.encode_queries([query_text])[0]
        scores, indices = self._vector_index.search(q_emb, k=k)

        meta = self._vector_index._meta
        results: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:  # FAISS returns -1 for padding when k > corpus size
                continue
            results.append(
                RetrievedChunk(
                    query_id=query_id,
                    chunk_id=meta.chunk_ids[idx],
                    document_id=meta.document_ids[idx],
                    rank=rank,
                    score=float(score),
                    category=meta.categories[idx],
                    section_path=meta.section_paths[idx],
                    text=meta.texts[idx][:200],
                )
            )
        return results

    def run(
        self,
        queries: list[dict],
        k: int = 10,
        method: str = "dense",
    ) -> RetrievalRun:
        """Run dense retrieval for all queries.

        Args:
            queries: List of dicts with 'query_id' and 'query'/'query_text' fields.
            k: Number of results per query.
            method: Method name to record in the run.

        Returns:
            RetrievalRun containing all results.
        """
        if not self._built:
            raise RuntimeError("Index not built. Call build() or load() first.")

        model = self._ensure_model()

        # Batch encode all queries at once for efficiency
        query_texts = [q.get("query", q.get("query_text", "")) for q in queries]
        query_ids = [q["query_id"] for q in queries]

        t0 = time.monotonic()
        q_embeddings = model.encode_queries(query_texts)
        encoding_elapsed = time.monotonic() - t0

        run = RetrievalRun(
            method=method,
            k=k,
            config={
                **asdict(self._config),
                "encoding_seconds": round(encoding_elapsed, 3),
                "n_queries": len(queries),
            },
        )

        for qid, q_emb in zip(query_ids, q_embeddings):
            results = self._retrieve_from_embedding(q_emb, qid, k=k)
            run.results.extend(results)

        return run

    def _retrieve_from_embedding(
        self,
        q_emb: np.ndarray,
        query_id: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Internal: retrieve from a precomputed query embedding."""
        scores, indices = self._vector_index.search(q_emb, k=k)
        meta = self._vector_index._meta
        results: list[RetrievedChunk] = []
        for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    query_id=query_id,
                    chunk_id=meta.chunk_ids[idx],
                    document_id=meta.document_ids[idx],
                    rank=rank,
                    score=float(score),
                    category=meta.categories[idx],
                    section_path=meta.section_paths[idx],
                    text=meta.texts[idx][:200],
                )
            )
        return results

    @property
    def model_info(self) -> dict:
        """Return embedding model metadata (after load or build)."""
        meta = self._vector_index._meta
        return {
            "model_name": meta.model_name,
            "embedding_dim": meta.embedding_dim,
            "corpus_size": meta.corpus_size,
            "build_duration_seconds": meta.build_duration_seconds,
        }


def build_dense_index(
    chunks: list[dict],
    index_dir: Path,
    config: DenseConfig | None = None,
) -> DenseIndex:
    """Build and save a dense index from a list of chunk dicts."""
    t0 = time.monotonic()
    index = DenseIndex(config=config)
    index.build(chunks)
    elapsed = time.monotonic() - t0
    index.save(index_dir)
    meta = index._vector_index._meta
    print(
        f"Dense index built in {elapsed:.1f}s "
        f"({meta.corpus_size} chunks, dim={meta.embedding_dim} → {index_dir})"
    )
    return index
