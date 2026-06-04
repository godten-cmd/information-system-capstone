"""Embedding model wrapper for dense retrieval.

Wraps sentence-transformers with batched encoding, normalization,
and reproducible query/document encoding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding model."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    device: str = "cpu"
    show_progress: bool = False
    # BGE models perform better with this query instruction for retrieval
    query_instruction: str = "Represent this sentence: "
    use_query_instruction: bool = True


@dataclass
class EmbeddingModelInfo:
    """Runtime metadata about a loaded embedding model."""

    model_name: str
    embedding_dim: int
    normalize: bool
    device: str
    load_duration_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class EmbeddingModel:
    """Thin wrapper around a SentenceTransformer for chunk and query encoding.

    Uses a query instruction prefix for BGE models to improve retrieval quality.
    Documents are encoded without a prefix (asymmetric retrieval).
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._config = config or EmbeddingConfig()
        self._model = None
        self._info: EmbeddingModelInfo | None = None

    def load(self) -> "EmbeddingModel":
        """Load the model. Idempotent."""
        if self._model is not None:
            return self
        from sentence_transformers import SentenceTransformer

        t0 = time.monotonic()
        self._model = SentenceTransformer(
            self._config.model_name,
            device=self._config.device,
        )
        elapsed = time.monotonic() - t0

        dim = self._model.get_embedding_dimension()
        self._info = EmbeddingModelInfo(
            model_name=self._config.model_name,
            embedding_dim=dim,
            normalize=self._config.normalize,
            device=self._config.device,
            load_duration_seconds=round(elapsed, 2),
        )
        return self

    @property
    def info(self) -> EmbeddingModelInfo:
        if self._info is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._info

    @property
    def embedding_dim(self) -> int:
        return self.info.embedding_dim

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode document/chunk texts. No instruction prefix."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        embeddings = self._model.encode(
            texts,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=self._config.show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Encode query texts, optionally prepending the instruction prefix."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        if self._config.use_query_instruction and self._config.query_instruction:
            prefixed = [self._config.query_instruction + t for t in texts]
        else:
            prefixed = texts
        embeddings = self._model.encode(
            prefixed,
            batch_size=self._config.batch_size,
            normalize_embeddings=self._config.normalize,
            show_progress_bar=self._config.show_progress,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)
