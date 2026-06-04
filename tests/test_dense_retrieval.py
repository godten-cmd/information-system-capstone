"""Tests for Phase 6: Dense retrieval baseline.

Tests use pre-computed random embeddings to avoid downloading actual models.
Only construction and config tests touch EmbeddingConfig/EmbeddingModel.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from enterprise_rag.retrieval.dense import DenseConfig, DenseIndex
from enterprise_rag.retrieval.embeddings import EmbeddingConfig, EmbeddingModel
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk
from enterprise_rag.retrieval.vector_index import VectorIndex, VectorIndexMeta


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_chunks(n: int = 8) -> list[dict]:
    texts = [
        "Annual leave policy grants 15 days per year to full-time employees.",
        "Security control HY-AC-01 requires multi-factor authentication for all users.",
        "Travel lodging reimbursement limit is USD 200 per night for US region.",
        "API endpoint POST /api/v2/alerts accepts JSON body with alert data.",
        "Project PRJ-HYD-ALPHA sprint planning notes with action items for deployment.",
        "Exception to the overtime policy requires VP approval and HR documentation.",
        "System HYDeploy uses HYID for authentication via OAuth2 protocol.",
        "Meeting notes PRJ-BILLING-REVAMP discuss invoice reconciliation timelines.",
    ]
    return [
        {
            "chunk_id": f"DOC-{i+1:03d}-C001",
            "document_id": f"DOC-{i+1:03d}",
            "category": [
                "HR Policies", "Security Policies", "Travel Policies",
                "API Documentation", "Project Meeting Notes",
                "HR Policies", "System Design Documents", "Project Meeting Notes",
            ][i],
            "section_path": [f"Section {i+1}"],
            "text": texts[i],
            "char_start": 0,
            "char_end": len(texts[i]),
            "token_estimate": 20,
        }
        for i in range(n)
    ]


def _random_embeddings(n: int, dim: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    emb = rng.random((n, dim)).astype(np.float32)
    # L2-normalize for cosine similarity via inner product
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


def _make_vector_index(chunks: list[dict], dim: int = 64) -> VectorIndex:
    """Build a VectorIndex with random embeddings (no model required)."""
    embeddings = _random_embeddings(len(chunks), dim=dim)
    vi = VectorIndex()
    vi.build(chunks, embeddings, model_name="test-model-dim64")
    return vi


class _FakeEmbeddingModel:
    """Deterministic embedding model for testing (no network access)."""

    def __init__(self, dim: int = 64, seed: int = 42) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(seed)

    def load(self) -> "_FakeEmbeddingModel":
        return self

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake-test-model"

    @property
    def info(self):
        info = MagicMock()
        info.embedding_dim = self._dim
        info.model_name = "fake-test-model"
        return info

    def _encode(self, texts: list[str]) -> np.ndarray:
        n = len(texts)
        # Deterministic: hash of text index → embedding
        embs = np.zeros((n, self._dim), dtype=np.float32)
        for i, t in enumerate(texts):
            rng = np.random.default_rng(abs(hash(t)) % (2**32))
            embs[i] = rng.random(self._dim).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-9)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)


# ── EmbeddingConfig tests ─────────────────────────────────────────────────────


def test_embedding_config_defaults() -> None:
    cfg = EmbeddingConfig()
    assert cfg.model_name == "BAAI/bge-small-en-v1.5"
    assert cfg.normalize is True
    assert cfg.batch_size > 0


def test_embedding_config_custom() -> None:
    cfg = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        device="cpu",
    )
    assert cfg.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert cfg.batch_size == 32


def test_embedding_model_raises_before_load() -> None:
    model = EmbeddingModel()
    with pytest.raises(RuntimeError, match="Model not loaded"):
        _ = model.info


# ── VectorIndex tests ─────────────────────────────────────────────────────────


def test_vector_index_builds() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    assert vi._meta.corpus_size == len(chunks)
    assert vi._meta.embedding_dim == 64
    assert len(vi._meta.chunk_ids) == len(chunks)


def test_vector_index_stores_chunk_ids_in_order() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    expected = [c["chunk_id"] for c in chunks]
    assert vi._meta.chunk_ids == expected


def test_vector_index_search_returns_k_results() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    q_emb = _random_embeddings(1, dim=64, seed=99)[0]
    scores, indices = vi.search(q_emb, k=5)
    assert len(scores) == 5
    assert len(indices) == 5


def test_vector_index_search_scores_in_range() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    q_emb = _random_embeddings(1, dim=64)[0]
    scores, _ = vi.search(q_emb, k=3)
    # Cosine similarities should be in [-1, 1]
    assert all(-1.0 <= float(s) <= 1.0 for s in scores)


def test_vector_index_search_same_vector_scores_near_1() -> None:
    chunks = _make_chunks()
    embeddings = _random_embeddings(len(chunks), dim=64)
    vi = VectorIndex()
    vi.build(chunks, embeddings, model_name="test")
    # Query with the first chunk's exact embedding
    scores, indices = vi.search(embeddings[0], k=1)
    assert float(scores[0]) == pytest.approx(1.0, abs=1e-5)
    assert int(indices[0]) == 0


def test_vector_index_raises_before_build() -> None:
    vi = VectorIndex()
    q = np.ones(64, dtype=np.float32)
    q /= np.linalg.norm(q)
    with pytest.raises(RuntimeError, match="Index not built"):
        vi.search(q, k=3)


def test_vector_index_save_and_load() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = Path(tmp)
        vi.save(idx_dir)
        assert (idx_dir / "faiss.index").exists()
        assert (idx_dir / "embeddings.npy").exists()
        assert (idx_dir / "meta.json").exists()

        loaded = VectorIndex.load(idx_dir)
        assert loaded._meta.corpus_size == vi._meta.corpus_size
        assert loaded._meta.chunk_ids == vi._meta.chunk_ids
        assert loaded._meta.embedding_dim == vi._meta.embedding_dim


def test_vector_index_load_roundtrip_consistent_results() -> None:
    chunks = _make_chunks()
    vi = _make_vector_index(chunks)
    q_emb = _random_embeddings(1, dim=64, seed=7)[0]
    scores_orig, idx_orig = vi.search(q_emb, k=3)

    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = Path(tmp)
        vi.save(idx_dir)
        loaded = VectorIndex.load(idx_dir)
        scores_loaded, idx_loaded = loaded.search(q_emb, k=3)

    np.testing.assert_array_almost_equal(scores_orig, scores_loaded, decimal=5)
    np.testing.assert_array_equal(idx_orig, idx_loaded)


# ── DenseConfig tests ─────────────────────────────────────────────────────────


def test_dense_config_defaults() -> None:
    cfg = DenseConfig()
    assert cfg.model_name == "BAAI/bge-small-en-v1.5"
    assert cfg.normalize is True
    assert cfg.max_k == 10


def test_dense_config_fallback_model() -> None:
    cfg = DenseConfig(model_name="sentence-transformers/all-MiniLM-L6-v2")
    assert "MiniLM" in cfg.model_name


# ── DenseIndex tests (using _FakeEmbeddingModel) ──────────────────────────────


def _build_dense_index_with_fake_model(
    chunks: list[dict],
    dim: int = 64,
) -> DenseIndex:
    """Build DenseIndex by injecting a fake embedding model."""
    fake_model = _FakeEmbeddingModel(dim=dim)
    texts = [
        " ".join(c.get("section_path", [])) + " " + c.get("text", "")
        for c in chunks
    ]
    embeddings = fake_model.encode_documents(texts)

    vi = VectorIndex()
    vi.build(chunks, embeddings, model_name="fake-test-model")

    di = DenseIndex()
    di._vector_index = vi
    di._model = fake_model
    di._built = True
    return di


def test_dense_index_retrieve_returns_retrieved_chunks() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    results = di.retrieve("annual leave policy", "Q-001", k=3)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, RetrievedChunk)
        assert r.query_id == "Q-001"
        assert r.rank >= 1


def test_dense_index_retrieve_results_sorted_by_rank() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    results = di.retrieve("security control", "Q-001", k=5)
    ranks = [r.rank for r in results]
    assert ranks == sorted(ranks)
    assert ranks[0] == 1


def test_dense_index_retrieve_at_most_k_results() -> None:
    chunks = _make_chunks(4)
    di = _build_dense_index_with_fake_model(chunks, dim=32)
    results = di.retrieve("test query", "Q-001", k=3)
    assert len(results) <= 3


def test_dense_index_retrieve_results_have_scores() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    results = di.retrieve("API endpoint", "Q-002", k=3)
    for r in results:
        assert isinstance(r.score, float)
        assert -1.0 <= r.score <= 1.0


def test_dense_index_raises_before_build() -> None:
    di = DenseIndex()
    with pytest.raises(RuntimeError, match="Index not built"):
        di.retrieve("test", "Q-001", k=3)


def test_dense_index_run_covers_all_queries() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    queries = [
        {"query_id": "Q-001", "query": "annual leave"},
        {"query_id": "Q-002", "query": "security HY-AC-01"},
        {"query_id": "Q-003", "query": "API alerts endpoint"},
    ]
    run = di.run(queries, k=5)
    query_ids = {r.query_id for r in run.results}
    assert query_ids == {"Q-001", "Q-002", "Q-003"}


def test_dense_index_run_returns_retrieval_run() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    run = di.run([{"query_id": "Q-1", "query": "test"}], k=3)
    assert isinstance(run, RetrievalRun)
    assert run.method == "dense"
    assert run.k == 3


def test_dense_index_run_records_model_name() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    run = di.run([{"query_id": "Q-1", "query": "test"}], k=3)
    assert "model_name" in run.config


def test_dense_index_save_and_load_roundtrip() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    with tempfile.TemporaryDirectory() as tmp:
        idx_dir = Path(tmp)
        di.save(idx_dir)
        assert (idx_dir / "faiss.index").exists()
        assert (idx_dir / "dense_config.json").exists()

        loaded = DenseIndex.load(idx_dir)
        assert loaded._built is True
        assert loaded._vector_index._meta.corpus_size == len(chunks)


def test_dense_index_model_info() -> None:
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    info = di.model_info
    assert "model_name" in info
    assert "embedding_dim" in info
    assert "corpus_size" in info
    assert info["corpus_size"] == len(chunks)


# ── Integration: build index + retrieve + evaluate ────────────────────────────


def test_end_to_end_dense_pipeline() -> None:
    from enterprise_rag.evaluation.retrieval_metrics import aggregate, evaluate_run

    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)

    queries = [
        {"query_id": "Q-001", "query": "annual leave days HR policy"},
        {"query_id": "Q-002", "query": "security control HY-AC-01 authentication"},
    ]
    run = di.run(queries, k=5)

    ground_truths = {
        "Q-001": ["DOC-001-C001"],
        "Q-002": ["DOC-002-C001"],
    }
    queries_meta = {
        "Q-001": {"query_id": "Q-001", "category": "HR Policies", "reasoning_type": "single_hop",
                  "difficulty": "easy", "retrieval_difficulty_factors": [], "answerable": True},
        "Q-002": {"query_id": "Q-002", "category": "Security Policies", "reasoning_type": "single_hop",
                  "difficulty": "medium", "retrieval_difficulty_factors": [], "answerable": True},
    }

    retrieval_results = [r.to_dict() for r in run.results]
    result_by_k = evaluate_run(retrieval_results, ground_truths, queries_meta, k_values=[1, 5])
    agg5 = aggregate(result_by_k[5], k=5)
    assert agg5.n_queries == 2
    # Results should be deterministic with fake model
    assert isinstance(agg5.recall, float)
    assert 0.0 <= agg5.recall <= 1.0


def test_dense_retrieval_consistency_across_calls() -> None:
    """Same query should return same top result on repeated calls."""
    chunks = _make_chunks()
    di = _build_dense_index_with_fake_model(chunks)
    r1 = di.retrieve("annual leave entitlement", "Q-001", k=3)
    r2 = di.retrieve("annual leave entitlement", "Q-001", k=3)
    assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


# ── Reporter comparison output ────────────────────────────────────────────────


def test_write_comparison_json_missing_bm25_is_silent() -> None:
    from enterprise_rag.evaluation.reporter import write_comparison_json
    from enterprise_rag.retrieval.result_schema import RetrievalRun

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        run = RetrievalRun(method="dense", k=10, config={"model_name": "test"})
        result_by_k = {}
        # Should not raise even if BM25 file doesn't exist
        path = write_comparison_json(result_by_k, run, Path(tmp) / "nonexistent.json", out_dir)
        assert isinstance(path, Path)
