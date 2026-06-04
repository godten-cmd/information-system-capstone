"""MCP tool: dense_search — wraps V1 DenseIndex as a callable MCP tool.

The embedding model (BGE-small-en-v1.5) is loaded lazily on first call.
Loading takes ~2s on CPU; subsequent calls are fast.

Direct-call API:
    from enterprise_rag.mcp.tools.dense_tool import dense_search
    results = dense_search("exception clause travel reimbursement", top_k=10)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from enterprise_rag.retrieval.dense import DenseIndex
from enterprise_rag.retrieval.result_schema import RetrievedChunk

# ── Singleton index ────────────────────────────────────────────────────────────

_index: DenseIndex | None = None


def _default_index_dir() -> Path:
    env = os.environ.get("V2_DENSE_INDEX_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    project_root = here.parents[4]
    return project_root / "outputs" / "evaluation" / "dense" / "index"


def _get_index() -> DenseIndex:
    global _index
    if _index is None:
        _index = DenseIndex.load(_default_index_dir())
    return _index


def reset_index() -> None:
    global _index
    _index = None


# ── Core function ──────────────────────────────────────────────────────────────


def dense_search(
    query: str,
    top_k: int = 10,
    query_id: str = "anon",
) -> list[dict]:
    """Retrieve top-k chunks using dense bi-encoder retrieval (BGE-small-en-v1.5).

    Args:
        query: Raw query text.
        top_k: Number of chunks to return (max 50).
        query_id: Identifier for tracing.

    Returns:
        List of chunk dicts with keys: chunk_id, document_id, rank, score,
        text, category, section_path, strategy_used, latency_ms.
    """
    top_k = min(top_k, 50)
    t0 = time.monotonic()
    idx = _get_index()
    chunks: list[RetrievedChunk] = idx.retrieve(query, query_id, k=top_k)
    latency_ms = (time.monotonic() - t0) * 1000

    return [
        {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "rank": c.rank,
            "score": round(c.score, 6),
            "text": c.text,
            "category": c.category,
            "section_path": c.section_path,
            "strategy_used": "dense",
            "latency_ms": round(latency_ms, 2),
        }
        for c in chunks
    ]
