"""MCP tool: bm25_search — wraps V1 BM25Index as a callable MCP tool.

Design: The V1 BM25Index is loaded once (lazy singleton on first call) and
reused for all subsequent calls. Index path is resolved from agent_config.yaml
or the V2_BM25_INDEX_DIR environment variable.

Direct-call API (no MCP overhead):
    from enterprise_rag.mcp.tools.bm25_tool import bm25_search
    results = bm25_search("hotel expense limit", top_k=10)

MCP API (via FastMCP server):
    Registered automatically by mcp/server.py.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from enterprise_rag.retrieval.bm25 import BM25Index
from enterprise_rag.retrieval.result_schema import RetrievedChunk

# ── Singleton index ────────────────────────────────────────────────────────────

_index: BM25Index | None = None
_index_dir: Path | None = None


def _default_index_dir() -> Path:
    env = os.environ.get("V2_BM25_INDEX_DIR")
    if env:
        return Path(env)
    # Resolve relative to project root: 3 levels up from this file
    here = Path(__file__).resolve()
    project_root = here.parents[4]
    return project_root / "outputs" / "evaluation" / "bm25" / "index"


def _get_index() -> BM25Index:
    global _index, _index_dir
    if _index is None:
        idx_dir = _default_index_dir()
        _index = BM25Index.load(idx_dir)
        _index_dir = idx_dir
    return _index


def reset_index() -> None:
    """Force reload on next call — used in tests to swap index directories."""
    global _index, _index_dir
    _index = None
    _index_dir = None


# ── Core function (callable directly without MCP) ─────────────────────────────


def bm25_search(
    query: str,
    top_k: int = 10,
    query_id: str = "anon",
) -> list[dict]:
    """Retrieve top-k chunks using BM25 sparse retrieval.

    Args:
        query: Raw query text.
        top_k: Number of chunks to return (max 50).
        query_id: Identifier for the query (used in result metadata).

    Returns:
        List of chunk dicts with keys: chunk_id, document_id, rank, score,
        text, category, section_path, strategy_used.
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
            "strategy_used": "bm25",
            "latency_ms": round(latency_ms, 2),
        }
        for c in chunks
    ]
