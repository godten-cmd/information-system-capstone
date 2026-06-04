"""MCP tool: hybrid_search — wraps V1 HybridIndex (BM25 + Dense RRF).

Reuses both singleton indexes from bm25_tool and dense_tool; does not
duplicate index loading. rrf_k defaults to 10 (V1 best-performing value).

Direct-call API:
    from enterprise_rag.mcp.tools.hybrid_tool import hybrid_search
    results = hybrid_search("multi-hop authentication query", top_k=10)
"""

from __future__ import annotations

import time

from enterprise_rag.mcp.tools.bm25_tool import _get_index as _get_bm25
from enterprise_rag.mcp.tools.dense_tool import _get_index as _get_dense
from enterprise_rag.retrieval.hybrid import HybridConfig, HybridIndex
from enterprise_rag.retrieval.result_schema import RetrievedChunk

# ── Singleton HybridIndex (shares BM25 + Dense singletons) ────────────────────

_hybrid: HybridIndex | None = None
_current_rrf_k: int | None = None


def _get_hybrid(rrf_k: int = 10) -> HybridIndex:
    global _hybrid, _current_rrf_k
    if _hybrid is None or _current_rrf_k != rrf_k:
        config = HybridConfig(rrf_k=rrf_k, candidate_k=100)
        _hybrid = HybridIndex(_get_bm25(), _get_dense(), config=config)
        _current_rrf_k = rrf_k
    return _hybrid


def reset_index() -> None:
    global _hybrid, _current_rrf_k
    _hybrid = None
    _current_rrf_k = None


# ── Core function ──────────────────────────────────────────────────────────────


def hybrid_search(
    query: str,
    top_k: int = 10,
    rrf_k: int = 10,
    query_id: str = "anon",
) -> list[dict]:
    """Retrieve top-k chunks by fusing BM25 and Dense results via RRF.

    Args:
        query: Raw query text.
        top_k: Number of chunks to return (max 50).
        rrf_k: RRF constant k in score = 1/(k + rank). V1 best: 10.
        query_id: Identifier for tracing.

    Returns:
        List of chunk dicts with keys: chunk_id, document_id, rank, score,
        text, category, section_path, strategy_used, latency_ms.
    """
    top_k = min(top_k, 50)
    t0 = time.monotonic()
    idx = _get_hybrid(rrf_k=rrf_k)
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
            "strategy_used": "hybrid",
            "latency_ms": round(latency_ms, 2),
        }
        for c in chunks
    ]
