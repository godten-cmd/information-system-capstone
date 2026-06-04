"""MCP server for enterprise retrieval tools.

Exposes three retrieval backends (BM25, Dense, Hybrid) as MCP-compliant tools
that can be invoked by any MCP client (LangGraph agent, Claude Desktop, etc.).

Run as standalone server:
    python -m enterprise_rag.mcp.server

Or import the `mcp` object and call tools directly in-process:
    from enterprise_rag.mcp.server import mcp
    result = await mcp.call_tool("bm25_search", {"query": "...", "top_k": 10})
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from enterprise_rag.mcp.tools.bm25_tool import bm25_search as _bm25_search
from enterprise_rag.mcp.tools.dense_tool import dense_search as _dense_search
from enterprise_rag.mcp.tools.hybrid_tool import hybrid_search as _hybrid_search

mcp = FastMCP(
    "enterprise-rag-retrieval",
    instructions=(
        "Retrieval tools for the SEKD enterprise knowledge corpus. "
        "Use bm25_search for keyword/temporal queries, dense_search for "
        "semantic/exception queries, and hybrid_search for multi-hop/aggregation queries."
    ),
)


@mcp.tool()
def bm25_search(query: str, top_k: int = 10, query_id: str = "anon") -> list[dict]:
    """BM25 sparse retrieval over the SEKD corpus.

    Best for: exact keyword matching, temporal queries (date/version numbers),
    policy codes, specific identifiers.

    Args:
        query: The search query text.
        top_k: Number of results to return (1–50).
        query_id: Optional trace identifier.
    """
    return _bm25_search(query=query, top_k=top_k, query_id=query_id)


@mcp.tool()
def dense_search(query: str, top_k: int = 10, query_id: str = "anon") -> list[dict]:
    """Dense bi-encoder retrieval using BGE-small-en-v1.5 embeddings.

    Best for: semantic similarity, exception clauses, conceptual comparisons,
    paraphrased queries.

    Args:
        query: The search query text.
        top_k: Number of results to return (1–50).
        query_id: Optional trace identifier.
    """
    return _dense_search(query=query, top_k=top_k, query_id=query_id)


@mcp.tool()
def hybrid_search(
    query: str,
    top_k: int = 10,
    rrf_k: int = 10,
    query_id: str = "anon",
) -> list[dict]:
    """Hybrid BM25+Dense retrieval fused via Reciprocal Rank Fusion.

    Best for: multi-hop queries, aggregation queries, and when both lexical
    and semantic signals are relevant.

    Args:
        query: The search query text.
        top_k: Number of results to return (1–50).
        rrf_k: RRF constant (default 10, as validated in V1 Phase 7).
        query_id: Optional trace identifier.
    """
    return _hybrid_search(query=query, top_k=top_k, rrf_k=rrf_k, query_id=query_id)


if __name__ == "__main__":
    mcp.run()
