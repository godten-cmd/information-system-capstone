"""Tests for MCP retrieval tool functions (direct-call mode, no MCP server).

These tests load real indexes, so they require the index files to exist.
Marked with `integration` marker; use -m "not integration" to skip.
"""

from __future__ import annotations

import pytest

from enterprise_rag.mcp.tools.bm25_tool import bm25_search, reset_index


# ── BM25 tool ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_bm25_search_returns_list():
    results = bm25_search("hotel daily expense limit", top_k=5)
    assert isinstance(results, list)
    assert len(results) <= 5


@pytest.mark.integration
def test_bm25_search_chunk_schema():
    results = bm25_search("authentication policy", top_k=3)
    assert len(results) > 0
    r = results[0]
    assert "chunk_id" in r
    assert "document_id" in r
    assert "rank" in r
    assert "score" in r
    assert "text" in r
    assert "category" in r
    assert "section_path" in r
    assert r["strategy_used"] == "bm25"
    assert r["rank"] == 1


@pytest.mark.integration
def test_bm25_search_top_k_respected():
    for k in [1, 3, 10]:
        results = bm25_search("policy", top_k=k)
        assert len(results) <= k


@pytest.mark.integration
def test_bm25_search_scores_descending():
    results = bm25_search("travel reimbursement expense", top_k=10)
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.integration
def test_bm25_search_latency_recorded():
    results = bm25_search("security two-factor authentication", top_k=5)
    assert results[0]["latency_ms"] >= 0.0


@pytest.mark.integration
def test_bm25_top_k_capped_at_50():
    results = bm25_search("policy", top_k=999)
    assert len(results) <= 50


@pytest.mark.integration
def test_bm25_search_empty_query_returns_list():
    results = bm25_search("", top_k=5)
    assert isinstance(results, list)


# ── Dense tool ────────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_dense_search_returns_list():
    from enterprise_rag.mcp.tools.dense_tool import dense_search
    results = dense_search("exception clause travel reimbursement", top_k=5)
    assert isinstance(results, list)
    assert len(results) <= 5


@pytest.mark.integration
def test_dense_search_chunk_schema():
    from enterprise_rag.mcp.tools.dense_tool import dense_search
    results = dense_search("semantic similarity policy exception", top_k=3)
    assert len(results) > 0
    assert results[0]["strategy_used"] == "dense"


# ── Hybrid tool ───────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_hybrid_search_returns_list():
    from enterprise_rag.mcp.tools.hybrid_tool import hybrid_search
    results = hybrid_search("multi-hop authentication endpoint", top_k=5)
    assert isinstance(results, list)
    assert len(results) <= 5


@pytest.mark.integration
def test_hybrid_search_strategy_label():
    from enterprise_rag.mcp.tools.hybrid_tool import hybrid_search
    results = hybrid_search("aggregation security policy", top_k=3)
    assert len(results) > 0
    assert results[0]["strategy_used"] == "hybrid"


# ── MCP server imports ────────────────────────────────────────────────────────


def test_mcp_server_importable():
    from enterprise_rag.mcp.server import mcp
    assert mcp is not None


def test_mcp_tool_functions_importable():
    from enterprise_rag.mcp.tools import bm25_search, dense_search, hybrid_search
    assert callable(bm25_search)
    assert callable(dense_search)
    assert callable(hybrid_search)
