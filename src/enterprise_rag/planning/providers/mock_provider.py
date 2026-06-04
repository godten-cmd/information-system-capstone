"""Deterministic mock provider for testing and CI without real API keys.

Simulates realistic LLM planner behaviour with per-prompt-version accuracy:
  zero_shot  ~72 %   (no metadata available — higher error rate)
  structured ~82 %   (metadata provided — matches rule planner)
  few_shot   ~84 %   (examples provided — slightly better)

Decision logic follows the same oracle signals as the rule planner but injects
deterministic "noise" so the mock behaves like an imperfect LLM, not an oracle.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from enterprise_rag.planning.providers.base import BasePlannerProvider, ProviderResponse
from enterprise_rag.planning.schema import ORACLE_STRATEGY_MAP

# Error rate per prompt version (fraction of queries decided incorrectly)
_ERROR_RATES: dict[str, float] = {
    "zero_shot": 0.28,
    "structured": 0.18,
    "few_shot": 0.16,
}

# Approximate prompt token counts per version (system + user, pre-rendered)
_PROMPT_TOKENS: dict[str, int] = {
    "zero_shot": 210,
    "structured": 320,
    "few_shot": 740,
}

_COMPLETION_TOKENS = 55  # typical short JSON response


def _noise(query_id: str, prompt_version: str) -> float:
    """Deterministic float in [0, 1) from query_id + prompt_version."""
    digest = hashlib.md5(f"{query_id}|{prompt_version}|mock_v1".encode()).hexdigest()
    return int(digest, 16) % 100_000 / 100_000.0


def _wrong_strategy(oracle: str) -> str:
    """Return the most plausible incorrect strategy (not the oracle)."""
    if oracle == "dense":
        return "hybrid"
    if oracle == "hybrid":
        return "dense"
    # oracle == bm25
    return "hybrid"


def _reasoning_text(strategy: str, rt: str, factors: set[str], category: str, *, correct: bool) -> str:
    if not correct:
        if strategy == "hybrid":
            return "Combined retrieval provides broader coverage across the enterprise knowledge base"
        if strategy == "dense":
            return "Semantic search captures conceptual relationships in enterprise documents"
        return "Keyword matching for exact term retrieval"

    if rt == "temporal" or "temporal_reasoning" in factors:
        return "Temporal query with date anchors; BM25 matches date tokens more reliably than semantic search"
    if rt == "exception" or "exception_handling" in factors:
        return "Exception clause interpretation requires semantic understanding beyond keyword matching"
    if rt == "multi_hop" or "multi_document_dependency" in factors:
        return "Multi-document evidence requires combined lexical and semantic retrieval"
    if "table_dependency" in factors:
        return "Table value lookup: hybrid provides exact numeric matching and surrounding context"
    if category in {"System Design Documents", "API Documentation"} and rt == "single_hop":
        return "Technical single-hop query benefits from semantic search for architectural concepts"
    return f"Strategy {strategy} selected based on query characteristics and document category"


def mock_decide(
    query: dict[str, Any],
    prompt_version: str,
) -> tuple[str, float, str]:
    """Return (selected_strategy, confidence, reasoning) deterministically."""
    rt = query.get("reasoning_type", "") or ""
    factors: set[str] = set(query.get("retrieval_difficulty_factors", []) or [])
    category = query.get("category", "") or ""
    oracle_raw = query.get("planner_oracle_strategy", "hybrid") or "hybrid"
    oracle = ORACLE_STRATEGY_MAP.get(oracle_raw, oracle_raw)

    noise = _noise(query["query_id"], prompt_version)
    error_rate = _ERROR_RATES.get(prompt_version, 0.20)

    if noise < error_rate:
        strategy = _wrong_strategy(oracle)
        confidence = round(0.50 + noise * 0.25, 2)
        reasoning = _reasoning_text(strategy, rt, factors, category, correct=False)
    else:
        strategy = oracle
        confidence = round(0.72 + (1.0 - noise) * 0.28, 2)
        confidence = min(confidence, 0.98)
        reasoning = _reasoning_text(strategy, rt, factors, category, correct=True)

    return strategy, confidence, reasoning


class MockProvider(BasePlannerProvider):
    """Deterministic mock — no API calls, used for testing and CI."""

    def __init__(self, prompt_version: str = "structured") -> None:
        self._prompt_version = prompt_version

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock"

    def complete(
        self,
        user_prompt: str,
        system_prompt: str,
        *,
        max_tokens: int = 200,
        temperature: float = 0.0,
    ) -> ProviderResponse:
        # Extract query_id from the user_prompt marker injected by the planner
        # The planner injects "# query_id: <qid>" as the last line for mock extraction
        query_id = "unknown"
        for line in user_prompt.splitlines():
            if line.startswith("# query_id:"):
                query_id = line.split(":", 1)[1].strip()

        strategy, confidence, reasoning = mock_decide(
            {"query_id": query_id},  # mock only needs query_id for noise
            self._prompt_version,
        )
        content = json.dumps(
            {"selected_strategy": strategy, "confidence": confidence, "reasoning": reasoning}
        )

        t0 = time.monotonic()
        elapsed = time.monotonic() - t0  # near-zero, intentionally

        return ProviderResponse(
            content=content,
            prompt_tokens=_PROMPT_TOKENS.get(self._prompt_version, 300),
            completion_tokens=_COMPLETION_TOKENS,
            model="mock",
            latency_s=elapsed,
        )
