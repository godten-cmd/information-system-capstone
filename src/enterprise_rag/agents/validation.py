"""Validation Agent: relevance scoring + smart re-retrieval trigger.

Scorer hierarchy:
  rank_proxy  Legacy stub (rank 1→1.0, rank 10→0.1).  Not strategy-aware.
  bm25        Normalized BM25 score for each chunk vs. the sub-query text.
  dense       Cosine similarity from chunk.score (already [0,1]).
  adaptive    Routes to bm25/dense/average based on chunk.strategy_used.
              Default in Phase 12C — no extra cost, strategy-appropriate.
  llm         GPT-4o-mini batch scoring (most accurate, optional).

Failure diagnosis keys:
  zero_recall          All chunks scored ≈ 0; query may have no overlap.
  wrong_strategy       Strategy mismatch (BM25 for exception, dense for temporal).
  insufficient_coverage Partial relevance; need more diverse coverage.
  missing_entity       Multi-hop slot not resolved into meaningful text.
  low_quality          Generic low-score failure; fallback to hybrid.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from enterprise_rag.agents.state import (
    ChunkResult,
    QueryPlan,
    RetrievalOutput,
    RetrievalPlan,
    ValidationDecision,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class ValidationConfig:
    """Tunable parameters for the Validation Agent."""

    relevance_threshold: float = 0.30   # minimum score to be "passed"
    min_passed_chunks: int = 3          # minimum passing chunks to skip re-retrieval
    scorer: str = "adaptive"            # rank_proxy | bm25 | dense | adaptive | llm
    max_loops: int = 3                  # enforced in graph/edges.py; referenced here for logging
    # Per-reasoning-type threshold overrides
    threshold_overrides: dict[str, float] = field(default_factory=lambda: {
        "temporal": 0.20,    # BM25 sparse for temporal; be lenient
        "exception": 0.22,   # exception clauses may have low lexical overlap
        "multi_hop": 0.20,   # slot-filled queries may lose precision
    })

    def threshold_for(self, reasoning_type: str) -> float:
        return self.threshold_overrides.get(reasoning_type, self.relevance_threshold)


# ── Scorer protocol ────────────────────────────────────────────────────────────


class RelevanceScorer(Protocol):
    def score(
        self,
        query: str,
        chunks: list[ChunkResult],
    ) -> list[float]:
        """Return a relevance score in [0, 1] per chunk (same order as input)."""
        ...


# ── Scorer implementations ─────────────────────────────────────────────────────


class RankProxyScorer:
    """Stub scorer: score = max(0, 1 - (rank - 1) * 0.1)."""

    name = "rank_proxy"

    def score(self, query: str, chunks: list[ChunkResult]) -> list[float]:
        return [max(0.0, 1.0 - (c.rank - 1) * 0.1) for c in chunks]


class BM25RelevanceScorer:
    """Score chunks using their normalized BM25 relevance against the query.

    Algorithm:
      1. Run BM25.get_scores(tokenize(query)) over the full corpus (fast, vectorised).
      2. Normalize each chunk's score by the corpus-max for this query.
      3. Chunks not found in the index (shouldn't happen) receive score 0.
    """

    name = "bm25"

    def score(self, query: str, chunks: list[ChunkResult]) -> list[float]:
        if not chunks:
            return []
        try:
            from enterprise_rag.mcp.tools.bm25_tool import _get_index
            from enterprise_rag.retrieval.tokenization import tokenize_query

            idx = _get_index()
            tokens = tokenize_query(query)
            if not tokens:
                return [0.0] * len(chunks)

            all_scores: np.ndarray = idx._bm25.get_scores(tokens)
            max_score = float(all_scores.max())
            if max_score < 1e-9:
                return [0.0] * len(chunks)

            chunk_id_to_pos: dict[str, int] = {
                cid: i for i, cid in enumerate(idx._meta.chunk_ids)
            }
            scores: list[float] = []
            for chunk in chunks:
                pos = chunk_id_to_pos.get(chunk.chunk_id)
                if pos is None:
                    scores.append(0.0)
                else:
                    scores.append(round(float(all_scores[pos]) / max_score, 4))
            return scores

        except Exception as exc:
            logger.warning("BM25RelevanceScorer failed: %s — falling back to rank proxy", exc)
            return RankProxyScorer().score(query, chunks)


class DenseRelevanceScorer:
    """Use chunk.score (cosine similarity) directly; it is already in [0, 1]."""

    name = "dense"

    def score(self, query: str, chunks: list[ChunkResult]) -> list[float]:
        return [round(max(0.0, min(1.0, c.score)), 4) for c in chunks]


class AdaptiveScorer:
    """Route each chunk to bm25 / dense scorer based on chunk.strategy_used.

    For hybrid chunks: average of BM25 and dense scores.
    Falls back to BM25 scorer if strategy_used is unrecognised.
    """

    name = "adaptive"

    def __init__(self) -> None:
        self._bm25 = BM25RelevanceScorer()
        self._dense = DenseRelevanceScorer()

    def score(self, query: str, chunks: list[ChunkResult]) -> list[float]:
        if not chunks:
            return []

        # Group chunks by strategy
        groups: dict[str, list[tuple[int, ChunkResult]]] = {}
        for i, c in enumerate(chunks):
            groups.setdefault(c.strategy_used or "bm25", []).append((i, c))

        scores = [0.0] * len(chunks)
        for strategy, indexed_chunks in groups.items():
            sub_chunks = [c for _, c in indexed_chunks]

            if strategy == "dense":
                sub_scores = self._dense.score(query, sub_chunks)
            elif strategy == "hybrid":
                b = self._bm25.score(query, sub_chunks)
                d = self._dense.score(query, sub_chunks)
                sub_scores = [(bv + dv) / 2 for bv, dv in zip(b, d)]
            else:  # bm25 or unknown
                sub_scores = self._bm25.score(query, sub_chunks)

            for (orig_idx, _), s in zip(indexed_chunks, sub_scores):
                scores[orig_idx] = s

        return scores


class LLMRelevanceScorer:
    """Batch-score chunks using an LLM (GPT-4o-mini style prompt).

    Sends up to BATCH_SIZE chunks per API call; falls back to adaptive scorer
    on any error.

    Cost note: ~$0.001 per 10 chunks with GPT-4o-mini at current pricing.
    """

    name = "llm"
    BATCH_SIZE = 5

    _SYSTEM = (
        "You are a relevance judge for enterprise document retrieval. "
        "Score each passage for relevance to the query: "
        "0.0 = not relevant, 0.5 = partially relevant, 1.0 = highly relevant. "
        "Respond ONLY with a JSON array of floats in the same order as the passages."
    )

    def __init__(self, provider: Any) -> None:
        self._provider = provider
        self._fallback = AdaptiveScorer()

    def score(self, query: str, chunks: list[ChunkResult]) -> list[float]:
        if not chunks:
            return []
        if self._provider is None:
            return self._fallback.score(query, chunks)

        results: list[float] = []
        for batch_start in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[batch_start : batch_start + self.BATCH_SIZE]
            batch_scores = self._score_batch(query, batch)
            results.extend(batch_scores)
        return results

    def _score_batch(self, query: str, batch: list[ChunkResult]) -> list[float]:
        passages = "\n".join(f"{i+1}. {c.text[:300]}" for i, c in enumerate(batch))
        user = f"Query: {query}\n\nPassages:\n{passages}"
        try:
            resp = self._provider.complete(
                user_prompt=user,
                system_prompt=self._SYSTEM,
                max_tokens=60 + len(batch) * 8,
                temperature=0.0,
            )
            raw = resp.content.strip()
            start, end = raw.find("["), raw.rfind("]")
            if start != -1 and end != -1:
                arr = json.loads(raw[start : end + 1])
                scores = [max(0.0, min(1.0, float(s))) for s in arr[: len(batch)]]
                # Pad if LLM returned fewer scores than expected
                while len(scores) < len(batch):
                    scores.append(0.5)
                return scores
        except Exception as exc:
            logger.warning("LLMRelevanceScorer batch failed: %s — using adaptive", exc)

        return self._fallback.score(query, batch)


# ── Failure diagnosis ──────────────────────────────────────────────────────────

_SLOT_RE = re.compile(r"\{(\w+)\}")

# Empirical oracle: which strategy works best per reasoning type (from V1)
_ORACLE_STRATEGY: dict[str, str] = {
    "temporal": "bm25",
    "exception": "dense",
    "multi_hop": "hybrid",
    "aggregation": "hybrid",
    "comparison": "hybrid",
    "single_hop": "hybrid",
}


def _diagnose_failure(
    decisions: list[ValidationDecision],
    chunks: list[ChunkResult],
    query_plan: QueryPlan,
    retrieval_plans: list[RetrievalPlan],
    retrieval_outputs: list[RetrievalOutput],
) -> tuple[str, str | None, str | None]:
    """Determine WHY validation failed and WHAT should change on re-retrieval.

    Returns:
        (failure_reason, suggested_strategy, query_expansion_hint)
        suggested_strategy: None means keep current but broaden query
        query_expansion_hint: "broader" to relax query, "re_slot_fill" for multi-hop
    """
    total = len(decisions)
    if total == 0:
        return "no_chunks", "hybrid", None

    passed_count = sum(1 for d in decisions if d.passed)
    zero_score_count = sum(1 for d in decisions if d.relevance_score < 0.05)
    rt = query_plan.reasoning_type
    current_strategy = retrieval_plans[0].strategy if retrieval_plans else "hybrid"
    oracle = _ORACLE_STRATEGY.get(rt, "hybrid")

    # ── Zero-recall: nothing relevant at all ──────────────────────────────────
    if zero_score_count == total:
        if current_strategy != oracle:
            return "wrong_strategy", oracle, None
        if current_strategy == "hybrid":
            return "too_narrow", None, "broader"
        return "wrong_strategy", "hybrid", None

    # ── Multi-hop: check if entity slots were not properly filled ─────────────
    if rt == "multi_hop":
        for output in retrieval_outputs:
            if _SLOT_RE.search(output.query_text):
                return "missing_entity", current_strategy, "re_slot_fill"

    # ── Strategy mismatch: used bm25 for exception or dense for temporal ──────
    if current_strategy != oracle and zero_score_count > total * 0.6:
        return "wrong_strategy", oracle, None

    # ── Insufficient coverage: some relevant but not enough ───────────────────
    if 0 < passed_count < 3:
        if current_strategy != "hybrid":
            return "insufficient_coverage", "hybrid", None
        return "insufficient_coverage", None, "broader"

    # ── Generic low quality ───────────────────────────────────────────────────
    if current_strategy != "hybrid":
        return "low_quality", "hybrid", None
    return "low_quality", None, "broader"


# ── ValidationAgent ────────────────────────────────────────────────────────────


class ValidationAgent:
    """Validate retrieved chunks and diagnose retrieval failures.

    Args:
        config: Thresholds and scorer selection.
        provider: LLM provider for 'llm' scorer. None = auto-disable llm scorer.
    """

    _SCORER_MAP: dict[str, type] = {
        "rank_proxy": RankProxyScorer,
        "bm25": BM25RelevanceScorer,
        "dense": DenseRelevanceScorer,
        "adaptive": AdaptiveScorer,
    }

    def __init__(
        self,
        config: ValidationConfig | None = None,
        provider: Any | None = None,
    ) -> None:
        self.config = config or ValidationConfig()
        self._provider = provider
        self._scorer = self._build_scorer()

    def _build_scorer(self) -> RelevanceScorer:
        stype = self.config.scorer
        if stype == "llm":
            if self._provider is not None:
                return LLMRelevanceScorer(self._provider)
            logger.warning("LLM scorer requested but no provider — falling back to adaptive")
            return AdaptiveScorer()
        cls = self._SCORER_MAP.get(stype, AdaptiveScorer)
        return cls()

    def validate(
        self,
        chunks: list[ChunkResult],
        query: str,
        query_plan: QueryPlan,
        retrieval_plans: list[RetrievalPlan],
        retrieval_outputs: list[RetrievalOutput],
    ) -> ValidationResult:
        """Score chunks and decide whether to proceed or trigger re-retrieval.

        The scoring query is the raw query for single-sub-query cases.
        For multi-sub-query cases, chunks are scored against their own
        sub-query text (matched via sub_query_id → RetrievalOutput.query_text).
        """
        threshold = self.config.threshold_for(query_plan.reasoning_type)
        min_passed = self.config.min_passed_chunks

        # Build sub-query text lookup (chunk's query_text may differ from raw query)
        subq_text: dict[str, str] = {o.sub_query_id: o.query_text for o in retrieval_outputs}
        subq_text["merged"] = query  # aggregation merged output

        # Score each chunk against its own sub-query text
        decisions: list[ValidationDecision] = []
        if chunks:
            # Group by sub_query_id to score with the right query text
            groups: dict[str, list[tuple[int, ChunkResult]]] = {}
            for i, c in enumerate(chunks):
                key = c.sub_query_id or "merged"
                groups.setdefault(key, []).append((i, c))

            scores = [0.0] * len(chunks)
            for sq_id, indexed_chunks in groups.items():
                sq_query = subq_text.get(sq_id, query)
                sub_chunks = [c for _, c in indexed_chunks]
                sub_scores = self._scorer.score(sq_query, sub_chunks)
                for (orig_idx, _), s in zip(indexed_chunks, sub_scores):
                    scores[orig_idx] = s

            for chunk, s in zip(chunks, scores):
                decisions.append(
                    ValidationDecision(
                        chunk_id=chunk.chunk_id,
                        relevance_score=round(s, 4),
                        passed=s >= threshold,
                        reason=self._scorer.name,  # type: ignore[attr-defined]
                    )
                )

        passed_chunks = [c for c, d in zip(chunks, decisions) if d.passed]
        validation_passed = len(passed_chunks) >= min_passed

        failure_reason = ""
        suggested_strategy = None
        query_expansion_hint = None

        if not validation_passed:
            failure_reason, suggested_strategy, query_expansion_hint = _diagnose_failure(
                decisions, chunks, query_plan, retrieval_plans, retrieval_outputs
            )
            logger.info(
                "Validation FAILED [%s] — reason=%s, suggest=%s, hint=%s",
                query_plan.query_id,
                failure_reason,
                suggested_strategy,
                query_expansion_hint,
            )
        else:
            logger.debug(
                "Validation PASSED [%s] — %d/%d chunks (threshold=%.2f)",
                query_plan.query_id,
                len(passed_chunks),
                len(chunks),
                threshold,
            )

        return ValidationResult(
            passed=validation_passed,
            decisions=decisions,
            passed_count=len(passed_chunks),
            total_count=len(chunks),
            threshold=threshold,
            min_passed=min_passed,
            failure_reason=failure_reason,
            suggested_strategy=suggested_strategy,
            suggested_query_expansion=query_expansion_hint,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_agent: ValidationAgent | None = None


def get_validation_agent(scorer: str | None = None) -> ValidationAgent:
    """Return module-level ValidationAgent, building once on first call.

    Args:
        scorer: Override the scorer type. If None, uses "adaptive".
    """
    global _agent
    if _agent is None or (scorer is not None and _agent.config.scorer != scorer):
        import os
        resolved_scorer = scorer or os.environ.get("V2_VALIDATION_SCORER", "adaptive")
        config = ValidationConfig(scorer=resolved_scorer)
        try:
            from enterprise_rag.planning.providers.factory import create_provider
            provider = create_provider(prompt_version="structured")
        except Exception:
            provider = None
        _agent = ValidationAgent(config=config, provider=provider)
    return _agent
