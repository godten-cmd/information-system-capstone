"""Answer Agent: LLM synthesis with inline citation markers.

Synthesis flow:
  1. Select top-N evidence chunks (sorted by rank).
  2. Build numbered evidence block: [1] (category) text…
  3. Call LLM with grounding system prompt → response with [N] markers.
  4. Extract cited chunk IDs from [N] markers.
  5. Compute confidence = mean score of cited chunks; coverage = cited / total evidence.

Fallback (no provider or LLM error):
  Extractive — concatenate top-3 chunks with [N] prefixes.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from enterprise_rag.agents.state import AgentAnswer, ChunkResult, QueryPlan

logger = logging.getLogger(__name__)

_CITE_RE = re.compile(r"\[(\d+)\]")

_SYSTEM_PROMPT = (
    "You are an enterprise document QA assistant. "
    "Answer the question using ONLY the numbered evidence passages provided. "
    "Cite sources with inline [N] markers (e.g., 'The limit is $200 [1].'). "
    "Be concise and factual. "
    "If the evidence does not contain enough information, state that explicitly."
)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class AnswerConfig:
    """Tunable parameters for the Answer Agent."""

    max_evidence_chunks: int = 10  # top-N evidence passed to the LLM context
    max_tokens: int = 512
    temperature: float = 0.1


# ── Helper functions ───────────────────────────────────────────────────────────


def _extract_cited_ids(
    text: str,
    idx_to_chunk: dict[int, ChunkResult],
) -> list[str]:
    """Return chunk_ids for every [N] citation marker found in text.

    Result is sorted by citation index (not by appearance order) and deduped.
    """
    indices = sorted({int(m.group(1)) for m in _CITE_RE.finditer(text)})
    return [idx_to_chunk[i].chunk_id for i in indices if i in idx_to_chunk]


def _mean_score(chunks: list[ChunkResult]) -> float:
    if not chunks:
        return 0.0
    return sum(c.score for c in chunks) / len(chunks)


# ── AnswerAgent ────────────────────────────────────────────────────────────────


class AnswerAgent:
    """Synthesize a grounded answer from validated evidence.

    Args:
        config:   Token and chunk limits.
        provider: LLM provider.  None → always use extractive fallback.
    """

    def __init__(
        self,
        config: AnswerConfig | None = None,
        provider: Any | None = None,
    ) -> None:
        self.config = config or AnswerConfig()
        self._provider = provider

    def synthesize(
        self,
        query: str,
        evidence: list[ChunkResult],
        query_plan: QueryPlan,
    ) -> AgentAnswer:
        """Produce an answer grounded in evidence.

        Returns an AgentAnswer with cited_chunk_ids populated.
        """
        query_id = query_plan.query_id

        if not evidence:
            return AgentAnswer(
                query_id=query_id,
                answer_text="No relevant evidence was found to answer this question.",
                cited_chunk_ids=[],
                confidence=0.0,
                evidence_coverage=0.0,
            )

        top_evidence = sorted(evidence, key=lambda c: c.rank)[: self.config.max_evidence_chunks]

        if self._provider is None:
            return self._extractive_answer(query_id, top_evidence, evidence)

        try:
            return self._llm_answer(query_id, query, top_evidence, evidence)
        except Exception as exc:
            logger.warning("AnswerAgent LLM failed: %s — using extractive fallback", exc)
            return self._extractive_answer(query_id, top_evidence, evidence)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _extractive_answer(
        self,
        query_id: str,
        top_evidence: list[ChunkResult],
        all_evidence: list[ChunkResult],
    ) -> AgentAnswer:
        top3 = top_evidence[:3]
        answer_text = " | ".join(f"[{i + 1}] {c.text[:200]}" for i, c in enumerate(top3))
        cited_ids = [c.chunk_id for c in top3]
        confidence = round(_mean_score(top3), 4)
        coverage = round(len(top3) / max(len(all_evidence), 1), 4)
        return AgentAnswer(
            query_id=query_id,
            answer_text=answer_text,
            cited_chunk_ids=cited_ids,
            confidence=confidence,
            evidence_coverage=coverage,
        )

    def _llm_answer(
        self,
        query_id: str,
        query: str,
        top_evidence: list[ChunkResult],
        all_evidence: list[ChunkResult],
    ) -> AgentAnswer:
        idx_to_chunk: dict[int, ChunkResult] = {}
        lines: list[str] = []
        for i, chunk in enumerate(top_evidence, start=1):
            idx_to_chunk[i] = chunk
            cat = chunk.category or "document"
            lines.append(f"[{i}] ({cat}) {chunk.text[:400]}")

        evidence_block = "\n".join(lines)
        user_prompt = f"Question: {query}\n\nEvidence:\n{evidence_block}\n\nAnswer:"

        resp = self._provider.complete(
            user_prompt=user_prompt,
            system_prompt=_SYSTEM_PROMPT,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        answer_text = resp.content.strip()
        cited_ids = _extract_cited_ids(answer_text, idx_to_chunk)

        # If the LLM cited nothing, default to top-3 chunk IDs
        if not cited_ids:
            cited_ids = [c.chunk_id for c in top_evidence[:3]]

        cited_set = {cid for cid in cited_ids}
        cited_chunks = [c for c in top_evidence if c.chunk_id in cited_set]
        confidence = round(_mean_score(cited_chunks) if cited_chunks else _mean_score(top_evidence), 4)
        coverage = round(len(cited_chunks) / max(len(all_evidence), 1), 4)

        return AgentAnswer(
            query_id=query_id,
            answer_text=answer_text,
            cited_chunk_ids=cited_ids,
            confidence=confidence,
            evidence_coverage=coverage,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────

_agent: AnswerAgent | None = None


def get_answer_agent() -> AnswerAgent:
    """Return module-level AnswerAgent, building once on first call."""
    global _agent
    if _agent is None:
        try:
            from enterprise_rag.planning.providers.factory import create_provider
            provider = create_provider(prompt_version="structured")
        except Exception:
            provider = None
        _agent = AnswerAgent(provider=provider)
    return _agent
