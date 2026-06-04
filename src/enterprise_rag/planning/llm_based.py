"""LLM-Based Retrieval Planner.

Selects a retrieval strategy (bm25 | dense | hybrid) for each query using an LLM.
Supports three prompt variants (zero_shot, structured, few_shot), provider
abstraction (OpenAI, Anthropic, Mock), on-disk caching, and automatic retry
with fallback on JSON parse failures.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from enterprise_rag.planning.cache import CacheRecord, PlannerCache
from enterprise_rag.planning.parser import parse_plan
from enterprise_rag.planning.prompts.templates import PROMPT_VERSIONS, get_system_prompt
from enterprise_rag.planning.providers.base import BasePlannerProvider
from enterprise_rag.planning.providers.mock_provider import mock_decide
from enterprise_rag.planning.schema import (
    AVAILABLE_STRATEGIES,
    ORACLE_DISCREPANCY_NOTES,
    ORACLE_STRATEGY_MAP,
    LLMDecision,
    estimate_cost,
)

_MAX_RETRIES = 3
_FALLBACK_STRATEGY = "hybrid"


def _classify_error(
    selected: str,
    oracle: str,
    rt: str,
    factors: list[str],
    category: str,
    prompt_version: str,
    oracle_source: str,
) -> str:
    """Assign a category to an incorrect LLM decision."""
    if prompt_version == "zero_shot":
        return "prompt_ambiguity"
    if oracle_source == "empirical_validation":
        return "oracle_disagreement"
    factor_set = set(factors)
    if rt in {"temporal"} or "temporal_reasoning" in factor_set:
        return "reasoning_confusion"
    if rt == "exception" or "exception_handling" in factor_set:
        return "reasoning_confusion"
    if "multi_document_dependency" in factor_set or rt == "multi_hop":
        return "difficulty_factor_confusion"
    if "table_dependency" in factor_set:
        return "difficulty_factor_confusion"
    if category in {"System Design Documents", "API Documentation"}:
        return "category_confusion"
    return "oracle_disagreement"


class LLMBasedPlanner:
    """Adaptive retrieval planner backed by an LLM provider.

    Args:
        provider: LLM provider instance (OpenAI, Anthropic, or Mock).
        prompt_version: "zero_shot", "structured", or "few_shot".
        cache_dir: Directory for caching decisions. None = no cache.
        max_retries: How many times to retry on JSON parse failure.
    """

    def __init__(
        self,
        provider: BasePlannerProvider,
        prompt_version: str = "structured",
        cache_dir: Path | None = None,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version must be one of {list(PROMPT_VERSIONS)}")
        self._provider = provider
        self._prompt_version = prompt_version
        self._cache = PlannerCache(cache_dir) if cache_dir is not None else None
        self._max_retries = max_retries
        self._system_prompt = get_system_prompt()
        self._template = PROMPT_VERSIONS[prompt_version]

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    @property
    def provider(self) -> BasePlannerProvider:
        return self._provider

    def plan(self, query: dict[str, Any]) -> LLMDecision:
        """Produce a single LLMDecision for one query."""
        query_id = query["query_id"]
        model = self._provider.model_name
        oracle_raw = query.get("planner_oracle_strategy", "hybrid") or "hybrid"
        oracle_mapped = ORACLE_STRATEGY_MAP.get(oracle_raw, oracle_raw)
        oracle_source = query.get("oracle_strategy_source", "") or ""
        rt = query.get("reasoning_type", "") or ""
        factors: list[str] = query.get("retrieval_difficulty_factors", []) or []
        category = query.get("category", "") or ""
        discrepancy = rt in ORACLE_DISCREPANCY_NOTES and oracle_source != "empirical_validation"
        disc_note = ORACLE_DISCREPANCY_NOTES.get(rt, "")

        # ── 1. Check cache ────────────────────────────────────────────────────
        if self._cache is not None:
            cached = self._cache.get(query_id, model, self._prompt_version)
            if cached is not None:
                strategy = cached.selected_strategy
                is_correct = strategy == oracle_mapped
                cost = estimate_cost(model, cached.prompt_tokens, cached.completion_tokens)
                return LLMDecision(
                    query_id=query_id,
                    selected_strategy=strategy,
                    matched_rules=[],
                    planner_confidence=cached.confidence,
                    oracle_strategy=oracle_raw,
                    oracle_strategy_mapped=oracle_mapped,
                    oracle_strategy_source=oracle_source,
                    is_correct=is_correct,
                    reasoning_type=rt,
                    category=category,
                    difficulty_factors=factors,
                    oracle_discrepancy=discrepancy,
                    discrepancy_note=disc_note,
                    prompt_version=self._prompt_version,
                    model=model,
                    raw_response=cached.raw_response,
                    parsed_reasoning=cached.reasoning,
                    parse_attempts=cached.parse_attempts,
                    parse_success=cached.parse_success,
                    fallback_used=not cached.parse_success,
                    prompt_tokens=cached.prompt_tokens,
                    completion_tokens=cached.completion_tokens,
                    latency_s=cached.latency_s,
                    estimated_cost_usd=cost,
                    error_category="" if is_correct else _classify_error(
                        strategy, oracle_mapped, rt, factors, category,
                        self._prompt_version, oracle_source,
                    ),
                )

        # ── 2. Call provider (mock short-circuits with full query context) ───────
        from enterprise_rag.planning.providers.mock_provider import MockProvider, mock_decide, _PROMPT_TOKENS, _COMPLETION_TOKENS  # noqa: PLC0415

        if isinstance(self._provider, MockProvider):
            # MockProvider gets the full query so it can use oracle for accuracy simulation.
            strategy, confidence, reasoning = mock_decide(query, self._prompt_version)
            raw_response = json.dumps({"selected_strategy": strategy, "confidence": confidence, "reasoning": reasoning})
            total_prompt_tokens = _PROMPT_TOKENS.get(self._prompt_version, 300)
            total_completion_tokens = _COMPLETION_TOKENS
            total_latency = 0.0001
            parse_success = True
            parse_attempts = 1
            fallback_used = False
        else:
            # ── 3. Build prompt and call real LLM with retries ────────────────────
            user_prompt = self._template.build(query)
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_latency = 0.0
            raw_response = ""
            parse_attempts = 0
            plan = None

            for attempt in range(1, self._max_retries + 1):
                parse_attempts = attempt
                resp = self._provider.complete(user_prompt, self._system_prompt)
                raw_response = resp.content
                total_prompt_tokens += resp.prompt_tokens
                total_completion_tokens += resp.completion_tokens
                total_latency += resp.latency_s

                plan = parse_plan(raw_response)
                if plan is not None:
                    break

                user_prompt = (
                    user_prompt
                    + f"\n\nYour previous response was not valid JSON: {raw_response!r}\n"
                    "Respond ONLY with valid JSON: "
                    '{"selected_strategy": "bm25|dense|hybrid", "confidence": 0.0-1.0, "reasoning": "..."}'
                )

            parse_success = plan is not None
            fallback_used = not parse_success

            if fallback_used:
                strategy = _FALLBACK_STRATEGY
                confidence = 0.5
                reasoning = f"Parse failure after {parse_attempts} attempts; fell back to {_FALLBACK_STRATEGY}"
            else:
                strategy = plan.selected_strategy  # type: ignore[union-attr]
                confidence = plan.confidence  # type: ignore[union-attr]
                reasoning = plan.reasoning  # type: ignore[union-attr]

        cost = estimate_cost(model, total_prompt_tokens, total_completion_tokens)
        is_correct = strategy == oracle_mapped

        # ── 4. Write to cache ─────────────────────────────────────────────────
        if self._cache is not None:
            cache_key = PlannerCache.make_key(query_id, model, self._prompt_version)
            self._cache.put(CacheRecord(
                cache_key=cache_key,
                query_id=query_id,
                model=model,
                prompt_version=self._prompt_version,
                raw_response=raw_response,
                selected_strategy=strategy,
                confidence=confidence,
                reasoning=reasoning,
                parse_success=parse_success,
                parse_attempts=parse_attempts,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                latency_s=total_latency,
                timestamp=datetime.now(timezone.utc).isoformat(),
            ))

        return LLMDecision(
            query_id=query_id,
            selected_strategy=strategy,
            matched_rules=[],
            planner_confidence=confidence,
            oracle_strategy=oracle_raw,
            oracle_strategy_mapped=oracle_mapped,
            oracle_strategy_source=oracle_source,
            is_correct=is_correct,
            reasoning_type=rt,
            category=category,
            difficulty_factors=factors,
            oracle_discrepancy=discrepancy,
            discrepancy_note=disc_note,
            prompt_version=self._prompt_version,
            model=model,
            raw_response=raw_response,
            parsed_reasoning=reasoning,
            parse_attempts=parse_attempts,
            parse_success=parse_success,
            fallback_used=fallback_used,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            latency_s=total_latency,
            estimated_cost_usd=cost,
            error_category="" if is_correct else _classify_error(
                strategy, oracle_mapped, rt, factors, category,
                self._prompt_version, oracle_source,
            ),
        )

    def plan_batch(
        self,
        queries: list[dict[str, Any]],
        *,
        verbose: bool = False,
    ) -> tuple[list[LLMDecision], float]:
        """Plan all queries; return (decisions, wall_clock_seconds)."""
        t0 = time.monotonic()
        decisions = []
        for i, q in enumerate(queries):
            decisions.append(self.plan(q))
            if verbose and (i + 1) % 50 == 0:
                elapsed = time.monotonic() - t0
                print(f"  [{i+1}/{len(queries)}] {elapsed:.1f}s elapsed")
        return decisions, time.monotonic() - t0


def run_prompt_comparison(
    queries: list[dict[str, Any]],
    provider: BasePlannerProvider,
    cache_dir: Path | None = None,
    max_retries: int = _MAX_RETRIES,
) -> dict[str, list[LLMDecision]]:
    """Run all three prompt versions and return decisions keyed by version."""
    results: dict[str, list[LLMDecision]] = {}
    for version in PROMPT_VERSIONS:
        planner = LLMBasedPlanner(
            provider=provider,
            prompt_version=version,
            cache_dir=cache_dir,
            max_retries=max_retries,
        )
        decisions, _ = planner.plan_batch(queries)
        results[version] = decisions
    return results
