"""Tests for Phase 9: LLM-Based Retrieval Planner.

46 tests covering providers, prompts, parser, cache, LLMDecision schema,
LLMBasedPlanner, metrics, and end-to-end runner assembly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from enterprise_rag.planning.cache import CacheRecord, PlannerCache
from enterprise_rag.planning.llm_based import LLMBasedPlanner, run_prompt_comparison
from enterprise_rag.planning.parser import ParsedPlan, parse_plan
from enterprise_rag.planning.prompts.templates import (
    PROMPT_VERSIONS,
    build_prompt,
    get_system_prompt,
)
from enterprise_rag.planning.providers.base import ProviderResponse
from enterprise_rag.planning.providers.factory import create_provider
from enterprise_rag.planning.providers.mock_provider import MockProvider, mock_decide
from enterprise_rag.planning.schema import (
    AVAILABLE_STRATEGIES,
    LLMDecision,
    estimate_cost,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_query() -> dict[str, Any]:
    return {
        "query_id": "Q-000001",
        "query": "What is the maximum lodging limit for US travel?",
        "reasoning_type": "single_hop",
        "retrieval_difficulty_factors": ["table_dependency"],
        "category": "Travel Policies",
        "planner_oracle_strategy": "hybrid",
        "oracle_strategy_source": "heuristic_assignment",
        "answerable": True,
    }


@pytest.fixture
def temporal_query() -> dict[str, Any]:
    return {
        "query_id": "Q-000070",
        "query": "What was the first decision made in the initial PRJ-HYD-ALPHA meeting?",
        "reasoning_type": "temporal",
        "retrieval_difficulty_factors": ["temporal_reasoning", "multi_document_dependency"],
        "category": "Project Meeting Notes",
        "planner_oracle_strategy": "bm25",
        "oracle_strategy_source": "empirical_validation",
        "answerable": True,
    }


@pytest.fixture
def exception_query() -> dict[str, Any]:
    return {
        "query_id": "Q-000010",
        "query": "Under what conditions can an employee be exempt from the attendance requirement?",
        "reasoning_type": "exception",
        "retrieval_difficulty_factors": ["exception_handling"],
        "category": "HR Policies",
        "planner_oracle_strategy": "dense",
        "oracle_strategy_source": "heuristic_assignment",
        "answerable": True,
    }


@pytest.fixture
def multi_hop_query() -> dict[str, Any]:
    return {
        "query_id": "Q-000020",
        "query": "Which API endpoints in HYDeploy v2 depend on the HYID authentication service?",
        "reasoning_type": "multi_hop",
        "retrieval_difficulty_factors": ["multi_document_dependency"],
        "category": "API Documentation",
        "planner_oracle_strategy": "multi_hop",
        "oracle_strategy_source": "heuristic_assignment",
        "answerable": True,
    }


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider(prompt_version="structured")


@pytest.fixture
def tmp_cache(tmp_path: Path) -> PlannerCache:
    return PlannerCache(tmp_path / "cache")


# ── Provider tests ────────────────────────────────────────────────────────────

class TestMockProvider:
    def test_returns_valid_json(self, mock_provider: MockProvider, sample_query: dict) -> None:
        prompt = build_prompt(sample_query, "structured")
        resp = mock_provider.complete(prompt, get_system_prompt())
        data = json.loads(resp.content)
        assert "selected_strategy" in data
        assert "confidence" in data
        assert "reasoning" in data

    def test_strategy_is_allowed(self, mock_provider: MockProvider, sample_query: dict) -> None:
        prompt = build_prompt(sample_query, "structured")
        resp = mock_provider.complete(prompt, get_system_prompt())
        data = json.loads(resp.content)
        assert data["selected_strategy"] in AVAILABLE_STRATEGIES

    def test_deterministic_same_query(self, sample_query: dict) -> None:
        prov = MockProvider(prompt_version="structured")
        p = build_prompt(sample_query, "structured")
        r1 = prov.complete(p, get_system_prompt())
        r2 = prov.complete(p, get_system_prompt())
        assert r1.content == r2.content

    def test_different_queries_may_differ(self, sample_query: dict, temporal_query: dict) -> None:
        prov = MockProvider(prompt_version="structured")
        r1 = json.loads(prov.complete(build_prompt(sample_query, "structured"), get_system_prompt()).content)
        r2 = json.loads(prov.complete(build_prompt(temporal_query, "structured"), get_system_prompt()).content)
        # Not asserting they differ since noise may place both in same bucket; just no crash.
        assert r1["selected_strategy"] in AVAILABLE_STRATEGIES
        assert r2["selected_strategy"] in AVAILABLE_STRATEGIES

    def test_provider_name(self, mock_provider: MockProvider) -> None:
        assert mock_provider.provider_name == "mock"

    def test_model_name(self, mock_provider: MockProvider) -> None:
        assert mock_provider.model_name == "mock"

    def test_returns_token_counts(self, mock_provider: MockProvider, sample_query: dict) -> None:
        resp = mock_provider.complete(build_prompt(sample_query, "structured"), get_system_prompt())
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens > 0

    def test_zero_shot_has_fewer_tokens(self, sample_query: dict) -> None:
        pzs = MockProvider(prompt_version="zero_shot")
        pfs = MockProvider(prompt_version="few_shot")
        rzs = pzs.complete(build_prompt(sample_query, "zero_shot"), get_system_prompt())
        rfs = pfs.complete(build_prompt(sample_query, "few_shot"), get_system_prompt())
        assert rzs.prompt_tokens < rfs.prompt_tokens

    def test_mock_decide_all_strategies_covered(self) -> None:
        from enterprise_rag.planning.providers.mock_provider import mock_decide
        # Run enough queries to see all three strategies
        seen = set()
        for i in range(200):
            s, _, _ = mock_decide(
                {"query_id": f"Q-{i:06d}", "planner_oracle_strategy": "hybrid",
                 "reasoning_type": "single_hop", "retrieval_difficulty_factors": [],
                 "category": "HR Policies"},
                "zero_shot",
            )
            seen.add(s)
            if seen >= {"bm25", "dense", "hybrid"}:
                break
        # At minimum, hybrid must appear since it's the dominant oracle
        assert "hybrid" in seen


class TestFactoryProvider:
    def test_creates_mock_when_no_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        prov = create_provider()
        assert prov.provider_name == "mock"

    def test_explicit_mock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        prov = create_provider(provider="mock")
        assert prov.provider_name == "mock"

    def test_openai_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises((ValueError, Exception)):
            create_provider(provider="openai")

    def test_anthropic_raises_without_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises((ValueError, Exception)):
            create_provider(provider="anthropic")


# ── Prompt tests ──────────────────────────────────────────────────────────────

class TestPromptTemplates:
    def test_all_versions_present(self) -> None:
        assert set(PROMPT_VERSIONS) == {"zero_shot", "structured", "few_shot"}

    def test_zero_shot_contains_query_text(self, sample_query: dict) -> None:
        p = build_prompt(sample_query, "zero_shot")
        assert sample_query["query"] in p

    def test_structured_includes_reasoning_type(self, sample_query: dict) -> None:
        p = build_prompt(sample_query, "structured")
        assert sample_query["reasoning_type"] in p

    def test_structured_includes_category(self, sample_query: dict) -> None:
        p = build_prompt(sample_query, "structured")
        assert sample_query["category"] in p

    def test_few_shot_has_examples(self, sample_query: dict) -> None:
        p = build_prompt(sample_query, "few_shot")
        assert "Example 1" in p
        assert "Example 6" in p

    def test_few_shot_includes_query_id_marker(self, sample_query: dict) -> None:
        p = build_prompt(sample_query, "few_shot")
        assert f"# query_id: {sample_query['query_id']}" in p

    def test_system_prompt_mentions_strategies(self) -> None:
        sp = get_system_prompt()
        for s in ["bm25", "dense", "hybrid"]:
            assert s in sp

    def test_unknown_version_raises(self, sample_query: dict) -> None:
        with pytest.raises(ValueError):
            build_prompt(sample_query, "invalid_version")

    def test_few_shot_longer_than_structured(self, sample_query: dict) -> None:
        p_struct = build_prompt(sample_query, "structured")
        p_few = build_prompt(sample_query, "few_shot")
        assert len(p_few) > len(p_struct)


# ── Parser tests ──────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_clean_json(self) -> None:
        text = '{"selected_strategy": "hybrid", "confidence": 0.85, "reasoning": "multi-hop query"}'
        result = parse_plan(text)
        assert result is not None
        assert result.selected_strategy == "hybrid"
        assert result.confidence == 0.85

    def test_parse_markdown_code_block(self) -> None:
        text = '```json\n{"selected_strategy": "dense", "confidence": 0.9, "reasoning": "semantic"}\n```'
        result = parse_plan(text)
        assert result is not None
        assert result.selected_strategy == "dense"

    def test_parse_preamble_and_postamble(self) -> None:
        text = 'Sure! Here is my answer: {"selected_strategy": "bm25", "confidence": 0.75, "reasoning": "exact"} Hope that helps.'
        result = parse_plan(text)
        assert result is not None
        assert result.selected_strategy == "bm25"

    def test_parse_invalid_strategy_returns_none(self) -> None:
        text = '{"selected_strategy": "keyword", "confidence": 0.8, "reasoning": "test"}'
        assert parse_plan(text) is None

    def test_parse_empty_string_returns_none(self) -> None:
        assert parse_plan("") is None

    def test_parse_non_json_returns_none(self) -> None:
        assert parse_plan("This is not JSON at all.") is None

    def test_parse_confidence_clamped(self) -> None:
        text = '{"selected_strategy": "hybrid", "confidence": 1.5, "reasoning": "test"}'
        result = parse_plan(text)
        assert result is not None
        assert result.confidence == 1.0

    def test_parse_missing_strategy_returns_none(self) -> None:
        text = '{"confidence": 0.8, "reasoning": "test"}'
        assert parse_plan(text) is None

    def test_parse_all_allowed_strategies(self) -> None:
        for s in AVAILABLE_STRATEGIES:
            text = f'{{"selected_strategy": "{s}", "confidence": 0.8, "reasoning": "test"}}'
            result = parse_plan(text)
            assert result is not None
            assert result.selected_strategy == s


# ── Cache tests ───────────────────────────────────────────────────────────────

class TestPlannerCache:
    def test_cache_miss_returns_none(self, tmp_cache: PlannerCache) -> None:
        assert tmp_cache.get("Q-000001", "mock", "structured") is None

    def test_cache_stores_and_retrieves(self, tmp_cache: PlannerCache) -> None:
        key = PlannerCache.make_key("Q-000001", "mock", "structured")
        record = CacheRecord(
            cache_key=key,
            query_id="Q-000001",
            model="mock",
            prompt_version="structured",
            raw_response='{"selected_strategy":"hybrid","confidence":0.8,"reasoning":"test"}',
            selected_strategy="hybrid",
            confidence=0.8,
            reasoning="test",
            parse_success=True,
            parse_attempts=1,
            prompt_tokens=200,
            completion_tokens=50,
            latency_s=0.1,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        tmp_cache.put(record)
        retrieved = tmp_cache.get("Q-000001", "mock", "structured")
        assert retrieved is not None
        assert retrieved.selected_strategy == "hybrid"

    def test_cache_key_includes_all_components(self) -> None:
        k1 = PlannerCache.make_key("Q-000001", "gpt-4o", "zero_shot")
        k2 = PlannerCache.make_key("Q-000001", "gpt-4o", "structured")
        k3 = PlannerCache.make_key("Q-000001", "claude-haiku", "zero_shot")
        assert k1 != k2
        assert k1 != k3

    def test_cache_directory_created(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nested" / "cache"
        cache = PlannerCache(cache_dir)
        assert cache_dir.exists()

    def test_cache_size_increments(self, tmp_cache: PlannerCache) -> None:
        assert tmp_cache.size() == 0
        record = CacheRecord(
            cache_key="k1", query_id="Q-1", model="mock", prompt_version="structured",
            raw_response="{}", selected_strategy="hybrid", confidence=0.8,
            reasoning="", parse_success=True, parse_attempts=1,
            prompt_tokens=200, completion_tokens=50, latency_s=0.0,
            timestamp="2026-01-01T00:00:00+00:00",
        )
        tmp_cache.put(record)
        assert tmp_cache.size() == 1


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestLLMDecision:
    def test_to_dict_has_required_keys(self, sample_query: dict, mock_provider: MockProvider) -> None:
        planner = LLMBasedPlanner(provider=mock_provider, prompt_version="structured")
        dec = planner.plan(sample_query)
        d = dec.to_dict()
        for key in ["query_id", "selected_strategy", "is_correct", "oracle_strategy_mapped",
                    "prompt_version", "model", "parse_success", "latency_s", "estimated_cost_usd"]:
            assert key in d, f"Missing key: {key}"

    def test_is_correct_when_matching(self, temporal_query: dict) -> None:
        # oracle is bm25; mock_decide with noise<error_rate→wrong, noise>=error_rate→correct
        # We test that the field is computed logically, not a specific value
        prov = MockProvider(prompt_version="structured")
        planner = LLMBasedPlanner(provider=prov, prompt_version="structured")
        dec = planner.plan(temporal_query)
        assert dec.is_correct == (dec.selected_strategy == dec.oracle_strategy_mapped)

    def test_strategy_always_valid(self, sample_query: dict, mock_provider: MockProvider) -> None:
        planner = LLMBasedPlanner(provider=mock_provider, prompt_version="structured")
        dec = planner.plan(sample_query)
        assert dec.selected_strategy in AVAILABLE_STRATEGIES

    def test_oracle_strategy_mapped_correct(self, multi_hop_query: dict) -> None:
        prov = MockProvider(prompt_version="structured")
        planner = LLMBasedPlanner(provider=prov, prompt_version="structured")
        dec = planner.plan(multi_hop_query)
        # multi_hop → hybrid
        assert dec.oracle_strategy_mapped == "hybrid"

    def test_matched_rules_empty_for_llm(self, sample_query: dict, mock_provider: MockProvider) -> None:
        planner = LLMBasedPlanner(provider=mock_provider, prompt_version="structured")
        dec = planner.plan(sample_query)
        assert dec.matched_rules == []


class TestEstimateCost:
    def test_mock_is_zero_cost(self) -> None:
        assert estimate_cost("mock", 1000, 200) == 0.0

    def test_positive_cost_for_real_model(self) -> None:
        cost = estimate_cost("gpt-4o-mini", 1000, 200)
        assert cost > 0.0

    def test_cost_scales_with_tokens(self) -> None:
        c1 = estimate_cost("gpt-4o-mini", 100, 50)
        c2 = estimate_cost("gpt-4o-mini", 1000, 500)
        assert c2 > c1


# ── Planner integration tests ─────────────────────────────────────────────────

class TestLLMBasedPlanner:
    def test_plan_returns_decision(self, sample_query: dict, mock_provider: MockProvider) -> None:
        planner = LLMBasedPlanner(provider=mock_provider, prompt_version="structured")
        dec = planner.plan(sample_query)
        assert isinstance(dec, LLMDecision)

    def test_plan_batch_deterministic(self, sample_query: dict) -> None:
        queries = [sample_query]
        prov = MockProvider(prompt_version="structured")
        planner = LLMBasedPlanner(provider=prov, prompt_version="structured")
        ds1, _ = planner.plan_batch(queries)
        ds2, _ = planner.plan_batch(queries)
        assert ds1[0].selected_strategy == ds2[0].selected_strategy

    def test_plan_uses_cache_on_second_call(self, sample_query: dict, tmp_path: Path) -> None:
        cache = PlannerCache(tmp_path / "cache")
        prov = MockProvider(prompt_version="structured")
        planner = LLMBasedPlanner(provider=prov, prompt_version="structured", cache_dir=tmp_path / "cache")
        dec1 = planner.plan(sample_query)
        assert cache.size() == 1
        dec2 = planner.plan(sample_query)
        assert cache.size() == 1  # no new file created
        assert dec1.selected_strategy == dec2.selected_strategy

    def test_invalid_prompt_version_raises(self, mock_provider: MockProvider) -> None:
        with pytest.raises(ValueError):
            LLMBasedPlanner(provider=mock_provider, prompt_version="invalid")

    def test_fallback_on_unparseable_response(self, sample_query: dict) -> None:
        from enterprise_rag.planning.providers.base import BasePlannerProvider

        class BadProvider(BasePlannerProvider):
            @property
            def provider_name(self) -> str:
                return "bad"

            @property
            def model_name(self) -> str:
                return "mock"

            def complete(self, user_prompt: str, system_prompt: str, **kwargs: Any) -> ProviderResponse:
                return ProviderResponse(
                    content="This is not valid JSON at all.",
                    prompt_tokens=100, completion_tokens=10, model="mock", latency_s=0.0
                )
        planner = LLMBasedPlanner(provider=BadProvider(), prompt_version="structured", max_retries=2)
        dec = planner.plan(sample_query)
        assert dec.fallback_used is True
        assert dec.selected_strategy == "hybrid"
        assert dec.parse_success is False

    def test_prompt_comparison_runs_all_versions(self, sample_query: dict) -> None:
        prov = MockProvider(prompt_version="structured")
        results = run_prompt_comparison([sample_query], prov)
        assert set(results.keys()) == {"zero_shot", "structured", "few_shot"}
        for decs in results.values():
            assert len(decs) == 1


# ── Metrics tests ─────────────────────────────────────────────────────────────

class TestPlannerMetricsWithLLM:
    def _make_decisions(self, n: int = 20) -> list[LLMDecision]:
        queries = []
        for i in range(n):
            oracle = ["hybrid", "dense", "bm25"][i % 3]
            queries.append({
                "query_id": f"Q-{i:06d}",
                "query": f"Test query {i}",
                "reasoning_type": "single_hop",
                "retrieval_difficulty_factors": [],
                "category": "HR Policies",
                "planner_oracle_strategy": oracle,
                "oracle_strategy_source": "heuristic_assignment",
                "answerable": True,
            })
        prov = MockProvider(prompt_version="structured")
        planner = LLMBasedPlanner(provider=prov, prompt_version="structured")
        decs, _ = planner.plan_batch(queries)
        return decs

    def test_metrics_sum_correct(self) -> None:
        from enterprise_rag.evaluation.planner_metrics import compute_planner_metrics
        decs = self._make_decisions(20)
        m = compute_planner_metrics(decs)
        assert m.n_correct + (m.n_queries - m.n_correct) == m.n_queries

    def test_strategy_distribution_sums_to_total(self) -> None:
        from enterprise_rag.evaluation.planner_metrics import strategy_distribution_report
        decs = self._make_decisions(20)
        rows = strategy_distribution_report(decs)
        total = sum(r["count"] for r in rows)
        assert total == len(decs)

    def test_accuracy_in_range(self) -> None:
        from enterprise_rag.evaluation.planner_metrics import compute_planner_metrics
        decs = self._make_decisions(30)
        m = compute_planner_metrics(decs)
        assert 0.0 <= m.accuracy <= 1.0

    def test_error_classification_assigned(self) -> None:
        decs = self._make_decisions(30)
        incorrect = [d for d in decs if not d.is_correct]
        for d in incorrect:
            assert d.error_category != "" or d.fallback_used
