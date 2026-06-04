"""Tests for QA Agent: classification, decomposition, parsing, heuristics."""

from __future__ import annotations

import json

import pytest

from enterprise_rag.agents.query_analysis import (
    QAAgent,
    _extract_json,
    _heuristic_classify,
    _heuristic_signals,
    _parse_qa_response,
)
from enterprise_rag.agents.state import QueryPlan


# ── Heuristic classifier ───────────────────────────────────────────────────────


@pytest.mark.parametrize("query,expected", [
    ("What is the maximum daily hotel expense allowed?", "single_hop"),
    ("Which API version was released before the Q3 2024 overhaul?", "temporal"),
    ("Under what conditions is an employee exempt from pre-approval?", "exception"),
    ("List all security policy sections that mandate two-factor authentication.", "aggregation"),
    ("Compare the leave entitlement for senior vs junior engineers.", "comparison"),
    ("Which endpoints use the same auth as the data export service?", "multi_hop"),
])
def test_heuristic_classify(query, expected):
    assert _heuristic_classify(query) == expected


def test_heuristic_signals_temporal():
    signals = _heuristic_signals("Which version was released before Q3 2024?")
    assert "temporal_reasoning" in signals


def test_heuristic_signals_exception():
    signals = _heuristic_signals("Under what conditions is an employee exempt?")
    assert "exception_handling" in signals


def test_heuristic_signals_empty_for_plain():
    signals = _heuristic_signals("What is the hotel limit?")
    assert signals == []


# ── JSON extraction ────────────────────────────────────────────────────────────


def test_extract_json_clean():
    raw = '{"reasoning_type": "single_hop", "difficulty_signals": []}'
    result = _extract_json(raw)
    assert result["reasoning_type"] == "single_hop"


def test_extract_json_with_preamble():
    raw = 'Sure! Here is the answer: {"reasoning_type": "temporal", "difficulty_signals": ["temporal_reasoning"]}'
    result = _extract_json(raw)
    assert result["reasoning_type"] == "temporal"


def test_extract_json_invalid_raises():
    with pytest.raises(ValueError):
        _extract_json("no json here at all")


# ── _parse_qa_response ────────────────────────────────────────────────────────


def _make_raw(
    reasoning_type="single_hop",
    signals=None,
    is_decomposed=False,
    sub_queries=None,
    confidence=0.9,
) -> str:
    if signals is None:
        signals = []
    if sub_queries is None:
        sub_queries = [
            {"sub_query_id": "q1", "text": "What is the hotel limit?",
             "strategy_hint": None, "target_category": None, "depends_on": None}
        ]
    return json.dumps({
        "reasoning_type": reasoning_type,
        "difficulty_signals": signals,
        "is_decomposed": is_decomposed,
        "sub_queries": sub_queries,
        "classification_confidence": confidence,
    })


def test_parse_single_hop():
    raw = _make_raw("single_hop", confidence=0.95)
    plan = _parse_qa_response(raw, "q1", "What is the hotel limit?")
    assert isinstance(plan, QueryPlan)
    assert plan.reasoning_type == "single_hop"
    assert len(plan.sub_queries) == 1
    assert plan.classification_confidence == pytest.approx(0.95)


def test_parse_temporal():
    raw = _make_raw("temporal", signals=["temporal_reasoning"])
    plan = _parse_qa_response(raw, "q2", "Which API version before Q3 2024?")
    assert plan.reasoning_type == "temporal"
    assert "temporal_reasoning" in plan.difficulty_signals


def test_parse_multi_hop_decomposed():
    sub_queries = [
        {"sub_query_id": "q1", "text": "What auth does data export use?",
         "strategy_hint": "dense", "target_category": "System Design Documents",
         "depends_on": None},
        {"sub_query_id": "q2", "text": "Which endpoints use {auth_method}?",
         "strategy_hint": "bm25", "target_category": "API Documentation",
         "depends_on": "q1"},
    ]
    raw = _make_raw("multi_hop", is_decomposed=True, sub_queries=sub_queries)
    plan = _parse_qa_response(raw, "q3", "Which endpoints use same auth as data export?")
    assert plan.reasoning_type == "multi_hop"
    assert plan.is_decomposed is True
    assert len(plan.sub_queries) == 2
    assert plan.sub_queries[1].depends_on == "q1"
    assert plan.sub_queries[1].strategy_hint == "bm25"


def test_parse_unknown_reasoning_type_defaults():
    raw = _make_raw("unknown_type_xyz")
    plan = _parse_qa_response(raw, "q1", "test")
    assert plan.reasoning_type == "single_hop"


def test_parse_invalid_strategy_hint_stripped():
    sub_queries = [
        {"sub_query_id": "q1", "text": "test", "strategy_hint": "colbert",
         "target_category": None, "depends_on": None}
    ]
    raw = _make_raw(sub_queries=sub_queries)
    plan = _parse_qa_response(raw, "q1", "test")
    assert plan.sub_queries[0].strategy_hint is None


def test_parse_confidence_clamped():
    raw = _make_raw(confidence=1.5)
    plan = _parse_qa_response(raw, "q1", "test")
    assert plan.classification_confidence == pytest.approx(1.0)

    raw2 = _make_raw(confidence=-0.5)
    plan2 = _parse_qa_response(raw2, "q1", "test")
    assert plan2.classification_confidence == pytest.approx(0.0)


def test_parse_missing_sub_queries_creates_fallback():
    raw = json.dumps({"reasoning_type": "single_hop", "difficulty_signals": [],
                      "is_decomposed": False, "classification_confidence": 0.8})
    plan = _parse_qa_response(raw, "q1", "fallback query")
    assert len(plan.sub_queries) == 1
    assert plan.sub_queries[0].text == "fallback query"


def test_parse_multi_hop_single_sub_query_marks_not_decomposed():
    # is_decomposed=True but only one sub-query → should be corrected to False
    sub_queries = [
        {"sub_query_id": "q1", "text": "test", "strategy_hint": None,
         "target_category": None, "depends_on": None}
    ]
    raw = _make_raw("multi_hop", is_decomposed=True, sub_queries=sub_queries)
    plan = _parse_qa_response(raw, "q1", "test")
    assert plan.is_decomposed is False


# ── QAAgent class ──────────────────────────────────────────────────────────────


def test_qa_agent_no_provider_uses_heuristic():
    agent = QAAgent(provider=None)
    plan = agent.analyze("q1", "What is the hotel limit?")
    assert isinstance(plan, QueryPlan)
    assert plan.query_id == "q1"
    assert plan.reasoning_type in {"single_hop", "aggregation", "comparison",
                                   "exception", "temporal", "multi_hop"}
    assert plan.classification_confidence == pytest.approx(0.6)  # heuristic confidence


def test_qa_agent_oracle_override_bypasses_llm():
    agent = QAAgent(provider=None)
    plan = agent.analyze("q1", "test query", oracle_reasoning_type="temporal")
    assert plan.reasoning_type == "temporal"
    assert plan.classification_confidence == pytest.approx(1.0)
    assert len(plan.sub_queries) == 1


def test_qa_agent_oracle_override_single_hop():
    agent = QAAgent(provider=None)
    plan = agent.analyze("q1", "some query", oracle_reasoning_type="single_hop")
    assert plan.reasoning_type == "single_hop"
    assert plan.is_decomposed is False


@pytest.mark.parametrize("query,expected_type", [
    ("What is the maximum daily hotel expense allowed under the travel policy?", "single_hop"),
    ("Which API version was released before the authentication overhaul in Q3 2024?", "temporal"),
    ("List all security policy sections that mandate two-factor authentication.", "aggregation"),
    ("Compare leave entitlement for senior engineers versus junior engineers.", "comparison"),
    ("Under what conditions is an employee exempt from the standard pre-approval requirement?", "exception"),
])
def test_qa_agent_heuristic_classification(query, expected_type):
    agent = QAAgent(provider=None)
    plan = agent.analyze("q_test", query)
    assert plan.reasoning_type == expected_type, f"Expected {expected_type} for: {query}"


# ── Mock provider integration ─────────────────────────────────────────────────


class _MockQAProvider:
    """Minimal mock provider that returns a fixed JSON response."""

    model_name = "mock-qa"

    def complete(self, user_prompt, system_prompt, **kwargs):
        from enterprise_rag.planning.providers.base import ProviderResponse
        payload = json.dumps({
            "reasoning_type": "aggregation",
            "difficulty_signals": ["multi_document_dependency"],
            "is_decomposed": False,
            "sub_queries": [
                {"sub_query_id": "q1", "text": user_prompt.split("Query: ")[-1].strip(),
                 "strategy_hint": "hybrid", "target_category": None, "depends_on": None}
            ],
            "classification_confidence": 0.88,
        })
        return ProviderResponse(
            content=payload,
            prompt_tokens=100,
            completion_tokens=60,
            model="mock-qa",
            latency_s=0.01,
        )


def test_qa_agent_with_mock_provider():
    agent = QAAgent(provider=_MockQAProvider())
    plan = agent.analyze("q1", "List all security policies mandating 2FA.")
    assert plan.reasoning_type == "aggregation"
    assert plan.classification_confidence == pytest.approx(0.88)
    assert "multi_document_dependency" in plan.difficulty_signals


def test_qa_agent_provider_failure_falls_back_to_heuristic():
    class _FailingProvider:
        model_name = "fail"
        def complete(self, *args, **kwargs):
            raise RuntimeError("API error")

    agent = QAAgent(provider=_FailingProvider())
    plan = agent.analyze("q1", "What is the hotel limit?")
    # Should not raise — falls back to heuristic
    assert isinstance(plan, QueryPlan)
    assert plan.classification_confidence == pytest.approx(0.6)


# ── Integration test (requires real indexes/API) ───────────────────────────────


@pytest.mark.integration
def test_qa_agent_end_to_end_no_api(monkeypatch):
    """QA Agent without API key uses heuristic (no external calls)."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Reset singleton so it picks up the monkeypatched env
    import enterprise_rag.agents.query_analysis as qa_mod
    qa_mod._agent = None

    agent = qa_mod.get_qa_agent()
    plan = agent.analyze("q_int", "What is the maximum hotel expense?")
    assert plan.reasoning_type in {"single_hop", "temporal", "exception",
                                   "aggregation", "comparison", "multi_hop"}
    qa_mod._agent = None  # clean up singleton
