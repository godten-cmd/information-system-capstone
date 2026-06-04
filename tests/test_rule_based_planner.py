"""Tests for the Rule-Based Retrieval Planner (Phase 8).

Covers: schema, oracle mapping, all 7 rules, rule precedence,
planner metrics, and end-to-end planner run assembly.
"""

from __future__ import annotations

import pytest

from enterprise_rag.planning.schema import (
    AVAILABLE_STRATEGIES,
    ORACLE_STRATEGY_MAP,
    PlannerDecision,
)
from enterprise_rag.planning.rule_based import (
    RULES,
    RULES_BY_ID,
    RuleBasedPlanner,
)
from enterprise_rag.evaluation.planner_metrics import (
    compute_planner_metrics,
    compute_rule_contributions,
    group_decisions_by,
    strategy_distribution_report,
)
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _query(**kwargs) -> dict:
    base = {
        "query_id": "Q-001",
        "query": "What is the policy?",
        "reasoning_type": "single_hop",
        "retrieval_difficulty_factors": [],
        "category": "HR Policies",
        "planner_oracle_strategy": "hybrid",
        "oracle_strategy_source": "heuristic_assignment",
        "answerable": True,
    }
    base.update(kwargs)
    return base


@pytest.fixture
def planner() -> RuleBasedPlanner:
    return RuleBasedPlanner()


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_available_strategies_set():
    assert AVAILABLE_STRATEGIES == {"bm25", "dense", "hybrid"}


def test_oracle_strategy_map_covers_all_extended():
    extended = {"keyword", "dense", "hybrid", "table_aware", "multi_hop",
                "metadata_filtered", "hierarchical"}
    assert extended.issubset(ORACLE_STRATEGY_MAP.keys())


def test_oracle_strategy_map_values_are_available():
    for v in ORACLE_STRATEGY_MAP.values():
        assert v in AVAILABLE_STRATEGIES


def test_planner_decision_to_dict_keys():
    d = PlannerDecision(
        query_id="Q-001", selected_strategy="bm25", matched_rules=["R01"],
        planner_confidence=0.80, oracle_strategy="keyword",
        oracle_strategy_mapped="bm25", oracle_strategy_source="heuristic_assignment",
        is_correct=True,
    )
    keys = set(d.to_dict().keys())
    assert "query_id" in keys
    assert "selected_strategy" in keys
    assert "matched_rules" in keys
    assert "planner_confidence" in keys
    assert "oracle_strategy_mapped" in keys
    assert "is_correct" in keys
    assert "oracle_discrepancy" in keys


# ── Oracle mapping tests ──────────────────────────────────────────────────────

@pytest.mark.parametrize("oracle,expected", [
    ("keyword", "bm25"),
    ("dense", "dense"),
    ("hybrid", "hybrid"),
    ("table_aware", "hybrid"),
    ("multi_hop", "hybrid"),
    ("metadata_filtered", "hybrid"),
    ("hierarchical", "dense"),
])
def test_oracle_mapping(oracle, expected):
    assert ORACLE_STRATEGY_MAP[oracle] == expected


# ── Rule unit tests ───────────────────────────────────────────────────────────

def test_rule_r01_temporal_reasoning_type(planner):
    d = planner.plan(_query(reasoning_type="temporal", planner_oracle_strategy="multi_hop"))
    assert d.selected_strategy == "bm25"
    assert "R01" in d.matched_rules


def test_rule_r01_temporal_factor(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        retrieval_difficulty_factors=["temporal_reasoning"],
        planner_oracle_strategy="multi_hop",
    ))
    assert d.selected_strategy == "bm25"
    assert "R01" in d.matched_rules


def test_rule_r01_first_match_wins(planner):
    """Temporal fires before multi_hop even if reasoning_type is multi_hop."""
    d = planner.plan(_query(
        reasoning_type="temporal",
        retrieval_difficulty_factors=["multi_document_dependency"],
        planner_oracle_strategy="multi_hop",
    ))
    assert d.selected_strategy == "bm25"
    assert d.matched_rules[0] == "R01"


def test_rule_r02_exception_reasoning_type(planner):
    d = planner.plan(_query(reasoning_type="exception", planner_oracle_strategy="dense"))
    assert d.selected_strategy == "dense"
    assert "R02" in d.matched_rules


def test_rule_r02_exception_factor(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        retrieval_difficulty_factors=["exception_handling"],
        planner_oracle_strategy="dense",
    ))
    assert d.selected_strategy == "dense"
    assert "R02" in d.matched_rules


def test_rule_r02_before_r04(planner):
    """Exception fires before table_dependency when both present."""
    d = planner.plan(_query(
        reasoning_type="exception",
        retrieval_difficulty_factors=["table_dependency", "exception_handling"],
        planner_oracle_strategy="dense",
    ))
    assert d.matched_rules[0] == "R02"
    assert d.selected_strategy == "dense"


def test_rule_r03_multi_hop_reasoning_type(planner):
    d = planner.plan(_query(reasoning_type="multi_hop", planner_oracle_strategy="multi_hop"))
    assert d.selected_strategy == "hybrid"
    assert "R03" in d.matched_rules


def test_rule_r03_multi_document_factor(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        retrieval_difficulty_factors=["multi_document_dependency"],
        planner_oracle_strategy="multi_hop",
    ))
    assert d.selected_strategy == "hybrid"
    assert "R03" in d.matched_rules


def test_rule_r04_table_dependency(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        retrieval_difficulty_factors=["table_dependency"],
        planner_oracle_strategy="table_aware",
    ))
    assert d.selected_strategy == "hybrid"
    assert "R04" in d.matched_rules


def test_rule_r05_comparison(planner):
    d = planner.plan(_query(reasoning_type="comparison", planner_oracle_strategy="metadata_filtered"))
    assert d.selected_strategy == "hybrid"
    assert "R05" in d.matched_rules


def test_rule_r05_aggregation(planner):
    d = planner.plan(_query(reasoning_type="aggregation", planner_oracle_strategy="table_aware"))
    assert d.selected_strategy == "hybrid"
    assert "R05" in d.matched_rules


def test_rule_r06_system_design_single_hop(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        category="System Design Documents",
        planner_oracle_strategy="hierarchical",
    ))
    assert d.selected_strategy == "dense"
    assert "R06" in d.matched_rules


def test_rule_r06_api_documentation_single_hop(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        category="API Documentation",
        planner_oracle_strategy="hierarchical",
    ))
    assert d.selected_strategy == "dense"
    assert "R06" in d.matched_rules


def test_rule_r06_does_not_fire_for_non_single_hop(planner):
    """R06 requires single_hop; multi_hop SysDesign should be caught by R03."""
    d = planner.plan(_query(
        reasoning_type="multi_hop",
        category="System Design Documents",
        retrieval_difficulty_factors=["multi_document_dependency"],
        planner_oracle_strategy="multi_hop",
    ))
    assert d.matched_rules[0] == "R03"


def test_rule_r07_default_hybrid(planner):
    d = planner.plan(_query(
        reasoning_type="single_hop",
        category="HR Policies",
        retrieval_difficulty_factors=[],
        planner_oracle_strategy="keyword",
    ))
    assert d.selected_strategy == "hybrid"
    assert "R07" in d.matched_rules


# ── Confidence and allowed values ─────────────────────────────────────────────

def test_selected_strategy_always_allowed(planner):
    test_cases = [
        _query(reasoning_type="temporal"),
        _query(reasoning_type="exception"),
        _query(reasoning_type="multi_hop"),
        _query(retrieval_difficulty_factors=["table_dependency"]),
        _query(reasoning_type="comparison"),
        _query(reasoning_type="aggregation"),
        _query(reasoning_type="single_hop", category="System Design Documents"),
        _query(reasoning_type="single_hop", category="Travel Policies"),
    ]
    for q in test_cases:
        d = planner.plan(q)
        assert d.selected_strategy in AVAILABLE_STRATEGIES, (
            f"Invalid strategy {d.selected_strategy!r} for query {q}"
        )


def test_confidence_in_range(planner):
    queries = [_query(reasoning_type=rt) for rt in
               ["temporal", "exception", "multi_hop", "comparison", "aggregation", "single_hop"]]
    for q in queries:
        d = planner.plan(q)
        assert 0.0 < d.planner_confidence <= 1.0


def test_matched_rules_non_empty(planner):
    for rt in ["temporal", "exception", "multi_hop", "comparison", "aggregation", "single_hop"]:
        d = planner.plan(_query(reasoning_type=rt))
        assert len(d.matched_rules) >= 1


# ── is_correct and oracle_discrepancy ────────────────────────────────────────

def test_is_correct_true_when_matches_oracle_mapped(planner):
    # exception → dense, oracle=dense → mapped=dense
    d = planner.plan(_query(reasoning_type="exception", planner_oracle_strategy="dense"))
    assert d.is_correct is True


def test_is_correct_false_when_mismatches_oracle_mapped(planner):
    # temporal → bm25, oracle=multi_hop → mapped=hybrid
    d = planner.plan(_query(reasoning_type="temporal", planner_oracle_strategy="multi_hop"))
    assert d.is_correct is False


def test_oracle_discrepancy_flagged_for_temporal(planner):
    d = planner.plan(_query(reasoning_type="temporal", planner_oracle_strategy="multi_hop"))
    assert d.oracle_discrepancy is True
    assert len(d.discrepancy_note) > 0


def test_oracle_discrepancy_false_for_non_temporal(planner):
    d = planner.plan(_query(reasoning_type="single_hop", planner_oracle_strategy="hybrid"))
    assert d.oracle_discrepancy is False


# ── Batch planning ────────────────────────────────────────────────────────────

def test_plan_batch_matches_individual(planner):
    queries = [_query(query_id=f"Q-{i:03d}", reasoning_type=rt)
               for i, rt in enumerate(["temporal", "exception", "multi_hop",
                                        "comparison", "single_hop"])]
    batch, _ = planner.plan_batch(queries)
    for q, b in zip(queries, batch):
        ind = planner.plan(q)
        assert b.selected_strategy == ind.selected_strategy
        assert b.matched_rules == ind.matched_rules


def test_plan_batch_returns_all_queries(planner):
    queries = [_query(query_id=f"Q-{i:03d}") for i in range(20)]
    batch, elapsed = planner.plan_batch(queries)
    assert len(batch) == 20
    assert elapsed >= 0.0


def test_planner_is_deterministic(planner):
    q = _query(reasoning_type="exception", retrieval_difficulty_factors=["exception_handling"])
    d1 = planner.plan(q)
    d2 = planner.plan(q)
    assert d1.selected_strategy == d2.selected_strategy
    assert d1.matched_rules == d2.matched_rules
    assert d1.planner_confidence == d2.planner_confidence


# ── Planner metrics ───────────────────────────────────────────────────────────

def _make_decisions(records: list[tuple[str, str, str]]) -> list[PlannerDecision]:
    """(query_id, selected, oracle_extended) → PlannerDecision list."""
    decisions = []
    for i, (qid, selected, oracle_ext) in enumerate(records):
        oracle_mapped = ORACLE_STRATEGY_MAP.get(oracle_ext, "hybrid")
        decisions.append(PlannerDecision(
            query_id=qid,
            selected_strategy=selected,
            matched_rules=[f"R0{i % 7 + 1}"],
            planner_confidence=0.75,
            oracle_strategy=oracle_ext,
            oracle_strategy_mapped=oracle_mapped,
            oracle_strategy_source="heuristic_assignment",
            is_correct=(selected == oracle_mapped),
        ))
    return decisions


def test_planner_metrics_accuracy():
    decisions = _make_decisions([
        ("Q-001", "dense", "dense"),    # correct
        ("Q-002", "bm25", "keyword"),   # correct
        ("Q-003", "hybrid", "table_aware"),  # correct
        ("Q-004", "dense", "hybrid"),   # wrong
        ("Q-005", "bm25", "multi_hop"), # wrong
    ])
    m = compute_planner_metrics(decisions)
    assert m.n_queries == 5
    assert m.n_correct == 3
    assert abs(m.accuracy - 0.6) < 1e-6


def test_planner_metrics_precision_recall():
    # 2 correct dense, 1 wrong (predicted dense but oracle hybrid)
    # 1 correct bm25, 0 hybrid
    decisions = _make_decisions([
        ("Q-001", "dense", "dense"),
        ("Q-002", "dense", "dense"),
        ("Q-003", "dense", "hybrid"),   # FP for dense, FN for hybrid
        ("Q-004", "bm25", "keyword"),
    ])
    m = compute_planner_metrics(decisions)
    # Dense precision: 2/3 ≈ 0.667
    assert abs(m.precision_by_strategy["dense"] - 2/3) < 1e-4
    # Dense recall: 2/2 = 1.0
    assert abs(m.recall_by_strategy["dense"] - 1.0) < 1e-4


def test_planner_metrics_empty():
    m = compute_planner_metrics([])
    assert m.n_queries == 0
    assert m.accuracy == 0.0


def test_strategy_distribution_report():
    decisions = _make_decisions([
        ("Q-001", "bm25", "keyword"),
        ("Q-002", "bm25", "keyword"),
        ("Q-003", "dense", "dense"),
        ("Q-004", "hybrid", "hybrid"),
        ("Q-005", "hybrid", "table_aware"),
    ])
    dist = strategy_distribution_report(decisions)
    counts = {r["selected_strategy"]: r["count"] for r in dist}
    assert counts["bm25"] == 2
    assert counts["dense"] == 1
    assert counts["hybrid"] == 2


def test_rule_contributions_usage():
    decisions = _make_decisions([
        ("Q-001", "bm25", "multi_hop"),   # R01 wins (discrepancy)
        ("Q-002", "dense", "dense"),       # R02 wins
        ("Q-003", "hybrid", "multi_hop"),  # R03 wins
    ])
    # manually override matched_rules to simulate planner output
    decisions[0].matched_rules = ["R01"]
    decisions[1].matched_rules = ["R02"]
    decisions[2].matched_rules = ["R03"]
    contribs = compute_rule_contributions(decisions)
    by_id = {rc.rule_id: rc for rc in contribs}
    assert by_id["R01"].usage_count == 1
    assert by_id["R02"].usage_count == 1
    assert by_id["R03"].usage_count == 1


def test_group_decisions_by_reasoning_type():
    decisions = _make_decisions([
        ("Q-001", "bm25", "keyword"),
        ("Q-002", "dense", "dense"),
        ("Q-003", "dense", "dense"),
    ])
    decisions[0].reasoning_type = "single_hop"
    decisions[1].reasoning_type = "exception"
    decisions[2].reasoning_type = "exception"
    groups = group_decisions_by(decisions, "reasoning_type")
    assert "exception" in groups
    assert groups["exception"].n_queries == 2


# ── Rule describe ─────────────────────────────────────────────────────────────

def test_describe_rules_has_all_rules(planner):
    rules = planner.describe_rules()
    ids = [r["rule_id"] for r in rules]
    assert set(ids) == {"R01", "R02", "R03", "R04", "R05", "R06", "R07"}


def test_describe_rules_strategies_are_allowed(planner):
    for r in planner.describe_rules():
        assert r["strategy"] in AVAILABLE_STRATEGIES


# ── End-to-end planner run assembly ──────────────────────────────────────────

def _make_retrieval_results(method: str, query_ids: list[str]) -> dict[str, list[dict]]:
    return {
        qid: [
            {"query_id": qid, "chunk_id": f"{method}-{qid}-C{i:03d}",
             "document_id": f"DOC-{i:03d}", "rank": i, "score": 1.0/i,
             "category": "HR Policies", "section_path": []}
            for i in range(1, 6)
        ]
        for qid in query_ids
    }


def test_assemble_planner_run_correct_strategy():
    """Verify that the assembled run uses the chunk from the selected method."""
    from scripts.run_rule_planner import _assemble_planner_run

    planner = RuleBasedPlanner()
    queries = [
        _query(query_id="Q-001", reasoning_type="temporal"),    # bm25
        _query(query_id="Q-002", reasoning_type="exception"),   # dense
        _query(query_id="Q-003", reasoning_type="multi_hop"),   # hybrid
    ]
    decisions, _ = planner.plan_batch(queries)

    qids = ["Q-001", "Q-002", "Q-003"]
    retrieval = {
        "bm25":   _make_retrieval_results("bm25", qids),
        "dense":  _make_retrieval_results("dense", qids),
        "hybrid": _make_retrieval_results("hybrid", qids),
    }

    run = _assemble_planner_run(decisions, queries, retrieval, k=5)

    by_q = {}
    for r in run.results:
        by_q.setdefault(r.query_id, []).append(r.chunk_id)

    # Q-001 temporal → bm25 → chunk IDs contain "bm25"
    assert all("bm25" in cid for cid in by_q["Q-001"])
    # Q-002 exception → dense → chunk IDs contain "dense"
    assert all("dense" in cid for cid in by_q["Q-002"])
    # Q-003 multi_hop → hybrid → chunk IDs contain "hybrid"
    assert all("hybrid" in cid for cid in by_q["Q-003"])


def test_assemble_planner_run_at_most_k():
    from scripts.run_rule_planner import _assemble_planner_run

    planner = RuleBasedPlanner()
    queries = [_query(query_id="Q-001", reasoning_type="single_hop")]
    decisions, _ = planner.plan_batch(queries)
    retrieval = {"hybrid": _make_retrieval_results("hybrid", ["Q-001"])}
    retrieval["bm25"] = _make_retrieval_results("bm25", ["Q-001"])
    retrieval["dense"] = _make_retrieval_results("dense", ["Q-001"])

    run = _assemble_planner_run(decisions, queries, retrieval, k=3)
    assert len(run.results) <= 3
