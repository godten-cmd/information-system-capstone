# Final Experiment Summary

## SEKD Adaptive Retrieval Planning Study — Phase 10

**Study Title**: Adaptive Retrieval Planning for Enterprise Knowledge Management: A Controlled Evaluation on the Synthetic Enterprise Knowledge Dataset (SEKD)

**Completed**: 2026-06-03

---

## Corpus Overview

| Property | Value |
|---|---|
| Dataset | SEKD (Synthetic Enterprise Knowledge Dataset) |
| Documents | 120 (6 categories × 20 documents) |
| Chunks | 1,206 (section-aware, with complete metadata) |
| Total Queries | 405 |
| Answerable Queries | 380 |
| Unanswerable Queries | 25 |
| Reasoning Types | 6 (single_hop, multi_hop, aggregation, comparison, exception, temporal) |
| Difficulty Factors | 8 (including none, table_dependency, exception_handling, temporal_reasoning, etc.) |

---

## Experiment Chronology

| Phase | System | Key Output | Status |
|---|---|---|---|
| 2 | Document Generator | 120 synthetic enterprise documents | Complete |
| 3 | Chunker | 1,206 section-aware chunks | Complete |
| 4 | Query Generator | 405 queries + ground truth | Complete |
| 5 | BM25 Baseline | Recall@10=0.713, MRR=0.457 | Complete |
| 6 | Dense Baseline | Recall@10=0.740, MRR=0.501 | Complete |
| 7 | Hybrid Baseline | Recall@10=0.745, MRR=0.530 | Complete |
| 8 | Rule Planner | Recall@10=0.755, SSA=78.4% | Complete |
| 9 | GPT-5 Planner | Recall@10=0.750, SSA=77.6%, MRR=0.533 | Complete |
| 10 | Final Report | All artifacts | Complete |

---

## System Performance Summary

### Primary Metrics (n=380, k=10)

| System | Recall@10 | MRR | nDCG@10 | Hit@10 | Cost |
|--------|-----------|-----|---------|--------|------|
| BM25 | 0.713 | 0.457 | 0.509 | 0.768 | $0 |
| Dense (BGE-small-v1.5) | 0.740 | 0.501 | 0.548 | 0.787 | $0 |
| **Hybrid (RRF k=10)** | 0.745 | **0.530** | 0.566 | 0.797 | $0 |
| **Rule Planner** | **0.755** | 0.522 | 0.563 | **0.808** | $0 |
| **GPT-5 Planner** | 0.750 | **0.533** | **0.570** | 0.803 | **$6.18** |

### System Rankings

| Metric | Rank 1 | Rank 2 | Rank 3 |
|--------|--------|--------|--------|
| Recall@10 | Rule Planner (0.755) | GPT-5 (0.750) | Hybrid (0.745) |
| MRR | GPT-5 (0.533) | Hybrid (0.530) | Rule (0.522) |
| nDCG@10 | GPT-5 (0.570) | Hybrid (0.566) | Rule (0.563) |
| Hit@10 | Rule (0.808) | GPT-5 (0.803) | Hybrid (0.797) |
| SSA | Rule (78.4%) | GPT-5 (77.6%) | — |
| Cost | Rule / BM25 / Dense / Hybrid ($0) | GPT-5 ($6.18) | — |

---

## Key Findings

### Finding 1: Hybrid Dominates Fixed-Strategy Retrieval
Hybrid RRF outperforms BM25 by +15.8% MRR and +23.4% Recall@1. It outperforms Dense by +5.6% MRR. Hybrid is the optimal fixed-strategy baseline.

### Finding 2: Rule Planner Is the Best Coverage System
Seven deterministic rules achieve SSA=78.4% and Recall@10=0.755 — the highest coverage of any system — at zero inference cost and sub-millisecond latency.

### Finding 3: GPT-5 Optimizes Ranking Quality
GPT-5 achieves the highest MRR (0.533) and nDCG@10 (0.570), outperforming the rule planner on ranking quality at the cost of coverage (−0.5% Recall@10) and $6.18 for 380 queries.

### Finding 4: Retrieval Architecture Dominates Planning Gain
BM25→Hybrid improvement (+15.8% MRR) is 8× larger than the best planner improvement (+2.0% MRR). Architecture selection is the primary performance lever.

### Finding 5: Exception and Temporal Queries Are Universal Failures
No system achieves Recall@10 > 0.412 on exception queries. Dense achieves 0.000 on temporal queries. These failure modes are structural and require specialized solutions.

---

## GPT-5 Evaluation Summary

| Property | Value |
|---|---|
| Provider | OpenAI (Chat Completions API) |
| Model | gpt-5 |
| Prompt Version | structured |
| Queries Evaluated | 380 |
| Parse Failure Rate | **0.0%** (0/380) |
| API Success Rate | 100% |
| SSA | 77.6% (295/380) |
| Recall@10 | 0.750 |
| MRR | 0.533 |
| nDCG@10 | 0.570 |
| Total Cost | $6.18 |
| Avg Latency | 5,649 ms |
| Wall Clock | 35.8 minutes |
| Cache Location | outputs/cache/planner_gpt5_structured_full/ |
| Key Fix Applied | Reasoning token exhaustion: max_completion_tokens increased to 2000 for gpt-5 |

---

## Reasoning-Type Performance Summary

| Type | n | Best Recall@10 | System | Worst (Dense@10) | Note |
|---|---|---|---|---|---|
| aggregation | 35 | 0.924 | BM25 | 0.895 | All methods ≥ 0.895 |
| comparison | 33 | 0.879 | Rule | 0.818 | All methods ≥ 0.818 |
| exception | 34 | 0.412 | BM25/Hybrid | 0.382 | Universal weakness |
| multi_hop | 53 | 0.538 | BM25 | 0.491 | Cross-doc challenge |
| single_hop | 220 | 0.836 | Rule/GPT-5 | 0.764 | Dense +8.9% over BM25 |
| temporal | 5 | 0.200 | BM25/Rule | **0.000** (Dense) | Dense fails completely |

---

## Research Question Summary

| RQ | Question | Answer |
|---|---|---|
| RQ1 | Does Hybrid outperform single-method baselines? | **Yes** — +15.8% MRR over BM25, +5.6% over Dense |
| RQ2 | Which method is best by category? | **Category-dependent** — Dense best for SysDesign/Security; Hybrid for Meeting Notes |
| RQ3 | How does performance vary by reasoning type? | **50 pp range** — aggregation (best) vs. temporal/exception (worst) |
| RQ4 | Can a rule planner outperform fixed-strategy? | **Yes** — Recall@10 +1.4%, Hit@10 +1.4% over Hybrid at $0 cost |
| RQ5 | Which prompt variant is optimal? | **Structured** — SSA=82.1% (mock), best cost/performance ratio |
| RQ6 | Does GPT-5 outperform the rule planner on SSA? | **No** — GPT-5 SSA=77.6% vs. Rule SSA=78.4%; GPT-5 wins on MRR/nDCG |
| RQ7 | What are GPT-5's primary failure modes? | **Category confusion** (55.3% of errors) and **temporal mis-routing** (0% SSA) |

---

## Output Files Index

### Experiment Outputs
| Path | Description |
|---|---|
| outputs/evaluation/bm25/ | BM25 metrics, per-category, per-reasoning-type |
| outputs/evaluation/dense/ | Dense metrics, comparison vs BM25 |
| outputs/evaluation/hybrid/ | Hybrid metrics, RRF sweep, contribution analysis |
| outputs/evaluation/rule_planner/ | Rule planner decisions, SSA, retrieval metrics |
| outputs/evaluation/llm_planner/ | GPT-5 decisions, SSA, retrieval metrics, cost |
| outputs/cache/planner_gpt5_structured_full/ | 380 cached GPT-5 decisions (JSON) |

### Phase 10 Final Report
| Path | Description |
|---|---|
| outputs/final_report/experiment_summary.md | This file |
| outputs/final_report/rq_answers.md | Detailed RQ1–RQ7 answers with evidence |
| outputs/final_report/results_chapter.md | Publication-ready results chapter |
| outputs/final_report/discussion_chapter.md | Discussion: why each system performed as it did |
| outputs/final_report/conclusion_chapter.md | Conclusion with principal findings |
| outputs/final_report/research_contributions.md | Theoretical, methodological, practical contributions |
| outputs/final_report/future_work.md | Four directions for extension |
| outputs/final_report/threats_to_validity.md | Internal, external, construct validity threats |
| outputs/final_report/statistical_analysis.md | Delta analysis, effect sizes, failure mode stats |
| outputs/final_report/tables/dataset_statistics.csv | SEKD corpus and query statistics |
| outputs/final_report/tables/retrieval_performance.csv | All systems × all k values |
| outputs/final_report/tables/reasoning_type_analysis.csv | Recall@10 + MRR by reasoning type |
| outputs/final_report/tables/category_analysis.csv | Recall@10 + MRR by category |
| outputs/final_report/tables/planner_comparison.csv | Rule vs. GPT-5 on all metrics |
| outputs/final_report/tables/cost_latency.csv | Cost and latency comparison |
| outputs/final_report/tables/tables.tex | All tables in publication-ready LaTeX |
| outputs/final_report/figures/fig1_retrieval_baselines.{png,svg} | BM25 vs Dense vs Hybrid |
| outputs/final_report/figures/fig2_planner_comparison.{png,svg} | Rule vs. GPT-5 planner |
| outputs/final_report/figures/fig3_reasoning_type_heatmap.{png,svg} | Reasoning-type heatmap |
| outputs/final_report/figures/fig4_category_performance.{png,svg} | Category-level bar chart |
| outputs/final_report/figures/fig5_cost_performance_tradeoff.{png,svg} | Cost vs. performance scatter |
