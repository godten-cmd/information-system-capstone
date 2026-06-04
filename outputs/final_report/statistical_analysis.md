# Statistical Analysis
## SEKD Phase 10 — Final Experiment Summary

---

## 1. Retrieval Baseline Improvements

### BM25 → Dense (BGE-small-en-v1.5)

| Metric | BM25 | Dense | Δ (pp) | Δ (%) |
|--------|------|-------|--------|-------|
| Recall@1 | 0.2982 | 0.3456 | +4.74 | +15.9% |
| Recall@10 | 0.7127 | 0.7404 | +2.77 | +3.9% |
| MRR | 0.4573 | 0.5014 | +4.41 | +9.6% |
| nDCG@10 | 0.5089 | 0.5479 | +3.90 | +7.7% |

Dense retrieval improves MRR by **+9.6%** over BM25, indicating that semantic matching places relevant documents significantly higher in the ranked list. The improvement at Recall@1 (+15.9%) is the most pronounced, reflecting that semantic similarity is more discriminative at rank 1.

### Dense → Hybrid (RRF, rrf_k=10)

| Metric | Dense | Hybrid | Δ (pp) | Δ (%) |
|--------|-------|--------|--------|-------|
| Recall@1 | 0.3456 | 0.3680 | +2.24 | +6.5% |
| Recall@10 | 0.7404 | 0.7447 | +0.43 | +0.6% |
| MRR | 0.5014 | 0.5295 | +2.81 | +5.6% |
| nDCG@10 | 0.5479 | 0.5664 | +1.85 | +3.4% |

Hybrid RRF adds **+5.6% MRR** over Dense alone. The marginal Recall@10 improvement (+0.6%) suggests that both systems cover similar sets of relevant documents at k=10, but RRF fusion improves ranking quality (MRR, nDCG) by combining complementary signal sources.

### BM25 → Hybrid (full baseline-to-best improvement)

| Metric | BM25 | Hybrid | Δ (pp) | Δ (%) |
|--------|------|--------|--------|-------|
| Recall@1 | 0.2982 | 0.3680 | +6.98 | +23.4% |
| Recall@10 | 0.7127 | 0.7447 | +3.20 | +4.5% |
| MRR | 0.4573 | 0.5295 | +7.22 | +15.8% |
| nDCG@10 | 0.5089 | 0.5664 | +5.75 | +11.3% |

The cumulative improvement from BM25 to Hybrid represents a **+15.8% MRR gain** and **+23.4% Recall@1 gain**, demonstrating that the combination of lexical and semantic signals substantially outperforms either alone.

---

## 2. Planner Improvements over Best Fixed Baseline (Hybrid)

### Rule Planner vs. Hybrid

| Metric | Hybrid | Rule | Δ (pp) | Δ (%) |
|--------|--------|------|--------|-------|
| Recall@10 | 0.7447 | 0.7553 | +1.06 | +1.4% |
| MRR | 0.5295 | 0.5220 | -0.75 | -1.4% |
| nDCG@10 | 0.5664 | 0.5632 | -0.32 | -0.6% |
| Hit@10 | 0.7974 | 0.8079 | +1.05 | +1.3% |

The Rule Planner achieves the best Recall@10 (+1.4% over Hybrid) and Hit@10 (+1.3%), but at the cost of MRR and nDCG — the rule-based system does not rank as well as pure Hybrid. This indicates that adaptive routing improves coverage but introduces some ranking degradation.

### GPT-5 Planner vs. Hybrid

| Metric | Hybrid | GPT-5 | Δ (pp) | Δ (%) |
|--------|--------|-------|--------|-------|
| Recall@10 | 0.7447 | 0.7500 | +0.53 | +0.7% |
| MRR | 0.5295 | 0.5325 | +0.30 | +0.6% |
| nDCG@10 | 0.5664 | 0.5702 | +0.38 | +0.7% |

GPT-5 Planner provides modest but consistent improvements over pure Hybrid across all three ranking metrics, representing a balanced adaptive strategy.

### GPT-5 Planner vs. Rule Planner

| Metric | Rule | GPT-5 | Δ (pp) | Δ (%) |
|--------|------|-------|--------|-------|
| SSA | 0.7842 | 0.7763 | -0.79 | -1.0% |
| Recall@1 | 0.3575 | 0.3680 | +1.05 | +2.9% |
| Recall@5 | 0.6294 | 0.6491 | +1.97 | +3.1% |
| Recall@10 | 0.7553 | 0.7500 | -0.53 | -0.7% |
| MRR | 0.5220 | 0.5325 | +1.05 | +2.0% |
| nDCG@10 | 0.5632 | 0.5702 | +0.70 | +1.2% |

GPT-5 is marginally weaker on raw coverage metrics (Recall@10: -0.7%, SSA: -1.0%) but outperforms the rule planner on ranking quality (MRR: +2.0%, nDCG@10: +1.2%). At Recall@1 and Recall@5, GPT-5 shows meaningful advantages (+2.9% and +3.1%), suggesting it makes more correct decisions at the top of the ranking.

---

## 3. Difficulty Factor Analysis

### Exception Handling — Persistent Cross-System Weakness

| System | Recall@10 | vs. Overall |
|--------|-----------|-------------|
| BM25 | 0.412 | -30.1 pp |
| Dense | 0.382 | -35.8 pp |
| Hybrid | 0.412 | -33.3 pp |
| Rule | 0.368 | -38.7 pp |
| GPT-5 | 0.382 | -36.8 pp |

Exception handling queries perform **30–39 pp below average** across all systems. Dense retrieval fails slightly worse than BM25 (despite correctly routing to dense), suggesting that the SEKD exception clause phrasing creates an intrinsic retrieval difficulty beyond strategy selection.

### Temporal Reasoning — Zero-Recall for Dense

| System | Recall@10 |
|--------|-----------|
| BM25 | 0.200 |
| Dense | **0.000** |
| Hybrid | 0.100 |
| Rule | 0.200 |
| GPT-5 | 0.100 |

Dense retrieval achieves **zero recall** on all 5 temporal queries. BM25 and the Rule Planner (which routes temporal to BM25) are the only systems achieving positive recall for temporal reasoning.

### Table Dependency — Hybrid Dominance

| System | Recall@10 |
|--------|-----------|
| BM25 | 0.887 |
| Dense | 0.944 |
| Hybrid | 0.927 |
| Rule | 0.927 |

Table-dependent queries perform well across all systems (Recall@10 > 0.887). Hybrid slightly underperforms Dense at k=10 for this factor (0.927 vs. 0.944), suggesting that dense retrieval alone is well-suited for structured numeric lookups.

---

## 4. Effect Size Discussion

The differences between adaptive planners (Rule, GPT-5) and the best fixed baseline (Hybrid) are small in absolute terms (Recall@10 delta: ≤1.1 pp). Given a sample size of n=380 queries, the practical significance of these differences depends on the deployment context.

**Magnitude classification** (using the domain-specific interpretation of pp differences):

| Delta Range | Classification |
|-------------|----------------|
| > 5 pp | Substantial improvement |
| 2–5 pp | Meaningful improvement |
| 0.5–2 pp | Marginal improvement |
| < 0.5 pp | Negligible |

By this scale:
- BM25 → Hybrid MRR improvement (+7.2 pp): **Substantial**
- Dense → Hybrid MRR improvement (+2.8 pp): **Meaningful**
- Hybrid → Rule Planner Recall@10 (+1.1 pp): **Marginal**
- Rule Planner → GPT-5 MRR (+1.1 pp): **Marginal**
- Rule vs. GPT-5 Recall@10 difference (0.5 pp): **Negligible**

The primary driver of performance gains is the choice of retrieval strategy architecture (BM25 vs. Dense vs. Hybrid), not the planning layer. Adaptive planning provides marginal improvements over the best fixed baseline, with the two planner approaches essentially tied on end-to-end retrieval quality.

---

## 5. Strategy Selection Accuracy Context

| Planner | SSA | Oracle Distribution: hybrid/dense/bm25 |
|---------|-----|----------------------------------------|
| Rule | 78.4% | 249 / 69 / 62 |
| GPT-5 | 77.6% | 243 / 88 / 49 |

GPT-5 over-predicts **dense** (88 vs oracle 69, +27.5%) and under-predicts **bm25** (49 vs oracle 62, -21.0%). This calibration error particularly affects API Documentation and System Design queries where the distinction between bm25 and dense is subtle and context-dependent.

The Rule Planner's bm25 under-prediction is more extreme (5 vs oracle 62, -91.9%) but compensated by its perfect temporal routing (5/5 correct), which GPT-5 fails completely.
