# Chapter: Results

## 4. Experimental Results

This chapter presents the results of all retrieval and planning experiments conducted on the SEKD corpus. All results are reported on 380 answerable queries unless otherwise noted. Metrics are computed at k ∈ {1, 3, 5, 10}. Primary comparison metrics are Recall@10 (coverage), MRR (ranking quality of first relevant result), and nDCG@10 (graded ranking quality).

---

## 4.1 Retrieval Baseline Results

### 4.1.1 BM25 (Sparse Retrieval)

BM25 with parameters k₁=1.5, b=0.75 serves as the sparse retrieval baseline. Table 4.1 reports overall performance.

**Table 4.1: BM25 Retrieval Performance**

| k | Recall | Precision | MRR | nDCG | Hit Rate |
|---|--------|-----------|-----|------|----------|
| 1 | 0.2982 | 0.3184 | 0.4573 | 0.3184 | 0.3184 |
| 3 | 0.5096 | 0.1868 | 0.4573 | 0.4295 | 0.5526 |
| 5 | 0.6022 | 0.1368 | 0.4573 | 0.4705 | 0.6526 |
| **10** | **0.7127** | 0.0834 | **0.4573** | **0.5089** | **0.7684** |

BM25 achieves Recall@10=0.713 and MRR=0.457, establishing the keyword-matching baseline. Performance is strong for document-version-conflict queries (Recall@1=0.583) and table-dependent queries (Recall@10=0.887), where exact keyword matching is effective. BM25 performs poorly on exception queries (Recall@10=0.412) and achieves zero Recall@10 on temporal queries (n=5, all requiring date sequence reasoning beyond keyword co-occurrence).

### 4.1.2 Dense Retrieval (BGE-small-en-v1.5)

Dense retrieval using BAAI/bge-small-en-v1.5 (384 dimensions, normalized) with cosine similarity improves upon BM25 on most metrics.

**Table 4.2: Dense Retrieval Performance**

| k | Recall | Precision | MRR | nDCG | Hit Rate |
|---|--------|-----------|-----|------|----------|
| 1 | 0.3456 | 0.3763 | 0.5014 | 0.3763 | 0.3763 |
| 3 | 0.5399 | 0.2018 | 0.5014 | 0.4712 | 0.5684 |
| 5 | 0.6294 | 0.1421 | 0.5014 | 0.5097 | 0.6658 |
| **10** | **0.7404** | 0.0850 | **0.5014** | **0.5479** | **0.7868** |

Dense retrieval achieves Recall@10=0.740 (+2.8 pp over BM25) and MRR=0.501 (+4.4 pp). The MRR improvement is substantial, indicating that semantic embeddings place relevant documents higher in the ranking. Dense retrieval particularly excels on comparison queries (MRR=0.730 vs. BM25's 0.618) and achieves perfect Recall@10 on API Documentation and Travel Policies categories. However, dense retrieval achieves **zero recall** on all five temporal queries, reflecting the well-documented failure of static dense embeddings to encode temporal ordering signals.

### 4.1.3 Hybrid Retrieval (RRF, rrf_k=10)

Hybrid Reciprocal Rank Fusion combines BM25 and Dense ranked lists using the RRF formula with rrf_k=10.

**Table 4.3: Hybrid Retrieval Performance**

| k | Recall | Precision | MRR | nDCG | Hit Rate |
|---|--------|-----------|-----|------|----------|
| 1 | 0.3680 | 0.4053 | 0.5295 | 0.4053 | 0.4053 |
| 3 | 0.5689 | 0.2132 | 0.5295 | 0.5006 | 0.6132 |
| 5 | 0.6228 | 0.1421 | 0.5295 | 0.5247 | 0.6632 |
| **10** | **0.7447** | 0.0863 | **0.5295** | **0.5664** | **0.7974** |

Hybrid achieves the best fixed-strategy performance: Recall@10=0.745, MRR=0.530, nDCG@10=0.566. Improvements over BM25 alone are +15.8% MRR and +23.4% Recall@1. Improvements over Dense alone are +5.6% MRR. Hybrid underperforms Dense on Recall@5 (0.623 vs. 0.629) but recovers at Recall@10, reflecting complementary coverage at higher k values.

### 4.1.4 Cross-Baseline Comparison by Reasoning Type

Table 4.4 summarizes Recall@10 by reasoning type across all baselines.

**Table 4.4: Recall@10 by Reasoning Type (Fixed Baselines)**

| Reasoning Type | n | BM25 | Dense | Hybrid |
|---|---|---|---|---|
| aggregation | 35 | 0.924 | 0.895 | **0.914** |
| comparison | 33 | 0.818 | **0.848** | 0.848 |
| exception | 34 | 0.412 | 0.382 | **0.412** |
| multi_hop | 53 | **0.538** | 0.491 | 0.519 |
| single_hop | 220 | 0.764 | **0.832** | 0.823 |
| temporal | 5 | **0.200** | 0.000 | 0.100 |

Key observations: (1) Dense is superior for single_hop queries (+8.9% over BM25); (2) BM25 is essential for temporal queries (Dense achieves 0.000); (3) exception handling is uniformly poor across all methods; (4) hybrid provides robust performance without catastrophic failure on any reasoning type.

---

## 4.2 Rule-Based Planner Results

### 4.2.1 Strategy Selection Accuracy

The seven-rule planner achieves SSA=78.4% (298/380 correct) with no oracle discrepancies.

**Table 4.5: Rule Planner SSA by Reasoning Type**

| Reasoning Type | n | SSA | Distribution |
|---|---|---|---|
| aggregation | 35 | **1.000** | hybrid (35) |
| comparison | 33 | 0.879 | hybrid (29), dense (4) |
| exception | 34 | **1.000** | dense (34) |
| multi_hop | 53 | **1.000** | hybrid (53) |
| single_hop | 220 | 0.645 | dense (70), hybrid (150) |
| temporal | 5 | **1.000** | bm25 (5) |

Four of six reasoning types achieve perfect SSA. Single_hop is the main weakness (64.5%), where the distinction between hybrid and dense for sub-category queries is ambiguous.

### 4.2.2 Retrieval Performance

**Table 4.6: Rule Planner Retrieval Performance**

| k | Recall | Precision | MRR | nDCG | Hit Rate |
|---|--------|-----------|-----|------|----------|
| 1 | 0.3575 | 0.3947 | 0.5220 | 0.3947 | 0.3947 |
| 3 | 0.5491 | 0.2053 | 0.5220 | 0.4844 | 0.5921 |
| 5 | 0.6294 | 0.1426 | 0.5220 | 0.5199 | 0.6684 |
| **10** | **0.7553** | 0.0874 | **0.5220** | **0.5632** | **0.8079** |

The rule planner achieves the highest Recall@10 (0.755) and Hit@10 (0.808) of all systems. It improves upon Hybrid by +1.4% Recall@10. However, MRR (0.522) and nDCG@10 (0.563) are slightly below Hybrid (0.530, 0.566), indicating that while the rule planner achieves better coverage, it does not improve ranking quality relative to Hybrid.

---

## 4.3 GPT-5 LLM Planner Results

### 4.3.1 API and Parsing Reliability

GPT-5 (structured prompt variant) was evaluated via real OpenAI API calls on all 380 answerable queries. All 380 calls returned valid JSON (parse failure rate: 0.0%), following remediation of the reasoning-token exhaustion issue described in Section 3.

### 4.3.2 Strategy Selection Accuracy

**Table 4.7: GPT-5 Planner SSA by Reasoning Type**

| Reasoning Type | n | SSA | Distribution |
|---|---|---|---|
| aggregation | 35 | **1.000** | hybrid (35) |
| comparison | 33 | **1.000** | hybrid (33) |
| exception | 34 | **1.000** | dense (34) |
| multi_hop | 53 | **1.000** | hybrid (53) |
| single_hop | 220 | 0.636 | bm25 (49), hybrid (117), dense (54) |
| temporal | 5 | **0.000** | hybrid (5) — all incorrect |

GPT-5 achieves SSA=77.6% (295/380 correct). Four reasoning types yield perfect SSA. Single_hop accuracy (63.6%) is similar to the rule planner (64.5%). The critical divergence is temporal: GPT-5 predicts hybrid for all 5 temporal queries, achieving 0% SSA, whereas the rule planner achieves 100% via its dedicated temporal rule.

**Table 4.8: GPT-5 Strategy Distribution vs. Oracle**

| Strategy | Oracle | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| bm25 | 62 | 49 | 0.4694 | 0.3710 | 0.4144 |
| dense | 69 | 88 | 0.6364 | 0.8116 | 0.7134 |
| hybrid | 249 | 243 | **0.8889** | **0.8675** | **0.8780** |

GPT-5 over-predicts dense (+19 over oracle) and under-predicts bm25 (−13 under oracle), reflecting systematic category confusion on API documentation and system design queries.

### 4.3.3 Retrieval Performance

**Table 4.9: GPT-5 Planner Retrieval Performance**

| k | Recall | Precision | MRR | nDCG | Hit Rate |
|---|--------|-----------|-----|------|----------|
| 1 | 0.3680 | 0.4053 | **0.5325** | 0.4053 | 0.4053 |
| 3 | 0.5662 | 0.2123 | **0.5325** | 0.4993 | 0.6105 |
| 5 | 0.6491 | 0.1474 | **0.5325** | 0.5356 | 0.6895 |
| **10** | 0.7500 | 0.0868 | **0.5325** | **0.5702** | 0.8026 |

GPT-5 achieves MRR=0.533 and nDCG@10=0.570 — the highest of all systems — while matching or slightly trailing the rule planner on Recall@10 (0.750 vs. 0.755).

### 4.3.4 Cost and Latency

| Metric | Value |
|---|---|
| Total cost (380 queries) | **$6.18** |
| Cost per query | $0.0163 |
| Average latency | 5,649 ms |
| Median latency | 5,044 ms |
| p95 latency | 10,368 ms |
| Total prompt tokens | 135,081 |
| Total completion tokens | 120,843 |

---

## 4.4 Full System Comparison

Table 4.10 provides the definitive cross-system comparison at k=10.

**Table 4.10: Complete System Comparison (n=380, k=10)**

| System | Recall@10 | MRR | nDCG@10 | Hit@10 | SSA | Cost |
|---|---|---|---|---|---|---|
| BM25 | 0.713 | 0.457 | 0.509 | 0.768 | — | $0 |
| Dense (BGE) | 0.740 | 0.501 | 0.548 | 0.787 | — | $0 |
| Hybrid (RRF) | 0.745 | 0.530 | 0.566 | 0.797 | — | $0 |
| Rule Planner | **0.755** | 0.522 | 0.563 | **0.808** | 78.4% | $0 |
| GPT-5 Planner | 0.750 | **0.533** | **0.570** | 0.803 | 77.6% | $6.18 |

The Rule Planner and GPT-5 Planner represent a performance tradeoff: the Rule Planner maximizes recall coverage at zero cost, while GPT-5 maximizes ranking quality (MRR, nDCG) at substantial computational cost. Both outperform all fixed-strategy baselines on their respective primary metrics.

---

## 4.5 Category-Level Results

**Table 4.11: Recall@10 by Category (Best System Bolded)**

| Category | n | BM25 | Dense | Hybrid | Rule | GPT-5 |
|---|---|---|---|---|---|---|
| API Documentation | 68 | 0.971 | **1.000** | **1.000** | **1.000** | **1.000** |
| HR Policies | 54 | **0.602** | 0.611 | 0.565 | 0.546 | 0.546 |
| Meeting Notes | 100 | 0.608 | 0.528 | 0.615 | 0.605 | 0.605 |
| Security Policies | 64 | 0.617 | **0.734** | 0.727 | 0.711 | 0.711 |
| System Design | 45 | 0.511 | 0.700 | 0.611 | **0.722** | **0.722** |
| Travel Policies | 49 | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

Ceiling effects (Recall@10=1.000) occur for API Documentation and Travel Policies across multiple systems, limiting discrimination at k=10 for these categories.
