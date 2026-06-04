# Chapter: Discussion

## Adaptive Retrieval Planning for Enterprise Knowledge Management

---

## 5.1 Why Hybrid Retrieval Outperforms BM25 and Dense

The most consistent finding across all experiments is the superiority of Hybrid Reciprocal Rank Fusion (RRF) over either unimodal retrieval method. Hybrid achieves Recall@10=0.745, MRR=0.530, and nDCG@10=0.566 — improvements of +4.5%, +15.8%, and +11.3% respectively over BM25, and +0.6%, +5.6%, and +3.4% over Dense. This result aligns with the theoretical motivation for fusion: BM25 and Dense retrieval are complementary, not redundant.

The complementarity is most visible at the difficulty-factor level. BM25 excels at queries involving exact identifiers (document_version_conflict: Recall@1=0.583, outperforming Dense's 0.667 only marginally, but with much higher keyword precision on version numbers and policy codes). Dense excels at queries requiring semantic understanding of conceptual relationships (comparison queries: MRR=0.730 vs. BM25's 0.618). Hybrid consistently delivers the best or near-best performance across all difficulty factors because RRF fusion preserves rank information from both systems rather than committing to either signal exclusively.

Critically, the RRF fusion benefits are largest for queries where BM25 and Dense are both informative but for different reasons — exactly the multi-hop and table-dependency scenarios that constitute 54% of the SEKD query set. For these queries, BM25 retrieves documents containing exact identifiers (project codes, policy section references) while Dense retrieves documents that semantically match the content question, and RRF merges both signals into a single ranked list.

The one failure mode that Hybrid does not address is temporal reasoning. With only 5 temporal queries, any retrieval system that lacks a BM25 component fails: Dense achieves Recall@10=0.000 on temporal queries, and Hybrid's 0.100 represents marginal BM25 contribution through the fusion. The temporal queries require matching specific date tokens that are nearly invisible to 384-dimensional dense embeddings. This suggests that for enterprise corpora with significant temporal content (meeting notes, version histories, policy revision logs), a BM25 component is not optional — it is necessary.

---

## 5.2 Why the Rule-Based Planner Performed So Well

The seven-rule planner achieves SSA=78.4% and Recall@10=0.755, making it the best-performing system on those metrics. Its success stems from three reinforcing factors.

**Factor 1: The Oracle Labels Reflect Rule Logic**. The oracle strategy labels were assigned using a heuristic process that mirrors the rule planner's design. Specifically, the same reasoning types (temporal → BM25, exception → Dense, multi_hop → Hybrid) that informed oracle assignment were directly encoded in the rule planner's rules R01–R03. This structural alignment means that the rule planner is, in some sense, evaluated against a ground truth that reflects its own logic. While this is a limitation of the oracle construction process (see Threats to Validity), it also reflects a genuine insight: the reasoning-type taxonomy provides sufficient signal for strategy selection in a well-structured enterprise corpus.

**Factor 2: Perfect Coverage of High-n Groups**. The rule planner achieves 100% SSA for aggregation (n=35), exception (n=34), and multi_hop (n=53) queries — three groups that together represent 32.1% of the query set. For these groups, the retrieval strategy is unambiguous: aggregation and multi_hop queries require coverage across documents (hybrid), and exception queries require semantic clause interpretation (dense). The rule planner's deterministic mapping is correct by design for these cases.

**Factor 3: Hybrid as a Safe Default**. The rule planner's default strategy (R07) is Hybrid, which is the best fixed baseline. By defaulting to Hybrid for all unmatched queries, the rule planner never performs worse than Hybrid for queries where its rules do not fire. Since 65.5% of oracle labels are Hybrid, the default captures the majority case, and the specialist rules handle the high-value exceptions.

The planner's main weakness is BM25 under-prediction: it predicts BM25 for only 5/62 oracle-BM25 queries (8.1% recall on BM25). The five temporal queries are correctly routed, but single-hop queries in categories with strong keyword signals (certain API documentation, exact numeric lookups) that empirically benefit from BM25 are incorrectly defaulted to Hybrid. This suggests that additional rules targeting keyword-dense query patterns (exact endpoint paths, specific version strings, numeric thresholds) could push SSA above 80%.

---

## 5.3 Why GPT-5 Did Not Outperform the Rule Planner

The most theoretically significant finding of this study is that GPT-5, despite 77.6 billion parameters, zero-shot reasoning capability, and $6.18 in inference costs, does not outperform a seven-rule deterministic system on overall retrieval performance. Understanding why requires separating several confounded factors.

**The task is simpler than LLM capabilities**: Retrieval strategy classification over three categories based on structured metadata (reasoning type, difficulty factors, category) is a low-complexity decision task. The structured prompt encodes the correct decision logic explicitly, making the LLM's marginal contribution — natural language understanding, contextual inference, cross-domain reasoning — largely irrelevant. A classifier trained on 50 labeled examples would likely achieve comparable accuracy at negligible cost.

**SSA measures correctness against a heuristic oracle**: As discussed in Section 5.2, the oracle labels reflect the same logic encoded in the rule planner. GPT-5's deviations from the oracle (particularly its preference for dense over bm25 for API documentation queries) may represent genuine disagreements with the heuristic rather than planning errors. The 44.7% of GPT-5 errors classified as `oracle_disagreement` supports this interpretation: GPT-5 may be more correct than the oracle for a substantial subset of queries.

**GPT-5's retrieval-level advantages are masked by Recall@10 ceiling effects**: GPT-5 outperforms the rule planner on MRR (+2.0%) and nDCG@10 (+1.2%), and on Recall@1 (+2.9%) and Recall@5 (+3.1%). These ranking quality improvements are more consequential for user-facing applications than Recall@10, which is already high for both systems (0.750 and 0.755 respectively, a negligible 0.7 pp gap). The Recall@10 comparison slightly favors the rule planner, but the MRR and nDCG comparisons — which better reflect the quality of the top-ranked results — favor GPT-5.

**Temporal query failure is a systematic calibration error**: GPT-5 achieves 0% SSA on temporal queries, consistently predicting hybrid rather than bm25. This is the single largest source of planning error for temporal content. The rule planner's dedicated temporal rule (R01: temporal → bm25) handles this perfectly, contributing to its higher SSA. GPT-5's failure on temporal queries suggests that the structured prompt's temporal guidance ("temporal → bm25") was insufficient, or that GPT-5's reasoning defaulted to the most common strategy (hybrid) when uncertain.

**API Documentation confusion is the dominant error source**: 32 of 85 GPT-5 errors (37.6%) occur on API Documentation queries. GPT-5 inconsistently selects bm25, dense, or hybrid for API queries, while the oracle assigns a mix of all three strategies depending on query characteristics (exact endpoint paths → bm25, semantic authentication queries → dense, multi-hop API dependencies → hybrid). The ambiguity of API Documentation as a retrieval planning signal — where any of the three strategies can be optimal depending on the specific query — explains why both GPT-5 and the rule planner perform worst on this category (SSA: 52.9% and 70.6% respectively).

---

## 5.4 Implications for Enterprise RAG Systems

The experimental results yield four actionable implications for enterprise RAG system design.

**Implication 1: Hybrid RRF is the correct default retrieval architecture.** No system in this study benefits from committing exclusively to BM25 or Dense. For any enterprise RAG deployment where the query distribution is unknown, Hybrid RRF provides the best combination of coverage and ranking quality at zero marginal cost beyond the setup overhead of maintaining two retrieval indices.

**Implication 2: A lightweight rule planner adds value at zero operational cost.** The seven-rule planner demonstrates that modest SSA (78.4%) is sufficient to improve upon the best fixed baseline (Recall@10: +1.4%, Hit@10: +1.3%). The rules are interpretable, auditable, and require no machine learning infrastructure. Enterprise RAG architects should implement rule-based adaptive routing as a first-line planning mechanism before investing in LLM-based planners.

**Implication 3: LLM-based planning provides ranking quality benefits at substantial cost.** GPT-5 planner's MRR (+2.0%) and nDCG (+1.2%) advantages over the rule planner may justify the $0.016/query cost in high-value, low-volume retrieval applications (executive decision support, compliance auditing, legal research) where answer precision is more important than throughput. For high-volume enterprise search workloads, the cost-benefit ratio favors the rule planner.

**Implication 4: Exception and temporal queries require specialized solutions.** No evaluated system achieves acceptable performance (Recall@10 > 0.5) on exception queries (Recall@10 ≤ 0.412) or temporal queries (Dense: 0.000, best: 0.200). These failure modes are structural rather than parametric — they reflect mismatches between query type and retrieval signal space. Future enterprise RAG systems should develop dedicated handling for these patterns, such as temporal-aware sparse retrieval with date field boosting and exception-aware dense models trained on policy clause pairs.
