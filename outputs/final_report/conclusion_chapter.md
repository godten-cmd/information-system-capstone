# Chapter: Conclusion

## Adaptive Retrieval Planning for Enterprise Knowledge Management: A Controlled Evaluation Study

---

## 6.1 Summary of Work

This thesis investigated adaptive retrieval planning as a mechanism for improving information retrieval in enterprise knowledge management systems. The study addressed seven research questions spanning retrieval architecture design, adaptive planning, planner comparison, and LLM-based planning using GPT-5. All experiments were conducted on the Synthetic Enterprise Knowledge Dataset (SEKD), a fully annotated, 120-document, 1,206-chunk corpus with 405 queries spanning six enterprise document categories and six reasoning types.

The experimental program progressed through nine phases, each building on the previous:

- **Phase 5 (BM25)**: Established the sparse retrieval baseline with Recall@10=0.713 and MRR=0.457, revealing persistent weaknesses on exception handling and temporal queries.
- **Phase 6 (Dense)**: Demonstrated that dense retrieval with BGE-small-en-v1.5 improves MRR by +9.6% to 0.501, with particular gains on semantic reasoning tasks (comparison, single_hop).
- **Phase 7 (Hybrid)**: Established Hybrid RRF as the superior fixed-strategy baseline with Recall@10=0.745 and MRR=0.530, achieving consistent improvements across most reasoning types and categories.
- **Phase 8 (Rule Planner)**: Demonstrated that a seven-rule deterministic planner achieves SSA=78.4% and Recall@10=0.755, outperforming Hybrid on coverage metrics at zero operational cost.
- **Phase 9 (LLM Planner)**: Evaluated GPT-5 as a zero-shot retrieval strategy planner, achieving SSA=77.6%, MRR=0.533, and nDCG@10=0.570 — marginally better than the rule planner on ranking quality, marginally worse on coverage, at a cost of $6.18 for 380 queries.

---

## 6.2 Research Question Answers

**RQ1: Does hybrid retrieval outperform single-method baselines?**
Yes. Hybrid RRF achieves Recall@10=0.745, MRR=0.530, and nDCG@10=0.566, outperforming both BM25 (Recall@10=0.713, MRR=0.457) and Dense (Recall@10=0.740, MRR=0.501) on the primary evaluation metrics. The MRR improvement of +15.8% over BM25 and +5.6% over Dense confirms that lexical and semantic signals are complementary in enterprise knowledge retrieval.

**RQ2: Which retrieval method performs best across query categories?**
No single method uniformly dominates all categories. BM25 achieves perfect Recall@10 on Travel Policies (1.000). Dense achieves perfect Recall@10 on API Documentation and Travel Policies (1.000). Hybrid provides the best balanced performance across all six categories. System Design Documents and HR Policies are the weakest categories for all methods (Recall@10 < 0.640 for most systems).

**RQ3: How does performance vary across reasoning types?**
Performance varies substantially by reasoning type. Aggregation and comparison queries achieve Recall@10 > 0.818 for all systems. Single_hop queries perform moderately well (Recall@10 = 0.764–0.836). Multi_hop queries are challenging (0.491–0.538). Exception and temporal queries are universally difficult (Recall@10 ≤ 0.412 and ≤ 0.200 respectively), with Dense achieving zero recall on temporal queries.

**RQ4: Can a rule-based planner outperform fixed-strategy retrieval?**
Yes, but marginally. The rule planner achieves Recall@10=0.755 (+1.4% over Hybrid) and Hit@10=0.808 (+1.3% over Hybrid). It performs comparably on most ranking metrics and offers a net improvement in coverage at zero marginal cost, validating adaptive planning as a practical enhancement for enterprise RAG systems.

**RQ5: What is the optimal prompt strategy for LLM-based planning?**
Among the three prompt variants evaluated with the mock provider (zero_shot, structured, few_shot), the structured prompt achieves the highest SSA (82.1%) at approximately half the token cost of few_shot. The structured prompt provides reasoning type, difficulty factors, and document category as explicit metadata, enabling the model to apply strategy selection guidance without relying on in-context examples.

**RQ6: Does GPT-5 outperform the rule planner on Strategy Selection Accuracy?**
No. GPT-5 achieves SSA=77.6%, marginally below the rule planner's 78.4% (−0.79 pp). GPT-5 outperforms the rule planner on ranking quality metrics (MRR: +2.0%, nDCG@10: +1.2%, Recall@1: +2.9%) but underperforms on raw coverage (Recall@10: −0.7%). The performance gap is within the range expected from oracle label ambiguity (44.7% of GPT-5 errors are classified as oracle_disagreement).

**RQ7: What are the primary failure modes of LLM-based planning?**
GPT-5's dominant failure modes are (1) category confusion on API Documentation and System Design queries (55.3% of errors), where the distinction between bm25, dense, and hybrid is ambiguous, and (2) temporal query failure (0% SSA, systematic preference for hybrid over bm25). Single_hop queries account for 94% of all GPT-5 planning errors, revealing that GPT-5's natural language reasoning provides little advantage over deterministic rules for well-structured, metadata-rich queries.

---

## 6.3 Principal Findings

Three principal findings emerge from the complete experimental program:

**Finding 1: Retrieval architecture is the primary driver of performance.** The improvement from BM25 to Hybrid RRF (+15.8% MRR) dwarfs the improvement from Hybrid to the best planner (+2.0% MRR). Enterprise RAG practitioners should prioritize retrieval architecture selection over planning intelligence as the first-order design decision.

**Finding 2: Deterministic planning is competitive with LLM planning at orders-of-magnitude lower cost.** A seven-rule planner and GPT-5 are functionally equivalent on end-to-end retrieval quality (Recall@10 within 0.7 pp, MRR within 1.1 pp) despite a 5.6 million× latency difference and $6.18 cost differential per 380 queries. For enterprise deployments with throughput requirements, the rule planner is the superior choice.

**Finding 3: Exception handling and temporal reasoning are universal failure modes.** Every system evaluated in this study fails systematically on exception clause queries (Recall@10 ≤ 0.412) and temporal reasoning queries (Recall@10 ≤ 0.200). These failure modes are not addressable through retrieval strategy selection alone and represent the primary remaining challenge for enterprise RAG on structured policy documents.

---

## 6.4 Conclusion

This study demonstrates that adaptive retrieval planning improves enterprise information retrieval beyond fixed-strategy baselines, but that the choice of planning mechanism is secondary to the choice of retrieval architecture. Hybrid Reciprocal Rank Fusion is the recommended retrieval foundation, and a lightweight rule-based planner provides the best cost-performance trade-off for production enterprise RAG deployment.

The finding that GPT-5 neither clearly outperforms nor clearly underperforms a seven-rule deterministic planner is not a negative result — it is an informative characterization of the current frontier. It suggests that the information available in the structured metadata (reasoning type, difficulty factors, document category) is sufficient to determine the optimal retrieval strategy for the majority of enterprise queries, and that LLM inference adds marginal value for this specific task in its current form.

Future work should address the exception handling and temporal reasoning failure modes, validate findings on real enterprise corpora, and explore fine-tuned planner models that combine the cost efficiency of the rule planner with the calibration quality advantages observed in GPT-5 planning.
