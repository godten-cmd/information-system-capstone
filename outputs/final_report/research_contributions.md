# Research Contributions

## SEKD Adaptive Retrieval Planning Study

---

## 1. Theoretical Contributions

### 1.1 Adaptive Retrieval as a Classification Problem

This work formally frames enterprise retrieval strategy selection as a multi-class classification problem over a discrete strategy space {bm25, dense, hybrid}. The framing introduces the concept of **Strategy Selection Accuracy (SSA)** as a planner-level evaluation metric that is orthogonal to end-to-end retrieval metrics. SSA enables direct comparison of planner intelligence independent of the underlying retrieval system quality, providing a new evaluation dimension for adaptive RAG research.

The key theoretical contribution is demonstrating that SSA and end-to-end retrieval performance are not strongly correlated at the planner-versus-planner comparison level: a 0.79 percentage point reduction in SSA (GPT-5 vs. Rule Planner) co-occurs with a 1.05 percentage point improvement in MRR. This finding suggests that planner calibration quality — not just accuracy — determines downstream retrieval outcomes.

### 1.2 Reasoning-Type Taxonomy for Retrieval Planning

The study proposes a six-category reasoning-type taxonomy (single_hop, multi_hop, aggregation, comparison, exception, temporal) grounded in enterprise query behavior. The taxonomy is validated empirically by demonstrating that reasoning type is a strong predictor of optimal retrieval strategy, with perfect SSA achievable for four of six types using deterministic rules. This provides a theoretical foundation for the claim that structured metadata is sufficient — and in some cases superior — to LLM inference for retrieval planning.

### 1.3 RRF as a Dominant Fusion Strategy

The experimental results provide empirical evidence that Reciprocal Rank Fusion with rrf_k=10 achieves the best trade-off among fixed retrieval strategies on a structured enterprise corpus, improving MRR by +15.8% over BM25 and +5.6% over Dense. The finding that hybrid RRF is the correct default strategy for 65.5% of queries (249/380) supports the theoretical argument that lexical and semantic retrieval signals are complementary in enterprise knowledge management, with neither modality dominating across all query types.

---

## 2. Methodological Contributions

### 2.1 SEKD: A Controlled Synthetic Enterprise Knowledge Dataset

This work contributes the Synthetic Enterprise Knowledge Dataset (SEKD), a controlled, fully-annotated benchmark corpus designed specifically for evaluating retrieval planning in enterprise contexts. SEKD provides:

- 120 synthetic enterprise documents across 6 categories
- 1,206 section-aware chunks with complete metadata
- 405 queries with explicit reasoning type, difficulty factors, oracle strategy labels, and verbatim evidence spans
- Ground truth chunk IDs enabling precise Recall@k evaluation
- 25 unanswerable queries for robustness testing

SEKD addresses the scarcity of publicly available, annotated enterprise RAG benchmarks by providing a reproducible, controlled corpus where the ground truth is exact rather than annotator-derived. The synthetic generation pipeline (Jinja2 templates, category-specific generators) enables controlled variation of document characteristics and supports ablation studies that are not possible with real proprietary corpora.

### 2.2 Unified Evaluation Framework

The evaluation framework developed across Phases 5–9 provides a unified pipeline for comparing fixed retrieval methods (BM25, Dense, Hybrid) with adaptive planning methods (Rule, LLM) on the same query set and ground truth. The framework computes Recall@k, Precision@k, MRR, nDCG@k, and Hit@k across multiple values of k, stratified by reasoning type, document category, and difficulty factor.

The framework's modular design — separate retrieval result files, planner decision files, and assembly scripts — enables new retrieval methods and planners to be evaluated without re-running upstream experiments, supporting efficient iteration and fair comparison.

### 2.3 GPT-5 Reasoning-Token Diagnosis and Fix

The diagnosis and remediation of the GPT-5 response parsing failure constitutes a methodological contribution to LLM-based system evaluation. The root cause (GPT-5 consuming the entire `max_completion_tokens=200` budget for internal reasoning, leaving zero tokens for visible output) was identified via structured debugging and validated empirically. The fix — increasing the budget to `max(requested_tokens + 2000, 2000)` for reasoning models — provides a reusable pattern for future evaluations using reasoning-capable LLMs via Chat Completions API.

The debug script (`scripts/debug_gpt5_response.py`) and fix pattern are directly applicable to any system integrating OpenAI reasoning models (o1, o3, o4-mini, gpt-5) with existing completion-based APIs.

---

## 3. Practical Contributions

### 3.1 Deployable Rule-Based Planner

The seven-rule planner achieves SSA=78.4% and Recall@10=0.755 at zero inference cost and sub-millisecond latency. Its rule set is fully interpretable, auditable, and modifiable by domain experts without machine learning expertise. The rules are grounded in empirically validated retrieval performance breakdowns by reasoning type and difficulty factor, making the planner immediately deployable in production enterprise RAG systems.

The planner's decision logic is:
1. Temporal queries → BM25 (date token matching)
2. Exception queries → Dense (semantic clause interpretation)
3. Multi-hop queries → Hybrid (cross-document evidence)
4. Table-dependent queries → Hybrid (exact values + context)
5. Aggregation/comparison queries → Hybrid (broad coverage)
6. Technical single-hop (API, System Design) → Dense
7. Default → Hybrid

This seven-rule system outperforms the best fixed baseline (Hybrid) on Recall@10 and Hit@10, providing a practical improvement with no operational cost.

### 3.2 Cost-Performance Characterization of LLM Planning

The study provides the first empirical cost-performance characterization of GPT-5 as a retrieval strategy planner at scale (380 queries). At $0.016 per query and 5,649 ms average latency, GPT-5 planning imposes a 5.6 million× latency overhead versus the rule planner for a marginal retrieval quality difference of ≤1 pp on most metrics. This characterization provides enterprise architects with concrete data for the build-vs-buy decision in adaptive RAG deployment.

The finding that GPT-5 improves MRR and nDCG relative to the rule planner — while being slightly worse on Recall@10 — suggests that LLM planning may be valuable in latency-tolerant, precision-sensitive applications (e.g., executive search, compliance review) but is likely over-specified for high-throughput, recall-focused enterprise search workloads.

### 3.3 Identification of Universal Retrieval Failure Modes

The cross-system analysis identifies two universal failure modes that persist regardless of retrieval architecture:

1. **Exception clause queries** (Recall@10 ≤ 0.412 across all systems): Exception clauses in enterprise policy documents are insufficiently differentiated from non-exception content in both BM25 token space and dense embedding space. This failure mode warrants specialized retrieval approaches (e.g., clause-level chunking, exception-aware prompting, or re-ranking with clause-detection models).

2. **Temporal reasoning queries** (Dense Recall@10 = 0.000): Dense retrieval completely fails on temporal ordering queries, which depend on date tokens that dense embeddings of 384 dimensions cannot reliably encode. Systems serving temporal enterprise queries must preserve BM25 or exact-match components.

These failure modes provide actionable guidance for enterprise RAG system designers and represent practical knowledge that cannot be derived from theoretical analysis alone.
