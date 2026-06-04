# Future Work

## SEKD Adaptive Retrieval Planning — Directions for Extension

---

## 1. Real Enterprise Dataset Validation

The most critical limitation of this study is its reliance on a synthetic corpus. Future work should validate the findings using real enterprise knowledge bases. Several directions are possible:

**Public enterprise corpora**: Emerging benchmarks such as BRIGHT (a dataset of realistic, diverse retrieval tasks), MIRAGE (enterprise-style multi-hop RAG), and datasets derived from corporate wiki exports (Confluence, SharePoint anonymization) provide increasingly realistic test conditions. The evaluation framework developed in this study is designed for easy adaptation to external datasets.

**Proprietary corpus pilot**: A limited pilot with a real enterprise partner — evaluating the same seven-rule planner and hybrid retrieval architecture on an internal knowledge base — would test generalizability without requiring full public disclosure. The planner pipeline is already production-ready (sub-millisecond latency, no external API dependencies for the rule planner).

**Multi-organization comparison**: Enterprise query patterns and document structures vary significantly by industry (legal, finance, healthcare, software). A multi-organization study comparing strategy performance across industry verticals would identify which findings are universal and which are domain-specific.

---

## 2. Multi-Agent Planning

The current planning architecture is a single-turn, single-model decision: one planner selects one strategy per query. Several multi-agent extensions merit investigation:

**Retrieval-then-replan**: A two-stage architecture in which an initial retrieval pass informs a second planning decision. If the initial retrieval returns low-confidence results (measured by score gaps or document diversity), a replan agent selects a different strategy or initiates a targeted re-query.

**Parallel multi-strategy retrieval with fusion**: Rather than selecting one strategy, a planning agent assigns confidence weights to all three strategies and performs weighted fusion of their ranked lists. This avoids the binary winner-take-all decision and may be especially beneficial for ambiguous single-hop queries where oracle disagreement is highest.

**Critic-reflective architecture**: A critic agent evaluates retrieved chunks for relevance before passing them to the generator. If the critic flags low relevance, the planner receives a feedback signal and retries with a different strategy. This closed-loop approach addresses the fundamental limitation that planner quality cannot be assessed without observing retrieval outcomes.

**Debate-based planning**: Two LLM agents argue for different retrieval strategies based on query metadata, with a third agent adjudicating. This approach may reduce category confusion errors (55% of GPT-5 planner errors) by forcing explicit articulation of strategy rationale.

---

## 3. Planner Fine-Tuning

The GPT-5 planner in this study uses zero-shot prompting with a structured prompt. Several fine-tuning approaches could substantially improve performance:

**Supervised fine-tuning on oracle labels**: Using the 380 oracle-labeled queries as training data, a smaller language model (e.g., GPT-4o-mini, Llama-3-8B, or Mistral-7B) could be fine-tuned specifically for retrieval strategy classification. This approach would reduce inference cost from $0.016/query to near-zero while potentially improving SSA above the 78.4% rule planner baseline.

**Retrieval outcome feedback**: Rather than supervising on oracle labels (which are heuristic), fine-tuning on the actual retrieval performance delta between strategies for each query provides stronger supervision. For each query, the optimal strategy is the one that achieves the highest Recall@1 (most discriminative signal), providing a retrieval-outcome-grounded training signal.

**LoRA adapters for domain specialization**: Lightweight LoRA adapters trained on enterprise-specific query-strategy pairs could enable efficient specialization of a general-purpose LLM to a specific enterprise corpus without full fine-tuning. The adapter could be updated incrementally as new document categories are added.

**Prompt optimization via DSPy or APE**: Automated prompt engineering frameworks (DSPy, Automatic Prompt Engineer) could optimize the structured prompt beyond the heuristic design used in this study, potentially improving SSA without model weight updates.

---

## 4. Adaptive Retrieval Learning

The current evaluation uses static retrieval systems evaluated on a fixed query set. Long-term adaptive retrieval learning is a richer problem:

**Online learning from user feedback**: If users provide implicit relevance signals (click-through, dwell time, follow-up query patterns), a planner can update its strategy weights online. Bandit algorithms (UCB, Thompson sampling) are natural candidates for online strategy selection with exploration-exploitation trade-offs.

**Query-adaptive chunking**: The fixed section-aware chunking (1,206 chunks) is query-agnostic. Hypothetical document embeddings (HyDE), late interaction models (ColBERT), or query-specific re-chunking could improve retrieval for complex multi-hop and exception queries where fixed chunks miss evidence across boundaries.

**Embedding model selection**: Rather than selecting among retrieval strategies, a meta-planner could also select among multiple embedding models (small/fast vs. large/slow), adapting latency and cost to query complexity. Queries identified as simple (single_hop, no difficulty factors) could use bge-small; complex multi-hop queries could route to a larger model.

**Continual learning with corpus drift**: Enterprise corpora change over time (new policies, updated API versions, revised meeting notes). A retrieval system that monitors distribution shift in query patterns and proactively retrains or updates the planner represents an important operational challenge for production deployments.

---

## 5. Generative RAG Integration

This study evaluates retrieval in isolation. Future work should close the loop to answer quality:

**End-to-end RAG evaluation**: Integrating a generator (GPT-4o, Claude Sonnet, Llama-3) into the pipeline and evaluating answer quality (faithfulness, correctness, citation coverage) would test whether retrieval metric improvements translate to answer quality improvements. The failure modes identified here (exception queries, temporal queries) may translate differently to answer quality than to retrieval recall.

**Unanswerable query handling**: The 25 unanswerable queries in SEKD were excluded from retrieval evaluation. A full RAG system must detect and handle unanswerable queries gracefully. Future work should evaluate whether planner metadata (confidence, reasoning type) can be used to route unanswerable queries to fallback responses without hallucination.

**Citation-grounded generation**: Enterprise RAG systems require that generated answers cite specific document passages. Retrieval planning that maximizes citation coverage (rather than general Recall@k) may require different optimization objectives and different planner designs.
