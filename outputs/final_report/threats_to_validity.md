# Threats to Validity

## SEKD Experiment — Phase 10

---

## 1. Internal Validity

### 1.1 Oracle Label Limitations

The oracle strategy labels used for Strategy Selection Accuracy (SSA) evaluation were assigned through a two-stage process: initial heuristic assignment based on reasoning type and difficulty factors, followed by empirical validation for a subset of edge cases. This process has several limitations.

**Heuristic assignment**: The majority of oracle labels (approximately 87%) were assigned by a deterministic mapping from reasoning type and difficulty factors to retrieval strategy. This mapping encodes the assumptions of the corpus designer rather than empirically optimal strategies derived from retrieval results. For example, all `single_hop` queries without special difficulty factors default to `hybrid`, even though dense or bm25 may perform equally well for specific query subsets.

**Oracle ambiguity**: The failure analysis identified that 38 of 85 GPT-5 planner errors (44.7%) were classified as `oracle_disagreement` — cases where the predicted strategy was plausible but differed from the oracle label. This indicates that oracle labels are not uniquely correct for a substantial fraction of queries, and SSA measurements may underestimate true planner quality.

**Empirical validation scope**: Only a subset of queries received empirical validation (5 temporal queries confirmed as bm25 via retrieval result analysis). The remaining oracle labels were not validated against actual retrieval performance. True oracle labels derived from held-out retrieval experiments would provide a stronger SSA evaluation baseline.

### 1.2 Measurement Bias from Chunk-Level Ground Truth

Ground truth was constructed as sets of required chunk IDs derived from verbatim evidence spans in the source documents. This introduces two measurement biases:

1. **Granularity mismatch**: Section-aware chunking may split evidence across chunk boundaries. Queries with evidence spanning chunk boundaries require retrieval of multiple chunks, artificially inflating the difficulty of Recall@1.

2. **Single-evidence assumption**: Most queries require exactly one primary chunk. Multi-hop queries requiring multiple chunks weight all required chunks equally, potentially masking partial-match scenarios where retrieval of any required chunk provides substantial answer coverage.

### 1.3 Retrieval Metric Ceilings

Several category-system combinations achieve Recall@10 = 1.000 (API Documentation, Travel Policies for multiple systems). Ceiling effects at k=10 prevent discrimination between methods for these categories, and the primary metric differentiation collapses to k<10 measurements for these groups.

---

## 2. External Validity

### 2.1 Synthetic Dataset Limitations

The SEKD corpus was synthetically generated using a controlled document generation pipeline. This introduces the following external validity concerns:

**Language distribution**: Synthetic documents may not capture the full linguistic diversity of real enterprise documents, including informal language, domain jargon, cross-referencing patterns, acronym variation, version-specific terminology, and redaction artifacts common in enterprise corpora.

**Topical coherence**: Real enterprise corpora exhibit complex inter-document dependencies (shared authors, evolving terminology, document lineage, versioning conflicts) that the synthetic SEKD corpus does not replicate. The controlled generation ensures clean ground truth but reduces ecological validity.

**Query naturalness**: The 405 queries were generated with explicit metadata (reasoning type, difficulty factors) that guided oracle assignment. Real user queries are unstructured and may not align cleanly with the six reasoning types defined in this study. The structured query generation may overstate retrieval difficulty in ways that do not correspond to actual user behavior.

**Corpus scale**: At 120 documents and 1,206 chunks, SEKD is a small-scale corpus. Retrieval methods may behave differently at enterprise scale (tens of thousands of documents), where vocabulary mismatch in BM25 and embedding space saturation in dense retrieval become more pronounced.

### 2.2 Domain Coverage Limitations

The six document categories (API Documentation, HR Policies, Project Meeting Notes, Security Policies, System Design Documents, Travel Policies) represent a limited slice of enterprise knowledge management. The following domains are not covered:

- Legal contracts and compliance documents
- Financial reports and earnings documentation
- Customer-facing support articles
- Code documentation and version histories
- Unstructured emails and chat logs

Retrieval strategy behavior may differ substantially for these document types. In particular, legal and financial documents involve domain-specific terminology and cross-reference structures that may favor BM25 more strongly than observed in SEKD.

### 2.3 Embedding Model Scope

Only one embedding model (BAAI/bge-small-en-v1.5, 384 dimensions) was evaluated for dense retrieval. Larger and more recent models (e.g., text-embedding-3-large, E5-large, GTE-Qwen) may produce substantially different rankings and would change the relative performance of dense and hybrid methods. The conclusions about dense retrieval limitations on temporal and exception queries may not generalize to more capable encoders.

---

## 3. Construct Validity

### 3.1 Strategy Selection Accuracy as Proxy

SSA measures whether the planner selects the strategy that matches the oracle label, not whether the planner's decision improves end-to-end retrieval performance. The weak correlation between SSA differences (GPT-5: -0.79 pp vs. Rule) and retrieval metric differences (MRR: +1.05 pp for GPT-5) illustrates this disconnect. A higher SSA does not necessarily translate to better user-facing retrieval quality.

### 3.2 Retrieval-as-Planning Framing

This study frames retrieval strategy selection as a classification problem over three discrete strategies (bm25, dense, hybrid). Real-world retrieval planning may involve continuous trade-offs (e.g., adjusting BM25 parameters per query, varying the RRF k parameter, or selecting a different embedding model). The discrete three-way classification may underestimate the potential gain from finer-grained adaptive retrieval.

---

## 4. Model-Version Limitations

### 4.1 GPT-5 Temporal Validity

The GPT-5 evaluation was conducted using the GPT-5 API as available in June 2026. GPT-5 is an active deployment; model behavior may change across versions. The 0% SSA on temporal queries and the specific error patterns observed (systematic preference for hybrid over bm25) may not persist in future GPT-5 versions.

Additionally, the structured prompt used in this study was optimized heuristically. A more systematic prompt optimization or fine-tuning of a smaller model specifically for retrieval strategy classification may substantially outperform the zero-shot GPT-5 approach.

### 4.2 Reasoning Token Cost

The parsing fix required for GPT-5 (increasing `max_completion_tokens` from 200 to 2000 to accommodate reasoning token overhead) had significant cost implications: the fixed baseline cost of $0 versus the GPT-5 cost of $6.18 for 380 queries. Future reasoning models may require different token budget configurations, affecting the cost-performance tradeoff reported here.

---

## 5. Conclusion on Generalizability

The experimental findings are most directly applicable to:
- Small-to-medium enterprise knowledge bases (< 10,000 documents)
- Well-structured, category-consistent document corpora
- English-language retrieval tasks
- Queries typed by users familiar with the document domain

Caution should be exercised when generalizing findings to multilingual corpora, noisy real-world enterprise data, very large-scale deployments, or user populations with adversarial or highly ambiguous query patterns.
