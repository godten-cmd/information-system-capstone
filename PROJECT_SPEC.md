# Project Specification

## Title

Agentic RAG 기반 기업 지식관리시스템 설계 및 평가: 문서 유형에 따른 Adaptive Retrieval 전략을 중심으로

## Version

1.2

## 1. Research Overview

This project designs and evaluates an enterprise knowledge management system based on Agentic Retrieval-Augmented Generation (Agentic RAG). The central research focus is whether retrieval strategies should adapt to document type, task intent, and information structure rather than relying on a single fixed retrieval pipeline.

Enterprise knowledge is distributed across heterogeneous documents such as policies, manuals, meeting notes, reports, FAQs, technical specifications, contracts, and internal announcements. These document types differ in structure, granularity, authority, update frequency, and expected user query patterns. A conventional RAG system often applies the same chunking, indexing, retrieval, and reranking strategy to all documents, which may reduce answer accuracy, evidence quality, and user trust.

The proposed system introduces an adaptive retrieval layer controlled by an agentic planner. The planner analyzes the user query and document context, selects an appropriate retrieval strategy, invokes retrieval tools, verifies evidence, and generates grounded answers with traceable citations.

## 2. Research Objectives

1. Design an Agentic RAG architecture for enterprise knowledge management.
2. Define document-type-aware adaptive retrieval strategies for heterogeneous enterprise documents.
3. Implement an experimental framework for comparing fixed retrieval and adaptive retrieval.
4. Evaluate answer quality, retrieval quality, citation reliability, and system efficiency.
5. Identify which document types benefit most from adaptive retrieval.
6. Provide practical design guidelines for enterprise RAG systems.
7. Evaluate whether performance improvements justify additional computational cost.

## 3. Research Questions

### RQ1. Retrieval Effectiveness

How does document-type-aware adaptive retrieval affect retrieval accuracy compared with a fixed retrieval strategy?

### RQ2. Answer Quality

Does Agentic RAG with adaptive retrieval improve answer correctness, completeness, and groundedness in enterprise knowledge tasks?

### RQ3. Document Type Sensitivity

Which enterprise document types show the largest performance difference between fixed retrieval and adaptive retrieval?

### RQ4. Agentic Planning

Can an agentic planner select retrieval strategies that are measurably better matched to query intent and document type?

### RQ5. Cost and Latency

What is the tradeoff between improved answer quality and additional latency, token usage, or computation cost introduced by agentic retrieval planning?

### RQ6. Planner Comparison

Can an LLM-based retrieval planner outperform a rule-based retrieval planner in enterprise knowledge management tasks?

### RQ7. Cost Justification

Do the gains achieved by adaptive retrieval planning justify the additional latency and inference cost?

## 4. Hypotheses

### H1. Adaptive Retrieval Improves Retrieval Quality

Document-type-aware adaptive retrieval will achieve higher Recall@K, MRR, and nDCG than a single fixed retrieval pipeline.

### H2. Adaptive Retrieval Improves Answer Groundedness

Answers generated using adaptive retrieval will contain fewer unsupported claims and more accurate citations than answers generated using fixed retrieval.

### H3. Structured and Semi-Structured Documents Benefit More

Document types with strong internal structure, such as policies, manuals, specifications, and contracts, will benefit more from adaptive retrieval than loosely structured documents such as informal meeting notes.

### H4. Agentic RAG Improves Complex Query Handling

For multi-hop, comparison, procedural, and policy-interpretation questions, Agentic RAG will outperform non-agentic RAG in correctness and completeness.

### H5. Adaptive Retrieval Introduces Measurable Overhead

Agentic adaptive retrieval will increase latency and token usage, but the quality improvement will justify the overhead for complex enterprise queries.

### H6. LLM-Based Planner Improves Complex Retrieval

The LLM-based retrieval planner will achieve higher retrieval effectiveness than a rule-based planner for complex and multi-document queries.

## 5. System Architecture

### 5.1 High-Level Architecture

The proposed system consists of six major layers:

1. Data ingestion layer
2. Document understanding and preprocessing layer
3. Indexing and storage layer
4. Agentic retrieval planning layer
5. Answer generation and verification layer
6. Evaluation and monitoring layer

### 5.2 Component Description

#### 5.2.1 Data Ingestion Layer

Responsible for collecting and normalizing enterprise documents.

Supported document sources:

- PDF documents
- Word documents
- Markdown documents
- HTML pages
- CSV or spreadsheet-based knowledge tables
- Internal FAQ files
- Meeting notes
- Policy documents
- Technical manuals

Main responsibilities:

- File loading
- Metadata extraction
- Document type classification
- Version tracking
- Source provenance preservation

#### 5.2.2 Document Understanding and Preprocessing Layer

This layer prepares documents for retrieval while preserving document-specific structure.

Main tasks:

- Text extraction
- Table extraction
- Heading hierarchy parsing
- Section segmentation
- Chunking
- Metadata enrichment
- Document type tagging
- Entity and keyword extraction

Document-type-specific preprocessing examples:

| Document Type | Preprocessing Strategy |
| --- | --- |
| Policy | Section-aware chunking, clause preservation, authority metadata |
| Manual | Step-based segmentation, heading hierarchy preservation |
| FAQ | Question-answer pair indexing |
| Meeting Notes | Date, participant, decision, and action item extraction |
| Report | Abstract, section, table, and figure-aware segmentation |
| Contract | Clause-level segmentation and obligation extraction |
| Technical Spec | API/entity/table-aware chunking |

#### 5.2.3 Indexing and Storage Layer

The system should support multiple retrieval indexes.

Recommended indexes:

- Dense vector index for semantic search
- Sparse keyword index for exact term matching
- Hybrid index combining dense and sparse retrieval
- Metadata index for filtering by document type, date, department, author, or authority level
- Optional graph index for entity and relation traversal

Each chunk should store:

- Chunk ID
- Source document ID
- Document type
- Section path
- Text content
- Metadata
- Embedding vector
- Creation or update timestamp
- Access permission metadata, if applicable

#### 5.2.4 Agentic Retrieval Planning Layer

The agentic planner decides how retrieval should be performed for a given query.

Planner inputs:

- User query
- Query intent
- Expected document type
- Required evidence type
- User role or permission
- Conversation history

Planner outputs:

- Retrieval strategy
- Index selection
- Query rewriting decision
- Metadata filters
- Number of retrieval rounds
- Reranking method
- Verification method

Possible retrieval strategies:

| Strategy | Description | Suitable Cases |
| --- | --- | --- |
| Dense Retrieval | Semantic vector search | Conceptual questions, paraphrased queries |
| Sparse Retrieval | Keyword-based search | Exact terms, codes, names, policy IDs |
| Hybrid Retrieval | Dense plus sparse retrieval | General enterprise queries |
| Metadata-Filtered Retrieval | Retrieval constrained by type, date, department, or authority | Policy, HR, compliance, role-specific questions |
| Hierarchical Retrieval | Retrieve document sections before chunks | Manuals, reports, long policies |
| Multi-Hop Retrieval | Multiple retrieval steps over related evidence | Comparison, cause-effect, dependency queries |
| FAQ Pair Retrieval | Retrieve complete Q-A pairs | Customer support and internal helpdesk knowledge |
| Table-Aware Retrieval | Retrieve rows, columns, or table summaries | Pricing, KPIs, schedules, inventories |

#### Retrieval Strategy Planner

The retrieval strategy planner should support two modes:

##### Rule-Based Planner

The rule-based planner selects retrieval strategies using predefined rules based on query intent, expected document type, query complexity, and metadata constraints. This mode provides deterministic behavior, easier debugging, and stronger experimental control.

Example rule-based decisions:

- Use metadata-filtered hybrid retrieval for policy and compliance questions.
- Use hierarchical retrieval for manuals, reports, and long structured documents.
- Use FAQ pair retrieval for helpdesk-style questions.
- Use table-aware retrieval for KPI, price, schedule, or inventory questions.
- Use multi-hop retrieval for comparison, dependency, or cross-document queries.

##### LLM-Based Planner

The LLM-based planner uses a language model to analyze the query, infer the likely evidence need, select retrieval tools, decide whether query decomposition is necessary, and determine whether retrieved evidence is sufficient. This mode is expected to be more flexible for ambiguous, complex, and multi-document enterprise queries.

The LLM-based planner should produce structured planning outputs, including:

- Predicted query intent
- Relevant document type candidates
- Selected retrieval strategy
- Metadata filters
- Query rewriting or decomposition plan
- Retrieval iteration decision
- Evidence sufficiency judgment

#### 5.2.5 Answer Generation and Verification Layer

This layer generates the final answer using retrieved evidence and verifies citation grounding.

Main responsibilities:

- Evidence compression
- Answer generation
- Citation attachment
- Faithfulness checking
- Conflict detection
- Insufficient-evidence detection
- Final response formatting

The system should avoid unsupported generation. If evidence is insufficient, the answer should explicitly state that the available knowledge base does not contain enough information.

#### 5.2.6 Evaluation and Monitoring Layer

This layer records experimental results and runtime behavior.

Tracked information:

- Query
- Document type
- Selected retrieval strategy
- Retrieved chunks
- Reranked chunks
- Generated answer
- Citation mapping
- Latency
- Token usage
- Evaluation scores
- Error category

## 6. Adaptive Retrieval Design

### 6.1 Query Classification

The system should classify each query by:

- Intent: factual, procedural, comparative, interpretive, summarization, troubleshooting
- Expected document type: policy, manual, FAQ, report, meeting note, contract, specification
- Complexity: single-hop or multi-hop
- Evidence need: exact citation, summarized evidence, table value, decision record
- Risk level: low, medium, high

### 6.2 Document-Type-Aware Strategy Mapping

| Query or Document Condition | Retrieval Strategy |
| --- | --- |
| Policy interpretation | Metadata-filtered hybrid retrieval plus clause reranking |
| Manual procedure question | Hierarchical retrieval plus step-preserving chunks |
| FAQ-like query | FAQ pair retrieval plus semantic matching |
| Meeting decision lookup | Metadata filtering by date/project plus entity matching |
| Report summary | Section-level retrieval plus abstractive evidence compression |
| Contract obligation query | Clause-level retrieval plus exact keyword matching |
| Technical troubleshooting | Hybrid retrieval plus multi-hop retrieval over related specs |
| KPI or table lookup | Table-aware retrieval plus row/column grounding |

### 6.3 Agent Control Loop

The agentic retrieval process should follow this loop:

1. Analyze query intent and constraints.
2. Predict relevant document types.
3. Select retrieval strategy.
4. Rewrite or decompose the query if needed.
5. Retrieve candidate evidence.
6. Rerank candidates.
7. Check evidence sufficiency.
8. Perform additional retrieval if evidence is weak.
9. Generate answer with citations.
10. Verify groundedness and return final answer.

## 7. Dataset Design

### 7.1 Dataset Goals

The dataset should represent realistic enterprise knowledge management scenarios while remaining suitable for controlled academic evaluation.

The dataset should support:

- Document-type comparison
- Retrieval evaluation
- Answer generation evaluation
- Citation grounding evaluation
- Cost and latency analysis

### 7.2 Document Collection

Recommended document categories:

| Category | Example Content | Target Count |
| --- | --- | --- |
| HR Policy | Leave policy, remote work policy, benefits policy | 20-30 docs |
| IT Manual | Account setup, VPN setup, security procedures | 20-30 docs |
| Internal FAQ | HR FAQ, IT FAQ, procurement FAQ | 20-30 docs |
| Meeting Notes | Project meetings, decision logs, sprint reviews | 30-50 docs |
| Business Reports | Quarterly summaries, market analysis, project reports | 20-30 docs |
| Technical Specs | API specs, system design docs, database schema docs | 20-30 docs |
| Contracts or Terms | Vendor terms, service agreements, SLA documents | 10-20 docs |
| Tables | KPI sheets, inventory, pricing, schedules | 10-20 files |

If real enterprise data is unavailable, synthetic documents should be created with controlled facts and known ground truth. Sensitive or confidential real documents should not be used unless properly anonymized and approved.

### 7.3 Query Dataset

The query set should include 300-500 questions.

Query categories:

- Factual lookup
- Procedural question
- Policy interpretation
- Comparative question
- Multi-hop reasoning
- Table lookup
- Decision or meeting history lookup
- Troubleshooting
- Summarization
- Unanswerable question

Each query should include:

- Query ID
- Query text
- Expected document type
- Query category
- Ground-truth answer
- Relevant document IDs
- Relevant chunk IDs or evidence spans
- Difficulty level
- Notes for evaluation

### 7.4 Ground Truth Construction

Ground truth should be built using a combination of:

- Manual annotation
- Controlled synthetic fact generation
- Human verification
- Evidence span labeling

Recommended annotation fields:

| Field | Description |
| --- | --- |
| answer | Reference answer |
| evidence_spans | Exact supporting source spans |
| relevant_docs | Relevant document IDs |
| relevant_chunks | Relevant chunk IDs |
| answerable | Whether the query is answerable from the corpus |
| document_type | Primary expected document type |
| reasoning_type | Single-hop, multi-hop, comparison, interpretation |

## 8. Experiment Plan

### 8.1 Experimental Conditions

The project should compare the following systems:

| System | Description |
| --- | --- |
| Baseline 1: Keyword Search RAG | Sparse retrieval only |
| Baseline 2: Dense RAG | Vector retrieval only |
| Baseline 3: Hybrid RAG | Dense plus sparse retrieval |
| Baseline 4: Fixed RAG with Reranking | Hybrid retrieval plus reranker |
| Baseline 5: Rule-Based Adaptive Retrieval | Rule-selected document-type-aware adaptive retrieval |
| Proposed: LLM-Based Adaptive Retrieval Planner | LLM-selected document-type-aware adaptive retrieval |

### 8.2 Independent Variables

- Retrieval strategy
- Document type
- Query type
- Query complexity
- Chunking method
- Reranking method
- Number of retrieved chunks

### 8.3 Dependent Variables

- Retrieval quality
- Answer correctness
- Answer completeness
- Citation accuracy
- Faithfulness
- Latency
- Token usage
- Cost

### 8.4 Controlled Variables

- Same document corpus
- Same query set
- Same embedding model
- Same generation model
- Same evaluation prompt, if LLM-based evaluation is used
- Same top-k retrieval setting where applicable
- Same hardware or runtime environment

### 8.5 Experiment Procedure

1. Build or collect the document corpus.
2. Annotate document types and metadata.
3. Generate or collect evaluation queries.
4. Label ground-truth answers and evidence spans.
5. Build indexes for each retrieval strategy.
6. Run all baseline systems on the query set.
7. Run the proposed Agentic Adaptive RAG system.
8. Record retrieved chunks, answers, citations, latency, and token usage.
9. Compute automatic metrics.
10. Conduct human evaluation on a sampled subset.
11. Analyze results by document type and query type.
12. Perform ablation studies.

### 8.6 Ablation Studies

Recommended ablations:

- Without agentic planning
- Without document type classification
- Without query rewriting
- Without reranking
- Without metadata filtering
- Dense-only adaptive retrieval
- Hybrid-only adaptive retrieval
- Single-round retrieval vs multi-round retrieval
- Fixed Hybrid Retrieval vs Rule-Based Planner vs LLM-Based Planner

### 8.6.1 Planner Ablation Study

The planner ablation study should compare the following conditions:

| Condition | Description |
| --- | --- |
| A. Fixed BM25 | Sparse keyword retrieval without planning |
| B. Fixed Dense Retrieval | Dense semantic retrieval without planning |
| C. Fixed Hybrid Retrieval | Fixed dense plus sparse retrieval without planning |
| D. Rule-Based Retrieval Planner | Deterministic strategy selection followed by selected retrieval |
| E. LLM-Based Retrieval Planner | LLM-selected retrieval strategy followed by selected retrieval |

Purpose:

The purpose of this ablation is to isolate the contribution of retrieval planning from retrieval quality itself. Fixed BM25, dense, and hybrid retrieval measure retrieval method strength without planning. Rule-based and LLM-based planners measure whether choosing a retrieval strategy per query improves results beyond strong fixed baselines.

Evaluation metrics:

- Recall@K
- Precision@K
- MRR
- nDCG
- Strategy Selection Accuracy

Expected analysis:

- When does planning help?
- When does planning hurt?
- Which query categories benefit most?

### 8.7 Error Analysis

Error categories:

- Wrong document type selected
- Relevant document not retrieved
- Relevant chunk retrieved but ignored
- Answer hallucination
- Incorrect citation
- Outdated evidence used
- Conflicting evidence not detected
- Query misunderstood
- Table value extraction error
- Excessive retrieval overhead

## 9. Evaluation Metrics

Retrieval metrics alone are insufficient for final conclusions because the thesis focuses on Agentic RAG, not only document retrieval. Final evaluation should combine retrieval effectiveness, planner accuracy, answer-level quality, citation reliability, and cost-latency tradeoffs.

### 9.1 Retrieval Metrics

| Metric | Purpose |
| --- | --- |
| Recall@K | Whether relevant evidence appears in top K |
| Precision@K | Proportion of top K results that are relevant |
| MRR | Rank of first relevant result |
| nDCG@K | Ranking quality with graded relevance |
| Hit Rate@K | Whether at least one relevant item was retrieved |

### 9.2 Answer Quality Metrics

| Metric | Purpose |
| --- | --- |
| Exact Match | Strict answer match for fact lookup |
| F1 Score | Token-level overlap with reference answer |
| Semantic Similarity | Meaning-level similarity to reference answer |
| Completeness | Coverage of required answer elements |
| Correctness | Factual accuracy |
| Relevance | Whether the answer directly addresses the query |
| Answer Correctness | Whether the final answer is factually correct against the reference answer |
| Answer Groundedness | Whether the answer is fully supported by retrieved evidence |
| RAGAS Faithfulness | Optional automated faithfulness score |
| RAGAS Context Precision | Optional automated context precision score |

### 9.3 Groundedness and Citation Metrics

| Metric | Purpose |
| --- | --- |
| Faithfulness | Whether claims are supported by retrieved evidence |
| Citation Precision | Whether cited sources actually support the answer |
| Citation Recall | Whether all necessary evidence is cited |
| Unsupported Claim Rate | Rate of claims not grounded in evidence |
| Refusal Accuracy | Correct handling of unanswerable questions |

### 9.4 Agent Metrics

| Metric | Purpose |
| --- | --- |
| Strategy Selection Accuracy | Agreement between selected and oracle retrieval strategy |
| Tool Call Count | Number of retrieval or verification actions |
| Retrieval Iteration Count | Number of retrieval rounds |
| Evidence Sufficiency Accuracy | Correct judgment of whether evidence is enough |

### 9.4.1 Oracle Validation Framework

The `planner_oracle_strategy` field should not be treated as absolute truth. It is an experimental oracle label that must carry provenance through `oracle_strategy_source`.

Allowed `oracle_strategy_source` values:

- manual_annotation
- heuristic_assignment
- empirical_validation

Validation protocol:

1. Assign initial oracle labels using manual annotation, heuristic assignment, or both.
2. Run Fixed BM25, Fixed Dense Retrieval, Fixed Hybrid Retrieval, Rule-Based Retrieval Planner, and LLM-Based Retrieval Planner.
3. Review a subset of queries after retrieval experiments.
4. If empirical results consistently contradict oracle labels, document the contradiction.
5. Update the oracle label only with a recorded `oracle_strategy_source` of `empirical_validation`.
6. Preserve prior labels in experiment notes or versioned dataset outputs.

Purpose:

This framework prevents circular evaluation of planner performance by making oracle labels auditable and revisable when experimental evidence shows that the original label was unreliable.

### 9.5 Efficiency Metrics

| Metric | Purpose |
| --- | --- |
| End-to-End Latency | Total response time |
| Retrieval Latency | Time spent retrieving evidence |
| Reranking Latency | Time spent reranking |
| Token Usage | Prompt and completion token count |
| Estimated Cost | Model and infrastructure cost per query |
| Mean Latency | Average response latency by method |
| Median Latency | Median response latency by method |
| Retrieval Cost | Cost of retrieval, embedding, reranking, or index access |
| Planner Cost | Cost of rule execution or LLM planner inference |
| Total Pipeline Cost | End-to-end cost for planning, retrieval, generation, and verification |

Dedicated cost-latency result table:

| Method | Recall@K | Planner Accuracy | Latency | Cost |
| --- | --- | --- | --- | --- |
| Fixed BM25 | TBD | N/A | TBD | TBD |
| Fixed Dense Retrieval | TBD | N/A | TBD | TBD |
| Fixed Hybrid Retrieval | TBD | N/A | TBD | TBD |
| Rule-Based Retrieval Planner | TBD | TBD | TBD | TBD |
| LLM-Based Retrieval Planner | TBD | TBD | TBD | TBD |

Purpose:

This table supports practical enterprise deployment analysis by showing whether adaptive retrieval planning provides enough quality improvement to justify additional latency and inference cost.

### 9.6 Human Evaluation

Human evaluators should rate sampled answers using a 1-5 Likert scale.

Recommended criteria:

- Correctness
- Completeness
- Clarity
- Usefulness
- Citation reliability
- Trustworthiness

At least two evaluators should review each sampled answer when possible. Inter-annotator agreement should be reported using Cohen's kappa or Krippendorff's alpha.

## 10. Development Roadmap

### Phase 1. Research Design and Scope Definition

Deliverables:

- Finalized research questions
- Finalized hypotheses
- Document type taxonomy
- Evaluation protocol
- Initial thesis outline

### Phase 2. Dataset Construction

Deliverables:

- Document corpus
- Metadata schema
- Query set
- Ground-truth answers
- Evidence annotations

### Phase 3. Baseline RAG Implementation

Deliverables:

- Keyword retrieval baseline
- Dense retrieval baseline
- Hybrid retrieval baseline
- Fixed reranking baseline
- Baseline evaluation scripts

### Phase 4. Adaptive Retrieval Design

Deliverables:

- Query classifier
- Document type classifier
- Strategy mapping rules
- Retrieval planner specification
- Adaptive retrieval pipeline

### Phase 5. Agentic RAG System Development

Deliverables:

- Agent planner
- Retrieval tool interface
- Multi-step retrieval loop
- Evidence sufficiency checker
- Citation-aware answer generator
- Groundedness verifier

### Phase 6. Experiment Execution

Deliverables:

- Baseline results
- Proposed system results
- Ablation study results
- Cost and latency logs
- Error analysis dataset

### Phase 7. Analysis and Thesis Writing

Deliverables:

- Quantitative analysis
- Qualitative error analysis
- Discussion of findings
- Limitations
- Final thesis manuscript

### Phase 8. Final Demo and Presentation

Deliverables:

- Demo scenario
- Final system prototype
- Presentation slides
- Reproducibility guide
- Final repository cleanup

## 11. Risks and Mitigation

| Risk | Mitigation |
| --- | --- |
| Lack of real enterprise data | Use controlled synthetic data with realistic templates |
| Poor ground truth quality | Use manual verification and evidence span labeling |
| Agent behavior instability | Log planner decisions and use deterministic strategy rules for experiments |
| Evaluation bias from LLM judge | Combine automatic metrics, human evaluation, and evidence-based checks |
| Excessive system complexity | Implement baselines first, then add adaptive components incrementally |
| High latency | Compare quality-latency tradeoffs and optimize only validated bottlenecks |

## 12. Expected Contributions

1. A document-type-aware Agentic RAG architecture for enterprise knowledge management.
2. An adaptive retrieval strategy framework mapped to enterprise document types.
3. A controlled dataset design for evaluating enterprise RAG systems.
4. Experimental evidence comparing fixed retrieval and adaptive retrieval.
5. Practical guidelines for deploying reliable enterprise knowledge management systems.

### 12.1 Research Contribution Clarification

Expected thesis contributions:

| Contribution | Description |
| --- | --- |
| Contribution 1: SEKD (Synthetic Enterprise Knowledge Dataset) | A synthetic enterprise dataset with document categories, metadata, queries, ground truth, reasoning labels, and oracle strategy annotations |
| Contribution 2: Enterprise Retrieval Benchmark | A benchmark comparing BM25, dense, hybrid, rule-based adaptive retrieval, and LLM-based adaptive retrieval |
| Contribution 3: Rule-Based Retrieval Planner | A deterministic planner that maps query and document features to retrieval strategies |
| Contribution 4: LLM-Based Retrieval Planner | A language-model-based planner that selects retrieval strategies from query context and metadata |
| Contribution 5: Empirical Analysis of Retrieval Strategy Selection | An analysis of when retrieval planning helps, when it hurts, and which enterprise query categories benefit most |

## 13. Final Thesis Mapping

Every research question must have a measurable evaluation path.

| Research Question | Dataset Fields | Experiments | Metrics | Expected Thesis Chapter |
| --- | --- | --- | --- | --- |
| RQ1. Retrieval Effectiveness | `required_chunk_ids`, `required_document_ids`, `planner_oracle_strategy`, `retrieval_difficulty_factors` | Fixed BM25, Fixed Dense, Fixed Hybrid, Rule-Based Planner, LLM-Based Planner | Recall@K, Precision@K, MRR, nDCG | Chapter 5: Experiments |
| RQ2. Answer Quality | `reference_answer`, `evidence`, `expected_citations`, `acceptable_answer_patterns`, `disallowed_claims` | End-to-end RAG answer generation using each retrieval method | Answer Correctness, Answer Groundedness, Citation Precision, Citation Recall | Chapter 5: Experiments |
| RQ3. Document-Type Sensitivity | `category`, category metadata, `query_type`, `difficulty` | Grouped evaluation by document category | Retrieval metrics, answer metrics by category | Chapter 5: Experiments and Chapter 6: Discussion |
| RQ4. Agentic Planning Effectiveness | `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type`, `retrieval_difficulty_factors` | Planner ablation study | Strategy Selection Accuracy, Planner Precision, Planner Recall | Chapter 4: System Design and Chapter 5: Experiments |
| RQ5. Cost and Latency Tradeoffs | `difficulty`, `reasoning_type`, `required_document_count`, run manifests | Cost-latency comparison across methods | Mean Latency, Median Latency, Retrieval Cost, Planner Cost, Total Pipeline Cost | Chapter 5: Experiments and Chapter 6: Discussion |
| RQ6. Rule-Based Planner vs LLM-Based Planner | `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type`, planner outputs | Rule-Based Planner vs LLM-Based Planner comparison | Strategy Selection Accuracy, Planner Precision, Planner Recall, downstream retrieval metrics | Chapter 5: Experiments |
| RQ7. Cost Justification | run manifests, token usage logs, latency logs, method outputs | Quality-cost tradeoff analysis | Recall@K, Planner Accuracy, Latency, Cost | Chapter 6: Discussion |

## 14. Optimal Repository Structure

The repository should separate thesis artifacts, system implementation, data, experiments, and evaluation outputs.

```text
.
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── .env.example
├── .gitignore
│
├── docs/
│   ├── thesis/
│   │   ├── outline.md
│   │   ├── chapter_1_introduction.md
│   │   ├── chapter_2_related_work.md
│   │   ├── chapter_3_methodology.md
│   │   ├── chapter_4_system_design.md
│   │   ├── chapter_5_experiments.md
│   │   └── chapter_6_conclusion.md
│   ├── architecture/
│   │   ├── system_architecture.md
│   │   ├── retrieval_strategies.md
│   │   └── data_schema.md
│   └── presentation/
│       └── final_presentation.md
│
├── data/
│   ├── raw/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   ├── annotations/
│   │   ├── queries.jsonl
│   │   ├── ground_truth.jsonl
│   │   └── evidence_spans.jsonl
│   └── synthetic/
│       ├── templates/
│       └── generated/
│
├── configs/
│   ├── retrieval/
│   │   ├── keyword.yaml
│   │   ├── dense.yaml
│   │   ├── hybrid.yaml
│   │   └── adaptive.yaml
│   ├── models.yaml
│   ├── dataset.yaml
│   └── experiment.yaml
│
├── src/
│   └── enterprise_rag/
│       ├── __init__.py
│       ├── ingestion/
│       │   ├── loaders.py
│       │   ├── metadata.py
│       │   └── document_classifier.py
│       ├── preprocessing/
│       │   ├── chunking.py
│       │   ├── table_extraction.py
│       │   └── text_normalization.py
│       ├── indexing/
│       │   ├── vector_index.py
│       │   ├── sparse_index.py
│       │   └── metadata_store.py
│       ├── retrieval/
│       │   ├── keyword_retriever.py
│       │   ├── dense_retriever.py
│       │   ├── hybrid_retriever.py
│       │   ├── hierarchical_retriever.py
│       │   ├── table_retriever.py
│       │   └── adaptive_retriever.py
│       ├── agent/
│       │   ├── planner.py
│       │   ├── tools.py
│       │   ├── query_analyzer.py
│       │   └── evidence_checker.py
│       ├── generation/
│       │   ├── answer_generator.py
│       │   ├── citation_builder.py
│       │   └── groundedness_verifier.py
│       └── evaluation/
│           ├── retrieval_metrics.py
│           ├── answer_metrics.py
│           ├── citation_metrics.py
│           └── human_eval_schema.py
│
├── experiments/
│   ├── baselines/
│   │   ├── keyword_rag.yaml
│   │   ├── dense_rag.yaml
│   │   ├── hybrid_rag.yaml
│   │   └── fixed_rerank_rag.yaml
│   ├── adaptive/
│   │   └── agentic_adaptive_rag.yaml
│   ├── ablations/
│   │   ├── no_agent_planner.yaml
│   │   ├── no_metadata_filter.yaml
│   │   ├── no_reranker.yaml
│   │   └── single_round_retrieval.yaml
│   └── results/
│       └── .gitkeep
│
├── scripts/
│   ├── build_dataset.py
│   ├── build_indexes.py
│   ├── run_experiment.py
│   ├── run_ablation.py
│   ├── evaluate_results.py
│   └── generate_report_tables.py
│
├── notebooks/
│   ├── dataset_analysis.ipynb
│   ├── retrieval_error_analysis.ipynb
│   └── results_visualization.ipynb
│
├── tests/
│   ├── test_chunking.py
│   ├── test_retrieval.py
│   ├── test_planner.py
│   └── test_metrics.py
│
└── outputs/
    ├── logs/
    ├── figures/
    ├── tables/
    └── reports/
```

### Repository Structure Rationale

- `docs/` stores thesis writing and architecture documentation separately from code.
- `data/` separates raw, processed, synthetic, and annotated datasets.
- `configs/` makes retrieval and experiment settings reproducible.
- `src/enterprise_rag/` contains the actual system modules.
- `experiments/` stores baseline, proposed, and ablation configurations.
- `scripts/` contains reproducible command-line workflows.
- `notebooks/` supports exploratory analysis and visualization.
- `tests/` validates critical system components.
- `outputs/` stores generated logs, figures, tables, and reports.

For thesis reproducibility, large raw documents, vector indexes, model outputs, and logs should be excluded from Git when they are too large or sensitive. The repository should include lightweight samples, schemas, and scripts that can regenerate the experimental artifacts.
