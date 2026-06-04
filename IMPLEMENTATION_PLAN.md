# IMPLEMENTATION_PLAN.md

## Project

Agentic RAG 기반 기업 지식관리시스템 설계 및 평가: 문서 유형에 따른 Adaptive Retrieval 전략을 중심으로

## Version

1.2

## Purpose

This implementation plan defines a step-by-step path for building a research prototype that evaluates Agentic RAG and document-type-aware Adaptive Retrieval for enterprise knowledge management. The plan prioritizes reproducibility, experimental rigor, traceable outputs, and clear separation between dataset generation, retrieval systems, planners, evaluation, and analysis.

The implementation must follow:

- `PROJECT_SPEC.md`
- `DATASET_SPEC.md`

No experimental result should be treated as valid unless the exact dataset version, seed, configuration, retrieval method, planner type, and evaluation script version are recorded.

## Reproducibility Principles

1. Every generated artifact must be reproducible from a seed and config file.
2. Every experiment must write a machine-readable run manifest.
3. Baselines and proposed methods must use the same corpus, queries, chunks, and evaluation logic.
4. Randomness must be controlled for document generation, query generation, embedding order, and experiment sampling.
5. Intermediate files should be saved so failures can be inspected without rerunning the whole pipeline.
6. Evaluation scripts should be deterministic wherever possible.
7. LLM-based planner outputs must be logged exactly, including prompt, response, parsed strategy, and parsing errors.

## Recommended Libraries and Tools

### Core Language and Packaging

- Python 3.11+
- `pyproject.toml`
- `uv` or `pip-tools` for dependency locking
- `ruff` for linting and formatting
- `pytest` for tests
- `pydantic` for schema validation
- `typer` for command-line interfaces
- `pyyaml` for config files

### Data Generation and Processing

- `faker` for synthetic names, dates, and role-like labels
- `jinja2` for document templates
- `pandas` for tabular data handling
- `jsonlines` for JSONL read/write
- `python-dateutil` for date handling

### Retrieval

- `rank-bm25` or `bm25s` for BM25 baseline
- `sentence-transformers` for dense embeddings
- `faiss-cpu` for vector indexing
- `scikit-learn` for normalization, metrics, and utility functions
- Optional: `tantivy` or `Pyserini` if stronger sparse retrieval is needed

### LLM and Experiment Interfaces

- `openai` or another model SDK selected by the researcher
- `tenacity` for retries
- `python-dotenv` for environment variables
- `tqdm` for progress bars

### Evaluation and Analysis

- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `seaborn`
- `plotly`
- `jupyter`
- Optional: `ragas` for supplementary RAG evaluation, not as the sole evaluator

### Development Quality

- `pre-commit`
- `mypy` or `pyright`
- `pytest-cov`
- Git for version control

## Development Phases

## Phase 1: Repository Setup

### Objectives

- Establish a reproducible Python research repository.
- Create the initial directory structure.
- Define package metadata, dependency management, linting, formatting, and test commands.
- Preserve `PROJECT_SPEC.md`, `DATASET_SPEC.md`, and this implementation plan as top-level research documents.

### Deliverables

- Python package scaffold
- Config directory
- Data directory placeholders
- Test directory
- README with basic usage
- Dependency files
- Initial reproducibility conventions

### Dependencies

- Finalized `PROJECT_SPEC.md`
- Finalized `DATASET_SPEC.md`

### Success Criteria

- Repository installs locally.
- `pytest` runs successfully.
- `ruff` runs successfully.
- Empty pipeline directories are present.
- README explains the intended command sequence at a high level.

### Complexity Estimate

Low

### Expected Output Files

```text
README.md
pyproject.toml
requirements.txt
.env.example
.gitignore
configs/dataset.yaml
configs/retrieval/bm25.yaml
configs/retrieval/dense.yaml
configs/retrieval/hybrid.yaml
configs/planners/rule_based.yaml
configs/planners/llm_based.yaml
configs/experiment.yaml
src/enterprise_rag/__init__.py
tests/test_repository_setup.py
```

## Phase 2: SEKD Synthetic Document Generator

### Objectives

- Implement deterministic generation of SEKD documents for HYTech Solutions.
- Generate all six document categories from `DATASET_SPEC.md`.
- Produce full documents with valid global and category-specific metadata.
- Ensure generated documents contain controlled facts that can later support ground truth generation.

### Deliverables

- Template-based document generator
- Category-specific generators
- Global metadata generator
- Validation logic for required fields and enum values
- CLI command for generating documents

### Dependencies

- Phase 1
- `DATASET_SPEC.md`

### Success Criteria

- Same seed produces identical `documents.jsonl`.
- Every document has all required global metadata.
- Every document has category-specific required fields.
- Generated categories include HR Policies, Travel Policies, Security Policies, API Documentation, System Design Documents, and Project Meeting Notes.
- No generated document contains real credentials, real personal data, or external proprietary data.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/dataset/generate_documents.py
src/enterprise_rag/dataset/templates/hr_policy.j2
src/enterprise_rag/dataset/templates/travel_policy.j2
src/enterprise_rag/dataset/templates/security_policy.j2
src/enterprise_rag/dataset/templates/api_doc.j2
src/enterprise_rag/dataset/templates/system_design.j2
src/enterprise_rag/dataset/templates/meeting_notes.j2
src/enterprise_rag/dataset/schemas.py
src/enterprise_rag/dataset/validators.py
scripts/generate_documents.py
data/sekd/documents.jsonl
data/sekd/metadata_schema.json
data/sekd/category_config.json
tests/test_document_generation.py
```

## Phase 3: Chunking & Metadata Pipeline

### Objectives

- Convert generated documents into retrievable chunks.
- Preserve section hierarchy, tables, endpoint blocks, policy clauses, and meeting-note structures.
- Attach inherited and chunk-specific metadata to each chunk.
- Support multiple chunking strategies while keeping one default canonical chunk set for experiments.

### Deliverables

- Section-aware chunker
- Table-aware chunker
- API endpoint block chunker
- Meeting note chunker
- Metadata inheritance logic
- Chunk validation logic

### Dependencies

- Phase 2

### Success Criteria

- Every chunk references an existing document.
- Every chunk has `chunk_id`, `document_id`, `category`, `section_path`, `text`, metadata, and character offsets.
- Tables remain interpretable after chunking.
- API endpoint blocks are not split across unrelated chunks.
- Chunk IDs follow `{document_id}-C{three_digit_number}`.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/preprocessing/chunking.py
src/enterprise_rag/preprocessing/section_parser.py
src/enterprise_rag/preprocessing/table_parser.py
src/enterprise_rag/preprocessing/metadata.py
scripts/build_chunks.py
data/sekd/chunks.jsonl
tests/test_chunking.py
tests/test_metadata_pipeline.py
```

## Phase 4: Query & Ground Truth Generator

### Objectives

- Generate evaluation queries from controlled document facts.
- Assign query taxonomy, difficulty level, oracle retrieval strategy, reasoning type, retrieval difficulty factors, and required document count.
- Generate ground truth answers with exact evidence spans.
- Enforce target distributions from `DATASET_SPEC.md` Version 1.2.

### Deliverables

- Query generation engine
- Category-specific query templates
- Ground truth builder
- Evidence span linker
- Distribution controller
- Query and ground truth validators

### Dependencies

- Phase 2
- Phase 3

### Success Criteria

- Same seed produces identical `queries.jsonl` and `ground_truth.jsonl`.
- Every answerable query has supporting document and chunk IDs.
- Every unanswerable query has empty required document and chunk IDs.
- Every query includes `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type`, `retrieval_difficulty_factors`, and `required_document_count`.
- Generated query distributions approximate the target distributions.
- Evidence spans map to valid chunks.

### Complexity Estimate

High

### Expected Output Files

```text
src/enterprise_rag/dataset/generate_queries.py
src/enterprise_rag/dataset/query_templates.py
src/enterprise_rag/dataset/ground_truth.py
src/enterprise_rag/dataset/distributions.py
scripts/generate_queries.py
data/sekd/queries.jsonl
data/sekd/ground_truth.jsonl
tests/test_query_generation.py
tests/test_ground_truth_validation.py
```

## Phase 5: BM25 Retrieval Baseline

### Objectives

- Implement sparse lexical retrieval baseline.
- Index generated chunks.
- Retrieve top-k chunks for each query.
- Save retrieval outputs in a standardized run format.

### Deliverables

- BM25 tokenizer
- BM25 index builder
- BM25 retriever
- Standard retrieval result writer

### Dependencies

- Phase 3
- Phase 4

### Success Criteria

- BM25 index builds from `chunks.jsonl`.
- Each query returns top-k ranked chunks.
- Output includes score, rank, chunk ID, document ID, and query ID.
- Retrieval evaluation can consume BM25 output.

### Complexity Estimate

Low

### Expected Output Files

```text
src/enterprise_rag/retrieval/bm25.py
src/enterprise_rag/retrieval/tokenization.py
src/enterprise_rag/retrieval/result_schema.py
scripts/build_bm25_index.py
scripts/run_bm25.py
outputs/indexes/bm25/
outputs/runs/bm25/retrieval_results.jsonl
tests/test_bm25_retrieval.py
```

## Phase 6: Dense Retrieval Baseline

### Objectives

- Implement semantic embedding retrieval baseline.
- Build vector embeddings for all chunks.
- Retrieve nearest chunks for each query.
- Ensure embedding model and index metadata are recorded.

### Deliverables

- Embedding wrapper
- Vector index builder
- Dense retriever
- Embedding cache
- Run manifest with model name and embedding dimension

### Dependencies

- Phase 3
- Phase 4

### Success Criteria

- Dense index builds from `chunks.jsonl`.
- Query embeddings are generated reproducibly for the same model.
- Each query returns top-k ranked chunks.
- Output format matches BM25 output.
- Manifest records model, distance function, index type, and timestamp.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/retrieval/embeddings.py
src/enterprise_rag/retrieval/vector_index.py
src/enterprise_rag/retrieval/dense.py
scripts/build_dense_index.py
scripts/run_dense.py
outputs/indexes/dense/
outputs/runs/dense/retrieval_results.jsonl
outputs/runs/dense/run_manifest.json
tests/test_dense_retrieval.py
```

## Phase 7: Hybrid Retrieval Baseline

### Objectives

- Combine BM25 and dense retrieval into a fixed hybrid retrieval baseline.
- Normalize and fuse sparse and dense scores.
- Support fixed top-k and optional reranking.
- Establish the main fixed retrieval baseline for comparison with planners.

### Deliverables

- Score normalization
- Reciprocal rank fusion or weighted score fusion
- Hybrid retriever
- Hybrid config

### Dependencies

- Phase 5
- Phase 6

### Success Criteria

- Hybrid retrieval consumes BM25 and dense indexes.
- Fusion method is configurable.
- Output format matches other retrieval systems.
- Hybrid baseline can be evaluated side-by-side with BM25 and dense baselines.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/retrieval/hybrid.py
src/enterprise_rag/retrieval/fusion.py
scripts/run_hybrid.py
configs/retrieval/hybrid.yaml
outputs/runs/hybrid/retrieval_results.jsonl
outputs/runs/hybrid/run_manifest.json
tests/test_hybrid_retrieval.py
```

## Phase 8: Rule-Based Retrieval Planner

### Objectives

- Implement deterministic retrieval strategy selection using rules.
- Map query metadata, reasoning type, difficulty factors, and category to retrieval strategies.
- Produce planner evaluation logs against `planner_oracle_strategy`.
- Route queries to the selected retrieval method.

### Deliverables

- Rule-based planner
- Strategy mapping config
- Planner output schema
- Planner evaluation output
- Adaptive retrieval runner for rule-based planning

### Dependencies

- Phase 5
- Phase 6
- Phase 7
- Phase 4

### Success Criteria

- Every query receives exactly one selected strategy.
- Selected strategies use only allowed oracle strategy values.
- Planner outputs include selected strategy, oracle strategy, correctness, and planner type.
- Rule-based adaptive retrieval produces standard retrieval results.
- Rules are documented and deterministic.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/planning/rule_based.py
src/enterprise_rag/planning/schema.py
src/enterprise_rag/retrieval/adaptive.py
configs/planners/rule_based.yaml
scripts/run_rule_based_planner.py
outputs/runs/rule_based_planner/planner_evaluation.jsonl
outputs/runs/rule_based_planner/retrieval_results.jsonl
outputs/runs/rule_based_planner/run_manifest.json
tests/test_rule_based_planner.py
```

## Phase 9: LLM-Based Retrieval Planner

### Objectives

- Implement an LLM-based planner that selects retrieval strategy from query text and metadata.
- Require structured output parsing.
- Log prompts, raw responses, parsed plans, and parsing failures.
- Compare LLM planner decisions against `planner_oracle_strategy`.

### Deliverables

- Prompt templates
- LLM planner client
- Structured response parser
- Retry and fallback handling
- Planner trace logs
- LLM-based adaptive retrieval runner

### Dependencies

- Phase 8
- Model provider credentials

### Success Criteria

- Every query receives a parseable selected strategy or a logged fallback.
- LLM planner output uses allowed strategy values only.
- Planner evaluation output matches the rule-based schema.
- Prompt version is recorded in run manifest.
- Token usage and latency are recorded when available.

### Complexity Estimate

High

### Expected Output Files

```text
src/enterprise_rag/planning/llm_based.py
src/enterprise_rag/planning/prompts/retrieval_planner.md
src/enterprise_rag/planning/parser.py
scripts/run_llm_planner.py
outputs/runs/llm_planner/planner_evaluation.jsonl
outputs/runs/llm_planner/planner_traces.jsonl
outputs/runs/llm_planner/retrieval_results.jsonl
outputs/runs/llm_planner/run_manifest.json
tests/test_llm_planner_parser.py
```

## Phase 10: Evaluation Framework

### Objectives

- Implement retrieval, planner, citation, and answer evaluation metrics.
- Ensure all baselines and planner methods are evaluated by the same code.
- Produce metric tables suitable for thesis analysis.
- Validate oracle strategy labels without treating them as absolute truth.
- Measure whether retrieval and answer quality improvements justify additional cost and latency.

### Deliverables

- Retrieval metrics: Recall@K, Precision@K, MRR, nDCG@K, Hit Rate@K
- Planner metrics: Strategy Selection Accuracy, Planner Precision, Planner Recall
- Required answer metrics: Answer Correctness, Answer Groundedness, Citation Precision, Citation Recall
- Optional answer metrics: RAGAS Faithfulness, RAGAS Context Precision
- Cost and latency metrics: Mean Latency, Median Latency, Retrieval Cost, Planner Cost, Total Pipeline Cost
- Oracle validation report using `oracle_strategy_source`
- Grouped evaluation by category, difficulty, reasoning type, oracle strategy, and difficulty factors

### Dependencies

- Phase 4
- Phase 5
- Phase 6
- Phase 7
- Phase 8
- Phase 9

### Success Criteria

- Evaluation scripts produce deterministic outputs.
- Metrics are computed from saved run outputs, not in-memory objects.
- Results include both aggregate and grouped metrics.
- Retrieval metrics, planner metrics, answer metrics, and cost-latency metrics are reported separately.
- Oracle validation identifies labels that require review after empirical retrieval results.
- Invalid records fail validation before scoring.

### Complexity Estimate

High

### Expected Output Files

```text
src/enterprise_rag/evaluation/retrieval_metrics.py
src/enterprise_rag/evaluation/planner_metrics.py
src/enterprise_rag/evaluation/answer_metrics.py
src/enterprise_rag/evaluation/cost_latency.py
src/enterprise_rag/evaluation/oracle_validation.py
src/enterprise_rag/evaluation/grouped_analysis.py
src/enterprise_rag/evaluation/schemas.py
scripts/evaluate_retrieval.py
scripts/evaluate_planners.py
scripts/evaluate_answers.py
scripts/evaluate_cost_latency.py
scripts/validate_oracle_labels.py
outputs/evaluation/retrieval_metrics.csv
outputs/evaluation/planner_metrics.csv
outputs/evaluation/answer_metrics.csv
outputs/evaluation/cost_latency_metrics.csv
outputs/evaluation/grouped_metrics.csv
outputs/evaluation/oracle_validation_report.md
tests/test_retrieval_metrics.py
tests/test_planner_metrics.py
tests/test_answer_metrics.py
tests/test_cost_latency_metrics.py
tests/test_oracle_validation.py
```

## Phase 11: Experiment Runner

### Objectives

- Create a single reproducible experiment runner.
- Execute dataset generation, chunking, indexing, retrieval, planning, and evaluation from config.
- Record run manifests for every experiment.
- Support repeated runs with different seeds.

### Deliverables

- Experiment orchestration CLI
- Run manifest schema
- Config loader
- Output directory manager
- Seed management
- Failure logging

### Dependencies

- Phase 10

### Success Criteria

- One command can run the full MVP experiment.
- Every run writes a manifest with seed, dataset version, config paths, method names, model names, and Git commit if available.
- Re-running the same config and seed produces the same non-LLM artifacts.
- Experiment outputs are isolated by run ID.

### Complexity Estimate

Medium

### Expected Output Files

```text
src/enterprise_rag/experiments/runner.py
src/enterprise_rag/experiments/manifest.py
src/enterprise_rag/experiments/config.py
scripts/run_experiment.py
configs/experiments/mvp.yaml
configs/experiments/full_comparison.yaml
outputs/experiments/{run_id}/run_manifest.json
outputs/experiments/{run_id}/retrieval_metrics.csv
outputs/experiments/{run_id}/planner_metrics.csv
outputs/experiments/{run_id}/answer_metrics.csv
outputs/experiments/{run_id}/cost_latency_metrics.csv
tests/test_experiment_runner.py
```

## Phase 12: Results Analysis & Visualization

### Objectives

- Analyze experimental results for thesis findings.
- Visualize retrieval quality, planner accuracy, answer quality, cost, and latency.
- Compare fixed retrieval, rule-based planner, and LLM-based planner.
- Generate tables and figures for the final thesis.

### Deliverables

- Analysis notebooks
- Plot generation scripts
- Thesis-ready tables
- Error analysis reports
- Result summary document

### Dependencies

- Phase 11

### Success Criteria

- Results are grouped by document category, reasoning type, oracle strategy, difficulty level, and retrieval difficulty factors.
- Figures are reproducible from CSV outputs.
- Error analysis identifies common failure modes.
- Final result tables directly answer RQ1-RQ7.
- The dedicated table `Method | Recall@K | Planner Accuracy | Latency | Cost` is generated for enterprise deployment analysis.

### Complexity Estimate

Medium

### Expected Output Files

```text
notebooks/results_analysis.ipynb
notebooks/error_analysis.ipynb
scripts/generate_figures.py
scripts/generate_result_tables.py
outputs/figures/retrieval_metrics_by_method.png
outputs/figures/planner_accuracy_by_reasoning_type.png
outputs/figures/latency_quality_tradeoff.png
outputs/tables/main_results.csv
outputs/tables/planner_comparison.csv
outputs/tables/cost_latency_tradeoff.csv
outputs/tables/planner_ablation.csv
outputs/reports/error_analysis.md
outputs/reports/final_results_summary.md
```

## Planner Ablation Study

The implementation must include a dedicated planner ablation study.

Comparison conditions:

| Condition | Method Key | Purpose |
| --- | --- | --- |
| A. Fixed BM25 | `fixed_bm25` | Measure sparse keyword retrieval without planning |
| B. Fixed Dense Retrieval | `fixed_dense` | Measure semantic retrieval without planning |
| C. Fixed Hybrid Retrieval | `fixed_hybrid` | Measure a strong fixed retrieval baseline |
| D. Rule-Based Retrieval Planner | `rule_based_planner` | Measure deterministic adaptive strategy selection |
| E. LLM-Based Retrieval Planner | `llm_based_planner` | Measure LLM-driven adaptive strategy selection |

Purpose:

The ablation isolates the contribution of retrieval planning from retrieval quality itself. It must make clear whether gains come from the underlying retriever or from selecting a better retriever per query.

Required metrics:

- Recall@K
- Precision@K
- MRR
- nDCG
- Strategy Selection Accuracy

Expected analysis questions:

- When does planning help?
- When does planning hurt?
- Which query categories benefit most?

Expected output files:

```text
configs/experiments/planner_ablation.yaml
outputs/tables/planner_ablation.csv
outputs/figures/planner_ablation_by_query_category.png
outputs/reports/planner_ablation_analysis.md
```

## Oracle Validation Framework

The implementation must not treat `planner_oracle_strategy` as absolute truth.

Required query metadata:

- `planner_oracle_strategy`
- `oracle_strategy_source`

Allowed `oracle_strategy_source` values:

- `manual_annotation`
- `heuristic_assignment`
- `empirical_validation`

Validation protocol:

1. Generate initial oracle strategy labels during Phase 4.
2. Record `oracle_strategy_source` for every query.
3. Run the planner ablation study.
4. Review a subset of queries where empirical results consistently contradict oracle labels.
5. Document the contradiction and the observed best-performing method.
6. If the oracle label is updated, set `oracle_strategy_source` to `empirical_validation`.
7. Save oracle validation findings separately from primary metric outputs.

Purpose:

This prevents circular evaluation of planner performance and keeps strategy labels auditable across dataset versions.

Expected output files:

```text
outputs/evaluation/oracle_validation_report.md
outputs/evaluation/oracle_validation_cases.jsonl
```

## Answer-Level Evaluation

The final thesis conclusions must not rely on retrieval metrics alone. Because the project evaluates Agentic RAG, retrieval effectiveness must be connected to final answer quality.

Required metrics:

- Answer Correctness
- Answer Groundedness
- Citation Precision
- Citation Recall

Optional metrics:

- RAGAS Faithfulness
- RAGAS Context Precision

Expected output files:

```text
outputs/evaluation/answer_metrics.csv
outputs/tables/answer_quality_by_method.csv
outputs/figures/answer_groundedness_by_method.png
```

## Cost and Latency Analysis

The implementation must evaluate whether performance improvements justify additional computational cost.

Required metrics:

- Mean Latency
- Median Latency
- Retrieval Cost
- Planner Cost
- Total Pipeline Cost

Dedicated result table:

| Method | Recall@K | Planner Accuracy | Latency | Cost |
| --- | --- | --- | --- | --- |
| Fixed BM25 | TBD | N/A | TBD | TBD |
| Fixed Dense Retrieval | TBD | N/A | TBD | TBD |
| Fixed Hybrid Retrieval | TBD | N/A | TBD | TBD |
| Rule-Based Retrieval Planner | TBD | TBD | TBD | TBD |
| LLM-Based Retrieval Planner | TBD | TBD | TBD | TBD |

Purpose:

This table supports practical enterprise deployment analysis by showing whether adaptive retrieval planning provides enough value to justify additional latency and inference cost.

Expected output files:

```text
outputs/tables/cost_latency_tradeoff.csv
outputs/figures/latency_quality_tradeoff.png
outputs/reports/cost_latency_analysis.md
```

## MVP Milestones

### MVP 1: Reproducible Dataset

Scope:

- Phase 1
- Phase 2
- Phase 3
- Phase 4

Exit criteria:

- `documents.jsonl`, `chunks.jsonl`, `queries.jsonl`, and `ground_truth.jsonl` are generated from one seed.
- All generated files pass validation.
- Query distributions are reported.

### MVP 2: Retrieval Baselines

Scope:

- Phase 5
- Phase 6
- Phase 7
- Initial Phase 10 retrieval metrics

Exit criteria:

- BM25, dense, and hybrid retrieval run over the same dataset.
- Retrieval metrics are computed for all three methods.
- Main baseline table is generated.

### MVP 3: Planner Comparison

Scope:

- Phase 8
- Phase 9
- Phase 10 planner metrics

Exit criteria:

- Rule-based and LLM-based planners produce selected strategies for all queries.
- Strategy Selection Accuracy, Planner Precision, and Planner Recall are computed.
- Planner outputs can be grouped by reasoning type and difficulty factors.

### MVP 4: End-to-End Experiment

Scope:

- Phase 11
- Phase 12

Exit criteria:

- Full experiment can be executed from config.
- Results are saved under a run ID.
- Thesis-ready result tables and figures are generated.

## Evaluation Checkpoints

### Checkpoint A: Dataset Validation

Occurs after Phase 4.

Required checks:

- Document schema validation
- Chunk schema validation
- Query schema validation
- Ground truth schema validation
- Evidence chunk existence
- Distribution report
- Duplicate ID check
- `oracle_strategy_source` validation

Blocking condition:

- Do not implement retrieval experiments until generated dataset files pass validation.

### Checkpoint B: Baseline Retrieval Sanity

Occurs after Phase 7.

Required checks:

- All retrieval methods return top-k results for every query.
- At least one simple exact-ID query succeeds under BM25.
- At least one paraphrased query succeeds under dense retrieval.
- Hybrid retrieval does not produce empty rankings.

Blocking condition:

- Do not compare planners until fixed baselines are stable.

### Checkpoint C: Planner Validity

Occurs after Phase 9.

Required checks:

- Every planner output uses an allowed strategy.
- Every planner output includes planner type.
- LLM parsing failures are counted and logged.
- Rule-based planner is deterministic across repeated runs.
- Planner accuracy is reported against oracle labels and oracle label provenance is retained.

Blocking condition:

- Do not report planner accuracy without parsing failure rates.

### Checkpoint D: Metric Reproducibility

Occurs after Phase 10.

Required checks:

- Metrics recompute identically from saved outputs.
- Grouped metrics aggregate to expected totals.
- Evaluation scripts fail on malformed inputs.
- Answer-level metrics are computed separately from retrieval metrics.
- Cost and latency metrics are computed separately from quality metrics.

Blocking condition:

- Do not run full experiments until metrics are validated with unit tests.

### Checkpoint E: Thesis Result Readiness

Occurs after Phase 12.

Required checks:

- Main tables answer RQ1-RQ7.
- Figures are generated from saved CSV files.
- Error analysis includes representative examples.
- Cost and latency are reported separately from accuracy metrics.
- The table `Method | Recall@K | Planner Accuracy | Latency | Cost` is available.

## Repository Structure Evolution by Phase

### After Phase 1

```text
.
├── PROJECT_SPEC.md
├── DATASET_SPEC.md
├── IMPLEMENTATION_PLAN.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── configs/
├── src/enterprise_rag/
└── tests/
```

### After Phase 4

```text
.
├── data/sekd/
│   ├── documents.jsonl
│   ├── chunks.jsonl
│   ├── queries.jsonl
│   ├── ground_truth.jsonl
│   ├── metadata_schema.json
│   └── category_config.json
├── src/enterprise_rag/dataset/
├── src/enterprise_rag/preprocessing/
└── scripts/
```

### After Phase 7

```text
.
├── src/enterprise_rag/retrieval/
├── outputs/indexes/
│   ├── bm25/
│   └── dense/
└── outputs/runs/
    ├── bm25/
    ├── dense/
    └── hybrid/
```

### After Phase 10

```text
.
├── src/enterprise_rag/planning/
├── src/enterprise_rag/evaluation/
└── outputs/evaluation/
    ├── retrieval_metrics.csv
    ├── planner_metrics.csv
    └── grouped_metrics.csv
```

### After Phase 12

```text
.
├── configs/experiments/
├── notebooks/
├── outputs/experiments/
├── outputs/figures/
├── outputs/tables/
└── outputs/reports/
```

## Implementation Risks and Mitigation Plans

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Synthetic data is too simple | Results may overstate planner performance | Add ambiguity, version conflicts, metadata dependency, tables, exceptions, and multi-document tasks |
| Ground truth evidence is incorrect | Retrieval and answer metrics become unreliable | Validate evidence spans against chunk text and add unit tests for generated examples |
| Query distributions drift from target | Planner evaluation may be biased | Implement distribution reports and fail when deviation exceeds configured tolerance |
| BM25 tokenization is weak for IDs | Keyword baseline underperforms unfairly | Preserve IDs, endpoint paths, control IDs, and hyphenated terms during tokenization |
| Dense model changes across runs | Results become non-reproducible | Record model name, version, embedding dimension, and cache embeddings |
| Hybrid fusion dominates planner comparison | Adaptive gains become unclear | Report fixed hybrid separately and run ablation against planner-selected strategies |
| Rule-based planner overfits oracle labels | Comparison becomes artificial | Define rules before reviewing full generated query set and report per-factor performance |
| LLM planner output is inconsistent | Strategy accuracy is noisy | Use structured prompts, strict parsing, retries, temperature 0, and log parse failures |
| LLM API cost grows | Experiments become expensive | Cache planner outputs and run LLM planner only after dataset and baselines are stable |
| Latency measurements are noisy | Cost-latency claims become weak | Run repeated measurements and report mean, median, and standard deviation |
| Evaluation code has hidden assumptions | Metrics may be invalid | Test metrics with small hand-written fixtures |
| Large generated files enter Git | Repository becomes hard to manage | Track only small samples and configs; ignore large outputs and indexes |

## Recommended Execution Order

1. Complete Phase 1 and commit the repository scaffold.
2. Complete Phase 2 with a small document count.
3. Complete Phase 3 and validate chunks manually on one document per category.
4. Complete Phase 4 with 30 sample queries before scaling to 300-600.
5. Complete BM25 first to validate exact evidence and IDs.
6. Add dense retrieval and verify paraphrased queries.
7. Add hybrid retrieval and establish the fixed baseline.
8. Implement rule-based planner and compare against oracle strategy.
9. Implement LLM-based planner only after all non-LLM components are stable.
10. Finalize evaluation metrics and grouped analysis.
11. Run full experiments with at least three seeds if time permits.
12. Generate thesis tables, figures, and error analysis.

## Final Research Outputs

The completed project should produce:

- A reproducible synthetic enterprise dataset
- A fixed retrieval baseline comparison
- A rule-based adaptive retrieval planner
- An LLM-based adaptive retrieval planner
- Retrieval effectiveness results
- Planner strategy selection results
- Grouped analysis by document type and reasoning type
- Error analysis
- Cost and latency analysis
- Thesis-ready figures and tables

## Research Contribution Clarification

Expected thesis contributions:

| Contribution | Implementation Output |
| --- | --- |
| Contribution 1: SEKD (Synthetic Enterprise Knowledge Dataset) | Deterministic synthetic documents, chunks, queries, ground truth, metadata, and validation scripts |
| Contribution 2: Enterprise Retrieval Benchmark | Fixed BM25, fixed dense, fixed hybrid, rule-based planner, and LLM-based planner comparison |
| Contribution 3: Rule-Based Retrieval Planner | Deterministic planner implementation, rules config, planner evaluation output |
| Contribution 4: LLM-Based Retrieval Planner | Structured LLM planner, prompt templates, planner traces, parsed strategy outputs |
| Contribution 5: Empirical Analysis of Retrieval Strategy Selection | Planner ablation, grouped analysis, oracle validation, answer-level evaluation, and cost-latency reports |

## Final Thesis Mapping

Every research question must map to dataset fields, experiments, metrics, and thesis chapters.

| Research Question | Dataset Fields | Experiments | Metrics | Expected Thesis Chapter |
| --- | --- | --- | --- | --- |
| RQ1. Retrieval Effectiveness | `required_document_ids`, `required_chunk_ids`, `planner_oracle_strategy` | Fixed BM25, Fixed Dense, Fixed Hybrid, Rule-Based Planner, LLM-Based Planner | Recall@K, Precision@K, MRR, nDCG | Chapter 5 |
| RQ2. Answer Quality | `reference_answer`, `evidence`, `expected_citations`, `disallowed_claims` | End-to-end RAG answer generation | Answer Correctness, Answer Groundedness, Citation Precision, Citation Recall | Chapter 5 |
| RQ3. Document-Type Sensitivity | `category`, category metadata, `query_type`, `difficulty` | Grouped evaluation by document type | Retrieval and answer metrics by category | Chapter 5 and Chapter 6 |
| RQ4. Agentic Planning Effectiveness | `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type` | Planner ablation study | Strategy Selection Accuracy, Planner Precision, Planner Recall | Chapter 4 and Chapter 5 |
| RQ5. Cost and Latency Tradeoffs | `difficulty`, `required_document_count`, run manifests | Cost-latency comparison across methods | Mean Latency, Median Latency, Retrieval Cost, Planner Cost, Total Pipeline Cost | Chapter 5 and Chapter 6 |
| RQ6. Rule-Based Planner vs LLM-Based Planner | planner outputs, `planner_oracle_strategy`, `oracle_strategy_source` | Rule-based vs LLM-based planner comparison | Strategy Selection Accuracy, Planner Precision, Planner Recall, downstream retrieval metrics | Chapter 5 |
| RQ7. Cost Justification | run manifests, token usage logs, latency logs, method outputs | Quality-cost tradeoff analysis | Recall@K, Planner Accuracy, Latency, Cost | Chapter 6 |
