# V2 Research Extension Plan
## Agentic RAG for Enterprise Knowledge Management

**Project**: Information System Capstone Design  
**Author**: Won Young Shin (신원영), Hanyang University — Department of Information Systems  
**Supervisor**: Prof. Ook Lee (이욱 교수님)  
**Date**: June 2026  
**Status**: Pre-implementation planning document

---

## Table of Contents

1. [V1 vs V2 Comparison](#1-v1-vs-v2-comparison)
2. [Proposed V2 Architecture](#2-proposed-v2-architecture)
3. [Research Objectives](#3-research-objectives)
4. [New Research Questions (RQ8–RQ13)](#4-new-research-questions-rq8rq13)
5. [Experimental Design](#5-experimental-design)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Required Code Changes](#7-required-code-changes)
8. [Estimated Implementation Effort](#8-estimated-implementation-effort)
9. [Risks and Mitigation](#9-risks-and-mitigation)
10. [Implementation Order Recommendation](#10-implementation-order-recommendation)

---

## 1. V1 vs V2 Comparison

### 1.1 Architecture at a Glance

| Dimension | V1 (Adaptive Retrieval) | V2 (Agentic RAG) |
|---|---|---|
| **Control flow** | Linear, single-pass | Graph-structured, multi-step |
| **Feedback loop** | None | Validation-driven re-retrieval |
| **Query handling** | Atomic (one query → one retrieval) | Decomposable (multi-hop → sub-queries) |
| **Planner role** | Selects retrieval backend once | Replans based on validation signal |
| **Retriever role** | Called directly by planner | Exposed as MCP tools, invoked by agent |
| **Answer generation** | Not present (retrieval only) | Dedicated Answer Agent |
| **Orchestration** | Custom sequential runner | LangGraph state machine |
| **Tool abstraction** | None | MCP (Model Context Protocol) |
| **Observability** | JSON logs per run | Structured agent execution traces |
| **State** | Stateless across steps | Shared graph state (AgentState) |

### 1.2 What V1 Proved

V1 established four empirical facts that directly motivate V2:

1. **Architecture dominates planning**: Moving from BM25 to Hybrid RRF improved MRR by +15.8%, while the best planner improvement over Hybrid was only +2.0%. This means the retrieval *loop* (validation, re-retrieval) may matter more than the initial strategy choice.

2. **Single-pass failure is structural**: Exception (Recall@10 ≤ 0.412) and temporal queries (Recall@10 ≤ 0.200) fail across all five V1 systems. No static strategy selection fixes this — the queries require multi-step reasoning or targeted re-retrieval.

3. **Multi-hop queries are underserved**: Multi-hop Recall@10 ranges 0.491–0.538 in V1, the weakest non-exception category. These queries decompose naturally into sequential sub-retrievals that V1 cannot express.

4. **LLM planning adds calibration but not coverage**: GPT-5 achieves better MRR (+2.0%) but lower Recall@10 (–0.7%) than the rule planner. An LLM agent with a feedback loop could leverage both: use rule-based fast paths where unambiguous, fall back to LLM reasoning where uncertain.

### 1.3 What V1 Did Not Address

| Gap | V1 Limitation | V2 Resolution |
|---|---|---|
| Retrieval quality validation | Chunks returned without relevance checking | Validation Agent scores and filters chunks |
| Multi-step retrieval | One retrieval call per query | Query Analysis Agent decomposes; N retrieval calls |
| Answer synthesis | No answer generated | Answer Agent synthesizes from validated evidence |
| Tool modularity | Retrievers coupled directly to planner | Retrievers exposed as MCP-compliant tools |
| Empirical oracle labels | Oracle circularity with rule planner | V2 uses empirical best-strategy labels from V1 outputs |
| Execution transparency | Metrics only; no decision trace | Full agent trace per query |
| Re-retrieval on failure | No mechanism | Validation Agent triggers re-retrieval loop |

### 1.4 V1 Baseline Numbers (for V2 benchmarking)

| System | Recall@10 | MRR | nDCG@10 | SSA | Cost/query |
|---|---|---|---|---|---|
| BM25 | 0.713 | 0.457 | 0.509 | — | ~$0 |
| Dense | 0.740 | 0.501 | 0.548 | — | ~$0 |
| Hybrid RRF | 0.745 | 0.530 | 0.566 | — | ~$0 |
| Rule Planner | 0.755 | 0.522 | 0.563 | 78.4% | ~$0 |
| GPT-5 Planner | 0.750 | 0.533 | 0.570 | 77.6% | ~$0.016 |
| **V2 target** | **≥ 0.800** | **≥ 0.560** | **≥ 0.600** | — | TBD |

---

## 2. Proposed V2 Architecture

### 2.1 System Overview

```
Query
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│                    LangGraph Graph                       │
│                                                         │
│  ┌──────────────────┐                                   │
│  │  Query Analysis  │  Classifies reasoning type,       │
│  │      Agent       │  detects sub-queries, flags       │
│  │   (QA Agent)     │  exception/temporal patterns      │
│  └────────┬─────────┘                                   │
│           │  QueryPlan                                   │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │    Planning      │  Selects retrieval strategy       │
│  │      Agent       │  per sub-query; may invoke        │
│  │  (Plan Agent)    │  rule planner or LLM planner      │
│  └────────┬─────────┘                                   │
│           │  RetrievalPlan[]                            │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │   Retrieval      │  Executes strategy via MCP        │
│  │      Agent       │  tools (bm25_tool, dense_tool,    │
│  │  (Ret Agent)     │  hybrid_tool); merges results     │
│  └────────┬─────────┘                                   │
│           │  RetrievedChunks[]                          │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │   Validation     │  Scores chunk relevance;          │
│  │      Agent       │  passes threshold → Answer Agent  │
│  │  (Val Agent)     │  fails threshold → re-retrieval   │
│  └────────┬─────────┘   (max N=3 loops)                │
│           │  ValidatedEvidence                          │
│           ▼                                             │
│  ┌──────────────────┐                                   │
│  │     Answer       │  Synthesizes grounded answer;     │
│  │      Agent       │  produces citations, confidence   │
│  │  (Ans Agent)     │  score, and evidence trace        │
│  └──────────────────┘                                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
  │
  ▼
AgentTrace + Answer + EvidencePackage
```

### 2.2 Agent Specifications

#### Agent 1: Query Analysis Agent (QA Agent)

**Input**: Raw query string + query metadata (from SEKD ground truth)  
**Output**: `QueryPlan` — structured decomposition

**Responsibilities**:
- Classify reasoning type (single_hop / multi_hop / aggregation / comparison / exception / temporal) — can reuse V1 oracle labels for evaluation, but must derive classification autonomously in inference
- Detect multi-hop structure: identify entity chains that require sequential lookups
- Decompose multi-hop query into ordered sub-queries (e.g., Q → [Q1: find entity A, Q2: using A, find B])
- Flag high-difficulty markers (exception clauses, temporal references, cross-document dependency)
- Emit a priority signal: which sub-query should be retrieved first

**Key design decision**: QA Agent should NOT see the oracle reasoning_type during inference — it must classify from text alone. This creates a clean test of LLM query understanding vs. V1's metadata-aware rule system.

#### Agent 2: Planning Agent (Plan Agent)

**Input**: `QueryPlan` from QA Agent  
**Output**: `RetrievalPlan[]` — one plan per sub-query

**Responsibilities**:
- Select retrieval strategy for each sub-query
- For simple sub-queries: delegate to rule planner (zero cost, deterministic)
- For ambiguous sub-queries: delegate to LLM planner (GPT-5 structured prompt)
- Attach search parameters: top-k, reranking flag, field restrictions
- Record the planning rationale in structured form (for trace analysis)

**Key design decision**: Hybrid dispatch — rule planner handles ≥70% of queries (those matching R01–R06 exactly), LLM planner handles the remainder. This preserves V1's cost efficiency while adding flexibility.

#### Agent 3: Retrieval Agent (Ret Agent)

**Input**: `RetrievalPlan[]`  
**Output**: `RetrievedChunks[]` per sub-query

**Responsibilities**:
- Invoke MCP-exposed retrieval tools (`bm25_tool`, `dense_tool`, `hybrid_tool`)
- For multi-hop queries: execute sub-retrievals sequentially, passing entity values extracted from Q1 results into Q2 search
- Handle parallel retrieval for aggregation queries (multiple independent retrievals merged)
- Return raw results with source metadata intact (chunk_id, document_id, score, source)

**Key design decision**: MCP tool interface enforces a clean input/output contract. Tools are stateless — all state lives in the LangGraph `AgentState`. This enables future tool swapping (e.g., replacing `dense_tool` with a ColBERT tool) without touching orchestration logic.

#### Agent 4: Validation Agent (Val Agent)

**Input**: `RetrievedChunks[]` + original `QueryPlan`  
**Output**: `ValidatedEvidence` OR re-retrieval signal

**Responsibilities**:
- Score each chunk for relevance to the (sub-)query using a lightweight relevance classifier
- Apply a relevance threshold (θ, tunable; default θ = 0.5)
- If ≥ k_min chunks pass (default k_min = 3): forward to Answer Agent
- If < k_min chunks pass: emit `RETRIEVAL_FAILED` signal with failure reason
- On failure: modify the `RetrievalPlan` (broaden query, switch strategy, relax constraints) and trigger re-retrieval
- Maximum re-retrieval loops: N = 3 (prevents infinite cycles)
- Log validation decisions with scores for trace analysis

**Key design decision**: Validation is the core addition V1 lacks. The re-retrieval loop is the mechanism that directly addresses exception and temporal query failures identified in V1. The failure reason (too narrow, wrong strategy, missing entity) feeds back into Plan Agent on retry.

#### Agent 5: Answer Agent (Ans Agent)

**Input**: `ValidatedEvidence`  
**Output**: `AgentAnswer` — grounded answer with citations

**Responsibilities**:
- Synthesize a factual answer from validated evidence chunks
- Generate inline citations (chunk_id references)
- Produce a confidence score (0.0–1.0) based on evidence coverage
- Flag contradictions if multiple evidence chunks conflict
- Record final evidence package for evaluation (which chunks were used, which were ignored)

**Key design decision**: Answer quality is evaluated independently from retrieval quality. This decouples V2's two contribution layers: (1) retrieval improvement from the agentic loop, (2) answer quality from synthesis. V1 metrics (Recall@10, MRR, nDCG) measure layer 1; new V2 metrics (Answer Accuracy, Citation F1) measure layer 2.

### 2.3 LangGraph State Schema

```
AgentState {
    query_id:           str
    raw_query:          str
    query_metadata:     dict          # from SEKD (reasoning_type, category, difficulty)
    query_plan:         QueryPlan     # output of QA Agent
    retrieval_plans:    list[RetrievalPlan]
    retrieved_chunks:   list[RetrievedChunks]
    validation_result:  ValidationResult
    validated_evidence: ValidatedEvidence
    answer:             AgentAnswer
    trace:              AgentTrace    # timestamped list of all agent decisions
    loop_count:         int           # re-retrieval loop counter (max 3)
    error:              str | None
}
```

### 2.4 MCP Tool Architecture

The three V1 retrievers are wrapped as MCP-compliant tools:

```
MCP Server: enterprise-rag-retrieval
  ├── tool: bm25_search
  │     input:  {query: str, top_k: int, filters: dict}
  │     output: {chunks: list[Chunk], latency_ms: float}
  │
  ├── tool: dense_search
  │     input:  {query: str, top_k: int, model: str}
  │     output: {chunks: list[Chunk], latency_ms: float}
  │
  └── tool: hybrid_search
        input:  {query: str, top_k: int, rrf_k: int, alpha: float}
        output: {chunks: list[Chunk], latency_ms: float}
```

**Why MCP**: The MCP standard decouples tool implementation from agent orchestration. The same MCP server can be consumed by LangGraph agents, Claude Desktop, or any MCP-compatible client without changes to the tool implementation. This future-proofs the retrieval layer.

---

## 3. Research Objectives

### Primary Objective
Investigate whether a multi-agent agentic architecture with validation-driven re-retrieval and query decomposition measurably improves enterprise retrieval performance over V1's single-pass adaptive retrieval, while remaining economically viable for enterprise deployment.

### Secondary Objectives

**O1 — Close the exception/temporal gap**: Demonstrate that V2's query decomposition and re-retrieval loop can improve Recall@10 for exception queries (V1 ceiling: 0.412) and temporal queries (V1 ceiling: 0.200) by at least +10 pp.

**O2 — Multi-hop specialization**: Show that sequential sub-query retrieval improves multi-hop Recall@10 beyond V1's ceiling of 0.538.

**O3 — Validate the MCP tool layer**: Confirm that wrapping retrievers as MCP tools does not introduce meaningful latency overhead compared to direct invocation (target: < 50ms added per tool call).

**O4 — Characterize the cost-quality frontier**: Map the relationship between number of re-retrieval loops (N=1,2,3), answer quality, and API cost. Identify the economically optimal N for enterprise deployment.

**O5 — LangGraph orchestration overhead**: Quantify the latency and complexity cost of LangGraph graph execution vs. V1's sequential runner for equivalent single-pass queries.

---

## 4. New Research Questions (RQ8–RQ13)

### RQ8: Does multi-step agentic retrieval improve retrieval coverage over single-pass adaptive retrieval?

**Hypothesis**: For multi-hop and aggregation queries, agentic sequential sub-retrieval will achieve higher Recall@10 than V1's best system (Rule Planner, 0.755 overall; 0.538 multi-hop). Single-hop and comparison queries will show negligible difference, as they do not benefit from decomposition.

**Measurement**: Recall@10, MRR, nDCG@10 per reasoning type. Primary comparison: V2 vs. Rule Planner (V1 best). Secondary comparison: V2 vs. GPT-5 Planner.

**Expected outcome**: +10–20 pp improvement on multi-hop; < 2 pp change on single-hop/comparison.

### RQ9: Does validation-driven re-retrieval reduce the failure rate on exception and temporal queries?

**Hypothesis**: V1's failure on exception queries stems from the initial strategy selection being suboptimal (dense is selected, but chunks containing exception clause language are missed). Val Agent's re-retrieval with broadened queries or switched strategy will recover at least some of these failures.

**Measurement**: Per-reasoning-type Recall@10 at first retrieval vs. after re-retrieval (loop count breakdown: N=0, N=1, N=2, N=3). Also: Val Agent re-retrieval trigger rate by reasoning type.

**Expected outcome**: Exception Recall@10 from 0.412 to ≥ 0.520; temporal Recall@10 from 0.200 to ≥ 0.350.

### RQ10: Does query decomposition by the QA Agent accurately mirror the oracle reasoning type classification?

**Hypothesis**: The QA Agent, operating from query text alone (no oracle metadata), will achieve ≥ 80% agreement with V1 oracle reasoning type labels. Temporal and exception queries may show lower agreement due to surface-level ambiguity.

**Measurement**: QA Agent classification agreement rate (analogous to SSA in V1). Per-class precision/recall/F1. Confusion matrix across 6 reasoning types.

**Expected outcome**: ~82% overall agreement; temporal < 70%, exception < 75%, aggregation > 90%.

### RQ11: What is the cost and latency overhead of the V2 agentic pipeline compared to V1?

**Hypothesis**: V2 will incur substantially higher per-query cost than V1 due to multiple LLM calls per query. The cost-quality relationship will be non-linear: N=1 re-retrieval captures most benefit; N=2 and N=3 offer diminishing returns.

**Measurement**: End-to-end latency per query (p50, p95). Total API cost per 380-query evaluation. Cost breakdown by agent. Comparison with V1 GPT-5 cost ($6.18 / 380 queries = $0.016/query).

**Expected outcome**: V2 cost $0.08–$0.25 per query (5–15× V1 GPT-5). Most cost in Answer Agent and Val Agent.

### RQ12: Does MCP tool wrapping add measurable latency overhead vs. direct retrieval?

**Hypothesis**: MCP introduces a thin serialization/deserialization layer. For in-process MCP (same machine), overhead will be < 20ms per call. For out-of-process MCP (separate server process), overhead may reach 50–100ms.

**Measurement**: Retrieval latency with direct call vs. MCP tool call for BM25, Dense, Hybrid individually. 100 repeated calls each, reporting mean ± std.

**Expected outcome**: In-process overhead < 20ms. If overhead exceeds 50ms, in-process MCP is recommended over out-of-process.

### RQ13: What proportion of V2's retrieval improvement is attributable to each agent component?

**Hypothesis**: Validation-driven re-retrieval (Val Agent) contributes the largest share of improvement over V1, followed by query decomposition (QA Agent). Answer Agent contribution is orthogonal (not captured by retrieval metrics).

**Measurement**: Ablation study — run V2 with agents progressively removed:
- V2-full (all 5 agents)
- V2-noVal (remove Val Agent; no re-retrieval)
- V2-noDecomp (remove QA Agent decomposition; treat all queries as single-hop)
- V2-baseline (QA Agent + direct retrieval only; effectively V1 + answer generation)

Report Recall@10, MRR, nDCG@10 for each ablation. Attribution = (V2-full − ablated version).

---

## 5. Experimental Design

### 5.1 Evaluation Corpus

V2 uses the same SEKD corpus as V1, unchanged:
- 120 documents, 1,206 chunks
- 380 answerable queries (25 unanswerable excluded from retrieval evaluation)
- Ground truth: chunk-level relevance labels from V1 Phase 4

No new data generation is required. SEKD's 6 reasoning types remain the primary stratification variable.

### 5.2 Evaluation Modes

**Mode A — Retrieval-only mode** (comparable to V1):  
Stop after Val Agent; measure Recall@10, MRR, nDCG@10 against V1 ground truth. This produces a fair apples-to-apples comparison with V1.

**Mode B — End-to-end agentic mode**:  
Run all 5 agents; measure answer quality metrics (RQ8–RQ9 in addition to retrieval metrics). Requires reference answers — generated from ground truth evidence chunks using a fixed GPT-5 call.

**Mode C — Ablation mode** (for RQ13):  
Systematically disable individual agents; record metrics per configuration.

### 5.3 Experiment Matrix

| Experiment | Mode | Agents Active | RQs Addressed | N queries |
|---|---|---|---|---|
| E1: Full V2 vs V1 baseline | A | All 5 | RQ8, RQ11 | 380 |
| E2: Exception/temporal deep dive | A | All 5 | RQ9 | 76 (exception) + 5 (temporal) |
| E3: QA Agent classification | — | QA Agent only | RQ10 | 380 |
| E4: MCP latency benchmark | — | Ret Agent tools | RQ12 | 100 per retriever |
| E5: Re-retrieval loop analysis | A | All 5, N=1,2,3 | RQ9, RQ11 | 380 × 3 configs |
| E6: Ablation study | A | Progressive | RQ13 | 380 × 4 configs |
| E7: Answer quality | B | All 5 | RQ8 (answer) | 100 (sampled) |

### 5.4 Query Decomposition Protocol

For multi-hop queries, QA Agent produces a `QueryPlan` with ordered sub-queries:

```
Query: "Which API endpoints require the same authentication method used by
        the data export service documented in the system design spec?"

QueryPlan:
  sub_queries:
    1. "What authentication method does the data export service use?"
       → retrieve from: System Design docs
       → strategy hint: dense (semantic lookup)
    2. "Which API endpoints use {auth_method_from_q1}?"
       → retrieve from: API Documentation
       → entity slot: {auth_method_from_q1} filled from Q1 results
       → strategy hint: bm25 (exact term matching)
  merge_strategy: intersection_by_endpoint_id
```

Entity slot filling is performed by Retrieval Agent between sub-query executions. This is the key mechanism for improving multi-hop Recall@10.

### 5.5 Validation Thresholds

Val Agent uses a configurable relevance scoring approach:

**Option A (Lightweight)**: BM25 relevance score of retrieved chunk vs. query as proxy for relevance (no additional LLM call). Fast, deterministic, comparable to V1.

**Option B (LLM-based)**: Small LLM call (GPT-4o-mini) to score each chunk 0–1 for relevance. Higher accuracy, higher cost (~$0.001 per chunk scored).

**Recommended default**: Option A for retrieval-mode evaluation (cost-controlled); Option B for end-to-end evaluation. This keeps E1–E6 directly comparable to V1 costs.

---

## 6. Evaluation Methodology

### 6.1 Retrieval Metrics (inherited from V1, unchanged)

All V1 metrics are retained to ensure comparability:

| Metric | Formula | Primary Use |
|---|---|---|
| Recall@k | |relevant ∩ retrieved@k| / |relevant| | Coverage |
| MRR | 1/N Σ 1/rank(first relevant) | Ranking quality |
| nDCG@10 | Normalized DCG with graded relevance | Ranking with position weighting |
| Hit@k | 1 if any relevant in top-k | Binary coverage |

Primary comparison metric: **Recall@10** (same as V1 primary metric).

### 6.2 New V2 Metrics

**Agent-level metrics**:

| Metric | Definition | Measures |
|---|---|---|
| QA Classification Accuracy | Agreement with oracle reasoning_type | RQ10 |
| Decomposition Rate | % queries decomposed into ≥ 2 sub-queries | QA Agent behavior |
| Re-retrieval Rate | % queries triggering ≥ 1 re-retrieval | Val Agent sensitivity |
| Val Pass Rate | % queries passing validation on first attempt | Retrieval quality |
| Mean Loop Count | Average re-retrieval loops per query | Efficiency |

**Answer quality metrics** (Mode B only):

| Metric | Definition |
|---|---|
| Answer Accuracy | Exact/partial match of agent answer vs. reference answer |
| Citation F1 | Precision/recall of cited chunk_ids vs. ground truth evidence |
| Confidence Calibration | Correlation of agent confidence score with answer accuracy |
| Hallucination Rate | % answers containing claims unsupported by retrieved evidence |

**Cost and efficiency metrics**:

| Metric | Definition |
|---|---|
| Tokens per query | Total LLM tokens consumed across all agent calls |
| Cost per query (USD) | Sum of all LLM API costs per query |
| End-to-end latency (ms) | Wall-clock time from query input to agent answer output |
| Retrieval-only latency (ms) | Time to validated evidence (before Answer Agent) |

### 6.3 Statistical Analysis

V1 thesis noted the absence of statistical significance testing as a critical gap. V2 addresses this:

- **Bootstrap confidence intervals** (n=1000 resamples) for all primary metrics
- **Permutation tests** for V2 vs. V1 differences (H₀: no difference)
- **McNemar's test** for per-query correct/incorrect comparisons between V2 and V1 Rule Planner
- Report: point estimate ± 95% CI for all headline numbers

Minimum detectable effect at n=380, α=0.05, power=0.80: approximately ±2.5 pp for Recall@10. Differences below this threshold are reported as "statistically indistinguishable."

### 6.4 Empirical Oracle Labels for V2

V1's oracle label circularity (rule planner SSA evaluated against oracle it designed) is resolved in V2:

**Empirical best-strategy oracle**: For each query in SEKD, the oracle label is defined as the V1 retrieval method that achieved the highest Recall@10 for that query's reasoning type class. Derived from V1 Phase 5–7 evaluation outputs already in `outputs/evaluation/`.

```
temporal    → bm25   (empirical: BM25 Recall=0.200 > Hybrid=0.100 > Dense=0.000)
exception   → dense  (empirical: Dense best for semantic exception clauses)
multi_hop   → hybrid (empirical: Hybrid 0.538 > Dense 0.527 > BM25 0.491)
aggregation → hybrid (empirical: Hybrid 0.818)
comparison  → hybrid (empirical: Hybrid 0.894)
single_hop  → hybrid (empirical: Hybrid 0.832)
```

V2 Planning Agent SSA is evaluated against this empirical oracle — no circularity.

---

## 7. Required Code Changes

### 7.1 New Modules (V2-only, additive)

```
src/enterprise_rag/
├── agents/                          # NEW — entire directory
│   ├── __init__.py
│   ├── state.py                     # AgentState, QueryPlan, RetrievalPlan, etc.
│   ├── query_analysis.py            # QA Agent implementation
│   ├── planning.py                  # Plan Agent (wraps V1 rule_based + llm_based)
│   ├── retrieval.py                 # Ret Agent (invokes MCP tools)
│   ├── validation.py                # Val Agent + relevance scoring
│   └── answer.py                    # Answer Agent
│
├── graph/                           # NEW — LangGraph orchestration
│   ├── __init__.py
│   ├── nodes.py                     # One function per agent (graph nodes)
│   ├── edges.py                     # Conditional edges (re-retrieval loop logic)
│   └── builder.py                   # StateGraph construction + compilation
│
├── mcp/                             # NEW — MCP tool server
│   ├── __init__.py
│   ├── server.py                    # MCP server entry point
│   └── tools/
│       ├── bm25_tool.py             # bm25_search MCP tool
│       ├── dense_tool.py            # dense_search MCP tool
│       └── hybrid_tool.py           # hybrid_search MCP tool
│
└── v2/                              # NEW — V2 experiment runner
    ├── __init__.py
    ├── runner.py                    # Orchestrates full V2 evaluation
    └── evaluation.py                # V2-specific metrics (answer quality, agent metrics)
```

### 7.2 V1 Modules — No Modification Required

The following V1 modules are consumed by V2 as-is:

| Module | Role in V2 |
|---|---|
| `retrieval/bm25.py` | Wrapped by `mcp/tools/bm25_tool.py` |
| `retrieval/dense.py` | Wrapped by `mcp/tools/dense_tool.py` |
| `retrieval/hybrid.py` | Wrapped by `mcp/tools/hybrid_tool.py` |
| `planning/rule_based.py` | Called by V2 Plan Agent for deterministic cases |
| `planning/llm_based.py` | Called by V2 Plan Agent for ambiguous cases |
| `planning/schema.py` | `PlannerDecision`, `LLMDecision` reused by Plan Agent |
| `evaluation/retrieval_metrics.py` | All V1 metrics reused unchanged |
| `evaluation/planner_metrics.py` | SSA calculation reused for QA Agent evaluation |
| `dataset/` | SEKD corpus loading unchanged |
| `preprocessing/` | Chunk pipeline unchanged |

**V1 is fully preserved as a baseline**. V2 adds a new execution path that calls V1 components as libraries.

### 7.3 Configuration Changes

New config files (additive):

```
configs/
├── v2/
│   ├── agent_config.yaml            # Agent parameters (thresholds, max loops, etc.)
│   ├── mcp_config.yaml              # MCP server settings
│   └── evaluation_v2.yaml           # V2 evaluation settings (Mode A/B/C)
```

### 7.4 New Dependencies

```toml
# Add to pyproject.toml [project.dependencies]
"langgraph>=0.2,<1.0"              # LangGraph orchestration
"mcp>=1.0,<2.0"                    # Model Context Protocol SDK
"langchain-core>=0.3,<1.0"        # Required by LangGraph

# Add to [project.optional-dependencies] dev
"pytest-asyncio>=0.24,<1.0"       # Async test support for LangGraph
```

### 7.5 Scripts

```
scripts/
├── run_v2_evaluation.py             # Full V2 evaluation (all experiments)
├── run_v2_ablation.py               # Ablation study (E6)
├── run_mcp_latency_benchmark.py     # MCP overhead benchmark (E4)
└── generate_reference_answers.py   # Reference answers for Mode B (E7)
```

---

## 8. Estimated Implementation Effort

### 8.1 Effort by Component

| Component | Complexity | Estimated Hours | Notes |
|---|---|---|---|
| `agents/state.py` — AgentState schema | Low | 3h | Straightforward Pydantic models |
| `agents/query_analysis.py` — QA Agent | High | 12h | LLM-based query decomposition; prompt engineering |
| `agents/planning.py` — Plan Agent | Medium | 6h | Wraps V1 planners; adds dispatch logic |
| `agents/retrieval.py` — Ret Agent | Medium | 8h | Entity slot filling is non-trivial |
| `agents/validation.py` — Val Agent | Medium | 8h | Relevance scoring + re-retrieval trigger logic |
| `agents/answer.py` — Answer Agent | Medium | 6h | Answer synthesis + citation extraction |
| `graph/nodes.py`, `edges.py`, `builder.py` | High | 10h | LangGraph graph wiring + conditional edges |
| `mcp/server.py` + 3 tools | Low–Medium | 8h | MCP SDK wrapper around V1 retrievers |
| `v2/runner.py` + `evaluation.py` | Medium | 8h | Adapts V1 evaluation framework to V2 outputs |
| Experiment scripts (5 scripts) | Low | 6h | Thin wrappers around runner.py |
| Tests | Medium | 12h | Agent unit tests + integration tests |
| Prompt engineering (QA, Val, Ans) | High | 10h | Iterative; hardest to estimate |
| **Total** | | **~97 hours** | |

### 8.2 Phase Timeline

| Phase | Contents | Duration | Cumulative |
|---|---|---|---|
| **Phase 12A** — Infrastructure | AgentState, MCP tools, LangGraph skeleton | 1.5 weeks | 1.5 weeks |
| **Phase 12B** — Core agents | QA Agent, Plan Agent, Ret Agent | 2 weeks | 3.5 weeks |
| **Phase 12C** — Validation loop | Val Agent + re-retrieval edges | 1 week | 4.5 weeks |
| **Phase 12D** — Answer Agent | Answer synthesis + citation | 1 week | 5.5 weeks |
| **Phase 12E** — Evaluation | All 7 experiments, stats analysis | 1.5 weeks | 7 weeks |
| **Phase 12F** — Thesis extension | V2 thesis chapters, LaTeX | 1 week | 8 weeks |

**Total estimated calendar time**: 8 weeks at full-time research pace (~40h/week), or ~12 weeks at part-time (25h/week).

---

## 9. Risks and Mitigation

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **LangGraph API churn**: LangGraph is under active development; API changes may break graph wiring | Medium | High | Pin `langgraph==0.2.x` at project start; run tests against pinned version |
| **QA Agent misclassification cascade**: If QA Agent misclassifies query type, all downstream agents operate on a wrong decomposition | High | Medium | Add oracle-override mode for evaluation: run V2 with oracle reasoning_type injected into QA Agent state to isolate QA Agent error from downstream agent error |
| **Re-retrieval loop non-termination**: Val Agent triggers loop; modified plan still fails; infinite loop | Low | High | Hard cap at N=3 loops; loop_count tracked in AgentState; graph edge checks `loop_count < 3` before allowing re-retrieval edge |
| **MCP out-of-process latency**: MCP server as separate process adds >100ms per retrieval call | Medium | Medium | Default to in-process MCP (function call wrapping same-process); out-of-process only for RQ12 benchmarking |
| **Answer Agent hallucination**: LLM synthesizes claims not in retrieved evidence | Medium | Medium | Citation F1 metric detects this; Val Agent pre-filters chunks to validated evidence only, constraining Answer Agent input |
| **Cost overrun**: V2 per-query cost exceeds budget for 380-query evaluation | Medium | Low | Cap E7 (answer quality) at 100 sampled queries; E1–E6 use Mode A (retrieval only), minimizing LLM calls |

### 9.2 Research Risks

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **V2 fails to improve over V1**: Agentic loop overhead offsets retrieval gains | Medium | High | V2 is designed to extend, not replace, V1. Null result is a valid research finding. Report as "V2 overhead cost not justified by marginal retrieval improvement for SEKD's query distribution." |
| **Exception/temporal queries remain hard**: Val Agent cannot fix structural retrieval failures | High | Medium | Document as a boundary condition. Propose chunk-level annotation of exception clauses as future work. |
| **SEKD corpus too small for agent evaluation**: 380 queries insufficient to detect meaningful differences in per-agent attribution | Medium | Medium | Use bootstrap CIs with wide reporting windows. Flag sub-group results (e.g., n=5 temporal) as preliminary. |
| **Empirical oracle labels contradict V1 oracle**: New oracle may change SSA comparisons | Low | Low | Report both oracle systems; label the difference explicitly. Empirical oracle is more defensible. |

### 9.3 Scope Risk

The largest scope risk is feature creep in the Answer Agent. Generating high-quality answers requires significant prompt engineering and is not the core research contribution (retrieval improvement is). **Mitigation**: scope Answer Agent to Mode B only; all primary experiments use Mode A (retrieval-only). This keeps V2 focused on its central claim (agentic retrieval > single-pass retrieval) and prevents thesis scope expansion beyond what is defensible.

---

## 10. Implementation Order Recommendation

### Recommended Order: Infrastructure-First, Evaluation-Anchored

The recommended sequence builds the system incrementally with evaluation capability at each step, so partial V2 results are obtainable early.

#### Step 1: MCP Tool Layer (Week 1)
**Why first**: The MCP tools are thin wrappers around V1 retrievers. They can be built and benchmarked independently (RQ12 → Experiment E4) before any agent logic exists. This validates the tool interface and resolves RQ12 early — if MCP latency is prohibitive, the tool design changes before downstream agents are built.

**Deliverable**: `mcp/` module complete, E4 latency benchmark results available.

#### Step 2: AgentState + LangGraph Skeleton (Week 1–2)
**Why second**: The state schema is the contract between all agents. Defining it before writing agent code prevents interface mismatches. The LangGraph skeleton (nodes wired to stub functions) can be compiled and tested end-to-end with stubs, confirming graph topology is correct before any agent logic is written.

**Deliverable**: `agents/state.py`, `graph/builder.py` with stub nodes; graph compiles without error.

#### Step 3: Retrieval Agent (Week 2)
**Why third**: Ret Agent is the simplest real agent — it just calls MCP tools and returns chunks. With Ret Agent working, a minimal pipeline (stub QA + stub Plan + real Ret + stub Val + stub Ans) can run end-to-end on real SEKD queries. This produces retrieval results immediately, even before sophisticated agents are ready.

**Deliverable**: `agents/retrieval.py`; minimal pipeline produces chunks for sample queries.

#### Step 4: Planning Agent (Week 2–3)
**Why fourth**: Plan Agent wraps V1 planners (already tested). The dispatch logic (rule vs. LLM) is the only new code. With Plan + Ret working, V2 in "no-decomposition, no-validation" mode already has the same behavior as V1 but running through the LangGraph graph. This is the V2-baseline ablation condition (RQ13).

**Deliverable**: `agents/planning.py`; V2-baseline ablation runnable → first V2 numbers.

#### Step 5: Validation Agent (Week 3)
**Why fifth**: Val Agent is the core V2 innovation. Building it after Plan + Ret means it can be tested immediately on real retrieved chunks. The re-retrieval loop (LangGraph conditional edge) can be validated with failure injection (force Val Agent to reject all results) to confirm loop termination logic works.

**Deliverable**: `agents/validation.py`, `graph/edges.py`; re-retrieval loop functional; E5 loop analysis runnable.

#### Step 6: Query Analysis Agent (Week 3–4)
**Why sixth**: QA Agent is the highest-risk component (LLM-based, prompt-dependent, classification quality unknown). Building it after the rest of the pipeline is stable means its output can be tested immediately against downstream agents. QA Agent prompt engineering is iterative — building it last prevents rework of downstream agents if decomposition format changes.

**Deliverable**: `agents/query_analysis.py`; E3 QA classification accuracy runnable; full V2 pipeline functional.

#### Step 7: Full Evaluation Suite (Week 4–5)
**Why seventh**: With all agents complete, run all 7 experiments. Start with E3 (QA classification, fast) and E4 (MCP benchmark, already done) to warm up. Run E1 (full V2 vs V1), E5 (loop analysis), E6 (ablation). Reserve E7 (answer quality) for last — it requires Answer Agent (Step 8) and reference answers.

**Deliverable**: E1–E6 results; primary RQ8–RQ13 answers (except RQ answer quality component).

#### Step 8: Answer Agent (Week 5–6)
**Why last**: Answer Agent is optional for retrieval evaluation (Mode A). Implementing it last ensures the main research results are in hand before spending effort on synthesis. If time is short, Answer Agent can be deferred entirely — V2's core contribution is the retrieval improvement, not answer generation.

**Deliverable**: `agents/answer.py`; E7 answer quality evaluation; Citation F1, hallucination rate.

#### Step 9: Thesis Extension Chapters (Week 7–8)
Write V2 methodology, results, and discussion chapters as an extension of the V1 thesis. The V1 thesis manuscript remains unchanged. V2 chapters are authored separately and can be presented as a conference paper or journal extension.

---

### Quick-Start Decision

If time is severely constrained (< 4 weeks), the minimum viable V2 delivers:

1. MCP tool layer (Step 1)
2. LangGraph skeleton (Step 2)
3. Ret Agent + Plan Agent (Steps 3–4)
4. Val Agent without re-retrieval (just filtering; Step 5 partial)
5. E1 (V2 vs V1) and E3 (QA classification) only

This still answers RQ8 (core question) and RQ12 (MCP overhead), and produces publishable results comparing agentic pipeline to V1 baseline.

---

## Appendix A: Key V1 Source References for V2 Integration

| V2 Agent | V1 Source File | Interface Used |
|---|---|---|
| Plan Agent | `src/enterprise_rag/planning/rule_based.py` | `RuleBasedPlanner.plan(query)` |
| Plan Agent | `src/enterprise_rag/planning/llm_based.py` | `LLMBasedPlanner.plan(query, metadata)` |
| Plan Agent | `src/enterprise_rag/planning/schema.py` | `PlannerDecision`, `LLMDecision` |
| Ret Agent / bm25_tool | `src/enterprise_rag/retrieval/bm25.py` | `BM25Retriever.retrieve(query, k)` |
| Ret Agent / dense_tool | `src/enterprise_rag/retrieval/dense.py` | `DenseRetriever.retrieve(query, k)` |
| Ret Agent / hybrid_tool | `src/enterprise_rag/retrieval/hybrid.py` | `HybridRetriever.retrieve(query, k)` |
| V2 evaluation | `src/enterprise_rag/evaluation/retrieval_metrics.py` | `compute_recall_at_k()`, `compute_mrr()`, `compute_ndcg()` |
| V2 evaluation | `src/enterprise_rag/evaluation/planner_metrics.py` | `compute_planner_metrics()` |
| All agents | `data/sekd/processed/` | Chunk corpus (1,206 chunks) |
| All agents | `data/sekd/raw/` | Document corpus (120 docs, 405 queries) |

## Appendix B: V1 Weakness → V2 Resolution Mapping

| V1 Weakness (from thesis review) | V2 Resolution |
|---|---|
| Oracle circularity in SSA | Empirical oracle labels derived from V1 Phase 5–7 outputs (Section 6.4) |
| No statistical significance testing | Bootstrap CIs + permutation tests required for all primary results (Section 6.3) |
| n=5 temporal queries | Temporal queries remain n=5 in SEKD; flag all temporal conclusions as preliminary |
| Single embedding model only | V2 uses same BGE-small model; ColBERT/SPLADE comparison deferred to future work |
| No answer generation | Answer Agent in Mode B (Section 5.2) |
| Single-pass; no feedback loop | Val Agent + conditional re-retrieval edge (core V2 contribution) |
