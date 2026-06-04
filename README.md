# Adaptive Retrieval Planning and Agentic Re-retrieval for Enterprise Knowledge Management

**한양대학교 정보시스템학과 캡스톤 디자인 (2026)**

> Agentic RAG 기반 기업 지식관리시스템 설계 및 평가: 문서 유형에 따른 Adaptive Retrieval 전략을 중심으로

---

## Overview

This repository contains the full experimental code, dataset, evaluation results, and thesis artifacts for a capstone research project investigating adaptive retrieval planning and agentic re-retrieval for enterprise knowledge corpora.

The study proceeds in two phases:

- **V1** — Controlled comparison of five retrieval systems on the SEKD benchmark
- **V2** — A five-agent LangGraph pipeline with iterative validation and re-retrieval

### Key Results

| System | Recall@10 | MRR | nDCG@10 |
|---|---|---|---|
| BM25 | 0.713 | 0.457 | 0.509 |
| Dense (BGE-small-en-v1.5) | 0.740 | 0.501 | 0.548 |
| Hybrid RRF | 0.745 | 0.530 | 0.566 |
| **Rule Planner** | **0.755** | 0.522 | 0.563 |
| GPT-5 Planner | 0.750 | **0.533** | **0.570** |
| V2 E1 (Oracle + Adaptive + Loop) | 0.743 | 0.529 | 0.566 |

**Main findings:**
1. Hybrid RRF delivers the largest single gain (+15.8% MRR over BM25), 8× larger than the best planner improvement
2. A 7-rule deterministic planner achieves the highest Recall@10 at zero operational cost
3. GPT-5 offers +2.0% MRR at $0.016/query but fails completely on temporal queries (0% SSA)
4. The V2 re-retrieval loop fires for only 5/380 queries (98.7% first-pass rate) — validation thresholds are too permissive to trigger corrective re-retrieval on hard query types

---

## Dataset: SEKD

The **Synthetic Enterprise Knowledge Dataset** is a fully annotated retrieval benchmark for enterprise corpora.

| Property | Value |
|---|---|
| Documents | 120 (6 categories × 20) |
| Chunks | 1,206 (section-aware, 3-tier strategy) |
| Queries | 405 (380 answerable + 25 unanswerable) |
| Reasoning types | 6 (single_hop, multi_hop, aggregation, comparison, exception, temporal) |
| Difficulty factors | 8 |
| Oracle strategy labels | BM25 (62), Dense (69), Hybrid (249) |

**Document categories:** API Documentation · HR Policies · Project Meeting Notes · Security Policies · System Design Documents · Travel Policies

Dataset files are in `data/sekd/`:
- `raw/documents.jsonl` — 120 generated documents
- `processed/chunks.jsonl` — 1,206 section-aware chunks
- `queries.jsonl` — 405 annotated queries
- `ground_truth.jsonl` — chunk-level ground truth per query

---

## Repository Structure

```
.
├── src/enterprise_rag/
│   ├── dataset/          # SEKD corpus generation (Jinja2 templates, schemas)
│   ├── preprocessing/    # Chunking pipeline (3-tier section-aware strategy)
│   ├── retrieval/        # BM25, Dense (FAISS), Hybrid RRF backends
│   ├── planning/         # Rule-based planner (7 rules) + GPT-5 LLM planner
│   ├── evaluation/       # Metrics (Recall@K, MRR, nDCG, Hit@K), reporters
│   ├── agents/           # V2: QAA, PA, RA, VA, AA agent implementations
│   ├── graph/            # V2: LangGraph StateGraph builder, edges, nodes
│   └── mcp/              # MCP tool wrappers for retrieval backends
├── scripts/              # One-shot evaluation runners
├── configs/              # YAML configs for retrieval, planners, V2 agents
├── tests/                # 200+ unit + integration tests
├── data/sekd/            # SEKD corpus and annotations
├── outputs/
│   ├── evaluation/       # Metric results for all 9 systems (JSON/CSV)
│   ├── thesis_latex/     # English thesis (75pp) + journal paper (26pp)
│   ├── thesis_latex_korean/ # Korean thesis (68pp) + Korean paper (18pp)
│   └── final_report/     # Chapter drafts, figures, tables
└── .gitignore            # Excludes: .venv, FAISS indexes, LLM cache, LaTeX artifacts
```

---

## Setup

**Requirements:** Python 3.14, pdflatex (for thesis), xelatex + kotex (for Korean thesis)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # add OPENAI_API_KEY if running GPT-5 planner
```

---

## Reproducing the Experiments

All evaluation outputs are already committed under `outputs/evaluation/`. To reproduce from scratch:

```bash
# 1. Generate corpus
python scripts/generate_documents.py
python scripts/build_chunks.py
python scripts/generate_queries.py

# 2. V1 baselines
python scripts/run_bm25.py
python scripts/run_dense.py
python scripts/run_hybrid.py
python scripts/run_rule_planner.py
python scripts/run_llm_planner.py   # requires OPENAI_API_KEY, costs ~$6

# 3. V2 agentic RAG (4 ablation experiments)
python scripts/run_v2_eval.py
```

Evaluation results are written to `outputs/evaluation/<system>/`.

---

## V2 Agentic RAG Architecture

A five-agent [LangGraph](https://github.com/langchain-ai/langgraph) pipeline:

```
QAA → PA → RA → VA ──(pass)──→ AA → end
               ↑                |
               └────(fail, loop < 3)─┘
```

| Agent | Role |
|---|---|
| QAA (Query Analysis) | Classifies reasoning type; oracle / heuristic / LLM modes |
| PA (Planning) | Maps sub-queries to retrieval strategy via V1 rules + loop feedback |
| RA (Retrieval) | Executes BM25 / Dense / Hybrid via MCP tools; sequential/parallel modes |
| VA (Validation) | Scores chunks against per-type thresholds; diagnoses failure on rejection |
| AA (Answer) | Synthesises response from validated evidence; extractive fallback |

**Four ablation experiments:**

| Exp | QA Mode | Scorer | Max Loops | Purpose |
|---|---|---|---|---|
| E1 | Oracle | Adaptive | 3 | Primary V2 configuration (RQ8) |
| E2 | Oracle | Adaptive | 0 | Loop contribution (RQ9) |
| E3 | Oracle | Rank-proxy | 3 | Scorer quality (RQ10) |
| E4 | Heuristic | Adaptive | 3 | Classification error cost (RQ11) |

---

## Thesis & Paper

| Artifact | File | Pages |
|---|---|---|
| English thesis | `outputs/thesis_latex/thesis.pdf` | 75 |
| Korean thesis | `outputs/thesis_latex_korean/thesis_ko.pdf` | 68 |
| English journal paper | `outputs/thesis_latex/paper.pdf` | 26 |
| Korean journal paper | `outputs/thesis_latex/paper_ko.pdf` | 18 |

Compile commands:
```bash
# English thesis / paper
cd outputs/thesis_latex
pdflatex thesis.tex && bibtex thesis && pdflatex thesis.tex && pdflatex thesis.tex
pdflatex paper.tex  && bibtex paper  && pdflatex paper.tex  && pdflatex paper.tex

# Korean thesis / paper (requires kotex + AppleMyungjo)
cd outputs/thesis_latex_korean
xelatex thesis_ko.tex && bibtex thesis_ko && xelatex thesis_ko.tex && xelatex thesis_ko.tex
cd ../thesis_latex
xelatex paper_ko.tex  && bibtex paper_ko  && xelatex paper_ko.tex  && xelatex paper_ko.tex
```

---

## Tests

```bash
pytest                    # all tests
pytest tests/v2/          # V2 agent tests only
ruff check .              # linting
```

---

## Author

Won Young Shin · Department of Information Systems, Hanyang University · `pingu090@hanyang.ac.kr`
