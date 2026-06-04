# Enterprise RAG Research Prototype

Research prototype for:

Agentic RAG 기반 기업 지식관리시스템 설계 및 평가: 문서 유형에 따른 Adaptive Retrieval 전략을 중심으로

This repository is currently at Phase 1: Repository Setup. It provides only the project foundation, configuration files, typed package skeleton, schema placeholders, and a minimal CLI entrypoint. Dataset generation, chunking, retrieval, planner logic, and experiment execution are intentionally deferred to later phases.

## Source Documents

- `PROJECT_SPEC.md`
- `DATASET_SPEC.md`
- `IMPLEMENTATION_PLAN.md`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Phase 1 Verification

```bash
pytest
ruff check .
enterprise-rag version
enterprise-rag status
```

## Intended Command Sequence

Later phases should use this high-level flow:

```bash
enterprise-rag dataset generate-documents --config configs/dataset.yaml
enterprise-rag dataset build-chunks --config configs/dataset.yaml
enterprise-rag dataset generate-queries --config configs/dataset.yaml
enterprise-rag retrieval run --config configs/experiment.yaml
enterprise-rag planning run --config configs/experiment.yaml
enterprise-rag evaluation run --config configs/experiment.yaml
enterprise-rag experiments run --config configs/experiment.yaml
```

The commands above are roadmap placeholders. Phase 1 only implements `version` and `status`.

## Reproducibility Conventions

- All later generated artifacts must be reproducible from a config file and seed.
- Experiment outputs should write a run manifest.
- Large generated data, indexes, and outputs should not be committed.
- Oracle labels and planner outputs must preserve provenance.

