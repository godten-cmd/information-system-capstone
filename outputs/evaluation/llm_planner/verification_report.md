# Phase 9 LLM Planner Verification Report

Generated: 2026-06-02

Scope: local file audit only. No external APIs were called during this verification.

## Verdict

Phase 9 was run with the mock planner provider, not a real OpenAI or Anthropic provider.

- Actual provider used: `MockProvider`
- Provider recorded in run artifacts: `mock`
- Model recorded in run artifacts: `mock`
- SSA / planner accuracy: `0.8211` (`82.11%`)
- Source of SSA result: simulated mock outputs, not real LLM outputs

## Evidence Summary

### Run Manifest

`outputs/evaluation/llm_planner/run_manifest.json` records:

- `provider`: `mock`
- `model`: `mock`
- `prompt_version`: `structured`
- `run_prompt_comparison`: `true`
- `prompt_versions_run`: `zero_shot`, `structured`, `few_shot`
- `n_queries`: `380`
- `total_wall_clock_s`: `0.06`
- `total_cost_usd`: `0.0`
- `cache_dir`: `outputs/cache/planner`

Relevant lines:

- `outputs/evaluation/llm_planner/run_manifest.json:4` -> `"provider": "mock"`
- `outputs/evaluation/llm_planner/run_manifest.json:5` -> `"model": "mock"`
- `outputs/evaluation/llm_planner/run_manifest.json:6` -> `"prompt_version": "structured"`
- `outputs/evaluation/llm_planner/run_manifest.json:14` -> `"total_wall_clock_s": 0.06`
- `outputs/evaluation/llm_planner/run_manifest.json:18` -> `"total_cost_usd": 0.0`

This is incompatible with a real API-backed run over 380 queries unless every call was cached or bypassed. The cache and decision evidence below show it was bypassed/simulated via mock records.

### Planner Metrics

`outputs/evaluation/llm_planner/planner_metrics.json` records:

- `planner_type`: `llm_based`
- `prompt_version`: `structured`
- `model`: `mock`
- `n_queries`: `380`
- `n_correct`: `312`
- `accuracy`: `0.8211`
- `parse_failures`: `0`
- `fallback_count`: `0`

Relevant lines:

- `outputs/evaluation/llm_planner/planner_metrics.json:4` -> `"model": "mock"`
- `outputs/evaluation/llm_planner/planner_metrics.json:9` -> `"n_queries": 380`
- `outputs/evaluation/llm_planner/planner_metrics.json:10` -> `"n_correct": 312`
- `outputs/evaluation/llm_planner/planner_metrics.json:12` -> `"accuracy": 0.8211`

The reported SSA is therefore 312 / 380 = 82.11%.

### Prompt Comparison

`outputs/evaluation/llm_planner/prompt_comparison.csv` shows the selected/best prompt version was `structured`:

```csv
prompt_version,n_queries,n_correct,accuracy,...,total_cost_usd
zero_shot,380,264,0.6947,...,0.0
structured,380,312,0.8211,...,0.0
few_shot,380,311,0.8184,...,0.0
```

Evidence:

- `outputs/evaluation/llm_planner/prompt_comparison.csv:2` -> `zero_shot` accuracy `0.6947`
- `outputs/evaluation/llm_planner/prompt_comparison.csv:3` -> `structured` accuracy `0.8211`
- `outputs/evaluation/llm_planner/prompt_comparison.csv:4` -> `few_shot` accuracy `0.8184`

All prompt versions have `total_cost_usd` of `0.0`.

### Cost And Latency

`outputs/evaluation/llm_planner/cost_analysis.json` records each prompt version with:

- `model`: `mock`
- `n_queries`: `380`
- mean latency: `0.1` ms
- total cost: `0.0`

Relevant lines:

- `outputs/evaluation/llm_planner/cost_analysis.json:16` -> zero-shot model `mock`
- `outputs/evaluation/llm_planner/cost_analysis.json:31` -> zero-shot total cost `0.0`
- `outputs/evaluation/llm_planner/cost_analysis.json:37` -> structured model `mock`
- `outputs/evaluation/llm_planner/cost_analysis.json:52` -> structured total cost `0.0`
- `outputs/evaluation/llm_planner/cost_analysis.json:58` -> few-shot model `mock`
- `outputs/evaluation/llm_planner/cost_analysis.json:73` -> few-shot total cost `0.0`
- `outputs/evaluation/llm_planner/cost_analysis.json:80` states mock latency is near-zero and real API latency is expected to be 200-2000 ms per query.

This supports simulated output, not real LLM output.

### Planner Decisions

`outputs/evaluation/llm_planner/llm_decisions.jsonl` contains 380 decision rows.

Every inspected row records:

- `prompt_version`: `structured`
- `model`: `mock`
- synthetic JSON in `raw_response`
- fixed token counts (`prompt_tokens`: `320`, `completion_tokens`: `55`)
- `latency_s`: `0.0001`
- `estimated_cost_usd`: `0.0`

Examples:

- `outputs/evaluation/llm_planner/llm_decisions.jsonl:1` -> model `mock`, latency `0.0001`, estimated cost `0.0`
- `outputs/evaluation/llm_planner/llm_decisions.jsonl:2` -> model `mock`, latency `0.0001`, estimated cost `0.0`
- `outputs/evaluation/llm_planner/llm_decisions.jsonl:3` -> model `mock`, latency `0.0001`, estimated cost `0.0`

The decision file has exactly 380 rows, matching the manifest and metrics.

### Cache Files

The cache directory is `outputs/cache/planner`.

Observed cache inventory:

- Total planner cache files: `1140`
- Provider/model segment in filenames: `1140 mock`
- Prompt versions:
  - `380 zero_shot`
  - `380 structured`
  - `380 few_shot`

This exactly matches 380 queries times 3 prompt versions.

Example cache file: `outputs/cache/planner/Q-000001__mock__structured.json`

Relevant lines:

- `outputs/cache/planner/Q-000001__mock__structured.json:2` -> cache key `Q-000001__mock__structured`
- `outputs/cache/planner/Q-000001__mock__structured.json:4` -> `"model": "mock"`
- `outputs/cache/planner/Q-000001__mock__structured.json:12` -> `"prompt_tokens": 320`
- `outputs/cache/planner/Q-000001__mock__structured.json:13` -> `"completion_tokens": 55`
- `outputs/cache/planner/Q-000001__mock__structured.json:14` -> `"latency_s": 0.0001`

No OpenAI or Anthropic cache filename pattern was observed in `outputs/cache/planner`.

### Provider Implementation

`src/enterprise_rag/planning/providers/factory.py` defines provider selection:

- `openai` -> `OpenAIProvider`
- `anthropic` -> `AnthropicProvider`
- otherwise -> `MockProvider`

Relevant lines:

- `src/enterprise_rag/planning/providers/factory.py:31-33` create `OpenAIProvider` only for `resolved == "openai"`
- `src/enterprise_rag/planning/providers/factory.py:35-37` create `AnthropicProvider` only for `resolved == "anthropic"`
- `src/enterprise_rag/planning/providers/factory.py:39-40` otherwise returns `MockProvider`
- `src/enterprise_rag/planning/providers/factory.py:43-48` auto-detects OpenAI/Anthropic API keys and otherwise returns `mock`

`src/enterprise_rag/planning/providers/mock_provider.py` explicitly identifies itself as simulated:

- `src/enterprise_rag/planning/providers/mock_provider.py:1-9` says it is a deterministic mock provider that simulates realistic LLM planner behavior.
- `src/enterprise_rag/planning/providers/mock_provider.py:22-27` defines prompt-version error rates including `structured: 0.18`.
- `src/enterprise_rag/planning/providers/mock_provider.py:103-115` defines `MockProvider`, with provider/model names returning `mock`.
- `src/enterprise_rag/planning/providers/mock_provider.py:143-149` returns a `ProviderResponse` with model `mock` and synthetic token/latency values.

`src/enterprise_rag/planning/llm_based.py` short-circuits real LLM calls for the mock provider:

- `src/enterprise_rag/planning/llm_based.py:147-160` checks `isinstance(self._provider, MockProvider)` and calls `mock_decide(query, self._prompt_version)` instead of provider completion.
- `src/enterprise_rag/planning/llm_based.py:161-199` is the separate real-LLM path, used only when the provider is not `MockProvider`.
- `src/enterprise_rag/planning/llm_based.py:204-222` writes the resulting synthetic response into the planner cache.

This is decisive evidence that the SSA was produced from simulated mock decisions.

### OpenAI And Anthropic Providers Were Not The Run Provider

The codebase contains OpenAI and Anthropic provider implementations, but the Phase 9 artifacts do not show either was used.

OpenAI provider evidence:

- `src/enterprise_rag/planning/providers/openai_provider.py:10` default model is `gpt-4o-mini`
- `src/enterprise_rag/planning/providers/openai_provider.py:30-35` provider/model names would be `openai` and the resolved OpenAI model
- `src/enterprise_rag/planning/providers/openai_provider.py:45-64` would call the OpenAI Chat Completions API and return the API response model

Anthropic provider evidence:

- `src/enterprise_rag/planning/providers/anthropic_provider.py:10` default model is `claude-haiku-4-5-20251001`
- `src/enterprise_rag/planning/providers/anthropic_provider.py:32-38` provider/model names would be `anthropic` and the resolved Anthropic model
- `src/enterprise_rag/planning/providers/anthropic_provider.py:48-65` would call the Anthropic Messages API and return the API response model

The Phase 9 run manifest, metrics, decisions, cost analysis, and cache all record `mock`, not either real provider or real model.

### Logs

No dedicated `.log` files were found outside virtualenv/cache directories. Local run evidence is limited to durable output artifacts:

- `outputs/evaluation/llm_planner/run_manifest.json`
- `outputs/evaluation/llm_planner/planner_metrics.json`
- `outputs/evaluation/llm_planner/prompt_comparison.csv`
- `outputs/evaluation/llm_planner/cost_analysis.json`
- `outputs/evaluation/llm_planner/llm_decisions.jsonl`
- `outputs/cache/planner/*.json`

## Final Determination

SSA `82.11%` was not produced from real OpenAI or Anthropic LLM outputs.

It was produced from deterministic simulated outputs generated by `MockProvider` / `mock_decide`, with the `structured` prompt version selected as best among three simulated prompt runs.
