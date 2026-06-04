# GPT-5 Parsing Fix — Mini-Pilot Report

Generated: 2026-06-02T15:10:38.361441+00:00

## Root Cause

GPT-5 is a reasoning model. `max_completion_tokens=200` was the combined budget for
internal reasoning tokens AND visible output tokens. GPT-5's reasoning process consumed
all 200 tokens, leaving zero for visible output — so `message.content` was always `""`.

**Evidence from debug run** (`scripts/debug_gpt5_response.py`):

| Call | `max_completion_tokens` | `reasoning_tokens` | Output content |
|---|---|---|---|
| Chat Completions (original) | 200 | **200 (exhausted)** | `""` (empty) |
| Chat Completions (patched) | 2000 | ~384 | valid JSON ✓ |

## Fix Applied

File: `src/enterprise_rag/planning/providers/openai_provider.py`

- Added `_is_reasoning_model()` helper that matches `gpt-5`, `o1`, `o3`, `o4` prefixes.
- For reasoning models: `max_completion_tokens = max(requested_tokens + 2000, 2000)`.
  This guarantees sufficient headroom for reasoning (~400 tokens) plus full JSON output.
- Added warning log when content is empty so future regressions are surfaced immediately.
- Non-reasoning models (gpt-4o, gpt-4o-mini, etc.) are unchanged.

## Mini-Pilot Results

- Provider: `OpenAIProvider`
- Model: `gpt-5`
- Prompt version: `structured`
- Sample size: `5` random queries (seed=42)
- API success rate: `100.0%` (5/5)
- JSON parse success rate: `100.0%` (5/5)
- Strategy accuracy vs oracle: `80.0%` (4/5)
- Average latency: `6.74s`
- Latency range: `3.95s` – `10.55s`
- Total prompt tokens: `1785`
- Total completion tokens: `1659`
- Estimated cost: `$0.084210`

## Per-Query Details

| Query ID | Parse | Strategy | Oracle | Correct | Latency | Tokens | Raw (first 120 chars) |
|---|---|---|---|---|---|---|---|
| Q-000348 | ✓ | dense | dense | ✓ | 5.50s | 350+249 | '{"selected_strategy":"dense","confidence":0.84,"reasoning":"This is an exception-focused HR policy question about eligib' |
| Q-000060 | ✓ | hybrid | dense | ✗ | 10.55s | 359+648 | '{"selected_strategy":"hybrid","confidence":0.75,"reasoning":"Exact endpoint tokens (POST /api/v1/streams) benefit from B' |
| Q-000013 | ✓ | dense | dense | ✓ | 6.58s | 350+191 | '{"selected_strategy":"dense","confidence":0.84,"reasoning":"Exception clauses about eligibility are often paraphrased an' |
| Q-000405 | ✓ | hybrid | hybrid | ✓ | 3.95s | 374+189 | '{"selected_strategy":"hybrid","confidence":0.9,"reasoning":"This requires multi-hop retrieval using lexical IDs/dates to' |
| Q-000147 | ✓ | hybrid | hybrid | ✓ | 7.10s | 352+382 | '{"selected_strategy":"hybrid","confidence":0.84,"reasoning":"Status code lookup in API docs is typically table-based; hy' |

## Verdict

**READY for full 380-query evaluation**

Parsing is now reliable. The full 380-query evaluation can proceed using:

```bash
python scripts/run_llm_planner.py --provider openai --model gpt-5 --prompt-version structured
```