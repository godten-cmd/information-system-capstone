# Real OpenAI GPT-5 Pilot Evaluation

Generated: 2026-06-02T15:01:17.978108+00:00

Scope: 20 random answerable queries, real `OpenAIProvider`, model `gpt-5`. No full 380-query planner evaluation was run.

## Result

- Provider used: `openai` (`OpenAIProvider`)
- Model used: `gpt-5`
- MockProvider used: `no`
- Prompt version: `structured`
- Sample size: `20` random queries
- Sample seed: `2915548840`
- API success rate: `100.00%` (20/20)
- JSON parse success rate: `0.00%` (0/20)
- Fallback count: `20`
- Average latency: `10.316s`
- Median latency: `9.444s`
- Latency range: `7.746s` - `16.558s`
- Total prompt tokens: `24207`
- Total completion tokens: `12000`
- Total tokens: `36207`
- Estimated cost: `$0.722070`
- Sample strategy-selection accuracy vs oracle: `55.00%`

## JSON Parsing Reliability

All 20 GPT-5 API calls returned usage metadata and were cached, but the planner received empty visible `raw_response` content under the current `max_completion_tokens=200` budget. Each query exhausted 3 parse attempts and fell back to `hybrid`.

This means the API path works, but JSON parsing reliability for the current GPT-5 provider settings is `0.00%`. The full run should not proceed until the provider/prompt settings are adjusted and another pilot reaches reliable JSON output.

## Cache Verification

- Cache directory: `outputs/cache/planner_openai_pilot_gpt5/20260602T145231Z`
- Cache files created: `20`
- Expected cache files: `20`
- Cached read decisions: `20`
- Cached read errors: `0`
- Cache hit rate: `100.00%` (20/20)
- Cached read elapsed time: `0.0010s`

## Query Sample

- `Q-000177`
- `Q-000078`
- `Q-000099`
- `Q-000013`
- `Q-000090`
- `Q-000366`
- `Q-000155`
- `Q-000307`
- `Q-000127`
- `Q-000243`
- `Q-000392`
- `Q-000058`
- `Q-000266`
- `Q-000197`
- `Q-000325`
- `Q-000171`
- `Q-000101`
- `Q-000005`
- `Q-000010`
- `Q-000068`

## Sample Planner Outputs

### Q-000177
- Selected strategy: `hybrid`
- Confidence: `0.5`
- Parse success: `False` after `3` attempt(s)
- Latency: `9.367s`
- Tokens: `1191` prompt + `600` completion
- Estimated cost: `$0.03591000`
- Parsed reasoning: Parse failure after 3 attempts; fell back to hybrid
- Raw response:
```json
<empty string>
```

### Q-000078
- Selected strategy: `hybrid`
- Confidence: `0.5`
- Parse success: `False` after `3` attempt(s)
- Latency: `9.3269s`
- Tokens: `1197` prompt + `600` completion
- Estimated cost: `$0.03597000`
- Parsed reasoning: Parse failure after 3 attempts; fell back to hybrid
- Raw response:
```json
<empty string>
```

### Q-000099
- Selected strategy: `hybrid`
- Confidence: `0.5`
- Parse success: `False` after `3` attempt(s)
- Latency: `9.2713s`
- Tokens: `1200` prompt + `600` completion
- Estimated cost: `$0.03600000`
- Parsed reasoning: Parse failure after 3 attempts; fell back to hybrid
- Raw response:
```json
<empty string>
```

### Q-000013
- Selected strategy: `hybrid`
- Confidence: `0.5`
- Parse success: `False` after `3` attempt(s)
- Latency: `9.2946s`
- Tokens: `1191` prompt + `600` completion
- Estimated cost: `$0.03591000`
- Parsed reasoning: Parse failure after 3 attempts; fell back to hybrid
- Raw response:
```json
<empty string>
```

### Q-000090
- Selected strategy: `hybrid`
- Confidence: `0.5`
- Parse success: `False` after `3` attempt(s)
- Latency: `7.7837s`
- Tokens: `1209` prompt + `600` completion
- Estimated cost: `$0.03609000`
- Parsed reasoning: Parse failure after 3 attempts; fell back to hybrid
- Raw response:
```json
<empty string>
```

## Stored Artifacts

- Planner decisions: `outputs/evaluation/llm_planner/pilot_decisions.jsonl`
- Pilot summary JSON: `outputs/evaluation/llm_planner/pilot_summary.json`
- Sample metadata: `outputs/evaluation/llm_planner/pilot_sample.json`
- Cache records: `outputs/cache/planner_openai_pilot_gpt5/20260602T145231Z`

## Recommendation

Do not proceed with the full 380-query run yet.
Reason: API success and cache verification passed, but JSON parse success was 0/20. Before a full 380-query run, increase the GPT-5 completion budget and/or use a GPT-5-compatible structured-output path so the model emits visible JSON reliably.
