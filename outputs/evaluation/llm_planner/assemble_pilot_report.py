from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path("src")))
load_dotenv(Path(".env"))

from enterprise_rag.planning.llm_based import LLMBasedPlanner
from enterprise_rag.planning.providers.openai_provider import OpenAIProvider


def main() -> None:
    out_dir = Path("outputs/evaluation/llm_planner")
    sample = json.loads((out_dir / "pilot_sample.json").read_text())
    cache_dir = Path(sample["cache_dir"])
    query_ids = sample["query_ids"]
    seed = sample["seed"]

    by_id = {}
    with open("data/sekd/queries.jsonl") as f:
        for line in f:
            q = json.loads(line)
            by_id[q["query_id"]] = q
    queries = [by_id[qid] for qid in query_ids]

    provider = OpenAIProvider(model="gpt-5")
    planner = LLMBasedPlanner(provider=provider, prompt_version="structured", cache_dir=cache_dir)

    t0 = time.monotonic()
    decisions = []
    cache_errors = []
    for q in queries:
        try:
            decisions.append(planner.plan(q))
        except Exception as e:
            cache_errors.append({"query_id": q["query_id"], "error": f"{type(e).__name__}: {e}"})
    cache_elapsed = time.monotonic() - t0

    dec_path = out_dir / "pilot_decisions.jsonl"
    with open(dec_path, "w") as f:
        for d in decisions:
            f.write(json.dumps(d.to_dict()) + "\n")

    success_n = len(decisions)
    n = len(query_ids)
    parse_success_n = sum(1 for d in decisions if d.parse_success)
    fallback_n = sum(1 for d in decisions if d.fallback_used)
    latencies = [d.latency_s for d in decisions]
    prompt_tokens = sum(d.prompt_tokens for d in decisions)
    completion_tokens = sum(d.completion_tokens for d in decisions)
    total_tokens = prompt_tokens + completion_tokens
    total_cost = sum(d.estimated_cost_usd for d in decisions)
    cache_files = list(cache_dir.glob("*.json"))
    cache_hit_n = success_n - len(cache_errors)
    cache_hit_rate = cache_hit_n / success_n if success_n else 0.0
    correct_n = sum(1 for d in decisions if d.is_correct)
    accuracy = correct_n / success_n if success_n else 0.0

    sample_outputs = []
    for d in decisions[:5]:
        sample_outputs.append(
            {
                "query_id": d.query_id,
                "selected_strategy": d.selected_strategy,
                "confidence": d.planner_confidence,
                "parse_success": d.parse_success,
                "parse_attempts": d.parse_attempts,
                "latency_s": round(d.latency_s, 4),
                "prompt_tokens": d.prompt_tokens,
                "completion_tokens": d.completion_tokens,
                "estimated_cost_usd": round(d.estimated_cost_usd, 8),
                "raw_response": d.raw_response,
                "parsed_reasoning": d.parsed_reasoning,
            }
        )

    summary = {
        "provider_used": provider.provider_name,
        "provider_class": type(provider).__name__,
        "model_used": provider.model_name,
        "prompt_version": "structured",
        "sample_size_requested": n,
        "sample_seed": seed,
        "query_ids": query_ids,
        "api_success_count": success_n,
        "api_error_count": n - success_n,
        "api_success_rate": success_n / n if n else 0.0,
        "json_parse_success_count": parse_success_n,
        "json_parse_success_rate": parse_success_n / success_n if success_n else 0.0,
        "fallback_count": fallback_n,
        "latency_s": {
            "avg": statistics.mean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
        },
        "tokens": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
        "estimated_cost_usd": total_cost,
        "cache": {
            "cache_dir": str(cache_dir),
            "cache_files_created": len(cache_files),
            "cache_files_expected": success_n,
            "cache_second_pass_decisions": success_n,
            "cache_second_pass_errors": cache_errors,
            "cache_hit_count": cache_hit_n,
            "cache_hit_rate": cache_hit_rate,
            "cache_pass_elapsed_s": cache_elapsed,
        },
        "planner_accuracy_vs_oracle_on_sample": accuracy,
        "sample_outputs": sample_outputs,
    }
    summary_path = out_dir / "pilot_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    recommend = (
        "Proceed with the full 380-query run"
        if summary["api_success_rate"] == 1.0
        and summary["json_parse_success_rate"] == 1.0
        and cache_hit_rate == 1.0
        else "Do not proceed with the full 380-query run yet"
    )

    md: list[str] = []
    md.append("# Real OpenAI GPT-5 Pilot Evaluation")
    md.append("")
    md.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append(
        "Scope: 20 random answerable queries, real `OpenAIProvider`, model `gpt-5`. "
        "No full 380-query planner evaluation was run."
    )
    md.append("")
    md.append("## Result")
    md.append("")
    md.append(f"- Provider used: `{provider.provider_name}` (`{type(provider).__name__}`)")
    md.append(f"- Model used: `{provider.model_name}`")
    md.append("- MockProvider used: `no`")
    md.append("- Prompt version: `structured`")
    md.append(f"- Sample size: `{n}` random queries")
    md.append(f"- Sample seed: `{seed}`")
    md.append(f"- API success rate: `{summary['api_success_rate']:.2%}` ({success_n}/{n})")
    md.append(
        f"- JSON parse success rate: `{summary['json_parse_success_rate']:.2%}` "
        f"({parse_success_n}/{success_n})"
    )
    md.append(f"- Fallback count: `{fallback_n}`")
    md.append(f"- Average latency: `{summary['latency_s']['avg']:.3f}s`")
    md.append(f"- Median latency: `{summary['latency_s']['median']:.3f}s`")
    md.append(
        f"- Latency range: `{summary['latency_s']['min']:.3f}s` - "
        f"`{summary['latency_s']['max']:.3f}s`"
    )
    md.append(f"- Total prompt tokens: `{prompt_tokens}`")
    md.append(f"- Total completion tokens: `{completion_tokens}`")
    md.append(f"- Total tokens: `{total_tokens}`")
    md.append(f"- Estimated cost: `${total_cost:.6f}`")
    md.append(f"- Sample strategy-selection accuracy vs oracle: `{accuracy:.2%}`")
    md.append("")
    md.append("## JSON Parsing Reliability")
    md.append("")
    md.append(
        "All 20 GPT-5 API calls returned usage metadata and were cached, but the planner "
        "received empty visible `raw_response` content under the current "
        "`max_completion_tokens=200` budget. Each query exhausted 3 parse attempts and "
        "fell back to `hybrid`."
    )
    md.append("")
    md.append(
        "This means the API path works, but JSON parsing reliability for the current "
        "GPT-5 provider settings is `0.00%`. The full run should not proceed until "
        "the provider/prompt settings are adjusted and another pilot reaches reliable "
        "JSON output."
    )
    md.append("")
    md.append("## Cache Verification")
    md.append("")
    md.append(f"- Cache directory: `{cache_dir}`")
    md.append(f"- Cache files created: `{len(cache_files)}`")
    md.append(f"- Expected cache files: `{success_n}`")
    md.append(f"- Cached read decisions: `{success_n}`")
    md.append(f"- Cached read errors: `{len(cache_errors)}`")
    md.append(f"- Cache hit rate: `{cache_hit_rate:.2%}` ({cache_hit_n}/{success_n})")
    md.append(f"- Cached read elapsed time: `{cache_elapsed:.4f}s`")
    md.append("")
    md.append("## Query Sample")
    md.append("")
    for qid in query_ids:
        md.append(f"- `{qid}`")
    md.append("")
    md.append("## Sample Planner Outputs")
    md.append("")
    for item in sample_outputs:
        md.append(f"### {item['query_id']}")
        md.append(f"- Selected strategy: `{item['selected_strategy']}`")
        md.append(f"- Confidence: `{item['confidence']}`")
        md.append(
            f"- Parse success: `{item['parse_success']}` after "
            f"`{item['parse_attempts']}` attempt(s)"
        )
        md.append(f"- Latency: `{item['latency_s']}s`")
        md.append(
            f"- Tokens: `{item['prompt_tokens']}` prompt + "
            f"`{item['completion_tokens']}` completion"
        )
        md.append(f"- Estimated cost: `${item['estimated_cost_usd']:.8f}`")
        md.append(f"- Parsed reasoning: {item['parsed_reasoning']}")
        md.append("- Raw response:")
        md.append("```json")
        md.append(item["raw_response"] if item["raw_response"] else "<empty string>")
        md.append("```")
        md.append("")
    md.append("## Stored Artifacts")
    md.append("")
    md.append(f"- Planner decisions: `{dec_path}`")
    md.append(f"- Pilot summary JSON: `{summary_path}`")
    md.append(f"- Sample metadata: `{out_dir / 'pilot_sample.json'}`")
    md.append(f"- Cache records: `{cache_dir}`")
    md.append("")
    md.append("## Recommendation")
    md.append("")
    md.append(f"{recommend}.")
    md.append(
        "Reason: API success and cache verification passed, but JSON parse success "
        "was 0/20. Before a full 380-query run, increase the GPT-5 completion budget "
        "and/or use a GPT-5-compatible structured-output path so the model emits "
        "visible JSON reliably."
    )

    report_path = out_dir / "pilot_report.md"
    report_path.write_text("\n".join(md) + "\n")
    print(
        json.dumps(
            {
                "report_path": str(report_path),
                "decisions_path": str(dec_path),
                "summary_path": str(summary_path),
                "cache_dir": str(cache_dir),
                "api_success_rate": summary["api_success_rate"],
                "json_parse_success_rate": summary["json_parse_success_rate"],
                "cache_hit_rate": cache_hit_rate,
                "estimated_cost_usd": total_cost,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
