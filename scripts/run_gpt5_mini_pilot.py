"""GPT-5 mini-pilot: 5 random queries to verify parsing fix.

Runs 5 queries through the fixed OpenAIProvider and measures:
  - API success rate
  - JSON parse success rate
  - Latency (mean, min, max)
  - Token usage
  - Estimated cost

Writes:
  outputs/evaluation/llm_planner/parsing_fix_report.md

Usage:
    python scripts/run_gpt5_mini_pilot.py [--model gpt-5] [--seed 42]
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from enterprise_rag.planning.llm_based import LLMBasedPlanner
from enterprise_rag.planning.parser import parse_plan
from enterprise_rag.planning.providers.openai_provider import OpenAIProvider
from enterprise_rag.planning.schema import estimate_cost

app = typer.Typer(add_completion=False)

_QUERIES_PATH = Path("data/sekd/queries.jsonl")
_OUT_DIR = Path("outputs/evaluation/llm_planner")
_REPORT_PATH = _OUT_DIR / "parsing_fix_report.md"
_N_QUERIES = 5


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


@app.command()
def main(
    model: str = typer.Option("gpt-5", help="OpenAI model to test"),
    seed: int = typer.Option(42, help="Random seed for query selection"),
    prompt_version: str = typer.Option("structured", help="zero_shot | structured | few_shot"),
) -> None:
    """Run 5-query GPT-5 mini-pilot to verify parsing fix."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        typer.echo("ERROR: OPENAI_API_KEY not set.", err=True)
        raise typer.Exit(1)

    typer.echo(f"GPT-5 Mini-Pilot  [model={model}, seed={seed}, prompt={prompt_version}]")
    typer.echo("=" * 60)

    # ── Load queries ──────────────────────────────────────────────────────────
    all_queries = _load_jsonl(_QUERIES_PATH)
    answerable = [q for q in all_queries if q.get("answerable", True)]
    rng = random.Random(seed)
    sample = rng.sample(answerable, min(_N_QUERIES, len(answerable)))
    typer.echo(f"Selected {len(sample)} queries (seed={seed})")

    # ── Build provider and planner (no cache so we get fresh responses) ───────
    provider = OpenAIProvider(model=model, api_key=api_key)
    planner = LLMBasedPlanner(
        provider=provider,
        prompt_version=prompt_version,
        cache_dir=None,  # no cache — we want fresh API calls
    )

    # ── Run queries one at a time, capturing all detail ───────────────────────
    results = []
    total_start = time.monotonic()

    for i, query in enumerate(sample, 1):
        qid = query["query_id"]
        typer.echo(f"\n[{i}/{len(sample)}] {qid}")
        t0 = time.monotonic()
        decision = planner.plan(query)
        elapsed = time.monotonic() - t0

        parse_ok = decision.parse_success
        raw = decision.raw_response or ""
        parsed = parse_plan(raw)

        typer.echo(f"  parse_success  : {parse_ok}")
        typer.echo(f"  raw_response   : {repr(raw[:200]) if raw else '<empty>'}")
        typer.echo(f"  parsed JSON    : {repr(decision.parsed_reasoning[:120])}")
        typer.echo(f"  selected       : {decision.selected_strategy} (oracle={decision.oracle_strategy_mapped})")
        typer.echo(f"  confidence     : {decision.planner_confidence}")
        typer.echo(f"  latency        : {decision.latency_s:.2f}s")
        typer.echo(f"  tokens         : {decision.prompt_tokens} prompt + {decision.completion_tokens} completion")
        typer.echo(f"  cost           : ${decision.estimated_cost_usd:.6f}")

        results.append({
            "query_id": qid,
            "parse_success": parse_ok,
            "raw_response": raw,
            "raw_response_length": len(raw),
            "selected_strategy": decision.selected_strategy,
            "oracle_strategy": decision.oracle_strategy_mapped,
            "is_correct": decision.is_correct,
            "confidence": decision.planner_confidence,
            "reasoning": decision.parsed_reasoning,
            "parse_attempts": decision.parse_attempts,
            "latency_s": decision.latency_s,
            "prompt_tokens": decision.prompt_tokens,
            "completion_tokens": decision.completion_tokens,
            "estimated_cost_usd": decision.estimated_cost_usd,
        })

    total_wall = time.monotonic() - total_start

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    n = len(results)
    n_parse_ok = sum(1 for r in results if r["parse_success"])
    n_api_ok = sum(1 for r in results if r["raw_response_length"] > 0)
    n_correct = sum(1 for r in results if r["is_correct"])
    latencies = [r["latency_s"] for r in results]
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_cost = sum(r["estimated_cost_usd"] for r in results)

    api_success_pct = 100 * n_api_ok / n if n else 0
    parse_success_pct = 100 * n_parse_ok / n if n else 0
    accuracy_pct = 100 * n_correct / n if n else 0

    typer.echo("\n" + "=" * 60)
    typer.echo("MINI-PILOT RESULTS")
    typer.echo("=" * 60)
    typer.echo(f"  Queries run            : {n}")
    typer.echo(f"  API success rate       : {api_success_pct:.1f}% ({n_api_ok}/{n})")
    typer.echo(f"  JSON parse success     : {parse_success_pct:.1f}% ({n_parse_ok}/{n})")
    typer.echo(f"  Strategy accuracy      : {accuracy_pct:.1f}% ({n_correct}/{n})")
    typer.echo(f"  Avg latency            : {mean(latencies):.2f}s")
    typer.echo(f"  Latency range          : {min(latencies):.2f}s – {max(latencies):.2f}s")
    typer.echo(f"  Total prompt tokens    : {total_prompt}")
    typer.echo(f"  Total completion tokens: {total_completion}")
    typer.echo(f"  Total cost             : ${total_cost:.6f}")

    ready = n_parse_ok == n and n_api_ok == n
    verdict = "READY for full 380-query evaluation" if ready else "NOT READY — investigate remaining failures"
    typer.echo(f"\n  Verdict: {verdict}")

    # ── Write markdown report ─────────────────────────────────────────────────
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    lines = [
        "# GPT-5 Parsing Fix — Mini-Pilot Report",
        "",
        f"Generated: {now}",
        "",
        "## Root Cause",
        "",
        "GPT-5 is a reasoning model. `max_completion_tokens=200` was the combined budget for",
        "internal reasoning tokens AND visible output tokens. GPT-5's reasoning process consumed",
        "all 200 tokens, leaving zero for visible output — so `message.content` was always `\"\"`.",
        "",
        "**Evidence from debug run** (`scripts/debug_gpt5_response.py`):",
        "",
        "| Call | `max_completion_tokens` | `reasoning_tokens` | Output content |",
        "|---|---|---|---|",
        "| Chat Completions (original) | 200 | **200 (exhausted)** | `\"\"` (empty) |",
        "| Chat Completions (patched) | 2000 | ~384 | valid JSON ✓ |",
        "",
        "## Fix Applied",
        "",
        "File: `src/enterprise_rag/planning/providers/openai_provider.py`",
        "",
        "- Added `_is_reasoning_model()` helper that matches `gpt-5`, `o1`, `o3`, `o4` prefixes.",
        "- For reasoning models: `max_completion_tokens = max(requested_tokens + 2000, 2000)`.",
        "  This guarantees sufficient headroom for reasoning (~400 tokens) plus full JSON output.",
        "- Added warning log when content is empty so future regressions are surfaced immediately.",
        "- Non-reasoning models (gpt-4o, gpt-4o-mini, etc.) are unchanged.",
        "",
        "## Mini-Pilot Results",
        "",
        f"- Provider: `OpenAIProvider`",
        f"- Model: `{model}`",
        f"- Prompt version: `{prompt_version}`",
        f"- Sample size: `{n}` random queries (seed={seed})",
        f"- API success rate: `{api_success_pct:.1f}%` ({n_api_ok}/{n})",
        f"- JSON parse success rate: `{parse_success_pct:.1f}%` ({n_parse_ok}/{n})",
        f"- Strategy accuracy vs oracle: `{accuracy_pct:.1f}%` ({n_correct}/{n})",
        f"- Average latency: `{mean(latencies):.2f}s`",
        f"- Latency range: `{min(latencies):.2f}s` – `{max(latencies):.2f}s`",
        f"- Total prompt tokens: `{total_prompt}`",
        f"- Total completion tokens: `{total_completion}`",
        f"- Estimated cost: `${total_cost:.6f}`",
        "",
        "## Per-Query Details",
        "",
        "| Query ID | Parse | Strategy | Oracle | Correct | Latency | Tokens | Raw (first 120 chars) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        raw_preview = repr(r["raw_response"][:120]).replace("|", "\\|") if r["raw_response"] else "`<empty>`"
        lines.append(
            f"| {r['query_id']} "
            f"| {'✓' if r['parse_success'] else '✗'} "
            f"| {r['selected_strategy']} "
            f"| {r['oracle_strategy']} "
            f"| {'✓' if r['is_correct'] else '✗'} "
            f"| {r['latency_s']:.2f}s "
            f"| {r['prompt_tokens']}+{r['completion_tokens']} "
            f"| {raw_preview} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
    ]
    if ready:
        lines += [
            "Parsing is now reliable. The full 380-query evaluation can proceed using:",
            "",
            "```bash",
            f"python scripts/run_llm_planner.py --provider openai --model {model} --prompt-version {prompt_version}",
            "```",
        ]
    else:
        n_fail = n - n_parse_ok
        lines += [
            f"{n_fail}/{n} queries still failing — do not run full evaluation yet.",
            "Review the per-query details above to identify remaining failure patterns.",
        ]

    _REPORT_PATH.write_text("\n".join(lines))
    typer.echo(f"\nReport written to: {_REPORT_PATH}")


if __name__ == "__main__":
    app()
