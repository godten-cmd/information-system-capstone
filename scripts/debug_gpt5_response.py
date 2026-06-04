"""Debug script: capture raw GPT-5 API response for parsing diagnostics.

Makes ONE real API call (no cache) and dumps the complete response object to
outputs/evaluation/llm_planner/raw_response_debug.json.

Usage:
    python scripts/debug_gpt5_response.py [--model gpt-5] [--api chat|responses]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = typer.Typer(add_completion=False)

_OUT = Path("outputs/evaluation/llm_planner/raw_response_debug.json")

# Fixed test prompt (simple, short, deterministic)
_SYSTEM = (
    "You are a retrieval strategy planner. "
    "Respond ONLY with valid JSON — no markdown, no extra text: "
    '{"selected_strategy": "bm25|dense|hybrid", "confidence": 0.0-1.0, "reasoning": "one sentence"}'
)
_USER = (
    "Query: Which API endpoints in HYDeploy v2 depend on the HYID authentication service?\n\n"
    "Query metadata:\n"
    "- Reasoning type: multi_hop\n"
    "- Retrieval difficulty factors: multi_document_dependency\n"
    "- Document category: API Documentation\n\n"
    "Select the best retrieval strategy.\n"
    "# query_id: Q-000001-debug"
)


def _safe_dict(obj) -> dict:
    """Convert an SDK response object to a plain dict, handling nested models."""
    try:
        return obj.model_dump()
    except AttributeError:
        pass
    try:
        return dict(obj)
    except (TypeError, ValueError):
        return {"repr": repr(obj)}


@app.command()
def main(
    model: str = typer.Option("gpt-5", help="Model to test"),
    api: str = typer.Option("both", help="chat | responses | both"),
    max_tokens: int = typer.Option(200, help="Token budget to test"),
) -> None:
    """Capture and dump raw GPT-5 API response for parsing diagnostics."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        typer.echo("ERROR: OPENAI_API_KEY not set.", err=True)
        raise typer.Exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        typer.echo("ERROR: openai package not installed.", err=True)
        raise typer.Exit(1)

    client = OpenAI(api_key=api_key)
    results: dict = {
        "model": model,
        "max_tokens_tested": max_tokens,
        "system_prompt_len": len(_SYSTEM),
        "user_prompt_len": len(_USER),
        "calls": [],
    }

    # ── A. Chat Completions ───────────────────────────────────────────────────
    if api in ("chat", "both"):
        typer.echo(f"\n[1] Testing Chat Completions API  (max_completion_tokens={max_tokens}) ...")
        payload_chat: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER},
            ],
            "max_completion_tokens": max_tokens,
        }
        t0 = time.monotonic()
        try:
            resp_chat = client.chat.completions.create(**payload_chat)
            latency = time.monotonic() - t0
            raw_dict = _safe_dict(resp_chat)
            content = resp_chat.choices[0].message.content if resp_chat.choices else None
            usage = resp_chat.usage
            usage_dict = _safe_dict(usage) if usage else {}

            typer.echo(f"  content (first 300 chars): {repr(content[:300]) if content else repr(content)}")
            typer.echo(f"  usage: {usage_dict}")
            typer.echo(f"  latency: {latency:.2f}s")

            results["calls"].append({
                "api": "chat_completions",
                "max_completion_tokens": max_tokens,
                "latency_s": round(latency, 3),
                "content": content,
                "content_length": len(content) if content else 0,
                "usage": usage_dict,
                "full_response": raw_dict,
            })
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)
            results["calls"].append({"api": "chat_completions", "error": str(e)})

    # ── B. Chat Completions with LARGER budget ────────────────────────────────
    if api in ("chat", "both"):
        large_budget = max(max_tokens * 10, 2000)
        typer.echo(f"\n[2] Testing Chat Completions API  (max_completion_tokens={large_budget}, LARGE budget) ...")
        payload_large: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER},
            ],
            "max_completion_tokens": large_budget,
        }
        t0 = time.monotonic()
        try:
            resp_large = client.chat.completions.create(**payload_large)
            latency = time.monotonic() - t0
            raw_dict = _safe_dict(resp_large)
            content = resp_large.choices[0].message.content if resp_large.choices else None
            usage = resp_large.usage
            usage_dict = _safe_dict(usage) if usage else {}

            typer.echo(f"  content (first 300 chars): {repr(content[:300]) if content else repr(content)}")
            typer.echo(f"  usage: {usage_dict}")
            typer.echo(f"  latency: {latency:.2f}s")

            results["calls"].append({
                "api": "chat_completions_large_budget",
                "max_completion_tokens": large_budget,
                "latency_s": round(latency, 3),
                "content": content,
                "content_length": len(content) if content else 0,
                "usage": usage_dict,
                "full_response": raw_dict,
            })
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)
            results["calls"].append({"api": "chat_completions_large_budget", "error": str(e)})

    # ── C. Responses API ──────────────────────────────────────────────────────
    if api in ("responses", "both"):
        typer.echo(f"\n[3] Testing Responses API  (max_output_tokens={max_tokens}) ...")
        t0 = time.monotonic()
        try:
            resp_responses = client.responses.create(
                model=model,
                instructions=_SYSTEM,
                input=_USER,
                max_output_tokens=max_tokens,
            )
            latency = time.monotonic() - t0
            raw_dict = _safe_dict(resp_responses)
            content = getattr(resp_responses, "output_text", None)
            usage = getattr(resp_responses, "usage", None)
            usage_dict = _safe_dict(usage) if usage else {}

            typer.echo(f"  output_text (first 300 chars): {repr(content[:300]) if content else repr(content)}")
            typer.echo(f"  usage: {usage_dict}")
            typer.echo(f"  latency: {latency:.2f}s")

            results["calls"].append({
                "api": "responses_api",
                "max_output_tokens": max_tokens,
                "latency_s": round(latency, 3),
                "output_text": content,
                "content_length": len(content) if content else 0,
                "usage": usage_dict,
                "full_response": raw_dict,
            })
        except Exception as e:
            typer.echo(f"  ERROR: {e}", err=True)
            results["calls"].append({"api": "responses_api", "error": str(e)})

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    _OUT.write_text(json.dumps(results, indent=2, default=str))
    typer.echo(f"\nFull debug dump written to: {_OUT}")

    # ── Summary ───────────────────────────────────────────────────────────────
    typer.echo("\n── Summary ─────────────────────────────────────────────────────")
    for call in results["calls"]:
        api_name = call.get("api", "?")
        err = call.get("error")
        if err:
            typer.echo(f"  {api_name}: ERROR — {err}")
        else:
            clen = call.get("content_length", 0)
            has_content = clen > 0
            status = "OK (content present)" if has_content else "EMPTY content"
            typer.echo(f"  {api_name}: {status} ({clen} chars)")
            usage = call.get("usage", {})
            if usage:
                typer.echo(f"    usage: {usage}")


if __name__ == "__main__":
    app()
