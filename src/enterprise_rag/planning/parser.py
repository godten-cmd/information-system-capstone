"""JSON response parser for LLM planner outputs.

Handles:
  - Clean JSON
  - JSON wrapped in markdown code fences
  - Extra preamble/postamble text
  - Missing or invalid fields (returns None so caller can retry)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from enterprise_rag.planning.schema import AVAILABLE_STRATEGIES

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_BRACES_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


@dataclass
class ParsedPlan:
    selected_strategy: str
    confidence: float
    reasoning: str


def parse_plan(text: str) -> ParsedPlan | None:
    """Parse a (potentially messy) LLM response into a ParsedPlan.

    Returns None if parsing fails or the result is invalid.
    """
    if not text or not text.strip():
        return None

    raw = text.strip()

    # 1. Try markdown code fence
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        raw = m.group(1).strip()

    # 2. Try to extract the first JSON object
    brace_m = _BRACES_RE.search(raw)
    if brace_m:
        raw = brace_m.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    strategy = str(data.get("selected_strategy", "")).strip().lower()
    if strategy not in AVAILABLE_STRATEGIES:
        return None

    raw_conf = data.get("confidence", 0.5)
    try:
        confidence = float(raw_conf)
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    reasoning = str(data.get("reasoning", "")).strip()

    return ParsedPlan(
        selected_strategy=strategy,
        confidence=confidence,
        reasoning=reasoning,
    )
