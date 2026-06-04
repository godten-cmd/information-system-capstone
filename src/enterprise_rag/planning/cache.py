"""On-disk cache for LLM planner decisions.

Cache key: {query_id}__{model}__{prompt_version}
Cache file: {cache_dir}/{cache_key}.json

Caches raw LLM responses so re-runs do not incur additional API cost.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheRecord:
    cache_key: str
    query_id: str
    model: str
    prompt_version: str
    raw_response: str
    selected_strategy: str
    confidence: float
    reasoning: str
    parse_success: bool
    parse_attempts: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    timestamp: str


class PlannerCache:
    """Simple JSON-file-per-decision cache."""

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(query_id: str, model: str, prompt_version: str) -> str:
        safe = lambda s: s.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"{safe(query_id)}__{safe(model)}__{safe(prompt_version)}"

    def get(self, query_id: str, model: str, prompt_version: str) -> CacheRecord | None:
        key = self.make_key(query_id, model, prompt_version)
        path = self._dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            data: dict[str, Any] = json.loads(path.read_text())
            return CacheRecord(**data)
        except Exception:
            return None

    def put(self, record: CacheRecord) -> None:
        key = record.cache_key
        path = self._dir / f"{key}.json"
        path.write_text(json.dumps(asdict(record), indent=2))

    def size(self) -> int:
        return sum(1 for _ in self._dir.glob("*.json"))
