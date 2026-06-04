"""SEKDGenerator: orchestrates all category generators and writes output artifacts."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import jsonlines

from ._assembler import DocumentAssembler
from ._rng import SeededRNG
from .generators import (
    APIDocGenerator,
    HRPolicyGenerator,
    MeetingNotesGenerator,
    SecurityPolicyGenerator,
    SystemDesignGenerator,
    TravelPolicyGenerator,
)
from .schemas import DocumentMetadata, EnterpriseDocument

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_CATEGORY_ORDER = [
    ("hr_policy", HRPolicyGenerator, 20),
    ("travel_policy", TravelPolicyGenerator, 15),
    ("security_policy", SecurityPolicyGenerator, 20),
    ("api_doc", APIDocGenerator, 20),
    ("system_design", SystemDesignGenerator, 15),
    ("meeting_notes", MeetingNotesGenerator, 30),
]

_DEFAULT_COUNTS = {
    "hr_policy": 20,
    "travel_policy": 15,
    "security_policy": 20,
    "api_doc": 20,
    "system_design": 15,
    "meeting_notes": 30,
}


def _distribute_counts(total: int) -> dict[str, int]:
    if total == 120:
        return dict(_DEFAULT_COUNTS)
    base_total = sum(_DEFAULT_COUNTS.values())
    counts: dict[str, int] = {}
    assigned = 0
    for key, default in _DEFAULT_COUNTS.items():
        if key == "meeting_notes":
            continue
        cnt = max(1, round(default * total / base_total))
        counts[key] = cnt
        assigned += cnt
    counts["meeting_notes"] = max(1, total - assigned)
    return counts


@dataclass
class DatasetConfig:
    company: str = "HYTech Solutions"
    document_count: int = 120
    seed: int = 42
    include_adversarial: bool = True
    language: str = "en"


@dataclass
class GenerationReport:
    total_documents: int
    per_category: dict[str, int]
    validation_errors: list[str]
    output_paths: dict[str, Path]
    seed: int
    duration_seconds: float


class SEKDGenerator:
    """Deterministic synthetic document generator for the SEKD dataset."""

    def __init__(self, config: DatasetConfig | None = None, seed: int = 42) -> None:
        self._config = config or DatasetConfig(seed=seed)
        self._config.seed = seed if config is None else self._config.seed

    def generate(self) -> list[EnterpriseDocument]:
        rng = SeededRNG(self._config.seed)
        assembler = DocumentAssembler(_TEMPLATES_DIR)
        counts = _distribute_counts(self._config.document_count)

        documents: list[EnterpriseDocument] = []
        all_doc_ids: list[str] = []

        for key, GenClass, _ in _CATEGORY_ORDER:
            count = counts.get(key, 1)
            cat_rng = rng.fork(key)
            generator = GenClass(cat_rng, assembler)
            for i in range(1, count + 1):
                doc = generator.generate_one(i, list(all_doc_ids))
                documents.append(doc)
                all_doc_ids.append(doc.document_id)

        return documents

    def write(
        self,
        documents: list[EnterpriseDocument],
        output_dir: Path,
    ) -> GenerationReport:
        start = time.monotonic()
        output_dir.mkdir(parents=True, exist_ok=True)

        validation_errors: list[str] = []
        seen_ids: set[str] = set()
        for doc in documents:
            if doc.document_id in seen_ids:
                validation_errors.append(f"Duplicate document_id: {doc.document_id}")
            seen_ids.add(doc.document_id)

        docs_path = output_dir / "documents.jsonl"
        with jsonlines.open(docs_path, mode="w") as writer:
            for doc in documents:
                writer.write(doc.model_dump(by_alias=True, mode="json"))

        meta_path = output_dir / "metadata_schema.json"
        meta_path.write_text(
            json.dumps(DocumentMetadata.model_json_schema(), indent=2, ensure_ascii=False)
        )

        per_category: dict[str, int] = {}
        for doc in documents:
            cat = doc.metadata.category.value
            per_category[cat] = per_category.get(cat, 0) + 1

        config_path = output_dir / "category_config.json"
        config_data = {
            "seed": self._config.seed,
            "total_documents": len(documents),
            "per_category": per_category,
            "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        config_path.write_text(json.dumps(config_data, indent=2, ensure_ascii=False))

        duration = time.monotonic() - start
        return GenerationReport(
            total_documents=len(documents),
            per_category=per_category,
            validation_errors=validation_errors,
            output_paths={
                "documents": docs_path,
                "metadata_schema": meta_path,
                "category_config": config_path,
            },
            seed=self._config.seed,
            duration_seconds=round(duration, 3),
        )
