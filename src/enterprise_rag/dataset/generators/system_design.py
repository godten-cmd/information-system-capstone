"""System Design document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import DEPARTMENTS, ROLE_NAMES, SYSTEM_DESIGN_SPECS
from .._rng import SeededRNG
from ..enums import (
    ArchitectureStatus,
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    Region,
    ReviewCycle,
)
from ..schemas import DocumentMetadata, EnterpriseDocument, SystemDesignMetadata

_ARCH_STATUS_MAP: dict[str, ArchitectureStatus] = {
    "proposed": ArchitectureStatus.PROPOSED,
    "approved": ArchitectureStatus.APPROVED,
    "implemented": ArchitectureStatus.IMPLEMENTED,
    "retired": ArchitectureStatus.RETIRED,
}

_BASE_DATE = date(2022, 3, 1)

# First 3 specs produce 3 docs each (9 total); last 3 produce 2 docs each (6 total) = 15
_DOCS_PER_SPEC = [3, 3, 3, 2, 2, 2]

_VARIANT_SUBTITLES = [
    "Architecture Overview",
    "Data Flow and Integration",
    "Decision Records and Failure Modes",
]

_ARCH_PATTERNS = ["microservices", "event-driven", "layered", "pipeline", "gateway"]
_INTEGRATION_STYLES = ["REST APIs", "gRPC", "message queues", "event streaming", "direct database access"]


def _build_adrs(spec: dict, variant: int, rng: SeededRNG) -> list[dict]:
    prefix = spec["adr_prefix"]
    base_adrs = [
        {
            "id": f"{prefix}-001",
            "title": f"Use {rng.choice(_ARCH_PATTERNS)} architecture for {spec['system_name']}",
            "decision": "Adopted after evaluation of monolithic and distributed approaches.",
            "rationale": "Aligns with HYTech scaling requirements and deployment tooling.",
            "superseded_by": None,
        },
        {
            "id": f"{prefix}-002",
            "title": f"Select primary data store for {spec['system_name']}",
            "decision": "PostgreSQL selected for transactional consistency.",
            "rationale": "Meets ACID requirements and integrates with existing HYDataLake pipelines.",
            "superseded_by": None,
        },
        {
            "id": f"{prefix}-003",
            "title": "Authentication strategy",
            "decision": "Delegate authentication to HYID using service-to-service mTLS.",
            "rationale": "Centralised identity management reduces security surface area.",
            "superseded_by": None,
        },
    ]
    if variant == 2:
        base_adrs.append({
            "id": f"{prefix}-004",
            "title": "Observability strategy",
            "decision": f"Use HYMonitor for all metrics and alerting.",
            "rationale": "Standardised observability reduces operational overhead.",
            "superseded_by": None,
        })
    return base_adrs[: 2 + variant]


def _build_data_flow_steps(spec: dict, variant: int) -> list[str]:
    system = spec["system_name"]
    components = spec["components"]
    if variant == 0:
        return [
            f"Client sends request to {components[0]['name']}.",
            f"{components[0]['name']} validates and forwards to {components[1]['name'] if len(components) > 1 else 'downstream service'}.",
            "Processed result is stored and returned to the caller.",
        ]
    elif variant == 1:
        return [
            f"Incoming event arrives at {components[0]['name']}.",
            "Event is validated, enriched, and routed based on type.",
            "Result is persisted and a completion notification is emitted.",
            "Monitoring metrics are published to HYMonitor.",
        ]
    else:
        return [
            f"Operator triggers the operation via the {system} API.",
            f"Request is authenticated through HYID.",
            "State machine advances; gate conditions are evaluated.",
            "Final state is committed and audit log is written.",
        ]


class SystemDesignGenerator:
    """Generates System Design documents — 6 specs × 2-3 variants = 15 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler
        # Pre-compute (spec_idx, variant_idx) pairs for each 1-based index
        self._index_map: list[tuple[int, int]] = []
        for spec_idx, count in enumerate(_DOCS_PER_SPEC):
            for variant in range(count):
                self._index_map.append((spec_idx, variant))

    def generate_one(self, index: int, all_doc_ids: list[str]) -> EnterpriseDocument:
        spec_idx, variant = self._index_map[(index - 1) % len(self._index_map)]
        spec = SYSTEM_DESIGN_SPECS[spec_idx]

        doc_id = f"SYS-DES-{index:03d}"
        arch_status_str = spec["architecture_status"]
        adr_prefix = spec["adr_prefix"]

        eff_date = _BASE_DATE + timedelta(days=(index - 1) * 65)
        created_dt = datetime(eff_date.year, eff_date.month, eff_date.day, 10, 0, 0)
        updated_dt = created_dt + timedelta(days=self._rng.randint(0, 30))

        version = f"1.{variant}"
        primary_author = self._rng.choice(ROLE_NAMES[7:11])
        reviewer_roles = self._rng.sample(ROLE_NAMES[10:16], 2)

        subtitle = _VARIANT_SUBTITLES[min(variant, len(_VARIANT_SUBTITLES) - 1)]
        title = f"{spec['system_name']} – {subtitle}"

        components = spec["components"]
        failure_modes_raw = spec["failure_modes"]
        failure_modes = [
            {
                "mode": fm["mode"],
                "impact": "Service degradation or data unavailability",
                "mitigation": fm["mitigation"],
                "residual_risk": "Low after mitigation",
            }
            for fm in failure_modes_raw[: 2 + variant]
        ]

        adrs = _build_adrs(spec, variant, self._rng)
        data_flow_steps = _build_data_flow_steps(spec, variant)

        related_doc_refs: list[str] = []
        api_doc_ids = [d for d in all_doc_ids if d.startswith("API-DOC")]
        if api_doc_ids:
            related_doc_refs = self._rng.sample(api_doc_ids, min(2, len(api_doc_ids)))

        security_controls = ["HY-AC-01", "HY-DP-01", "HY-IR-01"]

        tradeoffs = spec.get("tradeoffs", [])
        arch_pattern = self._rng.choice(_ARCH_PATTERNS)
        integration_style = self._rng.choice(_INTEGRATION_STYLES)

        rev_history = [
            {
                "version": "1.0",
                "date": str(eff_date - timedelta(days=30)),
                "author_role": primary_author,
                "change": "Initial draft",
            }
        ]
        if variant > 0:
            rev_history.append({
                "version": version,
                "date": str(eff_date),
                "author_role": primary_author,
                "change": f"Added {subtitle.lower()} section",
            })

        decision_record_ids = [a["id"] for a in adrs]
        related_services = spec.get("dependencies", ["HYID", "HYMonitor"])

        context = {
            "title": title,
            "design_doc_id": doc_id,
            "document_owner": primary_author,
            "department": DEPARTMENTS["System Design Documents"][0],
            "effective_date": str(eff_date),
            "version": version,
            "status": DocumentStatus.ACTIVE.value,
            "system_name": spec["system_name"],
            "architecture_status": arch_status_str,
            "primary_author_role": primary_author,
            "reviewer_roles": reviewer_roles,
            "decision_record_ids": decision_record_ids,
            "related_services": related_services,
            "system_description": spec["problem_statement"],
            "sla_uptime": spec["sla"],
            "sla_recovery_time": "1 hour",
            "data_volume": spec["data_volume"],
            "architecture_pattern": arch_pattern,
            "architecture_summary": (
                f"The {spec['system_name']} uses a {arch_pattern} architecture to meet "
                f"the scalability and reliability requirements defined in this document."
            ),
            "integration_style": integration_style,
            "components": [
                {
                    "name": c["name"],
                    "responsibility": c["responsibility"],
                    "dependencies": c.get("depends_on", []),
                }
                for c in components
            ],
            "data_flow_description": f"The {spec['system_name']} processes requests through the following flow:",
            "data_flow_steps": data_flow_steps,
            "adrs": adrs,
            "failure_modes": failure_modes,
            "scalability_description": f"The {spec['system_name']} scales horizontally.",
            "peak_throughput": spec["data_volume"].split(".")[0],
            "p99_latency_ms": self._rng.choice([50, 100, 200, 500]),
            "horizontal_scaling": "Supported via stateless service instances behind a load balancer",
            "security_notes": (
                "All inter-service communication is authenticated via HYID mTLS. "
                "Data at rest is encrypted using AES-256."
            ),
            "security_controls": security_controls,
            "related_doc_refs": related_doc_refs,
            "revision_history": rev_history,
        }

        body = self._assembler.render("system_design.j2", context)

        category_metadata = SystemDesignMetadata(
            design_doc_id=doc_id,
            system_name=spec["system_name"],
            architecture_status=_ARCH_STATUS_MAP[arch_status_str],
            primary_author_role=primary_author,
            reviewer_roles=reviewer_roles,
            related_services=list(related_services),
            decision_record_ids=decision_record_ids,
        )

        dept = DEPARTMENTS["System Design Documents"][0]
        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.SYSTEM_DESIGN,
            title=title,
            version=version,
            created_at=created_dt,
            updated_at=updated_dt,
            department=dept,
            document_owner=primary_author,
            confidentiality=ConfidentialityLevel.CONFIDENTIAL,
            region=Region.GLOBAL,
            language="en",
            effective_date=eff_date,
            review_cycle=ReviewCycle.SEMIANNUAL,
            status=DocumentStatus.ACTIVE,
            authority_level=AuthorityLevel.STANDARD,
            tags=[spec["system_name"].lower().replace(" ", "_"), arch_status_str, "architecture"],
            source_path=f"/data/design/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body, references=related_doc_refs)
