"""Security Policy document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import DEPARTMENTS, ROLE_NAMES, SECURITY_POLICY_SPECS
from .._rng import SeededRNG
from ..enums import (
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    Region,
    ReviewCycle,
    SecurityControlFamily,
    SecurityRiskLevel,
)
from ..schemas import DocumentMetadata, EnterpriseDocument, SecurityPolicyMetadata

_FAMILY_MAP: dict[str, SecurityControlFamily] = {
    "access_control": SecurityControlFamily.ACCESS_CONTROL,
    "data_protection": SecurityControlFamily.DATA_PROTECTION,
    "incident_response": SecurityControlFamily.INCIDENT_RESPONSE,
    "device_security": SecurityControlFamily.DEVICE_SECURITY,
}

_RISK_MAP: dict[str, SecurityRiskLevel] = {
    "critical": SecurityRiskLevel.CRITICAL,
    "high": SecurityRiskLevel.HIGH,
    "medium": SecurityRiskLevel.MEDIUM,
    "low": SecurityRiskLevel.LOW,
}

_REVIEW_DAYS: dict[str, int] = {
    "critical": 90,
    "high": 180,
    "medium": 180,
    "low": 365,
}

_REMEDIATION_DAYS: dict[str, int] = {
    "critical": 3,
    "high": 7,
    "medium": 30,
    "low": 90,
}

_MAX_EXCEPTION_DAYS: dict[str, int] = {
    "critical": 0,
    "high": 30,
    "medium": 90,
    "low": 180,
}

_RISK_LIKELIHOOD: dict[str, str] = {
    "critical": "Highly likely without mandatory controls",
    "high": "Likely if controls are absent",
    "medium": "Moderate likelihood with partial controls",
    "low": "Low likelihood with existing baseline controls",
}

_RISK_IMPACT: dict[str, str] = {
    "critical": "Catastrophic — potential for data breach, regulatory penalty, or service outage",
    "high": "Significant — possible data exposure or prolonged service disruption",
    "medium": "Moderate — limited data exposure or short-term service degradation",
    "low": "Minor — localised impact with low business effect",
}

_DETECTION_DIFFICULTY: dict[str, str] = {
    "critical": "Difficult — may go undetected without dedicated monitoring",
    "high": "Moderate — detectable with standard SIEM alerting",
    "medium": "Easy — typically surfaced by routine log review",
    "low": "Easy — visible through standard audit procedures",
}

_BASE_DATE = date(2021, 3, 1)

_FAMILIES = list(SECURITY_POLICY_SPECS.keys())
_DEPRECATED_INDICES = {3, 8}


class SecurityPolicyGenerator:
    """Generates Security Policy documents — 4 families × 5 policies = 20 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler

    def generate_one(self, index: int, _all_doc_ids: list[str]) -> EnterpriseDocument:
        family_idx = (index - 1) // 5
        policy_idx = (index - 1) % 5
        family_key = _FAMILIES[family_idx % len(_FAMILIES)]
        spec = SECURITY_POLICY_SPECS[family_key][policy_idx]

        doc_id = f"SEC-POL-{index:03d}"
        risk_level_str = spec["risk_level"]
        control_ids: list[str] = spec["control_ids"]
        exception_allowed: bool = spec["exception_allowed"]

        is_deprecated = index in _DEPRECATED_INDICES
        status = DocumentStatus.DEPRECATED if is_deprecated else DocumentStatus.ACTIVE

        review_days = _REVIEW_DAYS[risk_level_str]
        remediation_days = _REMEDIATION_DAYS[risk_level_str]
        max_exception_days = _MAX_EXCEPTION_DAYS[risk_level_str]

        eff_date = _BASE_DATE + timedelta(days=(index - 1) * 50)
        created_dt = datetime(eff_date.year, eff_date.month, eff_date.day, 10, 0, 0)
        updated_dt = created_dt + timedelta(days=self._rng.randint(0, 45))

        version = f"2.{policy_idx}" if is_deprecated else f"1.{policy_idx}"
        enforcement_owner = self._rng.choice(ROLE_NAMES[6:9])
        author_role = "Chief Information Security Officer"

        control_rows = [
            {
                "id": cid,
                "requirement": spec["mandatory_requirements"][i % len(spec["mandatory_requirements"])],
                "status": "Active" if not is_deprecated else "Deprecated",
            }
            for i, cid in enumerate(control_ids)
        ]

        title = spec["title"]
        if is_deprecated:
            title = f"{title} (Deprecated)"

        rev_history = [
            {
                "version": "1.0",
                "date": str(eff_date - timedelta(days=60)),
                "author_role": author_role,
                "change": "Initial release",
            }
        ]
        if is_deprecated:
            rev_history.append({
                "version": version,
                "date": str(eff_date),
                "author_role": author_role,
                "change": "Superseded by updated policy; marked deprecated",
            })

        context = {
            "title": title,
            "security_policy_id": doc_id,
            "document_owner": author_role,
            "department": self._rng.choice(DEPARTMENTS["Security Policies"]),
            "effective_date": str(eff_date),
            "version": version,
            "status": status.value,
            "control_family": family_key.replace("_", " ").title(),
            "risk_level": risk_level_str,
            "control_ids": control_ids,
            "review_frequency_days": review_days,
            "enforcement_owner": enforcement_owner,
            "control_rows": control_rows,
            "risk_likelihood": _RISK_LIKELIHOOD[risk_level_str],
            "risk_impact": _RISK_IMPACT[risk_level_str],
            "detection_difficulty": _DETECTION_DIFFICULTY[risk_level_str],
            "mandatory_requirements": spec["mandatory_requirements"],
            "prohibited_actions": spec["prohibited_actions"],
            "exception_allowed": exception_allowed,
            "max_exception_days": max_exception_days,
            "remediation_days": remediation_days,
            "revision_history": rev_history,
        }

        body = self._assembler.render("security_policy.j2", context)

        category_metadata = SecurityPolicyMetadata(
            security_policy_id=doc_id,
            control_family=_FAMILY_MAP[family_key],
            control_ids=control_ids,
            risk_level=_RISK_MAP[risk_level_str],
            enforcement_owner=enforcement_owner,
            exception_allowed=exception_allowed,
            review_frequency_days=review_days,
        )

        dept = self._rng.choice(DEPARTMENTS["Security Policies"])
        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.SECURITY_POLICY,
            title=title,
            version=version,
            created_at=created_dt,
            updated_at=updated_dt,
            department=dept,
            document_owner=author_role,
            confidentiality=ConfidentialityLevel.RESTRICTED,
            region=Region.GLOBAL,
            language="en",
            effective_date=eff_date,
            review_cycle=ReviewCycle.QUARTERLY if risk_level_str in ("critical", "high") else ReviewCycle.SEMIANNUAL,
            status=status,
            authority_level=AuthorityLevel.MANDATORY,
            tags=[family_key, risk_level_str, "security", "policy"],
            source_path=f"/data/security/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body)
