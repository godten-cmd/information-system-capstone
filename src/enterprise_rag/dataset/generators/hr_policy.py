"""HR Policy document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import (
    APPROVAL_BODIES,
    DEPARTMENTS,
    HR_ELIGIBILITY_INTROS,
    HR_EXCEPTION_POOL,
    HR_POLICY_TOPICS,
    HR_PROCEDURE_STEPS,
    HR_PURPOSE_TEMPLATES,
    ROLE_NAMES,
)
from .._rng import SeededRNG
from ..enums import (
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    EmployeeType,
    PolicyArea,
    Region,
    ReviewCycle,
)
from ..schemas import DocumentMetadata, EnterpriseDocument, HRPolicyMetadata

_REGIONS = [Region.US, Region.EU, Region.KR, Region.SG, Region.GLOBAL]
_VARIANT_REGIONS: list[tuple[Region, Region]] = [
    (Region.US, Region.EU),
    (Region.KR, Region.SG),
    (Region.EU, Region.US),
    (Region.GLOBAL, Region.KR),
    (Region.US, Region.KR),
    (Region.GLOBAL, Region.EU),
    (Region.EU, Region.KR),
    (Region.GLOBAL, Region.US),
    (Region.US, Region.SG),
    (Region.KR, Region.EU),
]

_POLICY_AREA_MAP: dict[str, PolicyArea] = {
    "leave": PolicyArea.LEAVE,
    "benefits": PolicyArea.BENEFITS,
    "remote_work": PolicyArea.REMOTE_WORK,
    "performance": PolicyArea.PERFORMANCE,
    "conduct": PolicyArea.CONDUCT,
}

_EMPLOYEE_TYPE_MAP: dict[str, EmployeeType] = {
    "contractor": EmployeeType.CONTRACTOR,
    "intern": EmployeeType.INTERN,
    "part_time": EmployeeType.PART_TIME,
}

_BASE_DATE = date(2021, 6, 1)


class HRPolicyGenerator:
    """Generates HR Policy documents — 10 topics × 2 regional variants = 20 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler

    def generate_one(self, index: int, _all_doc_ids: list[str]) -> EnterpriseDocument:
        topic_idx = (index - 1) % len(HR_POLICY_TOPICS)
        variant = (index - 1) // len(HR_POLICY_TOPICS)
        topic = HR_POLICY_TOPICS[topic_idx]
        region = _VARIANT_REGIONS[topic_idx][min(variant, 1)]

        doc_id = f"HR-POL-{index:03d}"
        supersedes = f"HR-POL-{index - 2:03d}" if index % 3 == 0 and index >= 3 else None
        status = DocumentStatus.ACTIVE

        eff_date = _BASE_DATE + timedelta(days=(index - 1) * 55)
        created_dt = datetime(eff_date.year, eff_date.month, eff_date.day, 9, 0, 0)
        updated_dt = created_dt + timedelta(days=self._rng.randint(0, 30))

        version = f"1.{(index - 1) % 5}"
        author_role = self._rng.choice(ROLE_NAMES[:5])
        approval_body = self._rng.choice(APPROVAL_BODIES[:2])

        policy_area_str = topic["policy_area"]
        policy_area = _POLICY_AREA_MAP.get(policy_area_str, PolicyArea.CONDUCT)

        entitlement_value = topic["entitlement_value"]
        entitlement_unit = topic["entitlement_unit"] or "N/A"

        region_overrides: dict = topic.get("region_overrides", {})
        region_key = region.value
        has_region_clause = (
            region_key in region_overrides
            and region_overrides[region_key] != entitlement_value
        )
        regional_value = region_overrides.get(region_key, entitlement_value)

        has_enhanced = (
            has_region_clause
            and regional_value is not None
            and entitlement_value is not None
            and regional_value > entitlement_value
        )

        purpose = self._rng.choice(HR_PURPOSE_TEMPLATES).format(name=topic["name"])
        eligibility_intro = self._rng.choice(HR_ELIGIBILITY_INTROS)

        proc_steps_raw = HR_PROCEDURE_STEPS.get(policy_area_str, HR_PROCEDURE_STEPS["leave"])
        notice_days = self._rng.randint(2, 5)
        steps = [s.format(notice_days=notice_days) for s in proc_steps_raw]

        exception_roles = topic.get("exception_roles", [])
        has_exceptions = bool(exception_roles)
        exceptions = [
            e for e in HR_EXCEPTION_POOL
            if any(r in e.lower() for r in exception_roles)
        ]
        if not exceptions and has_exceptions:
            exceptions = HR_EXCEPTION_POOL[:2]

        applicable_regions = [region.value]

        stmt_prefix = "a standard entitlement of" if entitlement_value else "the standards and procedures for"
        entitlement_display = f"{entitlement_value} {entitlement_unit}" if entitlement_value else topic["name"]
        policy_statement = (
            f"All eligible HYTech Solutions employees are entitled to {stmt_prefix} "
            f"{entitlement_display}. "
            f"This entitlement is subject to the eligibility criteria and approval procedures "
            f"described in this policy."
        )

        rev_history = [
            {"version": "1.0", "date": str(eff_date - timedelta(days=30)), "author_role": author_role, "change": "Initial release"},
        ]
        if index % 2 == 0:
            rev_history.append({
                "version": version,
                "date": str(eff_date),
                "author_role": self._rng.choice(ROLE_NAMES[:5]),
                "change": "Updated eligibility criteria and regional provisions",
            })

        context = {
            "title": f"{topic['name']} Policy – {region.value}",
            "policy_id": doc_id,
            "document_owner": author_role,
            "department": "Human Resources",
            "effective_date": str(eff_date),
            "version": version,
            "status": status.value,
            "applicable_regions": applicable_regions,
            "approval_body": approval_body,
            "supersedes": supersedes,
            "purpose": purpose,
            "policy_area": policy_area.value,
            "employee_type": EmployeeType.ALL.value,
            "eligibility_intro": eligibility_intro,
            "min_tenure_months": topic["min_tenure_months"],
            "has_region_clause": has_region_clause,
            "region_clause_region": region_key,
            "policy_statement": policy_statement,
            "entitlement_value": entitlement_value if entitlement_value is not None else "N/A",
            "entitlement_unit": entitlement_unit,
            "has_enhanced_entitlement": has_enhanced,
            "enhanced_region": region_key,
            "enhanced_value": regional_value,
            "steps": steps,
            "has_exceptions": has_exceptions,
            "exception_intro": "The following exceptions apply to the standard policy:",
            "exceptions": exceptions,
            "exception_retention_years": 7,
            "manager_approval_days": 5,
            "escalation_contact": "HR Operations Lead",
            "notice_days": notice_days,
            "revision_history": rev_history,
        }

        body = self._assembler.render("hr_policy.j2", context)

        category_metadata = HRPolicyMetadata(
            policy_id=doc_id,
            policy_area=policy_area,
            employee_type=EmployeeType.ALL,
            applicable_regions=[region],
            approval_body=approval_body,
            supersedes=supersedes,
            escalation_contact="HR Operations Lead",
        )

        dept = DEPARTMENTS["HR Policies"][0]
        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.HR_POLICY,
            title=context["title"],
            version=version,
            created_at=created_dt,
            updated_at=updated_dt,
            department=dept,
            document_owner=author_role,
            confidentiality=ConfidentialityLevel.CONFIDENTIAL,
            region=region,
            language="en",
            effective_date=eff_date,
            review_cycle=ReviewCycle.ANNUAL,
            status=status,
            authority_level=AuthorityLevel.MANDATORY,
            tags=[policy_area.value, region.value, "hr", "policy"],
            source_path=f"/data/hr/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body)
