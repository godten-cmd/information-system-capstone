"""Travel Policy document generator."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from .._assembler import DocumentAssembler
from .._facts import DEPARTMENTS, ROLE_NAMES, TRAVEL_EXPENSE_CONFIGS
from .._rng import SeededRNG
from ..enums import (
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    Region,
    ReviewCycle,
    TravelCurrency,
    TravelExpenseType,
)
from ..schemas import DocumentMetadata, EnterpriseDocument, TravelPolicyMetadata

_CURRENCY_MAP: dict[str, TravelCurrency] = {
    "USD": TravelCurrency.USD,
    "EUR": TravelCurrency.EUR,
    "KRW": TravelCurrency.KRW,
    "SGD": TravelCurrency.SGD,
}

_EXPENSE_MAP: dict[str, TravelExpenseType] = {
    "lodging": TravelExpenseType.LODGING,
    "meals": TravelExpenseType.MEALS,
    "airfare": TravelExpenseType.AIRFARE,
    "ground_transport": TravelExpenseType.GROUND_TRANSPORT,
    "visa": TravelExpenseType.VISA,
    "miscellaneous": TravelExpenseType.MISCELLANEOUS,
}

_REGION_MAP: dict[str, Region] = {
    "US": Region.US,
    "EU": Region.EU,
    "KR": Region.KR,
    "SG": Region.SG,
    "global": Region.GLOBAL,
}

_BASE_DATE = date(2021, 9, 1)

_NON_REIMBURSABLE: dict[str, list[str]] = {
    "lodging": [
        "Hotel room service and minibar charges",
        "Personal laundry or dry-cleaning services",
        "In-room entertainment not required for business",
    ],
    "meals": [
        "Alcohol purchased for personal consumption",
        "Meals for non-HYTech guests not on approved list",
        "Delivery service fees exceeding 15% of the meal cost",
    ],
    "airfare": [
        "Seat upgrade costs beyond the approved class",
        "Travel insurance not pre-approved by Finance",
        "Airport lounge access (except for Executive role-holders)",
    ],
    "ground_transport": [
        "Tolls for personal detours unrelated to the business destination",
        "Parking fines or traffic violations",
        "Car rental upgrade costs beyond the standard category",
    ],
    "visa": [
        "Expedite fees not pre-approved by Finance",
        "Passport renewal costs",
    ],
    "miscellaneous": [
        "Personal entertainment expenses",
        "Gifts or gratuities not covered by a separate policy",
    ],
}


class TravelPolicyGenerator:
    """Generates Travel Policy documents — one per expense config = 15 docs."""

    def __init__(self, rng: SeededRNG, assembler: DocumentAssembler) -> None:
        self._rng = rng
        self._assembler = assembler

    def generate_one(self, index: int, _all_doc_ids: list[str]) -> EnterpriseDocument:
        config_idx = (index - 1) % len(TRAVEL_EXPENSE_CONFIGS)
        config = TRAVEL_EXPENSE_CONFIGS[config_idx]

        doc_id = f"TRV-POL-{index:03d}"
        currency_str = config["currency"]
        expense_type_str = config["expense_type"]
        primary_region_str = config["primary_region"]
        approval_required: bool = config["approval_required"]
        receipt_threshold: float = config["receipt_threshold"]
        reimbursement_days: int = config["reimbursement_days"]

        currency = _CURRENCY_MAP[currency_str]
        expense_type = _EXPENSE_MAP[expense_type_str]
        region = _REGION_MAP.get(primary_region_str, Region.GLOBAL)

        eff_date = _BASE_DATE + timedelta(days=(index - 1) * 60)
        created_dt = datetime(eff_date.year, eff_date.month, eff_date.day, 9, 0, 0)
        updated_dt = created_dt + timedelta(days=self._rng.randint(0, 20))

        version = f"1.{(index - 1) % 4}"
        finance_owner = "Finance Controller"
        author_role = self._rng.choice(ROLE_NAMES[5:8])

        role_limits: dict = config["role_limits"]
        spending_limit_rows = []
        for role, limit in role_limits.items():
            spending_limit_rows.append({
                "expense_type": expense_type_str.replace("_", " ").title(),
                "role": role.replace("_", " ").title(),
                "region": primary_region_str.upper(),
                "limit": limit,
                "approval_required": "Yes" if approval_required else "No",
            })

        senior_threshold = max(role_limits.values()) * 1.2

        non_reimbursable = _NON_REIMBURSABLE.get(expense_type_str, _NON_REIMBURSABLE["miscellaneous"])
        title_region = primary_region_str.upper() if primary_region_str != "global" else "Global"
        title = f"{expense_type_str.replace('_', ' ').title()} Expense Policy – {title_region} ({currency_str})"

        rev_history = [
            {
                "version": "1.0",
                "date": str(eff_date - timedelta(days=30)),
                "author_role": author_role,
                "change": "Initial release",
            }
        ]
        if index % 3 == 0:
            rev_history.append({
                "version": version,
                "date": str(eff_date),
                "author_role": finance_owner,
                "change": f"Updated spending limits for {currency_str} region",
            })

        context = {
            "title": title,
            "travel_policy_id": doc_id,
            "document_owner": author_role,
            "department": DEPARTMENTS["Travel Policies"][0],
            "effective_date": str(eff_date),
            "version": version,
            "status": DocumentStatus.ACTIVE.value,
            "currency": currency_str,
            "expense_type": expense_type_str.replace("_", " ").lower(),
            "approval_required": approval_required,
            "receipt_required_threshold": receipt_threshold,
            "receipt_upload_days": 30,
            "spending_limit_rows": spending_limit_rows,
            "submission_deadline_days": 30,
            "digital_receipt_accepted": True,
            "senior_approval_threshold": round(senior_threshold),
            "pre_approval_lead_days": 3,
            "reimbursement_timeline_days": reimbursement_days,
            "finance_owner": finance_owner,
            "non_reimbursable": non_reimbursable,
            "audit_retention_years": 7,
            "revision_history": rev_history,
        }

        body = self._assembler.render("travel_policy.j2", context)

        category_metadata = TravelPolicyMetadata(
            travel_policy_id=doc_id,
            expense_type=expense_type,
            applicable_roles=list(role_limits.keys()),
            currency=currency,
            approval_required=approval_required,
            finance_owner=finance_owner,
            receipt_required_threshold=float(receipt_threshold),
        )

        dept = DEPARTMENTS["Travel Policies"][0]
        metadata = DocumentMetadata(
            document_id=doc_id,
            category=DocumentCategory.TRAVEL_POLICY,
            title=title,
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
            status=DocumentStatus.ACTIVE,
            authority_level=AuthorityLevel.MANDATORY,
            tags=[expense_type_str, currency_str.lower(), "travel", "expense"],
            source_path=f"/data/finance/{doc_id}.md",
            category_metadata=category_metadata,
        )

        return self._assembler.build_document(metadata, body)
