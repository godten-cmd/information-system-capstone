"""Per-category query recipe definitions and unanswerable query specs."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class UnanswerableSpec:
    """Specification for a query whose answer is not present in the SEKD corpus."""

    query_text: str
    evaluation_notes: str
    disallowed_claims: list[str] = field(default_factory=list)
    category_hint: str = "general"


# ── Unanswerable queries ──────────────────────────────────────────────────────
# These reference topics, policies, services, or facts that do not exist in
# the generated SEKD corpus.  They are assigned difficulty=adversarial.

UNANSWERABLE_SPECS: list[UnanswerableSpec] = [
    UnanswerableSpec(
        query_text="What is the surrogacy leave entitlement at HYTech Solutions?",
        evaluation_notes="Surrogacy leave is not a topic covered in the HR policy corpus.",
        disallowed_claims=["surrogacy", "days", "weeks"],
        category_hint="HR Policies",
    ),
    UnanswerableSpec(
        query_text="What is the pet relocation reimbursement policy at HYTech Solutions?",
        evaluation_notes="Pet relocation expenses are not covered in any travel or HR policy.",
        disallowed_claims=["pet", "relocation", "reimbursement amount"],
        category_hint="HR Policies",
    ),
    UnanswerableSpec(
        query_text="How many days of volunteer leave are full-time employees entitled to?",
        evaluation_notes="Volunteer leave is not defined in any SEKD policy document.",
        disallowed_claims=["volunteer", "days"],
        category_hint="HR Policies",
    ),
    UnanswerableSpec(
        query_text="What is the salary band for Platform Architects at HYTech Solutions?",
        evaluation_notes="Salary bands are not included in any SEKD document.",
        disallowed_claims=["salary", "band", "compensation"],
        category_hint="HR Policies",
    ),
    UnanswerableSpec(
        query_text="What is the remote work allowance for employees in Australia?",
        evaluation_notes="Australia is not a supported region in SEKD. Regions are KR, SG, US, EU, global.",
        disallowed_claims=["Australia", "AUD"],
        category_hint="HR Policies",
    ),
    UnanswerableSpec(
        query_text="What is the daily meal allowance for employees in Japan in JPY?",
        evaluation_notes="JPY currency and Japan region are not in any travel policy.",
        disallowed_claims=["JPY", "Japan", "¥"],
        category_hint="Travel Policies",
    ),
    UnanswerableSpec(
        query_text="What is the ground transport limit for executives in GBP?",
        evaluation_notes="GBP is not a currency covered in any travel policy document.",
        disallowed_claims=["GBP", "pounds", "sterling"],
        category_hint="Travel Policies",
    ),
    UnanswerableSpec(
        query_text="What is the premium economy flight policy for cross-continental travel?",
        evaluation_notes="Premium economy class and cross-continental specific rules are not defined.",
        disallowed_claims=["premium economy"],
        category_hint="Travel Policies",
    ),
    UnanswerableSpec(
        query_text="What does control HY-AC-15 require?",
        evaluation_notes="Control HY-AC-15 does not exist. Access Control controls only go up to HY-AC-10.",
        disallowed_claims=["HY-AC-15"],
        category_hint="Security Policies",
    ),
    UnanswerableSpec(
        query_text="What is the escalation timeline for low-risk incidents in the Network Security domain?",
        evaluation_notes="Network Security is not one of the four SEKD security control families.",
        disallowed_claims=["network security", "network"],
        category_hint="Security Policies",
    ),
    UnanswerableSpec(
        query_text="What are the requirements for quantum-safe encryption at HYTech Solutions?",
        evaluation_notes="Quantum-safe encryption is not mentioned in any security policy.",
        disallowed_claims=["quantum", "post-quantum"],
        category_hint="Security Policies",
    ),
    UnanswerableSpec(
        query_text="Is the HYID v4 API available for production use?",
        evaluation_notes="HYID v4 does not exist. The highest documented version is v3 (experimental).",
        disallowed_claims=["v4", "version 4"],
        category_hint="API Documentation",
    ),
    UnanswerableSpec(
        query_text="What is the HYCloud API endpoint for provisioning virtual machines?",
        evaluation_notes="HYCloud is not one of the four documented API services (HYID, HYDeploy, HYMonitor, HYDataLake).",
        disallowed_claims=["HYCloud", "virtual machine"],
        category_hint="API Documentation",
    ),
    UnanswerableSpec(
        query_text="What authentication method does the HYBilling API use?",
        evaluation_notes="HYBilling is not a documented API service in the SEKD corpus.",
        disallowed_claims=["HYBilling"],
        category_hint="API Documentation",
    ),
    UnanswerableSpec(
        query_text="What is the latency SLA for the HYDataLake real-time query service?",
        evaluation_notes="Real-time CDC replication and sub-second latency are explicitly out of scope for HYDataLake.",
        disallowed_claims=["real-time", "milliseconds", "sub-second"],
        category_hint="System Design Documents",
    ),
    UnanswerableSpec(
        query_text="What is the architecture of the HYSearch full-text indexing system?",
        evaluation_notes="HYSearch does not exist in the SEKD system design documents.",
        disallowed_claims=["HYSearch"],
        category_hint="System Design Documents",
    ),
    UnanswerableSpec(
        query_text="What were the Q3 2022 revenue figures discussed in the finance meeting?",
        evaluation_notes="Revenue figures are not tracked in any SEKD meeting notes.",
        disallowed_claims=["revenue", "Q3 2022", "financial results"],
        category_hint="Project Meeting Notes",
    ),
    UnanswerableSpec(
        query_text="What was decided in the PRJ-CLOUD-INFRA sprint planning meeting?",
        evaluation_notes="PRJ-CLOUD-INFRA does not exist. The five projects are PRJ-HYD-ALPHA, PRJ-HYID-BETA, PRJ-SECURITY-AUDIT, PRJ-DATALAKE-V2, PRJ-BILLING-REVAMP.",
        disallowed_claims=["PRJ-CLOUD-INFRA"],
        category_hint="Project Meeting Notes",
    ),
    UnanswerableSpec(
        query_text="Who is the CEO of HYTech Solutions?",
        evaluation_notes="Executive identity information is not present in any SEKD document.",
        disallowed_claims=["CEO", "chief executive"],
        category_hint="general",
    ),
    UnanswerableSpec(
        query_text="What is the stock ticker symbol for HYTech Solutions?",
        evaluation_notes="HYTech Solutions is a synthetic company with no publicly listed stock.",
        disallowed_claims=["ticker", "NASDAQ", "NYSE"],
        category_hint="general",
    ),
    UnanswerableSpec(
        query_text="What was discussed at the all-hands meeting on March 15, 2023?",
        evaluation_notes="All-hands meeting notes are not a document type in the SEKD corpus.",
        disallowed_claims=["all-hands"],
        category_hint="Project Meeting Notes",
    ),
    UnanswerableSpec(
        query_text="What is the approved VPN software for home office use?",
        evaluation_notes="VPN software selection is not documented in any SEKD policy.",
        disallowed_claims=["VPN software", "client", "app"],
        category_hint="Security Policies",
    ),
    UnanswerableSpec(
        query_text="How many employees does HYTech Solutions have in Berlin?",
        evaluation_notes="Employee headcount by office is not available in any SEKD document.",
        disallowed_claims=["headcount", "employees in Berlin"],
        category_hint="general",
    ),
    UnanswerableSpec(
        query_text="What is the expense limit for helicopter charter bookings?",
        evaluation_notes="Helicopter charter is not covered by any travel policy.",
        disallowed_claims=["helicopter"],
        category_hint="Travel Policies",
    ),
    UnanswerableSpec(
        query_text="What is the GDPR data subject access request response time for HYTech Solutions?",
        evaluation_notes="GDPR-specific response timelines are not defined in the SEKD security policies.",
        disallowed_claims=["GDPR", "data subject access request"],
        category_hint="Security Policies",
    ),
]


# ── Shared search patterns ────────────────────────────────────────────────────

# Regex anchors used by per-category extractors.  Each tuple is
# (anchor_text, stop_char_or_string) for prefix-match extraction.

HR_ENTITLEMENT_ANCHOR = "The standard entitlement under this policy is "
HR_ENHANCED_ANCHOR = " receive an enhanced entitlement of "
HR_TENURE_ANCHOR = "Minimum tenure: "
HR_EXCEPTION_ANCHOR = "Contractors engaged through third-party agencies are not eligible"
HR_MANAGER_ANCHOR = "Approving requests within "

TRAVEL_RECEIPT_ANCHOR = "Receipts are mandatory for all expenses exceeding "
TRAVEL_REIMBURSEMENT_ANCHOR = "Approved claims are reimbursed within "

SECURITY_ESCALATION_TABLE_ANCHOR = "| Critical | 1 hour |"
SECURITY_NO_EXCEPTIONS_ANCHOR = "No exceptions are permitted for"
SECURITY_EXCEPTIONS_ALLOWED_ANCHOR = "Exceptions to this policy may be approved"

API_DEPRECATION_ANCHOR = "DEPRECATION NOTICE:"
API_BASE_URL_ANCHOR = "Base URL: "
API_AUTH_METHOD_ANCHOR = "Authentication method: "

SYSDES_SLA_ANCHOR = "Target SLA: "
SYSDES_DATA_VOLUME_ANCHOR = "Expected data volume: "

MEETING_OVERVIEW_ANCHOR = "Project: "
MEETING_DECISION_ANCHOR = "1. **"
