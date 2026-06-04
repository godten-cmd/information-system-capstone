"""Controlled fact pools for all six SEKD document categories.

This module contains only pure data constants — no logic, no imports from
other enterprise_rag modules. Every numeric threshold, policy ID pool,
endpoint path, and role name that appears in generated documents originates
here, making all facts traceable for Phase 4 ground-truth generation.
"""

from __future__ import annotations

# ── Shared role names ─────────────────────────────────────────────────────────

ROLE_NAMES: list[str] = [
    "HR Operations Lead",
    "People Partner",
    "Compensation Manager",
    "Talent Acquisition Lead",
    "Employee Relations Manager",
    "Finance Controller",
    "Security Engineer",
    "Platform Architect",
    "Engineering Manager",
    "Product Manager",
    "Solutions Architect",
    "Site Reliability Engineer",
    "Legal Counsel",
    "Compliance Officer",
    "Data Engineering Lead",
    "DevOps Engineer",
    "Technical Program Manager",
    "Chief Information Security Officer",
    "Head of Finance",
    "VP of Engineering",
]

APPROVAL_BODIES: list[str] = [
    "HR Committee",
    "Executive Office",
    "Legal and Compliance",
    "Finance Committee",
    "Security Council",
    "Architecture Review Board",
]

DEPARTMENTS: dict[str, list[str]] = {
    "HR Policies": ["Human Resources"],
    "Travel Policies": ["Finance", "Human Resources"],
    "Security Policies": ["Security", "IT Operations"],
    "API Documentation": ["Platform Engineering", "Product Engineering"],
    "System Design Documents": ["Platform Engineering", "Solutions Engineering"],
    "Project Meeting Notes": [
        "Platform Engineering",
        "Product Engineering",
        "Human Resources",
        "Security",
        "Finance",
    ],
}

# ── HR Policy topics ──────────────────────────────────────────────────────────

HR_POLICY_TOPICS: list[dict] = [
    {
        "slug": "annual_leave",
        "name": "Annual Leave",
        "policy_area": "leave",
        "entitlement_value": 15,
        "entitlement_unit": "business days per calendar year",
        "min_tenure_months": 3,
        "reimbursement_cap": None,
        "exception_roles": ["contractor", "intern"],
        "region_overrides": {"KR": 15, "SG": 14, "US": 10, "EU": 20},
    },
    {
        "slug": "sick_leave",
        "name": "Sick Leave",
        "policy_area": "leave",
        "entitlement_value": 10,
        "entitlement_unit": "business days per calendar year",
        "min_tenure_months": 0,
        "reimbursement_cap": None,
        "exception_roles": ["contractor"],
        "region_overrides": {"KR": 10, "SG": 14, "US": 5, "EU": 15},
    },
    {
        "slug": "parental_leave",
        "name": "Parental Leave",
        "policy_area": "leave",
        "entitlement_value": 90,
        "entitlement_unit": "calendar days",
        "min_tenure_months": 12,
        "reimbursement_cap": None,
        "exception_roles": ["contractor", "intern", "part_time"],
        "region_overrides": {"KR": 90, "SG": 60, "US": 12, "EU": 112},
    },
    {
        "slug": "remote_work",
        "name": "Remote Work",
        "policy_area": "remote_work",
        "entitlement_value": 2,
        "entitlement_unit": "remote days per week",
        "min_tenure_months": 6,
        "reimbursement_cap": 500,
        "exception_roles": ["intern"],
        "region_overrides": {},
    },
    {
        "slug": "training_reimbursement",
        "name": "Training and Development Reimbursement",
        "policy_area": "benefits",
        "entitlement_value": 1500,
        "entitlement_unit": "USD per year",
        "min_tenure_months": 6,
        "reimbursement_cap": 1500,
        "exception_roles": ["contractor", "intern"],
        "region_overrides": {"KR": 1000, "SG": 1200, "US": 1500, "EU": 1200},
    },
    {
        "slug": "performance_review",
        "name": "Performance Review",
        "policy_area": "performance",
        "entitlement_value": 2,
        "entitlement_unit": "review cycles per year",
        "min_tenure_months": 3,
        "reimbursement_cap": None,
        "exception_roles": [],
        "region_overrides": {},
    },
    {
        "slug": "overtime",
        "name": "Overtime Compensation",
        "policy_area": "benefits",
        "entitlement_value": 150,
        "entitlement_unit": "percent of hourly rate",
        "min_tenure_months": 0,
        "reimbursement_cap": None,
        "exception_roles": ["contractor"],
        "region_overrides": {"KR": 150, "EU": 125},
    },
    {
        "slug": "employee_conduct",
        "name": "Employee Conduct and Disciplinary",
        "policy_area": "conduct",
        "entitlement_value": None,
        "entitlement_unit": None,
        "min_tenure_months": 0,
        "reimbursement_cap": None,
        "exception_roles": [],
        "region_overrides": {},
    },
    {
        "slug": "promotion",
        "name": "Promotion and Career Progression",
        "policy_area": "performance",
        "entitlement_value": 12,
        "entitlement_unit": "months minimum tenure before promotion eligibility",
        "min_tenure_months": 12,
        "reimbursement_cap": None,
        "exception_roles": ["contractor", "intern"],
        "region_overrides": {},
    },
    {
        "slug": "bereavement_leave",
        "name": "Bereavement Leave",
        "policy_area": "leave",
        "entitlement_value": 5,
        "entitlement_unit": "business days per bereavement event",
        "min_tenure_months": 0,
        "reimbursement_cap": None,
        "exception_roles": ["contractor"],
        "region_overrides": {"EU": 7, "KR": 5, "SG": 5, "US": 3},
    },
]

HR_ELIGIBILITY_INTROS: list[str] = [
    "All eligible employees must meet the following criteria to access this entitlement.",
    "Employees who meet the conditions below are entitled to the benefits described in this policy.",
    "Eligibility is determined by employment type, tenure, and region as specified below.",
]

HR_PURPOSE_TEMPLATES: list[str] = [
    (
        "The purpose of this policy is to define the entitlements and procedures for {name} "
        "at HYTech Solutions, ensuring consistency, fairness, and compliance with applicable "
        "regional employment regulations."
    ),
    (
        "This policy establishes the framework for {name} within HYTech Solutions, providing "
        "clear guidance to employees, managers, and the HR team on entitlements, procedures, "
        "and escalation paths."
    ),
    (
        "HYTech Solutions is committed to supporting its workforce through transparent {name} "
        "practices. This document outlines the applicable rules, entitlements, and approval "
        "workflows that govern this area."
    ),
]

HR_EXCEPTION_POOL: list[str] = [
    "Contractors engaged through third-party agencies are not eligible unless explicitly stated in the engagement contract.",
    "Interns on fixed-term agreements of less than six months are not eligible for this entitlement.",
    "Exceptions require written pre-approval from the HR Operations Lead and the employee's direct manager.",
    "Employees on a performance improvement plan are not eligible during the PIP period.",
    "In regions where local law grants a higher entitlement, the higher legal minimum shall apply.",
    "Part-time employees receive pro-rated entitlements based on their contracted hours.",
    "Employees who have given or received notice of termination are not eligible for new requests under this policy.",
]

HR_PROCEDURE_STEPS: dict[str, list[str]] = {
    "leave": [
        "Submit a leave request through HYPortal at least {notice_days} business days before the intended start date.",
        "Obtain line manager approval within HYPortal.",
        "HR Operations will confirm the leave balance deduction within 2 business days.",
        "Upon return, update your attendance record in HYPortal.",
    ],
    "benefits": [
        "Submit a reimbursement request through HYPortal with receipts or supporting documentation.",
        "Line manager reviews and approves the request within 5 business days.",
        "Finance processes the reimbursement in the next payroll cycle.",
        "Retain all original receipts for 2 years for audit purposes.",
    ],
    "remote_work": [
        "Discuss and agree on your remote work schedule with your line manager.",
        "Update your working location in HYPortal at least 2 business days before the first remote day.",
        "Ensure you have a reliable internet connection and a secure working environment.",
        "Attend all mandatory in-person meetings as required by your team.",
    ],
    "performance": [
        "Schedule the performance review meeting with your line manager through HYPortal.",
        "Complete self-assessment in HYPortal at least 5 business days before the review date.",
        "Line manager submits written feedback and ratings in HYPortal.",
        "HR Operations reviews and archives the completed review within 5 business days.",
    ],
    "conduct": [
        "The reporting manager documents the conduct concern in writing.",
        "HR Operations conducts a preliminary review within 5 business days.",
        "A formal meeting is scheduled with the employee and HR representative.",
        "Outcome and any disciplinary action are communicated in writing.",
        "Employee may appeal within 10 business days of the decision.",
    ],
}

# ── Travel Policy configs ─────────────────────────────────────────────────────

TRAVEL_EXPENSE_CONFIGS: list[dict] = [
    {
        "slug": "lodging_usd",
        "expense_type": "lodging",
        "currency": "USD",
        "primary_region": "US",
        "role_limits": {
            "employee": 150,
            "manager": 200,
            "executive": 350,
            "sales": 200,
            "engineer": 150,
        },
        "approval_required": False,
        "receipt_threshold": 50,
        "reimbursement_days": 30,
    },
    {
        "slug": "lodging_eur",
        "expense_type": "lodging",
        "currency": "EUR",
        "primary_region": "EU",
        "role_limits": {
            "employee": 120,
            "manager": 160,
            "executive": 280,
            "sales": 160,
            "engineer": 120,
        },
        "approval_required": False,
        "receipt_threshold": 40,
        "reimbursement_days": 30,
    },
    {
        "slug": "lodging_krw",
        "expense_type": "lodging",
        "currency": "KRW",
        "primary_region": "KR",
        "role_limits": {
            "employee": 120000,
            "manager": 180000,
            "executive": 350000,
            "sales": 180000,
            "engineer": 120000,
        },
        "approval_required": False,
        "receipt_threshold": 50000,
        "reimbursement_days": 30,
    },
    {
        "slug": "lodging_sgd",
        "expense_type": "lodging",
        "currency": "SGD",
        "primary_region": "SG",
        "role_limits": {
            "employee": 200,
            "manager": 280,
            "executive": 500,
            "sales": 280,
            "engineer": 200,
        },
        "approval_required": False,
        "receipt_threshold": 60,
        "reimbursement_days": 30,
    },
    {
        "slug": "meals_usd",
        "expense_type": "meals",
        "currency": "USD",
        "primary_region": "US",
        "role_limits": {
            "employee": 60,
            "manager": 80,
            "executive": 120,
            "sales": 100,
            "engineer": 60,
        },
        "approval_required": False,
        "receipt_threshold": 25,
        "reimbursement_days": 30,
    },
    {
        "slug": "meals_eur",
        "expense_type": "meals",
        "currency": "EUR",
        "primary_region": "EU",
        "role_limits": {
            "employee": 50,
            "manager": 70,
            "executive": 100,
            "sales": 80,
            "engineer": 50,
        },
        "approval_required": False,
        "receipt_threshold": 20,
        "reimbursement_days": 30,
    },
    {
        "slug": "meals_krw",
        "expense_type": "meals",
        "currency": "KRW",
        "primary_region": "KR",
        "role_limits": {
            "employee": 40000,
            "manager": 60000,
            "executive": 100000,
            "sales": 70000,
            "engineer": 40000,
        },
        "approval_required": False,
        "receipt_threshold": 20000,
        "reimbursement_days": 30,
    },
    {
        "slug": "airfare_usd",
        "expense_type": "airfare",
        "currency": "USD",
        "primary_region": "US",
        "role_limits": {
            "employee": 800,
            "manager": 1200,
            "executive": 3000,
            "sales": 1500,
            "engineer": 1000,
        },
        "approval_required": True,
        "receipt_threshold": 100,
        "reimbursement_days": 30,
    },
    {
        "slug": "airfare_eur",
        "expense_type": "airfare",
        "currency": "EUR",
        "primary_region": "EU",
        "role_limits": {
            "employee": 700,
            "manager": 1000,
            "executive": 2500,
            "sales": 1200,
            "engineer": 850,
        },
        "approval_required": True,
        "receipt_threshold": 80,
        "reimbursement_days": 30,
    },
    {
        "slug": "ground_transport_usd",
        "expense_type": "ground_transport",
        "currency": "USD",
        "primary_region": "US",
        "role_limits": {
            "employee": 50,
            "manager": 75,
            "executive": 120,
            "sales": 75,
            "engineer": 50,
        },
        "approval_required": False,
        "receipt_threshold": 20,
        "reimbursement_days": 30,
    },
    {
        "slug": "ground_transport_eur",
        "expense_type": "ground_transport",
        "currency": "EUR",
        "primary_region": "EU",
        "role_limits": {
            "employee": 40,
            "manager": 60,
            "executive": 100,
            "sales": 60,
            "engineer": 40,
        },
        "approval_required": False,
        "receipt_threshold": 15,
        "reimbursement_days": 30,
    },
    {
        "slug": "ground_transport_krw",
        "expense_type": "ground_transport",
        "currency": "KRW",
        "primary_region": "KR",
        "role_limits": {
            "employee": 30000,
            "manager": 50000,
            "executive": 80000,
            "sales": 50000,
            "engineer": 30000,
        },
        "approval_required": False,
        "receipt_threshold": 10000,
        "reimbursement_days": 30,
    },
    {
        "slug": "visa_usd",
        "expense_type": "visa",
        "currency": "USD",
        "primary_region": "global",
        "role_limits": {
            "employee": 200,
            "manager": 200,
            "executive": 500,
            "sales": 200,
            "engineer": 200,
        },
        "approval_required": True,
        "receipt_threshold": 0,
        "reimbursement_days": 30,
    },
    {
        "slug": "miscellaneous_usd",
        "expense_type": "miscellaneous",
        "currency": "USD",
        "primary_region": "global",
        "role_limits": {
            "employee": 30,
            "manager": 50,
            "executive": 100,
            "sales": 50,
            "engineer": 30,
        },
        "approval_required": False,
        "receipt_threshold": 10,
        "reimbursement_days": 30,
    },
    {
        "slug": "meals_sgd",
        "expense_type": "meals",
        "currency": "SGD",
        "primary_region": "SG",
        "role_limits": {
            "employee": 70,
            "manager": 100,
            "executive": 150,
            "sales": 110,
            "engineer": 70,
        },
        "approval_required": False,
        "receipt_threshold": 30,
        "reimbursement_days": 30,
    },
]

# ── Security Policy configs ───────────────────────────────────────────────────

SECURITY_CONTROL_IDS: dict[str, list[str]] = {
    "access_control": [f"HY-AC-{i:02d}" for i in range(1, 11)],
    "data_protection": [f"HY-DP-{i:02d}" for i in range(1, 11)],
    "incident_response": [f"HY-IR-{i:02d}" for i in range(1, 11)],
    "device_security": [f"HY-DS-{i:02d}" for i in range(1, 11)],
}

# Fixed escalation timelines per risk level (deterministic, not randomised)
SECURITY_ESCALATION_HOURS: dict[str, str] = {
    "critical": "1 hour",
    "high": "4 hours",
    "medium": "24 hours",
    "low": "3 business days",
}

# 5 policies per control family: [critical, high, high, medium, low]
SECURITY_POLICY_SPECS: dict[str, list[dict]] = {
    "access_control": [
        {
            "slug": "mfa_enforcement",
            "title": "Multi-Factor Authentication Enforcement",
            "risk_level": "critical",
            "control_ids": ["HY-AC-01", "HY-AC-02"],
            "objective": (
                "Ensure all HYTech Solutions systems require multi-factor authentication "
                "for all user and service account access."
            ),
            "mandatory_requirements": [
                "All employees must enroll in MFA within 5 business days of account creation.",
                "MFA must use TOTP or hardware security keys; SMS-based MFA is prohibited.",
                "Service accounts must use certificate-based authentication or service tokens.",
                "MFA bypass is prohibited without CISO approval.",
            ],
            "prohibited_actions": [
                "Sharing MFA codes or recovery codes with any other person.",
                "Disabling MFA on any production system without CISO written approval.",
                "Using SMS as a sole MFA factor for accounts with privileged access.",
            ],
            "exception_allowed": False,
        },
        {
            "slug": "privileged_access",
            "title": "Privileged Access Management",
            "risk_level": "high",
            "control_ids": ["HY-AC-03", "HY-AC-04"],
            "objective": (
                "Control and audit all privileged access to HYTech Solutions production "
                "systems and sensitive data repositories."
            ),
            "mandatory_requirements": [
                "All privileged access must be requested through the HYPortal access management workflow.",
                "Privileged access sessions must be recorded and retained for 90 days.",
                "Just-in-time access must be used for all production system changes.",
                "Access reviews must be conducted every 30 days for privileged accounts.",
            ],
            "prohibited_actions": [
                "Sharing privileged credentials with other users.",
                "Using personal accounts for production access.",
                "Retaining privileged access beyond the approved duration.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "password_policy",
            "title": "Password and Credential Management",
            "risk_level": "high",
            "control_ids": ["HY-AC-05", "HY-AC-06"],
            "objective": (
                "Define minimum password standards and credential lifecycle requirements "
                "for all HYTech Solutions accounts."
            ),
            "mandatory_requirements": [
                "Passwords must be at least 14 characters and include uppercase, lowercase, digits, and symbols.",
                "Passwords must be rotated every 90 days for standard accounts.",
                "Service account credentials must be stored in HYVault; plaintext storage is prohibited.",
                "Compromised credentials must be rotated within 1 hour of detection.",
            ],
            "prohibited_actions": [
                "Storing passwords in source code, configuration files, or unencrypted documents.",
                "Reusing any of the last 12 passwords.",
                "Sharing passwords between multiple users or services.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "access_review",
            "title": "User Access Review",
            "risk_level": "medium",
            "control_ids": ["HY-AC-07", "HY-AC-08"],
            "objective": (
                "Ensure that user access rights remain appropriate through regular review "
                "and timely revocation."
            ),
            "mandatory_requirements": [
                "Access reviews must be completed quarterly for all production systems.",
                "Access for terminated employees must be revoked within 1 business day.",
                "Access for role changes must be updated within 5 business days.",
                "Access review results must be documented in HYPortal.",
            ],
            "prohibited_actions": [
                "Approving access reviews without actively verifying the necessity of each access right.",
                "Delaying revocation of access for terminated employees beyond 1 business day.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "sso_enforcement",
            "title": "Single Sign-On Enforcement",
            "risk_level": "low",
            "control_ids": ["HY-AC-09", "HY-AC-10"],
            "objective": (
                "Require all internal SaaS and web applications to use HYID for centralised "
                "authentication where technically feasible."
            ),
            "mandatory_requirements": [
                "All new SaaS integrations must use HYID SSO before go-live.",
                "Applications that cannot support SSO must be submitted for security review.",
                "Local accounts in SSO-integrated applications must be disabled.",
            ],
            "prohibited_actions": [
                "Deploying new applications with local-only authentication without security review approval.",
            ],
            "exception_allowed": True,
        },
    ],
    "data_protection": [
        {
            "slug": "data_classification",
            "title": "Data Classification and Handling",
            "risk_level": "critical",
            "control_ids": ["HY-DP-01", "HY-DP-02"],
            "objective": (
                "Establish mandatory data classification labels and handling requirements "
                "for all HYTech Solutions data assets."
            ),
            "mandatory_requirements": [
                "All data assets must be classified as Public, Internal, Confidential, or Restricted.",
                "Restricted data must be encrypted at rest using AES-256 and in transit using TLS 1.2 or higher.",
                "Confidential data must not be stored in personal devices without CISO approval.",
                "Data handling procedures must be reviewed annually.",
            ],
            "prohibited_actions": [
                "Storing Restricted data in unencrypted cloud storage.",
                "Transmitting Confidential data over unencrypted channels.",
                "Sharing Restricted data with external parties without legal review.",
            ],
            "exception_allowed": False,
        },
        {
            "slug": "encryption_standards",
            "title": "Encryption Standards",
            "risk_level": "high",
            "control_ids": ["HY-DP-03", "HY-DP-04"],
            "objective": (
                "Define minimum encryption standards for data at rest, in transit, and in use "
                "across all HYTech Solutions environments."
            ),
            "mandatory_requirements": [
                "Data in transit must use TLS 1.2 or higher; TLS 1.0 and 1.1 are prohibited.",
                "Data at rest must use AES-256 for Confidential and Restricted data.",
                "Encryption keys must be rotated at least every 365 days.",
                "Key management must use HYVault or an approved HSM.",
            ],
            "prohibited_actions": [
                "Using deprecated cipher suites including RC4, DES, and 3DES.",
                "Storing encryption keys in source code or configuration files.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "data_retention",
            "title": "Data Retention and Deletion",
            "risk_level": "high",
            "control_ids": ["HY-DP-05", "HY-DP-06"],
            "objective": (
                "Define retention periods and secure deletion requirements for all data "
                "categories in the HYTech Solutions data estate."
            ),
            "mandatory_requirements": [
                "Customer data must be retained for no more than 7 years after contract termination.",
                "Employee records must be retained for 7 years after employment ends.",
                "Security logs must be retained for at least 1 year.",
                "Deletion must use cryptographic erasure for encrypted storage or NIST 800-88 for physical media.",
            ],
            "prohibited_actions": [
                "Retaining personal data beyond the defined retention period without legal justification.",
                "Deleting data subject to a legal hold.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "dlp_controls",
            "title": "Data Loss Prevention Controls",
            "risk_level": "medium",
            "control_ids": ["HY-DP-07", "HY-DP-08"],
            "objective": (
                "Prevent unauthorised exfiltration of sensitive data through technical and "
                "procedural controls."
            ),
            "mandatory_requirements": [
                "DLP monitoring must be enabled on all corporate endpoints.",
                "Bulk downloads of customer data require manager and security approval.",
                "External USB storage is prohibited on production-access workstations.",
                "DLP alert investigations must be completed within 24 hours.",
            ],
            "prohibited_actions": [
                "Disabling DLP agents without written approval from the Security team.",
                "Exporting Restricted data to personal cloud storage accounts.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "third_party_data",
            "title": "Third-Party Data Sharing",
            "risk_level": "low",
            "control_ids": ["HY-DP-09", "HY-DP-10"],
            "objective": (
                "Ensure that data shared with third parties complies with contractual, "
                "legal, and security requirements."
            ),
            "mandatory_requirements": [
                "All third-party data sharing must be covered by a signed Data Processing Agreement.",
                "Third parties must complete a security assessment before receiving Confidential or Restricted data.",
                "Third-party data access must be logged and reviewed quarterly.",
            ],
            "prohibited_actions": [
                "Sharing Confidential or Restricted data with third parties without a signed DPA.",
            ],
            "exception_allowed": True,
        },
    ],
    "incident_response": [
        {
            "slug": "incident_classification",
            "title": "Security Incident Classification and Response",
            "risk_level": "critical",
            "control_ids": ["HY-IR-01", "HY-IR-02"],
            "objective": (
                "Define incident severity classification levels and mandatory response timelines "
                "to ensure consistent and timely handling of security incidents."
            ),
            "mandatory_requirements": [
                "All security incidents must be reported to the Security team within 1 hour of discovery.",
                "Critical incidents require immediate escalation to CISO and Engineering VP.",
                "Incident severity must be classified using the defined severity matrix.",
                "Post-incident reviews must be completed within 5 business days for Severity 1 and 2 incidents.",
            ],
            "prohibited_actions": [
                "Investigating or remediating a security incident without informing the Security team.",
                "Disclosing incident details externally without CISO and Legal approval.",
            ],
            "exception_allowed": False,
        },
        {
            "slug": "breach_notification",
            "title": "Data Breach Notification",
            "risk_level": "high",
            "control_ids": ["HY-IR-03", "HY-IR-04"],
            "objective": (
                "Ensure timely and compliant notification of data breaches to affected parties "
                "and regulators."
            ),
            "mandatory_requirements": [
                "Suspected data breaches involving personal data must be reported to Legal within 4 hours.",
                "Regulatory notification must be completed within 72 hours where required by law.",
                "Customer notification must be completed within 48 hours of confirmed breach.",
                "All breach notifications must be reviewed by Legal and CISO before sending.",
            ],
            "prohibited_actions": [
                "Delaying breach notification beyond regulatory deadlines.",
                "Notifying customers without Legal review.",
            ],
            "exception_allowed": False,
        },
        {
            "slug": "forensic_investigation",
            "title": "Digital Forensics and Evidence Handling",
            "risk_level": "high",
            "control_ids": ["HY-IR-05", "HY-IR-06"],
            "objective": (
                "Define procedures for collecting, preserving, and handling digital evidence "
                "during security incident investigations."
            ),
            "mandatory_requirements": [
                "Digital evidence must be collected using approved forensic tools only.",
                "Chain of custody must be documented for all evidence.",
                "Systems under investigation must not be modified before evidence is collected.",
                "Forensic findings must be stored in the Security Evidence Repository for 3 years.",
            ],
            "prohibited_actions": [
                "Shutting down or reimaging systems under investigation without forensic evidence capture.",
                "Accessing evidence storage without documented authorisation.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "incident_runbook",
            "title": "Incident Response Runbook Maintenance",
            "risk_level": "medium",
            "control_ids": ["HY-IR-07", "HY-IR-08"],
            "objective": (
                "Ensure incident response runbooks are current, tested, and accessible to "
                "the Security team."
            ),
            "mandatory_requirements": [
                "All runbooks must be reviewed and updated at least every 6 months.",
                "Tabletop exercises must be conducted at least once per year.",
                "Runbooks must be stored in HYDrive with access restricted to the Security team.",
                "New incident types must have a runbook created within 30 days of discovery.",
            ],
            "prohibited_actions": [
                "Using runbooks that have not been reviewed in the past 12 months.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "vulnerability_management",
            "title": "Vulnerability Management",
            "risk_level": "low",
            "control_ids": ["HY-IR-09", "HY-IR-10"],
            "objective": (
                "Define the process for identifying, prioritising, and remediating vulnerabilities "
                "in HYTech Solutions systems."
            ),
            "mandatory_requirements": [
                "All production systems must be scanned for vulnerabilities at least every 30 days.",
                "Critical vulnerabilities must be remediated within 7 days of discovery.",
                "High vulnerabilities must be remediated within 30 days.",
                "Vulnerability scan reports must be reviewed by the Security team within 5 business days.",
            ],
            "prohibited_actions": [
                "Deploying systems with known critical vulnerabilities to production.",
            ],
            "exception_allowed": True,
        },
    ],
    "device_security": [
        {
            "slug": "endpoint_encryption",
            "title": "Endpoint Encryption Requirements",
            "risk_level": "critical",
            "control_ids": ["HY-DS-01", "HY-DS-02"],
            "objective": (
                "Ensure all corporate endpoints have full-disk encryption enabled to protect "
                "data in the event of device loss or theft."
            ),
            "mandatory_requirements": [
                "All corporate laptops must have full-disk encryption enabled using BitLocker or FileVault.",
                "Encryption keys must be escrowed in HYVault within 24 hours of device provisioning.",
                "Unencrypted devices must not be used to access corporate systems.",
                "Encryption compliance must be verified monthly through HYPortal MDM.",
            ],
            "prohibited_actions": [
                "Disabling disk encryption on any corporate device.",
                "Using personal devices without MDM enrollment to access Confidential or Restricted data.",
            ],
            "exception_allowed": False,
        },
        {
            "slug": "mobile_device_management",
            "title": "Mobile Device Management",
            "risk_level": "high",
            "control_ids": ["HY-DS-03", "HY-DS-04"],
            "objective": (
                "Define requirements for managing corporate and BYOD mobile devices used "
                "to access HYTech Solutions resources."
            ),
            "mandatory_requirements": [
                "All corporate mobile devices must be enrolled in HYPortal MDM.",
                "BYOD devices must meet minimum OS version requirements before accessing corporate email.",
                "Remote wipe must be enabled on all enrolled devices.",
                "MDM profiles must not be removed without Security team approval.",
            ],
            "prohibited_actions": [
                "Jailbreaking or rooting any device used to access corporate systems.",
                "Installing unapproved applications on corporate devices.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "patch_management",
            "title": "Patch Management",
            "risk_level": "high",
            "control_ids": ["HY-DS-05", "HY-DS-06"],
            "objective": (
                "Ensure all corporate endpoints receive timely security patches to reduce "
                "the attack surface."
            ),
            "mandatory_requirements": [
                "Critical OS patches must be applied within 72 hours of release.",
                "Standard OS patches must be applied within 14 days of release.",
                "Endpoints that fail to patch within the required window must be quarantined.",
                "Patch compliance must be reported to the Security team weekly.",
            ],
            "prohibited_actions": [
                "Indefinitely deferring critical security patches without CISO written approval.",
                "Disabling automatic updates without Security team approval.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "antimalware",
            "title": "Antimalware and EDR Requirements",
            "risk_level": "medium",
            "control_ids": ["HY-DS-07", "HY-DS-08"],
            "objective": (
                "Ensure all corporate endpoints are protected by approved antimalware and "
                "endpoint detection and response solutions."
            ),
            "mandatory_requirements": [
                "All corporate laptops must run the approved EDR agent.",
                "Antimalware definitions must be updated at least every 24 hours.",
                "EDR alerts must be reviewed by the Security team within 4 hours.",
                "Quarantined files must not be restored without Security team analysis.",
            ],
            "prohibited_actions": [
                "Disabling antimalware or EDR agents on corporate devices.",
                "Excluding directories from EDR scanning without Security approval.",
            ],
            "exception_allowed": True,
        },
        {
            "slug": "asset_inventory",
            "title": "Hardware Asset Inventory",
            "risk_level": "low",
            "control_ids": ["HY-DS-09", "HY-DS-10"],
            "objective": (
                "Maintain an accurate and up-to-date inventory of all corporate hardware assets."
            ),
            "mandatory_requirements": [
                "All corporate devices must be registered in the HYPortal asset inventory within 1 business day of provisioning.",
                "Asset inventory must be reconciled with physical assets quarterly.",
                "Decommissioned devices must be removed from inventory within 5 business days.",
            ],
            "prohibited_actions": [
                "Using unregistered devices to access corporate networks.",
            ],
            "exception_allowed": True,
        },
    ],
}

# ── API Documentation configs ─────────────────────────────────────────────────

API_SERVICES: dict[str, dict] = {
    "HYID": {
        "full_name": "HYID Identity Service",
        "department": "Platform Engineering",
        "description": "Centralised identity and access management service providing OAuth2 authentication, token issuance, and user directory management for all HYTech Solutions internal and external applications.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/v{ver}/auth/token",
                "purpose": "Issue an OAuth2 access token using client credentials or authorisation code flow.",
                "required_params": ["client_id", "client_secret", "grant_type"],
                "optional_params": ["scope", "redirect_uri"],
                "response": "JSON object containing access_token, token_type, expires_in, and scope.",
                "errors": ["400", "401", "403", "429"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/auth/refresh",
                "purpose": "Refresh an expired access token using a valid refresh token.",
                "required_params": ["refresh_token", "client_id"],
                "optional_params": ["scope"],
                "response": "JSON object containing new access_token and expires_in.",
                "errors": ["400", "401", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/users/{user_id}",
                "purpose": "Retrieve the profile of an authenticated user by user ID.",
                "required_params": ["user_id"],
                "optional_params": ["fields"],
                "response": "JSON object with user profile including id, email, roles, and department.",
                "errors": ["401", "403", "404", "429"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/users",
                "purpose": "Create a new user account in the HYID directory.",
                "required_params": ["email", "display_name", "department"],
                "optional_params": ["roles", "region"],
                "response": "JSON object with the created user including id and provisioning_status.",
                "errors": ["400", "401", "403", "409", "429"],
            },
            {
                "method": "DELETE",
                "path": "/api/v{ver}/users/{user_id}",
                "purpose": "Deactivate a user account and revoke all active sessions.",
                "required_params": ["user_id"],
                "optional_params": [],
                "response": "204 No Content on success.",
                "errors": ["401", "403", "404"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/auth/jwks",
                "purpose": "Retrieve the JSON Web Key Set for verifying HYID-issued tokens.",
                "required_params": [],
                "optional_params": [],
                "response": "JWKS JSON object with current signing keys.",
                "errors": ["503"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/auth/revoke",
                "purpose": "Revoke an active access token or refresh token.",
                "required_params": ["token"],
                "optional_params": ["token_type_hint"],
                "response": "200 OK on success.",
                "errors": ["400", "401"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/users/{user_id}/groups",
                "purpose": "List all security groups and roles assigned to a user.",
                "required_params": ["user_id"],
                "optional_params": ["page", "page_size"],
                "response": "Paginated JSON array of group objects.",
                "errors": ["401", "403", "404", "429"],
            },
        ],
        "error_codes": {
            "400": ("Bad Request", "Verify required parameters are present and correctly formatted."),
            "401": ("Unauthorized", "Provide a valid Bearer token in the Authorization header."),
            "403": ("Forbidden", "Request requires elevated permissions; contact your administrator."),
            "404": ("Not Found", "Verify the user_id or resource identifier is correct."),
            "409": ("Conflict", "A user with this email already exists."),
            "429": ("Too Many Requests", "Reduce request frequency; see rate limit headers for retry guidance."),
            "500": ("Internal Server Error", "Retry with exponential backoff; contact support if the issue persists."),
            "503": ("Service Unavailable", "Service is temporarily unavailable; retry after 30 seconds."),
        },
        "rate_limits": {
            "free": "100 requests/minute",
            "standard": "1000 requests/minute",
            "enterprise": "10000 requests/minute",
        },
        "auth_method": "OAuth2",
        "adr_prefix": "ADR-HYID",
    },
    "HYDeploy": {
        "full_name": "HYDeploy Release Management Service",
        "department": "Platform Engineering",
        "description": "Automated deployment and release management service for HYTech Solutions applications, supporting blue-green deployments, canary releases, and rollback operations.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/v{ver}/deployments",
                "purpose": "Trigger a new deployment for a specified application and environment.",
                "required_params": ["app_id", "environment", "artifact_id"],
                "optional_params": ["strategy", "canary_weight", "notify_channels"],
                "response": "JSON object with deployment_id, status, and estimated_duration_seconds.",
                "errors": ["400", "401", "403", "409", "422", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/deployments/{deployment_id}",
                "purpose": "Retrieve the status and details of a deployment by ID.",
                "required_params": ["deployment_id"],
                "optional_params": ["include_logs"],
                "response": "JSON object with deployment status, start_time, and step_results.",
                "errors": ["401", "403", "404"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/deployments/{deployment_id}/rollback",
                "purpose": "Roll back a failed or problematic deployment to the previous successful version.",
                "required_params": ["deployment_id"],
                "optional_params": ["reason"],
                "response": "JSON object with rollback_id and status.",
                "errors": ["400", "401", "403", "404", "409"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/apps/{app_id}/deployments",
                "purpose": "List all deployment history for a specific application.",
                "required_params": ["app_id"],
                "optional_params": ["environment", "status", "page", "page_size"],
                "response": "Paginated JSON array of deployment summary objects.",
                "errors": ["401", "403", "404", "429"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/apps/{app_id}/promote",
                "purpose": "Promote a canary or staging deployment to full production traffic.",
                "required_params": ["app_id", "deployment_id"],
                "optional_params": ["approval_token"],
                "response": "JSON object with promotion_id and new_traffic_weight.",
                "errors": ["400", "401", "403", "404", "422"],
            },
            {
                "method": "DELETE",
                "path": "/api/v{ver}/deployments/{deployment_id}",
                "purpose": "Cancel an in-progress deployment.",
                "required_params": ["deployment_id"],
                "optional_params": [],
                "response": "204 No Content on successful cancellation.",
                "errors": ["401", "403", "404", "409"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/artifacts",
                "purpose": "List available build artifacts for deployment.",
                "required_params": ["app_id"],
                "optional_params": ["branch", "status", "page"],
                "response": "Paginated JSON array of artifact objects with id, version, and build_status.",
                "errors": ["401", "403", "429"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/gates/{app_id}",
                "purpose": "Evaluate deployment gate conditions before proceeding to the next stage.",
                "required_params": ["app_id", "gate_type"],
                "optional_params": ["override_token"],
                "response": "JSON object with gate_result and blocking_conditions.",
                "errors": ["400", "401", "403", "422"],
            },
        ],
        "error_codes": {
            "400": ("Bad Request", "Verify all required parameters are present."),
            "401": ("Unauthorized", "Provide a valid service token in the Authorization header."),
            "403": ("Forbidden", "The requestor lacks the deploy permission for this application."),
            "404": ("Not Found", "Verify the app_id or deployment_id is correct."),
            "409": ("Conflict", "A deployment is already in progress for this application and environment."),
            "422": ("Unprocessable Entity", "Deployment failed gate validation; check gate conditions."),
            "429": ("Too Many Requests", "Reduce deployment frequency; maximum 10 concurrent deployments per app."),
            "500": ("Internal Server Error", "Retry after 60 seconds; contact Platform Engineering if unresolved."),
            "503": ("Service Unavailable", "HYDeploy service is undergoing maintenance; retry after 5 minutes."),
        },
        "rate_limits": {
            "free": "50 requests/minute",
            "standard": "500 requests/minute",
            "enterprise": "5000 requests/minute",
        },
        "auth_method": "service_token",
        "adr_prefix": "ADR-HYD",
    },
    "HYMonitor": {
        "full_name": "HYMonitor Observability Service",
        "department": "Platform Engineering",
        "description": "Centralised observability platform providing metrics ingestion, alerting, log aggregation, and dashboard management for all HYTech Solutions services.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/v{ver}/metrics",
                "purpose": "Ingest a batch of time-series metrics data points.",
                "required_params": ["metrics"],
                "optional_params": ["resolution", "labels"],
                "response": "JSON object with ingested_count and failed_count.",
                "errors": ["400", "401", "413", "429", "503"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/metrics/query",
                "purpose": "Execute a metrics query and return time-series results.",
                "required_params": ["query", "start", "end"],
                "optional_params": ["step", "timeout"],
                "response": "JSON object with result_type and data arrays.",
                "errors": ["400", "401", "403", "408", "429"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/alerts",
                "purpose": "Create a new alerting rule for a metrics condition.",
                "required_params": ["name", "query", "threshold", "severity"],
                "optional_params": ["labels", "notify_channels", "silence_duration_minutes"],
                "response": "JSON object with alert_id and evaluation_interval_seconds.",
                "errors": ["400", "401", "403", "409", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/alerts/{alert_id}",
                "purpose": "Retrieve the current state and history of an alert rule.",
                "required_params": ["alert_id"],
                "optional_params": ["include_history"],
                "response": "JSON object with alert state, last_fired, and recent_events.",
                "errors": ["401", "403", "404"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/logs/query",
                "purpose": "Query aggregated log data using a structured filter expression.",
                "required_params": ["filter", "start", "end"],
                "optional_params": ["limit", "fields", "sort"],
                "response": "JSON object with log entries and total_count.",
                "errors": ["400", "401", "403", "408", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/dashboards",
                "purpose": "List all monitoring dashboards accessible to the caller.",
                "required_params": [],
                "optional_params": ["tag", "page", "page_size"],
                "response": "Paginated JSON array of dashboard metadata objects.",
                "errors": ["401", "429"],
            },
        ],
        "error_codes": {
            "400": ("Bad Request", "Verify the query syntax and required parameters."),
            "401": ("Unauthorized", "Provide a valid API key in the X-API-Key header."),
            "403": ("Forbidden", "The API key lacks read or write permission for this resource."),
            "404": ("Not Found", "Verify the alert_id or dashboard_id is correct."),
            "408": ("Request Timeout", "Narrow the query time range or add more specific filters."),
            "409": ("Conflict", "An alert with this name already exists."),
            "413": ("Payload Too Large", "Reduce batch size to fewer than 10000 metrics per request."),
            "429": ("Too Many Requests", "Reduce ingestion rate; see X-RateLimit headers."),
            "503": ("Service Unavailable", "HYMonitor ingestion pipeline is degraded; retry in 60 seconds."),
        },
        "rate_limits": {
            "free": "200 requests/minute",
            "standard": "2000 requests/minute",
            "enterprise": "20000 requests/minute",
        },
        "auth_method": "API_key",
        "adr_prefix": "ADR-HYM",
    },
    "HYDataLake": {
        "full_name": "HYDataLake Ingestion Service",
        "department": "Data Engineering Lead",
        "description": "Large-scale data ingestion and partitioning service for the HYTech Solutions enterprise data lake, supporting batch and streaming data pipelines.",
        "endpoints": [
            {
                "method": "POST",
                "path": "/api/v{ver}/ingest/batch",
                "purpose": "Submit a batch ingestion job for a structured or semi-structured dataset.",
                "required_params": ["source_uri", "dataset_id", "format"],
                "optional_params": ["partition_by", "schema_id", "notify_on_complete"],
                "response": "JSON object with job_id, status, and estimated_duration_seconds.",
                "errors": ["400", "401", "403", "409", "413", "422", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/ingest/jobs/{job_id}",
                "purpose": "Retrieve the status and metrics of a batch ingestion job.",
                "required_params": ["job_id"],
                "optional_params": ["include_metrics"],
                "response": "JSON object with job status, rows_ingested, and errors.",
                "errors": ["401", "403", "404"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/streams",
                "purpose": "Register a new streaming data source for continuous ingestion.",
                "required_params": ["stream_name", "source_type", "schema_id"],
                "optional_params": ["buffer_size_mb", "flush_interval_seconds", "partition_by"],
                "response": "JSON object with stream_id and ingestion_endpoint.",
                "errors": ["400", "401", "403", "409", "429"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/datasets/{dataset_id}/partitions",
                "purpose": "List available data partitions for a dataset.",
                "required_params": ["dataset_id"],
                "optional_params": ["date_from", "date_to", "page"],
                "response": "Paginated JSON array of partition objects with path and row_count.",
                "errors": ["401", "403", "404", "429"],
            },
            {
                "method": "DELETE",
                "path": "/api/v{ver}/datasets/{dataset_id}/partitions/{partition_id}",
                "purpose": "Delete a specific data partition from the lake.",
                "required_params": ["dataset_id", "partition_id"],
                "optional_params": ["purge"],
                "response": "204 No Content on success.",
                "errors": ["401", "403", "404", "409"],
            },
            {
                "method": "GET",
                "path": "/api/v{ver}/schemas/{schema_id}",
                "purpose": "Retrieve the registered schema definition for a dataset.",
                "required_params": ["schema_id"],
                "optional_params": ["version"],
                "response": "JSON object with schema fields, types, and validation rules.",
                "errors": ["401", "403", "404"],
            },
            {
                "method": "POST",
                "path": "/api/v{ver}/schemas",
                "purpose": "Register a new schema or update an existing schema version.",
                "required_params": ["schema_name", "fields"],
                "optional_params": ["version", "description"],
                "response": "JSON object with schema_id and version.",
                "errors": ["400", "401", "403", "409"],
            },
        ],
        "error_codes": {
            "400": ("Bad Request", "Verify source_uri, format, and schema_id parameters."),
            "401": ("Unauthorized", "Provide a valid mTLS client certificate or service token."),
            "403": ("Forbidden", "The caller lacks ingestion or read permission for this dataset."),
            "404": ("Not Found", "Verify the job_id, dataset_id, or schema_id."),
            "409": ("Conflict", "A job or stream with the same name is already active."),
            "413": ("Payload Too Large", "Use streaming ingestion for datasets exceeding 10 GB."),
            "422": ("Unprocessable Entity", "Schema validation failed; check the error details."),
            "429": ("Too Many Requests", "Reduce job submission rate; maximum 5 concurrent batch jobs per dataset."),
            "503": ("Service Unavailable", "Ingestion pipeline is degraded; retry after 120 seconds."),
        },
        "rate_limits": {
            "free": "50 requests/minute",
            "standard": "500 requests/minute",
            "enterprise": "5000 requests/minute",
        },
        "auth_method": "mTLS",
        "adr_prefix": "ADR-HDL",
    },
}

API_VERSIONS: list[dict] = [
    {"version": "v1", "stability": "deprecated", "note": "Deprecated as of 2023-06-01. Migrate to v2."},
    {"version": "v2", "stability": "stable", "note": "Current stable version."},
    {"version": "v2", "stability": "stable", "note": "Current stable version."},
    {"version": "v3", "stability": "beta", "note": "Feature-complete beta; breaking changes possible."},
    {"version": "v3", "stability": "experimental", "note": "Experimental preview; do not use in production."},
]

# ── System Design configs ─────────────────────────────────────────────────────

SYSTEM_DESIGN_SPECS: list[dict] = [
    {
        "slug": "hydeploy_pipeline",
        "system_name": "HYDeploy Deployment Pipeline",
        "service_key": "HYDeploy",
        "adr_prefix": "ADR-HYD",
        "architecture_status": "implemented",
        "problem_statement": (
            "HYTech Solutions requires a standardised, automated deployment pipeline that "
            "supports multiple deployment strategies (blue-green, canary, rolling) with "
            "integrated approval gates and rollback capabilities across all production environments."
        ),
        "goals": [
            "Support blue-green, canary, and rolling deployment strategies for all application types.",
            "Provide automated gate evaluation including smoke tests and metric threshold checks.",
            "Enable one-click rollback to any previous successful deployment within 30 seconds.",
            "Achieve 99.9% deployment service uptime.",
        ],
        "non_goals": [
            "Build management and artifact creation are out of scope; HYDeploy consumes pre-built artifacts only.",
            "Infrastructure provisioning is handled by the separate HYCloud service.",
        ],
        "components": [
            {"name": "Deployment Orchestrator", "responsibility": "Manages deployment lifecycle state machine and step sequencing.", "depends_on": ["Artifact Registry", "Gate Evaluator"]},
            {"name": "Artifact Registry", "responsibility": "Stores and versions build artifacts for all applications.", "depends_on": []},
            {"name": "Gate Evaluator", "responsibility": "Evaluates pre- and post-deployment conditions including metric thresholds and smoke tests.", "depends_on": ["HYMonitor"]},
            {"name": "Traffic Manager", "responsibility": "Controls traffic routing weights for canary and blue-green deployments.", "depends_on": ["Load Balancer"]},
            {"name": "Rollback Manager", "responsibility": "Executes rollback operations by restoring the previous deployment state.", "depends_on": ["Deployment Orchestrator", "Traffic Manager"]},
            {"name": "Notification Router", "responsibility": "Sends deployment status notifications to configured channels.", "depends_on": []},
        ],
        "dependencies": ["HYMonitor", "HYID", "HYDrive"],
        "sla": "99.9% uptime",
        "data_volume": "Handles up to 500 deployment events per day across all production environments.",
        "failure_modes": [
            {"mode": "Gate evaluation timeout", "mitigation": "Gates time out after 300 seconds and block the deployment; operator manual override is required."},
            {"mode": "Artifact registry unavailable", "mitigation": "Deployments are queued for up to 15 minutes; if the registry remains unavailable, the deployment fails with error code HYD-503."},
            {"mode": "Traffic manager rollback failure", "mitigation": "A secondary static routing configuration is applied; on-call engineer is paged immediately."},
        ],
        "tradeoffs": [
            {"decision": "Use blue-green by default over rolling updates", "benefit": "Zero-downtime deployments with instant rollback", "cost": "Doubles the compute resources required during deployment"},
            {"decision": "Synchronous gate evaluation", "benefit": "Prevents bad deployments from progressing", "cost": "Increases deployment duration by 60-120 seconds per gate"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
    {
        "slug": "hymonitor_ingestion",
        "system_name": "HYMonitor Event Ingestion Design",
        "service_key": "HYMonitor",
        "adr_prefix": "ADR-HYM",
        "architecture_status": "implemented",
        "problem_statement": (
            "HYTech Solutions requires a scalable, low-latency metrics and log ingestion "
            "system capable of handling 500,000 events per second at peak load with less "
            "than 5 seconds end-to-end latency."
        ),
        "goals": [
            "Ingest 500,000 events per second at peak with less than 5 seconds end-to-end latency.",
            "Provide exactly-once delivery semantics for critical alert metrics.",
            "Support dynamic scaling without service interruption.",
            "Retain raw metrics for 30 days and aggregated metrics for 1 year.",
        ],
        "non_goals": [
            "Real-time streaming queries with sub-second latency are not supported in this version.",
            "Log analysis and anomaly detection are handled by a separate ML pipeline.",
        ],
        "components": [
            {"name": "Ingest Gateway", "responsibility": "Authenticates and validates incoming metric and log payloads.", "depends_on": ["HYID"]},
            {"name": "Write Buffer", "responsibility": "Buffers incoming events in memory before batch writes to storage.", "depends_on": []},
            {"name": "Partition Router", "responsibility": "Routes events to time-partitioned storage buckets based on timestamp and service label.", "depends_on": ["HYDataLake"]},
            {"name": "Aggregation Engine", "responsibility": "Computes rollup aggregations (1m, 5m, 1h, 1d) over raw time-series data.", "depends_on": ["Write Buffer"]},
            {"name": "Alert Evaluator", "responsibility": "Evaluates alert rule conditions against incoming metrics in near-real time.", "depends_on": ["Aggregation Engine"]},
        ],
        "dependencies": ["HYID", "HYDataLake", "HYDrive"],
        "sla": "99.95% uptime for ingestion; 99.9% for query",
        "data_volume": "500,000 events per second at peak; 5 PB total storage.",
        "failure_modes": [
            {"mode": "Write buffer overflow", "mitigation": "Back-pressure signals are sent to producers; events are dropped with error code HYM-503 if the buffer remains full for more than 30 seconds."},
            {"mode": "Partition router failure", "mitigation": "Events are routed to a fallback partition; manual rebalancing is required after recovery."},
            {"mode": "Alert evaluator lag", "mitigation": "Alerts are delayed but not lost; a lag threshold of 60 seconds triggers an operational alert."},
        ],
        "tradeoffs": [
            {"decision": "Use time-partitioned storage over full-text indexing", "benefit": "Dramatically reduces storage cost and query latency for time-range queries", "cost": "Full-text searches require a separate indexing pipeline"},
            {"decision": "Eventually consistent aggregation", "benefit": "Higher throughput without blocking writes", "cost": "Aggregation results may lag raw data by up to 30 seconds"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
    {
        "slug": "hydatalake_partitioning",
        "system_name": "HYDataLake Partitioning Architecture",
        "service_key": "HYDataLake",
        "adr_prefix": "ADR-HDL",
        "architecture_status": "approved",
        "problem_statement": (
            "The HYDataLake must support petabyte-scale storage with efficient partitioning "
            "for both analytical workloads and operational data pipelines, while maintaining "
            "data lineage and schema evolution capabilities."
        ),
        "goals": [
            "Support datasets up to 10 PB with partition pruning reducing query scan cost by at least 80%.",
            "Enforce schema validation on all ingested data.",
            "Provide data lineage tracking for all transformations.",
            "Support time-travel queries up to 30 days in the past.",
        ],
        "non_goals": [
            "Online transaction processing (OLTP) workloads are not supported.",
            "Real-time CDC replication is handled by a separate service.",
        ],
        "components": [
            {"name": "Schema Registry", "responsibility": "Stores and versions dataset schemas; validates incoming data against registered schemas.", "depends_on": []},
            {"name": "Partition Manager", "responsibility": "Creates and maintains partition metadata; routes data to correct partition paths.", "depends_on": ["Schema Registry"]},
            {"name": "Lineage Tracker", "responsibility": "Records data transformation history and dependencies.", "depends_on": []},
            {"name": "Compaction Service", "responsibility": "Merges small files into optimal Parquet files to reduce query overhead.", "depends_on": ["Partition Manager"]},
            {"name": "Time-Travel Store", "responsibility": "Maintains snapshots for point-in-time query support.", "depends_on": ["Partition Manager"]},
            {"name": "Access Controller", "responsibility": "Enforces column-level and row-level access policies.", "depends_on": ["HYID"]},
        ],
        "dependencies": ["HYID", "HYMonitor"],
        "sla": "99.9% uptime for read; 99.5% for write",
        "data_volume": "Current storage: 5 PB. Expected growth: 2 PB per year.",
        "failure_modes": [
            {"mode": "Compaction service backlog", "mitigation": "Read performance degrades gracefully; compaction is prioritised during off-peak hours."},
            {"mode": "Schema registry unavailable", "mitigation": "Ingestion is paused; buffered events are retried for up to 30 minutes."},
            {"mode": "Partition metadata corruption", "mitigation": "Partition metadata is rebuilt from object storage; recovery takes up to 2 hours for large datasets."},
        ],
        "tradeoffs": [
            {"decision": "Use Parquet columnar format over row-based storage", "benefit": "3-10x query performance improvement for analytical workloads", "cost": "Write amplification during compaction"},
            {"decision": "Eventual consistency for partition metadata", "benefit": "Higher write throughput", "cost": "Newly ingested data may not be visible to queries for up to 60 seconds"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
    {
        "slug": "hyid_authentication",
        "system_name": "HYID Authentication Architecture",
        "service_key": "HYID",
        "adr_prefix": "ADR-HYID",
        "architecture_status": "implemented",
        "problem_statement": (
            "HYTech Solutions requires a centralised, standards-compliant identity and "
            "access management system that supports OAuth2, SAML 2.0, and service-to-service "
            "authentication for over 1,200 employees and 50 internal services."
        ),
        "goals": [
            "Issue and validate OAuth2 tokens with sub-10ms latency at the 99th percentile.",
            "Support SAML 2.0 federation with external identity providers.",
            "Provide service-to-service mTLS authentication for all internal APIs.",
            "Achieve 99.99% uptime for token validation.",
        ],
        "non_goals": [
            "HYID does not manage application-level authorisation policies; those are managed by each service.",
            "Customer-facing identity management uses a separate B2C service.",
        ],
        "components": [
            {"name": "Token Issuer", "responsibility": "Issues JWT access tokens and refresh tokens after successful authentication.", "depends_on": ["User Directory", "Key Manager"]},
            {"name": "User Directory", "responsibility": "Stores user profiles, credentials, and group memberships.", "depends_on": []},
            {"name": "Key Manager", "responsibility": "Manages signing keys for JWT tokens; rotates keys on a 90-day schedule.", "depends_on": ["HYVault"]},
            {"name": "SAML Bridge", "responsibility": "Translates SAML 2.0 assertions to HYID tokens for federated identity providers.", "depends_on": ["Token Issuer"]},
            {"name": "Session Manager", "responsibility": "Tracks active sessions and handles logout and forced revocation.", "depends_on": ["User Directory"]},
            {"name": "Audit Logger", "responsibility": "Records all authentication events to HYDataLake for security auditing.", "depends_on": ["HYDataLake"]},
        ],
        "dependencies": ["HYDataLake", "HYMonitor", "HYVault"],
        "sla": "99.99% uptime for token validation; 99.9% for token issuance",
        "data_volume": "Issues up to 50,000 tokens per minute at peak.",
        "failure_modes": [
            {"mode": "Key manager unavailable", "mitigation": "Token issuance fails; existing tokens remain valid until expiry. Key Manager has a 3-node HA deployment."},
            {"mode": "User directory replication lag", "mitigation": "Read replicas may serve stale data; write-through caching reduces but does not eliminate the risk."},
            {"mode": "Session store overflow", "mitigation": "Oldest sessions are evicted; affected users must re-authenticate."},
        ],
        "tradeoffs": [
            {"decision": "Use short-lived JWTs (15 minutes) over long-lived session tokens", "benefit": "Reduced blast radius for compromised tokens", "cost": "Higher refresh token request volume"},
            {"decision": "Separate token issuance from token validation path", "benefit": "Token validation can be performed without a network call using public key verification", "cost": "Key rotation requires coordination with all consuming services"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
    {
        "slug": "notification_routing",
        "system_name": "Notification Routing Design",
        "service_key": "HYDeploy",
        "adr_prefix": "ADR-NTF",
        "architecture_status": "proposed",
        "problem_statement": (
            "HYTech Solutions internal services require a unified notification routing layer "
            "that can deliver alerts and status updates to multiple channels (email, Slack, "
            "PagerDuty) with guaranteed at-least-once delivery and deduplication."
        ),
        "goals": [
            "Support email, Slack, and PagerDuty delivery channels.",
            "Guarantee at-least-once delivery with idempotency keys to prevent duplicate notifications.",
            "Deliver notifications within 10 seconds of event receipt at the 95th percentile.",
            "Provide per-channel delivery rate limiting to prevent notification fatigue.",
        ],
        "non_goals": [
            "Customer-facing notifications are handled by a separate customer communications service.",
            "SMS delivery is out of scope for this version.",
        ],
        "components": [
            {"name": "Notification Gateway", "responsibility": "Accepts notification requests from internal services and validates the payload.", "depends_on": ["HYID"]},
            {"name": "Deduplication Store", "responsibility": "Stores idempotency keys to prevent duplicate delivery within a 24-hour window.", "depends_on": []},
            {"name": "Channel Router", "responsibility": "Routes notifications to the appropriate channel adapter based on the configured routing rules.", "depends_on": ["Deduplication Store"]},
            {"name": "Email Adapter", "responsibility": "Formats and delivers email notifications via the corporate SMTP gateway.", "depends_on": []},
            {"name": "Slack Adapter", "responsibility": "Delivers formatted messages to Slack channels via the Slack Web API.", "depends_on": []},
            {"name": "PagerDuty Adapter", "responsibility": "Creates and resolves PagerDuty incidents for on-call alerts.", "depends_on": []},
        ],
        "dependencies": ["HYID", "HYMonitor"],
        "sla": "99.9% delivery rate; 10-second delivery latency at p95",
        "data_volume": "Up to 10,000 notifications per hour during peak incident periods.",
        "failure_modes": [
            {"mode": "Channel adapter timeout", "mitigation": "Failed deliveries are retried with exponential backoff up to 5 times; undelivered notifications are moved to a dead-letter queue."},
            {"mode": "Deduplication store unavailable", "mitigation": "Deduplication is bypassed; duplicate notifications may be delivered. An alert is raised for manual review."},
        ],
        "tradeoffs": [
            {"decision": "Use a push-based delivery model over polling", "benefit": "Lower latency and reduced infrastructure cost", "cost": "Requires reliable connectivity to all channel endpoints"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
    {
        "slug": "billing_reconciliation",
        "system_name": "Billing Reconciliation Design",
        "service_key": "HYDataLake",
        "adr_prefix": "ADR-BIL",
        "architecture_status": "approved",
        "problem_statement": (
            "HYTech Solutions requires an automated billing reconciliation system that "
            "aggregates usage data from all internal services, applies pricing rules, "
            "and produces accurate monthly invoices for internal chargebacks and customer billing."
        ),
        "goals": [
            "Reconcile usage data from all six internal services monthly with 99.99% accuracy.",
            "Complete reconciliation run within 4 hours of month end.",
            "Support retroactive corrections for up to 90 days.",
            "Produce auditable invoice records stored in HYDataLake.",
        ],
        "non_goals": [
            "Real-time usage metering is handled by each individual service.",
            "Payment processing is out of scope; this system produces invoices only.",
        ],
        "components": [
            {"name": "Usage Aggregator", "responsibility": "Collects and normalises usage records from HYDataLake usage partitions.", "depends_on": ["HYDataLake"]},
            {"name": "Pricing Engine", "responsibility": "Applies current pricing rules to aggregated usage quantities.", "depends_on": []},
            {"name": "Invoice Generator", "responsibility": "Produces structured invoice documents in JSON and PDF formats.", "depends_on": ["Pricing Engine", "Usage Aggregator"]},
            {"name": "Correction Handler", "responsibility": "Processes retroactive corrections and regenerates affected invoices.", "depends_on": ["Invoice Generator"]},
            {"name": "Audit Trail", "responsibility": "Records all reconciliation runs and corrections with timestamps and operator IDs.", "depends_on": ["HYDataLake"]},
        ],
        "dependencies": ["HYDataLake", "HYID", "HYMonitor"],
        "sla": "99.99% accuracy; 4-hour reconciliation window",
        "data_volume": "Processes up to 500 million usage events per monthly reconciliation run.",
        "failure_modes": [
            {"mode": "Usage data incomplete at month end", "mitigation": "Reconciliation is paused for up to 2 hours; partial data is flagged and a manual review is triggered."},
            {"mode": "Pricing engine configuration error", "mitigation": "Invoice generation fails with a validation error; Finance is notified immediately."},
        ],
        "tradeoffs": [
            {"decision": "Batch reconciliation over real-time streaming", "benefit": "Simpler consistency model and lower operational cost", "cost": "Billing information is available monthly, not in real time"},
        ],
        "related_api_doc_prefix": "API-DOC",
    },
]

# ── Meeting Notes configs ─────────────────────────────────────────────────────

PROJECT_CODES: list[str] = [
    "PRJ-HYD-ALPHA",
    "PRJ-HYID-BETA",
    "PRJ-SECURITY-AUDIT",
    "PRJ-DATALAKE-V2",
    "PRJ-BILLING-REVAMP",
]

PROJECT_DESCRIPTIONS: dict[str, str] = {
    "PRJ-HYD-ALPHA": "HYDeploy v2 rollout and blue-green deployment automation",
    "PRJ-HYID-BETA": "HYID v3 authentication upgrade and SAML 2.0 federation",
    "PRJ-SECURITY-AUDIT": "Annual security policy review and control gap remediation",
    "PRJ-DATALAKE-V2": "HYDataLake partitioning architecture migration",
    "PRJ-BILLING-REVAMP": "Billing reconciliation system rewrite and automation",
}

# One sequence of meeting types per project: 6 meetings total
MEETING_SEQUENCES: list[dict] = [
    {"type": "sprint_planning", "offset_weeks": 0},
    {"type": "sprint_planning", "offset_weeks": 4},
    {"type": "architecture_review", "offset_weeks": 8},
    {"type": "architecture_review", "offset_weeks": 12},
    {"type": "postmortem", "offset_weeks": 16},
    {"type": "launch_review", "offset_weeks": 20},
]

# Base dates per project (spread over 18 months from 2023-01-15)
PROJECT_BASE_DATES: dict[str, str] = {
    "PRJ-HYD-ALPHA": "2023-01-15",
    "PRJ-HYID-BETA": "2023-04-10",
    "PRJ-SECURITY-AUDIT": "2023-07-03",
    "PRJ-DATALAKE-V2": "2023-10-09",
    "PRJ-BILLING-REVAMP": "2024-01-22",
}

MEETING_DECISION_POOL: dict[str, list[str]] = {
    "sprint_planning": [
        "Agreed to prioritise the authentication refactor in this sprint.",
        "Capacity confirmed for 32 story points across the team.",
        "Dependency on the Security team review resolved; blocking issue cleared.",
        "Sprint goal accepted: complete API endpoint migration to v2.",
        "Technical debt items deferred to next sprint by consensus.",
        "Deployment freeze window accepted: no releases on Fridays.",
    ],
    "architecture_review": [
        "Architecture approved with the condition that load test results are reviewed before go-live.",
        "Decision to use blue-green deployment over rolling updates for zero-downtime guarantee.",
        "ADR recorded: use PostgreSQL over DynamoDB for transactional billing records.",
        "Security reviewer approved the proposed mTLS configuration.",
        "Agreed to defer the streaming ingestion design to Phase 2.",
        "Rollback procedure approved and added to the runbook.",
    ],
    "postmortem": [
        "Root cause identified: configuration change deployed without gate evaluation.",
        "Action items assigned to prevent recurrence within 2 weeks.",
        "Agreed to add automated rollback trigger when error rate exceeds 5% for 60 seconds.",
        "Communication gap identified: on-call engineer was not notified within the 4-hour SLA.",
        "Postmortem document signed off and published to the incident library.",
    ],
    "launch_review": [
        "Go/No-Go decision: Go. All gate criteria met.",
        "Rollback plan confirmed and on-call rotation set for launch weekend.",
        "Customer communication draft approved by Product and Legal.",
        "Monitoring dashboards reviewed; alert thresholds adjusted for launch traffic.",
        "Launch date confirmed: production deployment scheduled for Monday 09:00 KST.",
    ],
}

MEETING_BLOCKER_POOL: list[str] = [
    "Security review not yet completed for the new API endpoints.",
    "Dependency on HYDataLake migration still pending; blocking the billing pipeline.",
    "Infrastructure provisioning for the staging environment is delayed by 1 week.",
    "Legal review of the updated data retention policy is pending.",
    "HYID SAML bridge has a known issue with IdP metadata refresh; fix targeted for next sprint.",
    "Performance test results for the new partitioning scheme are not yet available.",
]

ACTION_STATUS_OPTIONS: list[str] = ["completed", "pending", "blocked"]
