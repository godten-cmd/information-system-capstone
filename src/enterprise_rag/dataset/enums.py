"""Enumeration types for the SEKD dataset domain.

All enums inherit from ``str`` so they serialise directly to JSON strings and
compare equal to their raw string values (e.g. ``DocumentCategory.HR_POLICY ==
"HR Policies"`` is True).
"""

from __future__ import annotations

from enum import Enum


# ── Document-level enums ─────────────────────────────────────────────────────


class DocumentCategory(str, Enum):
    """Six document categories that make up the SEKD corpus."""

    HR_POLICY = "HR Policies"
    TRAVEL_POLICY = "Travel Policies"
    SECURITY_POLICY = "Security Policies"
    API_DOCUMENTATION = "API Documentation"
    SYSTEM_DESIGN = "System Design Documents"
    MEETING_NOTES = "Project Meeting Notes"


class ConfidentialityLevel(str, Enum):
    """Document sensitivity classification applied across all categories."""

    PUBLIC_INTERNAL = "public_internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class DocumentStatus(str, Enum):
    """Document lifecycle status."""

    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class AuthorityLevel(str, Enum):
    """Enforcement weight of the document."""

    ADVISORY = "advisory"
    STANDARD = "standard"
    MANDATORY = "mandatory"


class ReviewCycle(str, Enum):
    """How frequently the document is scheduled for review."""

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMIANNUAL = "semiannual"
    ANNUAL = "annual"


class Region(str, Enum):
    """HYTech Solutions geographic regions and global scope."""

    GLOBAL = "global"
    KR = "KR"
    SG = "SG"
    US = "US"
    EU = "EU"


# ── Query-level enums ─────────────────────────────────────────────────────────


class QueryType(str, Enum):
    """Primary query taxonomy label — exactly one per query."""

    FACTUAL_LOOKUP = "factual_lookup"
    POLICY_INTERPRETATION = "policy_interpretation"
    PROCEDURAL = "procedural"
    NUMERIC_LOOKUP = "numeric_lookup"
    TABLE_LOOKUP = "table_lookup"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    TEMPORAL = "temporal"
    OWNER_LOOKUP = "owner_lookup"
    EXCEPTION_LOOKUP = "exception_lookup"
    TROUBLESHOOTING = "troubleshooting"
    SUMMARIZATION = "summarization"
    UNANSWERABLE = "unanswerable"


class QueryDifficulty(str, Enum):
    """Retrieval difficulty level for a query.

    Difficulty reflects retrieval complexity, not answer length.
    """

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    ADVERSARIAL = "adversarial"


class ReasoningType(str, Enum):
    """Reasoning complexity required to answer the query correctly."""

    SINGLE_HOP = "single_hop"
    MULTI_HOP = "multi_hop"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    EXCEPTION = "exception"
    AGGREGATION = "aggregation"


class OracleStrategy(str, Enum):
    """Retrieval strategy labels used for oracle annotation and planner comparison.

    The ``planner_oracle_strategy`` field in a Query carries one of these values.
    It is a research annotation, not absolute ground truth; see
    ``OracleStrategySource`` for its provenance.
    """

    KEYWORD = "keyword"
    DENSE = "dense"
    HYBRID = "hybrid"
    METADATA_FILTERED = "metadata_filtered"
    HIERARCHICAL = "hierarchical"
    TABLE_AWARE = "table_aware"
    MULTI_HOP = "multi_hop"


class OracleStrategySource(str, Enum):
    """Provenance of a ``planner_oracle_strategy`` annotation.

    Labels must carry provenance so that planner evaluation remains auditable
    and oracle labels can be corrected after empirical validation without
    creating circular evaluation.
    """

    MANUAL_ANNOTATION = "manual_annotation"
    HEURISTIC_ASSIGNMENT = "heuristic_assignment"
    EMPIRICAL_VALIDATION = "empirical_validation"


class RetrievalDifficultyFactor(str, Enum):
    """Fine-grained difficulty dimensions that complicate retrieval for a query.

    These are independent of the coarse ``QueryDifficulty`` label. A medium-
    difficulty query may still carry ``metadata_dependency`` while a hard query
    may combine several factors.
    """

    KEYWORD_AMBIGUITY = "keyword_ambiguity"
    DOCUMENT_VERSION_CONFLICT = "document_version_conflict"
    METADATA_DEPENDENCY = "metadata_dependency"
    TABLE_DEPENDENCY = "table_dependency"
    MULTI_DOCUMENT_DEPENDENCY = "multi_document_dependency"
    EXCEPTION_HANDLING = "exception_handling"
    TEMPORAL_REASONING = "temporal_reasoning"


class RequiredDocumentCount(str, Enum):
    """Number of documents required to correctly answer a query.

    Queries with ``TWO`` or ``THREE_PLUS`` must include
    ``RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY``.
    """

    ONE = "1"
    TWO = "2"
    THREE_PLUS = "3+"


# ── Planner enums ─────────────────────────────────────────────────────────────


class PlannerType(str, Enum):
    """Retrieval planner implementation variant."""

    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"


# ── Category-specific metadata enums ─────────────────────────────────────────


class PolicyArea(str, Enum):
    """HR policy subject area."""

    LEAVE = "leave"
    BENEFITS = "benefits"
    REMOTE_WORK = "remote_work"
    PERFORMANCE = "performance"
    CONDUCT = "conduct"


class EmployeeType(str, Enum):
    """Employment type covered by an HR policy."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACTOR = "contractor"
    INTERN = "intern"
    ALL = "all"


class TravelExpenseType(str, Enum):
    """Primary expense category covered by a travel policy."""

    AIRFARE = "airfare"
    LODGING = "lodging"
    MEALS = "meals"
    GROUND_TRANSPORT = "ground_transport"
    VISA = "visa"
    MISCELLANEOUS = "miscellaneous"


class TravelCurrency(str, Enum):
    """Currency used in travel policy spending limits."""

    KRW = "KRW"
    USD = "USD"
    EUR = "EUR"
    SGD = "SGD"


class SecurityControlFamily(str, Enum):
    """Security control domain family."""

    ACCESS_CONTROL = "access_control"
    DATA_PROTECTION = "data_protection"
    INCIDENT_RESPONSE = "incident_response"
    DEVICE_SECURITY = "device_security"


class SecurityRiskLevel(str, Enum):
    """Risk severity level for a security policy."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class APIServiceName(str, Enum):
    """Internal HYTech Solutions service names covered by API documentation."""

    HYID = "HYID"
    HYDEPLOY = "HYDeploy"
    HYMONITOR = "HYMonitor"
    HYDATALAKE = "HYDataLake"


class APIAuthMethod(str, Enum):
    """Authentication method used by a HYTech API."""

    OAUTH2 = "OAuth2"
    API_KEY = "API_key"
    MTLS = "mTLS"
    SERVICE_TOKEN = "service_token"


class APIStability(str, Enum):
    """API lifecycle stability status."""

    EXPERIMENTAL = "experimental"
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class ArchitectureStatus(str, Enum):
    """Lifecycle status of a system design document."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    IMPLEMENTED = "implemented"
    RETIRED = "retired"


class MeetingType(str, Enum):
    """Type of project meeting captured in meeting notes."""

    SPRINT_PLANNING = "sprint_planning"
    ARCHITECTURE_REVIEW = "architecture_review"
    POSTMORTEM = "postmortem"
    LAUNCH_REVIEW = "launch_review"
