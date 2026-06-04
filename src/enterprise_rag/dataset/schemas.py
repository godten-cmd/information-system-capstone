"""Canonical Pydantic v2 domain schemas for the SEKD dataset.

Six core models correspond to the six dataset artifacts:

    DocumentMetadata    – global and category-specific metadata for one document
    EnterpriseDocument  – full document with content, structure, and metadata
    Chunk               – retrievable text fragment produced from a document
    Query               – evaluation query with oracle annotations
    GroundTruth         – reference answer with evidence spans
    PlannerEvaluation   – planner strategy selection record

Supporting types:

    HRPolicyMetadata / TravelPolicyMetadata / SecurityPolicyMetadata /
    APIDocumentationMetadata / SystemDesignMetadata / MeetingNotesMetadata –
        category-specific sub-schemas; selected via the CategoryMetadata
        discriminated union on the ``kind`` field.

    DocumentSection, DocumentTable – structural elements extracted from a document.
    EvidenceSpan                   – single evidence reference inside GroundTruth.

JSON field-name mapping (Python name → DATASET_SPEC JSON name):
    DocumentMetadata.author_role          → document_owner
    DocumentMetadata.confidentiality_level → confidentiality
    EnterpriseDocument.content            → body
    Query.query_text                      → query
    GroundTruth.answer                    → reference_answer
    GroundTruth.evidence_spans            → evidence

All models that carry aliases set ``populate_by_name=True`` so both the Python
name and the JSON alias are accepted as constructor arguments.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import (
    APIAuthMethod,
    APIServiceName,
    APIStability,
    ArchitectureStatus,
    AuthorityLevel,
    ConfidentialityLevel,
    DocumentCategory,
    DocumentStatus,
    EmployeeType,
    MeetingType,
    OracleStrategy,
    OracleStrategySource,
    PlannerType,
    PolicyArea,
    QueryDifficulty,
    QueryType,
    ReasoningType,
    Region,
    RequiredDocumentCount,
    RetrievalDifficultyFactor,
    ReviewCycle,
    SecurityControlFamily,
    SecurityRiskLevel,
    TravelCurrency,
    TravelExpenseType,
)

_CHUNK_ID_RE = re.compile(r"^.+-C\d{3}$")


# ── Category-specific metadata models ────────────────────────────────────────
# Each model carries a ``kind`` discriminator equal to its DocumentCategory
# value so the CategoryMetadata union can be resolved deterministically.


class HRPolicyMetadata(BaseModel):
    """Category-specific metadata for HR Policy documents (e.g. HR-POL-001).

    Captures policy ownership, scope, eligibility, and approval chain.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.HR_POLICY] = Field(
        default=DocumentCategory.HR_POLICY,
        description="Discriminator — always 'HR Policies'",
    )
    policy_id: str = Field(
        description="HR policy identifier, e.g. HR-POL-001",
        pattern=r"^HR-POL-\d+$",
    )
    policy_area: PolicyArea = Field(description="HR policy subject area")
    employee_type: EmployeeType = Field(
        description="Employment type this policy covers"
    )
    applicable_regions: list[Region] = Field(
        description="Regions where this policy applies",
        min_length=1,
    )
    approval_body: str = Field(
        description="Body that approved this policy, e.g. HR Committee"
    )
    supersedes: str | None = Field(
        default=None,
        description="Previous policy ID this document replaces, if any",
    )
    escalation_contact: str = Field(
        description="HR role to contact for escalations"
    )


class TravelPolicyMetadata(BaseModel):
    """Category-specific metadata for Travel Policy documents (e.g. TRV-POL-001).

    Captures expense category, applicable roles, spending limits, and approval.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.TRAVEL_POLICY] = Field(
        default=DocumentCategory.TRAVEL_POLICY,
        description="Discriminator — always 'Travel Policies'",
    )
    travel_policy_id: str = Field(
        description="Travel policy identifier, e.g. TRV-POL-001",
        pattern=r"^TRV-POL-\d+$",
    )
    expense_type: TravelExpenseType = Field(
        description="Primary expense category covered by this policy"
    )
    applicable_roles: list[str] = Field(
        description="Employee roles this policy applies to",
        min_length=1,
    )
    currency: TravelCurrency = Field(description="Currency used for spending limits")
    approval_required: bool = Field(
        description="Whether pre-approval is required for this expense type"
    )
    finance_owner: str = Field(description="Finance role that owns this policy")
    receipt_required_threshold: float = Field(
        description="Amount above which receipts are mandatory",
        ge=0.0,
    )


class SecurityPolicyMetadata(BaseModel):
    """Category-specific metadata for Security Policy documents (e.g. SEC-POL-001).

    Captures control domain, risk level, enforcement ownership, and exception rules.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.SECURITY_POLICY] = Field(
        default=DocumentCategory.SECURITY_POLICY,
        description="Discriminator — always 'Security Policies'",
    )
    security_policy_id: str = Field(
        description="Security policy identifier, e.g. SEC-POL-001",
        pattern=r"^SEC-POL-\d+$",
    )
    control_family: SecurityControlFamily = Field(
        description="Security control domain family"
    )
    control_ids: list[str] = Field(
        description="Synthetic control IDs referenced by this policy, e.g. HY-AC-01",
        min_length=1,
    )
    risk_level: SecurityRiskLevel = Field(description="Risk severity level")
    enforcement_owner: str = Field(
        description="Security team role responsible for enforcement"
    )
    exception_allowed: bool = Field(
        description="Whether exceptions to this policy can be approved"
    )
    review_frequency_days: int = Field(
        description="Maximum days between scheduled reviews",
        gt=0,
    )


class APIDocumentationMetadata(BaseModel):
    """Category-specific metadata for API Documentation (e.g. API-DOC-001).

    Captures service identity, API version, authentication, and stability.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.API_DOCUMENTATION] = Field(
        default=DocumentCategory.API_DOCUMENTATION,
        description="Discriminator — always 'API Documentation'",
    )
    api_doc_id: str = Field(
        description="API document identifier, e.g. API-DOC-001",
        pattern=r"^API-DOC-\d+$",
    )
    service_name: APIServiceName = Field(
        description="HYTech Solutions internal service name"
    )
    api_version: str = Field(
        description="API version string, e.g. v1, v2, v3",
        pattern=r"^v\d+$",
    )
    endpoint_count: int = Field(
        description="Number of endpoints described in this document",
        ge=1,
    )
    auth_method: APIAuthMethod = Field(description="Authentication method required")
    owner_team: str = Field(description="Engineering team that owns this API")
    stability: APIStability = Field(description="API lifecycle stability status")


class SystemDesignMetadata(BaseModel):
    """Category-specific metadata for System Design Documents (e.g. SYS-DES-001).

    Captures system identity, architecture status, ownership, and decision records.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.SYSTEM_DESIGN] = Field(
        default=DocumentCategory.SYSTEM_DESIGN,
        description="Discriminator — always 'System Design Documents'",
    )
    design_doc_id: str = Field(
        description="Design document identifier, e.g. SYS-DES-001",
        pattern=r"^SYS-DES-\d+$",
    )
    system_name: str = Field(description="Name of the system being designed")
    architecture_status: ArchitectureStatus = Field(
        description="Current lifecycle status of this architecture"
    )
    primary_author_role: str = Field(
        description="Synthetic role of the primary author; must not be a real name"
    )
    reviewer_roles: list[str] = Field(
        description="Roles that reviewed this document",
        min_length=1,
    )
    related_services: list[str] = Field(
        description="HYTech services this system depends on",
        min_length=1,
    )
    decision_record_ids: list[str] = Field(
        description="Synthetic ADR IDs referenced by this document, e.g. ADR-HYD-001",
        default_factory=list,
    )


class MeetingNotesMetadata(BaseModel):
    """Category-specific metadata for Project Meeting Notes (e.g. MTG-001).

    Captures project identity, meeting type, participants, and cross-document links.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal[DocumentCategory.MEETING_NOTES] = Field(
        default=DocumentCategory.MEETING_NOTES,
        description="Discriminator — always 'Project Meeting Notes'",
    )
    meeting_id: str = Field(
        description="Meeting note identifier, e.g. MTG-001",
        pattern=r"^MTG-\d+$",
    )
    project_code: str = Field(
        description="Synthetic project code, e.g. PRJ-HYD-ALPHA"
    )
    meeting_type: MeetingType = Field(description="Type of project meeting")
    meeting_date: date = Field(description="Date the meeting took place (ISO 8601)")
    participants: list[str] = Field(
        description="Synthetic role names of participants; must not include real names",
        min_length=1,
    )
    decisions_count: int = Field(
        description="Number of decisions recorded in this note",
        ge=0,
    )
    action_items_count: int = Field(
        description="Number of action items recorded in this note",
        ge=0,
    )
    related_documents: list[str] = Field(
        description="IDs of related policy, API, or design documents referenced",
        default_factory=list,
    )


CategoryMetadata = Annotated[
    Union[
        HRPolicyMetadata,
        TravelPolicyMetadata,
        SecurityPolicyMetadata,
        APIDocumentationMetadata,
        SystemDesignMetadata,
        MeetingNotesMetadata,
    ],
    Field(discriminator="kind"),
]
"""Discriminated union of all six category-specific metadata models.

Pydantic selects the correct sub-model based on the ``kind`` field value,
which must equal the parent ``DocumentMetadata.category`` value.
"""


# ── Supporting types for EnterpriseDocument ──────────────────────────────────


class DocumentSection(BaseModel):
    """A logical section within a document, preserving its heading hierarchy."""

    model_config = ConfigDict(extra="forbid")

    heading: str = Field(description="Section heading text")
    level: int = Field(description="Heading depth: 1 = top-level, 2 = sub-section, etc.", ge=1)
    path: list[str] = Field(
        description="Full path from document root to this section, e.g. ['3. Data Flow', '3.1 Ingestion']",
        min_length=1,
    )
    content: str = Field(description="Full text body of this section")
    char_start: int = Field(
        description="Start character offset of this section in the document body",
        ge=0,
    )
    char_end: int = Field(
        description="End character offset of this section in the document body",
        ge=0,
    )

    @model_validator(mode="after")
    def _offsets_valid(self) -> DocumentSection:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class DocumentTable(BaseModel):
    """A table extracted from a document with its surrounding section context.

    Row count is not constrained to allow empty tables (e.g. template stubs),
    but every row must have the same number of cells as the header row.
    """

    model_config = ConfigDict(extra="forbid")

    caption: str | None = Field(
        default=None,
        description="Table caption or title text, if present",
    )
    section_path: list[str] = Field(
        description="Section path where this table appears",
        min_length=1,
    )
    headers: list[str] = Field(
        description="Column header labels",
        min_length=1,
    )
    rows: list[list[str]] = Field(
        description="Table data rows; each row must contain exactly len(headers) cells",
        default_factory=list,
    )
    char_start: int = Field(
        description="Start character offset of this table in the document body",
        ge=0,
    )
    char_end: int = Field(
        description="End character offset of this table in the document body",
        ge=0,
    )

    @model_validator(mode="after")
    def _validate_rows_and_offsets(self) -> DocumentTable:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        for i, row in enumerate(self.rows):
            if len(row) != len(self.headers):
                raise ValueError(
                    f"Row {i} has {len(row)} cell(s) but the table has "
                    f"{len(self.headers)} header(s)"
                )
        return self


# ── Core model 1: DocumentMetadata ───────────────────────────────────────────


class DocumentMetadata(BaseModel):
    """Complete metadata for one SEKD document.

    Combines the DATASET_SPEC top-level document fields with the mandatory
    global metadata schema that every document must carry. The
    ``category_metadata`` field holds the category-specific sub-schema,
    validated as a discriminated union and required to match ``category``.

    JSON alias mapping (Python field → DATASET_SPEC JSON key):
        author_role            → document_owner
        confidentiality_level  → confidentiality
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # ── Task-required fields ─────────────────────────────────────────────────

    document_id: str = Field(description="Unique document identifier, e.g. HR-POL-001")
    category: DocumentCategory = Field(description="Primary document category")
    title: str = Field(description="Human-readable document title", min_length=1)
    version: str = Field(
        description="Semantic or policy version, e.g. '1.0', '2.3', 'v2'",
        min_length=1,
    )
    created_at: datetime = Field(description="ISO 8601 datetime when the document was created")
    updated_at: datetime = Field(description="ISO 8601 datetime of the most recent update")
    department: str = Field(
        description="Owning department within HYTech Solutions, e.g. 'Human Resources'",
        min_length=1,
    )
    author_role: str = Field(
        description=(
            "Synthetic role name of the document owner "
            "(maps to 'document_owner' in DATASET_SPEC JSON)"
        ),
        alias="document_owner",
        min_length=1,
    )
    confidentiality_level: ConfidentialityLevel = Field(
        description=(
            "Document sensitivity classification "
            "(maps to 'confidentiality' in DATASET_SPEC JSON)"
        ),
        alias="confidentiality",
    )
    region: Region = Field(description="Primary geographic region this document applies to")
    language: str = Field(
        description="ISO 639-1 language code, e.g. 'en', 'ko', or BCP-47 tag 'en-US'",
        pattern=r"^[a-z]{2}(-[A-Z]{2})?$",
    )

    # ── Global metadata (DATASET_SPEC §Global Metadata Schema) ──────────────

    company: Literal["HYTech Solutions"] = Field(
        default="HYTech Solutions",
        description="Must always be 'HYTech Solutions'",
    )
    effective_date: date = Field(description="Date the document became effective (ISO 8601)")
    review_cycle: ReviewCycle = Field(description="How frequently the document is reviewed")
    status: DocumentStatus = Field(description="Document lifecycle status")
    authority_level: AuthorityLevel = Field(description="Enforcement weight of the document")
    tags: list[str] = Field(
        description="Search and filtering tags; at least one required",
        min_length=1,
    )
    source_path: str = Field(
        description="Synthetic filesystem source path, e.g. '/data/hr/HR-POL-001.md'",
        min_length=1,
    )

    # ── Category-specific metadata ───────────────────────────────────────────

    category_metadata: CategoryMetadata | None = Field(
        default=None,
        description=(
            "Category-specific sub-schema. When present, its ``kind`` field "
            "must equal this document's ``category`` value."
        ),
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _updated_at_not_before_created_at(self) -> DocumentMetadata:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self

    @model_validator(mode="after")
    def _category_metadata_kind_matches_category(self) -> DocumentMetadata:
        if self.category_metadata is None:
            return self
        # Use .value for version-stable comparison; str(StrEnum) behaviour
        # changed across Python releases.
        actual = self.category_metadata.kind.value  # type: ignore[union-attr]
        expected = self.category.value
        if actual != expected:
            raise ValueError(
                f"category_metadata.kind '{actual}' does not match "
                f"category '{expected}'"
            )
        return self


# ── Core model 2: EnterpriseDocument ─────────────────────────────────────────


class EnterpriseDocument(BaseModel):
    """A full SEKD document as written to ``documents.jsonl``.

    Contains complete metadata, the full text body, extracted structural
    elements (sections and tables), and a list of cross-document reference IDs.

    JSON alias mapping (Python field → DATASET_SPEC JSON key):
        content → body
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    metadata: DocumentMetadata = Field(description="Complete document metadata")
    content: str = Field(
        description="Full document body text (maps to 'body' in DATASET_SPEC JSON)",
        alias="body",
        min_length=1,
    )
    sections: list[DocumentSection] = Field(
        description="Hierarchical sections extracted from the document body",
        default_factory=list,
    )
    tables: list[DocumentTable] = Field(
        description="Tables extracted from the document body",
        default_factory=list,
    )
    references: list[str] = Field(
        description="IDs of other SEKD documents referenced by this document",
        default_factory=list,
    )

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def document_id(self) -> str:
        """Shortcut to ``metadata.document_id``."""
        return self.metadata.document_id

    @property
    def category(self) -> DocumentCategory:
        """Shortcut to ``metadata.category``."""
        return self.metadata.category


# ── Core model 3: Chunk ───────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A retrievable text chunk produced from an EnterpriseDocument.

    Written to ``chunks.jsonl``.  Chunk IDs must follow the format
    ``{document_id}-C{NNN}`` where NNN is a zero-padded three-digit sequence
    number (e.g. ``TRV-POL-004-C003``).

    ``inherited_metadata`` carries both document-level metadata fields
    forwarded from the parent document and any chunk-specific enrichment
    (e.g. table row context, endpoint block label).

    JSON field aliases (Python field → DATASET_SPEC JSON key):
        start_offset → char_start
        end_offset   → char_end
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    chunk_id: str = Field(description="Unique chunk identifier: {document_id}-C{NNN}")
    document_id: str = Field(description="Parent document identifier")
    category: DocumentCategory = Field(
        description="Document category inherited from the parent document"
    )
    section_path: list[str] = Field(
        description="Hierarchical section path from document root to the containing section",
        min_length=1,
    )
    text: str = Field(description="Chunk text content", min_length=1)
    inherited_metadata: dict[str, Any] = Field(
        description=(
            "Metadata fields inherited from the parent document plus any "
            "chunk-specific annotations (maps to 'metadata' in DATASET_SPEC JSON)"
        ),
        default_factory=dict,
    )
    start_offset: int = Field(
        description="Start character offset within the document body (char_start in spec)",
        alias="char_start",
        ge=0,
    )
    end_offset: int = Field(
        description="End character offset within the document body (char_end in spec)",
        alias="char_end",
        ge=0,
    )
    token_estimate: int = Field(
        description="Approximate token count for the chunk text",
        ge=0,
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("chunk_id")
    @classmethod
    def _chunk_id_matches_pattern(cls, v: str) -> str:
        if not _CHUNK_ID_RE.match(v):
            raise ValueError(
                f"chunk_id '{v}' does not match expected pattern "
                "'{{document_id}}-C{{NNN}}' (NNN = 3 digits)"
            )
        return v

    @model_validator(mode="after")
    def _end_offset_after_start(self) -> Chunk:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        return self

    @model_validator(mode="after")
    def _chunk_id_belongs_to_document(self) -> Chunk:
        prefix = self.chunk_id.rsplit("-C", 1)[0]
        if prefix != self.document_id:
            raise ValueError(
                f"chunk_id prefix '{prefix}' does not match document_id '{self.document_id}'"
            )
        return self


# ── Core model 4: Query ───────────────────────────────────────────────────────


class Query(BaseModel):
    """An evaluation query for the SEKD benchmark.

    Written to ``queries.jsonl``.

    Oracle strategy labels are research annotations, not absolute ground truth.
    Every label must carry provenance via ``oracle_strategy_source`` so that
    planner evaluation remains auditable and labels can be corrected after
    empirical experiments without creating circular evaluation
    (DATASET_SPEC §Oracle Validation Framework).

    Unanswerable queries must have empty ``required_document_ids`` and
    ``required_chunk_ids``.  Queries with ``required_document_count`` of TWO or
    THREE_PLUS must include ``multi_document_dependency`` in
    ``retrieval_difficulty_factors``.

    JSON alias mapping (Python field → DATASET_SPEC JSON key):
        query_text → query
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # ── Task-required fields ─────────────────────────────────────────────────

    query_id: str = Field(description="Unique query identifier, e.g. Q-000001")
    query_text: str = Field(
        description="User-facing query string (maps to 'query' in DATASET_SPEC JSON)",
        alias="query",
        min_length=1,
    )
    difficulty: QueryDifficulty = Field(description="Retrieval difficulty level")
    reasoning_type: ReasoningType = Field(
        description="Reasoning complexity required to correctly answer this query"
    )
    planner_oracle_strategy: OracleStrategy = Field(
        description=(
            "Retrieval strategy expected to perform best for this query. "
            "Treated as a research annotation; see oracle_strategy_source."
        )
    )
    oracle_strategy_source: OracleStrategySource = Field(
        description="Provenance of the planner_oracle_strategy label"
    )
    retrieval_difficulty_factors: list[RetrievalDifficultyFactor] = Field(
        description="Fine-grained dimensions that make retrieval harder for this query",
        default_factory=list,
    )
    required_document_count: RequiredDocumentCount = Field(
        description="Number of documents required to answer correctly: '1', '2', or '3+'"
    )

    # ── Additional DATASET_SPEC fields ──────────────────────────────────────

    category: DocumentCategory = Field(
        description="Primary expected document category for this query"
    )
    query_type: QueryType = Field(description="Primary query taxonomy label")
    answerable: bool = Field(description="Whether the answer exists in the corpus")
    required_document_ids: list[str] = Field(
        description="Document IDs expected to contain supporting evidence",
        default_factory=list,
    )
    required_chunk_ids: list[str] = Field(
        description="Chunk IDs expected to directly support the answer",
        default_factory=list,
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _answerable_requires_evidence_ids(self) -> Query:
        if self.answerable:
            if not self.required_document_ids:
                raise ValueError(
                    "answerable=True requires at least one entry in required_document_ids"
                )
            if not self.required_chunk_ids:
                raise ValueError(
                    "answerable=True requires at least one entry in required_chunk_ids"
                )
        else:
            if self.required_document_ids:
                raise ValueError(
                    "answerable=False must have empty required_document_ids"
                )
            if self.required_chunk_ids:
                raise ValueError(
                    "answerable=False must have empty required_chunk_ids"
                )
        return self

    @model_validator(mode="after")
    def _multi_doc_flag_required(self) -> Query:
        """Queries spanning 2+ documents must declare the multi_document_dependency factor."""
        is_multi = self.required_document_count in (
            RequiredDocumentCount.TWO,
            RequiredDocumentCount.THREE_PLUS,
        )
        has_flag = (
            RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY
            in self.retrieval_difficulty_factors
        )
        if is_multi and not has_flag:
            raise ValueError(
                "required_document_count '2' or '3+' requires "
                "'multi_document_dependency' in retrieval_difficulty_factors"
            )
        return self


# ── Supporting type for GroundTruth ──────────────────────────────────────────


class EvidenceSpan(BaseModel):
    """A single supporting evidence span linking a ground-truth claim to a source chunk.

    Every factual claim in a GroundTruth ``answer`` must map to at least one
    EvidenceSpan.  The ``char_start`` / ``char_end`` offsets refer to character
    positions within the parent document's full body text.
    """

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(description="Source document identifier")
    chunk_id: str = Field(description="Source chunk identifier")
    section_path: list[str] = Field(
        description="Section path identifying where the evidence lives",
        min_length=1,
    )
    quote: str = Field(
        description="Verbatim text from the source that constitutes the evidence",
        min_length=1,
    )
    char_start: int = Field(
        description="Start character offset of the quote within the document body",
        ge=0,
    )
    char_end: int = Field(
        description="End character offset of the quote within the document body",
        ge=0,
    )
    supports: str = Field(
        description="Short description of the claim supported by this evidence",
        min_length=1,
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _offsets_valid(self) -> EvidenceSpan:
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self

    @model_validator(mode="after")
    def _chunk_belongs_to_document(self) -> EvidenceSpan:
        if "-C" not in self.chunk_id:
            raise ValueError(
                f"chunk_id '{self.chunk_id}' does not match the expected "
                "pattern '{{document_id}}-C{{NNN}}'"
            )
        prefix = self.chunk_id.rsplit("-C", 1)[0]
        if prefix != self.document_id:
            raise ValueError(
                f"chunk_id prefix '{prefix}' does not match document_id '{self.document_id}'"
            )
        return self


# ── Core model 5: GroundTruth ─────────────────────────────────────────────────


class GroundTruth(BaseModel):
    """Reference answer and evidence annotations for one SEKD query.

    Written to ``ground_truth.jsonl``.

    Every factual claim in ``answer`` must be grounded by at least one entry in
    ``evidence_spans``.  ``required_document_ids`` and ``required_chunk_ids``
    must be consistent with the document IDs and chunk IDs appearing in
    ``evidence_spans``.  ``expected_citations`` must be a subset of the chunk IDs
    in ``evidence_spans``.

    JSON alias mapping (Python field → DATASET_SPEC JSON key):
        answer         → reference_answer
        evidence_spans → evidence
    """

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # ── Task-required fields ─────────────────────────────────────────────────

    query_id: str = Field(
        description="Matching query identifier from queries.jsonl"
    )
    answer: str = Field(
        description="Gold reference answer (maps to 'reference_answer' in DATASET_SPEC JSON)",
        alias="reference_answer",
        min_length=1,
    )
    required_document_ids: list[str] = Field(
        description="Document IDs that must be retrieved to support this answer",
        default_factory=list,
    )
    required_chunk_ids: list[str] = Field(
        description="Chunk IDs that must be retrieved to support this answer",
        default_factory=list,
    )
    evidence_spans: list[EvidenceSpan] = Field(
        description=(
            "Detailed evidence spans linking specific claims to source chunks "
            "(maps to 'evidence' in DATASET_SPEC JSON)"
        ),
        alias="evidence",
        default_factory=list,
    )

    # ── Additional DATASET_SPEC fields ──────────────────────────────────────

    expected_citations: list[str] = Field(
        description=(
            "Chunk or document IDs that should appear in the generated answer's citations. "
            "Must be a subset of chunk IDs present in evidence_spans."
        ),
        default_factory=list,
    )
    acceptable_answer_patterns: list[str] = Field(
        description="Regex or literal phrase patterns that a correct answer must satisfy",
        default_factory=list,
    )
    disallowed_claims: list[str] = Field(
        description="Phrases or claims that must not appear in a correct answer",
        default_factory=list,
    )
    evaluation_notes: str = Field(
        description="Human-readable grading guidance for evaluators",
        default="",
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _required_ids_covered_by_evidence(self) -> GroundTruth:
        evidence_doc_ids = {span.document_id for span in self.evidence_spans}
        evidence_chunk_ids = {span.chunk_id for span in self.evidence_spans}

        missing_docs = set(self.required_document_ids) - evidence_doc_ids
        if missing_docs:
            raise ValueError(
                f"required_document_ids {sorted(missing_docs)} have no corresponding "
                "evidence_span entries"
            )
        missing_chunks = set(self.required_chunk_ids) - evidence_chunk_ids
        if missing_chunks:
            raise ValueError(
                f"required_chunk_ids {sorted(missing_chunks)} have no corresponding "
                "evidence_span entries"
            )
        return self

    @model_validator(mode="after")
    def _expected_citations_are_evidence_chunks(self) -> GroundTruth:
        evidence_chunk_ids = {span.chunk_id for span in self.evidence_spans}
        unknown = set(self.expected_citations) - evidence_chunk_ids
        if unknown:
            raise ValueError(
                f"expected_citations {sorted(unknown)} are not present in evidence_spans"
            )
        return self


# ── Core model 6: PlannerEvaluation ──────────────────────────────────────────


class PlannerEvaluation(BaseModel):
    """Evaluation record comparing a planner's strategy selection against the oracle.

    Written to ``planner_evaluation.jsonl`` after each planner run.

    ``strategy_correct`` is derived from ``selected_strategy == oracle_strategy``.
    The validator enforces this invariant so callers cannot set an inconsistent value.

    Note: ``oracle_strategy`` is copied from ``Query.planner_oracle_strategy``
    and inherits that label's provenance constraints.  A correct planner decision
    does not necessarily mean the oracle label is reliable; see
    ``oracle_strategy_source`` on the parent Query.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(description="Query that was evaluated by this planner run")
    planner_type: PlannerType = Field(description="Planner implementation variant")
    selected_strategy: OracleStrategy = Field(
        description="Retrieval strategy selected by the planner for this query"
    )
    oracle_strategy: OracleStrategy = Field(
        description="Oracle strategy value copied from Query.planner_oracle_strategy"
    )
    strategy_correct: bool = Field(
        description="True when selected_strategy equals oracle_strategy"
    )

    # ── Validator ────────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _strategy_correct_is_consistent(self) -> PlannerEvaluation:
        expected = self.selected_strategy == self.oracle_strategy
        if self.strategy_correct != expected:
            raise ValueError(
                f"strategy_correct={self.strategy_correct} is inconsistent: "
                f"selected='{self.selected_strategy.value}' "
                f"oracle='{self.oracle_strategy.value}' "
                f"(expected strategy_correct={expected})"
            )
        return self
