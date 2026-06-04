"""SEKDQueryGenerator: deterministic query and ground truth generation for SEKD."""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from enterprise_rag.dataset.distributions import DistributionConfig, adjust_oracle_strategy_distribution
from enterprise_rag.dataset.enums import (
    DocumentCategory,
    OracleStrategy,
    OracleStrategySource,
    QueryDifficulty,
    QueryType,
    ReasoningType,
    RequiredDocumentCount,
    RetrievalDifficultyFactor,
)
from enterprise_rag.dataset.ground_truth import build_evidence_spans, find_quote_in_body
from enterprise_rag.dataset.query_templates import UNANSWERABLE_SPECS
from enterprise_rag.dataset.schemas import (
    Chunk,
    EnterpriseDocument,
    GroundTruth,
    Query,
)

# ── Internal raw query representation ────────────────────────────────────────


@dataclass
class _RawQuery:
    """Internal representation before Query/GroundTruth serialization."""

    category: DocumentCategory
    doc_ids: list[str]
    query_text: str
    answer_text: str
    # Each tuple: (doc_id_for_quote, quote_text, supports_label)
    quotes: list[tuple[str, str, str]]
    reasoning_type: ReasoningType
    query_type: QueryType
    difficulty: QueryDifficulty
    oracle_strategy: OracleStrategy
    oracle_strategy_source: OracleStrategySource = OracleStrategySource.HEURISTIC_ASSIGNMENT
    difficulty_factors: list[RetrievalDifficultyFactor] = field(default_factory=list)
    answerable: bool = True
    disallowed_claims: list[str] = field(default_factory=list)
    acceptable_patterns: list[str] = field(default_factory=list)
    evaluation_notes: str = ""


# ── Generator config ─────────────────────────────────────────────────────────


@dataclass
class QueryConfig:
    seed: int = 42
    distribution: DistributionConfig = field(default_factory=DistributionConfig)


# ── Extraction helpers ────────────────────────────────────────────────────────


def _extract_to_stop(body: str, anchor: str, stop: str = ".", max_len: int = 300) -> str | None:
    """Find anchor in body, then return the text from anchor to the next stop char."""
    idx = body.find(anchor)
    if idx == -1:
        return None
    start = idx
    stop_idx = body.find(stop, idx + len(anchor))
    if stop_idx == -1:
        stop_idx = min(idx + len(anchor) + max_len, len(body))
    end = stop_idx + (1 if stop in (".", "\n") else 0)
    return body[start:end]


def _extract_line(body: str, anchor: str) -> str | None:
    """Return the full line containing anchor."""
    idx = body.find(anchor)
    if idx == -1:
        return None
    line_start = body.rfind("\n", 0, idx) + 1
    line_end = body.find("\n", idx)
    if line_end == -1:
        line_end = len(body)
    return body[line_start:line_end].strip()


def _find_table_row(body: str, key: str) -> str | None:
    """Find the first table row that starts with '| key' (case-insensitive)."""
    pattern = re.compile(
        r"(\| " + re.escape(key) + r" \|[^\n]+)", re.IGNORECASE
    )
    m = pattern.search(body)
    return m.group(1).strip() if m else None


def _first_control_row(body: str) -> tuple[str, str] | None:
    """Extract (control_id, requirement) from the first control table row."""
    m = re.search(r"\| (HY-[A-Z]+-\d+) \| (.+?) \| \w+ \|", body)
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()[:120]


def _extract_endpoint_block(body: str) -> tuple[str, str, str] | None:
    """
    Extract (method, path, first_param_line) from the first endpoint block.
    Returns None if no endpoint found.
    """
    m = re.search(r"### (GET|POST|PUT|DELETE|PATCH) (/[^\n]+)\n(.+?)(?:\n\n|\Z)", body, re.DOTALL)
    if not m:
        return None
    method = m.group(1)
    path = m.group(2).strip()
    # Extract purpose line (first non-empty line after header)
    block_lines = m.group(3).strip().split("\n")
    purpose = next((ln.strip() for ln in block_lines if ln.strip() and not ln.strip().startswith("**")), "")
    return method, path, purpose


def _extract_error_row(body: str, code: str = "401") -> str | None:
    """Find error code table row."""
    return _find_table_row(body, code)


def _meeting_key_decision(body: str) -> str | None:
    """Extract the first decision text from meeting notes."""
    m = re.search(r"\d+\. \*\*[^*]+\*\*: (.+?)(?:\n|$)", body)
    return m.group(1).strip()[:200] if m else None


def _meeting_first_action(body: str) -> str | None:
    """Extract the first full action items table row."""
    # Table row: | Owner Role | Action | Due Date | Status |
    m = re.search(r"(\| [A-Z][^|]+ \| Complete [^|]+ \| \d{4}-\d{2}-\d{2} \| \w+ \|)", body)
    return m.group(1).strip() if m else None


def _meeting_project_code(body: str) -> str | None:
    m = re.search(r"Project: (PRJ-[\w-]+)", body)
    return m.group(1) if m else None


def _sysdes_sla(body: str) -> str | None:
    return _extract_to_stop(body, "Target SLA: ", " with")


def _sysdes_first_component(body: str) -> tuple[str, str] | None:
    """Return (component_name, dependency) from first component table row."""
    m = re.search(r"\| ([A-Z][^|]+?) \| ([^|]+?) \| ([^|]*?) \|", body)
    if not m:
        return None
    return m.group(1).strip(), m.group(3).strip()


# ── Per-category query generators ────────────────────────────────────────────


class SEKDQueryGenerator:
    """Deterministic query and ground truth generator for the SEKD corpus."""

    def __init__(self, config: QueryConfig | None = None) -> None:
        self._config = config or QueryConfig()
        self._rng = random.Random(self._config.seed)

    # ── Public interface ──────────────────────────────────────────────────────

    def generate(
        self,
        docs: list[EnterpriseDocument],
        chunks: list[Chunk],
    ) -> tuple[list[Query], list[GroundTruth]]:
        """Generate all queries and ground truths for the corpus."""
        doc_index: dict[str, EnterpriseDocument] = {d.document_id: d for d in docs}
        chunks_by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for c in chunks:
            chunks_by_doc[c.document_id].append(c)

        raw: list[_RawQuery] = []

        # Per-document queries
        for doc in docs:
            cat = doc.metadata.category
            if cat == DocumentCategory.HR_POLICY:
                raw.extend(self._gen_hr(doc))
            elif cat == DocumentCategory.TRAVEL_POLICY:
                raw.extend(self._gen_travel(doc))
            elif cat == DocumentCategory.SECURITY_POLICY:
                raw.extend(self._gen_security(doc))
            elif cat == DocumentCategory.API_DOCUMENTATION:
                raw.extend(self._gen_api(doc, doc_index))
            elif cat == DocumentCategory.SYSTEM_DESIGN:
                raw.extend(self._gen_sysdes(doc, doc_index))
            elif cat == DocumentCategory.MEETING_NOTES:
                raw.extend(self._gen_meeting(doc, doc_index))

        # Cross-document queries
        raw.extend(self._gen_hr_comparisons(docs))
        raw.extend(self._gen_travel_comparisons(docs))
        raw.extend(self._gen_sec_comparisons(docs))
        raw.extend(self._gen_mtg_temporal(docs))
        raw.extend(self._gen_mtg_aggregation(docs))

        # Unanswerable queries
        raw.extend(self._gen_unanswerable())

        # Shuffle deterministically, apply distribution adjustment
        self._rng.shuffle(raw)
        adjust_oracle_strategy_distribution(raw, self._config.distribution, self._rng)

        # Serialize to Query + GroundTruth objects
        queries: list[Query] = []
        ground_truths: list[GroundTruth] = []
        seq = 1
        for rq in raw:
            query_id = f"Q-{seq:06d}"
            result = self._build(query_id, rq, doc_index, chunks_by_doc)
            if result is not None:
                q, gt = result
                queries.append(q)
                ground_truths.append(gt)
                seq += 1

        return queries, ground_truths

    # ── HR Policy queries ─────────────────────────────────────────────────────

    def _gen_hr(self, doc: EnterpriseDocument) -> list[_RawQuery]:
        body = doc.content
        title = doc.metadata.title
        region = doc.metadata.region.value
        doc_id = doc.document_id

        # Strip region suffix from title: "Annual Leave Policy – US" → "Annual Leave"
        policy_name = re.sub(r"\s*[–-]\s*\w+\s*$", "", title).replace(" Policy", "").strip()

        queries: list[_RawQuery] = []

        # Query 1: entitlement value lookup
        entitlement_sentence = _extract_to_stop(body, "The standard entitlement under this policy is ")
        if entitlement_sentence:
            queries.append(_RawQuery(
                category=DocumentCategory.HR_POLICY,
                doc_ids=[doc_id],
                query_text=f"What is the {policy_name.lower()} entitlement for employees at HYTech Solutions?",
                answer_text=entitlement_sentence,
                quotes=[(doc_id, entitlement_sentence, "standard entitlement value")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.NUMERIC_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.KEYWORD,
                difficulty_factors=[],
                answerable=True,
                evaluation_notes="Answer must state the exact numeric entitlement value and unit.",
            ))

        # Query 2: enhanced / region-specific entitlement (if present)
        m_enhanced = re.search(
            r"Employees in (\w+) receive an enhanced entitlement of (.+?) in compliance", body
        )
        if m_enhanced:
            enh_region = m_enhanced.group(1)
            enh_value = m_enhanced.group(2)
            enh_quote = m_enhanced.group(0)
            queries.append(_RawQuery(
                category=DocumentCategory.HR_POLICY,
                doc_ids=[doc_id],
                query_text=f"What {policy_name.lower()} entitlement do employees in {enh_region} receive?",
                answer_text=f"Employees in {enh_region} receive an enhanced entitlement of {enh_value}.",
                quotes=[(doc_id, enh_quote, f"enhanced entitlement for {enh_region} region")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.NUMERIC_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.METADATA_FILTERED,
                difficulty_factors=[RetrievalDifficultyFactor.METADATA_DEPENDENCY],
                answerable=True,
                evaluation_notes=f"Answer must cite the enhanced entitlement for the {enh_region} region specifically.",
            ))

        # Query 3: contractor exception
        exc_sentence = _extract_to_stop(
            body,
            "Contractors engaged through third-party agencies are not eligible",
            ".",
        )
        if exc_sentence:
            queries.append(_RawQuery(
                category=DocumentCategory.HR_POLICY,
                doc_ids=[doc_id],
                query_text=f"Are contractors eligible for the {policy_name.lower()} policy at HYTech Solutions?",
                answer_text="Contractors engaged through third-party agencies are not eligible unless explicitly stated in the engagement contract.",
                quotes=[(doc_id, exc_sentence, "contractor exclusion clause")],
                reasoning_type=ReasoningType.EXCEPTION,
                query_type=QueryType.EXCEPTION_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.DENSE,
                difficulty_factors=[RetrievalDifficultyFactor.EXCEPTION_HANDLING],
                answerable=True,
                evaluation_notes="Answer must state that contractors are not eligible unless the engagement contract says otherwise.",
            ))
        else:
            # Fallback: minimum tenure query
            tenure_line = _extract_line(body, "Minimum tenure: ")
            if tenure_line:
                queries.append(_RawQuery(
                    category=DocumentCategory.HR_POLICY,
                    doc_ids=[doc_id],
                    query_text=f"What is the minimum tenure requirement to be eligible for the {policy_name.lower()} policy?",
                    answer_text=tenure_line,
                    quotes=[(doc_id, tenure_line, "minimum tenure requirement")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.EASY,
                    oracle_strategy=OracleStrategy.KEYWORD,
                    difficulty_factors=[],
                    answerable=True,
                ))

        return queries

    # ── Travel Policy queries ─────────────────────────────────────────────────

    def _gen_travel(self, doc: EnterpriseDocument) -> list[_RawQuery]:
        body = doc.content
        doc_id = doc.document_id
        meta = doc.metadata.category_metadata  # TravelPolicyMetadata
        currency = meta.currency.value  # type: ignore[union-attr]
        expense_type = meta.expense_type.value  # type: ignore[union-attr]
        queries: list[_RawQuery] = []

        # Query 1: receipt threshold
        receipt_sentence = _extract_to_stop(body, "Receipts are mandatory for all expenses exceeding ")
        if receipt_sentence:
            queries.append(_RawQuery(
                category=DocumentCategory.TRAVEL_POLICY,
                doc_ids=[doc_id],
                query_text=f"What is the receipt requirement threshold for {expense_type} expenses in {currency}?",
                answer_text=receipt_sentence,
                quotes=[(doc_id, receipt_sentence, "receipt threshold")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.NUMERIC_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.KEYWORD,
                difficulty_factors=[],
                answerable=True,
                evaluation_notes="Answer must state the exact threshold amount and currency.",
            ))

        # Query 2: manager spending limit (from table)
        manager_row = _find_table_row(body, "Manager")
        if manager_row:
            queries.append(_RawQuery(
                category=DocumentCategory.TRAVEL_POLICY,
                doc_ids=[doc_id],
                query_text=f"What is the maximum {expense_type} reimbursement for managers in {currency}?",
                answer_text=f"According to the spending limits table: {manager_row}",
                quotes=[(doc_id, manager_row, f"manager {expense_type} limit in {currency}")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.TABLE_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes="Answer must state the exact monetary limit for the Manager role from the spending limits table.",
            ))

        # Query 3: employee vs executive comparison (within doc)
        employee_row = _find_table_row(body, "Employee")
        executive_row = _find_table_row(body, "Executive")
        if employee_row and executive_row:
            quotes_list = [
                (doc_id, employee_row, f"employee {expense_type} limit"),
                (doc_id, executive_row, f"executive {expense_type} limit"),
            ]
            queries.append(_RawQuery(
                category=DocumentCategory.TRAVEL_POLICY,
                doc_ids=[doc_id],
                query_text=f"How does the {expense_type} limit differ between employees and executives in {currency}?",
                answer_text=f"Employee row: {employee_row}. Executive row: {executive_row}",
                quotes=quotes_list,
                reasoning_type=ReasoningType.COMPARISON,
                query_type=QueryType.COMPARISON,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes="Answer must compare the exact limit values for Employee and Executive roles.",
            ))
        else:
            # Fallback: reimbursement timeline
            reimb_sentence = _extract_to_stop(body, "Approved claims are reimbursed within ")
            if reimb_sentence:
                queries.append(_RawQuery(
                    category=DocumentCategory.TRAVEL_POLICY,
                    doc_ids=[doc_id],
                    query_text=f"How long does reimbursement take for {expense_type} expenses in {currency}?",
                    answer_text=reimb_sentence,
                    quotes=[(doc_id, reimb_sentence, "reimbursement timeline")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.EASY,
                    oracle_strategy=OracleStrategy.KEYWORD,
                    difficulty_factors=[],
                    answerable=True,
                ))

        return queries

    # ── Security Policy queries ───────────────────────────────────────────────

    def _gen_security(self, doc: EnterpriseDocument) -> list[_RawQuery]:
        body = doc.content
        doc_id = doc.document_id
        meta = doc.metadata.category_metadata  # SecurityPolicyMetadata
        risk_level = meta.risk_level.value  # type: ignore[union-attr]
        policy_title = doc.metadata.title

        queries: list[_RawQuery] = []

        # Query 1: control requirement from table
        ctrl_result = _first_control_row(body)
        if ctrl_result:
            control_id, requirement = ctrl_result
            ctrl_row = _find_table_row(body, control_id) or f"| {control_id} | {requirement} | Active |"
            queries.append(_RawQuery(
                category=DocumentCategory.SECURITY_POLICY,
                doc_ids=[doc_id],
                query_text=f"What does control {control_id} require under the {policy_title}?",
                answer_text=f"{control_id} requires: {requirement}",
                quotes=[(doc_id, ctrl_row, f"requirement for control {control_id}")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.FACTUAL_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes=f"Answer must quote the specific requirement for control {control_id}.",
            ))

        # Query 2: escalation timeline comparison (Critical vs High within same doc)
        critical_row = _find_table_row(body, "Critical")
        high_row = _find_table_row(body, "High")
        if critical_row and high_row:
            quotes_list = [
                (doc_id, critical_row, "critical incident escalation deadline"),
                (doc_id, high_row, "high incident escalation deadline"),
            ]
            queries.append(_RawQuery(
                category=DocumentCategory.SECURITY_POLICY,
                doc_ids=[doc_id],
                query_text=f"What is the escalation deadline for critical security incidents under the {policy_title}?",
                answer_text="Critical incidents must be escalated within 1 hour. High incidents must be escalated within 4 hours.",
                quotes=quotes_list,
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.TABLE_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes="Answer must state the 1-hour escalation deadline for critical incidents.",
            ))
        else:
            # Fallback: risk level info
            risk_section = _extract_to_stop(body, f"This policy addresses {risk_level} risk events", "\n\n")
            if risk_section:
                queries.append(_RawQuery(
                    category=DocumentCategory.SECURITY_POLICY,
                    doc_ids=[doc_id],
                    query_text=f"What risk level does the {policy_title} address?",
                    answer_text=risk_section[:200],
                    quotes=[(doc_id, risk_section[:200], "risk classification")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.EASY,
                    oracle_strategy=OracleStrategy.KEYWORD,
                    difficulty_factors=[],
                    answerable=True,
                ))

        # Query 3: exception handling
        if "No exceptions are permitted" in body:
            no_exc_sentence = _extract_to_stop(body, "No exceptions are permitted for ")
            quote = no_exc_sentence or "No exceptions are permitted"
            queries.append(_RawQuery(
                category=DocumentCategory.SECURITY_POLICY,
                doc_ids=[doc_id],
                query_text=f"Can exceptions be approved for the {policy_title}?",
                answer_text=f"No. Exceptions are not permitted for {risk_level} risk controls. All requirements are non-negotiable.",
                quotes=[(doc_id, quote, "exception policy")],
                reasoning_type=ReasoningType.EXCEPTION,
                query_type=QueryType.EXCEPTION_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.DENSE,
                difficulty_factors=[RetrievalDifficultyFactor.EXCEPTION_HANDLING],
                answerable=True,
                evaluation_notes="Answer must state that no exceptions are permitted for this policy.",
            ))
        elif "Exceptions to this policy may be approved" in body:
            exc_sentence = _extract_to_stop(body, "Exceptions to this policy may be approved")
            queries.append(_RawQuery(
                category=DocumentCategory.SECURITY_POLICY,
                doc_ids=[doc_id],
                query_text=f"Under what conditions can exceptions to the {policy_title} be approved?",
                answer_text="Exceptions may be approved by the Chief Information Security Officer with documented business justification, compensating controls, and a time-limited scope.",
                quotes=[(doc_id, exc_sentence or "Exceptions to this policy may be approved", "exception conditions")],
                reasoning_type=ReasoningType.EXCEPTION,
                query_type=QueryType.EXCEPTION_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.DENSE,
                difficulty_factors=[RetrievalDifficultyFactor.EXCEPTION_HANDLING],
                answerable=True,
                evaluation_notes="Answer must state the CISO approval requirement and conditions for exceptions.",
            ))

        return queries

    # ── API Documentation queries ─────────────────────────────────────────────

    def _gen_api(
        self,
        doc: EnterpriseDocument,
        doc_index: dict[str, EnterpriseDocument],
    ) -> list[_RawQuery]:
        body = doc.content
        doc_id = doc.document_id
        meta = doc.metadata.category_metadata  # APIDocumentationMetadata
        service = meta.service_name.value  # type: ignore[union-attr]
        version = meta.api_version  # type: ignore[union-attr]
        stability = meta.stability.value  # type: ignore[union-attr]
        auth_method = meta.auth_method.value  # type: ignore[union-attr]

        queries: list[_RawQuery] = []

        # Query 1: endpoint parameters
        ep_result = _extract_endpoint_block(body)
        if ep_result:
            method, path, purpose = ep_result
            # Find the endpoint block in the body
            ep_anchor = f"### {method} {path}"
            ep_start = body.find(ep_anchor)
            ep_end = body.find("\n### ", ep_start + 1) if ep_start != -1 else -1
            if ep_end == -1:
                ep_end = min(ep_start + 600, len(body)) if ep_start != -1 else 0
            ep_block = body[ep_start:ep_end].strip() if ep_start != -1 else ep_anchor
            ep_block = ep_block[:400]  # truncate for quote

            queries.append(_RawQuery(
                category=DocumentCategory.API_DOCUMENTATION,
                doc_ids=[doc_id],
                query_text=f"What are the parameters for the {method} {path} endpoint in the {service} {version} API?",
                answer_text=f"{purpose} Parameters are documented in the {service} {version} API reference.",
                quotes=[(doc_id, ep_block, f"endpoint definition for {method} {path}")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.FACTUAL_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.HIERARCHICAL,
                difficulty_factors=[RetrievalDifficultyFactor.METADATA_DEPENDENCY],
                answerable=True,
                evaluation_notes=f"Answer must include the required parameters for {method} {path}.",
            ))

        # Query 2: error code meaning
        error_row = _extract_error_row(body, "401")
        if error_row:
            queries.append(_RawQuery(
                category=DocumentCategory.API_DOCUMENTATION,
                doc_ids=[doc_id],
                query_text=f"What does a 401 error response mean from the {service} {version} API?",
                answer_text=f"A 401 response means Unauthorized. {error_row}",
                quotes=[(doc_id, error_row, "error code 401 meaning")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.TABLE_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes="Answer must state Unauthorized and the recommended action.",
            ))

        # Query 3: deprecation or authentication
        if stability == "deprecated":
            depr_line = _extract_line(body, "DEPRECATION NOTICE:")
            if depr_line:
                queries.append(_RawQuery(
                    category=DocumentCategory.API_DOCUMENTATION,
                    doc_ids=[doc_id],
                    query_text=f"Is the {service} {version} API still recommended for production use?",
                    answer_text=f"No. {depr_line}",
                    quotes=[(doc_id, depr_line, "deprecation notice")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.MEDIUM,
                    oracle_strategy=OracleStrategy.KEYWORD,
                    difficulty_factors=[RetrievalDifficultyFactor.DOCUMENT_VERSION_CONFLICT],
                    answerable=True,
                    evaluation_notes="Answer must confirm the API is deprecated and mention the recommended migration path.",
                ))
            # Multi-hop: find stable v2 doc for same service
            stable_doc = self._find_stable_api_doc(service, doc_index)
            if stable_doc and stable_doc.document_id != doc_id:
                # Use metadata to confirm the stable version auth method
                stable_body = stable_doc.content
                stable_auth_line = _extract_line(stable_body, "Authentication method: ")
                if stable_auth_line:
                    quote_deprecated = depr_line or f"DEPRECATION NOTICE: This {version} API is deprecated."
                    queries.append(_RawQuery(
                        category=DocumentCategory.API_DOCUMENTATION,
                        doc_ids=[doc_id, stable_doc.document_id],
                        query_text=f"The {service} {version} API is deprecated. What authentication method should be used with the current stable {service} API?",
                        answer_text=f"The stable {service} API uses {auth_method} authentication. {stable_auth_line}",
                        quotes=[
                            (doc_id, quote_deprecated, "deprecation notice"),
                            (stable_doc.document_id, stable_auth_line, "stable API auth method"),
                        ],
                        reasoning_type=ReasoningType.MULTI_HOP,
                        query_type=QueryType.MULTI_HOP,
                        difficulty=QueryDifficulty.HARD,
                        oracle_strategy=OracleStrategy.MULTI_HOP,
                        difficulty_factors=[
                            RetrievalDifficultyFactor.DOCUMENT_VERSION_CONFLICT,
                            RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                        ],
                        answerable=True,
                        evaluation_notes="Answer requires checking the deprecated API doc and the stable version doc.",
                    ))
                # Second multi_hop: compare rate limits between deprecated and stable
                depr_rate_row = _find_table_row(body, "Standard")
                stable_rate_row = _find_table_row(stable_body, "Standard")
                if depr_rate_row and stable_rate_row:
                    queries.append(_RawQuery(
                        category=DocumentCategory.API_DOCUMENTATION,
                        doc_ids=[doc_id, stable_doc.document_id],
                        query_text=f"How do the rate limits differ between the deprecated {service} {version} and the current stable version?",
                        answer_text=f"Deprecated {version}: {depr_rate_row}. Stable: {stable_rate_row}",
                        quotes=[
                            (doc_id, depr_rate_row, f"deprecated {version} rate limit"),
                            (stable_doc.document_id, stable_rate_row, "stable API rate limit"),
                        ],
                        reasoning_type=ReasoningType.MULTI_HOP,
                        query_type=QueryType.COMPARISON,
                        difficulty=QueryDifficulty.HARD,
                        oracle_strategy=OracleStrategy.MULTI_HOP,
                        difficulty_factors=[
                            RetrievalDifficultyFactor.DOCUMENT_VERSION_CONFLICT,
                            RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                        ],
                        answerable=True,
                        evaluation_notes="Compare the Standard tier rate limits between the deprecated and stable API versions.",
                    ))
        else:
            auth_line = _extract_line(body, "Authentication method: ")
            if auth_line:
                queries.append(_RawQuery(
                    category=DocumentCategory.API_DOCUMENTATION,
                    doc_ids=[doc_id],
                    query_text=f"What authentication method does the {service} {version} API require?",
                    answer_text=auth_line,
                    quotes=[(doc_id, auth_line, "authentication method")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.EASY,
                    oracle_strategy=OracleStrategy.KEYWORD,
                    difficulty_factors=[],
                    answerable=True,
                ))

        return queries

    def _find_stable_api_doc(
        self, service: str, doc_index: dict[str, EnterpriseDocument]
    ) -> EnterpriseDocument | None:
        for doc in doc_index.values():
            if doc.metadata.category != DocumentCategory.API_DOCUMENTATION:
                continue
            cm = doc.metadata.category_metadata
            if cm is None:
                continue
            if cm.service_name.value == service and cm.stability.value == "stable":  # type: ignore[union-attr]
                return doc
        return None

    # ── System Design queries ─────────────────────────────────────────────────

    def _gen_sysdes(
        self,
        doc: EnterpriseDocument,
        doc_index: dict[str, EnterpriseDocument],
    ) -> list[_RawQuery]:
        body = doc.content
        doc_id = doc.document_id
        system_name = doc.metadata.title
        queries: list[_RawQuery] = []

        # Query 1: SLA
        sla_text = _sysdes_sla(body)
        if sla_text:
            queries.append(_RawQuery(
                category=DocumentCategory.SYSTEM_DESIGN,
                doc_ids=[doc_id],
                query_text=f"What is the uptime SLA for the {system_name}?",
                answer_text=sla_text,
                quotes=[(doc_id, sla_text, "SLA uptime commitment")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.FACTUAL_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.KEYWORD,
                difficulty_factors=[],
                answerable=True,
                evaluation_notes="Answer must state the exact SLA percentage.",
            ))

        # Query 2: first component dependency
        comp_result = _sysdes_first_component(body)
        if comp_result:
            comp_name, comp_deps = comp_result
            comp_row = _find_table_row(body, comp_name)
            if comp_row:
                queries.append(_RawQuery(
                    category=DocumentCategory.SYSTEM_DESIGN,
                    doc_ids=[doc_id],
                    query_text=f"What does the {comp_name} component depend on in the {system_name}?",
                    answer_text=f"{comp_name} depends on: {comp_deps if comp_deps and comp_deps.lower() not in ('none', '') else 'no external dependencies'}.",
                    quotes=[(doc_id, comp_row, f"{comp_name} component dependencies")],
                    reasoning_type=ReasoningType.SINGLE_HOP,
                    query_type=QueryType.FACTUAL_LOOKUP,
                    difficulty=QueryDifficulty.MEDIUM,
                    oracle_strategy=OracleStrategy.HIERARCHICAL,
                    difficulty_factors=[],
                    answerable=True,
                    evaluation_notes=f"Answer must list the dependencies of the {comp_name} component.",
                ))

        # Query 3: multi-hop — which external service handles authentication?
        # Check if the system depends on HYID
        # Try services in order; generate multi_hop for the first one found
        for service_name in ("HYID", "HYDeploy", "HYMonitor", "HYDataLake"):
            if service_name not in body:
                continue
            svc_doc = self._find_api_doc_by_service(service_name, doc_index)
            if svc_doc is None:
                continue
            svc_auth_line = _extract_line(svc_doc.content, "Authentication method: ")
            related_line = _find_table_row(body, service_name)
            if not related_line:
                m = re.search(r"(" + re.escape(service_name) + r"[^\n]*)", body)
                related_line = m.group(1)[:100] if m else service_name
            if svc_auth_line:
                queries.append(_RawQuery(
                    category=DocumentCategory.SYSTEM_DESIGN,
                    doc_ids=[doc_id, svc_doc.document_id],
                    query_text=f"What authentication service and method does {system_name} use for identity management?",
                    answer_text=f"{system_name} uses {service_name} for identity management. {svc_auth_line}",
                    quotes=[
                        (doc_id, related_line, f"{service_name} dependency reference"),
                        (svc_doc.document_id, svc_auth_line, f"{service_name} authentication method"),
                    ],
                    reasoning_type=ReasoningType.MULTI_HOP,
                    query_type=QueryType.MULTI_HOP,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.MULTI_HOP,
                    difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                    answerable=True,
                    evaluation_notes=f"Answer requires checking the system design doc for {service_name} dependency and the {service_name} API doc for auth method.",
                ))
                break  # one multi_hop per SYS-DES doc

        return queries

    def _find_api_doc_by_service(
        self, service: str, doc_index: dict[str, EnterpriseDocument]
    ) -> EnterpriseDocument | None:
        for doc in doc_index.values():
            if doc.metadata.category != DocumentCategory.API_DOCUMENTATION:
                continue
            cm = doc.metadata.category_metadata
            if cm is None:
                continue
            if cm.service_name.value == service and cm.stability.value == "stable":  # type: ignore[union-attr]
                return doc
        return None

    # ── Meeting Notes queries ─────────────────────────────────────────────────

    def _gen_meeting(
        self,
        doc: EnterpriseDocument,
        doc_index: dict[str, EnterpriseDocument],
    ) -> list[_RawQuery]:
        body = doc.content
        doc_id = doc.document_id
        cm = doc.metadata.category_metadata  # MeetingNotesMetadata
        project_code = cm.project_code  # type: ignore[union-attr]
        meeting_type = cm.meeting_type.value  # type: ignore[union-attr]
        meeting_date = str(cm.meeting_date)  # type: ignore[union-attr]
        related_docs = cm.related_documents  # type: ignore[union-attr]

        queries: list[_RawQuery] = []

        # Query 1: key decision
        decision_text = _meeting_key_decision(body)
        if decision_text:
            # Build quote: find the full "1. **...**:" line
            m_decision = re.search(r"(\d+\. \*\*[^*]+\*\*: [^\n]+)", body)
            decision_quote = m_decision.group(1)[:200] if m_decision else decision_text[:200]
            queries.append(_RawQuery(
                category=DocumentCategory.MEETING_NOTES,
                doc_ids=[doc_id],
                query_text=f"What was the first decision recorded in the {meeting_type.replace('_', ' ')} for {project_code} on {meeting_date}?",
                answer_text=decision_text[:200],
                quotes=[(doc_id, decision_quote, "first recorded decision")],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.FACTUAL_LOOKUP,
                difficulty=QueryDifficulty.EASY,
                oracle_strategy=OracleStrategy.KEYWORD,
                difficulty_factors=[],
                answerable=True,
                evaluation_notes="Answer must quote the decision text from the meeting record.",
            ))

        # Query 2: action items or multi-hop reference lookup
        action_row = _meeting_first_action(body)
        if action_row:
            queries.append(_RawQuery(
                category=DocumentCategory.MEETING_NOTES,
                doc_ids=[doc_id],
                query_text=f"What action items were assigned in the {meeting_type.replace('_', ' ')} for {project_code}?",
                answer_text=f"Action item: {action_row}",
                quotes=[(doc_id, action_row, "action item assignment")],
                reasoning_type=ReasoningType.AGGREGATION,
                query_type=QueryType.TABLE_LOOKUP,
                difficulty=QueryDifficulty.MEDIUM,
                oracle_strategy=OracleStrategy.TABLE_AWARE,
                difficulty_factors=[RetrievalDifficultyFactor.TABLE_DEPENDENCY],
                answerable=True,
                evaluation_notes="Answer must list the action items from the meeting notes table.",
            ))

        # Multi-hop: if meeting references another doc, ask a cross-doc question.
        # Iterate up to 2 references; stop after the first successful generation.
        if related_docs:
            generated = 0
            for ref_id in related_docs[:3]:
                if generated >= 1:
                    break
                ref_doc = doc_index.get(ref_id)
                if ref_doc is None:
                    continue
                ref_cat = ref_doc.metadata.category
                ref_title = ref_doc.metadata.title
                refs_line = _extract_line(body, ref_id)
                if not refs_line:
                    continue

                if ref_cat == DocumentCategory.HR_POLICY:
                    ent_sentence = _extract_to_stop(
                        ref_doc.content,
                        "The standard entitlement under this policy is ",
                    )
                    if ent_sentence:
                        queries.append(_RawQuery(
                            category=DocumentCategory.MEETING_NOTES,
                            doc_ids=[doc_id, ref_id],
                            query_text=f"The {project_code} meeting on {meeting_date} referenced {ref_id}. What entitlement does the {ref_title} define?",
                            answer_text=ent_sentence,
                            quotes=[
                                (doc_id, refs_line, f"reference to {ref_id}"),
                                (ref_id, ent_sentence, "referenced policy entitlement"),
                            ],
                            reasoning_type=ReasoningType.MULTI_HOP,
                            query_type=QueryType.MULTI_HOP,
                            difficulty=QueryDifficulty.HARD,
                            oracle_strategy=OracleStrategy.MULTI_HOP,
                            difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                            answerable=True,
                            evaluation_notes="Answer requires reading the meeting notes to find the referenced document and then reading that document.",
                        ))
                        generated += 1

                elif ref_cat == DocumentCategory.TRAVEL_POLICY:
                    receipt_sent = _extract_to_stop(
                        ref_doc.content,
                        "Receipts are mandatory for all expenses exceeding ",
                    )
                    if receipt_sent:
                        queries.append(_RawQuery(
                            category=DocumentCategory.MEETING_NOTES,
                            doc_ids=[doc_id, ref_id],
                            query_text=f"The {project_code} meeting on {meeting_date} referenced {ref_id}. What is the receipt threshold defined in {ref_title}?",
                            answer_text=receipt_sent,
                            quotes=[
                                (doc_id, refs_line, f"reference to {ref_id}"),
                                (ref_id, receipt_sent, "referenced travel policy receipt threshold"),
                            ],
                            reasoning_type=ReasoningType.MULTI_HOP,
                            query_type=QueryType.MULTI_HOP,
                            difficulty=QueryDifficulty.HARD,
                            oracle_strategy=OracleStrategy.MULTI_HOP,
                            difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                            answerable=True,
                            evaluation_notes="Multi-hop: find the referenced travel policy and extract its receipt threshold.",
                        ))
                        generated += 1

                elif ref_cat == DocumentCategory.SECURITY_POLICY:
                    ctrl_result = _first_control_row(ref_doc.content)
                    if ctrl_result:
                        control_id, requirement = ctrl_result
                        ctrl_row = _find_table_row(ref_doc.content, control_id) or requirement[:120]
                        queries.append(_RawQuery(
                            category=DocumentCategory.MEETING_NOTES,
                            doc_ids=[doc_id, ref_id],
                            query_text=f"The {project_code} meeting on {meeting_date} referenced {ref_id}. What does control {control_id} require under that security policy?",
                            answer_text=f"{control_id} requires: {requirement}",
                            quotes=[
                                (doc_id, refs_line, f"reference to {ref_id}"),
                                (ref_id, ctrl_row, f"control {control_id} requirement"),
                            ],
                            reasoning_type=ReasoningType.MULTI_HOP,
                            query_type=QueryType.MULTI_HOP,
                            difficulty=QueryDifficulty.HARD,
                            oracle_strategy=OracleStrategy.MULTI_HOP,
                            difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                            answerable=True,
                            evaluation_notes="Multi-hop: find the referenced security policy and extract the control requirement.",
                        ))
                        generated += 1

                elif ref_cat == DocumentCategory.API_DOCUMENTATION:
                    auth_line = _extract_line(ref_doc.content, "Authentication method: ")
                    if auth_line:
                        ref_meta = ref_doc.metadata.category_metadata
                        service = ref_meta.service_name.value  # type: ignore[union-attr]
                        version = ref_meta.api_version  # type: ignore[union-attr]
                        queries.append(_RawQuery(
                            category=DocumentCategory.MEETING_NOTES,
                            doc_ids=[doc_id, ref_id],
                            query_text=f"The {project_code} meeting on {meeting_date} referenced {ref_id}. What authentication method does the {service} {version} API use?",
                            answer_text=auth_line,
                            quotes=[
                                (doc_id, refs_line, f"reference to {ref_id}"),
                                (ref_id, auth_line, "API authentication method"),
                            ],
                            reasoning_type=ReasoningType.MULTI_HOP,
                            query_type=QueryType.MULTI_HOP,
                            difficulty=QueryDifficulty.HARD,
                            oracle_strategy=OracleStrategy.MULTI_HOP,
                            difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                            answerable=True,
                            evaluation_notes="Multi-hop: find the referenced API doc and extract its authentication method.",
                        ))
                        generated += 1

                elif ref_cat == DocumentCategory.SYSTEM_DESIGN:
                    sla_text = _sysdes_sla(ref_doc.content)
                    if sla_text:
                        queries.append(_RawQuery(
                            category=DocumentCategory.MEETING_NOTES,
                            doc_ids=[doc_id, ref_id],
                            query_text=f"The {project_code} meeting on {meeting_date} referenced {ref_id}. What is the SLA defined in {ref_title}?",
                            answer_text=sla_text,
                            quotes=[
                                (doc_id, refs_line, f"reference to {ref_id}"),
                                (ref_id, sla_text, "system design SLA"),
                            ],
                            reasoning_type=ReasoningType.MULTI_HOP,
                            query_type=QueryType.MULTI_HOP,
                            difficulty=QueryDifficulty.HARD,
                            oracle_strategy=OracleStrategy.MULTI_HOP,
                            difficulty_factors=[RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY],
                            answerable=True,
                            evaluation_notes="Multi-hop: find the referenced system design doc and extract its SLA.",
                        ))
                        generated += 1

        return queries

    # ── Cross-document comparison queries ─────────────────────────────────────

    def _gen_hr_comparisons(self, docs: list[EnterpriseDocument]) -> list[_RawQuery]:
        """Generate region-comparison queries for HR policies with the same topic."""
        hr_docs = [d for d in docs if d.metadata.category == DocumentCategory.HR_POLICY]

        # Group by policy name (title sans region)
        by_topic: dict[str, list[EnterpriseDocument]] = defaultdict(list)
        for doc in hr_docs:
            topic_key = re.sub(r"\s*[–-]\s*\w+\s*$", "", doc.metadata.title).replace(" Policy", "").strip()
            by_topic[topic_key].append(doc)

        queries: list[_RawQuery] = []
        for topic, topic_docs in by_topic.items():
            if len(topic_docs) < 2:
                continue
            d1, d2 = topic_docs[0], topic_docs[1]
            r1, r2 = d1.metadata.region.value, d2.metadata.region.value
            e1 = _extract_to_stop(d1.content, "The standard entitlement under this policy is ")
            e2 = _extract_to_stop(d2.content, "The standard entitlement under this policy is ")
            if e1 and e2:
                queries.append(_RawQuery(
                    category=DocumentCategory.HR_POLICY,
                    doc_ids=[d1.document_id, d2.document_id],
                    query_text=f"How does the {topic.lower()} entitlement differ between the {r1} and {r2} regions at HYTech Solutions?",
                    answer_text=f"{r1} region: {e1}. {r2} region: {e2}",
                    quotes=[
                        (d1.document_id, e1, f"{r1} entitlement"),
                        (d2.document_id, e2, f"{r2} entitlement"),
                    ],
                    reasoning_type=ReasoningType.COMPARISON,
                    query_type=QueryType.COMPARISON,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.METADATA_FILTERED,
                    difficulty_factors=[
                        RetrievalDifficultyFactor.KEYWORD_AMBIGUITY,
                        RetrievalDifficultyFactor.METADATA_DEPENDENCY,
                        RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                    ],
                    answerable=True,
                    evaluation_notes=f"Answer must compare the specific entitlement values for {r1} and {r2} regions.",
                ))

        return queries

    def _gen_travel_comparisons(self, docs: list[EnterpriseDocument]) -> list[_RawQuery]:
        """Generate currency-comparison queries for travel policies with same expense type."""
        trv_docs = [d for d in docs if d.metadata.category == DocumentCategory.TRAVEL_POLICY]
        by_type: dict[str, list[EnterpriseDocument]] = defaultdict(list)
        for doc in trv_docs:
            cm = doc.metadata.category_metadata
            by_type[cm.expense_type.value].append(doc)  # type: ignore[union-attr]

        queries: list[_RawQuery] = []
        for etype, etype_docs in by_type.items():
            if len(etype_docs) < 2:
                continue
            d1, d2 = etype_docs[0], etype_docs[1]
            c1 = d1.metadata.category_metadata.currency.value  # type: ignore[union-attr]
            c2 = d2.metadata.category_metadata.currency.value  # type: ignore[union-attr]
            if c1 == c2:
                continue
            mgr1 = _find_table_row(d1.content, "Manager")
            mgr2 = _find_table_row(d2.content, "Manager")
            if mgr1 and mgr2:
                queries.append(_RawQuery(
                    category=DocumentCategory.TRAVEL_POLICY,
                    doc_ids=[d1.document_id, d2.document_id],
                    query_text=f"How does the {etype} limit for managers compare between {c1} and {c2} policies?",
                    answer_text=f"{c1} policy: {mgr1}. {c2} policy: {mgr2}",
                    quotes=[
                        (d1.document_id, mgr1, f"manager {etype} limit in {c1}"),
                        (d2.document_id, mgr2, f"manager {etype} limit in {c2}"),
                    ],
                    reasoning_type=ReasoningType.COMPARISON,
                    query_type=QueryType.COMPARISON,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.METADATA_FILTERED,
                    difficulty_factors=[
                        RetrievalDifficultyFactor.TABLE_DEPENDENCY,
                        RetrievalDifficultyFactor.METADATA_DEPENDENCY,
                        RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                    ],
                    answerable=True,
                    evaluation_notes=f"Answer must compare manager {etype} limits between {c1} and {c2} policies.",
                ))

        return queries

    def _gen_sec_comparisons(self, docs: list[EnterpriseDocument]) -> list[_RawQuery]:
        """Generate cross-risk-level comparison queries across security policy pairs."""
        sec_docs = [d for d in docs if d.metadata.category == DocumentCategory.SECURITY_POLICY]
        by_family: dict[str, list[EnterpriseDocument]] = defaultdict(list)
        for doc in sec_docs:
            cm = doc.metadata.category_metadata
            by_family[cm.control_family.value].append(doc)  # type: ignore[union-attr]

        queries: list[_RawQuery] = []
        for family, family_docs in by_family.items():
            if len(family_docs) < 2:
                continue
            d1 = family_docs[0]  # typically critical
            d2 = family_docs[1]  # typically high
            r1 = d1.metadata.category_metadata.risk_level.value  # type: ignore[union-attr]
            r2 = d2.metadata.category_metadata.risk_level.value  # type: ignore[union-attr]
            t1 = d1.metadata.title
            t2 = d2.metadata.title
            c1 = _first_control_row(d1.content)
            c2 = _first_control_row(d2.content)
            if c1 and c2:
                queries.append(_RawQuery(
                    category=DocumentCategory.SECURITY_POLICY,
                    doc_ids=[d1.document_id, d2.document_id],
                    query_text=f"What is the difference in exception handling between the {t1} ({r1}) and {t2} ({r2}) policies?",
                    answer_text=f"The {t1} ({r1}) policy does not allow exceptions. The {t2} ({r2}) policy allows exceptions with CISO approval.",
                    quotes=[
                        (d1.document_id, c1[1][:100], f"control requirement under {r1} policy"),
                        (d2.document_id, c2[1][:100], f"control requirement under {r2} policy"),
                    ],
                    reasoning_type=ReasoningType.COMPARISON,
                    query_type=QueryType.COMPARISON,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.METADATA_FILTERED,
                    difficulty_factors=[
                        RetrievalDifficultyFactor.EXCEPTION_HANDLING,
                        RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                    ],
                    answerable=True,
                    evaluation_notes="Answer must compare exception handling rules for both policies.",
                ))

        return queries

    def _gen_mtg_temporal(self, docs: list[EnterpriseDocument]) -> list[_RawQuery]:
        """Generate temporal queries comparing first and last meeting for each project."""
        mtg_docs = [d for d in docs if d.metadata.category == DocumentCategory.MEETING_NOTES]
        by_project: dict[str, list[EnterpriseDocument]] = defaultdict(list)
        for doc in mtg_docs:
            cm = doc.metadata.category_metadata
            by_project[cm.project_code].append(doc)  # type: ignore[union-attr]

        queries: list[_RawQuery] = []
        for project, project_docs in by_project.items():
            if len(project_docs) < 2:
                continue
            # Sort by meeting_date
            sorted_docs = sorted(
                project_docs,
                key=lambda d: d.metadata.category_metadata.meeting_date,  # type: ignore[union-attr]
            )
            first_doc = sorted_docs[0]
            last_doc = sorted_docs[-1]
            first_cm = first_doc.metadata.category_metadata
            last_cm = last_doc.metadata.category_metadata

            first_dec = _meeting_key_decision(first_doc.content)
            last_dec = _meeting_key_decision(last_doc.content)

            if first_dec and last_dec:
                m_first = re.search(r"(\d+\. \*\*[^*]+\*\*: [^\n]+)", first_doc.content)
                m_last = re.search(r"(\d+\. \*\*[^*]+\*\*: [^\n]+)", last_doc.content)
                q1 = m_first.group(1)[:200] if m_first else first_dec[:200]
                q2 = m_last.group(1)[:200] if m_last else last_dec[:200]

                queries.append(_RawQuery(
                    category=DocumentCategory.MEETING_NOTES,
                    doc_ids=[first_doc.document_id, last_doc.document_id],
                    query_text=(
                        f"What was the first decision made in the initial {project} meeting "
                        f"and what was the key outcome of the final meeting?"
                    ),
                    answer_text=(
                        f"Initial meeting ({first_cm.meeting_date}, {first_cm.meeting_type.value}): {first_dec}. "
                        f"Final meeting ({last_cm.meeting_date}, {last_cm.meeting_type.value}): {last_dec}."
                    ),
                    quotes=[
                        (first_doc.document_id, q1, f"first {project} meeting decision"),
                        (last_doc.document_id, q2, f"last {project} meeting outcome"),
                    ],
                    reasoning_type=ReasoningType.TEMPORAL,
                    query_type=QueryType.TEMPORAL,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.MULTI_HOP,
                    difficulty_factors=[
                        RetrievalDifficultyFactor.TEMPORAL_REASONING,
                        RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                    ],
                    answerable=True,
                    evaluation_notes=f"Requires reading both the first and last {project} meeting notes.",
                ))

        return queries

    def _gen_mtg_aggregation(self, docs: list[EnterpriseDocument]) -> list[_RawQuery]:
        """Generate aggregation queries counting items across project meetings."""
        mtg_docs = [d for d in docs if d.metadata.category == DocumentCategory.MEETING_NOTES]
        by_project: dict[str, list[EnterpriseDocument]] = defaultdict(list)
        for doc in mtg_docs:
            cm = doc.metadata.category_metadata
            by_project[cm.project_code].append(doc)  # type: ignore[union-attr]

        queries: list[_RawQuery] = []
        for project, project_docs in by_project.items():
            if len(project_docs) < 3:
                continue

            # Count total decisions across all meetings
            total_decisions = 0
            quotes_list: list[tuple[str, str, str]] = []
            all_doc_ids: list[str] = []

            for doc in project_docs:
                cm = doc.metadata.category_metadata
                count = cm.decisions_count  # type: ignore[union-attr]
                total_decisions += count
                # Get the decisions section header as a quote anchor
                decisions_anchor = "3. Decisions\nThe following decisions were recorded"
                anchor_pos = doc.content.find(decisions_anchor)
                if anchor_pos != -1:
                    q_text = doc.content[anchor_pos:anchor_pos + 80]
                    quotes_list.append((doc.document_id, q_text, f"decisions section in {doc.document_id}"))
                    all_doc_ids.append(doc.document_id)

            if quotes_list and total_decisions > 0:
                queries.append(_RawQuery(
                    category=DocumentCategory.MEETING_NOTES,
                    doc_ids=all_doc_ids[:3],  # cap at 3 for required_document_count
                    query_text=f"How many total decisions were recorded across all {project} project meetings?",
                    answer_text=f"Across all {len(project_docs)} {project} meetings, {total_decisions} decisions were recorded.",
                    quotes=quotes_list[:3],
                    reasoning_type=ReasoningType.AGGREGATION,
                    query_type=QueryType.SUMMARIZATION,
                    difficulty=QueryDifficulty.HARD,
                    oracle_strategy=OracleStrategy.MULTI_HOP,
                    difficulty_factors=[
                        RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY,
                    ],
                    answerable=True,
                    evaluation_notes=f"Requires reading all {project} meeting notes and summing decision counts.",
                ))

        return queries

    # ── Unanswerable queries ──────────────────────────────────────────────────

    def _gen_unanswerable(self) -> list[_RawQuery]:
        queries: list[_RawQuery] = []
        for spec in UNANSWERABLE_SPECS:
            cat = DocumentCategory.MEETING_NOTES  # default
            try:
                for dc in DocumentCategory:
                    if dc.value == spec.category_hint:
                        cat = dc
                        break
            except Exception:
                pass

            queries.append(_RawQuery(
                category=cat,
                doc_ids=[],
                query_text=spec.query_text,
                answer_text="The answer to this question is not available in the SEKD corpus.",
                quotes=[],
                reasoning_type=ReasoningType.SINGLE_HOP,
                query_type=QueryType.UNANSWERABLE,
                difficulty=QueryDifficulty.ADVERSARIAL,
                oracle_strategy=OracleStrategy.DENSE,
                difficulty_factors=[],
                answerable=False,
                disallowed_claims=list(spec.disallowed_claims),
                evaluation_notes=spec.evaluation_notes,
            ))
        return queries

    # ── Build Query + GroundTruth objects ─────────────────────────────────────

    def _build(
        self,
        query_id: str,
        rq: _RawQuery,
        doc_index: dict[str, EnterpriseDocument],
        chunks_by_doc: dict[str, list[Chunk]],
    ) -> tuple[Query, GroundTruth] | None:
        """Convert a _RawQuery to (Query, GroundTruth). Returns None if invalid."""
        if not rq.answerable:
            return self._build_unanswerable(query_id, rq)

        # Collect evidence spans from all quotes
        all_spans = []
        all_chunk_ids: list[str] = []
        all_doc_ids: list[str] = []

        # Group quotes by doc_id
        quotes_by_doc: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for doc_id, quote_text, supports_label in rq.quotes:
            quotes_by_doc[doc_id].append((quote_text, supports_label))

        for doc_id, quote_specs in quotes_by_doc.items():
            doc = doc_index.get(doc_id)
            if doc is None:
                continue
            spans, chunk_ids, doc_ids = build_evidence_spans(
                doc_id,
                doc.content,
                chunks_by_doc.get(doc_id, []),
                quote_specs,
            )
            all_spans.extend(spans)
            for cid in chunk_ids:
                if cid not in all_chunk_ids:
                    all_chunk_ids.append(cid)
            for did in doc_ids:
                if did not in all_doc_ids:
                    all_doc_ids.append(did)

        # Skip if no evidence found for answerable query
        if not all_spans:
            return None

        # Determine required_document_count
        n_docs = len(all_doc_ids)
        if n_docs >= 3:
            req_doc_count = RequiredDocumentCount.THREE_PLUS
        elif n_docs == 2:
            req_doc_count = RequiredDocumentCount.TWO
        else:
            req_doc_count = RequiredDocumentCount.ONE

        # Ensure multi-doc flag is present when needed
        difficulty_factors = list(rq.difficulty_factors)
        if req_doc_count in (RequiredDocumentCount.TWO, RequiredDocumentCount.THREE_PLUS):
            if RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY not in difficulty_factors:
                difficulty_factors.append(RetrievalDifficultyFactor.MULTI_DOCUMENT_DEPENDENCY)

        try:
            query = Query(
                query_id=query_id,
                query=rq.query_text,
                difficulty=rq.difficulty,
                reasoning_type=rq.reasoning_type,
                planner_oracle_strategy=rq.oracle_strategy,
                oracle_strategy_source=rq.oracle_strategy_source,
                retrieval_difficulty_factors=difficulty_factors,
                required_document_count=req_doc_count,
                category=rq.category,
                query_type=rq.query_type,
                answerable=True,
                required_document_ids=all_doc_ids,
                required_chunk_ids=all_chunk_ids,
            )
        except Exception:
            return None

        acceptable_patterns = rq.acceptable_patterns
        if not acceptable_patterns and rq.answer_text:
            # Add a simple pattern matching the first significant token of the answer
            first_word = rq.answer_text.split()
            if first_word:
                pass  # leave empty for flexibility

        try:
            gt = GroundTruth(
                query_id=query_id,
                reference_answer=rq.answer_text,
                required_document_ids=all_doc_ids,
                required_chunk_ids=all_chunk_ids,
                evidence=all_spans,
                expected_citations=all_chunk_ids[:3],  # cite up to 3 chunks
                acceptable_answer_patterns=acceptable_patterns,
                disallowed_claims=rq.disallowed_claims,
                evaluation_notes=rq.evaluation_notes,
            )
        except Exception:
            return None

        return query, gt

    def _build_unanswerable(
        self, query_id: str, rq: _RawQuery
    ) -> tuple[Query, GroundTruth] | None:
        try:
            query = Query(
                query_id=query_id,
                query=rq.query_text,
                difficulty=rq.difficulty,
                reasoning_type=rq.reasoning_type,
                planner_oracle_strategy=rq.oracle_strategy,
                oracle_strategy_source=rq.oracle_strategy_source,
                retrieval_difficulty_factors=rq.difficulty_factors,
                required_document_count=RequiredDocumentCount.ONE,
                category=rq.category,
                query_type=rq.query_type,
                answerable=False,
                required_document_ids=[],
                required_chunk_ids=[],
            )
            gt = GroundTruth(
                query_id=query_id,
                reference_answer="The information requested is not available in the SEKD corpus.",
                required_document_ids=[],
                required_chunk_ids=[],
                evidence=[],
                expected_citations=[],
                acceptable_answer_patterns=[],
                disallowed_claims=rq.disallowed_claims,
                evaluation_notes=rq.evaluation_notes,
            )
            return query, gt
        except Exception:
            return None


# ── Generation report ─────────────────────────────────────────────────────────


@dataclass
class QueryGenerationReport:
    total_queries: int
    answerable_count: int
    unanswerable_count: int
    per_category: dict[str, int]
    per_reasoning_type: dict[str, int]
    per_difficulty: dict[str, int]
    per_oracle_strategy: dict[str, int]
    duration_seconds: float


def build_generation_report(
    queries: list[Query],
    elapsed: float,
) -> QueryGenerationReport:
    per_cat: Counter = Counter(q.category.value for q in queries)
    per_reason: Counter = Counter(q.reasoning_type.value for q in queries)
    per_diff: Counter = Counter(q.difficulty.value for q in queries)
    per_strat: Counter = Counter(q.planner_oracle_strategy.value for q in queries)
    answerable = sum(1 for q in queries if q.answerable)

    return QueryGenerationReport(
        total_queries=len(queries),
        answerable_count=answerable,
        unanswerable_count=len(queries) - answerable,
        per_category=dict(per_cat),
        per_reasoning_type=dict(per_reason),
        per_difficulty=dict(per_diff),
        per_oracle_strategy=dict(per_strat),
        duration_seconds=elapsed,
    )
