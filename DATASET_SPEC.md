# DATASET_SPEC.md

## Dataset Name

SEKD: Synthetic Enterprise Knowledge Dataset

## Version

1.2

## Purpose

SEKD is designed to evaluate Agentic RAG and Retrieval Strategy Planning for Enterprise Knowledge Management Systems. The dataset models a synthetic company knowledge base with heterogeneous document types, structured metadata, controlled facts, and annotated ground truth for retrieval and answer generation evaluation.

The dataset should support experiments comparing:

- Fixed keyword retrieval
- Fixed dense retrieval
- Fixed hybrid retrieval
- Rule-based adaptive retrieval planning
- LLM-based adaptive retrieval planning
- Multi-hop and document-type-aware retrieval strategies

## Synthetic Company

Company name: HYTech Solutions

Company profile:

- Industry: Enterprise software, cloud infrastructure, and AI platform services
- Size: 1,200 employees
- Locations: Seoul, Singapore, San Francisco, Berlin
- Departments: Human Resources, Finance, Security, Platform Engineering, Product Engineering, Solutions Engineering, Legal, Sales Operations
- Internal systems: HYPortal, HYDrive, HYVPN, HYID, HYDeploy, HYMonitor, HYDataLake

All documents must be synthetic. Do not include real employee names, real customer data, real access credentials, or real confidential business information.

## Dataset Output Format

The dataset generator should produce the following files:

```text
data/sekd/
├── documents.jsonl
├── chunks.jsonl
├── queries.jsonl
├── ground_truth.jsonl
├── metadata_schema.json
├── category_config.json
└── README.md
```

### documents.jsonl

Each line represents one full document.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| document_id | string | Unique document ID |
| category | string | One of the six document categories |
| title | string | Document title |
| body | string | Full document text |
| metadata | object | Category-specific metadata |
| created_at | string | ISO 8601 date |
| updated_at | string | ISO 8601 date |
| version | string | Semantic or policy version |
| source_path | string | Synthetic source path |
| sensitivity | string | public_internal, confidential, restricted |

### chunks.jsonl

Each line represents a retrievable chunk.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| chunk_id | string | Unique chunk ID |
| document_id | string | Parent document ID |
| category | string | Document category |
| section_path | array[string] | Hierarchical section path |
| text | string | Chunk text |
| metadata | object | Inherited and chunk-specific metadata |
| token_estimate | integer | Approximate token count |
| char_start | integer | Start character offset in document body |
| char_end | integer | End character offset in document body |

### queries.jsonl

Each line represents one evaluation query.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| query_id | string | Unique query ID |
| query | string | User-facing query |
| category | string | Primary expected document category |
| query_type | string | Query taxonomy label |
| difficulty | string | easy, medium, hard, adversarial |
| expected_strategy | string | Expected retrieval strategy |
| planner_oracle_strategy | string | Retrieval strategy that should theoretically perform best for answering the query |
| oracle_strategy_source | string | Provenance of the oracle strategy label |
| reasoning_type | string | Reasoning complexity required to answer the query |
| retrieval_difficulty_factors | array[string] | Optional difficulty dimensions affecting retrieval |
| required_document_count | string | Number of documents required to answer correctly: 1, 2, or 3+ |
| answerable | boolean | Whether answer exists in corpus |
| required_document_ids | array[string] | Documents expected to support the answer |
| required_chunk_ids | array[string] | Chunks expected to support the answer |

Allowed `planner_oracle_strategy` values:

- keyword
- dense
- hybrid
- metadata_filtered
- hierarchical
- table_aware
- multi_hop

Allowed `oracle_strategy_source` values:

- manual_annotation
- heuristic_assignment
- empirical_validation

Allowed `reasoning_type` values:

- single_hop
- multi_hop
- comparison
- temporal
- exception
- aggregation

Example query record:

```json
{
  "query_id": "Q-000001",
  "query": "What is the hotel limit for EU managers?",
  "planner_oracle_strategy": "table_aware",
  "oracle_strategy_source": "heuristic_assignment",
  "reasoning_type": "single_hop",
  "retrieval_difficulty_factors": [
    "metadata_dependency",
    "table_dependency"
  ],
  "required_document_count": "1"
}
```

The `planner_oracle_strategy` field will later be used to evaluate Rule-Based Planner and LLM-Based Planner performance with Strategy Selection Accuracy, Planner Precision, and Planner Recall.

The `planner_oracle_strategy` field must not be treated as absolute truth. The `oracle_strategy_source` field records whether the oracle was assigned by manual annotation, heuristic assignment, or empirical validation.

### ground_truth.jsonl

Each line represents the reference answer and evidence.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| query_id | string | Matching query ID |
| reference_answer | string | Gold answer |
| evidence | array[object] | Supporting evidence spans |
| expected_citations | array[string] | Document or chunk IDs that should be cited |
| acceptable_answer_patterns | array[string] | Optional regex or phrase patterns |
| disallowed_claims | array[string] | Claims that should not appear |
| evaluation_notes | string | Human-readable grading note |

## Global Metadata Schema

All documents must include these metadata fields:

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| company | string | yes | Must be HYTech Solutions |
| department | string | yes | Owning department |
| document_owner | string | yes | Synthetic role or team owner |
| region | string | yes | global, KR, SG, US, EU |
| effective_date | string | yes | Date document became effective |
| review_cycle | string | yes | monthly, quarterly, semiannual, annual |
| status | string | yes | draft, active, deprecated, archived |
| authority_level | string | yes | advisory, standard, mandatory |
| confidentiality | string | yes | public_internal, confidential, restricted |
| tags | array[string] | yes | Search and filtering tags |

## Document Categories

## 1. HR Policies

### Business Purpose

HR Policies define employee rights, benefits, responsibilities, workplace rules, performance processes, and employment lifecycle procedures. These documents support employee self-service, HR operations, manager decision-making, and compliance with internal standards.

### Document Characteristics

- Formal policy language
- Section-based structure
- Many eligibility rules and exceptions
- Region-specific clauses
- Effective dates and version history
- Frequently queried by employees and managers

Typical topics:

- Paid time off
- Sick leave
- Parental leave
- Remote work
- Performance review
- Promotion process
- Employee conduct
- Training reimbursement

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| policy_id | string | yes | HR policy identifier, e.g. HR-POL-001 |
| policy_area | string | yes | leave, benefits, remote_work, performance, conduct |
| employee_type | string | yes | full_time, part_time, contractor, intern, all |
| applicable_regions | array[string] | yes | Regions where policy applies |
| approval_body | string | yes | HR Committee, Legal, Executive Office |
| supersedes | string or null | yes | Previous policy ID if applicable |
| escalation_contact | string | yes | Responsible HR role |

### Required Fields

- policy_id
- title
- effective_date
- scope
- eligibility
- policy_statement
- procedure
- exceptions
- approval_body
- revision_history

### Document Template

```text
Title: {title}
Policy ID: {policy_id}
Owner: {document_owner}
Effective Date: {effective_date}
Version: {version}
Applicable Regions: {applicable_regions}

1. Purpose
{purpose}

2. Scope
This policy applies to {employee_type} employees in {applicable_regions}.

3. Eligibility
{eligibility_rules}

4. Policy Statement
{policy_statement}

5. Procedure
{numbered_steps}

6. Exceptions
{exception_rules}

7. Manager Responsibilities
{manager_responsibilities}

8. Employee Responsibilities
{employee_responsibilities}

9. Escalation
Questions must be sent to {escalation_contact}.

10. Revision History
{revision_history}
```

### Realistic Generation Rules

- Generate 20-40 HR policy documents.
- Each policy must have 5-12 sections.
- Include at least one numeric entitlement or threshold per document.
- Include region-specific differences in 30-50% of documents.
- Include exception clauses in 60-80% of documents.
- Include version history with 2-4 revisions.
- Use synthetic role names such as HR Operations Lead, People Partner, or Compensation Manager.
- Avoid using real personal names.
- Ensure some policies supersede older policies.

### Retrieval Challenges

- Similar policy names may cause retrieval ambiguity.
- Region-specific clauses require metadata filtering.
- Exceptions may override general rules.
- Numeric thresholds must be retrieved exactly.
- Users may ask using informal terms instead of official policy names.

## 2. Travel Policies

### Business Purpose

Travel Policies define business travel approval, booking, reimbursement, expense limits, allowed vendors, and documentation requirements. These documents support employees, finance teams, and managers who need consistent travel expense decisions.

### Document Characteristics

- Rule-heavy policy documents
- Tables for spending limits
- Region and role-dependent conditions
- Many approval workflows
- Strong need for exact values and exceptions

Typical topics:

- Flight class eligibility
- Hotel nightly limits
- Meal allowance
- Taxi and rideshare reimbursement
- International travel approval
- Receipt requirements
- Currency conversion

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| travel_policy_id | string | yes | Travel policy identifier, e.g. TRV-POL-001 |
| expense_type | string | yes | airfare, lodging, meals, ground_transport, visa, miscellaneous |
| applicable_roles | array[string] | yes | employee, manager, executive, sales, engineer |
| currency | string | yes | KRW, USD, EUR, SGD |
| approval_required | boolean | yes | Whether approval is required |
| finance_owner | string | yes | Finance role responsible |
| receipt_required_threshold | number | yes | Amount above which receipt is required |

### Required Fields

- travel_policy_id
- title
- purpose
- covered_expenses
- spending_limits
- approval_workflow
- receipt_requirements
- reimbursement_timeline
- exceptions
- audit_rules

### Document Template

```text
Title: {title}
Travel Policy ID: {travel_policy_id}
Owner: {finance_owner}
Effective Date: {effective_date}
Version: {version}
Currency: {currency}

1. Purpose
{purpose}

2. Covered Expenses
{covered_expenses}

3. Spending Limits
| Expense Type | Region | Limit | Approval Required |
| --- | --- | --- | --- |
{spending_limit_rows}

4. Approval Workflow
{approval_workflow}

5. Receipt Requirements
Receipts are required for expenses above {receipt_required_threshold} {currency}.

6. Reimbursement Timeline
{reimbursement_timeline}

7. Exceptions
{exception_rules}

8. Audit and Compliance
{audit_rules}
```

### Realistic Generation Rules

- Generate 15-30 travel policy documents.
- At least 70% of documents must contain tables.
- Spending limits must vary by region and expense type.
- Include approval thresholds for at least three employee roles.
- Include receipt rules and reimbursement deadlines.
- Include conflicting-looking but resolvable rules, such as general meal limits and higher client-meal limits.
- Include currency-specific values.

### Retrieval Challenges

- Table retrieval is required for expense limits.
- Queries often require exact numeric values.
- Role, region, and expense type must be combined.
- General rules and exception rules may conflict.
- Currency and reimbursement deadlines require precise grounding.

## 3. Security Policies

### Business Purpose

Security Policies define mandatory controls for access management, data protection, incident response, device security, vendor security, and compliance. These documents support security operations, engineering teams, auditors, and employees.

### Document Characteristics

- Mandatory language
- High authority level
- Control IDs and compliance references
- Strict exception and escalation paths
- Strong emphasis on exact terms

Typical topics:

- Password and MFA requirements
- Data classification
- Incident reporting
- Laptop encryption
- Source code access
- Vendor security review
- Production access
- Secrets management

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| security_policy_id | string | yes | Security policy identifier, e.g. SEC-POL-001 |
| control_family | string | yes | access_control, data_protection, incident_response, device_security |
| control_ids | array[string] | yes | Synthetic control IDs, e.g. HY-AC-01 |
| risk_level | string | yes | low, medium, high, critical |
| enforcement_owner | string | yes | Security team role |
| exception_allowed | boolean | yes | Whether exceptions are allowed |
| review_frequency_days | integer | yes | Review frequency |

### Required Fields

- security_policy_id
- title
- control_objective
- scope
- mandatory_requirements
- prohibited_actions
- exception_process
- incident_escalation
- control_mapping
- enforcement

### Document Template

```text
Title: {title}
Security Policy ID: {security_policy_id}
Owner: {enforcement_owner}
Effective Date: {effective_date}
Version: {version}
Control Family: {control_family}
Risk Level: {risk_level}

1. Control Objective
{control_objective}

2. Scope
{scope}

3. Mandatory Requirements
{mandatory_requirements}

4. Prohibited Actions
{prohibited_actions}

5. Exception Process
{exception_process}

6. Incident Escalation
{incident_escalation}

7. Control Mapping
{control_mapping}

8. Enforcement
{enforcement_rules}
```

### Realistic Generation Rules

- Generate 20-35 security policy documents.
- Every document must include at least two synthetic control IDs.
- Include mandatory requirements using words such as must, required, prohibited, and mandatory.
- Include escalation timelines such as 1 hour, 4 hours, 24 hours, or 3 business days.
- Include exception workflows in 40-70% of documents.
- Include deprecated controls in a small subset to test version and status filtering.
- Use restricted confidentiality for high-risk documents.

### Retrieval Challenges

- Exact control ID matching is important.
- Deprecated and active policies may be similar.
- Queries may require mandatory vs recommended distinction.
- Incident timelines must be retrieved exactly.
- Security exceptions require multi-section retrieval.

## 4. API Documentation

### Business Purpose

API Documentation describes internal and external service interfaces, authentication, endpoints, request and response formats, error codes, rate limits, and integration examples. These documents support developers, solutions engineers, and platform teams.

### Document Characteristics

- Highly structured technical content
- Endpoint paths and HTTP methods
- Parameter tables
- Code-like examples
- Error code references
- Versioned APIs

Typical topics:

- HYID identity API
- HYDeploy release API
- HYMonitor metrics API
- HYDataLake ingestion API
- Billing API
- Notification API

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| api_doc_id | string | yes | API document identifier, e.g. API-DOC-001 |
| service_name | string | yes | HYID, HYDeploy, HYMonitor, HYDataLake |
| api_version | string | yes | v1, v2, v3 |
| endpoint_count | integer | yes | Number of endpoints described |
| auth_method | string | yes | OAuth2, API key, mTLS, service token |
| owner_team | string | yes | Owning engineering team |
| stability | string | yes | experimental, beta, stable, deprecated |

### Required Fields

- api_doc_id
- service_name
- api_version
- authentication
- endpoint_reference
- request_schema
- response_schema
- error_codes
- rate_limits
- examples
- changelog

### Document Template

```text
Title: {service_name} API Reference {api_version}
API Doc ID: {api_doc_id}
Owner Team: {owner_team}
Stability: {stability}
Effective Date: {effective_date}
Version: {version}

1. Overview
{overview}

2. Authentication
Authentication method: {auth_method}
{authentication_details}

3. Endpoint Reference
{endpoint_blocks}

4. Request Schema
{request_schema}

5. Response Schema
{response_schema}

6. Error Codes
| Code | Meaning | Recommended Action |
| --- | --- | --- |
{error_code_rows}

7. Rate Limits
{rate_limits}

8. Examples
{examples}

9. Changelog
{changelog}
```

Endpoint block template:

```text
Endpoint: {http_method} {path}
Purpose: {endpoint_purpose}
Required Parameters: {required_parameters}
Optional Parameters: {optional_parameters}
Response: {response_summary}
Errors: {error_codes}
```

### Realistic Generation Rules

- Generate 20-40 API documents.
- Each document must contain 3-10 endpoints.
- Endpoint paths must use realistic internal service names.
- Include parameter names using snake_case or camelCase consistently per service.
- Include 4-8 error codes per document.
- Include rate limits in at least 80% of documents.
- Include deprecated endpoints in 20-30% of documents.
- Include examples but never real credentials.

### Retrieval Challenges

- Exact endpoint and parameter matching is required.
- Similar endpoint names across API versions may confuse retrieval.
- Deprecated endpoint filtering is important.
- Error code questions require table or list retrieval.
- Auth and rate limit answers may live in different sections.

## 5. System Design Documents

### Business Purpose

System Design Documents describe architecture decisions, components, data flows, dependencies, scalability assumptions, failure handling, and tradeoffs for HYTech Solutions systems. These documents support engineering onboarding, architecture review, incident analysis, and technical decision-making.

### Document Characteristics

- Long-form technical explanations
- Component and dependency lists
- Architecture decision records
- Sequence and data-flow descriptions
- Tradeoff analysis
- Non-functional requirements

Typical topics:

- HYDeploy deployment pipeline design
- HYMonitor event ingestion design
- HYDataLake partitioning design
- HYID authentication architecture
- Notification routing design
- Billing reconciliation design

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| design_doc_id | string | yes | Design document identifier, e.g. SYS-DES-001 |
| system_name | string | yes | System being described |
| architecture_status | string | yes | proposed, approved, implemented, retired |
| primary_author_role | string | yes | Synthetic role, not person name |
| reviewer_roles | array[string] | yes | Architect, Security Reviewer, SRE, Product Manager |
| related_services | array[string] | yes | Dependent HYTech services |
| decision_record_ids | array[string] | yes | Synthetic ADR IDs |

### Required Fields

- design_doc_id
- system_name
- problem_statement
- goals
- non_goals
- architecture_overview
- components
- data_flow
- dependencies
- tradeoffs
- failure_modes
- security_considerations
- open_questions
- decision_records

### Document Template

```text
Title: {system_name} System Design
Design Doc ID: {design_doc_id}
Owner: {document_owner}
Architecture Status: {architecture_status}
Effective Date: {effective_date}
Version: {version}

1. Problem Statement
{problem_statement}

2. Goals
{goals}

3. Non-Goals
{non_goals}

4. Architecture Overview
{architecture_overview}

5. Components
{component_descriptions}

6. Data Flow
{data_flow}

7. Dependencies
{dependencies}

8. Tradeoffs
{tradeoffs}

9. Failure Modes
{failure_modes}

10. Security Considerations
{security_considerations}

11. Open Questions
{open_questions}

12. Decision Records
{decision_records}
```

### Realistic Generation Rules

- Generate 15-30 system design documents.
- Each document must include 4-8 components.
- Include at least three explicit dependencies per document.
- Include 2-5 tradeoffs.
- Include 2-4 failure modes.
- Include at least one security consideration.
- Include synthetic ADR IDs such as ADR-HYD-001.
- Some documents should reference the same services as API documentation to enable multi-document queries.

### Retrieval Challenges

- Long documents require hierarchical retrieval.
- Queries may require linking components, dependencies, and tradeoffs.
- Multi-hop retrieval is needed across design docs and API docs.
- Goals and non-goals must not be confused.
- Architecture status filtering is important.

## 6. Project Meeting Notes

### Business Purpose

Project Meeting Notes capture decisions, action items, blockers, timelines, ownership, and follow-up items from HYTech Solutions projects. These documents support project memory, accountability, and decision traceability.

### Document Characteristics

- Semi-structured notes
- Date-centered
- Action items and owners
- Decisions and blockers
- Informal wording
- References to projects, systems, and policies

Typical topics:

- Sprint planning
- Architecture review
- Incident postmortem
- Product launch readiness
- Security review
- Customer escalation review

### Metadata Schema

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| meeting_id | string | yes | Meeting note identifier, e.g. MTG-001 |
| project_code | string | yes | Synthetic project code, e.g. PRJ-HYD-ALPHA |
| meeting_type | string | yes | sprint_planning, architecture_review, postmortem, launch_review |
| meeting_date | string | yes | ISO 8601 date |
| participants | array[string] | yes | Synthetic role names |
| decisions_count | integer | yes | Number of decisions |
| action_items_count | integer | yes | Number of action items |
| related_documents | array[string] | yes | Related policy, API, or design doc IDs |

### Required Fields

- meeting_id
- project_code
- meeting_date
- meeting_type
- agenda
- participants
- discussion_summary
- decisions
- action_items
- blockers
- follow_ups
- related_documents

### Document Template

```text
Title: {project_code} {meeting_type} Meeting Notes
Meeting ID: {meeting_id}
Date: {meeting_date}
Participants: {participants}
Related Documents: {related_documents}

1. Agenda
{agenda}

2. Discussion Summary
{discussion_summary}

3. Decisions
{decisions}

4. Action Items
| Item | Owner Role | Due Date | Status |
| --- | --- | --- | --- |
{action_item_rows}

5. Blockers
{blockers}

6. Follow-Ups
{follow_ups}
```

### Realistic Generation Rules

- Generate 30-60 meeting note documents.
- Meeting dates should span 6-18 months.
- Each note must include 1-5 decisions.
- Each note must include 2-8 action items.
- Participants must be role names, not real names.
- At least 50% of notes should reference another document ID.
- Some action items should be completed, some pending, and some blocked.
- Generate recurring project codes to enable timeline questions.

### Retrieval Challenges

- Informal wording may not match query wording.
- Date and project metadata filtering is important.
- Decisions may require chronological retrieval across multiple meetings.
- Action item ownership requires table-aware retrieval.
- Related document references enable multi-hop queries.

## Query Taxonomy

Each generated query must be assigned exactly one primary query type.

| Query Type | Description | Example |
| --- | --- | --- |
| factual_lookup | Ask for a specific fact from one document | What is the receipt threshold for domestic meals? |
| policy_interpretation | Ask how a policy applies under conditions | Can a Singapore contractor request remote work reimbursement? |
| procedural | Ask for steps or workflow | How should an employee request parental leave? |
| numeric_lookup | Ask for exact limits, thresholds, counts, or deadlines | What is the hotel limit in Berlin? |
| table_lookup | Ask for a value from a table | Which error action is recommended for HYD-429? |
| comparison | Compare rules, systems, versions, or regions | How do KR and EU travel meal limits differ? |
| multi_hop | Require evidence from two or more documents | Which API rate limit affects the HYMonitor design decision? |
| temporal | Ask about decisions over time | What changed after the March launch review? |
| owner_lookup | Ask who owns a policy, action item, or system | Which team owns the HYDeploy release API? |
| exception_lookup | Ask about exceptions to a general rule | When can business class travel be approved? |
| troubleshooting | Ask how to resolve an error or operational issue | What should a developer do after receiving HYD-503? |
| summarization | Ask for a concise summary of a document or section | Summarize the failure modes of HYDataLake ingestion. |
| unanswerable | Ask for information absent from the corpus | What is the CEO's private phone number? |

## Query Difficulty Levels

| Difficulty | Definition | Retrieval Expectation |
| --- | --- | --- |
| easy | One clear fact in one document, direct wording overlap | Keyword, dense, or hybrid retrieval should succeed |
| medium | One document, paraphrased query, section-level reasoning | Dense or hybrid retrieval with reranking should succeed |
| hard | Multiple constraints, tables, exceptions, or multiple sections | Adaptive retrieval should outperform fixed retrieval |
| adversarial | Ambiguous, deprecated, conflicting, or unanswerable query | Planner must filter, verify, or refuse correctly |

Difficulty distribution:

- easy: 30%
- medium: 35%
- hard: 25%
- adversarial: 10%

## Query Generation Rules

The generator should create 300-600 queries.

General rules:

- Generate at least 40 queries per document category.
- Generate at least 20 unanswerable queries.
- Generate at least 50 multi-hop queries.
- Generate at least 50 table or numeric lookup queries.
- Generate paraphrased queries that do not exactly match source wording.
- Include document ID, policy ID, control ID, API endpoint, project code, or region in some queries.
- For every answerable query, store exact supporting document IDs and chunk IDs.
- For every unanswerable query, set required document IDs and chunk IDs to empty arrays.
- Avoid queries requiring external world knowledge.

Category-specific query rules:

- HR Policies: emphasize eligibility, exceptions, region-specific rules, and procedures.
- Travel Policies: emphasize numeric limits, approval thresholds, receipts, and reimbursements.
- Security Policies: emphasize mandatory controls, exceptions, incident timelines, and control IDs.
- API Documentation: emphasize endpoints, parameters, error codes, auth methods, and rate limits.
- System Design Documents: emphasize components, dependencies, tradeoffs, goals, and failure modes.
- Project Meeting Notes: emphasize decisions, action items, blockers, dates, and project history.

Expected retrieval strategy labels:

| Label | Use When |
| --- | --- |
| keyword | Exact IDs, endpoint paths, control IDs, or error codes |
| dense | Paraphrased conceptual queries |
| hybrid | General enterprise questions requiring both semantic and exact matching |
| metadata_filtered | Region, department, date, status, role, or document category constraints |
| hierarchical | Long structured documents with section-level context |
| table_aware | Tables, rows, numeric limits, error codes, action items |
| multi_hop | Evidence across multiple documents or sections |
| faq_pair | Reserved for future FAQ extensions |

## Planner Ablation Study Labels

SEKD should support a dedicated planner ablation study with the following comparison conditions:

| Label | Condition | Description |
| --- | --- | --- |
| fixed_bm25 | A. Fixed BM25 | Sparse keyword retrieval without planning |
| fixed_dense | B. Fixed Dense Retrieval | Dense semantic retrieval without planning |
| fixed_hybrid | C. Fixed Hybrid Retrieval | Fixed dense plus sparse retrieval without planning |
| rule_based_planner | D. Rule-Based Retrieval Planner | Deterministic strategy selection followed by selected retrieval |
| llm_based_planner | E. LLM-Based Retrieval Planner | LLM-selected retrieval strategy followed by selected retrieval |

Purpose:

This ablation isolates the contribution of retrieval planning from retrieval quality itself.

Evaluation metrics:

- Recall@K
- Precision@K
- MRR
- nDCG
- Strategy Selection Accuracy

Expected analysis:

- When does planning help?
- When does planning hurt?
- Which query categories benefit most?

## Retrieval Difficulty Factors

Each generated query should optionally include `retrieval_difficulty_factors` to support deeper experimental analysis.

Allowed values:

| Factor | Description |
| --- | --- |
| keyword_ambiguity | Query contains terms that match multiple documents, versions, systems, or policies |
| document_version_conflict | Active, deprecated, draft, or archived documents contain similar content |
| metadata_dependency | Correct answer requires filtering by region, date, role, department, status, or category |
| table_dependency | Correct answer requires interpreting a table, row, column, or numeric cell |
| multi_document_dependency | Correct answer requires evidence from more than one document |
| exception_handling | Correct answer depends on exception clauses or override rules |
| temporal_reasoning | Correct answer depends on dates, sequence, latest status, or historical change |

Example:

```json
[
  "metadata_dependency",
  "table_dependency"
]
```

These factors should be generated independently from the coarse difficulty label. For example, a query may be `medium` difficulty but still include `metadata_dependency`, while a `hard` query may include `multi_document_dependency`, `exception_handling`, and `temporal_reasoning`.

## Multi-Document Dependency Annotation

Each query must include `required_document_count`.

Allowed values:

- 1
- 2
- 3+

Description:

`required_document_count` is the number of documents required to answer correctly. This field separates simple retrieval tasks from true multi-hop enterprise tasks.

Generation rules:

- Use `1` when all required evidence is in a single document.
- Use `2` when the answer requires cross-checking or combining two documents.
- Use `3+` when the answer requires aggregation, chronology, or comparison across three or more documents.
- If `required_document_count` is `2` or `3+`, include `multi_document_dependency` in `retrieval_difficulty_factors`.
- If `answerable` is false, set `required_document_count` to `1` unless the query is intentionally designed as an unanswerable multi-document task.

## Experimental Query Distribution

The dataset generator should attempt to satisfy the following target distributions.

Reasoning type distribution:

| Reasoning Type | Target |
| --- | --- |
| single_hop | 35% |
| multi_hop | 20% |
| comparison | 15% |
| temporal | 10% |
| exception | 10% |
| aggregation | 10% |

Oracle strategy distribution:

| Planner Oracle Strategy | Target |
| --- | --- |
| keyword | 15% |
| dense | 15% |
| hybrid | 30% |
| metadata_filtered | 15% |
| hierarchical | 10% |
| table_aware | 10% |
| multi_hop | 5% |

The generator should treat these as target distributions, not hard constraints. If the requested query count is too small to satisfy every percentage exactly, it should round counts while preserving the overall distribution as closely as possible.

## Planner Evaluation Schema

Planner evaluation records should be generated after a planner selects a strategy for each query.

Required fields:

| Field | Type | Description |
| --- | --- | --- |
| query_id | string | Query evaluated by the planner |
| selected_strategy | string | Strategy selected by the planner |
| oracle_strategy | string | Value copied from `planner_oracle_strategy` |
| strategy_correct | boolean | Whether selected strategy equals oracle strategy |
| planner_type | string | Planner implementation type |

Allowed `planner_type` values:

- rule_based
- llm_based

Purpose:

This schema allows direct comparison between planner implementations.

Recommended output file:

```text
planner_evaluation.jsonl
```

Example:

```json
{
  "query_id": "Q-000001",
  "selected_strategy": "table_aware",
  "oracle_strategy": "table_aware",
  "strategy_correct": true,
  "planner_type": "rule_based"
}
```

## Oracle Validation Framework

The `planner_oracle_strategy` label is a research annotation, not absolute truth. It should be validated after retrieval experiments.

Required query metadata field:

| Field | Type | Description |
| --- | --- | --- |
| oracle_strategy_source | string | Provenance of `planner_oracle_strategy` |

Allowed values:

- manual_annotation
- heuristic_assignment
- empirical_validation

Validation protocol:

1. Generate initial `planner_oracle_strategy` labels using manual annotation, heuristic rules, or both.
2. Run Fixed BM25, Fixed Dense Retrieval, Fixed Hybrid Retrieval, Rule-Based Retrieval Planner, and LLM-Based Retrieval Planner.
3. Review a subset of queries after retrieval experiments.
4. Identify cases where empirical retrieval results consistently contradict the oracle strategy label.
5. Document the contradiction in experiment notes.
6. If the oracle label is updated, set `oracle_strategy_source` to `empirical_validation`.
7. Preserve the original dataset version so planner results remain auditable.

Purpose:

This protocol prevents circular evaluation of planner performance by requiring oracle strategy labels to carry provenance and by allowing correction only through documented validation.

## Answer-Level Evaluation Support

Retrieval metrics alone are insufficient for final thesis conclusions because the project evaluates Agentic RAG, not only retrieval.

Required answer-level metrics:

- Answer Correctness
- Answer Groundedness
- Citation Precision
- Citation Recall

Optional answer-level metrics:

- RAGAS Faithfulness
- RAGAS Context Precision

Dataset fields supporting answer-level evaluation:

- `reference_answer`
- `evidence`
- `expected_citations`
- `acceptable_answer_patterns`
- `disallowed_claims`
- `evaluation_notes`

Answer-level evaluation should verify that generated answers are factually correct, grounded in retrieved evidence, and supported by precise citations.

## Ground Truth Schema

Ground truth must be explicit enough for automatic evaluation.

```json
{
  "query_id": "Q-000001",
  "reference_answer": "Employees in the EU region must submit receipts for lodging expenses above 75 EUR.",
  "evidence": [
    {
      "document_id": "TRV-POL-004",
      "chunk_id": "TRV-POL-004-C003",
      "section_path": ["5. Receipt Requirements"],
      "quote": "Receipts are required for lodging expenses above 75 EUR in the EU region.",
      "char_start": 1204,
      "char_end": 1278,
      "supports": "receipt threshold"
    }
  ],
  "expected_citations": ["TRV-POL-004-C003"],
  "acceptable_answer_patterns": [
    "above 75 EUR",
    "75 EUR"
  ],
  "disallowed_claims": [
    "receipt is always optional",
    "threshold is 100 EUR"
  ],
  "evaluation_notes": "Answer must include the numeric threshold, currency, region, and receipt requirement."
}
```

### Evidence Object Fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| document_id | string | yes | Source document |
| chunk_id | string | yes | Source chunk |
| section_path | array[string] | yes | Evidence location |
| quote | string | yes | Exact evidence span |
| char_start | integer | yes | Character start offset |
| char_end | integer | yes | Character end offset |
| supports | string | yes | Claim supported by this evidence |

## Annotation Guidelines

### Document Annotation

- Assign exactly one primary category to each document.
- Preserve all generated IDs exactly as written.
- Mark document status as active, deprecated, draft, or archived.
- Use metadata fields consistently across the same category.
- Ensure title, body, metadata, version, dates, and sensitivity are always present.

### Chunk Annotation

- Chunk boundaries should preserve semantic units.
- Policy and design documents should be chunked by section or subsection.
- API documentation should keep endpoint blocks intact.
- Tables should be stored with enough surrounding context to interpret row and column meaning.
- Meeting note chunks should preserve decisions, action item rows, and blockers.
- Chunk IDs should follow `{document_id}-C{three_digit_number}`.

### Query Annotation

- Each query must have one primary category and one primary query type.
- Each query must include `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type`, and `required_document_count`.
- Each query may include one or more `retrieval_difficulty_factors`.
- Difficulty must reflect retrieval complexity, not answer length.
- Set `answerable` to false only when no document supports the answer.
- For ambiguous queries, ground truth should specify the intended interpretation in `evaluation_notes`.
- For multi-hop queries, include all required evidence chunks.

### Answer Annotation

- Reference answers should be concise but complete.
- Numeric answers must include units and currencies.
- Policy answers must include applicable conditions and exceptions.
- API answers must include method, endpoint, parameter, or error code where relevant.
- Meeting answers must include date or project code when needed for disambiguation.
- Unanswerable answers should state that the dataset does not contain enough information.

### Citation Annotation

- Every factual claim in the reference answer must map to at least one evidence object.
- Prefer the smallest sufficient evidence span.
- If a claim requires two documents, cite both chunks.
- Do not cite whole documents when chunk-level evidence is available.
- Deprecated documents should not be cited unless the query explicitly asks about deprecated content.

## Recommended Dataset Scale

| Artifact | Recommended Count |
| --- | --- |
| Documents | 120-235 |
| Chunks | 800-2,000 |
| Queries | 300-600 |
| Ground-truth answers | 300-600 |
| Multi-hop queries | 50-100 |
| Unanswerable queries | 20-60 |

## Generator Determinism

The Python generator should accept:

- `--seed`
- `--document-count`
- `--query-count`
- `--output-dir`
- `--include-adversarial`

Generation must be reproducible for the same seed. IDs, dates, generated facts, query assignments, and ground truth evidence should remain stable across runs with the same configuration.

## Validation Rules

The generator should validate:

- Every document has all required global metadata fields.
- Every category-specific document has all required fields.
- Every chunk references an existing document.
- Every answerable query has at least one required document and chunk.
- Every query uses an allowed `planner_oracle_strategy` value.
- Every query uses an allowed `oracle_strategy_source` value.
- Every query uses an allowed `reasoning_type` value.
- Every query uses an allowed `required_document_count` value.
- Every `retrieval_difficulty_factors` item uses an allowed value.
- Every ground-truth evidence object references an existing chunk.
- Every expected citation references an evidence chunk.
- No unanswerable query has required chunks.
- No document contains real credentials, personal phone numbers, or private email addresses.
- Date fields are valid ISO 8601 strings.
- Enum fields use only allowed values.

## Notes for Retrieval Evaluation

SEKD should intentionally contain:

- Similar document titles across regions
- Deprecated and active versions
- Numeric values with different units
- Tables requiring row and column interpretation
- Multi-document references
- Informal meeting-note language
- Exact IDs requiring keyword retrieval
- Paraphrased queries requiring dense retrieval
- Conflicting-looking rules resolved by metadata or exceptions

These properties make the dataset suitable for evaluating whether retrieval strategy planning improves enterprise RAG performance.

## Alignment With Research Questions

SEKD is aligned with the project research questions as follows.

### RQ1. Retrieval Effectiveness

Supported by:

- `planner_oracle_strategy`
- `expected_strategy`
- `required_document_ids`
- `required_chunk_ids`
- `retrieval_difficulty_factors`
- `required_document_count`

These fields enable retrieval metrics such as Recall@K, Precision@K, MRR, nDCG@K, and Hit Rate@K across fixed retrieval and adaptive retrieval methods.

### RQ2. Answer Quality

Supported by:

- `reference_answer`
- `evidence`
- `expected_citations`
- `acceptable_answer_patterns`
- `disallowed_claims`
- `evaluation_notes`

These fields enable answer correctness, completeness, faithfulness, refusal accuracy, and citation quality evaluation.

### RQ3. Document-Type Sensitivity

Supported by:

- `category`
- category-specific metadata schemas
- `query_type`
- `difficulty`
- `retrieval_difficulty_factors`

These fields enable analysis of retrieval and answer quality by document category, such as HR Policies, Travel Policies, Security Policies, API Documentation, System Design Documents, and Project Meeting Notes.

### RQ4. Agentic Planning Effectiveness

Supported by:

- `planner_oracle_strategy`
- `reasoning_type`
- `retrieval_difficulty_factors`
- Planner Evaluation Schema fields

These fields enable measurement of whether the planner selects an appropriate retrieval strategy for each query and whether strategy choice varies correctly by reasoning category and difficulty factor.

### RQ5. Cost and Latency Tradeoffs

Supported by:

- `difficulty`
- `reasoning_type`
- `required_document_count`
- `retrieval_difficulty_factors`
- planner evaluation outputs

These fields allow cost and latency to be analyzed by query complexity, retrieval strategy, planner type, and multi-document dependency.

### RQ6. Rule-Based Planner vs LLM-Based Planner

Supported by:

- `planner_oracle_strategy`
- `selected_strategy`
- `oracle_strategy`
- `strategy_correct`
- `planner_type`
- `reasoning_type`
- `retrieval_difficulty_factors`

These fields allow direct comparison of Rule-Based Planner and LLM-Based Planner performance using Strategy Selection Accuracy, Planner Precision, Planner Recall, and downstream retrieval or answer quality metrics.

### RQ7. Cost Justification

Supported by:

- `difficulty`
- `reasoning_type`
- `required_document_count`
- `retrieval_difficulty_factors`
- planner evaluation outputs
- experiment run manifests
- latency and cost logs

These fields and outputs allow analysis of whether gains from adaptive retrieval planning justify additional latency and inference cost using Mean Latency, Median Latency, Retrieval Cost, Planner Cost, and Total Pipeline Cost.

## Research Contribution Clarification

SEKD supports the expected thesis contributions as follows:

| Contribution | Dataset Support |
| --- | --- |
| Contribution 1: SEKD (Synthetic Enterprise Knowledge Dataset) | The dataset itself, including documents, chunks, queries, ground truth, metadata, and oracle strategy labels |
| Contribution 2: Enterprise Retrieval Benchmark | Query labels, ground truth evidence, oracle strategies, and difficulty factors for comparing fixed and adaptive retrieval |
| Contribution 3: Rule-Based Retrieval Planner | `planner_oracle_strategy`, `oracle_strategy_source`, and retrieval difficulty fields used to design and evaluate deterministic strategy rules |
| Contribution 4: LLM-Based Retrieval Planner | Query text, metadata, reasoning labels, and oracle strategies used to evaluate LLM strategy selection |
| Contribution 5: Empirical Analysis of Retrieval Strategy Selection | Grouping fields such as `category`, `query_type`, `reasoning_type`, `retrieval_difficulty_factors`, and `required_document_count` |

## Final Thesis Mapping

| Research Question | Dataset Fields | Experiments | Metrics | Expected Thesis Chapter |
| --- | --- | --- | --- | --- |
| RQ1. Retrieval Effectiveness | `required_document_ids`, `required_chunk_ids`, `planner_oracle_strategy` | Fixed BM25, Fixed Dense, Fixed Hybrid, planners | Recall@K, Precision@K, MRR, nDCG | Chapter 5 |
| RQ2. Answer Quality | `reference_answer`, `evidence`, `expected_citations`, `disallowed_claims` | End-to-end RAG generation | Answer Correctness, Answer Groundedness, Citation Precision, Citation Recall | Chapter 5 |
| RQ3. Document-Type Sensitivity | `category`, metadata schemas, `query_type` | Grouped category analysis | Retrieval and answer metrics by category | Chapter 5 and Chapter 6 |
| RQ4. Agentic Planning Effectiveness | `planner_oracle_strategy`, `oracle_strategy_source`, `reasoning_type` | Planner ablation study | Strategy Selection Accuracy | Chapter 4 and Chapter 5 |
| RQ5. Cost and Latency Tradeoffs | `difficulty`, `required_document_count`, run manifests | Cost-latency comparison | Mean Latency, Median Latency, Total Pipeline Cost | Chapter 5 and Chapter 6 |
| RQ6. Rule-Based Planner vs LLM-Based Planner | planner evaluation outputs, `planner_oracle_strategy`, `oracle_strategy_source` | Rule-based vs LLM-based planner comparison | Planner Precision, Planner Recall, Strategy Selection Accuracy | Chapter 5 |
| RQ7. Cost Justification | run manifests, latency logs, cost logs, grouped query fields | Quality-cost tradeoff analysis | Recall@K, Planner Accuracy, Latency, Cost | Chapter 6 |
