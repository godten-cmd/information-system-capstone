"""Prompt templates: zero_shot, structured, few_shot.

Each template receives a query dict and returns a user prompt string.
The system prompt is shared across all three variants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


# ── System prompt (shared) ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a retrieval strategy planner for an enterprise knowledge \
management system (HYTech Solutions SEKD corpus).

Select the most appropriate retrieval strategy for the given query.

Available strategies:
- bm25   Keyword-based sparse retrieval. Best for exact terms, policy/control IDs \
(e.g. HY-AC-01), version numbers, dates, names, and temporal event ordering.
- dense  Semantic vector retrieval. Best for exception clause interpretation, \
conceptual questions, and paraphrased policy lookups where exact keywords may differ.
- hybrid Combined BM25 + Dense retrieval. Best for multi-hop evidence across documents, \
table value lookups, comparison, aggregation, and general enterprise queries.

Respond with ONLY a valid JSON object — no markdown, no extra text:
{"selected_strategy": "bm25|dense|hybrid", "confidence": 0.0-1.0, "reasoning": "one sentence"}"""


def get_system_prompt() -> str:
    return _SYSTEM_PROMPT


# ── Prompt A: Zero-Shot ───────────────────────────────────────────────────────

def _zero_shot(query: dict[str, Any]) -> str:
    qid = query.get("query_id", "")
    text = query.get("query", "") or query.get("query_text", "")
    return (
        f"Query: {text}\n\n"
        "Select the best retrieval strategy for this enterprise knowledge query.\n"
        f"# query_id: {qid}"
    )


# ── Prompt B: Structured ──────────────────────────────────────────────────────

def _structured(query: dict[str, Any]) -> str:
    qid = query.get("query_id", "")
    text = query.get("query", "") or query.get("query_text", "")
    rt = query.get("reasoning_type", "unknown") or "unknown"
    factors = query.get("retrieval_difficulty_factors", []) or []
    factors_str = ", ".join(factors) if factors else "none"
    category = query.get("category", "unknown") or "unknown"

    return (
        f"Query: {text}\n\n"
        "Query metadata:\n"
        f"- Reasoning type: {rt}\n"
        f"- Retrieval difficulty factors: {factors_str}\n"
        f"- Document category: {category}\n\n"
        "Strategy selection guidance:\n"
        "- temporal → bm25  (date tokens and event sequences match better with keywords)\n"
        "- exception → dense  (exception clauses use paraphrased language, not policy headings)\n"
        "- multi_hop / multi_document_dependency → hybrid  (lexical IDs + semantic content)\n"
        "- table_dependency → hybrid  (exact numeric values + surrounding context)\n"
        "- single_hop in System Design or API Documentation → dense  (semantic architecture concepts)\n"
        "- default → hybrid  (broadest enterprise coverage)\n\n"
        "Select the best retrieval strategy.\n"
        f"# query_id: {qid}"
    )


# ── Prompt C: Few-Shot ────────────────────────────────────────────────────────

_FEW_SHOT_EXAMPLES = """Examples of correct retrieval strategy selection for enterprise queries:

Example 1:
Query: "What was the first decision made in the initial PRJ-HYD-ALPHA meeting?"
Reasoning: temporal | Factors: temporal_reasoning, multi_document_dependency | Category: Project Meeting Notes
→ {"selected_strategy": "bm25", "confidence": 0.85, "reasoning": "Temporal query with first/initial anchors; BM25 matches date tokens and project names more reliably than semantic search"}

Example 2:
Query: "Under what conditions can an employee be exempt from the remote work attendance requirement?"
Reasoning: exception | Factors: exception_handling | Category: HR Policies
→ {"selected_strategy": "dense", "confidence": 0.90, "reasoning": "Exception clause interpretation: exceptions are phrased differently from policy headings so semantic matching outperforms keyword search"}

Example 3:
Query: "Which API endpoints in HYDeploy v2 depend on the HYID authentication service?"
Reasoning: multi_hop | Factors: multi_document_dependency | Category: API Documentation
→ {"selected_strategy": "hybrid", "confidence": 0.90, "reasoning": "Multi-document dependency: needs exact endpoint path matching (BM25) plus semantic understanding of service dependencies (Dense)"}

Example 4:
Query: "What is the maximum lodging reimbursement limit for US business travel?"
Reasoning: single_hop | Factors: table_dependency | Category: Travel Policies
→ {"selected_strategy": "hybrid", "confidence": 0.85, "reasoning": "Table value lookup: hybrid provides BM25 for exact amounts and Dense for contextual table interpretation"}

Example 5:
Query: "How does the HYDeploy pipeline handle deployment rollbacks?"
Reasoning: single_hop | Factors: none | Category: System Design Documents
→ {"selected_strategy": "dense", "confidence": 0.80, "reasoning": "Technical single-hop in System Design; semantic search outperforms keyword matching for architectural and procedural concepts"}

Example 6:
Query: "Which control IDs are associated with the HY-AC access control family?"
Reasoning: single_hop | Factors: metadata_dependency | Category: Security Policies
→ {"selected_strategy": "hybrid", "confidence": 0.80, "reasoning": "Control IDs (e.g. HY-AC-01) benefit from BM25 exact match while hybrid adds Dense context for policy interpretation"}

Now for this query:"""


def _few_shot(query: dict[str, Any]) -> str:
    qid = query.get("query_id", "")
    text = query.get("query", "") or query.get("query_text", "")
    rt = query.get("reasoning_type", "unknown") or "unknown"
    factors = query.get("retrieval_difficulty_factors", []) or []
    factors_str = ", ".join(factors) if factors else "none"
    category = query.get("category", "unknown") or "unknown"

    return (
        f"{_FEW_SHOT_EXAMPLES}\n"
        f"Query: {text}\n"
        f"Reasoning: {rt} | Factors: {factors_str} | Category: {category}\n\n"
        "Select the best retrieval strategy.\n"
        f"# query_id: {qid}"
    )


# ── Registry ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptTemplate:
    version: str
    description: str
    builder: Callable[[dict[str, Any]], str]

    def build(self, query: dict[str, Any]) -> str:
        return self.builder(query)


PROMPT_VERSIONS: dict[str, PromptTemplate] = {
    "zero_shot": PromptTemplate(
        version="zero_shot",
        description="Minimal prompt with query text only; no query metadata provided.",
        builder=_zero_shot,
    ),
    "structured": PromptTemplate(
        version="structured",
        description="Prompt includes reasoning type, difficulty factors, category, and selection guidance.",
        builder=_structured,
    ),
    "few_shot": PromptTemplate(
        version="few_shot",
        description="6 labeled examples covering all reasoning types, plus structured metadata.",
        builder=_few_shot,
    ),
}


def build_prompt(query: dict[str, Any], version: str) -> str:
    """Build a user prompt for the given query and version."""
    tmpl = PROMPT_VERSIONS.get(version)
    if tmpl is None:
        raise ValueError(f"Unknown prompt version '{version}'. Choose from: {list(PROMPT_VERSIONS)}")
    return tmpl.build(query)
