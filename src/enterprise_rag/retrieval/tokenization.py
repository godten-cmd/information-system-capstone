"""Enterprise-aware tokenizer for BM25 sparse retrieval.

Preserves structured identifiers (document IDs, control IDs, API paths, project codes)
that are critical for exact-match retrieval in the SEKD corpus.
"""

from __future__ import annotations

import re

# Patterns to extract and preserve as atomic tokens before general word splitting
_ID_PATTERNS = [
    re.compile(r"\b(HY-[A-Z]+-\d+)\b"),               # control IDs: HY-AC-01, HY-DP-05
    re.compile(r"\b(HR-POL-\d+|TRAVEL-POL-\d+|SEC-POL-\d+|API-DOC-\d+|SYS-DES-\d+|MTG-\d+)\b"),  # doc IDs
    re.compile(r"\b(PRJ-[\w-]+)\b"),                    # project codes: PRJ-HYD-ALPHA
    re.compile(r"\b(HYID|HYDeploy|HYMonitor|HYDataLake)\b"),  # service names
    re.compile(r"\b([A-Z]+-\d{3})\b"),                 # generic ID pattern
    re.compile(r"(/api/v\d+[^\s,;\"'<>]*)"),            # API endpoint paths
    re.compile(r"\b(v\d+(?:\.\d+)*)\b"),               # version strings: v1, v2.1
    re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"),            # ISO dates: 2023-01-15
    re.compile(r"\b(ADR-[A-Z]+-\d+)\b"),               # ADR IDs
]

# English stop words to remove (minimal set — keep domain terms)
_STOP_WORDS = frozenset(
    [
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "can", "could", "this", "that",
        "these", "those", "it", "its", "as", "if", "not", "no", "nor",
        "all", "any", "each", "few", "more", "most", "other", "such",
        "than", "too", "very", "so", "while", "when", "where", "which",
        "who", "whom", "what", "how", "then", "there", "here", "their",
        "they", "we", "our", "us", "you", "your", "he", "she", "him", "her",
        "his", "hers", "i", "me", "my", "mine",
    ]
)

_WORD_SPLIT_RE = re.compile(r"[^a-z0-9_/-]+")


def tokenize(text: str, keep_stop_words: bool = False) -> list[str]:
    """Tokenize text for BM25 indexing.

    Strategy:
    1. Extract structured identifiers as atomic tokens (IDs, paths, dates).
    2. Lowercase remaining text and split on non-alphanumeric characters.
    3. Optionally remove stop words.
    4. Filter tokens shorter than 2 characters.

    Args:
        text: Input text to tokenize.
        keep_stop_words: If True, retain stop words (useful for query time).

    Returns:
        List of tokens.
    """
    tokens: list[str] = []
    placeholder_map: dict[str, str] = {}
    working = text

    # Step 1: extract structured identifiers as placeholders
    # Use lowercase placeholder keys so they survive working.lower() below
    for i, pattern in enumerate(_ID_PATTERNS):
        for match in pattern.finditer(working):
            tok = match.group(1).lower()
            placeholder = f"__id{i}_{len(placeholder_map)}__"
            placeholder_map[placeholder] = tok
            working = working.replace(match.group(0), f" {placeholder} ", 1)

    # Step 2: lowercase and split
    working_lower = working.lower()
    raw_tokens = _WORD_SPLIT_RE.split(working_lower)

    for t in raw_tokens:
        # Check for placeholder before stripping (strip removes the __ delimiters)
        if t.startswith("__id") and t.endswith("__"):
            resolved = placeholder_map.get(t)
            if resolved:
                tokens.append(resolved)
        else:
            t = t.strip("_-")
            if len(t) >= 2:
                if keep_stop_words or t not in _STOP_WORDS:
                    tokens.append(t)

    return tokens


def tokenize_query(text: str) -> list[str]:
    """Tokenize a query string. Retains stop words to preserve question phrasing."""
    return tokenize(text, keep_stop_words=True)


def tokenize_document(text: str) -> list[str]:
    """Tokenize document/chunk text. Removes stop words to reduce index noise."""
    return tokenize(text, keep_stop_words=False)
