"""Metadata inheritance logic for the SEKD chunking pipeline."""

from __future__ import annotations

from enterprise_rag.dataset.schemas import EnterpriseDocument

# All 9 required global fields from DATASET_SPEC §Global Metadata Schema
# plus title, version, and source_path for chunk-level traceability.
_INHERITED_KEYS = (
    "title",
    "version",
    "region",
    "status",
    "department",
    "document_owner",
    "confidentiality",
    "authority_level",
    "effective_date",
    "review_cycle",
    "tags",
    "company",
    "source_path",
)


def build_inherited_metadata(doc: EnterpriseDocument) -> dict:
    """Return the base metadata dict to attach to every chunk from *doc*.

    Inherits all 9 required global metadata fields from DATASET_SPEC,
    plus title, version, and source_path.  The result is stored in
    ``Chunk.inherited_metadata`` and may be extended with chunk-specific
    annotations by the caller.
    """
    m = doc.metadata
    return {
        "title": m.title,
        "version": m.version,
        "region": m.region.value,
        "status": m.status.value,
        "department": m.department,
        "document_owner": m.author_role,
        "confidentiality": m.confidentiality_level.value,
        "authority_level": m.authority_level.value,
        "effective_date": str(m.effective_date),
        "review_cycle": m.review_cycle.value,
        "tags": list(m.tags),
        "company": m.company,
        "source_path": m.source_path,
    }


def validate_inherited_metadata(meta: dict) -> list[str]:
    """Return a list of missing required keys in *meta* (empty list = valid)."""
    return [k for k in _INHERITED_KEYS if k not in meta]
