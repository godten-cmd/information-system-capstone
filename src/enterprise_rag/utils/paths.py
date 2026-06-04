"""Path helpers for repository-relative files."""

from pathlib import Path


def project_root() -> Path:
    """Return the repository root inferred from this source file."""
    return Path(__file__).resolve().parents[3]


def resolve_from_root(*parts: str) -> Path:
    """Resolve a path under the repository root."""
    return project_root().joinpath(*parts)

