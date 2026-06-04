"""Smoke tests for Phase 1 repository setup."""

from __future__ import annotations

from pathlib import Path


def test_required_top_level_files_exist() -> None:
    """Verify Phase 1 top-level files exist."""
    root = Path(__file__).resolve().parents[1]
    required_files = [
        "PROJECT_SPEC.md",
        "DATASET_SPEC.md",
        "IMPLEMENTATION_PLAN.md",
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        ".env.example",
        ".gitignore",
    ]
    missing = [path for path in required_files if not root.joinpath(path).is_file()]
    assert not missing


def test_required_package_directories_exist() -> None:
    """Verify Phase 1 package directories exist."""
    root = Path(__file__).resolve().parents[1]
    packages = [
        "dataset",
        "preprocessing",
        "retrieval",
        "planning",
        "evaluation",
        "experiments",
        "utils",
    ]
    for package in packages:
        assert root.joinpath("src", "enterprise_rag", package, "__init__.py").is_file()


def test_required_config_files_exist() -> None:
    """Verify Phase 1 config files exist."""
    root = Path(__file__).resolve().parents[1]
    config_files = [
        "configs/dataset.yaml",
        "configs/experiment.yaml",
        "configs/retrieval/bm25.yaml",
        "configs/retrieval/dense.yaml",
        "configs/retrieval/hybrid.yaml",
        "configs/planners/rule_based.yaml",
        "configs/planners/llm_based.yaml",
    ]
    missing = [path for path in config_files if not root.joinpath(path).is_file()]
    assert not missing

