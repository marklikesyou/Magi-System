from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_python_targets_are_aligned() -> None:
    pyproject = _read("pyproject.toml")
    mypy_ini = _read("mypy.ini")
    ci = _read(".github/workflows/ci.yml")

    assert 'requires-python = ">=3.10"' in pyproject
    assert 'target-version = "py310"' in pyproject
    assert "python_version = 3.10" in mypy_ini
    assert 'python-version: ["3.10", "3.13"]' in ci


def test_container_and_ci_include_production_guards() -> None:
    dockerfile = _read("Dockerfile")
    ci = _read(".github/workflows/ci.yml")

    assert "USER magi" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "magi/eval/production_scenarios.yaml" in ci
    assert "--max-p95-latency-ms" in ci
    assert "--max-total-cost-usd" in ci


def test_docker_context_excludes_runtime_data_and_secrets() -> None:
    dockerignore = _read(".dockerignore")

    for pattern in (
        ".env",
        ".env.*",
        ".venv/",
        "__pycache__/",
        "magi/storage/",
        "magi/artifacts/",
        "magi/eval/artifacts/",
        "datasets/",
        "build/",
        "dist/",
    ):
        assert pattern in dockerignore


def test_issue_files_are_trackable() -> None:
    gitignore = _read(".gitignore")

    assert "issues/issue-*.md" not in gitignore
