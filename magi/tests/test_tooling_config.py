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
