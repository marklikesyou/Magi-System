from __future__ import annotations

import zipfile
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_built_wheel_contains_runtime_assets_without_tests(tmp_path: Path) -> None:
    source_tree = tmp_path / "source"
    shutil.copytree(
        ROOT,
        source_tree,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "__pycache__",
            "*.pyc",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            "build",
            "dist",
            "*.egg-info",
            "magi/artifacts",
            "magi/storage",
            "magi/eval/artifacts",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    completed = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=source_tree,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    assert "magi/profiles/security-review.yaml" in names
    assert "magi/eval/production_scenarios.yaml" in names
    assert "magi/eval/retrieval_corpus/pilot_brief.txt" in names
    assert "magi/infra/docker-compose.yml" in names
    assert not any(name.startswith("magi/tests/") for name in names)
