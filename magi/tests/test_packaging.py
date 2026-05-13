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
    assert "magi/README.md" not in completed.stderr
    wheels = sorted(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    wheel_path = wheels[0]

    with zipfile.ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())
        metadata_path = next(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        metadata = wheel.read(metadata_path).decode("utf-8")

    assert "magi/profiles/security-review.yaml" in names
    assert "magi/eval/ACCEPTANCE.md" in names
    assert "magi/eval/acceptance_audit.py" in names
    assert "magi/eval/build_adversarial_semantic_suite.py" in names
    assert "magi/eval/run_gauntlet.py" in names
    assert "magi/eval/verify_gauntlet_manifest.py" in names
    assert "magi/eval/production_scenarios.yaml" in names
    assert "magi/eval/retrieval_corpus/pilot_brief.txt" in names
    assert "magi/infra/docker-compose.yml" in names
    assert not any(name.startswith("magi/tests/") for name in names)
    assert "MAGI is a command line assistant" in metadata
