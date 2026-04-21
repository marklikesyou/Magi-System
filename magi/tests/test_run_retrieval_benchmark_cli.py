from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def test_run_retrieval_benchmark_cli_writes_json_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_retrieval_benchmark.py"
    dataset = Path(__file__).resolve().parents[1] / "eval" / "retrieval_benchmark.yaml"
    report_out = tmp_path / "retrieval_benchmark_report.json"
    env = dict(os.environ)
    env["MAGI_FORCE_HASH_EMBEDDER"] = "1"
    env["DATABASE_URL"] = ""

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--store",
            str(tmp_path / "benchmark_store.json"),
            "--report-out",
            str(report_out),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["summary"]["overall_score"] == 1.0
    assert payload["summary"]["retrieval_hit_rate"] == 1.0
    assert payload["summary"]["retrieval_top_source_accuracy"] == 1.0
    assert "report_saved" in completed.stdout


def test_run_retrieval_benchmark_cli_fails_when_threshold_is_not_met(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_retrieval_benchmark.py"
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "weekly_report.txt").write_text(
        "The rollout status and monitoring cadence are tracked in a weekly report.\n",
        encoding="utf-8",
    )
    (corpus_root / "rollout_notes.txt").write_text(
        "The rollout status is green and monitored by ops.\n",
        encoding="utf-8",
    )
    dataset = tmp_path / "retrieval_cases.yaml"
    report_out = tmp_path / "retrieval_report.json"
    dataset.write_text(
        yaml.safe_dump(
            {
                "corpus": {
                    "root": str(corpus_root),
                    "chunk_size": 5000,
                    "chunk_overlap": 200,
                    "documents": [
                        {"id": "weekly_report", "path": "weekly_report.txt"},
                        {"id": "rollout_notes", "path": "rollout_notes.txt"},
                    ],
                },
                "cases": [
                    {
                        "id": "ranked_source",
                        "query": "Explain rollout status and monitoring cadence.",
                        "expected_sources_any": ["rollout_notes"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ)
    env["MAGI_FORCE_HASH_EMBEDDER"] = "1"
    env["DATABASE_URL"] = ""

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--store",
            str(tmp_path / "benchmark_store.json"),
            "--report-out",
            str(report_out),
            "--min-retrieval-top-source-accuracy",
            "1.0",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )

    assert completed.returncode == 1
    assert "threshold_failed" in completed.stderr
    assert "retrieval_top_source_accuracy" in completed.stderr
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["summary"]["retrieval_top_source_accuracy"] == 0.0
