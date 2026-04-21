from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def test_run_scenarios_cli_fails_when_threshold_is_not_met(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_scenarios.py"
    dataset = tmp_path / "ranked_cases.yaml"
    report_out = tmp_path / "scenario_report.json"
    payload = {
        "cases": [
            {
                "id": "ranked_source",
                "query": "Explain rollout status and monitoring cadence.",
                "expected_verdict": "approve",
                "evidence": [
                    {
                        "source": "weekly_report",
                        "text": "The rollout status and monitoring cadence are tracked in a weekly report.",
                    },
                    {
                        "source": "rollout_notes",
                        "text": "The rollout status is green and monitored by ops.",
                    },
                ],
                "checks": {"required_sources_any": ["rollout_notes"]},
            }
        ]
    }
    dataset.write_text(yaml.safe_dump(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--mode",
            "stub",
            "--report-out",
            str(report_out),
            "--min-retrieval-top-source-accuracy",
            "1.0",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 1
    assert "threshold_failed" in completed.stderr
    assert "retrieval_top_source_accuracy" in completed.stderr
    assert report_out.exists()
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["summary"]["retrieval_top_source_accuracy"] == 0.0


def test_run_scenarios_cli_can_gate_answer_support_metrics(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_scenarios.py"
    dataset = tmp_path / "unsupported_cases.yaml"
    payload = {
        "cases": [
            {
                "id": "evidence_gap_revision",
                "query": "What is MAGI's guaranteed p95 latency SLA?",
                "expected_verdict": "revise",
            }
        ]
    }
    dataset.write_text(yaml.safe_dump(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--mode",
            "stub",
            "--min-average-answer-support-score",
            "0.1",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 1
    assert "threshold_failed" in completed.stderr
    assert "average_answer_support_score" in completed.stderr
