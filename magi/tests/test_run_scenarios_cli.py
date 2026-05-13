from __future__ import annotations

from argparse import Namespace
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml  # type: ignore[import-untyped]

from magi.eval.run_scenarios import _threshold_failures


def _threshold_args(**overrides: object) -> Namespace:
    defaults: dict[str, object] = {
        "min_overall_score": None,
        "min_verdict_accuracy": None,
        "min_requirement_pass_rate": None,
        "min_retrieval_hit_rate": None,
        "min_retrieval_top_source_accuracy": None,
        "min_retrieval_source_recall": None,
        "min_average_citation_hit_rate": None,
        "min_average_answer_support_score": None,
        "min_supported_answer_rate": None,
        "min_cached_replay_hit_rate": None,
        "max_p50_latency_ms": None,
        "max_p95_latency_ms": None,
        "max_cached_p95_latency_ms": None,
        "max_max_latency_ms": None,
        "max_average_cost_usd": None,
        "max_total_cost_usd": None,
        "max_live_fallbacks": None,
        "allow_live_fallbacks": False,
        "max_uncited_approvals": 0,
        "max_empty_final_answers": 0,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


def test_threshold_failures_gate_uncited_approvals_and_empty_answers() -> None:
    report = SimpleNamespace(
        summary=SimpleNamespace(
            effective_mode="stub",
            overall_score=1.0,
            verdict_accuracy=1.0,
            requirement_pass_rate=1.0,
            retrieval_hit_rate=1.0,
            retrieval_top_source_accuracy=1.0,
            retrieval_source_recall=1.0,
            average_citation_hit_rate=1.0,
            average_answer_support_score=1.0,
            supported_answer_rate=1.0,
            cached_replay_hit_rate=1.0,
            latency_p50_ms=1.0,
            latency_p95_ms=1.0,
            cached_latency_p95_ms=1.0,
            latency_max_ms=1.0,
            average_estimated_cost_usd=0.0,
            total_estimated_cost_usd=0.0,
            live_fallback_count=0,
            uncited_approval_count=1,
            empty_final_answer_count=1,
        )
    )

    failures = _threshold_failures(report, _threshold_args())

    assert ("uncited_approval_count", 1.0, 0.0, "maximum") in failures
    assert ("empty_final_answer_count", 1.0, 0.0, "maximum") in failures


def test_threshold_failures_gate_cached_replay_metrics() -> None:
    report = SimpleNamespace(
        summary=SimpleNamespace(
            effective_mode="stub",
            overall_score=1.0,
            verdict_accuracy=1.0,
            requirement_pass_rate=1.0,
            retrieval_hit_rate=1.0,
            retrieval_top_source_accuracy=1.0,
            retrieval_source_recall=1.0,
            average_citation_hit_rate=1.0,
            average_answer_support_score=1.0,
            supported_answer_rate=1.0,
            cached_replay_hit_rate=0.5,
            latency_p50_ms=1.0,
            latency_p95_ms=1.0,
            cached_latency_p95_ms=300.0,
            latency_max_ms=1.0,
            average_estimated_cost_usd=0.0,
            total_estimated_cost_usd=0.0,
            live_fallback_count=0,
            uncited_approval_count=0,
            empty_final_answer_count=0,
        )
    )

    failures = _threshold_failures(
        report,
        _threshold_args(
            min_cached_replay_hit_rate=1.0,
            max_cached_p95_latency_ms=250.0,
        ),
    )

    assert ("cached_replay_hit_rate", 0.5, 1.0, "minimum") in failures
    assert ("cached_latency_p95_ms", 300.0, 250.0, "maximum") in failures


def test_threshold_failures_gate_requested_live_effective_mode() -> None:
    report = SimpleNamespace(
        summary=SimpleNamespace(
            effective_mode="stub",
            overall_score=1.0,
            verdict_accuracy=1.0,
            requirement_pass_rate=1.0,
            retrieval_hit_rate=1.0,
            retrieval_top_source_accuracy=1.0,
            retrieval_source_recall=1.0,
            average_citation_hit_rate=1.0,
            average_answer_support_score=1.0,
            supported_answer_rate=1.0,
            cached_replay_hit_rate=1.0,
            latency_p50_ms=1.0,
            latency_p95_ms=1.0,
            cached_latency_p95_ms=1.0,
            latency_max_ms=1.0,
            average_estimated_cost_usd=0.0,
            total_estimated_cost_usd=0.0,
            live_fallback_count=0,
            uncited_approval_count=0,
            empty_final_answer_count=0,
        )
    )

    failures = _threshold_failures(report, _threshold_args(mode="live"))

    assert ("effective_mode_live", 0.0, 1.0, "minimum") in failures


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
            "--file",
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


def test_run_scenarios_cli_can_gate_latency_metrics(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_scenarios.py"
    dataset = tmp_path / "latency_cases.yaml"
    payload = {
        "cases": [
            {
                "id": "latency_gate",
                "query": "Summarize MAGI in one sentence.",
                "expected_verdict": "approve",
                "evidence": [
                    {
                        "source": "README",
                        "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                    }
                ],
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
            "--max-p95-latency-ms",
            "0",
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 1
    assert "threshold_failed" in completed.stderr
    assert "latency_p95_ms" in completed.stderr
    assert "maximum=0.0000" in completed.stderr
