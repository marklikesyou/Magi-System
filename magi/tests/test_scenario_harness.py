from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from magi.eval.scenario_harness import load_scenario_dataset, run_scenario_suite


def test_live_scenarios_suite_passes_in_stub_mode() -> None:
    dataset_path = Path(__file__).resolve().parents[1] / "eval" / "live_scenarios.yaml"
    dataset = load_scenario_dataset(dataset_path)

    report = run_scenario_suite(dataset, force_stub=True, requested_mode="stub")

    assert report.summary.effective_mode == "stub"
    assert report.summary.verdict_accuracy == 1.0
    assert report.summary.overall_score == 1.0
    assert report.summary.requirement_pass_rate == 1.0


def test_run_scenarios_cli_writes_json_report(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "eval" / "run_scenarios.py"
    dataset = Path(__file__).resolve().parents[1] / "eval" / "live_scenarios.yaml"
    report_out = tmp_path / "scenario_report.json"

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
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(report_out.read_text(encoding="utf-8"))
    assert payload["summary"]["overall_score"] == 1.0
    assert payload["summary"]["effective_mode"] == "stub"
    assert "report_saved" in completed.stdout
