from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from magi.core.clients import LLMClient
from magi.eval.scenario_harness import (
    ScenarioDataset,
    load_scenario_dataset,
    run_scenario_suite,
)


class _SequenceClient(LLMClient):
    model = "gpt-4o-mini-2024-07-18"

    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self._payloads = list(payloads)

    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        assert messages
        assert response_format is not None
        if not self._payloads:
            raise AssertionError("unexpected extra client call")
        payload = self._payloads.pop(0)
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


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


def test_live_scenarios_suite_passes_with_fake_live_client() -> None:
    dataset = ScenarioDataset.model_validate(
        {
            "cases": [
                {
                    "id": "live_grounded_summary",
                    "query": "Summarize MAGI in one sentence.",
                    "expected_verdict": "approve",
                    "expected_residual_risk": "low",
                    "evidence": [
                        {
                            "source": "README",
                            "text": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
                        }
                    ],
                    "checks": {
                        "required_terms_any": ["magi", "reasoning engine"],
                        "min_citations": 1,
                    },
                }
            ]
        }
    )
    client = _SequenceClient(
        [
            {
                "analysis": "The evidence defines MAGI directly.",
                "answer_outline": ["Summarize MAGI from [1]."],
                "confidence": 0.8,
                "evidence_quotes": ['[1] "MAGI is a multi persona reasoning engine."'],
                "stance": "approve",
                "actions": ["Answer with citation [1]."],
            },
            {
                "plan": "Answer in one sentence with citation [1].",
                "communication_plan": ["Keep it concise.", "Cite [1]."],
                "cost_estimate": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Keep it concise.", "Cite [1]."],
            },
            {
                "risks": ["Low risk of over-claiming beyond [1]."],
                "mitigations": ["Stay within [1]."],
                "residual_risk": "low",
                "confidence": 0.8,
                "stance": "approve",
                "actions": ["Stay within [1]."],
                "outstanding_questions": [],
            },
            {
                "verdict": "approve",
                "justification": "The evidence in [1] is enough for a one-sentence summary.",
                "confidence": 0.8,
                "final_answer": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base [1].",
                "next_steps": ["Answer directly."],
                "consensus_points": ["[1] supports the summary."],
                "disagreements": [],
                "residual_risk": "low",
                "risks": ["Low risk of over-claiming beyond [1]."],
                "mitigations": ["Stay within [1]."],
            },
            {
                "final_answer": "MAGI is a multi persona reasoning engine for assessing user requests against an evidence base [1].",
                "justification": "The evidence in [1] is enough for a one-sentence summary.",
                "next_steps": ["Answer directly."],
            },
        ]
    )

    report = run_scenario_suite(
        dataset,
        force_stub=False,
        client=client,
        requested_mode="live",
    )

    assert report.summary.effective_mode == "live"
    assert report.summary.overall_score == 1.0
    assert report.summary.verdict_accuracy == 1.0
    assert report.cases[0].citation_count >= 1
