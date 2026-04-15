from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]


def parse_accuracy(output: str) -> float:
    for line in output.splitlines():
        if line.lower().startswith("accuracy"):
            return float(line.split()[-1].strip("%")) / 100.0
    raise AssertionError("accuracy line not found in output")


def build_sample_dataset(path: Path) -> None:
    payload = {
        "cases": [
            {
                "id": "unit_case",
                "query": "Should we publish release notes?",
                "constraints": "",
                "expected_verdict": "approve",
                "fused": {
                    "verdict": "approve",
                    "justification": "Alignment across personas",
                    "confidence": 0.6,
                    "residual_risk": "low",
                },
                "personas": {
                    "melchior": {
                        "text": "[APPROVE] evidence supports release",
                        "confidence": 0.7,
                    },
                    "balthasar": {
                        "text": "[APPROVE] stakeholders ready",
                        "confidence": 0.7,
                    },
                    "casper": {"text": "[APPROVE] risks mitigated", "confidence": 0.6},
                },
            }
        ]
    }
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_decision_bench_accuracy(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "eval" / "run_bench.py"
    dataset = tmp_path / "cases.yaml"
    features_out = tmp_path / "features.jsonl"
    build_sample_dataset(dataset)
    bench = subprocess.run(
        [
            sys.executable,
            str(script),
            "--cases",
            str(dataset),
            "--features-out",
            str(features_out),
        ],
        text=True,
        capture_output=True,
        check=False,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert bench.returncode == 0, f"run_bench failed: {bench.stderr}"
    accuracy = parse_accuracy(bench.stdout)
    assert accuracy >= 0.85
    assert features_out.exists()
