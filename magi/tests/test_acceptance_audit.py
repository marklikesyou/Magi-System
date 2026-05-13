from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from magi.eval.acceptance_audit import build_acceptance_audit, main
from magi.eval.verify_gauntlet_manifest import EXPECTED_SEMANTIC_CATEGORIES


def _scenario_summary(*, effective_mode: str = "stub") -> dict[str, object]:
    return {
        "overall_score": 1.0,
        "verdict_accuracy": 1.0,
        "requirement_pass_rate": 1.0,
        "requested_mode": effective_mode,
        "effective_mode": effective_mode,
        "total_cases": 4,
        "passed_cases": 4,
        "retrieval_hit_rate": 1.0,
        "retrieval_top_source_accuracy": 1.0,
        "retrieval_source_recall": 1.0,
        "average_citation_hit_rate": 1.0,
        "average_answer_support_score": 0.25,
        "supported_answer_rate": 1.0,
        "latency_p95_ms": 24.0,
        "cached_latency_p95_ms": 1.0,
        "cached_replay_hit_rate": 1.0,
        "live_fallback_count": 0,
        "empty_final_answer_count": 0,
        "uncited_approval_count": 0,
        "total_estimated_cost_usd": 0.0,
    }


def _retrieval_summary() -> dict[str, object]:
    return {
        "total_cases": 3,
        "passed_cases": 3,
        "overall_score": 1.0,
        "retrieval_evaluable_cases": 3,
        "cases_with_retrieval_hits": 3,
        "retrieval_hit_rate": 1.0,
        "retrieval_ranked_cases": 3,
        "retrieval_top_source_accuracy": 1.0,
        "retrieval_mrr": 1.0,
        "retrieval_source_recall": 1.0,
        "ingested_document_count": 4,
        "ingested_chunk_count": 4,
    }


def _semantic_summary() -> dict[str, object]:
    return {
        "total_cases": 1000,
        "passed_cases": 960,
        "pass_rate": 0.96,
        "latency_p95_ms": 900.0,
        "empty_final_answer_count": 0,
        "uncited_approval_count": 0,
        "live_fallback_count": 0,
        "average_answer_quality_score": 0.91,
        "average_grounding_score": 0.90,
        "average_safety_score": 0.98,
        "effective_mode": "stub",
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _semantic_cases() -> list[dict[str, object]]:
    return [
        {
            "id": f"case-{index}",
            "query": f"How should MAGI audit semantic case {index}?",
            "expected_behavior": "Judge semantic behavior from evidence.",
            "tags": [
                EXPECTED_SEMANTIC_CATEGORIES[
                    index % len(EXPECTED_SEMANTIC_CATEGORIES)
                ]
            ],
        }
        for index in range(1000)
    ]


def _write_valid_manifest(tmp_path: Path) -> Path:
    production_report = tmp_path / "production_report.json"
    live_report = tmp_path / "live_report.json"
    retrieval_report = tmp_path / "retrieval_benchmark_report.json"
    semantic_report = tmp_path / "adversarial_semantic_report.json"
    semantic_suite = tmp_path / "adversarial_semantic.yaml"
    manifest = tmp_path / "gauntlet_manifest.json"

    production_summary = _scenario_summary(effective_mode="stub")
    live_summary = _scenario_summary(effective_mode="live")
    retrieval_summary = _retrieval_summary()
    semantic_summary = _semantic_summary()
    semantic_cases = _semantic_cases()
    _write_json(production_report, {"summary": production_summary})
    _write_json(live_report, {"summary": live_summary})
    _write_json(retrieval_report, {"summary": retrieval_summary, "cases": []})
    _write_json(semantic_report, {"summary": semantic_summary, "cases": semantic_cases})
    semantic_suite.write_text(yaml.safe_dump({"cases": semantic_cases}), encoding="utf-8")
    _write_json(
        manifest,
        {
            "metadata": {
                "suite_type": "production_acceptance_gauntlet",
                "status": "passed",
                "adversarial_semantic_cases": 1000,
                "criteria": [
                    {"id": "production_scenarios"},
                    {"id": "retrieval_benchmark"},
                    {"id": "live_scenarios", "provider": "openai_responses_api"},
                    {"id": "semantic_suite"},
                    {"id": "no_uncited_approvals"},
                    {"id": "no_empty_final_answers"},
                    {"id": "stub_latency"},
                    {"id": "cached_replay_latency"},
                ],
            },
            "inputs": {"semantic_cases": str(semantic_suite), "semantic_mode": "stub"},
            "results": [
                {
                    "gate": "production_stub",
                    "status": "passed",
                    "report": str(production_report),
                    "summary": production_summary,
                    "failures": [],
                },
                {
                    "gate": "retrieval_benchmark",
                    "status": "passed",
                    "report": str(retrieval_report),
                    "summary": retrieval_summary,
                    "failures": [],
                },
                {
                    "gate": "live_provider",
                    "status": "passed",
                    "report": str(live_report),
                    "summary": live_summary,
                    "failures": [],
                },
                {
                    "gate": "semantic",
                    "status": "passed",
                    "report": str(semantic_report),
                    "summary": semantic_summary,
                    "failures": [],
                },
            ],
        },
    )
    return manifest


def test_acceptance_audit_accepts_verified_manifest(tmp_path: Path) -> None:
    manifest = _write_valid_manifest(tmp_path)

    audit = build_acceptance_audit(manifest)

    assert audit["metadata"]["status"] == "passed"
    assert audit["metadata"]["manifest_verified"] is True
    checklist = {item["id"]: item for item in audit["checklist"]}
    assert checklist["live_scenarios"]["status"] == "passed"
    assert checklist["semantic_suite"]["gate_summary"]["total_cases"] == 1000


def test_acceptance_audit_fails_without_manifest(tmp_path: Path) -> None:
    audit = build_acceptance_audit(tmp_path / "missing_manifest.json")

    assert audit["metadata"]["status"] == "failed"
    assert audit["metadata"]["manifest_verified"] is False
    assert any("manifest invalid" in failure for failure in audit["failures"])


def test_acceptance_audit_cli_writes_json(tmp_path: Path, capsys) -> None:
    manifest = _write_valid_manifest(tmp_path)
    out = tmp_path / "acceptance_audit.json"

    status = main(["--manifest", str(manifest), "--out", str(out)])

    captured = capsys.readouterr()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert status == 0
    assert "acceptance_audit_saved" in captured.out
    assert payload["metadata"]["status"] == "passed"
