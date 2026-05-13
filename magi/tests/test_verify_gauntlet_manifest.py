from __future__ import annotations

import json
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from magi.eval.verify_gauntlet_manifest import (
    EXPECTED_SEMANTIC_CATEGORIES,
    main,
    verify_gauntlet_manifest,
)


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def _semantic_cases() -> list[dict[str, object]]:
    return [
        {
            "id": f"case-{index}",
            "query": f"How should MAGI handle semantic validation case {index}?",
            "expected_behavior": "Judge semantic behavior using evidence rather than keywords.",
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
    _write_json(
        semantic_report,
        {"summary": semantic_summary, "cases": semantic_cases},
    )
    semantic_suite.write_text(
        yaml.safe_dump(
            {"cases": semantic_cases}
        ),
        encoding="utf-8",
    )
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


def test_verify_gauntlet_manifest_accepts_valid_manifest(tmp_path: Path) -> None:
    manifest = _write_valid_manifest(tmp_path)

    assert verify_gauntlet_manifest(manifest) == []


def test_verify_gauntlet_manifest_rejects_stale_report_metrics(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    live_report = tmp_path / "live_report.json"
    payload = json.loads(live_report.read_text(encoding="utf-8"))
    payload["summary"]["effective_mode"] = "stub"
    payload["summary"]["live_fallback_count"] = 2
    _write_json(live_report, payload)

    failures = verify_gauntlet_manifest(manifest)

    assert "live_provider.effective_mode expected 'live', got 'stub'" in failures
    assert "live_provider.live_fallback_count expected 0, got 2" in failures


def test_verify_gauntlet_manifest_rejects_retrieval_metric_regression(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    retrieval_report = tmp_path / "retrieval_benchmark_report.json"
    payload = json.loads(retrieval_report.read_text(encoding="utf-8"))
    payload["summary"]["retrieval_top_source_accuracy"] = 0.5
    _write_json(retrieval_report, payload)

    failures = verify_gauntlet_manifest(manifest)

    assert (
        "retrieval_benchmark.retrieval_top_source_accuracy 0.5000 below 1.0000"
        in failures
    )


def test_verify_gauntlet_manifest_rejects_missing_semantic_categories(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    semantic_suite = tmp_path / "adversarial_semantic.yaml"
    semantic_suite.write_text(
        yaml.safe_dump(
            {
                "cases": [
                    {"id": f"case-{index}", "tags": ["summary"]}
                    for index in range(1000)
                ]
            }
        ),
        encoding="utf-8",
    )

    failures = verify_gauntlet_manifest(manifest)

    assert any(
        failure.startswith("semantic_cases missing categories:")
        for failure in failures
    )


def test_verify_gauntlet_manifest_rejects_semantic_report_case_mismatch(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    semantic_report = tmp_path / "adversarial_semantic_report.json"
    payload = json.loads(semantic_report.read_text(encoding="utf-8"))
    payload["cases"][0]["id"] = "unexpected-case"
    _write_json(semantic_report, payload)

    failures = verify_gauntlet_manifest(manifest)

    assert any(
        failure.startswith("semantic.report case IDs do not match semantic_cases")
        for failure in failures
    )


def test_verify_gauntlet_manifest_rejects_low_diversity_semantic_suite(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    semantic_suite = tmp_path / "adversarial_semantic.yaml"
    payload = yaml.safe_load(semantic_suite.read_text(encoding="utf-8"))
    for case in payload["cases"]:
        case["query"] = "Repeated semantic validation query."
    semantic_suite.write_text(yaml.safe_dump(payload), encoding="utf-8")

    failures = verify_gauntlet_manifest(manifest)

    assert any(
        failure.startswith("semantic_cases unique query count")
        for failure in failures
    )


def test_verify_gauntlet_manifest_rejects_failed_extra_gate(tmp_path: Path) -> None:
    manifest = _write_valid_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["results"].append(
        {
            "gate": "provider_preflight",
            "status": "failed",
            "failures": [{"metric": "live_provider_configured"}],
        }
    )
    _write_json(manifest, payload)

    failures = verify_gauntlet_manifest(manifest)

    assert "provider_preflight.status expected 'passed', got 'failed'" in failures
    assert "provider_preflight.failures contains 1 item(s)" in failures


def test_verify_gauntlet_manifest_rejects_live_semantic_stub_report(
    tmp_path: Path,
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["inputs"]["semantic_mode"] = "live"
    _write_json(manifest, payload)

    failures = verify_gauntlet_manifest(manifest)

    assert "semantic.effective_mode expected 'live', got 'stub'" in failures


def test_verify_gauntlet_manifest_can_check_manifest_only(tmp_path: Path) -> None:
    manifest = _write_valid_manifest(tmp_path)
    (tmp_path / "live_report.json").unlink()

    assert verify_gauntlet_manifest(manifest, check_report_files=False) == []


def test_verify_gauntlet_manifest_cli_reports_failures(
    tmp_path: Path, capsys
) -> None:
    manifest = _write_valid_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["metadata"]["status"] = "failed"
    _write_json(manifest, payload)

    status = main(["--manifest", str(manifest), "--skip-report-file-check"])

    captured = capsys.readouterr()
    assert status == 1
    assert "metadata.status expected 'passed'" in captured.err
