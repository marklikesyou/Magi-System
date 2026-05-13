from __future__ import annotations

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from magi.eval.build_adversarial_semantic_suite import (
    CATEGORIES,
    build_adversarial_semantic_suite,
    write_adversarial_semantic_suite,
)
from magi.eval.semantic_harness import (
    SemanticReport,
    SemanticSummary,
    load_semantic_dataset,
    semantic_threshold_failures,
)


def test_adversarial_semantic_suite_generator_builds_balanced_1000_cases() -> None:
    payload = build_adversarial_semantic_suite(total=1000)
    cases = payload["cases"]
    metadata = payload["metadata"]

    assert len(cases) == 1000
    assert metadata["case_count"] == 1000
    assert metadata["unique_query_count"] == 1000
    assert set(metadata["categories"]) == set(CATEGORIES)
    assert min(metadata["categories"].values()) >= 142
    assert len({case["id"] for case in cases}) == 1000
    assert len({case["query"] for case in cases}) == 1000
    assert {case["tags"][0] for case in cases} == set(CATEGORIES)
    assert all(case["expected_behavior"].strip() for case in cases)


def test_adversarial_semantic_suite_file_validates(tmp_path: Path) -> None:
    suite_path = tmp_path / "adversarial_semantic.yaml"

    write_adversarial_semantic_suite(suite_path, total=14)

    raw = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    assert raw["metadata"]["case_count"] == 14
    dataset = load_semantic_dataset(suite_path)
    assert len(dataset.cases) == 14
    assert {case.tags[0] for case in dataset.cases} == set(CATEGORIES)


def test_semantic_threshold_failures_report_gate_misses() -> None:
    report = SemanticReport(
        summary=SemanticSummary(
            total_cases=1000,
            passed_cases=940,
            pass_rate=0.94,
            average_answer_quality_score=0.9,
            average_grounding_score=0.9,
            average_safety_score=0.9,
            latency_p50_ms=100.0,
            latency_p95_ms=260.0,
            latency_max_ms=300.0,
            empty_final_answer_count=1,
            uncited_approval_count=2,
            live_fallback_count=3,
        )
    )

    failures = semantic_threshold_failures(
        report,
        min_pass_rate=0.95,
        max_p95_latency_ms=250.0,
        max_uncited_approvals=0,
        max_empty_final_answers=0,
        max_live_fallbacks=0,
    )

    assert failures == [
        ("pass_rate", 0.94, 0.95, "minimum"),
        ("latency_p95_ms", 260.0, 250.0, "maximum"),
        ("uncited_approval_count", 2.0, 0.0, "maximum"),
        ("empty_final_answer_count", 1.0, 0.0, "maximum"),
        ("live_fallback_count", 3.0, 0.0, "maximum"),
    ]


def test_semantic_threshold_failures_require_live_effective_mode() -> None:
    report = SemanticReport(
        summary=SemanticSummary(
            total_cases=1000,
            passed_cases=1000,
            pass_rate=1.0,
            average_answer_quality_score=1.0,
            average_grounding_score=1.0,
            average_safety_score=1.0,
            latency_p50_ms=10.0,
            latency_p95_ms=20.0,
            latency_max_ms=30.0,
            effective_mode="stub",
        )
    )

    failures = semantic_threshold_failures(report, requested_mode="live")

    assert failures == [("effective_mode_live", 0.0, 1.0, "minimum")]
