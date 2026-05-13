from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import magi.eval.run_gauntlet as run_gauntlet
from magi.eval.build_adversarial_semantic_suite import (
    write_adversarial_semantic_suite,
)
from magi.eval.run_gauntlet import (
    ADVERSARIAL_SEMANTIC_CASES,
    EXPECTED_SEMANTIC_CATEGORIES,
    _prepare_semantic_suite,
    _require_live_provider_configured,
    _retrieval_gate_thresholds,
    _scenario_gate_args,
    _semantic_gate_thresholds,
    parse_args,
)


def test_scenario_gauntlet_gate_args_are_fixed_acceptance_thresholds() -> None:
    production = _scenario_gate_args("production_stub")
    live = _scenario_gate_args("live_provider")

    assert production.min_overall_score == 1.0
    assert production.min_verdict_accuracy == 1.0
    assert production.min_requirement_pass_rate == 1.0
    assert production.min_retrieval_hit_rate == 1.0
    assert production.min_cached_replay_hit_rate == 1.0
    assert production.max_cached_p95_latency_ms == 250.0
    assert production.max_live_fallbacks == 0
    assert production.max_uncited_approvals == 0
    assert production.max_empty_final_answers == 0
    assert production.allow_live_fallbacks is False
    assert production.mode == "stub"

    assert live.min_average_citation_hit_rate == 1.0
    assert live.min_supported_answer_rate == 1.0
    assert live.max_p95_latency_ms == 60000.0
    assert live.max_average_cost_usd == 0.02
    assert live.max_total_cost_usd is None
    assert live.max_live_fallbacks == 0
    assert live.mode == "live"


def test_semantic_gauntlet_defaults_require_live_judge_quality_gate() -> None:
    args = parse_args(["--manifest-out", ".magi/custom_manifest.json"])
    thresholds = _semantic_gate_thresholds()
    retrieval_thresholds = _retrieval_gate_thresholds()

    assert args.semantic_mode == "stub"
    assert args.semantic_concurrency == 8
    assert args.manifest_out == Path(".magi/custom_manifest.json")
    assert args.retrieval_cases == run_gauntlet.DEFAULT_RETRIEVAL_CASES
    assert thresholds == {
        "min_pass_rate": 0.95,
        "max_p95_latency_ms": 1000.0,
        "max_uncited_approvals": 0,
        "max_empty_final_answers": 0,
        "max_live_fallbacks": 0,
    }
    assert retrieval_thresholds == {
        "min_overall_score": 1.0,
        "min_retrieval_hit_rate": 1.0,
        "min_retrieval_top_source_accuracy": 1.0,
        "min_retrieval_mrr": 1.0,
        "min_retrieval_source_recall": 1.0,
    }
    assert ADVERSARIAL_SEMANTIC_CASES == 1000


def test_prebuilt_semantic_gauntlet_suite_must_have_1000_cases(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "small_semantic.yaml"
    write_adversarial_semantic_suite(suite_path, total=14)
    args = parse_args(["--semantic-cases", str(suite_path)])

    with pytest.raises(ValueError, match="requires 1000 cases"):
        _prepare_semantic_suite(args, suite_path)


def test_prebuilt_semantic_gauntlet_suite_must_cover_categories(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "single_category_semantic.yaml"
    cases = [
        {
            "id": f"summary_{index}",
            "query": "Summarize the status note.",
            "expected_behavior": "Return a grounded summary.",
            "tags": ["summary"],
        }
        for index in range(ADVERSARIAL_SEMANTIC_CASES)
    ]
    suite_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    args = parse_args(["--semantic-cases", str(suite_path)])

    with pytest.raises(ValueError, match="missing:"):
        _prepare_semantic_suite(args, suite_path)


def test_prebuilt_semantic_gauntlet_suite_must_have_diverse_queries(
    tmp_path: Path,
) -> None:
    suite_path = tmp_path / "duplicate_query_semantic.yaml"
    cases = []
    for index in range(ADVERSARIAL_SEMANTIC_CASES):
        category = EXPECTED_SEMANTIC_CATEGORIES[
            index % len(EXPECTED_SEMANTIC_CATEGORIES)
        ]
        cases.append(
            {
                "id": f"{category}_{index}",
                "query": f"Repeated query for {category}.",
                "expected_behavior": "Judge semantic behavior.",
                "tags": [category],
            }
        )
    suite_path.write_text(json.dumps({"cases": cases}), encoding="utf-8")
    args = parse_args(["--semantic-cases", str(suite_path)])

    with pytest.raises(ValueError, match="unique queries"):
        _prepare_semantic_suite(args, suite_path)


def test_gauntlet_requires_live_provider_before_running(monkeypatch) -> None:
    monkeypatch.setattr(
        run_gauntlet,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="", google_api_key="google-key"),
    )

    with pytest.raises(RuntimeError, match="requires OPENAI_API_KEY"):
        _require_live_provider_configured()


def test_run_gauntlet_writes_failure_manifest_for_missing_provider(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        run_gauntlet,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="", google_api_key=""),
    )
    args = parse_args(["--artifact-dir", str(tmp_path)])

    status = run_gauntlet.run_gauntlet(args)

    manifest = json.loads(
        (tmp_path / "gauntlet_manifest.json").read_text(encoding="utf-8")
    )
    assert status == 1
    assert manifest["metadata"]["status"] == "failed"
    assert manifest["results"][0]["gate"] == "provider_preflight"
    assert manifest["results"][0]["failures"][0]["metric"] == "live_provider_configured"


def test_run_gauntlet_writes_success_manifest(monkeypatch, tmp_path: Path) -> None:
    args = parse_args(["--artifact-dir", str(tmp_path)])
    monkeypatch.setattr(run_gauntlet, "_require_live_provider_configured", lambda: None)

    def fake_scenario_gate(**kwargs: object) -> tuple[bool, dict[str, object]]:
        gate = str(kwargs["gate"])
        return True, {
            "gate": gate,
            "status": "passed",
            "report": str(kwargs["report_out"]),
            "summary": {"overall_score": 1.0},
            "failures": [],
        }

    def fake_semantic_gate(
        _args: object, suite_path: Path, report_out: Path
    ) -> tuple[bool, dict[str, object]]:
        return True, {
            "gate": "semantic",
            "status": "passed",
            "cases": str(suite_path),
            "report": str(report_out),
            "summary": {"pass_rate": 0.95},
            "failures": [],
        }

    def fake_retrieval_gate(**kwargs: object) -> tuple[bool, dict[str, object]]:
        return True, {
            "gate": "retrieval_benchmark",
            "status": "passed",
            "cases": str(kwargs["cases"]),
            "report": str(kwargs["report_out"]),
            "summary": {"overall_score": 1.0, "retrieval_hit_rate": 1.0},
            "failures": [],
        }

    monkeypatch.setattr(run_gauntlet, "_run_scenario_gate", fake_scenario_gate)
    monkeypatch.setattr(run_gauntlet, "_run_retrieval_gate", fake_retrieval_gate)
    monkeypatch.setattr(run_gauntlet, "_run_semantic_gate", fake_semantic_gate)

    status = run_gauntlet.run_gauntlet(args)

    manifest = json.loads(
        (tmp_path / "gauntlet_manifest.json").read_text(encoding="utf-8")
    )
    assert status == 0
    assert manifest["metadata"]["status"] == "passed"
    assert manifest["metadata"]["adversarial_semantic_cases"] == 1000
    semantic_criterion = next(
        item
        for item in manifest["metadata"]["criteria"]
        if item["id"] == "semantic_suite"
    )
    assert semantic_criterion["categories"] == list(EXPECTED_SEMANTIC_CATEGORIES)
    assert [result["gate"] for result in manifest["results"]] == [
        "production_stub",
        "retrieval_benchmark",
        "live_provider",
        "semantic",
    ]
    assert manifest["reports"]["retrieval"].endswith("retrieval_benchmark_report.json")
    assert manifest["reports"]["manifest"].endswith("gauntlet_manifest.json")
