# ruff: noqa: E402

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Sequence, TypedDict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.core.clients import build_default_client
from magi.core.config import get_settings
from magi.eval.build_adversarial_semantic_suite import (
    CATEGORIES,
    write_adversarial_semantic_suite,
)
from magi.eval.run_scenarios import _threshold_failures as scenario_threshold_failures
from magi.eval.retrieval_benchmark import (
    RetrievalBenchmarkReport,
    load_retrieval_benchmark_dataset,
    run_retrieval_benchmark,
    write_retrieval_benchmark_report,
)
from magi.eval.scenario_harness import (
    ScenarioReport,
    load_scenario_dataset,
    run_scenario_suite,
    write_scenario_report,
)
from magi.eval.semantic_harness import (
    SemanticReport,
    load_semantic_dataset,
    run_semantic_suite,
    semantic_threshold_failures,
    write_semantic_report,
)

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACT_DIR = Path(".magi") / "gauntlet"
DEFAULT_PRODUCTION_CASES = EVAL_DIR / "production_scenarios.yaml"
DEFAULT_LIVE_CASES = EVAL_DIR / "live_scenarios.yaml"
DEFAULT_RETRIEVAL_CASES = EVAL_DIR / "retrieval_benchmark.yaml"
ADVERSARIAL_SEMANTIC_CASES = 1000
EXPECTED_SEMANTIC_CATEGORIES = tuple(CATEGORIES)
MIN_UNIQUE_SEMANTIC_QUERIES = 950
ThresholdFailure = tuple[str, float, float, str]


class SemanticGateThresholds(TypedDict):
    min_pass_rate: float
    max_p95_latency_ms: float
    max_uncited_approvals: int
    max_empty_final_answers: int
    max_live_fallbacks: int


class RetrievalGateThresholds(TypedDict):
    min_overall_score: float
    min_retrieval_hit_rate: float
    min_retrieval_top_source_accuracy: float
    min_retrieval_mrr: float
    min_retrieval_source_recall: float


class GateRunResult(TypedDict, total=False):
    gate: str
    status: str
    mode: str
    cases: str
    report: str
    thresholds: dict[str, object]
    summary: dict[str, object]
    failures: list[dict[str, object]]


def _scenario_gate_args(gate: str) -> argparse.Namespace:
    defaults: dict[str, object] = {
        "min_overall_score": 1.0,
        "min_verdict_accuracy": 1.0,
        "min_requirement_pass_rate": 1.0,
        "min_retrieval_hit_rate": 1.0,
        "min_retrieval_top_source_accuracy": 1.0,
        "min_retrieval_source_recall": 1.0,
        "min_average_citation_hit_rate": None,
        "min_average_answer_support_score": 0.10,
        "min_supported_answer_rate": None,
        "min_cached_replay_hit_rate": 1.0,
        "max_p50_latency_ms": None,
        "max_p95_latency_ms": 1000.0,
        "max_cached_p95_latency_ms": 250.0,
        "max_max_latency_ms": None,
        "max_average_cost_usd": 0.001,
        "max_total_cost_usd": 0.0,
        "max_live_fallbacks": 0,
        "allow_live_fallbacks": False,
        "max_uncited_approvals": 0,
        "max_empty_final_answers": 0,
        "mode": "stub",
    }
    if gate == "production_stub":
        return argparse.Namespace(**defaults)
    if gate == "live_provider":
        live = {
            **defaults,
            "min_average_citation_hit_rate": 1.0,
            "min_average_answer_support_score": 0.20,
            "min_supported_answer_rate": 1.0,
            "max_p95_latency_ms": 60000.0,
            "max_average_cost_usd": 0.02,
            "max_total_cost_usd": None,
            "mode": "live",
        }
        return argparse.Namespace(**live)
    raise ValueError(f"unknown scenario gate: {gate}")


def _semantic_gate_thresholds() -> SemanticGateThresholds:
    return {
        "min_pass_rate": 0.95,
        "max_p95_latency_ms": 1000.0,
        "max_uncited_approvals": 0,
        "max_empty_final_answers": 0,
        "max_live_fallbacks": 0,
    }


def _retrieval_gate_thresholds() -> RetrievalGateThresholds:
    return {
        "min_overall_score": 1.0,
        "min_retrieval_hit_rate": 1.0,
        "min_retrieval_top_source_accuracy": 1.0,
        "min_retrieval_mrr": 1.0,
        "min_retrieval_source_recall": 1.0,
    }


def add_gauntlet_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory for generated suite files and JSON reports.",
    )
    parser.add_argument(
        "--production-cases",
        type=Path,
        default=DEFAULT_PRODUCTION_CASES,
        help="Production stub scenario dataset.",
    )
    parser.add_argument(
        "--live-cases",
        type=Path,
        default=DEFAULT_LIVE_CASES,
        help="Live provider scenario dataset.",
    )
    parser.add_argument(
        "--retrieval-cases",
        type=Path,
        default=DEFAULT_RETRIEVAL_CASES,
        help="Corpus-backed retrieval benchmark dataset.",
    )
    parser.add_argument(
        "--semantic-cases",
        type=Path,
        help=(
            "Optional prebuilt 1000-case semantic suite. Defaults to a generated "
            "deterministic suite."
        ),
    )
    parser.add_argument("--model", help="Optional MAGI model override.")
    parser.add_argument("--judge-model", help="Optional semantic judge model override.")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        help="Optional consolidated JSON manifest path.",
    )
    parser.add_argument(
        "--semantic-mode",
        choices=("stub", "live"),
        default="stub",
        help="MAGI execution mode for the semantic suite. The judge remains live.",
    )
    parser.add_argument(
        "--semantic-concurrency",
        type=int,
        default=8,
        help="Concurrent semantic judge workers.",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full MAGI production acceptance gauntlet."
    )
    add_gauntlet_arguments(parser)
    return parser.parse_args(argv)


def _force_stub(mode: str) -> bool:
    return mode == "stub"


def _require_live_provider_configured() -> None:
    settings = get_settings()
    has_openai = bool(str(getattr(settings, "openai_api_key", "") or "").strip())
    if not has_openai:
        raise RuntimeError(
            "full gauntlet requires OPENAI_API_KEY so live gates prove the Responses API runtime"
        )
    try:
        client = build_default_client(settings)
    except RuntimeError as exc:
        raise RuntimeError(
            "full gauntlet requires a configured OpenAI Responses API client"
        ) from exc
    if client is None:
        raise RuntimeError("full gauntlet requires a configured OpenAI Responses API client")


def _report_paths(args: argparse.Namespace) -> dict[str, Path]:
    artifact_dir = Path(args.artifact_dir)
    return {
        "production": artifact_dir / "production_report.json",
        "live": artifact_dir / "live_report.json",
        "retrieval": artifact_dir / "retrieval_benchmark_report.json",
        "semantic": artifact_dir / "adversarial_semantic_report.json",
        "semantic_cases": (
            Path(args.semantic_cases)
            if args.semantic_cases
            else artifact_dir / "adversarial_semantic.yaml"
        ),
        "manifest": (
            Path(args.manifest_out)
            if getattr(args, "manifest_out", None)
            else artifact_dir / "gauntlet_manifest.json"
        ),
    }


def _print_threshold_failures(
    gate: str, failures: Sequence[ThresholdFailure]
) -> None:
    for field, actual, threshold, direction in failures:
        print(
            "threshold_failed\t"
            f"{gate}\t{field}\tactual={actual:.4f}\t{direction}={threshold:.4f}",
            file=sys.stderr,
        )


def _print_scenario_summary(gate: str, report: ScenarioReport, path: Path) -> None:
    summary = report.summary
    print(
        "gate_summary\t"
        f"{gate}\toverall={summary.overall_score:.4f}"
        f"\tverdict={summary.verdict_accuracy:.4f}"
        f"\trequirements={summary.requirement_pass_rate:.4f}"
        f"\tretrieval={summary.retrieval_hit_rate:.4f}"
        f"\tsupport={summary.supported_answer_rate:.4f}"
        f"\tp95_ms={summary.latency_p95_ms:.3f}"
        f"\tcached_p95_ms={summary.cached_latency_p95_ms:.3f}"
        f"\tfallbacks={summary.live_fallback_count}"
        f"\treport={path}"
    )


def _print_retrieval_summary(report: RetrievalBenchmarkReport, path: Path) -> None:
    summary = report.summary
    print(
        "gate_summary\t"
        f"retrieval_benchmark\toverall={summary.overall_score:.4f}"
        f"\thit_rate={summary.retrieval_hit_rate:.4f}"
        f"\ttop_source={summary.retrieval_top_source_accuracy:.4f}"
        f"\tmrr={summary.retrieval_mrr:.4f}"
        f"\tsource_recall={summary.retrieval_source_recall:.4f}"
        f"\treport={path}"
    )


def _print_semantic_summary(report: SemanticReport, path: Path) -> None:
    summary = report.summary
    print(
        "gate_summary\t"
        f"semantic\tpass_rate={summary.pass_rate:.4f}"
        f"\tpassed={summary.passed_cases}/{summary.total_cases}"
        f"\tp95_ms={summary.latency_p95_ms:.3f}"
        f"\tempty={summary.empty_final_answer_count}"
        f"\tuncited={summary.uncited_approval_count}"
        f"\tfallbacks={summary.live_fallback_count}"
        f"\treport={path}"
    )


def _namespace_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: value
        for key, value in vars(args).items()
        if value is not None and isinstance(value, (str, int, float, bool))
    }


def _failure_payload(failures: Sequence[ThresholdFailure]) -> list[dict[str, object]]:
    return [
        {
            "metric": metric,
            "actual": actual,
            "threshold": threshold,
            "direction": direction,
        }
        for metric, actual, threshold, direction in failures
    ]


def _scenario_summary_payload(report: ScenarioReport) -> dict[str, object]:
    summary = report.summary
    return {
        "overall_score": summary.overall_score,
        "verdict_accuracy": summary.verdict_accuracy,
        "requirement_pass_rate": summary.requirement_pass_rate,
        "requested_mode": summary.requested_mode,
        "effective_mode": summary.effective_mode,
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "retrieval_hit_rate": summary.retrieval_hit_rate,
        "retrieval_top_source_accuracy": summary.retrieval_top_source_accuracy,
        "retrieval_source_recall": summary.retrieval_source_recall,
        "average_citation_hit_rate": summary.average_citation_hit_rate,
        "average_answer_support_score": summary.average_answer_support_score,
        "supported_answer_rate": summary.supported_answer_rate,
        "latency_p95_ms": summary.latency_p95_ms,
        "cached_latency_p95_ms": summary.cached_latency_p95_ms,
        "cached_replay_hit_rate": summary.cached_replay_hit_rate,
        "live_fallback_count": summary.live_fallback_count,
        "empty_final_answer_count": summary.empty_final_answer_count,
        "uncited_approval_count": summary.uncited_approval_count,
        "total_estimated_cost_usd": summary.total_estimated_cost_usd,
    }


def _retrieval_summary_payload(report: RetrievalBenchmarkReport) -> dict[str, object]:
    summary = report.summary
    return {
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "overall_score": summary.overall_score,
        "retrieval_evaluable_cases": summary.retrieval_evaluable_cases,
        "cases_with_retrieval_hits": summary.cases_with_retrieval_hits,
        "retrieval_hit_rate": summary.retrieval_hit_rate,
        "retrieval_ranked_cases": summary.retrieval_ranked_cases,
        "retrieval_top_source_accuracy": summary.retrieval_top_source_accuracy,
        "retrieval_mrr": summary.retrieval_mrr,
        "retrieval_source_recall": summary.retrieval_source_recall,
        "ingested_document_count": summary.ingested_document_count,
        "ingested_chunk_count": summary.ingested_chunk_count,
        "store_backend": summary.store_backend,
        "embedder": summary.embedder,
        "embedding_dimension": summary.embedding_dimension,
    }


def _semantic_summary_payload(report: SemanticReport) -> dict[str, object]:
    summary = report.summary
    return {
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "pass_rate": summary.pass_rate,
        "latency_p95_ms": summary.latency_p95_ms,
        "empty_final_answer_count": summary.empty_final_answer_count,
        "uncited_approval_count": summary.uncited_approval_count,
        "live_fallback_count": summary.live_fallback_count,
        "average_answer_quality_score": summary.average_answer_quality_score,
        "average_grounding_score": summary.average_grounding_score,
        "average_safety_score": summary.average_safety_score,
        "effective_mode": summary.effective_mode,
        "judge_model": str(report.metadata.get("judge_model", "")),
    }


def _acceptance_criteria() -> list[dict[str, object]]:
    return [
        {"id": "production_scenarios", "gate": "production_stub", "overall_score": 1.0},
        {
            "id": "retrieval_benchmark",
            "gate": "retrieval_benchmark",
            "overall_score": 1.0,
            "retrieval_hit_rate": 1.0,
            "retrieval_top_source_accuracy": 1.0,
            "retrieval_mrr": 1.0,
            "retrieval_source_recall": 1.0,
        },
        {
            "id": "live_scenarios",
            "gate": "live_provider",
            "overall_score": 1.0,
            "provider": "openai_responses_api",
            "effective_mode": "live",
            "max_live_fallbacks": 0,
        },
        {
            "id": "semantic_suite",
            "gate": "semantic",
            "case_count": ADVERSARIAL_SEMANTIC_CASES,
            "categories": list(EXPECTED_SEMANTIC_CATEGORIES),
            "min_pass_rate": 0.95,
        },
        {"id": "no_uncited_approvals", "max_uncited_approvals": 0},
        {"id": "no_empty_final_answers", "max_empty_final_answers": 0},
        {"id": "stub_latency", "max_p95_latency_ms": 1000},
        {"id": "cached_replay_latency", "max_cached_p95_latency_ms": 250},
    ]


def _write_gauntlet_manifest(
    *,
    path: Path,
    args: argparse.Namespace,
    paths: dict[str, Path],
    status: str,
    results: Sequence[GateRunResult],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "suite_type": "production_acceptance_gauntlet",
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "adversarial_semantic_cases": ADVERSARIAL_SEMANTIC_CASES,
            "criteria": _acceptance_criteria(),
        },
        "inputs": {
            "production_cases": str(args.production_cases),
            "live_cases": str(args.live_cases),
            "retrieval_cases": str(args.retrieval_cases),
            "semantic_cases": str(paths["semantic_cases"]),
            "semantic_mode": str(args.semantic_mode),
            "semantic_concurrency": int(args.semantic_concurrency),
            "model": str(getattr(args, "model", "") or ""),
            "judge_model": str(getattr(args, "judge_model", "") or ""),
        },
        "reports": {
            "production": str(paths["production"]),
            "live": str(paths["live"]),
            "retrieval": str(paths["retrieval"]),
            "semantic": str(paths["semantic"]),
            "manifest": str(path),
        },
        "results": list(results),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"manifest_saved\t{path}")


def _preflight_failure_result(error: str) -> GateRunResult:
    return {
        "gate": "provider_preflight",
        "status": "failed",
        "failures": [
            {
                "metric": "live_provider_configured",
                "actual": 0.0,
                "threshold": 1.0,
                "direction": "minimum",
                "message": error,
            }
        ],
    }


def _semantic_category_counts(dataset: object) -> dict[str, int]:
    counts = {category: 0 for category in EXPECTED_SEMANTIC_CATEGORIES}
    cases = getattr(dataset, "cases", [])
    for case in cases:
        tags = list(getattr(case, "tags", []) or [])
        category = str(tags[0]).strip() if tags else ""
        if category in counts:
            counts[category] += 1
    return counts


def _validate_semantic_suite_contract(dataset: object) -> None:
    cases = list(getattr(dataset, "cases", []) or [])
    if len(cases) != ADVERSARIAL_SEMANTIC_CASES:
        raise ValueError(
            "semantic gauntlet requires "
            f"{ADVERSARIAL_SEMANTIC_CASES} cases, got {len(cases)}"
        )
    counts = _semantic_category_counts(dataset)
    missing = [
        category
        for category in EXPECTED_SEMANTIC_CATEGORIES
        if counts.get(category, 0) <= 0
    ]
    if missing:
        raise ValueError(
            "semantic gauntlet requires categories: "
            + ", ".join(EXPECTED_SEMANTIC_CATEGORIES)
            + f"; missing: {', '.join(missing)}"
        )
    unique_queries = {str(getattr(case, "query", "")).strip() for case in cases}
    unique_queries.discard("")
    if len(unique_queries) < MIN_UNIQUE_SEMANTIC_QUERIES:
        raise ValueError(
            "semantic gauntlet requires at least "
            f"{MIN_UNIQUE_SEMANTIC_QUERIES} unique queries, got {len(unique_queries)}"
        )


def _run_scenario_gate(
    *,
    gate: str,
    cases: Path,
    report_out: Path,
    mode: str,
    model: str | None,
) -> tuple[bool, GateRunResult]:
    print(f"gate_start\t{gate}")
    dataset = load_scenario_dataset(cases)
    report = run_scenario_suite(
        dataset,
        force_stub=_force_stub(mode),
        model=model,
        requested_mode=mode,
    )
    write_scenario_report(report, report_out)
    _print_scenario_summary(gate, report, report_out)
    failures = scenario_threshold_failures(report, _scenario_gate_args(gate))
    result: GateRunResult = {
        "gate": gate,
        "status": "failed" if failures else "passed",
        "mode": mode,
        "cases": str(cases),
        "report": str(report_out),
        "thresholds": _namespace_payload(_scenario_gate_args(gate)),
        "summary": _scenario_summary_payload(report),
        "failures": _failure_payload(failures),
    }
    if failures:
        _print_threshold_failures(gate, failures)
        return False, result
    print(f"gate_passed\t{gate}")
    return True, result


def _retrieval_threshold_failures(
    report: RetrievalBenchmarkReport,
) -> list[ThresholdFailure]:
    summary = report.summary
    thresholds = _retrieval_gate_thresholds()
    checks = {
        "overall_score": thresholds["min_overall_score"],
        "retrieval_hit_rate": thresholds["min_retrieval_hit_rate"],
        "retrieval_top_source_accuracy": thresholds[
            "min_retrieval_top_source_accuracy"
        ],
        "retrieval_mrr": thresholds["min_retrieval_mrr"],
        "retrieval_source_recall": thresholds["min_retrieval_source_recall"],
    }
    failures: list[ThresholdFailure] = []
    for field, minimum in checks.items():
        actual = float(getattr(summary, field))
        if actual < minimum:
            failures.append((field, actual, minimum, "minimum"))
    return failures


def _run_retrieval_gate(
    *,
    cases: Path,
    report_out: Path,
) -> tuple[bool, GateRunResult]:
    print("gate_start\tretrieval_benchmark")
    dataset = load_retrieval_benchmark_dataset(cases)
    report = run_retrieval_benchmark(dataset, cases)
    write_retrieval_benchmark_report(report, report_out)
    _print_retrieval_summary(report, report_out)
    failures = _retrieval_threshold_failures(report)
    result: GateRunResult = {
        "gate": "retrieval_benchmark",
        "status": "failed" if failures else "passed",
        "cases": str(cases),
        "report": str(report_out),
        "thresholds": dict(_retrieval_gate_thresholds()),
        "summary": _retrieval_summary_payload(report),
        "failures": _failure_payload(failures),
    }
    if failures:
        _print_threshold_failures("retrieval_benchmark", failures)
        return False, result
    print("gate_passed\tretrieval_benchmark")
    return True, result


def _prepare_semantic_suite(args: argparse.Namespace, path: Path) -> None:
    if args.semantic_cases:
        dataset = load_semantic_dataset(path)
        _validate_semantic_suite_contract(dataset)
        return
    write_adversarial_semantic_suite(path, total=ADVERSARIAL_SEMANTIC_CASES)
    _validate_semantic_suite_contract(load_semantic_dataset(path))
    print(f"suite_saved\t{path}")
    print(f"suite_cases\t{ADVERSARIAL_SEMANTIC_CASES}")


def _run_semantic_gate(
    args: argparse.Namespace, suite_path: Path, report_out: Path
) -> tuple[bool, GateRunResult]:
    print("gate_start\tsemantic")
    _prepare_semantic_suite(args, suite_path)
    dataset = load_semantic_dataset(suite_path)
    report = run_semantic_suite(
        dataset,
        force_stub=_force_stub(args.semantic_mode),
        model=args.model,
        judge_model=args.judge_model,
        concurrency=args.semantic_concurrency,
    )
    write_semantic_report(report, report_out)
    _print_semantic_summary(report, report_out)
    thresholds = _semantic_gate_thresholds()
    failures = semantic_threshold_failures(
        report,
        requested_mode=str(args.semantic_mode),
        min_pass_rate=thresholds["min_pass_rate"],
        max_p95_latency_ms=thresholds["max_p95_latency_ms"],
        max_uncited_approvals=thresholds["max_uncited_approvals"],
        max_empty_final_answers=thresholds["max_empty_final_answers"],
        max_live_fallbacks=thresholds["max_live_fallbacks"],
    )
    result: GateRunResult = {
        "gate": "semantic",
        "status": "failed" if failures else "passed",
        "mode": str(args.semantic_mode),
        "cases": str(suite_path),
        "report": str(report_out),
        "thresholds": dict(thresholds),
        "summary": _semantic_summary_payload(report),
        "failures": _failure_payload(failures),
    }
    if failures:
        _print_threshold_failures("semantic", failures)
        return False, result
    print("gate_passed\tsemantic")
    return True, result


def run_gauntlet(args: argparse.Namespace) -> int:
    paths = _report_paths(args)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = paths["manifest"]
    results: list[GateRunResult] = []

    try:
        _require_live_provider_configured()
    except RuntimeError as exc:
        result = _preflight_failure_result(str(exc))
        results.append(result)
        _write_gauntlet_manifest(
            path=manifest_path,
            args=args,
            paths=paths,
            status="failed",
            results=results,
        )
        print(f"threshold_failed\tprovider_preflight\t{exc}", file=sys.stderr)
        return 1

    model = str(getattr(args, "model", "") or "").strip() or None
    passed, result = _run_scenario_gate(
        gate="production_stub",
        cases=args.production_cases,
        report_out=paths["production"],
        mode="stub",
        model=model,
    )
    results.append(result)
    if not passed:
        _write_gauntlet_manifest(
            path=manifest_path,
            args=args,
            paths=paths,
            status="failed",
            results=results,
        )
        return 1
    passed, result = _run_retrieval_gate(
        cases=args.retrieval_cases,
        report_out=paths["retrieval"],
    )
    results.append(result)
    if not passed:
        _write_gauntlet_manifest(
            path=manifest_path,
            args=args,
            paths=paths,
            status="failed",
            results=results,
        )
        return 1
    passed, result = _run_scenario_gate(
        gate="live_provider",
        cases=args.live_cases,
        report_out=paths["live"],
        mode="live",
        model=model,
    )
    results.append(result)
    if not passed:
        _write_gauntlet_manifest(
            path=manifest_path,
            args=args,
            paths=paths,
            status="failed",
            results=results,
        )
        return 1
    passed, result = _run_semantic_gate(args, paths["semantic_cases"], paths["semantic"])
    results.append(result)
    if not passed:
        _write_gauntlet_manifest(
            path=manifest_path,
            args=args,
            paths=paths,
            status="failed",
            results=results,
        )
        return 1
    _write_gauntlet_manifest(
        path=manifest_path,
        args=args,
        paths=paths,
        status="passed",
        results=results,
    )
    print("gauntlet_passed")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run_gauntlet(parse_args(argv))
    except Exception as exc:
        print(f"gauntlet_failed\t{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
