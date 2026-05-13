# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import yaml  # type: ignore[import-untyped]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.eval.build_adversarial_semantic_suite import CATEGORIES

ADVERSARIAL_SEMANTIC_CASES = 1000
EXPECTED_SEMANTIC_CATEGORIES = tuple(CATEGORIES)
MIN_UNIQUE_SEMANTIC_QUERIES = 950
REQUIRED_GATES = (
    "production_stub",
    "retrieval_benchmark",
    "live_provider",
    "semantic",
)


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _fail_min(
    failures: list[str], gate: str, summary: Mapping[str, Any], field: str, minimum: float
) -> None:
    actual = _float(summary.get(field))
    if actual < minimum:
        failures.append(f"{gate}.{field} {actual:.4f} below {minimum:.4f}")


def _fail_max(
    failures: list[str], gate: str, summary: Mapping[str, Any], field: str, maximum: float
) -> None:
    actual = _float(summary.get(field))
    if actual > maximum:
        failures.append(f"{gate}.{field} {actual:.4f} above {maximum:.4f}")


def _fail_equal(
    failures: list[str], gate: str, summary: Mapping[str, Any], field: str, expected: Any
) -> None:
    actual = summary.get(field)
    if actual != expected:
        failures.append(f"{gate}.{field} expected {expected!r}, got {actual!r}")


def _resolve_manifest_path(reference: str, manifest_path: Path) -> Path:
    candidate = Path(reference)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    sibling = manifest_path.parent / candidate.name
    if sibling.exists():
        return sibling
    return candidate


def _gate_results(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for item in _sequence(manifest.get("results")):
        result = _mapping(item)
        gate = str(result.get("gate", "")).strip()
        if gate:
            results[gate] = result
    return results


def _load_result_report(
    *,
    gate: str,
    result: Mapping[str, Any],
    manifest_path: Path,
    failures: list[str],
) -> Mapping[str, Any]:
    report_reference = str(result.get("report", "") or "").strip()
    if not report_reference:
        failures.append(f"{gate}.report missing")
        return {}
    report_path = _resolve_manifest_path(report_reference, manifest_path)
    if not report_path.exists():
        failures.append(f"{gate}.report file not found: {report_reference}")
        return {}
    try:
        return _load_json(report_path)
    except Exception as exc:
        failures.append(f"{gate}.report invalid JSON: {exc}")
        return {}


def _report_summary(
    *,
    gate: str,
    result: Mapping[str, Any],
    manifest_path: Path,
    failures: list[str],
    check_report_files: bool,
) -> Mapping[str, Any]:
    manifest_summary = _mapping(result.get("summary"))
    if not check_report_files:
        return manifest_summary
    report = _load_result_report(
        gate=gate,
        result=result,
        manifest_path=manifest_path,
        failures=failures,
    )
    if not report:
        return manifest_summary
    report_summary = _mapping(report.get("summary"))
    if not report_summary:
        failures.append(f"{gate}.report summary missing")
        return manifest_summary
    return report_summary


def _check_result_envelope(
    failures: list[str], gate: str, result: Mapping[str, Any]
) -> None:
    if result.get("status") != "passed":
        failures.append(f"{gate}.status expected 'passed', got {result.get('status')!r}")
    result_failures = _sequence(result.get("failures"))
    if result_failures:
        failures.append(f"{gate}.failures contains {len(result_failures)} item(s)")


def _verify_production_summary(
    failures: list[str], summary: Mapping[str, Any]
) -> None:
    gate = "production_stub"
    for field in (
        "overall_score",
        "verdict_accuracy",
        "requirement_pass_rate",
        "retrieval_hit_rate",
        "retrieval_top_source_accuracy",
        "retrieval_source_recall",
        "cached_replay_hit_rate",
    ):
        _fail_min(failures, gate, summary, field, 1.0)
    _fail_min(failures, gate, summary, "average_answer_support_score", 0.10)
    _fail_max(failures, gate, summary, "latency_p95_ms", 1000.0)
    _fail_max(failures, gate, summary, "cached_latency_p95_ms", 250.0)
    for field in (
        "live_fallback_count",
        "empty_final_answer_count",
        "uncited_approval_count",
    ):
        _fail_equal(failures, gate, summary, field, 0)


def _verify_live_summary(failures: list[str], summary: Mapping[str, Any]) -> None:
    gate = "live_provider"
    for field in (
        "overall_score",
        "verdict_accuracy",
        "requirement_pass_rate",
        "retrieval_hit_rate",
        "retrieval_top_source_accuracy",
        "retrieval_source_recall",
        "average_citation_hit_rate",
        "supported_answer_rate",
        "cached_replay_hit_rate",
    ):
        _fail_min(failures, gate, summary, field, 1.0)
    _fail_min(failures, gate, summary, "average_answer_support_score", 0.20)
    _fail_max(failures, gate, summary, "latency_p95_ms", 60000.0)
    _fail_max(failures, gate, summary, "cached_latency_p95_ms", 250.0)
    _fail_equal(failures, gate, summary, "effective_mode", "live")
    for field in (
        "live_fallback_count",
        "empty_final_answer_count",
        "uncited_approval_count",
    ):
        _fail_equal(failures, gate, summary, field, 0)


def _verify_retrieval_summary(
    failures: list[str], summary: Mapping[str, Any]
) -> None:
    gate = "retrieval_benchmark"
    for field in (
        "overall_score",
        "retrieval_hit_rate",
        "retrieval_top_source_accuracy",
        "retrieval_mrr",
        "retrieval_source_recall",
    ):
        _fail_min(failures, gate, summary, field, 1.0)
    _fail_min(failures, gate, summary, "total_cases", 1.0)
    _fail_min(failures, gate, summary, "ingested_document_count", 1.0)
    _fail_min(failures, gate, summary, "ingested_chunk_count", 1.0)


def _verify_semantic_summary(failures: list[str], summary: Mapping[str, Any]) -> None:
    gate = "semantic"
    _fail_equal(failures, gate, summary, "total_cases", ADVERSARIAL_SEMANTIC_CASES)
    _fail_min(failures, gate, summary, "pass_rate", 0.95)
    _fail_max(failures, gate, summary, "latency_p95_ms", 1000.0)
    for field in (
        "live_fallback_count",
        "empty_final_answer_count",
        "uncited_approval_count",
    ):
        _fail_equal(failures, gate, summary, field, 0)


def _verify_semantic_suite_file(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    failures: list[str],
    *,
    check_report_files: bool,
) -> set[str] | None:
    if not check_report_files:
        return None
    inputs = _mapping(manifest.get("inputs"))
    reference = str(inputs.get("semantic_cases", "") or "").strip()
    if not reference:
        failures.append("semantic_cases input missing")
        return None
    suite_path = _resolve_manifest_path(reference, manifest_path)
    if not suite_path.exists():
        failures.append(f"semantic_cases file not found: {reference}")
        return None
    try:
        payload = yaml.safe_load(suite_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        failures.append(f"semantic_cases invalid YAML: {exc}")
        return None
    if not isinstance(payload, Mapping):
        failures.append("semantic_cases file must contain a mapping")
        return None
    cases = _sequence(payload.get("cases"))
    if len(cases) != ADVERSARIAL_SEMANTIC_CASES:
        failures.append(
            "semantic_cases case count "
            f"{len(cases)} != {ADVERSARIAL_SEMANTIC_CASES}"
        )
    category_counts = {category: 0 for category in EXPECTED_SEMANTIC_CATEGORIES}
    identifiers: list[str] = []
    queries: list[str] = []
    missing_expected_behavior = 0
    for item in cases:
        case = _mapping(item)
        case_id = str(case.get("id", "") or "").strip()
        if case_id:
            identifiers.append(case_id)
        query = str(case.get("query", "") or "").strip()
        if query:
            queries.append(query)
        if not str(case.get("expected_behavior", "") or "").strip():
            missing_expected_behavior += 1
        tags = _sequence(case.get("tags"))
        category = str(tags[0]).strip() if tags else ""
        if category in category_counts:
            category_counts[category] += 1
    if len(identifiers) != len(cases):
        failures.append("semantic_cases contains missing case identifiers")
    if len(set(identifiers)) != len(identifiers):
        failures.append("semantic_cases contains duplicate case identifiers")
    if len(queries) != len(cases):
        failures.append("semantic_cases contains missing queries")
    unique_query_count = len(set(queries))
    if unique_query_count < MIN_UNIQUE_SEMANTIC_QUERIES:
        failures.append(
            "semantic_cases unique query count "
            f"{unique_query_count} below {MIN_UNIQUE_SEMANTIC_QUERIES}"
        )
    if missing_expected_behavior:
        failures.append(
            "semantic_cases contains "
            f"{missing_expected_behavior} missing expected_behavior value(s)"
        )
    missing = [
        category
        for category in EXPECTED_SEMANTIC_CATEGORIES
        if category_counts.get(category, 0) <= 0
    ]
    if missing:
        failures.append(
            "semantic_cases missing categories: " + ", ".join(sorted(missing))
        )
    return set(identifiers)


def _verify_semantic_report_cases(
    *,
    result: Mapping[str, Any],
    manifest_path: Path,
    suite_case_ids: set[str] | None,
    failures: list[str],
    check_report_files: bool,
) -> None:
    if not check_report_files:
        return
    report = _load_result_report(
        gate="semantic",
        result=result,
        manifest_path=manifest_path,
        failures=failures,
    )
    if not report:
        return
    cases = _sequence(report.get("cases"))
    if len(cases) != ADVERSARIAL_SEMANTIC_CASES:
        failures.append(
            "semantic.report case count "
            f"{len(cases)} != {ADVERSARIAL_SEMANTIC_CASES}"
        )
    category_counts = {category: 0 for category in EXPECTED_SEMANTIC_CATEGORIES}
    identifiers: list[str] = []
    for item in cases:
        case = _mapping(item)
        case_id = str(case.get("id", "") or "").strip()
        if case_id:
            identifiers.append(case_id)
        tags = _sequence(case.get("tags"))
        category = str(tags[0]).strip() if tags else ""
        if category in category_counts:
            category_counts[category] += 1
    if len(identifiers) != len(cases):
        failures.append("semantic.report contains missing case identifiers")
    report_ids = set(identifiers)
    if len(report_ids) != len(identifiers):
        failures.append("semantic.report contains duplicate case identifiers")
    if suite_case_ids is not None and report_ids != suite_case_ids:
        missing_count = len(suite_case_ids - report_ids)
        extra_count = len(report_ids - suite_case_ids)
        failures.append(
            "semantic.report case IDs do not match semantic_cases "
            f"(missing={missing_count}, extra={extra_count})"
        )
    missing_categories = [
        category
        for category in EXPECTED_SEMANTIC_CATEGORIES
        if category_counts.get(category, 0) <= 0
    ]
    if missing_categories:
        failures.append(
            "semantic.report missing categories: "
            + ", ".join(sorted(missing_categories))
        )


def verify_gauntlet_manifest(
    manifest_path: Path, *, check_report_files: bool = True
) -> list[str]:
    failures: list[str] = []
    try:
        manifest = _load_json(manifest_path)
    except Exception as exc:
        return [f"manifest invalid: {exc}"]

    metadata = _mapping(manifest.get("metadata"))
    if metadata.get("suite_type") != "production_acceptance_gauntlet":
        failures.append("metadata.suite_type is not production_acceptance_gauntlet")
    if metadata.get("status") != "passed":
        failures.append(f"metadata.status expected 'passed', got {metadata.get('status')!r}")
    if _int(metadata.get("adversarial_semantic_cases")) != ADVERSARIAL_SEMANTIC_CASES:
        failures.append("metadata.adversarial_semantic_cases must be 1000")

    criteria_ids = {
        str(item.get("id", "")).strip()
        for item in _sequence(metadata.get("criteria"))
        if isinstance(item, Mapping)
    }
    for expected in (
        "production_scenarios",
        "retrieval_benchmark",
        "live_scenarios",
        "semantic_suite",
        "no_uncited_approvals",
        "no_empty_final_answers",
        "stub_latency",
        "cached_replay_latency",
    ):
        if expected not in criteria_ids:
            failures.append(f"criteria missing {expected}")
    live_criteria = [
        _mapping(item)
        for item in _sequence(metadata.get("criteria"))
        if isinstance(item, Mapping)
        and str(item.get("id", "")).strip() == "live_scenarios"
    ]
    if not live_criteria:
        failures.append("criteria missing live_scenarios provider")
    elif live_criteria[0].get("provider") != "openai_responses_api":
        failures.append("criteria live_scenarios.provider must be openai_responses_api")

    results = _gate_results(manifest)
    for gate in REQUIRED_GATES:
        if gate not in results:
            failures.append(f"results missing {gate}")
    for gate, result in results.items():
        _check_result_envelope(failures, gate, result)

    if "production_stub" in results:
        summary = _report_summary(
            gate="production_stub",
            result=results["production_stub"],
            manifest_path=manifest_path,
            failures=failures,
            check_report_files=check_report_files,
        )
        _verify_production_summary(failures, summary)
    if "live_provider" in results:
        summary = _report_summary(
            gate="live_provider",
            result=results["live_provider"],
            manifest_path=manifest_path,
            failures=failures,
            check_report_files=check_report_files,
        )
        _verify_live_summary(failures, summary)
    if "retrieval_benchmark" in results:
        summary = _report_summary(
            gate="retrieval_benchmark",
            result=results["retrieval_benchmark"],
            manifest_path=manifest_path,
            failures=failures,
            check_report_files=check_report_files,
        )
        _verify_retrieval_summary(failures, summary)
    suite_case_ids = _verify_semantic_suite_file(
        manifest,
        manifest_path,
        failures,
        check_report_files=check_report_files,
    )
    if "semantic" in results:
        summary = _report_summary(
            gate="semantic",
            result=results["semantic"],
            manifest_path=manifest_path,
            failures=failures,
            check_report_files=check_report_files,
        )
        _verify_semantic_summary(failures, summary)
        semantic_mode = str(_mapping(manifest.get("inputs")).get("semantic_mode", ""))
        if semantic_mode == "live":
            _fail_equal(failures, "semantic", summary, "effective_mode", "live")
        _verify_semantic_report_cases(
            result=results["semantic"],
            manifest_path=manifest_path,
            suite_case_ids=suite_case_ids,
            failures=failures,
            check_report_files=check_report_files,
        )
    return failures


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a MAGI production gauntlet manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".magi") / "gauntlet" / "gauntlet_manifest.json",
        help="Path to gauntlet_manifest.json.",
    )
    parser.add_argument(
        "--skip-report-file-check",
        action="store_true",
        help="Validate manifest contents only, without opening linked reports or suite files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    failures = verify_gauntlet_manifest(
        args.manifest,
        check_report_files=not args.skip_report_file_check,
    )
    if failures:
        for failure in failures:
            print(f"gauntlet_manifest_failed\t{failure}", file=sys.stderr)
        return 1
    print(f"gauntlet_manifest_verified\t{args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
