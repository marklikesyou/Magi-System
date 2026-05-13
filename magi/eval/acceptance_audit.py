# ruff: noqa: E402

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.eval.verify_gauntlet_manifest import verify_gauntlet_manifest

OBJECTIVE = (
    "Production-grade MAGI evidence-decision engine with preserved Melchior, "
    "Balthasar, and Casper contracts; Responses API-backed live runtime; "
    "deterministic offline parity; calibrated evidence gates; replayable "
    "operator artifacts; and a manifest-verifiable acceptance gauntlet."
)


@dataclass(frozen=True)
class ChecklistSpec:
    id: str
    requirement: str
    evidence: tuple[str, ...]
    required_paths: tuple[str, ...] = ()
    manifest_gate: str = ""
    requires_verified_manifest: bool = False


CHECKLIST: tuple[ChecklistSpec, ...] = (
    ChecklistSpec(
        id="persona_contracts",
        requirement="Melchior, Balthasar, and Casper names, responsibilities, schemas, and public outputs are preserved.",
        evidence=(
            "magi/tests/test_persona_contracts.py",
            "magi/dspy_programs/personas.py",
            "magi/dspy_programs/schemas.py",
            "magi/dspy_programs/signatures.py",
        ),
        required_paths=(
            "magi/tests/test_persona_contracts.py",
            "magi/dspy_programs/personas.py",
            "magi/dspy_programs/schemas.py",
            "magi/dspy_programs/signatures.py",
        ),
    ),
    ChecklistSpec(
        id="responses_api_live_runtime",
        requirement="Responses-API-backed live runtime with strict structured outputs is available and proven by the live gate.",
        evidence=(
            "magi/core/clients.py",
            "magi/tests/test_clients.py",
            "gauntlet_manifest.json live_provider gate",
        ),
        required_paths=("magi/core/clients.py", "magi/tests/test_clients.py"),
        manifest_gate="live_provider",
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="parallel_personas",
        requirement="Live persona execution runs Melchior, Balthasar, and Casper in parallel without changing their contracts.",
        evidence=(
            "magi/dspy_programs/runtime.py",
            "magi/tests/test_runtime.py::test_live_personas_run_in_parallel_when_client_supports_it",
        ),
        required_paths=("magi/dspy_programs/runtime.py", "magi/tests/test_runtime.py"),
    ),
    ChecklistSpec(
        id="trace_ids_spans",
        requirement="Trace IDs and execution spans are captured in decision records and run artifacts.",
        evidence=(
            "magi/app/service.py",
            "magi/app/artifacts.py",
            "magi/tests/test_artifacts.py",
            "magi/tests/test_magi_integration.py",
        ),
        required_paths=(
            "magi/app/service.py",
            "magi/app/artifacts.py",
            "magi/tests/test_artifacts.py",
            "magi/tests/test_magi_integration.py",
        ),
    ),
    ChecklistSpec(
        id="deterministic_offline_parity",
        requirement="Deterministic offline mode remains testable with stub DSPy and hashing embeddings.",
        evidence=("MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q",),
        required_paths=("magi/tests",),
    ),
    ChecklistSpec(
        id="calibrated_thresholds",
        requirement="Approval, abstention, citation, answer-support, cost, fallback, and latency thresholds are enforced.",
        evidence=(
            "magi/eval/run_scenarios.py",
            "magi/eval/scenario_harness.py",
            "magi/eval/run_gauntlet.py",
            "magi/eval/verify_gauntlet_manifest.py",
        ),
        required_paths=(
            "magi/eval/run_scenarios.py",
            "magi/eval/scenario_harness.py",
            "magi/eval/run_gauntlet.py",
            "magi/eval/verify_gauntlet_manifest.py",
        ),
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="retrieval_reranking_citations",
        requirement="Evidence retrieval, reranking, source recall, and citation verification are gated.",
        evidence=(
            "magi/core/rag.py",
            "retrieval_benchmark_report.json",
            "gauntlet_manifest.json retrieval_benchmark gate",
        ),
        required_paths=("magi/core/rag.py", "magi/eval/retrieval_benchmark.yaml"),
        manifest_gate="retrieval_benchmark",
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="profile_rendering",
        requirement="Profile-aware answer rendering is implemented for operator workflows.",
        evidence=(
            "magi/app/presentation.py",
            "magi/profiles/*.yaml",
            "magi/tests/test_presentation.py",
            "magi/tests/test_profiles.py",
        ),
        required_paths=(
            "magi/app/presentation.py",
            "magi/profiles/security-review.yaml",
            "magi/tests/test_presentation.py",
            "magi/tests/test_profiles.py",
        ),
    ),
    ChecklistSpec(
        id="replay_diff_debug_artifacts",
        requirement="Replay, diff, explain, and debug artifacts are available through CLI and rendered artifacts.",
        evidence=(
            "magi/app/artifacts.py",
            "magi/app/cli.py",
            "magi/tests/test_artifacts.py",
            "magi/tests/test_cli_entrypoints.py",
        ),
        required_paths=(
            "magi/app/artifacts.py",
            "magi/app/cli.py",
            "magi/tests/test_artifacts.py",
            "magi/tests/test_cli_entrypoints.py",
        ),
    ),
    ChecklistSpec(
        id="production_scenarios",
        requirement="Production scenario gate passes at 100% with strict deterministic thresholds.",
        evidence=("production_report.json", "gauntlet_manifest.json production_stub gate"),
        manifest_gate="production_stub",
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="live_scenarios",
        requirement="OpenAI live scenario gate passes at 100% with zero live fallbacks.",
        evidence=("live_report.json", "gauntlet_manifest.json live_provider gate"),
        manifest_gate="live_provider",
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="semantic_suite",
        requirement="1,000-case adversarial semantic suite passes at >=95% across summary, extract, fact-check, recommend, decision, injection, and harmful prompts.",
        evidence=(
            "adversarial_semantic.yaml",
            "adversarial_semantic_report.json",
            "gauntlet_manifest.json semantic gate",
        ),
        manifest_gate="semantic",
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="zero_bad_outputs",
        requirement="Acceptance reports prove zero uncited approvals, zero live fallbacks in live gates, and no empty final answers.",
        evidence=("gauntlet_manifest.json", "magi/eval/verify_gauntlet_manifest.py"),
        required_paths=("magi/eval/verify_gauntlet_manifest.py",),
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="latency_gates",
        requirement="Stub p95 latency is below 1s and cached/replay p95 latency is below 250ms.",
        evidence=("production_report.json", "live_report.json", "gauntlet_manifest.json"),
        requires_verified_manifest=True,
    ),
    ChecklistSpec(
        id="operator_workflows",
        requirement="Operator workflows for setup, evaluation, gauntlet verification, replay, diff, and audit are documented.",
        evidence=("README.md", "magi/eval/ACCEPTANCE.md"),
        required_paths=("README.md", "magi/eval/ACCEPTANCE.md"),
    ),
)


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, list) else []


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _gate_results(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    results: dict[str, Mapping[str, Any]] = {}
    for item in _sequence(manifest.get("results")):
        result = _mapping(item)
        gate = str(result.get("gate", "")).strip()
        if gate:
            results[gate] = result
    return results


def _missing_paths(paths: Sequence[str]) -> list[str]:
    return [path for path in paths if not (ROOT / path).exists()]


def _checklist_item(
    spec: ChecklistSpec,
    *,
    manifest_verified: bool,
    manifest_failures: Sequence[str],
    gates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    failures: list[str] = []
    missing_paths = _missing_paths(spec.required_paths)
    if missing_paths:
        failures.append("missing paths: " + ", ".join(missing_paths))
    gate_summary: Mapping[str, Any] = {}
    if spec.manifest_gate:
        gate = gates.get(spec.manifest_gate)
        if gate is None:
            failures.append(f"manifest gate missing: {spec.manifest_gate}")
        elif gate.get("status") != "passed":
            failures.append(
                f"manifest gate {spec.manifest_gate} status is {gate.get('status')!r}"
            )
        else:
            gate_summary = _mapping(gate.get("summary"))
    if spec.requires_verified_manifest and not manifest_verified:
        failures.extend(str(item) for item in manifest_failures)
    return {
        "id": spec.id,
        "requirement": spec.requirement,
        "status": "failed" if failures else "passed",
        "evidence": list(spec.evidence),
        "gate": spec.manifest_gate,
        "gate_summary": dict(gate_summary),
        "failures": failures,
    }


def build_acceptance_audit(
    manifest_path: Path,
    *,
    check_report_files: bool = True,
) -> dict[str, Any]:
    manifest_failures = verify_gauntlet_manifest(
        manifest_path,
        check_report_files=check_report_files,
    )
    manifest: Mapping[str, Any] = {}
    if manifest_path.exists():
        try:
            manifest = _load_json(manifest_path)
        except Exception as exc:
            manifest_failures = [*manifest_failures, f"manifest unreadable: {exc}"]
    manifest_verified = not manifest_failures
    gates = _gate_results(manifest)
    checklist = [
        _checklist_item(
            spec,
            manifest_verified=manifest_verified,
            manifest_failures=manifest_failures,
            gates=gates,
        )
        for spec in CHECKLIST
    ]
    failures = [
        f"{item['id']}: {failure}"
        for item in checklist
        for failure in _sequence(item.get("failures"))
    ]
    return {
        "metadata": {
            "suite_type": "acceptance_audit",
            "status": "failed" if failures else "passed",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "manifest_verified": manifest_verified,
            "checked_report_files": check_report_files,
        },
        "objective": OBJECTIVE,
        "checklist": checklist,
        "failures": failures,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a MAGI objective-to-artifact acceptance audit."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".magi") / "gauntlet" / "gauntlet_manifest.json",
        help="Path to gauntlet_manifest.json.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path to write the acceptance audit JSON.",
    )
    parser.add_argument(
        "--skip-report-file-check",
        action="store_true",
        help="Validate manifest contents only, without opening linked reports or suite files.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = build_acceptance_audit(
        args.manifest,
        check_report_files=not args.skip_report_file_check,
    )
    text = json.dumps(audit, ensure_ascii=True, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"acceptance_audit_saved\t{args.out}")
    else:
        print(text)
    return 0 if audit["metadata"]["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
