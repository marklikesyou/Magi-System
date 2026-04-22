from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Iterable, Mapping

from magi.app.service import ChatSessionResult
from magi.core.storage import save_json_document

DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def artifact_dir(settings: object | None = None) -> Path:
    configured = str(getattr(settings, "run_artifact_dir", "") or "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_ARTIFACT_DIR


def decision_payload(result: ChatSessionResult) -> dict[str, object]:
    return {
        "decision": result.final_decision.model_dump(mode="json"),
        "fused": result.fused.model_dump(mode="json"),
        "decision_trace": asdict(result.decision_trace),
        "effective_mode": result.effective_mode,
        "model": result.model,
    }


def build_run_id(query_hash: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{query_hash[:12]}"


def build_run_artifact(
    *,
    result: ChatSessionResult,
    query: str,
    constraints: str,
    store_path: Path,
    store_metadata: Mapping[str, object] | None = None,
    profile_name: str = "",
    requested_route: str = "",
    artifact_path: Path | None = None,
) -> dict[str, object]:
    trace = result.decision_trace
    run_id = build_run_id(trace.query_hash)
    payload = decision_payload(result)
    payload.update(
        {
            "artifact_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifact_path": str(artifact_path) if artifact_path is not None else "",
            "input": {
                "query": query,
                "constraints": constraints,
                "profile": profile_name,
                "requested_route": requested_route,
            },
            "store": {
                "path": str(store_path),
                "metadata": dict(store_metadata or {}),
            },
            "summary": {
                "verdict": result.final_decision.verdict,
                "query_mode": trace.query_mode,
                "citation_hit_rate": trace.citation_hit_rate,
                "answer_support_score": trace.answer_support_score,
                "answer_supported": trace.answer_supported,
                "requires_human_review": trace.requires_human_review,
                "abstained": trace.abstained,
                "abstention_reason": trace.abstention_reason,
                "effective_mode": result.effective_mode,
                "model": result.model,
            },
        }
    )
    return payload


def persist_run_artifact(
    settings: object | None,
    *,
    result: ChatSessionResult,
    query: str,
    constraints: str,
    store_path: Path,
    store_metadata: Mapping[str, object] | None = None,
    profile_name: str = "",
    requested_route: str = "",
) -> Path:
    directory = artifact_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = build_run_id(result.decision_trace.query_hash)
    path = directory / f"{run_id}.json"
    payload = build_run_artifact(
        result=result,
        query=query,
        constraints=constraints,
        store_path=store_path,
        store_metadata=store_metadata,
        profile_name=profile_name,
        requested_route=requested_route,
        artifact_path=path,
    )
    save_json_document(path, payload)
    return path


def load_run_artifact(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"artifact at {path} is not a JSON object")
    return payload


def resolve_artifact_path(reference: str, *, settings: object | None = None) -> Path:
    path = Path(reference).expanduser()
    if path.exists():
        return path.resolve()
    candidate = artifact_dir(settings) / f"{reference}.json"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"artifact not found: {reference}")


def _safe_float(value: object) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def _safe_str(value: object) -> str:
    return str(value or "").strip()


def _safe_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _safe_str(value).lower()
    return text in {"1", "true", "yes", "on"}


def _format_mapping(mapping: Mapping[str, object]) -> str:
    parts = []
    for key, value in sorted(mapping.items()):
        parts.append(f"{key}={value}")
    return ", ".join(parts)


def _bullet_lines(title: str, items: Iterable[object]) -> list[str]:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return []
    return [f"{title}:"] + [f"  - {value}" for value in values]


def _count_items(value: object) -> int:
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    return 0


def diff_run_artifacts(
    left: Mapping[str, object],
    right: Mapping[str, object],
) -> dict[str, object]:
    left_summary = left.get("summary", {})
    right_summary = right.get("summary", {})
    if not isinstance(left_summary, Mapping):
        left_summary = {}
    if not isinstance(right_summary, Mapping):
        right_summary = {}
    left_trace = left.get("decision_trace", {})
    right_trace = right.get("decision_trace", {})
    if not isinstance(left_trace, Mapping):
        left_trace = {}
    if not isinstance(right_trace, Mapping):
        right_trace = {}
    left_input = left.get("input", {})
    right_input = right.get("input", {})
    if not isinstance(left_input, Mapping):
        left_input = {}
    if not isinstance(right_input, Mapping):
        right_input = {}

    return {
        "left_run_id": str(left.get("run_id", "")),
        "right_run_id": str(right.get("run_id", "")),
        "left_profile": _safe_str(left_input.get("profile")) or "default",
        "right_profile": _safe_str(right_input.get("profile")) or "default",
        "left_requested_route": _safe_str(left_input.get("requested_route")) or "auto",
        "right_requested_route": _safe_str(right_input.get("requested_route")) or "auto",
        "left_verdict": _safe_str(left_summary.get("verdict")) or "unknown",
        "right_verdict": _safe_str(right_summary.get("verdict")) or "unknown",
        "left_query_mode": _safe_str(left_summary.get("query_mode")) or "unknown",
        "right_query_mode": _safe_str(right_summary.get("query_mode")) or "unknown",
        "left_effective_mode": _safe_str(left_summary.get("effective_mode")) or "unknown",
        "right_effective_mode": _safe_str(right_summary.get("effective_mode")) or "unknown",
        "left_requires_human_review": _safe_bool(
            left_summary.get("requires_human_review")
        ),
        "right_requires_human_review": _safe_bool(
            right_summary.get("requires_human_review")
        ),
        "left_abstained": _safe_bool(left_summary.get("abstained")),
        "right_abstained": _safe_bool(right_summary.get("abstained")),
        "verdict_changed": left_summary.get("verdict") != right_summary.get("verdict"),
        "query_mode_changed": left_summary.get("query_mode")
        != right_summary.get("query_mode"),
        "effective_mode_changed": left_summary.get("effective_mode")
        != right_summary.get("effective_mode"),
        "profile_changed": left_input.get("profile") != right_input.get("profile"),
        "requested_route_changed": left_input.get("requested_route")
        != right_input.get("requested_route"),
        "human_review_changed": left_summary.get("requires_human_review")
        != right_summary.get("requires_human_review"),
        "abstained_changed": left_summary.get("abstained")
        != right_summary.get("abstained"),
        "citation_hit_rate_delta": round(
            _safe_float(right_summary.get("citation_hit_rate"))
            - _safe_float(left_summary.get("citation_hit_rate")),
            4,
        ),
        "answer_support_score_delta": round(
            _safe_float(right_summary.get("answer_support_score"))
            - _safe_float(left_summary.get("answer_support_score")),
            4,
        ),
        "latency_ms_delta": round(
            _safe_float(right_trace.get("end_to_end_ms", 0.0))
            - _safe_float(left_trace.get("end_to_end_ms", 0.0)),
            3,
        ),
    }


def render_run_artifact(payload: Mapping[str, object]) -> str:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}
    input_payload = payload.get("input", {})
    if not isinstance(input_payload, Mapping):
        input_payload = {}
    store_payload = payload.get("store", {})
    if not isinstance(store_payload, Mapping):
        store_payload = {}
    decision_payload = payload.get("decision", {})
    if not isinstance(decision_payload, Mapping):
        decision_payload = {}
    fused_payload = payload.get("fused", {})
    if not isinstance(fused_payload, Mapping):
        fused_payload = {}
    trace_payload = payload.get("decision_trace", {})
    if not isinstance(trace_payload, Mapping):
        trace_payload = {}

    lines = [
        "=" * 60,
        "RUN ARTIFACT",
        "=" * 60,
        f"Run ID: {payload.get('run_id', '')}",
        f"Created: {payload.get('created_at', '')}",
    ]
    artifact_path = _safe_str(payload.get("artifact_path"))
    if artifact_path:
        lines.append(f"Artifact Path: {artifact_path}")
    lines.extend(
        [
            "",
            "Input:",
            f"  Query: {input_payload.get('query', '')}",
            f"  Constraints: {input_payload.get('constraints', '') or 'None'}",
            f"  Profile: {input_payload.get('profile', '') or 'default'}",
            f"  Requested Route: {input_payload.get('requested_route', '') or 'auto'}",
            "",
            "Outcome:",
            f"  Verdict: {summary.get('verdict', '') or 'unknown'}",
            f"  Residual Risk: {decision_payload.get('residual_risk', '') or 'unknown'}",
            f"  Resolved Mode: {summary.get('query_mode', '') or 'unknown'}",
            (
                "  Human Review: "
                f"{'required' if summary.get('requires_human_review') else 'not required'}"
            ),
            f"  Abstained: {'yes' if summary.get('abstained') else 'no'}",
            f"  Abstention Reason: {summary.get('abstention_reason', '') or 'None'}",
            f"  Effective Mode: {summary.get('effective_mode', '') or 'unknown'}",
            f"  Model: {summary.get('model', '') or 'unknown'}",
            f"  Latency: {_safe_float(trace_payload.get('end_to_end_ms')):.3f} ms",
            "",
            "Grounding:",
            f"  Citation Hit Rate: {_safe_float(summary.get('citation_hit_rate')):.2f}",
            f"  Answer Support: {_safe_float(summary.get('answer_support_score')):.2f}",
            (
                "  Evidence Counts: "
                f"retrieved={_count_items(trace_payload.get('retrieved_evidence_ids'))} "
                f"used={_count_items(trace_payload.get('used_evidence_ids'))} "
                f"cited={_count_items(trace_payload.get('cited_evidence_ids'))} "
                f"blocked={_count_items(trace_payload.get('blocked_evidence_ids'))}"
            ),
        ]
    )
    lines.extend(["", "Routing:"])
    rationale = _safe_str(trace_payload.get("routing_rationale"))
    lines.append(f"  Rationale: {rationale or 'None'}")
    scores = trace_payload.get("routing_scores", {})
    if isinstance(scores, Mapping) and scores:
        lines.append(f"  Scores: {_format_mapping(scores)}")
    signals = trace_payload.get("routing_signals", [])
    if isinstance(signals, list) and signals:
        lines.append(f"  Signals: {', '.join(str(item) for item in signals)}")
    justification = _safe_str(decision_payload.get("justification"))
    if justification:
        lines.extend(["", "Decision:", f"  Justification: {justification}"])
    final_answer = _safe_str(fused_payload.get("final_answer"))
    if final_answer:
        lines.append(f"  Final Answer: {final_answer}")
    lines.extend(_bullet_lines("Next Steps", fused_payload.get("next_steps", [])))
    lines.extend(_bullet_lines("Risks", decision_payload.get("risks", [])))
    lines.extend(_bullet_lines("Mitigations", decision_payload.get("mitigations", [])))
    lines.extend(["", f"Store Path: {store_payload.get('path', '')}"])
    return "\n".join(lines)


def render_artifact_diff(diff: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "=" * 60,
            "ARTIFACT DIFF",
            "=" * 60,
            f"Left Run: {diff.get('left_run_id', '')}",
            f"Right Run: {diff.get('right_run_id', '')}",
            "",
            "Identity:",
            f"  Profile: {diff.get('left_profile', 'default')} -> {diff.get('right_profile', 'default')}",
            (
                "  Requested Route: "
                f"{diff.get('left_requested_route', 'auto')} -> {diff.get('right_requested_route', 'auto')}"
            ),
            "",
            "Outcome Changes:",
            f"  Verdict: {diff.get('left_verdict', 'unknown')} -> {diff.get('right_verdict', 'unknown')}",
            (
                "  Resolved Mode: "
                f"{diff.get('left_query_mode', 'unknown')} -> {diff.get('right_query_mode', 'unknown')}"
            ),
            (
                "  Effective Mode: "
                f"{diff.get('left_effective_mode', 'unknown')} -> {diff.get('right_effective_mode', 'unknown')}"
            ),
            (
                "  Human Review: "
                f"{'required' if diff.get('left_requires_human_review') else 'not required'}"
                " -> "
                f"{'required' if diff.get('right_requires_human_review') else 'not required'}"
            ),
            (
                "  Abstained: "
                f"{'yes' if diff.get('left_abstained') else 'no'}"
                " -> "
                f"{'yes' if diff.get('right_abstained') else 'no'}"
            ),
            "",
            "Change Flags:",
            f"  Profile Changed: {'yes' if diff.get('profile_changed') else 'no'}",
            (
                "  Requested Route Changed: "
                f"{'yes' if diff.get('requested_route_changed') else 'no'}"
            ),
            f"  Verdict Changed: {'yes' if diff.get('verdict_changed') else 'no'}",
            f"  Query Mode Changed: {'yes' if diff.get('query_mode_changed') else 'no'}",
            (
                "  Effective Mode Changed: "
                f"{'yes' if diff.get('effective_mode_changed') else 'no'}"
            ),
            (
                "  Human Review Changed: "
                f"{'yes' if diff.get('human_review_changed') else 'no'}"
            ),
            f"  Abstained Changed: {'yes' if diff.get('abstained_changed') else 'no'}",
            "",
            "Metric Deltas:",
            (
                "  Citation Hit Rate Delta: "
                f"{_safe_float(diff.get('citation_hit_rate_delta')):+.4f}"
            ),
            (
                "  Answer Support Delta: "
                f"{_safe_float(diff.get('answer_support_score_delta')):+.4f}"
            ),
            f"  Latency Delta: {_safe_float(diff.get('latency_ms_delta')):+.3f} ms",
        ]
    )


__all__ = [
    "artifact_dir",
    "build_run_artifact",
    "decision_payload",
    "diff_run_artifacts",
    "load_run_artifact",
    "persist_run_artifact",
    "render_artifact_diff",
    "render_run_artifact",
    "resolve_artifact_path",
]
