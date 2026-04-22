from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_report(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"report at {path} is not a JSON object")
    return payload


def summary_metrics(payload: Mapping[str, object]) -> dict[str, float]:
    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
            continue
        try:
            metrics[key] = float(str(value))
        except (TypeError, ValueError):
            continue
    return metrics


def compare_reports(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    baseline_metadata = baseline.get("metadata", {})
    candidate_metadata = candidate.get("metadata", {})
    if not isinstance(baseline_metadata, Mapping):
        baseline_metadata = {}
    if not isinstance(candidate_metadata, Mapping):
        candidate_metadata = {}
    left = summary_metrics(baseline)
    right = summary_metrics(candidate)
    shared = sorted(set(left).intersection(right))
    deltas = {
        key: round(right[key] - left[key], 4)
        for key in shared
    }
    regressions = {key: value for key, value in deltas.items() if value < 0.0}
    improvements = {key: value for key, value in deltas.items() if value > 0.0}
    return {
        "baseline_name": str(baseline_metadata.get("name", "")),
        "candidate_name": str(candidate_metadata.get("name", "")),
        "metrics": deltas,
        "regressions": regressions,
        "improvements": improvements,
    }


def render_report_comparison(payload: Mapping[str, object]) -> str:
    lines = []
    metrics = payload.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    for key in sorted(metrics):
        value = metrics[key]
        try:
            delta = float(value)
        except (TypeError, ValueError):
            continue
        lines.append(f"{key}\t{delta:+.4f}")
    return "\n".join(lines)


def failing_regressions(
    comparison: Mapping[str, object],
    thresholds: Mapping[str, float],
) -> list[tuple[str, float, float]]:
    metrics = comparison.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return []
    failures: list[tuple[str, float, float]] = []
    for key, minimum_delta in thresholds.items():
        try:
            actual = float(metrics.get(key, 0.0))
        except (TypeError, ValueError):
            actual = 0.0
        if actual < minimum_delta:
            failures.append((key, actual, minimum_delta))
    return failures


__all__ = [
    "compare_reports",
    "failing_regressions",
    "load_report",
    "render_report_comparison",
    "summary_metrics",
]
