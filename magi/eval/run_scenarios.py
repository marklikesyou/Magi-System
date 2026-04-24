# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.eval.scenario_harness import (
    load_scenario_dataset,
    render_scenario_report,
    run_scenario_suite,
    write_scenario_report,
)


def _bounded_rate(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("thresholds must be between 0.0 and 1.0")
    return rate


def _threshold_failures(
    report, args: argparse.Namespace
) -> list[tuple[str, float, float, str]]:
    summary = report.summary
    min_thresholds = {
        "overall_score": args.min_overall_score,
        "verdict_accuracy": args.min_verdict_accuracy,
        "requirement_pass_rate": args.min_requirement_pass_rate,
        "retrieval_hit_rate": args.min_retrieval_hit_rate,
        "retrieval_top_source_accuracy": args.min_retrieval_top_source_accuracy,
        "retrieval_source_recall": args.min_retrieval_source_recall,
        "average_citation_hit_rate": args.min_average_citation_hit_rate,
        "average_answer_support_score": args.min_average_answer_support_score,
        "supported_answer_rate": args.min_supported_answer_rate,
    }
    max_thresholds = {
        "latency_p50_ms": args.max_p50_latency_ms,
        "latency_p95_ms": args.max_p95_latency_ms,
        "latency_max_ms": args.max_max_latency_ms,
        "average_estimated_cost_usd": args.max_average_cost_usd,
        "total_estimated_cost_usd": args.max_total_cost_usd,
    }
    failures: list[tuple[str, float, float, str]] = []
    for field, minimum in min_thresholds.items():
        if minimum is None:
            continue
        actual = float(getattr(summary, field))
        if actual < minimum:
            failures.append((field, actual, minimum, "minimum"))
    for field, maximum in max_thresholds.items():
        if maximum is None:
            continue
        actual = float(getattr(summary, field))
        if actual > maximum:
            failures.append((field, actual, maximum, "maximum"))
    return failures


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reusable end-to-end MAGI scenario evaluations."
    )
    parser.add_argument(
        "--cases",
        "--file",
        dest="cases",
        type=Path,
        required=True,
        help="Path to YAML dataset containing scenario cases.",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "stub", "live"),
        default="auto",
        help="Execution mode: auto uses live reasoning when credentials are available, otherwise stub.",
    )
    parser.add_argument(
        "--model",
        help="Optional model override for live runs.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        help="Optional path to write the full JSON report.",
    )
    parser.add_argument(
        "--min-overall-score",
        type=_bounded_rate,
        help="Fail if overall_score falls below this threshold.",
    )
    parser.add_argument(
        "--min-verdict-accuracy",
        type=_bounded_rate,
        help="Fail if verdict_accuracy falls below this threshold.",
    )
    parser.add_argument(
        "--min-requirement-pass-rate",
        type=_bounded_rate,
        help="Fail if requirement_pass_rate falls below this threshold.",
    )
    parser.add_argument(
        "--min-retrieval-hit-rate",
        type=_bounded_rate,
        help="Fail if retrieval_hit_rate falls below this threshold.",
    )
    parser.add_argument(
        "--min-retrieval-top-source-accuracy",
        type=_bounded_rate,
        help="Fail if retrieval_top_source_accuracy falls below this threshold.",
    )
    parser.add_argument(
        "--min-retrieval-source-recall",
        type=_bounded_rate,
        help="Fail if retrieval_source_recall falls below this threshold.",
    )
    parser.add_argument(
        "--min-average-citation-hit-rate",
        type=_bounded_rate,
        help="Fail if average_citation_hit_rate falls below this threshold.",
    )
    parser.add_argument(
        "--min-average-answer-support-score",
        type=_bounded_rate,
        help="Fail if average_answer_support_score falls below this threshold.",
    )
    parser.add_argument(
        "--min-supported-answer-rate",
        type=_bounded_rate,
        help="Fail if supported_answer_rate falls below this threshold.",
    )
    parser.add_argument(
        "--max-p50-latency-ms",
        type=float,
        help="Fail if latency_p50_ms exceeds this threshold.",
    )
    parser.add_argument(
        "--max-p95-latency-ms",
        type=float,
        help="Fail if latency_p95_ms exceeds this threshold.",
    )
    parser.add_argument(
        "--max-max-latency-ms",
        type=float,
        help="Fail if latency_max_ms exceeds this threshold.",
    )
    parser.add_argument(
        "--max-average-cost-usd",
        type=float,
        help="Fail if average_estimated_cost_usd exceeds this threshold.",
    )
    parser.add_argument(
        "--max-total-cost-usd",
        type=float,
        help="Fail if total_estimated_cost_usd exceeds this threshold.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_scenario_dataset(args.cases)
    force_stub = None
    if args.mode == "stub":
        force_stub = True
    elif args.mode == "live":
        force_stub = False

    report = run_scenario_suite(
        dataset,
        force_stub=force_stub,
        model=args.model,
        requested_mode=args.mode,
    )
    print(render_scenario_report(report))
    if args.report_out:
        write_scenario_report(report, args.report_out)
        print(f"report_saved\t{args.report_out}")
    failures = _threshold_failures(report, args)
    if failures:
        for field, actual, threshold, direction in failures:
            print(
                f"threshold_failed\t{field}\tactual={actual:.4f}\t{direction}={threshold:.4f}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
