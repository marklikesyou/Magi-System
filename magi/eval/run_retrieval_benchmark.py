# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.eval.retrieval_benchmark import (
    load_retrieval_benchmark_dataset,
    render_retrieval_benchmark_report,
    run_retrieval_benchmark,
    write_retrieval_benchmark_report,
)


def _bounded_rate(value: str) -> float:
    rate = float(value)
    if not 0.0 <= rate <= 1.0:
        raise argparse.ArgumentTypeError("thresholds must be between 0.0 and 1.0")
    return rate


def _threshold_failures(
    report, args: argparse.Namespace
) -> list[tuple[str, float, float]]:
    summary = report.summary
    thresholds = {
        "overall_score": args.min_overall_score,
        "retrieval_hit_rate": args.min_retrieval_hit_rate,
        "retrieval_top_source_accuracy": args.min_retrieval_top_source_accuracy,
        "retrieval_mrr": args.min_retrieval_mrr,
        "retrieval_source_recall": args.min_retrieval_source_recall,
    }
    failures: list[tuple[str, float, float]] = []
    for field, minimum in thresholds.items():
        if minimum is None:
            continue
        actual = float(getattr(summary, field))
        if actual < minimum:
            failures.append((field, actual, minimum))
    return failures


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a corpus-backed MAGI retrieval benchmark."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Path to YAML dataset describing the retrieval benchmark corpus and cases.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="Optional logical store path to use for the benchmark corpus.",
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
        "--min-retrieval-mrr",
        type=_bounded_rate,
        help="Fail if retrieval_mrr falls below this threshold.",
    )
    parser.add_argument(
        "--min-retrieval-source-recall",
        type=_bounded_rate,
        help="Fail if retrieval_source_recall falls below this threshold.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_retrieval_benchmark_dataset(args.cases)
    report = run_retrieval_benchmark(
        dataset,
        args.cases,
        store_path=args.store,
    )
    print(render_retrieval_benchmark_report(report))
    if args.report_out:
        write_retrieval_benchmark_report(report, args.report_out)
        print(f"report_saved\t{args.report_out}")
    failures = _threshold_failures(report, args)
    if failures:
        for field, actual, minimum in failures:
            print(
                f"threshold_failed\t{field}\tactual={actual:.4f}\tminimum={minimum:.4f}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
