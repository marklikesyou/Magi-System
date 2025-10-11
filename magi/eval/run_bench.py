from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.decision.aggregator import resolve_verdict_with_details
from magi.eval.dataset import (
    EvaluationDataset,
    build_persona_outputs,
    export_feature_log,
    load_dataset,
)
from magi.eval.metrics import accuracy


def evaluate_dataset(dataset: EvaluationDataset) -> Tuple[List[str], List[str], List[Tuple[str, str, str]], List[Dict[str, object]]]:
    predictions: List[str] = []
    gold: List[str] = []
    rows: List[Tuple[str, str, str]] = []
    feature_rows: List[Dict[str, object]] = []
    for case in dataset.cases:
        persona_outputs = build_persona_outputs(case)
        verdict, details = resolve_verdict_with_details(case.fused, case.personas, persona_outputs)
        predictions.append(verdict)
        gold.append(case.expected_verdict)
        rows.append((case.id, case.expected_verdict, verdict))
        feature_row = dict(details)
        feature_row["case_id"] = case.id
        feature_row["expected_verdict"] = case.expected_verdict
        feature_rows.append(feature_row)
    return predictions, gold, rows, feature_rows


def report(rows: List[Tuple[str, str, str]], score: float, total: int) -> None:
    print("case_id\tgold\tpredicted")
    for case_id, gold, predicted in rows:
        print(f"{case_id}\t{gold}\t{predicted}")
    print(f"\naccuracy\t{score:.2%}\ncount\t{total}")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MAGI decisions against a labeled dataset.")
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Path to YAML dataset containing evaluation cases.",
    )
    parser.add_argument(
        "--features-out",
        type=Path,
        help="Optional path to write decision feature log (JSONL).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.cases)
    predictions, gold, rows, features = evaluate_dataset(dataset)
    score = accuracy(predictions, gold)
    report(rows, score, len(dataset.cases))
    if args.features_out:
        export_feature_log(features, args.features_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
