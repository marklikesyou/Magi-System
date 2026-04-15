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


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reusable end-to-end MAGI scenario evaluations.")
    parser.add_argument(
        "--cases",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
