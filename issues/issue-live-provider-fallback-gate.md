# Gate Live Provider Fallbacks

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want live production gates to prove provider-backed execution instead of silently passing on deterministic fallback.

## Goal

Make live provider failures visible in decision traces and make live scenario gates fail when deterministic fallback is used unexpectedly.

## Requirements

• Live-mode LLM call failures must be counted and exposed in run metadata and decision traces.
• Scenario reports must summarize fallback count across cases.
• Live scenario gates must fail by default when any live fallback occurs.
• Stub/offline scenario gates must continue to allow deterministic fallback behavior.
• Existing deterministic fallback behavior for normal chat resilience must remain available outside strict live gates.

## Acceptance Criteria

• A live-mode run with a configured client whose call fails reports a nonzero fallback count.
• A live scenario suite with provider call failures fails its quality gate by default.
• A live scenario suite can explicitly allow fallback when configured for diagnostic runs.
• Existing stub production gates continue to pass.

## Test Plan

• Add runtime/service coverage for fallback count metadata and trace fields.
• Add scenario harness/CLI coverage that live mode fails when fallback count is above the threshold.
• Run the new failing tests before implementation.
• Run targeted tests after implementation, then the full stub test suite, ruff, mypy, and live smoke where available.

## Notes

This follows the production-readiness review dated 2026-05-06. P1/P2 review findings were fixed separately in `issues/issue-review-p1-p2-production-readiness.md`.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_magi_integration.py::test_live_chat_session_reports_provider_fallback -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_scenario_harness.py::test_live_scenario_gate_fails_when_provider_fallback_occurs -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_run_scenarios_cli.py magi/tests/test_scenario_harness.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --max-p95-latency-ms 1000 --max-total-cost-usd 0.0`
• `MAGI_RUN_LIVE_SMOKE=1 MAGI_FORCE_DSPY_STUB=0 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_live_provider_smoke.py -q`
• `MAGI_FORCE_DSPY_STUB=0 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/live_scenarios.yaml --mode live --min-overall-score 0.80 --min-verdict-accuracy 0.80 --min-requirement-pass-rate 0.80 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-citation-hit-rate 1.0 --min-average-answer-support-score 0.20 --min-supported-answer-rate 1.0 --max-p95-latency-ms 60000 --max-average-cost-usd 0.02`

The live scenario gate reported `live_fallback_count 0`.
