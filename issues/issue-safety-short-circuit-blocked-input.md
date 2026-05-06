# Short-Circuit Blocked Input Before Retrieval

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want blocked sensitive or unsafe user input to stop before retrieval, embedding, persona, or responder work so that the safety gate is a real pre-execution boundary.

## Goal

Prevent input-blocked requests from invoking retrieval or provider-backed embedding/model work.

## Requirements

• When input safety marks a query as blocked, MAGI must return the blocked-input response without calling the retriever.
• Blocked sensitive input must not be sent to embedding providers through retrieval.
• Decision traces for input-blocked requests must not report retrieved evidence or approving persona stances from unrelated safe evidence.
• Harmful-request rejection behavior must continue to pass.

## Acceptance Criteria

• A blocked input with a fake retriever that raises if called returns a reject or revise response without raising.
• A blocked input trace has `safety_outcome` set to `input_blocked` and empty retrieved, used, and cited evidence ids.
• Existing retrieval-injection tests still prove unsafe retrieved content is blocked when the input itself is safe.

## Test Plan

• Add runtime or integration coverage proving input-blocked queries do not call retrieval.
• Add trace coverage for empty retrieval fields on blocked input.
• Run `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_runtime.py magi/tests/test_magi_integration.py -q`.
• Run the full stub suite, ruff, mypy, and the production scenario gate.

## Notes

Review on 2026-05-06 found `MagiProgram._retrieve_with_safety` analyzes input safety but still calls `_safe_retrieve` before `forward` checks `is_blocked(input_report)`. In live/default configurations this can send a blocked query to the embedder through retrieval before returning the safety response.

Implemented by returning immediately from input safety analysis when the input is blocked, before invoking the retriever.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py::test_detect_sensitive_leak_allows_sensitive_policy_topics magi/tests/test_magi_integration.py::test_chat_session_blocks_sensitive_input_before_retrieval magi/tests/test_magi_integration.py::test_chat_session_allows_benign_password_policy_question -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py magi/tests/test_runtime.py magi/tests/test_magi_integration.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-answer-support-score 0.10 --max-p95-latency-ms 1000 --max-average-cost-usd 0.001 --max-total-cost-usd 0.0`
• `MAGI_FORCE_DSPY_STUB=0 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/live_scenarios.yaml --mode live --min-overall-score 0.80 --min-verdict-accuracy 0.80 --min-requirement-pass-rate 0.80 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-citation-hit-rate 1.0 --min-average-answer-support-score 0.20 --min-supported-answer-rate 1.0 --max-p95-latency-ms 60000 --max-average-cost-usd 0.02`
