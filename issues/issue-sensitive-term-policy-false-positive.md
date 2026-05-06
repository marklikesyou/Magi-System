# Narrow Sensitive-Term Safety False Positives

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI policy or security-review user, I want to ask benign questions about password, token, API-key, or confidential-data policies without MAGI treating the topic name itself as leaked secret material.

## Goal

Distinguish bare sensitive-topic words from actual leaked credentials or unsafe secret-exfiltration intent.

## Requirements

• Benign policy questions such as `What does the password policy require?` must not be blocked solely because they contain the word `password`.
• Actual secret-looking strings such as `password=123`, `api_key: ...`, or token assignments must continue to be flagged and blocked.
• Harmful requests for credential theft or bypassing controls must continue to reject.
• The fix must remain generic and must not special-case a single prompt string.

## Acceptance Criteria

• A benign password-policy question reaches retrieval/grounding and returns an evidence-based answer or abstention instead of a safety rephrase response.
• A query containing an explicit credential assignment is blocked.
• A malicious credential-exfiltration request is rejected.
• Existing safety, runtime, and production scenario tests pass.

## Test Plan

• Add safety unit coverage for bare sensitive-topic words versus credential assignment patterns.
• Add integration coverage for a benign password-policy query.
• Run `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py magi/tests/test_magi_integration.py -q`.
• Run the full stub suite, ruff, mypy, and production scenario gate.

## Notes

Review on 2026-05-06 found `detect_sensitive_leak` flags any occurrence of default banned markers including `password`, `api_key`, `token`, and `confidential`. A CLI smoke query, `What does the password policy require?`, returned `safety_outcome=input_blocked` and the generic rephrase response, which is too broad for MAGI's documented policy-triage and security-review use cases.

Implemented by treating bare sensitive-topic words as allowable policy/security topics while continuing to flag credential assignments, SSN-like values, credit-card-like values, and malicious credential-exfiltration requests.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py::test_detect_sensitive_leak_allows_sensitive_policy_topics magi/tests/test_magi_integration.py::test_chat_session_blocks_sensitive_input_before_retrieval magi/tests/test_magi_integration.py::test_chat_session_allows_benign_password_policy_question -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py magi/tests/test_runtime.py magi/tests/test_magi_integration.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-answer-support-score 0.10 --max-p95-latency-ms 1000 --max-average-cost-usd 0.001 --max-total-cost-usd 0.0`
• `MAGI_FORCE_DSPY_STUB=0 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/live_scenarios.yaml --mode live --min-overall-score 0.80 --min-verdict-accuracy 0.80 --min-requirement-pass-rate 0.80 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-citation-hit-rate 1.0 --min-average-answer-support-score 0.20 --min-supported-answer-rate 1.0 --max-p95-latency-ms 60000 --max-average-cost-usd 0.02`
