# Fix Unsupported Informational Smoke Approvals

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want unsupported natural-language questions to abstain or request better evidence instead of approving weakly related retrieved text.

## Goal

Prevent unsupported informational smoke prompts from approving unrelated or weakly related evidence.

## Requirements

• Informational questions whose key requested terms are not supported by retrieved evidence must revise or abstain instead of approve.
• Short alphanumeric identifiers such as quarter labels must count as specificity signals even when they are too short for normal semantic support terms.
• Benign sensitive-topic policy questions must still pass input safety, but must not approve unrelated evidence when the corpus lacks the requested policy.
• The fix must remain generic and must not add prompt-specific, scenario-specific, source-name, variable-name, exact-match, or stopword-table rules.

## Acceptance Criteria

• `What is MAGI Q4 procurement budget?` against the bundled retrieval corpus returns revise or abstain, not approve.
• `What does the password policy require?` against the bundled retrieval corpus returns revise or abstain, not approve, while safety outcome remains passed.
• Existing positive evidence cases, including real password-policy evidence and source-qualified overview summaries, continue to approve.
• Existing runtime, integration, scenario, lint, and type checks pass.

## Test Plan

• Add runtime coverage for short-identifier unsupported detail wording without possessive punctuation.
• Add integration coverage for a benign policy-topic query with no supporting policy evidence.
• Run the new tests before implementation and confirm they fail for the expected reasons.
• Run targeted runtime and integration tests after implementation.
• Run the full stub suite, ruff, mypy, and the production scenario gate.

## Notes

This follows the production-readiness review on 2026-05-06. The implementation must improve generic evidence-support thresholds and query-specificity detection without adding exact prompt checks, retrieval-corpus source labels, scenario IDs, or stopword lists.

Implemented with a generic strict query-term coverage path for compact field-style informational questions and raw alphanumeric identifier questions. No prompt-specific, source-specific, scenario-specific, or stopword-list rules were added.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_runtime.py::test_short_identifier_detail_question_revises_without_possessive_punctuation -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_magi_integration.py::test_chat_session_abstains_when_policy_topic_lacks_supporting_policy_evidence -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_runtime.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_magi_integration.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_safety.py magi/tests/test_routing.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-answer-support-score 0.10 --max-p95-latency-ms 1000 --max-average-cost-usd 0.001 --max-total-cost-usd 0.0`
• Offline CLI smoke replay confirmed `What is MAGI Q4 procurement budget?` and `What does the password policy require?` both abstain with `safety_outcome=passed` and no cited unrelated sources.
