# Prevent Irrelevant Evidence Approvals

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want unsupported questions to be revised or abstained instead of approved from unrelated evidence.

## Goal

Prevent MAGI from approving an answer when retrieved evidence is grounded but does not address the specific user question.

## Requirements

• Unsupported detail questions must not receive an approve verdict from unrelated retrieved evidence.
• Fact-check-style yes/no questions should only approve when the evidence directly verifies the claim.
• Existing grounded summary and pilot-decision approvals must keep working.

## Acceptance Criteria

• A Q4 procurement budget question against pilot-only evidence returns revise or abstain, not approve.
• A direct rollout-status fact check still returns a supported answer.
• Existing runtime and integration tests pass.

## Test Plan

• Add a runtime regression test for an unsupported procurement-budget question.
• Add a runtime regression test for a direct rollout-status yes/no fact check.
• Run `uv run pytest magi/tests/test_runtime.py -q`.
• Run `uv run pytest -q`.
• Run `uv run ruff check .` and `uv run mypy magi`.

## Notes

The smoke test showed `What is MAGI's Q4 procurement budget?` approving with pilot-guardrail evidence. The answer was grounded in retrieved text, but that text did not answer the requested budget detail.
