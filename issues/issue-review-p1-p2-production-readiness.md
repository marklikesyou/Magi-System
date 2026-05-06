# Fix Review P1/P2 Production Readiness Findings

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want decision traces to identify actually cited evidence and package builds to ship clean metadata.

## Goal

Fix the accepted production-readiness review findings for audit trace evidence usage and package readme metadata.

## Requirements

• Decision traces must not report every safe retrieved chunk as used evidence.
• `used_evidence_ids` must reflect evidence actually cited in the final answer when citations are present.
• Package metadata must use the tracked root README and build without a missing-readme warning.
• Packaging tests must guard against regressing to empty or missing long-description metadata.

## Acceptance Criteria

• A chat answer that cites only one retrieved source reports only that source's document id in `used_evidence_ids`.
• Retrieved evidence remains available separately through `retrieved_evidence_ids`.
• `uv build --wheel` no longer warns that `magi/README.md` is missing.
• Built wheel metadata includes the README long description.

## Test Plan

• Add or update integration coverage for `used_evidence_ids` matching cited evidence.
• Add packaging coverage for wheel metadata containing README content.
• Run the new failing tests before implementation.
• Run targeted tests after implementation, then the full stub test suite, ruff, mypy, and `uv build --wheel`.

## Notes

This follows the code review dated 2026-05-06. The P0 live-mode fallback finding is intentionally out of scope for this issue.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_magi_integration.py::test_chat_session_excludes_social_distractor_from_pilot_decision -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_packaging.py::test_built_wheel_contains_runtime_assets_without_tests -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `uv build --wheel --out-dir /private/tmp/magi_review_build_20260506`
