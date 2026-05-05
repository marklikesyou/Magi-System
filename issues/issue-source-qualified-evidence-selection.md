# Prefer Source-Qualified Evidence

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want questions that name a source or document title to cite that source when it contains relevant evidence.

## Goal

Ensure source-qualified informational queries prefer the matching source/document instead of a merely related document.

## Requirements

• Informational queries that include source-like terms must prefer evidence whose source label and text best match the query.
• The fix must remain generic and must not hardcode retrieval-corpus source names.
• Existing unsupported-detail abstention and distractor exclusion behavior must keep passing.

## Acceptance Criteria

• `Give me the key points from the MAGI overview.` cites the MAGI overview evidence, not the pilot brief.
• Existing routing, runtime, and evidence-relevance tests pass.

## Test Plan

• Add a runtime regression test for source-qualified key-point selection.
• Run the new failing test before implementation.
• Run targeted runtime tests, then the full stub test suite.
• Rerun realistic MAGI CLI smoke queries.

## Notes

The real-user smoke pass after `issue-gepa-prompt-context-hygiene` showed the source-qualified overview query selecting the pilot brief because evidence ranking considered text overlap before source-label overlap.

Implemented by applying a generic source-qualification ranking pass after semantic evidence ranking. Source labels now help choose between already-supporting evidence chunks without hardcoding any corpus source names.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_runtime.py::test_source_qualified_key_points_prefer_matching_source_label -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_routing.py magi/tests/test_runtime.py magi/tests/test_optimization.py magi/tests/test_magi_integration.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• Offline CLI smoke confirmed `Give me the key points from the MAGI overview.` cites `magi/eval/retrieval_corpus/magi_overview.txt`.
