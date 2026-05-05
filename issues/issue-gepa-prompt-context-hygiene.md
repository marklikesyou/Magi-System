# Keep Routing Heuristics Out of Optimizer Context

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want DSPy/GEPA-facing prompts to learn from task behavior and evidence, not from exact-match routing or stopword internals.

## Goal

Prevent exact matched routing patterns, route score details, and token-filter internals from being included in model-facing prompt context while preserving routing metadata for debugging.

## Requirements

• Model-facing prompt context must include the selected route mode and answer style only.
• Model-facing prompt context must not include exact matched routing phrases, routing score maps, routing signal lists, or stopword/token-filter tables.
• Routing scores and signals must remain available in run metadata for diagnostics.
• The optimization helper should expose a GEPA option when DSPy provides `dspy.GEPA`.
• Existing routing, grounding, and runtime behavior must continue to pass.

## Acceptance Criteria

• `mode_prompt_brief` does not include exact matched patterns or route score/debug signal details.
• Live MAGI prompts sent through the structured runner do not contain routing scores or routing signals.
• `create_optimization_pipeline(..., optimization_method="gepa")` routes to a GEPA compile helper.
• Multiple realistic MAGI CLI queries run successfully in offline/stub mode after the change.

## Test Plan

• Add routing coverage for sanitized model-facing route context.
• Add runtime coverage with a capturing client to verify live prompts omit route debug data.
• Add optimization coverage for the GEPA method dispatch using a fake DSPy module.
• Run the new failing tests before implementation.
• Run targeted tests after implementation, then `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`.
• Run multiple realistic `magi chat` queries in stub mode against the retrieval corpus.

## Notes

Existing completed issues require routing and evidence fixes to remain generic, without source-name, scenario-id, prompt-string, or stopword-list patches. This issue keeps deterministic lexical logic outside the prompt/optimizer surface rather than removing diagnostic metadata or routing behavior.

Implemented by sanitizing `mode_prompt_brief` so model-facing prompts receive only route mode and answer style, while run metadata still records routing rationale, scores, and signals for diagnostics. Added `optimization_method="gepa"` dispatch through a `dspy.GEPA` compile helper.

Verification passed:
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_routing.py magi/tests/test_runtime.py magi/tests/test_optimization.py magi/tests/test_magi_integration.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• Offline CLI smoke queries against `magi/eval/retrieval_corpus/*.txt`.
