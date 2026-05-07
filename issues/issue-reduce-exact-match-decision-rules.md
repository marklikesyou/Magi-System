# Reduce Exact-Match Decision Rules

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want persona decision-making to generalize from query structure and evidence support instead of brittle scenario words or exact prompt fragments.

## Goal

Refactor routing and deterministic persona support logic to remove broad exact-match scenario/domain word tables from decision-making while preserving observable MAGI behavior.

## Requirements

• Routing must not depend on long scenario/domain phrase tables for decision, recommendation, summary, extraction, or fact-check intent.
• Persona fallback decision support must not depend on broad scenario/domain word lists such as specific rollout, pilot, proposal, budget, or source-corpus terms.
• Remaining lexical checks must be narrow protocol/safety/output-shape checks or generic structural markers, not prompt-string, source-name, scenario-id, canned answer, or corpus-specific patches.
• Existing human-like routing, evidence relevance, safety, grounding, production gates, and persona stance alignment must continue to pass.

## Acceptance Criteria

• `magi/core/routing.py` no longer contains large exact-match intent pattern tables.
• `magi/dspy_programs/heuristic_signals.py` no longer contains large exact-match scenario/domain decision-control and recommendation-support tables.
• Human-like regression tests continue to cover paraphrased decision, summary, negated fact-check, unsupported-detail, injection, and harmful-input behavior.
• The full stub suite, lint, type checks, and production scenario gate pass.

## Test Plan

• Add or preserve tests proving paraphrased prompts work without source-name or prompt-string branches.
• Run targeted routing/runtime/scenario tests after refactor.
• Run the full stub suite, `uv run ruff check .`, `uv run mypy magi`, and the production scenario gate.
• Run a human-like smoke sweep covering decision, informational, fact-check, unsupported, injection, and harmful input.

## Notes

The previous issue fixed concrete smoke failures, but review found that too much exact-match vocabulary remained in routing and fallback persona signal code. This issue removes the broad phrase tables and keeps behavior driven by query form, selected route mode, evidence relevance, safety gates, and generic evidence-quality structure.

Implemented with `magi/core/semantic.py` as the shared lightweight semantic scorer. Routing, persona fallback signals, aggregation, RAG scoring, and eval support scoring now use compact semantic profiles plus generic query-form or protocol checks instead of broad exact-match scenario/domain phrase tables. The old `magi/core/text_signals.py` exact phrase helper was removed.

Added a local semantic sensitive-exfiltration intent check in `magi/core/safety.py` after smoke testing found that harmful credential-extraction phrasing could reach retrieval without a literal leaked secret value. Literal secret and prompt-injection regexes remain as narrow safety/protocol checks.

Removed scenario-specific pilot/rollout examples from live persona system prompts and replaced service high-stakes model routing keyword checks with semantic profile scoring.

Verification completed:

• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q` -> 243 passed, 1 skipped.
• `uv run ruff check .` -> passed.
• `uv run mypy magi` -> passed with existing untyped-body notes only.
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-answer-support-score 0.10 --max-p95-latency-ms 1000 --max-average-cost-usd 0.001 --max-total-cost-usd 0.0` -> 4/4 cases and 21/21 requirements passed.
• Human-like smoke sweep passed for paraphrased decision, production decision, source-framed negation, unsupported detail, summary injection filtering, and harmful credential-extraction input.
• Exact-match audit found no scenario-specific decision/routing vocabulary in the refactored core surfaces; remaining matches are safety/protocol markers, persona stance tags, negation grammar, and scenario expected/forbidden term evaluation checks.
