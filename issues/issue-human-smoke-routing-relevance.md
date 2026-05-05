# Fix Human Smoke Routing and Evidence Relevance

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want natural prompts to use the right query mode, cite relevant evidence, and answer yes/no questions directly.

## Goal

Fix human-typed smoke failures where MAGI used decision approval wording for status/extraction questions, cited distractor evidence, or failed to answer a negative yes/no fact check directly.

## Requirements

• A rollout-status question must answer from rollout-status evidence, not pilot approval evidence.
• A guarded pilot decision must not cite team-social distractor evidence.
• A production-incidents yes/no question must answer no when the evidence says no incidents were recorded.
• The three persona stances must remain aligned with the final answer for these prompts.
• The fix must use general routing, evidence relevance, and polarity behavior instead of source-name or prompt-string exact matches.
• Runtime decision support and distractor detection must not depend on scenario-specific source labels, named domains, or canned approval wording.

## Acceptance Criteria

• `What is the MAGI rollout status right now?` returns an approve answer that cites rollout-status evidence and does not use bounded-pilot approval wording.
• `Should we pilot MAGI for internal policy triage next month?` cites pilot/control evidence and excludes team-social evidence from the final answer.
• `Did production incidents happen during the latest review window?` returns an approve answer that starts with or clearly contains `No` and cites rollout-status evidence.
• Existing grounding, production scenario, runtime, and integration tests pass.

## Test Plan

• Add regression coverage for natural rollout-status routing and answer wording.
• Add regression coverage that guarded pilot decisions exclude social distractors.
• Add regression coverage for negative production-incident yes/no wording.
• Add regression coverage with paraphrased prompts and an unnamed distractor source to guard against exact-match fixes.
• Remove remaining runtime source-name and scenario-domain exact matches while preserving observable routing and grounding behavior.
• Run the narrow regression tests before implementation and confirm they fail for the expected reasons.
• Run targeted tests after implementation, then `uv run pytest -q`, `uv run ruff check .`, and `uv run mypy magi`.

## Notes

An offline CLI smoke batch against `magi/eval/retrieval_corpus/*.txt` processed 12 prompts but exposed three answer-quality failures: status prompts used decision approval wording, pilot decisions cited `team_social`, and incident yes/no prompts summarized evidence instead of directly answering no.

Follow-up refactor removed the large exact-match heuristic lists from `magi/dspy_programs/runtime.py` and replaced the bulky grounding stopword tables with a compact low-signal-token predicate in `magi/dspy_programs/grounding.py`. Remaining lexical vocabulary is isolated in `magi/dspy_programs/heuristic_signals.py` as generic offline fallback signal categories, not scenario-source or prompt-specific branches.
