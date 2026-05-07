# Fix Paraphrased Decision and Negated Fact Checks

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want natural decision phrasing and negated evidence checks to produce answers aligned with the selected route and cited evidence.

## Goal

Fix human-like smoke failures where constrained decision paraphrases receive summary wording and source-framed fact checks invert negated evidence.

## Requirements

• A query routed as a decision because of constraints must synthesize a decision-style answer instead of falling back to summary wording.
• A fact check asking whether a source says a claim is true must answer no when the cited evidence directly negates that claim.
• A polite summary request must stay informational when it contains decision-domain words such as rollout status.
• The fix must use route-level decision context and claim-structure comparison, not prompt-string, source-name, scenario-id, exact-match, or stopword-table patches.
• Existing persona stance alignment, grounded approvals, unsupported-question revisions, and safety behavior must continue to pass.

## Acceptance Criteria

• `Could we sign off on the vendor renewal for next month?` with audit-logging constraints returns an approve decision answer that cites renewal controls and does not start with summary wording.
• `Is there any source saying customer escalations were logged?` returns an approve fact-check answer that starts with `No.` when the evidence says no customer escalations were logged.
• `Can you summarize the rollout status for me?` routes as summarize and returns status/summary wording, not decision approval wording.
• The three deterministic persona stances remain aligned with the final verdict for these prompts.
• The full stub test suite, lint, type checks, and production scenario gate pass.

## Test Plan

• Add runtime regression coverage for constrained paraphrased decision synthesis.
• Add runtime regression coverage for source-framed negated fact-check polarity.
• Add routing and runtime regression coverage for polite summary requests that contain decision-domain terms.
• Run the new tests before implementation and confirm they fail for the expected reasons.
• Run targeted runtime/routing tests after implementation.
• Run the full stub suite, `uv run ruff check .`, `uv run mypy magi`, and the production scenario gate.

## Notes

Manual smoke on 2026-05-07 found that constrained decision paraphrases selected `decision` at routing time but produced `Summary:` answers because downstream persona helpers re-routed from the query text alone. The same smoke found a source-framed fact check answered `Yes` to evidence whose core claim was negated.

Implemented by carrying the selected route mode into downstream persona synthesis, tightening decision evidence selection to require clustered decision-control support, comparing negated fact checks around the claim event structure, and calibrating routing so explicit summary intent outranks a single decision-domain word.

Verification passed:
• Red tests first failed for constrained paraphrased decision synthesis, source-framed negated fact-check polarity, and polite rollout summary routing.
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_routing.py::test_route_query_prioritizes_explicit_summary_intent_over_domain_word magi/tests/test_runtime.py::test_polite_summary_request_with_rollout_word_avoids_decision_approval magi/tests/test_runtime.py::test_source_framed_negative_fact_check_preserves_claim_polarity magi/tests/test_runtime.py::test_constrained_paraphrased_decision_uses_decision_synthesis -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest magi/tests/test_routing.py magi/tests/test_runtime.py magi/tests/test_scenario_harness.py -q`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`
• `uv run ruff check .`
• `uv run mypy magi`
• `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --min-retrieval-hit-rate 1.0 --min-retrieval-top-source-accuracy 1.0 --min-retrieval-source-recall 1.0 --min-average-answer-support-score 0.10 --max-p95-latency-ms 1000 --max-average-cost-usd 0.001 --max-total-cost-usd 0.0`
• Human-like smoke sweep passed for paraphrased signoff decisions, production rollout decisions, source-framed negated fact checks, unsupported details, retrieval-injection filtering, and harmful-input rejection.
