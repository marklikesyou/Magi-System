# Fix Answer Grounding Smoke Failures

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want answers to cite the right source, preserve yes/no polarity, and avoid unsupported decision wording.

## Goal

Fix the smoke-test failures where MAGI approved wrong evidence, inverted a negative fact check, or added unsupported budget wording.

## Requirements

• A team-social agenda question must answer from team-social evidence, not rollout evidence.
• A fact check asking whether production incidents happened must answer no when the evidence says no incidents were recorded.
• Decision approval text must not mention an approved budget unless the cited evidence states budget support.

## Acceptance Criteria

• The team-social agenda query approves only when the answer cites the team-social source and includes the agenda details.
• The production-incidents fact check approves a negative answer when the cited evidence directly negates the claim.
• The internal-policy pilot decision answer does not invent approved-budget wording when budget is absent from cited evidence.
• Existing evidence-relevance behavior continues to pass.

## Test Plan

• Add integration coverage for team-social agenda evidence selection.
• Add runtime coverage for negative yes/no fact-check polarity.
• Add runtime coverage for decision approval text without budget evidence.
• Run the narrow failing tests before implementation.
• Run targeted tests after implementation, then `uv run pytest -q`, `uv run ruff check .`, and `uv run mypy magi`.

## Notes

Manual CLI smoke testing found three failures after the previous evidence-relevance fix: a team-social question cited rollout status, a production-incidents fact check answered yes despite negative evidence, and security-review output included unsupported approved-budget wording.
