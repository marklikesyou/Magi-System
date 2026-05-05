# Hide Routing Debug From Chat Report

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI CLI user, I want normal chat output to show the answer and evidence without routing debug details.

## Goal

Remove routing rationale debug lines from user-facing chat reports while preserving structured routing metadata for artifacts and traces.

## Requirements

• Standard chat reports must not print routing rationale.
• Executive chat reports must not print route rationale when profile settings request it.
• Decision trace metadata and explain artifacts may continue to include routing diagnostics.

## Acceptance Criteria

• `format_chat_report` standard output omits `Routing Rationale`.
• `format_chat_report` executive output omits `Why This Route`.
• Existing presentation and CLI behavior tests pass.

## Test Plan

• Update presentation tests for the expected report output.
• Run `uv run pytest magi/tests/test_presentation.py -q`.
• Run the full stub test suite and static checks before committing.

## Notes

This is a user-facing presentation cleanup only. It does not remove routing rationale, scores, or signals from run metadata.

Implemented by removing routing rationale lines from standard and executive report renderers while leaving trace metadata untouched.
