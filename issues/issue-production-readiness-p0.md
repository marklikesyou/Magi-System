# Production Readiness P0 Hardening

*Status:* Done
*Type:* AFK
*Blocked By:* None
*User Stories Addressed:* As a MAGI operator, I want production-blocking packaging, Docker, storage, and quality-gate failures fixed before customer-facing use.

## Goal

Make the release-blocking readiness failures locally verifiable and reduce production data and packaging risk.

## Requirements

• Built-in profiles, eval scenario files, retrieval fixtures, and Docker Compose assets must be included in packaged installs.
• Production wheels must not include `magi/tests`.
• Docker builds must exclude secrets, virtualenvs, caches, local vector stores, artifacts, datasets, and eval reports from build context.
• Default local vector-store and run-artifact paths must live outside the Python package tree unless explicitly configured.
• `--reset-store` must fail loudly or reset the database namespace when PostgreSQL storage is enabled.
• Scenario support metrics must treat unsupported abstentions/revisions generically, without scenario-specific IDs, source labels, canned prompts, or stopword-list fixes.
• Durable issue files must be trackable by git.

## Acceptance Criteria

• Built wheel metadata includes the required non-Python MAGI assets and excludes test modules.
• Default store and artifact directories resolve under `MAGI_DATA_DIR` or the user data directory, not `magi/`.
• `.dockerignore` prevents ignored runtime data and secrets from entering the Docker context.
• PostgreSQL-backed ingest reset clears the active logical namespace or reports a deterministic error.
• A paraphrased unsupported-SLA scenario counts as supported when MAGI correctly abstains/revises due to missing evidence.
• The production scenario gate still passes in stub mode.

## Test Plan

• Add packaging metadata tests for package data and excluded test modules.
• Add config/artifact tests for default data locations and `MAGI_DATA_DIR`.
• Add pgvector/storage tests for namespace reset behavior.
• Add Docker ignore coverage for runtime-data exclusions.
• Add scenario-harness support-rate coverage for generic unsupported-answer handling.
• Run targeted tests, then `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q`, `uv run ruff check .`, and `uv run mypy magi`.

## Notes

The production-readiness review dated 2026-05-05 is the source artifact for this slice. AI decision-making changes must remain generic semantic/structural behavior and must not add exact prompt, source, scenario-id, or stopword-list patches.
