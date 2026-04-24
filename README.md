# MAGI System

<p align="center">
  <img src="assets/magi-hero.svg" alt="MAGI terminal hero image with a self-destruct warning" width="720">
</p>

## Overview
MAGI is a multi persona reasoning engine for assessing user requests against an evidence base. It retrieves context from a local vector store, convenes three specialized personas (scientist, pragmatist, guardian), negotiates consensus, and returns a final verdict along with an explanation.

## Features
- Command line helper (`run_magi.py`) that supports optional document ingestion before answering a query.
- Retrieval augmented generation backed by DSPy personas or a lightweight stub when DSPy is unavailable.
- Consensus loop that enforces multi persona agreement before issuing a verdict.
- Persistent vector store on disk with hashing based embeddings for offline mode.
- Query routing across summary, extraction, fact-check, recommendation, and decision modes.
- Run artifacts with replayable `explain`, `replay`, and `diff` CLI workflows.
- Profile comparison workflow for running the same prompt across multiple CLI profiles.
- Built-in domain profiles for security review, policy triage, incident review, executive briefs, and vendor review.

## Requirements
- Python 3.10 or newer (CI runs on Python 3.10 and 3.13).
- uv package manager.
- Optional: OpenAI or Google credentials when running live reasoning; OpenAI is required for provider-backed embeddings.

## Setup
1. Sync the pinned environment with uv: `uv sync --extra dev`
2. Add provider extras only if you need live model calls:
   - OpenAI: `uv sync --extra openai`
   - Gemini: `uv sync --extra google`
   - Optional calibrator training: `uv sync --extra torch`
3. Copy the example environment file and edit only the values you need: `cp .env.example .env.local`
4. Activate the environment: `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)

The project runs offline by default when provider credentials are unset. Add keys only when you want live reasoning or provider-backed embeddings.

## Usage
### Ingest documents and ask a question
```bash
python run_magi.py \
  --docs path/to/doc1.txt path/to/doc2.pdf \
  --query "Explain the safety implications of section 4"
```

Interactive mode is also available by invoking `python run_magi.py` with no arguments. The CLI will prompt for document paths, a query, and optional persona constraints.

### CLI subcommands
```bash
python -m magi.app.cli ingest path/to/doc.txt
python -m magi.app.cli chat "What risks should I consider?"
python -m magi.app.cli chat "What risks should I consider?" --json
python -m magi.app.cli profiles
python -m magi.app.cli chat "Should we deploy the pilot?" --profile security-review
python -m magi.app.cli compare "Should we deploy the pilot?" --include-default --profiles security-review exec-brief
python -m magi.app.cli explain <run-id>
python -m magi.app.cli diff <run-a> <run-b>
```

After installation, the native `magi` command exposes the same subcommands. Run
`magi` with no subcommand to open the interactive shell; from there, enter a
command such as `profiles security-review` or type a plain question to run
`chat`.

### Profiles
Use built-in profiles to bias retrieval, routing, and decision thresholds toward real workflows:

```bash
python -m magi.app.cli profiles
python -m magi.app.cli profiles policy-triage
python -m magi.app.cli chat "What does the handbook actually require?" --profile policy-triage
python -m magi.app.cli chat "Summarize the incident for leadership." --profile incident-review
python -m magi.app.cli chat "Give me the executive takeaway." --profile exec-brief
python -m magi.app.cli compare "Should we deploy the pilot?" --profiles security-review vendor-review --full
```

### Toggling DSPy
- Default behavior uses structured provider outputs when OpenAI or Google credentials are available.
- Set `MAGI_FORCE_DSPY_STUB=1` to force the deterministic offline fallback without external LLM calls.
- Set `MAGI_FORCE_HASH_EMBEDDER=1` to use the deterministic hashing embedder instead of OpenAI embeddings.

### Runtime Controls
- `MAGI_PROVIDER_MAX_RETRIES`: retry budget for provider calls. Default `3`.
- `MAGI_PROVIDER_RETRY_INITIAL_DELAY`: initial retry backoff in seconds. Default `1.0`.
- `MAGI_PROVIDER_REQUESTS_PER_MINUTE`: provider-side rate limit. Default `0` disables throttling.
- `MAGI_DECISION_TRACE_DIR`: optional directory for persisted decision records when using the CLI.
- `MAGI_RUN_ARTIFACT_DIR`: optional directory for persisted run artifacts used by `explain`, `replay`, and `diff`.
- `MAGI_PROFILE_DIR`: optional directory for workspace-local profile YAML files.
- `MAGI_ENABLE_MODEL_ROUTING`: when `true`, live OpenAI runs route harder decision, recommendation, and fact-check prompts to the strong/high-stakes model. Default `true`.
- `MAGI_OPENAI_FAST_MODEL`: model for summary/extraction-style live OpenAI runs. Default `gpt-5-mini`.
- `MAGI_OPENAI_STRONG_MODEL`: model for decision, recommendation, and fact-check runs. Default `gpt-5.2`.
- `MAGI_OPENAI_HIGH_STAKES_MODEL`: model for high-stakes decision/recommendation/fact-check runs. Default `gpt-5.2`.
- `MAGI_ENABLE_LIVE_PERSONAS`: when `true`, live runs call the provider for each persona before fusion. Default `false`; deterministic personas plus live fusion are much faster.
- `MAGI_ENABLE_RESPONDER_LLM`: when `true`, live runs make a final responder LLM call after fusion. Default `false`; the deterministic responder is faster and cheaper.
- `MAGI_APPROVE_MIN_CITATION_HIT_RATE`: minimum valid citation hit rate required for `approve`. Default `1.0`.
- `MAGI_APPROVE_MIN_ANSWER_SUPPORT_SCORE`: minimum lexical evidence-overlap score required for `approve`. Default `0.2`.
- `MAGI_REQUIRE_HUMAN_REVIEW_FOR_APPROVALS`: when `true`, grounded approvals are still marked for human review. Default `true`.

When an `approve` answer fails the citation or support thresholds, the production path downgrades it to `revise`. When an `approve` answer passes those thresholds, the default behavior is still to mark it as requiring human review.

## Testing
Run the full suite with:
```bash
uv run pytest -q
```

Run static checks with:
```bash
uv run ruff check .
uv run mypy magi
```

Run the reusable scenario harness with:
```bash
python magi/eval/run_scenarios.py \
  --cases magi/eval/live_scenarios.yaml \
  --mode auto \
  --report-out artifacts/live_scenarios.json
```

Run the production gate suite with quality, latency, and cost thresholds:
```bash
python magi/eval/run_scenarios.py \
  --cases magi/eval/production_scenarios.yaml \
  --mode stub \
  --min-overall-score 1.0 \
  --min-verdict-accuracy 1.0 \
  --min-requirement-pass-rate 1.0 \
  --max-p95-latency-ms 1000 \
  --max-total-cost-usd 0.0
```

## Project Structure
- `run_magi.py` entry point for one shot runs.
- `magi/app/cli.py` reusable CLI with ingest, chat, profiles, artifact, batch, and eval commands.
- `magi/dspy_programs/` persona definitions, consensus logic, and DSPy wiring.
- `magi/decision/` verdict aggregation logic and optional learned calibrators.
- `magi/core/` embeddings, storage, retrieval, and utilities.
- `magi/eval/` reusable benchmark and scenario harness tooling.

## Notes
- Vector store artifacts live in `magi/storage/` and are ignored by git.
- Run artifacts live in `magi/artifacts/` by default and are ignored by git.
- Evaluation artifacts and decision model weights are also ignored to keep the repository lightweight.
- Example retrieval documents can be placed in `docs/`; everything in that directory is ignored by git so you can add PDFs or text files for local testing without affecting version control.
