# MAGI System

<p align="center">
  <img src="assets/magi-hero.svg" alt="MAGI terminal hero image with a self-destruct warning" width="720">
</p>

MAGI is a command line assistant for answering questions from your own documents.
It retrieves relevant evidence, asks three internal reviewer personas to reason over
the evidence, and returns a verdict with citations.

Use it for internal research, policy triage, incident review, security review, and
other workflows where answers should stay grounded in source material.

## requirements

- python 3.10 or newer
- `uv`
- an openai or google api key

## install

Install the user-level command:

```bash
curl -fsSL https://raw.githubusercontent.com/marklikesyou/Magi-System/master/scripts/install.sh | sh
```

Then configure a provider key:

```bash
magi setup
```

For local development from this repo:

```bash
uv sync --extra dev --extra openai --extra google
uv run magi setup
```

## quick start

Add documents:

```bash
magi ingest docs/briefing.txt
```

Ask a question:

```bash
magi ask "what risks should i consider?"
```

Use a built-in profile:

```bash
magi chat "should we deploy the pilot?" --profile security-review
```

Compare profiles:

```bash
magi compare "should we deploy the pilot?" --profiles security-review exec-brief
```

Inspect saved runs:

```bash
magi explain <run-id>
magi diff <run-a> <run-b>
```

## common commands

```bash
magi status
magi ingest path/to/file.txt
magi ask "summarize this"
magi chat "what does the policy require?" --json
magi profiles
magi profiles security-review
magi compare "what should we do?" --include-default --profiles security-review exec-brief
```

`magi ask` is an alias for `magi chat`.
`magi docs add` is an alias for `magi ingest`.
`magi runs show` is an alias for `magi explain`.

## data and config

`magi setup` writes provider keys to:

```text
~/.config/magi-system/.env
```

Project-local `.env` and `.env.local` files still work for development.

Default runtime data lives outside the package tree:

```text
~/.local/share/magi-system
```

Set these when you need explicit locations:

- `MAGI_DATA_DIR`: base directory for the local vector store and default artifacts
- `MAGI_RUN_ARTIFACT_DIR`: directory for replayable run artifacts
- `MAGI_DECISION_TRACE_DIR`: directory for structured decision records
- `MAGI_PROFILE_DIR`: directory for custom profile yaml files
- `DATABASE_URL`: use postgres with pgvector instead of the local json store

## offline and ci mode

Use deterministic local behavior for tests or development without provider calls:

```bash
MAGI_ALLOW_OFFLINE=1 \
MAGI_FORCE_DSPY_STUB=1 \
MAGI_FORCE_HASH_EMBEDDER=1 \
uv run pytest -q
```

## test

Run unit tests:

```bash
uv run pytest -q
```

Run lint and type checks:

```bash
uv run ruff check .
uv run mypy magi
```

Run the stub production gate:

```bash
MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_scenarios.py \
  --cases magi/eval/production_scenarios.yaml \
  --mode stub \
  --min-overall-score 1.0 \
  --min-verdict-accuracy 1.0 \
  --min-requirement-pass-rate 1.0 \
  --max-p95-latency-ms 1000 \
  --max-total-cost-usd 0.0
```

## project layout

- `magi/app`: cli commands and service orchestration
- `magi/core`: config, providers, embeddings, storage, retrieval, and safety
- `magi/dspy_programs`: personas, grounding, and runtime reasoning
- `magi/decision`: verdict aggregation and decision schemas
- `magi/eval`: scenario gates, benchmarks, and reports
- `magi/profiles`: built-in workflow profiles
