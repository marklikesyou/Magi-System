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

OpenAI live mode uses the OpenAI Responses API through `openai>=2.0.0`.
MAGI sends persona and fusion calls as strict structured outputs, records trace
IDs with phase spans, and stores replayable run artifacts locally, not in the
provider. Live mode runs Melchior, Balthasar, and Casper through the provider by
default and executes them in parallel when the client supports concurrent calls.
Set `MAGI_ENABLE_LIVE_PERSONAS=false` to force deterministic persona parity while
keeping live fusion.

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

Run the full production acceptance gauntlet. This requires a configured OpenAI
key because it proves the Responses API-backed live runtime, live provider
scenarios, and a live structured semantic judge over the generated 1,000-case
adversarial suite:

```bash
MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_gauntlet.py
```

The gauntlet writes individual JSON reports plus a consolidated manifest under
`.magi/gauntlet/`. CI should treat `.magi/gauntlet/gauntlet_manifest.json`
`metadata.status == "passed"` as the acceptance artifact; failed preflight or
threshold checks are recorded in the same manifest.

Verify an existing gauntlet manifest before accepting it:

```bash
uv run python magi/eval/verify_gauntlet_manifest.py \
  --manifest .magi/gauntlet/gauntlet_manifest.json

uv run python magi/eval/acceptance_audit.py \
  --manifest .magi/gauntlet/gauntlet_manifest.json \
  --out .magi/gauntlet/acceptance_audit.json
```

The verifier checks that the live gate used the OpenAI Responses API path, that
linked reports pass every threshold, and that the semantic suite has at least
950 unique queries with the same 1,000 case IDs and category coverage in the
semantic report. It also requires the corpus-backed retrieval benchmark to pass
with perfect hit rate, top-source accuracy, MRR, and source recall.

The same gate is available through the installed CLI:

```bash
magi eval gauntlet
magi eval verify-gauntlet
magi eval audit
```

The full acceptance checklist lives in `magi/eval/ACCEPTANCE.md`.

Run the stub production gate:

```bash
MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_scenarios.py \
  --cases magi/eval/production_scenarios.yaml \
  --mode stub \
  --min-overall-score 1.0 \
  --min-verdict-accuracy 1.0 \
  --min-requirement-pass-rate 1.0 \
  --min-retrieval-hit-rate 1.0 \
  --min-retrieval-top-source-accuracy 1.0 \
  --min-retrieval-source-recall 1.0 \
  --min-average-answer-support-score 0.10 \
  --max-p95-latency-ms 1000 \
  --min-cached-replay-hit-rate 1.0 \
  --max-cached-p95-latency-ms 250 \
  --max-average-cost-usd 0.001 \
  --max-uncited-approvals 0 \
  --max-empty-final-answers 0 \
  --max-total-cost-usd 0.0
```

Run the retrieval benchmark gate:

```bash
MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_retrieval_benchmark.py \
  --cases magi/eval/retrieval_benchmark.yaml \
  --report-out retrieval_benchmark_report.json \
  --min-overall-score 1.0 \
  --min-retrieval-hit-rate 1.0 \
  --min-retrieval-top-source-accuracy 1.0 \
  --min-retrieval-mrr 1.0 \
  --min-retrieval-source-recall 1.0
```

Run the OpenAI live provider gate. This fails by default if any provider call
falls back to deterministic behavior:

```bash
MAGI_FORCE_DSPY_STUB=0 MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_scenarios.py \
  --cases magi/eval/live_scenarios.yaml \
  --mode live \
  --min-overall-score 1.0 \
  --min-verdict-accuracy 1.0 \
  --min-requirement-pass-rate 1.0 \
  --min-retrieval-hit-rate 1.0 \
  --min-retrieval-top-source-accuracy 1.0 \
  --min-retrieval-source-recall 1.0 \
  --min-average-citation-hit-rate 1.0 \
  --min-average-answer-support-score 0.20 \
  --min-supported-answer-rate 1.0 \
  --max-p95-latency-ms 60000 \
  --min-cached-replay-hit-rate 1.0 \
  --max-cached-p95-latency-ms 250 \
  --max-live-fallbacks 0 \
  --max-uncited-approvals 0 \
  --max-empty-final-answers 0 \
  --max-average-cost-usd 0.02
```

Build and run the 1,000-case adversarial semantic gate with a live structured
judge:

```bash
uv run python magi/eval/build_adversarial_semantic_suite.py \
  --out .magi/adversarial_semantic.yaml \
  --total 1000

MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/semantic_harness.py \
  --cases .magi/adversarial_semantic.yaml \
  --mode stub \
  --concurrency 8 \
  --min-pass-rate 0.95 \
  --max-p95-latency-ms 1000 \
  --max-uncited-approvals 0 \
  --max-empty-final-answers 0 \
  --max-live-fallbacks 0 \
  --report-out .magi/adversarial_semantic_report.json
```

## project layout

- `magi/app`: cli commands and service orchestration
- `magi/core`: config, providers, embeddings, storage, retrieval, and safety
- `magi/dspy_programs`: personas, grounding, and runtime reasoning
- `magi/decision`: verdict aggregation and decision schemas
- `magi/eval`: scenario gates, benchmarks, and reports
- `magi/profiles`: built-in workflow profiles
