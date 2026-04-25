# MAGI Decision System

This repository scaffolds a MAGI-inspired multi-agent decision system using DSPy, OpenAI, and Google Gemini clients. The layout mirrors the plan you provided:

- `app/`: CLI entrypoint that runs the MAGI pipeline.
- `core/`: Shared configuration, LLM clients, safety helpers, RAG building blocks.
- `dspy_programs/`: Persona signatures, modules, and compilation helpers.
- `decision/`: Aggregation logic, schemas, and optional PyTorch calibrators.
- `data_pipeline/`: Ingestion, chunking, and embedding utilities for RAG.
- `eval/`: Task suites and metrics to track quality.
- `tests/`: Unit tests for aggregation, signatures, and safety heuristics.

The current codebase is a working CLI-oriented prototype with offline fallback, provider-backed reasoning, grounding checks, and evaluation tooling. It still leaves room for stronger telemetry and production packaging, but it is no longer just a directory skeleton.

## Getting started

Install the user-level CLI:

```bash
curl -fsSL https://raw.githubusercontent.com/marklikesyou/Magi-System/main/scripts/install.sh | sh
```

Then configure an AI provider key once:

```bash
magi setup
```

For local development from a checkout:

1. Install dependencies: `uv sync --extra dev --extra openai --extra google`
2. Optional calibrator training: `uv sync --extra torch`
3. Configure a provider key: `uv run magi setup`
4. Ingest documents and chat with them from the terminal:

```bash
python -m magi.app.cli ingest docs/briefing.pdf
python -m magi.app.cli chat "Should we deploy the latest patch?" --constraints "Budget <= 50k"
python -m magi.app.cli chat "Should we deploy the latest patch?" --json
python -m magi.app.cli profiles
python -m magi.app.cli chat "Should we deploy the latest patch?" --profile security-review
python -m magi.app.cli compare "Should we deploy the latest patch?" --include-default --profiles security-review exec-brief
```

The native CLI command is also available through the package entry point. After
install and setup, run:

```bash
magi
```

`magi` opens the interactive shell on a real terminal. Without activating the
environment, use `uv run magi`. For a user-level command outside the repository,
run `uv tool install --force "magi-system[openai,google] @ git+https://github.com/marklikesyou/Magi-System.git@main"` once, then run `magi setup` and `magi` from any shell. For scripts or piped input, use `magi shell` to force shell mode.

MAGI requires an OpenAI or Google key before running user-facing ingest/chat workflows. `magi setup` writes the key to `~/.config/magi-system/.env`; project-local `.env` and `.env.local` files still override that user config during repository development. Use `MAGI_ALLOW_OFFLINE=1` with `MAGI_FORCE_DSPY_STUB=1` and `MAGI_FORCE_HASH_EMBEDDER=1` only for local offline development or CI.

The default local vector store persists entries as JSON and builds a NumPy matrix for exact cosine search at runtime. Set `DATABASE_URL` to switch to the PostgreSQL + pgvector backend for larger shared stores.

### Runtime controls

The most relevant environment variables for production-style runs are:

- `MAGI_PROVIDER_MAX_RETRIES` and `MAGI_PROVIDER_RETRY_INITIAL_DELAY` for provider retry policy.
- `MAGI_PROVIDER_REQUESTS_PER_MINUTE` for provider-side throttling.
- `MAGI_ENABLE_MODEL_ROUTING`, `MAGI_OPENAI_FAST_MODEL`, `MAGI_OPENAI_STRONG_MODEL`,
  and `MAGI_OPENAI_HIGH_STAKES_MODEL` for route-aware live OpenAI model selection.
- `MAGI_ENABLE_LIVE_PERSONAS` to opt into provider calls for each persona before fusion. It defaults to `false` so live runs use deterministic personas plus live fusion.
- `MAGI_ENABLE_RESPONDER_LLM` to opt into the extra live responder call after fusion. It defaults to `false` so live runs use the faster deterministic responder.
- `MAGI_DECISION_TRACE_DIR` to persist structured decision records from CLI runs.
- `MAGI_RUN_ARTIFACT_DIR` to persist replayable run artifacts from CLI runs.
- `MAGI_PROFILE_DIR` to expose workspace-local profile YAML files to the CLI.
- `MAGI_APPROVE_MIN_CITATION_HIT_RATE` and `MAGI_APPROVE_MIN_ANSWER_SUPPORT_SCORE` to gate `approve` on cited-evidence quality.
- `MAGI_REQUIRE_HUMAN_REVIEW_FOR_APPROVALS` to keep grounded approvals flagged for human review.

With the current defaults, unsupported approvals are downgraded to `revise`, and supported approvals are still marked as requiring human review.

### Testing

Run the lightweight unit suite anytime with:

```bash
uv run pytest -q
```

Run the static checks with:

```bash
uv run ruff check .
uv run mypy magi
```

### Evaluation workflow

- Provide your own dataset matching `magi/eval/cases.template.yaml` and run:
  - `python magi/eval/run_bench.py --cases /path/to/cases.yaml --features-out artifacts/features.jsonl`
  - `python magi/eval/train_decision_model.py --cases /path/to/cases.yaml --model-out magi/decision/model_weights.json`
- Run the end-to-end scenario harness against prompt-sensitive live scenarios:
  - `python magi/eval/run_scenarios.py --cases magi/eval/live_scenarios.yaml --mode auto --report-out artifacts/live_scenarios.json`
  - `python magi/eval/run_scenarios.py --cases /path/to/scenarios.yaml --mode live --model gpt-5-mini`
- Run stricter production gates with latency/cost thresholds:
  - `python magi/eval/run_scenarios.py --cases magi/eval/production_scenarios.yaml --mode stub --min-overall-score 1.0 --min-verdict-accuracy 1.0 --min-requirement-pass-rate 1.0 --max-p95-latency-ms 1000 --max-total-cost-usd 0.0`
- Enforce the benchmark accuracy threshold in CI: `pytest magi/tests/test_decision_bench.py`
- See `magi/eval/cases.template.yaml` and `magi/eval/scenarios.template.yaml` for dataset format guidance.

### Profiles and artifacts

- Inspect built-in profiles: `python -m magi.app.cli profiles`
- Open the native interactive shell: `magi`
- Show one profile in detail: `python -m magi.app.cli profiles incident-review`
- Compare multiple profiles on the same prompt: `python -m magi.app.cli compare "Should we deploy the pilot?" --profiles security-review exec-brief`
- Explain a saved run: `python -m magi.app.cli explain <run-id>`
- Replay a saved run through the current code: `python -m magi.app.cli replay <run-id>`
- Diff two saved runs: `python -m magi.app.cli diff <run-a> <run-b>`
