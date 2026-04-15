# MAGI Decision System

This repository scaffolds a MAGI-inspired multi-agent decision system using DSPy, OpenAI, and Google Gemini clients. The layout mirrors the plan you provided:

- `app/`: CLI entrypoint that runs the MAGI pipeline.
- `core/`: Shared configuration, LLM clients, safety helpers, RAG building blocks.
- `dspy_programs/`: Persona signatures, modules, and compilation helpers.
- `decision/`: Aggregation logic, schemas, and optional PyTorch calibrators.
- `data_pipeline/`: Ingestion, chunking, and embedding utilities for RAG.
- `eval/`: Task suites and metrics to track quality.
- `tests/`: Unit tests for aggregation, signatures, and safety heuristics.

Most modules are lightweight skeletons intended to be fleshed out with real implementations, embeddings, and telemetry once provider credentials are available.

## Getting started

1. Create and edit the `.env` file in the repository root with your API credentials.
2. Install dependencies: `uv sync --extra dev --extra openai` (add `--extra google` for Gemini support, `--extra torch` for calibrators).
3. Ingest documents and chat with them from the terminal:

```bash
python -m magi.app.cli ingest docs/briefing.pdf
python -m magi.app.cli chat "Should we deploy the latest patch?" --constraints "Budget <= 50k"
```

By default the system stays completely offline using a deterministic hashing embedder and a deterministic reasoning fallback. To activate the provider-backed stack, set your keys in `.env` and run commands with `MAGI_FORCE_DSPY_STUB=0` (use `MAGI_FORCE_HASH_EMBEDDER=1` if you ever need to fall back to hashing).

```bash
export MAGI_FORCE_DSPY_STUB=0
python -m magi.app.cli ingest docs/briefing.pdf
python -m magi.app.cli chat "Should we deploy the latest patch?" --constraints "Budget <= 50k"
```

### Testing

Run the lightweight unit suite anytime with:

```bash
pytest
```

### Evaluation workflow

- Provide your own dataset matching `magi/eval/cases.template.yaml` and run:
  - `python magi/eval/run_bench.py --cases /path/to/cases.yaml --features-out artifacts/features.jsonl`
  - `python magi/eval/train_decision_model.py --cases /path/to/cases.yaml --model-out magi/decision/model_weights.json`
- Run the end-to-end scenario harness against prompt-sensitive live scenarios:
  - `python magi/eval/run_scenarios.py --cases magi/eval/live_scenarios.yaml --mode auto --report-out artifacts/live_scenarios.json`
  - `python magi/eval/run_scenarios.py --cases /path/to/scenarios.yaml --mode live --model gpt-4o-mini-2024-07-18`
- Enforce the benchmark accuracy threshold in CI: `pytest magi/tests/test_decision_bench.py`
- See `magi/eval/cases.template.yaml` and `magi/eval/scenarios.template.yaml` for dataset format guidance.
