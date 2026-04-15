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

## Requirements
- Python 3.11 or newer (the project targets Python 3.13 in development).
- uv package manager.
- Optional: OpenAI or Google credentials when running live reasoning; OpenAI is required for provider-backed embeddings.

## Setup
1. Sync the pinned environment with uv: `uv sync --extra dev --extra openai`
2. Activate the environment: `source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows)
3. Copy `.env` and populate any required keys (see `magi/core/config.py` for supported variables).

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
```

### Toggling DSPy
- Default behavior uses structured provider outputs when OpenAI or Google credentials are available.
- Set `MAGI_FORCE_DSPY_STUB=1` to force the deterministic offline fallback without external LLM calls.
- Set `MAGI_FORCE_HASH_EMBEDDER=1` to use the deterministic hashing embedder instead of OpenAI embeddings.

## Testing
Run the full suite with:
```bash
pytest
```

Run the reusable scenario harness with:
```bash
python magi/eval/run_scenarios.py \
  --cases magi/eval/live_scenarios.yaml \
  --mode auto \
  --report-out artifacts/live_scenarios.json
```

## Project Structure
- `run_magi.py` entry point for one shot runs.
- `magi/app/cli.py` reusable CLI with ingest and chat commands.
- `magi/dspy_programs/` persona definitions, consensus logic, and DSPy wiring.
- `magi/decision/` verdict aggregation logic and optional learned calibrators.
- `magi/core/` embeddings, storage, retrieval, and utilities.
- `magi/eval/` reusable benchmark and scenario harness tooling.

## Notes
- Vector store artifacts live in `magi/storage/` and are ignored by git.
- Evaluation artifacts and decision model weights are also ignored to keep the repository lightweight.
- Example retrieval documents can be placed in `docs/`; everything in that directory is ignored by git so you can add PDFs or text files for local testing without affecting version control.
