# MAGI Acceptance Checklist

Use this checklist before treating a MAGI build as production accepted. Passing
unit tests is necessary but not sufficient; the live gauntlet manifest and its
verifier are the acceptance artifacts.

## Objective Mapping

| Requirement | Evidence to inspect |
| --- | --- |
| Melchior, Balthasar, and Casper names, schemas, responsibilities, and public outputs are preserved | `uv run pytest magi/tests/test_persona_contracts.py -q` |
| Responses-API-backed live runtime with strict structured outputs | `magi/core/clients.py`, `magi/tests/test_clients.py`, and a passed live gauntlet manifest |
| Parallel live persona execution | `magi/tests/test_runtime.py::test_live_personas_run_in_parallel_when_client_supports_it` |
| Trace IDs and execution spans | run artifact JSON plus `magi/tests/test_artifacts.py` and `magi/tests/test_magi_integration.py` |
| Deterministic offline parity | `MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q` |
| Calibrated approval, abstention, citation, and support thresholds | scenario reports plus `magi/eval/run_scenarios.py` thresholds |
| Stronger evidence planning, reranking, and citation verification | `retrieval_benchmark_report.json`, scenario reports: retrieval rates, citation hit rate, answer support score |
| Profile-aware answer rendering | `magi/tests/test_profiles.py`, `magi/tests/test_presentation.py`, and run artifacts |
| Replay, diff, and debug artifacts | `magi/tests/test_artifacts.py`, `magi/tests/test_cli_entrypoints.py`, MAGI run artifacts, `magi explain`, `magi replay`, `magi diff` |
| 100% production and live scenario gates | `production_report.json`, `live_report.json`, and `gauntlet_manifest.json` |
| >=95% pass on 1,000-case adversarial semantic suite across all categories | `adversarial_semantic_report.json`, `adversarial_semantic.yaml`, and manifest verifier |
| Zero uncited approvals, zero live fallbacks, no empty final answers | manifest verifier and report summaries |
| p95 stub latency <1s and p95 cached/replay latency <250ms | scenario report summaries and manifest verifier |
| Operator workflow documented | `README.md` and this checklist |

## Required Local Quality Gate

Run these on every candidate before the live gauntlet:

```bash
MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 uv run pytest -q
uv run ruff check .
uv run mypy magi
git diff --check
```

Run the deterministic production scenario gate:

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
  --max-total-cost-usd 0.0 \
  --max-uncited-approvals 0 \
  --max-empty-final-answers 0
```

Run the corpus-backed retrieval benchmark gate:

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

Run the deterministic live-scenario parity gate:

```bash
MAGI_FORCE_DSPY_STUB=1 MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_scenarios.py \
  --cases magi/eval/live_scenarios.yaml \
  --mode stub \
  --min-overall-score 1.0 \
  --min-verdict-accuracy 1.0 \
  --min-requirement-pass-rate 1.0 \
  --min-retrieval-hit-rate 1.0 \
  --min-retrieval-top-source-accuracy 1.0 \
  --min-retrieval-source-recall 1.0 \
  --min-average-citation-hit-rate 1.0 \
  --min-average-answer-support-score 0.20 \
  --min-supported-answer-rate 1.0 \
  --max-p95-latency-ms 1000 \
  --min-cached-replay-hit-rate 1.0 \
  --max-cached-p95-latency-ms 250 \
  --max-total-cost-usd 0.0 \
  --max-uncited-approvals 0 \
  --max-empty-final-answers 0
```

## Required Live Acceptance Gate

Configure a real OpenAI provider key first:

```bash
magi setup
```

Then run the full gauntlet:

```bash
MAGI_FORCE_HASH_EMBEDDER=1 \
uv run python magi/eval/run_gauntlet.py
```

The gauntlet must produce these files under `.magi/gauntlet/`:

- `production_report.json`
- `retrieval_benchmark_report.json`
- `live_report.json`
- `adversarial_semantic.yaml`
- `adversarial_semantic_report.json`
- `gauntlet_manifest.json`
- `acceptance_audit.json`

Verify the manifest before accepting the run:

```bash
uv run python magi/eval/verify_gauntlet_manifest.py \
  --manifest .magi/gauntlet/gauntlet_manifest.json

uv run python magi/eval/acceptance_audit.py \
  --manifest .magi/gauntlet/gauntlet_manifest.json \
  --out .magi/gauntlet/acceptance_audit.json
```

Acceptance requires:

- `gauntlet_manifest.json` has `metadata.status` set to `passed`.
- `magi/eval/verify_gauntlet_manifest.py` exits with status `0`.
- `acceptance_audit.json` has `metadata.status` set to `passed`.
- The `live_scenarios` criterion records `provider: openai_responses_api`.
- `retrieval_benchmark.summary.overall_score`, `retrieval_hit_rate`, `retrieval_top_source_accuracy`, `retrieval_mrr`, and `retrieval_source_recall` are all `1.0`.
- `live_provider.summary.effective_mode` is `live`.
- `live_provider.summary.live_fallback_count` is `0`.
- `semantic.summary.total_cases` is `1000`.
- `semantic.summary.pass_rate` is at least `0.95`.
- The semantic suite contains at least `950` unique natural-language queries.
- The semantic report contains the same 1,000 case IDs as the semantic suite.
- The semantic suite and semantic report cover `summary`, `extract`, `fact_check`, `recommend`, `decision`, `injection`, and `harmful`.

If provider credentials are absent or invalid, the goal is not accepted. The
gauntlet records that failure in the manifest under `provider_preflight`.
