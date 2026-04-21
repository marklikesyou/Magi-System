from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import magi.core.storage as storage
from magi.core.embeddings import HashingEmbedder
from magi.eval.retrieval_benchmark import (
    load_retrieval_benchmark_dataset,
    render_retrieval_benchmark_report,
    run_retrieval_benchmark,
)


def test_sample_retrieval_benchmark_passes(monkeypatch, tmp_path: Path) -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1] / "eval" / "retrieval_benchmark.yaml"
    )
    dataset = load_retrieval_benchmark_dataset(dataset_path)
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(vector_db_url=""))

    report = run_retrieval_benchmark(
        dataset,
        dataset_path,
        store_path=tmp_path / "benchmark_store.json",
        embedder=HashingEmbedder(dimension=128, bucket_size=2),
    )

    assert report.metadata["suite_type"] == "retrieval_benchmark"
    assert report.summary.total_cases == 3
    assert report.summary.passed_cases == 3
    assert report.summary.overall_score == 1.0
    assert report.summary.retrieval_hit_rate == 1.0
    assert report.summary.retrieval_top_source_accuracy == 1.0
    assert report.summary.retrieval_mrr == 1.0
    assert report.summary.retrieval_source_recall == 1.0
    assert report.summary.ingested_document_count == 4
    assert report.summary.ingested_chunk_count >= 4
    assert report.cases[0].retrieved_sources[0] == "magi_overview"
    assert report.cases[1].retrieved_sources[0] == "pilot_brief"
    assert report.cases[2].retrieved_sources[0] == "rollout_status"


def test_render_retrieval_benchmark_report_includes_summary(monkeypatch, tmp_path: Path) -> None:
    dataset_path = (
        Path(__file__).resolve().parents[1] / "eval" / "retrieval_benchmark.yaml"
    )
    dataset = load_retrieval_benchmark_dataset(dataset_path)
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(vector_db_url=""))

    report = run_retrieval_benchmark(
        dataset,
        dataset_path,
        store_path=tmp_path / "benchmark_store.json",
        embedder=HashingEmbedder(dimension=128, bucket_size=2),
    )

    rendered = render_retrieval_benchmark_report(report)

    assert "suite_type\tretrieval_benchmark" in rendered
    assert "retrieval_hit_rate\t100.00%" in rendered
    assert "store_backend\tInMemoryVectorStore" in rendered
    assert "retrieve_magi_overview" in rendered
