from __future__ import annotations
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Sequence, cast

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError, model_validator

from magi.core.config import get_settings
from magi.core.embeddings import build_embedder
from magi.core.rag import RagRetriever
from magi.core.storage import describe_store_destination, initialize_store, persist_store
from magi.core.vectorstore import RetrievedChunk, VectorEntry
from magi.data_pipeline.chunkers import sliding_window_chunk
from magi.data_pipeline.embed import embed_chunks
from magi.data_pipeline.ingest import ingest_paths


def _normalize_text(text: str) -> str:
    return " ".join(text.replace("’", "'").strip().lower().split())


class RetrievalCorpusDocument(BaseModel):
    id: str
    path: str

    @model_validator(mode="after")
    def _normalize(self) -> "RetrievalCorpusDocument":
        doc_id = self.id.strip()
        doc_path = self.path.strip()
        if not doc_id:
            raise ValueError("corpus document id must not be empty")
        if not doc_path:
            raise ValueError("corpus document path must not be empty")
        object.__setattr__(self, "id", doc_id)
        object.__setattr__(self, "path", doc_path)
        return self


class RetrievalCorpusConfig(BaseModel):
    root: str = ""
    chunk_size: int = Field(default=1500, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    documents: List[RetrievalCorpusDocument] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "RetrievalCorpusConfig":
        if not self.documents:
            raise ValueError("retrieval benchmark corpus contains no documents")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        object.__setattr__(self, "root", self.root.strip())
        return self


class RetrievalBenchmarkCase(BaseModel):
    id: str
    description: str = ""
    query: str
    top_k: int = Field(default=8, gt=0)
    expected_sources_all: List[str] = Field(default_factory=list)
    expected_sources_any: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "RetrievalBenchmarkCase":
        case_id = self.id.strip()
        query = self.query.strip()
        if not case_id:
            raise ValueError("benchmark case id must not be empty")
        if not query:
            raise ValueError("benchmark case query must not be empty")
        expected_all = [
            str(item).strip()
            for item in self.expected_sources_all
            if str(item).strip()
        ]
        expected_any = [
            str(item).strip()
            for item in self.expected_sources_any
            if str(item).strip()
        ]
        if not expected_all and not expected_any:
            raise ValueError(
                "benchmark case must define expected_sources_all or expected_sources_any"
            )
        object.__setattr__(self, "id", case_id)
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "expected_sources_all", expected_all)
        object.__setattr__(self, "expected_sources_any", expected_any)
        return self


class RetrievalBenchmarkDataset(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    corpus: RetrievalCorpusConfig
    cases: List[RetrievalBenchmarkCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "RetrievalBenchmarkDataset":
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("benchmark case identifiers must be unique")
        if not self.cases:
            raise ValueError("retrieval benchmark contains no cases")
        return self


class RetrievalBenchmarkCaseResult(BaseModel):
    id: str
    description: str
    query: str
    top_k: int
    passed: bool
    expected_match_mode: str
    expected_sources: List[str] = Field(default_factory=list)
    retrieved_sources: List[str] = Field(default_factory=list)
    retrieved_document_ids: List[str] = Field(default_factory=list)
    retrieved_chunk_count: int = 0
    retrieved_relevant_chunk_count: int = 0
    first_expected_source_rank: int | None = None
    retrieval_source_recall: float = 0.0


class RetrievalBenchmarkSummary(BaseModel):
    total_cases: int
    passed_cases: int
    overall_score: float
    retrieval_evaluable_cases: int = 0
    cases_with_retrieval_hits: int = 0
    retrieval_hit_rate: float = 0.0
    retrieval_ranked_cases: int = 0
    retrieval_top_source_accuracy: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_source_recall: float = 0.0
    ingested_document_count: int = 0
    ingested_chunk_count: int = 0
    store_backend: str = ""
    store_destination: str = ""
    embedder: str = ""
    embedding_dimension: int = 0


class RetrievalBenchmarkReport(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: RetrievalBenchmarkSummary
    cases: List[RetrievalBenchmarkCaseResult] = Field(default_factory=list)


def load_retrieval_benchmark_dataset(path: Path) -> RetrievalBenchmarkDataset:
    if not path.exists():
        raise FileNotFoundError(f"retrieval benchmark file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        return RetrievalBenchmarkDataset.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid retrieval benchmark dataset: {exc}") from exc


def write_retrieval_benchmark_report(
    report: RetrievalBenchmarkReport, path: Path
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _vector_entries(payload: Iterable[Dict[str, object]]) -> List[VectorEntry]:
    entries: list[VectorEntry] = []
    for record in payload:
        metadata = dict(cast(Dict[str, object], record.get("metadata", {})))
        metadata.setdefault("source", str(record["id"]).split("::")[0])
        entries.append(
            VectorEntry(
                document_id=str(record["id"]),
                embedding=list(cast(Sequence[float], record["embedding"])),
                text=str(record["text"]),
                metadata=metadata,
            )
        )
    return entries


def _resolve_corpus_root(dataset_path: Path, corpus: RetrievalCorpusConfig) -> Path:
    if corpus.root:
        root = Path(corpus.root)
        if not root.is_absolute():
            root = dataset_path.parent / root
        return root
    return dataset_path.parent


def _resolve_corpus_paths(
    dataset_path: Path, corpus: RetrievalCorpusConfig
) -> list[tuple[RetrievalCorpusDocument, Path]]:
    root = _resolve_corpus_root(dataset_path, corpus)
    resolved: list[tuple[RetrievalCorpusDocument, Path]] = []
    for document in corpus.documents:
        path = Path(document.path)
        if not path.is_absolute():
            path = root / path
        resolved.append((document, path.resolve()))
    return resolved


def _benchmark_source(chunk: RetrievedChunk) -> str:
    alias = str(chunk.metadata.get("benchmark_source", "")).strip()
    if alias:
        return alias
    source = str(chunk.metadata.get("source", "")).strip()
    if source:
        source_path = Path(source)
        return source_path.stem or source_path.name or source
    return str(chunk.document_id).strip()


def _retrieval_details(
    chunks: Sequence[RetrievedChunk],
) -> tuple[list[str], list[str]]:
    sources: list[str] = []
    document_ids: list[str] = []
    seen_sources: set[str] = set()
    seen_ids: set[str] = set()
    for chunk in chunks:
        source = _benchmark_source(chunk)
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)
        document_id = str(chunk.document_id).strip()
        if document_id and document_id not in seen_ids:
            seen_ids.add(document_id)
            document_ids.append(document_id)
    return sources, document_ids


def _expected_sources(case: RetrievalBenchmarkCase) -> tuple[str, list[str], set[str]]:
    expected = case.expected_sources_all or case.expected_sources_any
    mode = "all" if case.expected_sources_all else "any"
    normalized = {_normalize_text(source) for source in expected}
    return mode, list(expected), normalized


def _retrieval_expectations(
    case: RetrievalBenchmarkCase, retrieved_sources: Sequence[str]
) -> tuple[str, list[str], int | None, float, bool]:
    mode, expected_sources, normalized_expected = _expected_sources(case)
    normalized_retrieved = [_normalize_text(source) for source in retrieved_sources]
    matched = [
        source
        for source in normalized_retrieved
        if source in normalized_expected
    ]
    first_rank: int | None = None
    for index, source in enumerate(normalized_retrieved, start=1):
        if source in normalized_expected:
            first_rank = index
            break
    recall = (
        0.0
        if not normalized_expected
        else len(set(matched)) / len(normalized_expected)
    )
    if mode == "all":
        passed = recall == 1.0
    else:
        passed = first_rank is not None
    return mode, expected_sources, first_rank, recall, passed


def _load_corpus_records(
    dataset: RetrievalBenchmarkDataset, dataset_path: Path
) -> list[Dict[str, object]]:
    resolved_documents = _resolve_corpus_paths(dataset_path, dataset.corpus)
    path_to_id = {
        str(path): document.id for document, path in resolved_documents
    }
    records = ingest_paths(path for _, path in resolved_documents)
    for record in records:
        metadata = dict(record.get("metadata", {}))
        source = str(metadata.get("source", "")).strip()
        benchmark_source = path_to_id.get(str(Path(source).resolve()))
        if benchmark_source:
            metadata["benchmark_source"] = benchmark_source
        record["metadata"] = metadata
    return records


def run_retrieval_benchmark(
    dataset: RetrievalBenchmarkDataset,
    dataset_path: Path,
    *,
    store_path: Path | None = None,
    embedder: Any | None = None,
) -> RetrievalBenchmarkReport:
    active_embedder = embedder or build_embedder(get_settings())
    temp_dir: TemporaryDirectory[str] | None = None
    effective_store_path: Path
    if store_path is None:
        temp_dir = TemporaryDirectory()
        effective_store_path = Path(temp_dir.name) / "retrieval_benchmark_store.json"
    else:
        effective_store_path = store_path

    try:
        store = initialize_store(effective_store_path, active_embedder)
        store.load([])
        persist_store(effective_store_path, store)

        records = _load_corpus_records(dataset, dataset_path)
        chunks: list[Dict[str, object]] = []
        for record in records:
            chunks.extend(
                sliding_window_chunk(
                    record,
                    chunk_size=dataset.corpus.chunk_size,
                    overlap=dataset.corpus.chunk_overlap,
                )
            )
        embedded = embed_chunks(chunks, active_embedder)
        entries = _vector_entries(embedded)
        store.add(entries)
        persist_store(effective_store_path, store)

        retriever = RagRetriever(active_embedder, store)
        case_results: list[RetrievalBenchmarkCaseResult] = []
        retrieval_evaluable_cases = 0
        cases_with_hits = 0
        retrieval_ranked_cases = 0
        retrieval_top_source_hits = 0
        retrieval_rr_total = 0.0
        retrieval_source_recall_total = 0.0

        for case in dataset.cases:
            retrieved_chunks = retriever.retrieve(case.query, top_k=case.top_k)
            retrieved_sources, retrieved_document_ids = _retrieval_details(retrieved_chunks)
            (
                match_mode,
                expected_sources,
                first_expected_source_rank,
                retrieval_source_recall,
                passed,
            ) = _retrieval_expectations(case, retrieved_sources)
            expected_source_set = {
                _normalize_text(source) for source in expected_sources
            }
            retrieved_relevant_chunk_count = sum(
                1
                for chunk in retrieved_chunks
                if _normalize_text(_benchmark_source(chunk)) in expected_source_set
            )
            if expected_sources:
                retrieval_evaluable_cases += 1
                retrieval_ranked_cases += 1
                retrieval_source_recall_total += retrieval_source_recall
                if first_expected_source_rank is not None:
                    cases_with_hits += 1
                    retrieval_rr_total += 1.0 / first_expected_source_rank
                    if first_expected_source_rank == 1:
                        retrieval_top_source_hits += 1
            case_results.append(
                RetrievalBenchmarkCaseResult(
                    id=case.id,
                    description=case.description,
                    query=case.query,
                    top_k=case.top_k,
                    passed=passed,
                    expected_match_mode=match_mode,
                    expected_sources=expected_sources,
                    retrieved_sources=retrieved_sources,
                    retrieved_document_ids=retrieved_document_ids,
                    retrieved_chunk_count=len(retrieved_chunks),
                    retrieved_relevant_chunk_count=retrieved_relevant_chunk_count,
                    first_expected_source_rank=first_expected_source_rank,
                    retrieval_source_recall=retrieval_source_recall,
                )
            )

        passed_cases = sum(1 for item in case_results if item.passed)
        summary = RetrievalBenchmarkSummary(
            total_cases=len(case_results),
            passed_cases=passed_cases,
            overall_score=(
                0.0 if not case_results else passed_cases / len(case_results)
            ),
            retrieval_evaluable_cases=retrieval_evaluable_cases,
            cases_with_retrieval_hits=cases_with_hits,
            retrieval_hit_rate=(
                0.0
                if retrieval_evaluable_cases == 0
                else cases_with_hits / retrieval_evaluable_cases
            ),
            retrieval_ranked_cases=retrieval_ranked_cases,
            retrieval_top_source_accuracy=(
                0.0
                if retrieval_ranked_cases == 0
                else retrieval_top_source_hits / retrieval_ranked_cases
            ),
            retrieval_mrr=(
                0.0
                if retrieval_ranked_cases == 0
                else retrieval_rr_total / retrieval_ranked_cases
            ),
            retrieval_source_recall=(
                0.0
                if retrieval_ranked_cases == 0
                else retrieval_source_recall_total / retrieval_ranked_cases
            ),
            ingested_document_count=len(records),
            ingested_chunk_count=len(entries),
            store_backend=type(store).__name__,
            store_destination=describe_store_destination(effective_store_path, store),
            embedder=type(active_embedder).__name__,
            embedding_dimension=int(getattr(active_embedder, "dimension", 0) or 0),
        )
        metadata = dict(dataset.metadata)
        metadata["suite_type"] = "retrieval_benchmark"
        metadata["corpus_root"] = str(_resolve_corpus_root(dataset_path, dataset.corpus))
        metadata["store_path"] = str(effective_store_path)
        return RetrievalBenchmarkReport(
            metadata=metadata,
            summary=summary,
            cases=case_results,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def render_retrieval_benchmark_report(report: RetrievalBenchmarkReport) -> str:
    lines = [
        f"suite_type\t{report.metadata.get('suite_type', 'retrieval_benchmark')}",
        f"total_cases\t{report.summary.total_cases}",
        f"passed_cases\t{report.summary.passed_cases}",
        f"overall_score\t{report.summary.overall_score:.2%}",
        f"ingested_documents\t{report.summary.ingested_document_count}",
        f"ingested_chunks\t{report.summary.ingested_chunk_count}",
        f"store_backend\t{report.summary.store_backend}",
        f"store_destination\t{report.summary.store_destination}",
        f"embedder\t{report.summary.embedder}",
        f"embedding_dimension\t{report.summary.embedding_dimension}",
        "retrieval_hits\t"
        f"{report.summary.cases_with_retrieval_hits}/{report.summary.retrieval_evaluable_cases}",
        f"retrieval_hit_rate\t{report.summary.retrieval_hit_rate:.2%}",
        "retrieval_top_source_accuracy\t"
        f"{report.summary.retrieval_top_source_accuracy:.2%}",
        f"retrieval_mrr\t{report.summary.retrieval_mrr:.4f}",
        f"retrieval_source_recall\t{report.summary.retrieval_source_recall:.2%}",
        "",
        "case_id\tpassed\texpected\tretrieved\trelevant_chunks\tfirst_expected_rank\tsource_recall",
    ]
    for case in report.cases:
        lines.append(
            "\t".join(
                [
                    case.id,
                    "yes" if case.passed else "no",
                    ",".join(case.expected_sources),
                    ",".join(case.retrieved_sources),
                    str(case.retrieved_relevant_chunk_count),
                    "" if case.first_expected_source_rank is None else str(case.first_expected_source_rank),
                    f"{case.retrieval_source_recall:.2f}",
                ]
            )
        )
    return "\n".join(lines)
