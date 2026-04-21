from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable, List

from .config import get_settings
from .pgvectorstore import PgVectorStore
from .vectorstore import InMemoryVectorStore, VectorEntry, VectorStore

logger = logging.getLogger(__name__)


def load_entries(path: Path) -> List[VectorEntry]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [VectorEntry.from_dict(raw) for raw in payload]


def save_json_document(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
            tmp_path = Path(handle.name)
        tmp_path.replace(path)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def save_entries(path: Path, entries: Iterable[VectorEntry]) -> None:
    serialized = [entry.to_dict() for entry in entries]
    save_json_document(path, serialized)


def _vector_db_url() -> str:
    settings = get_settings()
    return str(getattr(settings, "vector_db_url", "") or "").strip()


def initialize_store(path: Path, embedder) -> VectorStore:
    database_url = _vector_db_url()
    if database_url:
        return PgVectorStore(
            database_url,
            getattr(embedder, "dimension"),
            store_path=path,
        )
    store = InMemoryVectorStore(getattr(embedder, "dimension"))
    entries = load_entries(path)
    if entries:
        stored_dimensions = sorted({len(entry.embedding) for entry in entries})
        if stored_dimensions != [store.dim]:
            raise RuntimeError(
                "vector store at "
                f"{path} uses embedding dimension(s) {stored_dimensions}, but the "
                f"active embedder expects {store.dim}. Re-ingest the documents with "
                "the current embedder or restore the previous embedder configuration."
            )
    store.load(entries)
    return store


def persist_store(path: Path, store: VectorStore) -> None:
    if isinstance(store, InMemoryVectorStore):
        save_entries(path, store.entries)


def describe_store_destination(path: Path, store: VectorStore) -> str:
    if isinstance(store, PgVectorStore):
        return f"Store persisted to PostgreSQL namespace {Path(path).resolve()}"
    return f"Store persisted to {path}"
