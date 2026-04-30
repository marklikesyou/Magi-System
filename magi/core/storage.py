from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable, List, Mapping

from .config import get_settings
from .embeddings import HashingEmbedder
from .pgvectorstore import PgVectorStore
from .vectorstore import InMemoryVectorStore, VectorEntry, VectorStore

logger = logging.getLogger(__name__)

STORE_FORMAT = "magi_vector_store_v2"


def _coerce_int(value: object) -> int:
    try:
        return int(str(value or 0))
    except (TypeError, ValueError):
        return 0


def load_store_bundle(path: Path) -> tuple[dict[str, object], List[VectorEntry]]:
    if not path.exists():
        return {}, []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        return {}, [VectorEntry.from_dict(raw) for raw in payload]
    if not isinstance(payload, dict):
        raise RuntimeError(f"store payload at {path} must be a JSON object or list")
    metadata = payload.get("metadata", {})
    raw_entries = payload.get("entries", [])
    return (
        dict(metadata) if isinstance(metadata, dict) else {},
        [VectorEntry.from_dict(raw) for raw in raw_entries if isinstance(raw, dict)],
    )


def load_entries(path: Path) -> List[VectorEntry]:
    _, entries = load_store_bundle(path)
    return entries


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


def save_entries(
    path: Path,
    entries: Iterable[VectorEntry],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    serialized = [entry.to_dict() for entry in entries]
    if metadata is None:
        save_json_document(path, serialized)
        return
    payload = {
        "format": STORE_FORMAT,
        "metadata": dict(metadata),
        "entries": serialized,
    }
    save_json_document(path, payload)


def _vector_db_url() -> str:
    settings = get_settings()
    return str(getattr(settings, "vector_db_url", "") or "").strip()


def embedder_fingerprint(embedder: object) -> dict[str, object]:
    if isinstance(embedder, HashingEmbedder):
        return {
            "kind": "hashing",
            "dimension": int(getattr(embedder, "dimension", 0)),
            "bucket_size": int(getattr(embedder, "bucket_size", 0)),
            "model": "hashing",
        }
    return {
        "kind": "provider",
        "dimension": int(getattr(embedder, "dimension", 0)),
        "bucket_size": 0,
        "model": str(getattr(embedder, "model", "")).strip() or "provider",
    }


def _fingerprint_label(fingerprint: Mapping[str, object]) -> str:
    kind = str(fingerprint.get("kind", "unknown")).strip() or "unknown"
    model = str(fingerprint.get("model", "unknown")).strip() or "unknown"
    dimension = _coerce_int(fingerprint.get("dimension", 0))
    if kind == "hashing":
        bucket_size = _coerce_int(fingerprint.get("bucket_size", 0))
        return f"hashing(dim={dimension}, bucket_size={bucket_size})"
    return f"{kind}(model={model}, dim={dimension})"


def _prepare_store_metadata(
    existing: Mapping[str, object] | None,
    *,
    embedder: object,
    entry_count: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(existing or {})
    metadata["format"] = STORE_FORMAT
    metadata["entry_count"] = entry_count
    metadata["updated_at"] = now
    metadata.setdefault("created_at", now)
    metadata["embedder"] = embedder_fingerprint(embedder)
    chunking = metadata.get("chunking", {})
    if not isinstance(chunking, dict):
        chunking = {}
    if chunk_size is not None:
        chunking["chunk_size"] = int(chunk_size)
    if chunk_overlap is not None:
        chunking["chunk_overlap"] = int(chunk_overlap)
    metadata["chunking"] = chunking
    return metadata


def _refresh_store_metadata(
    existing: Mapping[str, object] | None,
    *,
    entry_count: int,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(existing or {})
    metadata["format"] = STORE_FORMAT
    metadata["entry_count"] = entry_count
    metadata["updated_at"] = now
    metadata.setdefault("created_at", now)
    chunking = metadata.get("chunking", {})
    if not isinstance(chunking, dict):
        chunking = {}
    if chunk_size is not None:
        chunking["chunk_size"] = int(chunk_size)
    if chunk_overlap is not None:
        chunking["chunk_overlap"] = int(chunk_overlap)
    metadata["chunking"] = chunking
    return metadata


def store_metadata(store: object) -> dict[str, object]:
    value = getattr(store, "metadata", {})
    return dict(value) if isinstance(value, dict) else {}


def _set_store_metadata(store: object, metadata: Mapping[str, object]) -> None:
    setattr(store, "metadata", dict(metadata))


def _ensure_store_compatible(
    path: Path,
    metadata: Mapping[str, object],
    entries: Iterable[VectorEntry],
    embedder: object,
) -> None:
    entry_list = list(entries)
    active = embedder_fingerprint(embedder)
    stored = metadata.get("embedder", {})
    if isinstance(stored, dict) and stored:
        stored_dimension = _coerce_int(stored.get("dimension", 0))
        active_dimension = _coerce_int(active.get("dimension", 0))
        stored_model = str(stored.get("model", "")).strip()
        active_model = str(active.get("model", "")).strip()
        stored_kind = str(stored.get("kind", "")).strip()
        active_kind = str(active.get("kind", "")).strip()
        if (
            stored_dimension != active_dimension
            or stored_model != active_model
            or stored_kind != active_kind
        ):
            raise RuntimeError(
                "vector store at "
                f"{path} was created with {_fingerprint_label(stored)}, but the "
                f"active embedder is {_fingerprint_label(active)}. Re-ingest with "
                "the active embedder or point MAGI at a compatible store."
            )
        return

    stored_dimensions = sorted({len(entry.embedding) for entry in entry_list})
    active_dimension = _coerce_int(active.get("dimension", 0))
    if entry_list and stored_dimensions != [active_dimension]:
        raise RuntimeError(
            "vector store at "
            f"{path} uses embedding dimension(s) {stored_dimensions}, but the "
            f"active embedder expects {active_dimension}. Re-ingest the documents "
            "with the current embedder or restore the previous embedder configuration."
        )


def initialize_store(path: Path, embedder: object) -> VectorStore:
    database_url = _vector_db_url()
    if database_url:
        return PgVectorStore(
            database_url,
            getattr(embedder, "dimension"),
            store_path=path,
        )
    store = InMemoryVectorStore(getattr(embedder, "dimension"))
    metadata, entries = load_store_bundle(path)
    _ensure_store_compatible(path, metadata, entries, embedder)
    store.load(entries)
    _set_store_metadata(
        store,
        _prepare_store_metadata(metadata, embedder=embedder, entry_count=len(entries)),
    )
    return store


def persist_store(
    path: Path,
    store: VectorStore,
    *,
    embedder: object | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> None:
    if isinstance(store, InMemoryVectorStore):
        metadata = store_metadata(store)
        active_embedder = embedder
        fingerprint = metadata.get("embedder", {})
        if active_embedder is None and isinstance(fingerprint, dict):
            if fingerprint.get("kind") == "hashing":
                active_embedder = HashingEmbedder(
                    dimension=int(fingerprint.get("dimension", 384) or 384),
                    bucket_size=int(fingerprint.get("bucket_size", 2) or 2),
                )
        if active_embedder is None:
            prepared = _refresh_store_metadata(
                metadata,
                entry_count=len(store.entries),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        else:
            prepared = _prepare_store_metadata(
                metadata,
                embedder=active_embedder,
                entry_count=len(store.entries),
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        _set_store_metadata(store, prepared)
        save_entries(path, store.entries, metadata=prepared)


def describe_store_destination(path: Path, store: VectorStore) -> str:
    if isinstance(store, PgVectorStore):
        return f"Store persisted to PostgreSQL namespace {Path(path).resolve()}"
    metadata = store_metadata(store)
    fingerprint = metadata.get("embedder", {})
    if isinstance(fingerprint, dict) and fingerprint:
        return (
            f"Store persisted to {path} "
            f"({_fingerprint_label(fingerprint)}, entries={len(store.entries)})"
        )
    return f"Store persisted to {path}"


__all__ = [
    "describe_store_destination",
    "embedder_fingerprint",
    "initialize_store",
    "load_entries",
    "load_store_bundle",
    "persist_store",
    "save_entries",
    "save_json_document",
    "store_metadata",
]
