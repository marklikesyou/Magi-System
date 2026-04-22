"""Tests for magi.core.storage – persistence helpers for the vector store."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import magi.core.storage as storage
from magi.core.embeddings import HashingEmbedder
from magi.core.storage import (
    describe_store_destination,
    load_store_bundle,
    initialize_store,
    load_entries,
    persist_store,
    save_entries,
    save_json_document,
)
from magi.core.vectorstore import InMemoryVectorStore, VectorEntry


def test_save_and_load_entries(sample_entries, vector_store_path):
    """Round-trip save followed by load preserves every entry."""
    save_entries(vector_store_path, sample_entries)
    loaded = load_entries(vector_store_path)

    assert len(loaded) == len(sample_entries)
    for original, restored in zip(sample_entries, loaded):
        assert original.document_id == restored.document_id
        assert original.text == restored.text
        assert original.embedding == restored.embedding
        assert original.metadata == restored.metadata


def test_load_missing_file(tmp_path):
    """Loading from a path that does not exist returns an empty list."""
    missing = tmp_path / "does_not_exist.json"
    assert load_entries(missing) == []


def test_save_atomic(sample_entries, vector_store_path):
    """save_entries writes via a temporary file (atomic rename)."""
    save_entries(vector_store_path, sample_entries)

    assert vector_store_path.exists()
    data = json.loads(vector_store_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == len(sample_entries)


def test_initialize_store_fresh(tmp_path, monkeypatch):
    """A brand-new store (no file on disk) has the correct dimension and is empty."""
    path = tmp_path / "fresh_store.json"
    embedder = HashingEmbedder(dimension=32)
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(vector_db_url=""))
    store = initialize_store(path, embedder)
    assert store.dim == 32
    assert len(store.entries) == 0


def test_initialize_store_dimension_mismatch(tmp_path, monkeypatch):
    """When stored embeddings have a different dimension the store fails loudly."""
    path = tmp_path / "mismatch_store.json"

    old_embedder = HashingEmbedder(dimension=64)
    old_entry = VectorEntry(
        document_id="old",
        embedding=old_embedder("stale data"),
        text="stale data",
    )
    save_entries(path, [old_entry])

    new_embedder = HashingEmbedder(dimension=32)
    monkeypatch.setattr(storage, "get_settings", lambda: SimpleNamespace(vector_db_url=""))
    with pytest.raises(RuntimeError, match="active embedder expects 32"):
        initialize_store(path, new_embedder)


def test_save_json_document_writes_atomic_json(tmp_path):
    path = tmp_path / "records" / "decision.json"

    save_json_document(path, {"verdict": "approve", "score": 0.9})

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {"verdict": "approve", "score": 0.9}


def test_persist_store_writes_json_for_in_memory_store(sample_entries, tmp_path):
    path = tmp_path / "persisted_store.json"
    store = InMemoryVectorStore(dim=len(sample_entries[0].embedding))
    store.metadata = {
        "embedder": {
            "kind": "hashing",
            "dimension": len(sample_entries[0].embedding),
            "bucket_size": 2,
            "model": "hashing",
        }
    }
    store.add(sample_entries)

    persist_store(path, store)

    payload = load_entries(path)
    assert [entry.document_id for entry in payload] == [
        entry.document_id for entry in sample_entries
    ]


def test_persist_store_writes_provenance_metadata(sample_entries, tmp_path):
    path = tmp_path / "store.json"
    embedder = HashingEmbedder(dimension=len(sample_entries[0].embedding))
    store = initialize_store(path, embedder)
    store.add(sample_entries)

    persist_store(path, store, embedder=embedder, chunk_size=512, chunk_overlap=64)

    metadata, entries = load_store_bundle(path)
    assert metadata["format"] == "magi_vector_store_v2"
    assert metadata["embedder"]["kind"] == "hashing"  # type: ignore[index]
    assert metadata["chunking"]["chunk_size"] == 512  # type: ignore[index]
    assert metadata["chunking"]["chunk_overlap"] == 64  # type: ignore[index]
    assert len(entries) == len(sample_entries)


def test_initialize_store_uses_pgvector_backend_when_database_url_configured(
    monkeypatch, tmp_path
):
    embedder = HashingEmbedder(dimension=32)
    captured: dict[str, object] = {}

    class FakePgVectorStore:
        def __init__(self, dsn: str, dim: int, *, store_path) -> None:
            captured["dsn"] = dsn
            captured["dim"] = dim
            captured["store_path"] = store_path

    monkeypatch.setattr(
        storage, "get_settings", lambda: SimpleNamespace(vector_db_url="postgresql://db")
    )
    monkeypatch.setattr(storage, "PgVectorStore", FakePgVectorStore)

    store_obj = initialize_store(tmp_path / "logical-store.json", embedder)

    assert isinstance(store_obj, FakePgVectorStore)
    assert captured == {
        "dsn": "postgresql://db",
        "dim": 32,
        "store_path": tmp_path / "logical-store.json",
    }


def test_describe_store_destination_uses_database_namespace(monkeypatch, tmp_path):
    class FakePgVectorStore:
        pass

    monkeypatch.setattr(storage, "PgVectorStore", FakePgVectorStore)
    message = describe_store_destination(tmp_path / "store.json", FakePgVectorStore())

    assert "PostgreSQL namespace" in message
