"""Tests for magi.core.storage – persistence helpers for the vector store."""

from __future__ import annotations

import json

from magi.core.embeddings import HashingEmbedder
from magi.core.storage import initialize_store, load_entries, save_entries
from magi.core.vectorstore import VectorEntry


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


def test_initialize_store_fresh(tmp_path):
    """A brand-new store (no file on disk) has the correct dimension and is empty."""
    path = tmp_path / "fresh_store.json"
    embedder = HashingEmbedder(dimension=32)
    store = initialize_store(path, embedder)
    assert store.dim == 32
    assert len(store.entries) == 0


def test_initialize_store_dimension_mismatch(tmp_path):
    """When stored embeddings have a different dimension the store starts fresh."""
    path = tmp_path / "mismatch_store.json"

    old_embedder = HashingEmbedder(dimension=64)
    old_entry = VectorEntry(
        document_id="old",
        embedding=old_embedder("stale data"),
        text="stale data",
    )
    save_entries(path, [old_entry])

    new_embedder = HashingEmbedder(dimension=32)
    store = initialize_store(path, new_embedder)

    assert store.dim == 32
    assert len(store.entries) == 0
