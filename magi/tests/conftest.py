"""Shared fixtures for the Magi test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest





os.environ.setdefault("MAGI_FORCE_DSPY_STUB", "1")
os.environ.setdefault("MAGI_FORCE_HASH_EMBEDDER", "1")

from magi.core.embeddings import HashingEmbedder
from magi.core.vectorstore import VectorEntry






@pytest.fixture()
def small_embedder() -> HashingEmbedder:
    """A lightweight hashing embedder with a small dimension for fast tests."""
    return HashingEmbedder(dimension=32, bucket_size=1)


@pytest.fixture()
def sample_entries(small_embedder: HashingEmbedder) -> list[VectorEntry]:
    """A handful of pre-embedded VectorEntry objects for store tests."""
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require large datasets.",
        "Python is a popular programming language.",
        "Neural networks consist of layers of neurons.",
        "The weather today is sunny and warm.",
    ]
    entries: list[VectorEntry] = []
    for idx, text in enumerate(texts):
        embedding = small_embedder(text)
        entries.append(
            VectorEntry(
                document_id=f"doc-{idx}",
                embedding=embedding,
                text=text,
                metadata={"source": f"test-{idx}"},
            )
        )
    return entries


@pytest.fixture()
def vector_store_path(tmp_path: Path) -> Path:
    """A temporary file path suitable for persisting a vector store."""
    return tmp_path / "test_store.json"
