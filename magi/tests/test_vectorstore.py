"""Tests for magi.core.vectorstore – InMemoryVectorStore and helpers."""

from __future__ import annotations

import math

import pytest

from magi.core.vectorstore import (
    InMemoryVectorStore,
    RetrievedChunk,
    VectorEntry,
    cosine_similarity,
)


def test_cosine_similarity_identical():
    """Identical vectors have cosine similarity of 1.0."""
    vec = [0.3, 0.4, 0.5]
    assert math.isclose(cosine_similarity(vec, vec), 1.0, abs_tol=1e-9)


def test_cosine_similarity_orthogonal():
    """Orthogonal vectors have cosine similarity of 0.0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert math.isclose(cosine_similarity(a, b), 0.0, abs_tol=1e-9)


def test_cosine_similarity_opposite():
    """Opposite vectors have cosine similarity of -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert math.isclose(cosine_similarity(a, b), -1.0, abs_tol=1e-9)


def test_add_and_search(small_embedder, sample_entries):
    """Adding entries and searching returns the most relevant result."""
    store = InMemoryVectorStore(dim=32)
    store.add(sample_entries)

    query_vec = small_embedder("programming language")
    results = store.search(query_vec, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], RetrievedChunk)

    assert results[0].text


def test_dimension_mismatch_raises():
    """Adding an entry whose embedding dimension differs from the store raises."""
    store = InMemoryVectorStore(dim=4)
    bad_entry = VectorEntry(
        document_id="bad",
        embedding=[0.1, 0.2],
        text="oops",
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add([bad_entry])


def test_search_top_k(small_embedder, sample_entries):
    """top_k limits the number of returned results."""
    store = InMemoryVectorStore(dim=32)
    store.add(sample_entries)

    query_vec = small_embedder("fox dog")
    results_1 = store.search(query_vec, top_k=1)
    results_3 = store.search(query_vec, top_k=3)

    assert len(results_1) == 1
    assert len(results_3) == 3

    results_all = store.search(query_vec, top_k=100)
    assert len(results_all) == len(sample_entries)


def test_empty_store_search(small_embedder):
    """Searching an empty store returns an empty list."""
    store = InMemoryVectorStore(dim=32)
    query_vec = small_embedder("anything")
    assert store.search(query_vec) == []


def test_dump_and_load(small_embedder, sample_entries):
    """dump() produces dicts that can be loaded back into a fresh store."""
    store = InMemoryVectorStore(dim=32)
    store.add(sample_entries)

    dumped = store.dump()
    assert len(dumped) == len(sample_entries)

    restored_entries = [VectorEntry.from_dict(d) for d in dumped]
    store2 = InMemoryVectorStore(dim=32)
    store2.load(restored_entries)

    assert len(store2.entries) == len(sample_entries)

    for original, restored in zip(sample_entries, store2.entries):
        assert original.document_id == restored.document_id
        assert original.text == restored.text
        assert original.embedding == restored.embedding


def test_revision_changes_when_entries_change(sample_entries):
    store = InMemoryVectorStore(dim=32)
    initial_revision = store.revision

    store.add(sample_entries[:1])
    after_add = store.revision
    store.load(sample_entries)

    assert after_add > initial_revision
    assert store.revision > after_add


def test_add_replaces_existing_document_id(small_embedder):
    store = InMemoryVectorStore(dim=32)
    first = VectorEntry(document_id="doc", embedding=small_embedder("old"), text="old")
    second = VectorEntry(document_id="doc", embedding=small_embedder("new"), text="new")

    store.add([first])
    store.add([second])

    assert len(store.entries) == 1
    assert store.entries[0].text == "new"


def test_search_results_sorted_by_score(small_embedder, sample_entries):
    """Search results are returned in descending score order."""
    store = InMemoryVectorStore(dim=32)
    store.add(sample_entries)

    query_vec = small_embedder("neural network layers")
    results = store.search(query_vec, top_k=5)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_filters_by_metadata(small_embedder):
    store = InMemoryVectorStore(dim=32)
    store.add(
        [
            VectorEntry(
                document_id="doc-a",
                embedding=small_embedder("release approved"),
                text="release approved",
                metadata={"source": "doc-a.pdf", "page": 1},
            ),
            VectorEntry(
                document_id="doc-b",
                embedding=small_embedder("release approved"),
                text="release approved",
                metadata={"source": "doc-b.pdf", "page": 2},
            ),
        ]
    )

    results = store.search(
        small_embedder("release approved"),
        top_k=5,
        metadata_filters={"source": "doc-b.pdf", "page": 2},
    )

    assert [item.document_id for item in results] == ["doc-b"]


def test_search_filters_accept_multiple_allowed_values(small_embedder):
    store = InMemoryVectorStore(dim=32)
    store.add(
        [
            VectorEntry(
                document_id="doc-a",
                embedding=small_embedder("policy rollout"),
                text="policy rollout",
                metadata={"source": "doc-a.pdf"},
            ),
            VectorEntry(
                document_id="doc-b",
                embedding=small_embedder("policy rollout"),
                text="policy rollout",
                metadata={"source": "doc-b.pdf"},
            ),
            VectorEntry(
                document_id="doc-c",
                embedding=small_embedder("policy rollout"),
                text="policy rollout",
                metadata={"source": "doc-c.pdf"},
            ),
        ]
    )

    results = store.search(
        small_embedder("policy rollout"),
        top_k=5,
        metadata_filters={"source": ["doc-a.pdf", "doc-c.pdf"]},
    )

    assert [item.document_id for item in results] == ["doc-a", "doc-c"]
