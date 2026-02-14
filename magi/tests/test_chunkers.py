"""Tests for magi.data_pipeline.chunkers – sliding-window chunking logic."""

from __future__ import annotations

import pytest

from magi.data_pipeline.chunkers import sliding_window_chunk






def _make_doc(text: str, doc_id: str = "doc-1") -> dict[str, str]:
    return {"id": doc_id, "text": text}






def test_sliding_window_basic():
    """Basic chunking of a long text produces more than one chunk."""
    text = "word " * 500
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=200, overlap=50)
    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk["id"].startswith("doc-1::chunk-")
        assert len(chunk["text"]) > 0


def test_sliding_window_overlap():
    """Consecutive chunks share overlapping content."""


    text = "abcdefgh" * 200
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=400, overlap=100)
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        tail = chunks[i]["text"][-100:]
        head = chunks[i + 1]["text"][:100]
        assert tail == head, (
            f"Overlap mismatch between chunk {i} and {i + 1}"
        )


def test_sliding_window_sentence_boundary():
    """Chunks prefer to break at sentence boundaries when possible."""
    sentences = [
        "First sentence here. ",
        "Second sentence here. ",
        "Third sentence here. ",
        "Fourth sentence here. ",
        "Fifth sentence here. ",
    ]
    text = "".join(sentences * 10)
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=150, overlap=30)

    ends_at_sentence = sum(
        1 for c in chunks[:-1]
        if c["text"].rstrip().endswith(".")
    )
    assert ends_at_sentence > 0, "Expected at least one chunk to end on a sentence boundary"


def test_sliding_window_small_document():
    """A document smaller than chunk_size produces exactly one chunk."""
    text = "Short text."
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=1500, overlap=200)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text
    assert chunks[0]["id"] == "doc-1::chunk-0"


def test_sliding_window_empty_text():
    """Empty text produces an empty list (the while-loop condition is never true)."""
    chunks = sliding_window_chunk(_make_doc(""), chunk_size=500, overlap=100)
    assert chunks == []


def test_sliding_window_custom_sizes():
    """Custom chunk_size and overlap are respected."""
    text = "x" * 1000
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=300, overlap=50)

    for chunk in chunks[:-1]:
        assert len(chunk["text"]) <= 300

    assert len(chunks) >= 3


def test_overlap_larger_than_chunk_raises():
    """overlap >= chunk_size must raise ValueError."""
    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        sliding_window_chunk(_make_doc("some text"), chunk_size=100, overlap=100)

    with pytest.raises(ValueError, match="overlap must be smaller than chunk_size"):
        sliding_window_chunk(_make_doc("some text"), chunk_size=100, overlap=150)


def test_sliding_window_chunk_ids_are_sequential():
    """Chunk ids increment from chunk-0 onward."""
    text = "hello world. " * 200
    chunks = sliding_window_chunk(_make_doc(text, "mydoc"), chunk_size=200, overlap=30)
    for idx, chunk in enumerate(chunks):
        assert chunk["id"] == f"mydoc::chunk-{idx}"


def test_sliding_window_no_data_loss():
    """All characters of the original text appear in at least one chunk."""
    text = "The quick brown fox jumps over the lazy dog. " * 50
    chunks = sliding_window_chunk(_make_doc(text), chunk_size=200, overlap=50)
    reconstructed = set()
    for chunk in chunks:
        for i, ch in enumerate(text):
            if chunk["text"].find(text[i : i + 10]) != -1:
                for j in range(i, min(i + 10, len(text))):
                    reconstructed.add(j)

    assert len(reconstructed) == len(text)
