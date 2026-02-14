"""Tests for magi.core.embeddings – HashingEmbedder and build_embedder."""

from __future__ import annotations

import math

from magi.core.embeddings import HashingEmbedder, build_embedder
from magi.core.config import Settings






def test_hashing_embedder_deterministic():
    """The same text always produces the exact same embedding vector."""
    embedder = HashingEmbedder(dimension=64)
    vec_a = embedder("hello world")
    vec_b = embedder("hello world")
    assert vec_a == vec_b


def test_hashing_embedder_dimension():
    """Output vector length matches the configured dimension."""
    for dim in (16, 64, 256):
        embedder = HashingEmbedder(dimension=dim)
        vec = embedder("test input")
        assert len(vec) == dim


def test_hashing_embedder_normalized():
    """Output vector is L2-normalized (unit length)."""
    embedder = HashingEmbedder(dimension=128)
    vec = embedder("some longer sentence with multiple words")
    l2_norm = math.sqrt(sum(x * x for x in vec))
    assert math.isclose(l2_norm, 1.0, abs_tol=1e-9)


def test_hashing_embedder_different_texts():
    """Different texts produce different embedding vectors."""
    embedder = HashingEmbedder(dimension=128)
    vec_a = embedder("machine learning is great")
    vec_b = embedder("the weather is cold today")
    assert vec_a != vec_b


def test_hashing_embedder_embed_batch():
    """embed_batch returns one vector per input text, matching sequential results."""
    embedder = HashingEmbedder(dimension=64)
    texts = ["alpha", "beta", "gamma"]
    batch_results = embedder.embed_batch(texts)
    assert len(batch_results) == 3
    for text, batch_vec in zip(texts, batch_results):
        assert batch_vec == embedder(text)






def test_build_embedder_forces_hash(monkeypatch):
    """MAGI_FORCE_HASH_EMBEDDER=1 causes build_embedder to return a HashingEmbedder."""
    monkeypatch.setenv("MAGI_FORCE_HASH_EMBEDDER", "1")
    settings = Settings(openai_api_key="sk-fake-key-for-testing")
    embedder = build_embedder(settings)
    assert isinstance(embedder, HashingEmbedder)


def test_build_embedder_no_key_returns_hash(monkeypatch):
    """Without an OpenAI API key, build_embedder falls back to HashingEmbedder."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MAGI_FORCE_HASH_EMBEDDER", "0")
    settings = Settings(openai_api_key="", force_hash_embeddings=False)
    embedder = build_embedder(settings)
    assert isinstance(embedder, HashingEmbedder)
