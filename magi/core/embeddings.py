from __future__ import annotations

import hashlib
import os
from typing import Any, Iterable, List

from httpx import Timeout

from .config import Settings

OPENAI_EMBED_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class HashingEmbedder:
    def __init__(self, dimension: int = 384, bucket_size: int = 2):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if bucket_size not in (1, 2, 3):
            raise ValueError("bucket_size must be 1, 2, or 3")
        self.dimension = dimension
        self.bucket_size = bucket_size

    def _ngrams(self, tokens: List[str]) -> Iterable[str]:
        length = len(tokens)
        for size in range(1, self.bucket_size + 1):
            for start in range(length - size + 1):
                yield " ".join(tokens[start : start + size])

    def __call__(self, text: str) -> List[float]:
        tokens = text.lower().split()
        vector = [0.0] * self.dimension
        for gram in self._ngrams(tokens):
            digest = hashlib.sha1(gram.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimension
            vector[bucket] += 1.0

        norm = sum(x * x for x in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts by calling the single embed method in a loop."""
        return [self(text) for text in texts]


class OpenAIEmbedder:
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        api_base: str | None = None,
        organization: str | None = None,
        timeout: float = 60.0,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY must use OpenAI embeddings.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package is required for OpenAIEmbedder."
            ) from exc

        if model not in OPENAI_EMBED_DIMENSIONS:
            raise ValueError(
                f"embedding model '{model}' is not recognized. "
                f"known models: {', '.join(sorted(OPENAI_EMBED_DIMENSIONS))}."
            )

        kwargs: dict[str, Any] = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        if organization:
            kwargs["organization"] = organization
        if timeout is not None:
            kwargs["timeout"] = Timeout(timeout)
        self.model = model
        self.client = OpenAI(**kwargs)
        self.dimension = OPENAI_EMBED_DIMENSIONS[model]

    def __call__(self, text: str) -> List[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return list(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts in a single OpenAI API call.

        The OpenAI embeddings API accepts ``input`` as a list of strings,
        returning one embedding per input in a single round-trip.  This is
        dramatically faster than calling the API once per text.
        """
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model, input=texts)

        sorted_data = sorted(response.data, key=lambda d: d.index)
        return [list(item.embedding) for item in sorted_data]


def build_embedder(settings: Settings):
    force_hash = settings.force_hash_embeddings
    env_override = os.getenv("MAGI_FORCE_HASH_EMBEDDER")
    if env_override is not None:
        force_hash = env_override.strip().lower() in {"1", "true", "yes", "on"}

    if force_hash or not settings.openai_api_key:
        return HashingEmbedder()
    return OpenAIEmbedder(
        settings.openai_embedding_model,
        settings.openai_api_key,
        api_base=settings.openai_api_base or None,
        organization=settings.openai_organization or None,
        timeout=settings.openai_request_timeout,
    )
