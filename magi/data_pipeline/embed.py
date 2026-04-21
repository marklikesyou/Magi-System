from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Protocol, Sequence, cast

EmbedFn = Callable[[str], Sequence[float]]


class BatchEmbedFn(Protocol):
    def __call__(self, text: str) -> Sequence[float]: ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]: ...


_BATCH_SIZE = 100


def embed_chunks(
    chunks: Iterable[Dict[str, object]], embed_fn: EmbedFn | BatchEmbedFn
) -> List[Dict[str, object]]:
    """Embed every chunk, using batch embedding when the embed function supports it.

    If *embed_fn* exposes an ``embed_batch`` method (as both ``OpenAIEmbedder``
    and ``HashingEmbedder`` do), chunks are grouped into batches of
    ``_BATCH_SIZE`` and embedded in bulk -- this gives a 10-50x speed-up for
    OpenAI embeddings because multiple texts are sent in a single API call.

    If *embed_fn* is a plain callable without ``embed_batch``, the function
    falls back to the original one-at-a-time approach for backward
    compatibility.
    """
    chunk_list = list(chunks)

    has_batch = hasattr(embed_fn, "embed_batch") and callable(
        getattr(embed_fn, "embed_batch")
    )

    if has_batch:
        embedded: List[Dict[str, object]] = []
        batch_embedder = cast(BatchEmbedFn, embed_fn)
        for start in range(0, len(chunk_list), _BATCH_SIZE):
            batch = chunk_list[start : start + _BATCH_SIZE]
            texts = [str(c["text"]) for c in batch]
            vectors = batch_embedder.embed_batch(texts)
            if len(vectors) != len(batch):
                raise RuntimeError(
                    f"embed_batch returned {len(vectors)} vector(s) for {len(batch)} chunk(s)"
                )
            for chunk, vector in zip(batch, vectors):
                embedded.append({**chunk, "embedding": list(vector)})
        return embedded

    embedded = []
    for chunk in chunk_list:
        vector = list(embed_fn(str(chunk["text"])))
        embedded.append({**chunk, "embedding": vector})
    return embedded
