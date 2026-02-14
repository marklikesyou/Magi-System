

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

EmbedFn = Callable[[str], Sequence[float]]


_BATCH_SIZE = 100


def embed_chunks(chunks: Iterable[Dict[str, str]], embed_fn: EmbedFn) -> List[Dict[str, object]]:
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

    has_batch = hasattr(embed_fn, "embed_batch") and callable(getattr(embed_fn, "embed_batch"))

    if has_batch:
        embedded: List[Dict[str, object]] = []
        for start in range(0, len(chunk_list), _BATCH_SIZE):
            batch = chunk_list[start : start + _BATCH_SIZE]
            texts = [c["text"] for c in batch]
            vectors = embed_fn.embed_batch(texts)
            for chunk, vector in zip(batch, vectors):
                embedded.append({**chunk, "embedding": list(vector)})
        return embedded


    embedded = []
    for chunk in chunk_list:
        vector = list(embed_fn(chunk["text"]))
        embedded.append({**chunk, "embedding": vector})
    return embedded
