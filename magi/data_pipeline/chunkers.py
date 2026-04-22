from __future__ import annotations

import re
from typing import Any, Dict, List, cast

_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?]\s+|\n\n+")


def _find_sentence_break(text: str, max_pos: int, min_pos: int) -> int:
    """Return the best sentence-boundary position in *text* up to *max_pos*.

    Searches for the last sentence boundary (period / exclamation / question
    mark followed by whitespace, or a paragraph break) that falls between
    *min_pos* and *max_pos*.  If no boundary is found in that range the
    function returns *max_pos* (i.e. falls back to the raw character limit).
    """
    best = -1
    for m in _SENTENCE_BOUNDARY_RE.finditer(text, 0, max_pos):
        end = m.end()
        if end <= max_pos and end >= min_pos:
            best = end
    if best == -1:
        return max_pos
    return best


def sliding_window_chunk(
    document: Dict[str, object],
    *,
    chunk_size: int = 1500,
    overlap: int = 200,
) -> List[Dict[str, object]]:

    text = str(document["text"])
    doc_id = str(document["id"])
    base_metadata = dict(cast(Dict[str, Any], document.get("metadata", {})))
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    min_ratio = 0.80

    chunks: List[Dict[str, object]] = []
    start = 0
    idx = 0
    while start < len(text):
        raw_end = start + chunk_size
        if raw_end >= len(text):
            chunk_text = text[start:]
            chunks.append(
                {
                    "id": f"{doc_id}::chunk-{idx}",
                    "text": chunk_text,
                    "metadata": {**base_metadata, "chunk_id": f"chunk-{idx}"},
                }
            )
            break

        min_break = start + int(chunk_size * min_ratio)
        break_pos = _find_sentence_break(text, raw_end, min_break)

        chunk_text = text[start:break_pos]
        chunks.append(
            {
                "id": f"{doc_id}::chunk-{idx}",
                "text": chunk_text,
                "metadata": {**base_metadata, "chunk_id": f"chunk-{idx}"},
            }
        )

        start = break_pos - overlap
        start = max(start, 0)
        idx += 1
    return chunks
