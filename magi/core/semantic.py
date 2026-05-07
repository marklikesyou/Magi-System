from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def semantic_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def character_ngram_counts(text: str, *, n: int = 3) -> dict[str, int]:
    normalized = semantic_text(text)
    if not normalized:
        return {}
    padded = f"  {normalized} "
    counts: dict[str, int] = {}
    for index in range(len(padded) - n + 1):
        gram = padded[index : index + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def cosine_counts(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_similarity(text: str, profiles: Sequence[str]) -> float:
    text_counts = character_ngram_counts(text)
    if not text_counts:
        return 0.0
    return max(
        (
            cosine_counts(text_counts, character_ngram_counts(profile))
            for profile in profiles
            if profile.strip()
        ),
        default=0.0,
    )


__all__ = [
    "character_ngram_counts",
    "cosine_counts",
    "semantic_similarity",
    "semantic_text",
]
