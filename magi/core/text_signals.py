from __future__ import annotations

from typing import Sequence

INSUFFICIENT_INFORMATION_PATTERNS = (
    "insufficient",
    "does not specify",
    "doesn't specify",
    "not specified",
    "not enough evidence",
    "not enough information",
    "missing information",
    "need additional",
    "need more evidence",
    "cannot determine",
    "can't determine",
    "not directly supported",
    "not explicitly stated",
)

REFUSAL_PATTERNS = (
    "i can't assist",
    "i cannot assist",
    "can't help with that request",
    "cannot help with that request",
    "decline the request",
    "do not operationalize",
    "refuse the request",
)


def contains_pattern(text: str, patterns: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(pattern in lowered for pattern in patterns)


__all__ = [
    "INSUFFICIENT_INFORMATION_PATTERNS",
    "REFUSAL_PATTERNS",
    "contains_pattern",
]
