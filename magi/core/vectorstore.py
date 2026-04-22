from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, TypeGuard


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must share the same dimensionality.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _is_filter_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _coerce_embedding(raw: object) -> List[float]:
    if not _is_filter_sequence(raw):
        return []
    values: List[float] = []
    for value in raw:
        number = float(value) if isinstance(value, (int, float)) else float(str(value))
        values.append(number)
    return values


def _coerce_metadata(raw: object) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def metadata_matches_filters(
    metadata: Mapping[str, Any],
    metadata_filters: Mapping[str, object] | None,
) -> bool:
    if not metadata_filters:
        return True
    for key, expected in metadata_filters.items():
        if key not in metadata:
            return False
        actual = metadata.get(key)
        if _is_filter_sequence(expected):
            options = list(expected)
            if not options:
                return False
            if not any(
                metadata_matches_filters(metadata, {key: option}) for option in options
            ):
                return False
            continue
        if _is_filter_sequence(actual):
            values = list(actual)
            if not values:
                return False
            if not any(metadata_matches_filters({key: item}, {key: expected}) for item in values):
                return False
            continue
        if actual == expected:
            continue
        if actual is None or expected is None:
            return False
        if str(actual) != str(expected):
            return False
    return True


@dataclass
class VectorEntry:
    document_id: str
    embedding: List[float]
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "embedding": self.embedding,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "VectorEntry":
        return cls(
            document_id=str(payload["document_id"]),
            embedding=_coerce_embedding(payload.get("embedding", [])),
            text=str(payload["text"]),
            metadata=_coerce_metadata(payload.get("metadata", {})),
        )


@dataclass
class RetrievedChunk:
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class VectorStore(Protocol):
    dim: int

    def add(self, entries: Iterable[VectorEntry]) -> None: ...

    def load(self, entries: Iterable[VectorEntry]) -> None: ...

    @property
    def revision(self) -> int: ...

    @property
    def entries(self) -> List[VectorEntry]: ...

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]: ...


class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._entries: List[VectorEntry] = []
        self._revision = 0

    def add(self, entries: Iterable[VectorEntry]) -> None:
        pending = list(entries)
        for entry in pending:
            if len(entry.embedding) != self.dim:
                raise ValueError("embedding dimension mismatch.")
        if pending:
            positions = {
                entry.document_id: idx for idx, entry in enumerate(self._entries)
            }
            for entry in pending:
                existing_idx = positions.get(entry.document_id)
                if existing_idx is None:
                    positions[entry.document_id] = len(self._entries)
                    self._entries.append(entry)
                else:
                    self._entries[existing_idx] = entry
            self._revision += 1

    def load(self, entries: Iterable[VectorEntry]) -> None:
        pending = list(entries)
        for entry in pending:
            if len(entry.embedding) != self.dim:
                raise ValueError("embedding dimension mismatch.")
        self._entries = pending
        self._revision += 1

    def dump(self) -> List[Dict[str, Any]]:
        return [entry.to_dict() for entry in self._entries]

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def entries(self) -> List[VectorEntry]:
        return list(self._entries)

    def search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]:
        if len(query_embedding) != self.dim:
            raise ValueError("query dimension mismatch.")

        scored = [
            RetrievedChunk(
                document_id=entry.document_id,
                text=entry.text,
                score=cosine_similarity(query_embedding, entry.embedding),
                metadata=entry.metadata,
            )
            for entry in self._entries
            if metadata_matches_filters(entry.metadata, metadata_filters)
        ]
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:top_k]
