from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Protocol, Sequence, TypeGuard

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


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


def _query_tokens(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.findall(text.lower()) if len(token) > 2}


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

    def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]: ...

    def page_search(
        self,
        page_numbers: Iterable[str],
        top_k: int = 20,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]: ...


class InMemoryVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self._entries: List[VectorEntry] = []
        self._embedding_matrix = np.empty((0, dim), dtype=np.float32)
        self._normalized_matrix = np.empty((0, dim), dtype=np.float32)
        self._revision = 0

    def _rebuild_matrix(self) -> None:
        if not self._entries:
            self._embedding_matrix = np.empty((0, self.dim), dtype=np.float32)
            self._normalized_matrix = np.empty((0, self.dim), dtype=np.float32)
            return

        matrix = np.asarray(
            [entry.embedding for entry in self._entries],
            dtype=np.float32,
        )
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(norms == 0.0, np.float32(1.0), norms)
        self._embedding_matrix = matrix
        self._normalized_matrix = (matrix / safe_norms).astype(np.float32)

    def _normalized_query(self, query_embedding: Sequence[float]) -> np.ndarray:
        query = np.asarray(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(query))
        if norm == 0.0:
            return np.zeros((self.dim,), dtype=np.float32)
        return query / norm

    @staticmethod
    def _top_score_positions(
        scores: np.ndarray,
        source_indices: Sequence[int],
        top_k: int,
    ) -> list[int]:
        available = len(source_indices)
        limit = min(top_k, available)
        if limit <= 0:
            return []
        if limit < available:
            candidate_positions = np.argpartition(scores, -limit)[-limit:]
        else:
            candidate_positions = np.arange(available)
        return sorted(
            (int(position) for position in candidate_positions),
            key=lambda position: (-float(scores[position]), source_indices[position]),
        )

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
            self._rebuild_matrix()
            self._revision += 1

    def load(self, entries: Iterable[VectorEntry]) -> None:
        pending = list(entries)
        for entry in pending:
            if len(entry.embedding) != self.dim:
                raise ValueError("embedding dimension mismatch.")
        self._entries = pending
        self._rebuild_matrix()
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
        if top_k <= 0:
            return []

        if metadata_filters:
            source_indices: Sequence[int] = [
                index
                for index, entry in enumerate(self._entries)
                if metadata_matches_filters(entry.metadata, metadata_filters)
            ]
            matrix = self._normalized_matrix[list(source_indices)]
        else:
            source_indices = range(len(self._entries))
            matrix = self._normalized_matrix
        if not source_indices:
            return []

        query = self._normalized_query(query_embedding)
        scores = matrix @ query
        ranked_positions = self._top_score_positions(
            scores,
            source_indices,
            top_k,
        )
        results: list[RetrievedChunk] = []
        for position in ranked_positions:
            entry_index = source_indices[position]
            entry = self._entries[entry_index]
            results.append(
                RetrievedChunk(
                    document_id=entry.document_id,
                    text=entry.text,
                    score=float(scores[position]),
                    metadata=entry.metadata,
                )
            )
        return results

    def keyword_search(
        self,
        query: str,
        top_k: int = 20,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []
        query_tokens = _query_tokens(query)
        if not query_tokens:
            return []

        def score_entry(entry: VectorEntry) -> RetrievedChunk | None:
            if not metadata_matches_filters(entry.metadata, metadata_filters):
                return None
            text_tokens = _query_tokens(entry.text)
            if not text_tokens:
                return None
            overlap = len(query_tokens.intersection(text_tokens))
            if overlap <= 0:
                return None
            return RetrievedChunk(
                document_id=entry.document_id,
                text=entry.text,
                score=min(1.0, overlap / max(1, len(query_tokens))),
                metadata=entry.metadata,
            )

        scored = (
            chunk
            for chunk in (score_entry(entry) for entry in self._entries)
            if chunk is not None
        )
        return heapq.nlargest(top_k, scored, key=lambda chunk: chunk.score)

    def page_search(
        self,
        page_numbers: Iterable[str],
        top_k: int = 20,
        *,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> List[RetrievedChunk]:
        if top_k <= 0:
            return []
        requested = {str(value).strip() for value in page_numbers if str(value).strip()}
        if not requested:
            return []
        page_markers = [
            (f"page {value}".lower(), f"page-{value}".lower())
            for value in sorted(requested)
        ]

        def match_entry(entry: VectorEntry) -> RetrievedChunk | None:
            if not metadata_matches_filters(entry.metadata, metadata_filters):
                return None
            page = str(entry.metadata.get("page", "")).strip()
            source = str(entry.metadata.get("source", "")).lower()
            document_id = entry.document_id.lower()
            text = entry.text.lower()
            if page in requested:
                matched = True
            else:
                matched = any(
                    label in text
                    or label in source
                    or suffix in document_id
                    or f"#{suffix}" in document_id
                    or f"/{suffix}" in document_id
                    for label, suffix in page_markers
                )
            if not matched:
                return None
            return RetrievedChunk(
                document_id=entry.document_id,
                text=entry.text,
                score=1.2,
                metadata=entry.metadata,
            )

        scored = (
            chunk
            for chunk in (match_entry(entry) for entry in self._entries)
            if chunk is not None
        )
        return heapq.nlargest(top_k, scored, key=lambda chunk: chunk.score)
