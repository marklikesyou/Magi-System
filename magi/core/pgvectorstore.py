from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, TypeGuard

from .vectorstore import RetrievedChunk, VectorEntry

_STATE_TABLE = "magi_vector_store_state"


def _vector_literal(values: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.12g}" for value in values) + "]"


def _parse_vector(raw: object) -> List[float]:
    text = str(raw or "").strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(item) for item in text.split(",")]


def _coerce_metadata(raw: object) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if isinstance(payload, dict):
            return dict(payload)
    return {}


def _is_filter_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _metadata_filter_text(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


class PgVectorStore:
    def __init__(self, dsn: str, dim: int, *, store_path: Path):
        if dim <= 0:
            raise ValueError("dimension must be positive")
        normalized_dsn = dsn.strip()
        if not normalized_dsn:
            raise ValueError("database DSN must not be empty")
        self.dsn = normalized_dsn
        self.dim = dim
        self.store_path = Path(store_path)
        self.store_key = str(self.store_path.resolve())
        self._entry_table = f"magi_vector_entries_{dim}"
        self._entry_index = f"{self._entry_table}_embedding_idx"
        self._metadata_index = f"{self._entry_table}_metadata_idx"
        self._revision = 0
        self._ensure_schema()

    @staticmethod
    def _load_psycopg():
        try:
            import psycopg
        except ImportError as exc:
            raise RuntimeError(
                "psycopg is required for PostgreSQL vector store support. "
                "Install the PostgreSQL dependencies and retry."
            ) from exc
        return psycopg

    @contextmanager
    def _connection(self) -> Iterator[Any]:
        psycopg = self._load_psycopg()
        with psycopg.connect(self.dsn) as conn:
            yield conn

    def _bump_revision(self, cursor: Any) -> None:
        cursor.execute(
            f"""
            UPDATE {_STATE_TABLE}
            SET revision = revision + 1,
                updated_at = NOW()
            WHERE store_key = %s
            RETURNING revision
            """,
            (self.store_key,),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"vector store state row missing for logical store {self.store_key}"
            )
        self._revision = int(row[0])

    def _ensure_schema(self) -> None:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {_STATE_TABLE} (
                        store_key TEXT PRIMARY KEY,
                        dim INTEGER NOT NULL,
                        revision BIGINT NOT NULL DEFAULT 0,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self._entry_table} (
                        store_key TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        embedding vector({self.dim}) NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        PRIMARY KEY (store_key, document_id)
                    )
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._entry_index}
                    ON {self._entry_table}
                    USING hnsw (embedding vector_cosine_ops)
                    """
                )
                cursor.execute(
                    f"""
                    CREATE INDEX IF NOT EXISTS {self._metadata_index}
                    ON {self._entry_table}
                    USING gin (metadata)
                    """
                )
                cursor.execute(
                    f"""
                    INSERT INTO {_STATE_TABLE} (store_key, dim, revision)
                    VALUES (%s, %s, 0)
                    ON CONFLICT (store_key) DO NOTHING
                    """,
                    (self.store_key, self.dim),
                )
                cursor.execute(
                    f"""
                    SELECT dim, revision
                    FROM {_STATE_TABLE}
                    WHERE store_key = %s
                    """,
                    (self.store_key,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError(
                        f"vector store state row missing for logical store {self.store_key}"
                    )
                stored_dim, revision = int(row[0]), int(row[1])
                if stored_dim != self.dim:
                    raise RuntimeError(
                        "vector store at "
                        f"{self.store_path} uses embedding dimension {stored_dim}, but the "
                        f"active embedder expects {self.dim}. Re-ingest the documents with "
                        "the current embedder or restore the previous embedder configuration."
                    )
                self._revision = revision

    def add(self, entries: Iterable[VectorEntry]) -> None:
        pending = list(entries)
        for entry in pending:
            if len(entry.embedding) != self.dim:
                raise ValueError("embedding dimension mismatch.")
        if not pending:
            return
        rows = [
            (
                self.store_key,
                entry.document_id,
                _vector_literal(entry.embedding),
                entry.text,
                json.dumps(entry.metadata, ensure_ascii=True),
            )
            for entry in pending
        ]
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.executemany(
                    f"""
                    INSERT INTO {self._entry_table} (
                        store_key,
                        document_id,
                        embedding,
                        text,
                        metadata
                    )
                    VALUES (%s, %s, %s::vector, %s, %s::jsonb)
                    ON CONFLICT (store_key, document_id)
                    DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        text = EXCLUDED.text,
                        metadata = EXCLUDED.metadata,
                        updated_at = NOW()
                    """,
                    rows,
                )
                self._bump_revision(cursor)

    def load(self, entries: Iterable[VectorEntry]) -> None:
        pending = list(entries)
        for entry in pending:
            if len(entry.embedding) != self.dim:
                raise ValueError("embedding dimension mismatch.")
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"DELETE FROM {self._entry_table} WHERE store_key = %s",
                    (self.store_key,),
                )
                if pending:
                    cursor.executemany(
                        f"""
                        INSERT INTO {self._entry_table} (
                            store_key,
                            document_id,
                            embedding,
                            text,
                            metadata
                        )
                        VALUES (%s, %s, %s::vector, %s, %s::jsonb)
                        """,
                        [
                            (
                                self.store_key,
                                entry.document_id,
                                _vector_literal(entry.embedding),
                                entry.text,
                                json.dumps(entry.metadata, ensure_ascii=True),
                            )
                            for entry in pending
                        ],
                    )
                self._bump_revision(cursor)

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def entries(self) -> List[VectorEntry]:
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT document_id, embedding::text, text, metadata
                    FROM {self._entry_table}
                    WHERE store_key = %s
                    ORDER BY document_id
                    """,
                    (self.store_key,),
                )
                rows = cursor.fetchall()
        return [
            VectorEntry(
                document_id=str(document_id),
                embedding=_parse_vector(raw_embedding),
                text=str(text),
                metadata=_coerce_metadata(metadata),
            )
            for document_id, raw_embedding, text, metadata in rows
        ]

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
        query_vector = _vector_literal(query_embedding)
        filter_clauses = ["store_key = %s"]
        filter_params: list[object] = [self.store_key]
        for key, expected in (metadata_filters or {}).items():
            if _is_filter_sequence(expected):
                options = [_metadata_filter_text(item) for item in expected]
                if not options:
                    return []
                placeholders = ", ".join(["%s"] * len(options))
                filter_clauses.append(
                    "(metadata ? %s AND jsonb_extract_path_text(metadata, %s) "
                    f"IN ({placeholders}))"
                )
                filter_params.extend([key, key, *options])
                continue
            filter_clauses.append(
                "(metadata ? %s AND jsonb_extract_path_text(metadata, %s) = %s)"
            )
            filter_params.extend([key, key, _metadata_filter_text(expected)])
        where_clause = " AND ".join(filter_clauses)
        with self._connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        document_id,
                        text,
                        metadata,
                        1 - (embedding <=> %s::vector) AS score
                    FROM {self._entry_table}
                    WHERE {where_clause}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_vector, *filter_params, query_vector, top_k),
                )
                rows = cursor.fetchall()
        return [
            RetrievedChunk(
                document_id=str(document_id),
                text=str(text),
                score=float(score),
                metadata=_coerce_metadata(metadata),
            )
            for document_id, text, metadata, score in rows
        ]
