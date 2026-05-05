from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from magi.core.pgvectorstore import PgVectorStore


class _FakeCursor:
    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[object, ...]]] = []
        self.fetchone_row: tuple[object, ...] | None = (1,)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def execute(self, sql: str, params=()) -> None:
        self.statements.append((sql, tuple(params)))

    def fetchall(self):
        return [
            (
                "doc-1",
                "MAGI policy triage guardrails include human review.",
                {"source": "doc"},
                0.75,
            )
        ]

    def fetchone(self):
        return self.fetchone_row


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def cursor(self):
        return self._cursor


def _store_with_cursor(cursor: _FakeCursor) -> PgVectorStore:
    store = PgVectorStore.__new__(PgVectorStore)
    store.dsn = "postgresql://db"
    store.dim = 32
    store.store_path = Path("logical-store.json")
    store.store_key = "/tmp/logical-store.json"
    store._entry_table = "magi_vector_entries_32"
    store._entry_index = "magi_vector_entries_32_embedding_idx"
    store._metadata_index = "magi_vector_entries_32_metadata_idx"
    store._search_index = "magi_vector_entries_32_search_idx"
    store._revision = 0

    @contextmanager
    def connection():
        yield _FakeConnection(cursor)

    cast(Any, store)._connection = connection
    return store


def test_pgvector_keyword_search_uses_full_text_index_path() -> None:
    cursor = _FakeCursor()
    store = _store_with_cursor(cursor)

    results = store.keyword_search("policy triage guardrails", top_k=5)

    sql, params = cursor.statements[-1]
    assert "websearch_to_tsquery" in sql
    assert "search_vector @@" in sql
    assert "LOWER(text) LIKE" not in sql
    assert params[-1] == 5
    assert results[0].document_id == "doc-1"


def test_pgvector_reset_deletes_namespace_and_bumps_revision() -> None:
    cursor = _FakeCursor()
    store = _store_with_cursor(cursor)

    store.reset()

    statements = [sql for sql, _params in cursor.statements]
    assert any("DELETE FROM magi_vector_entries_32" in sql for sql in statements)
    assert any("UPDATE magi_vector_store_state" in sql for sql in statements)
    assert cursor.statements[0][1] == ("/tmp/logical-store.json",)
    assert store.revision == 1
