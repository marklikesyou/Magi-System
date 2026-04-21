from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Iterable, List

from .vectorstore import InMemoryVectorStore, VectorEntry

logger = logging.getLogger(__name__)


def load_entries(path: Path) -> List[VectorEntry]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [VectorEntry.from_dict(raw) for raw in payload]


def save_entries(path: Path, entries: Iterable[VectorEntry]) -> None:
    serialized = [entry.to_dict() for entry in entries]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
        ) as handle:
            json.dump(serialized, handle, ensure_ascii=True, indent=2)
            tmp_path = Path(handle.name)
        tmp_path.replace(path)
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass


def initialize_store(path: Path, embedder) -> InMemoryVectorStore:
    store = InMemoryVectorStore(getattr(embedder, "dimension"))
    entries = load_entries(path)
    if entries:
        stored_dimensions = sorted({len(entry.embedding) for entry in entries})
        if stored_dimensions != [store.dim]:
            raise RuntimeError(
                "vector store at "
                f"{path} uses embedding dimension(s) {stored_dimensions}, but the "
                f"active embedder expects {store.dim}. Re-ingest the documents with "
                "the current embedder or restore the previous embedder configuration."
            )
    store.load(entries)
    return store
