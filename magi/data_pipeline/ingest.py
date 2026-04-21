from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List

IngestRecord = Dict[str, Any]


def _content_hash(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def _metadata_for_text(
    path: Path,
    text: str,
    *,
    ingested_at: str,
    page: int | None = None,
) -> Dict[str, object]:
    digest = _content_hash(text)
    metadata: Dict[str, object] = {
        "source": str(path),
        "ingested_at": ingested_at,
        "content_hash": digest,
        "document_version": digest,
    }
    if page is not None:
        metadata["page"] = page
    return metadata


def _extract_pdf_pages(path: Path) -> List[IngestRecord]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf missing") from exc

    reader = PdfReader(str(path))
    ingested_at = datetime.now(timezone.utc).isoformat()
    records: List[IngestRecord] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        labeled = f"[Page {index}]\n{text.strip()}"
        records.append(
            {
                "id": f"{path}#page-{index}",
                "text": labeled,
                "metadata": _metadata_for_text(
                    path,
                    text.strip(),
                    ingested_at=ingested_at,
                    page=index,
                ),
            }
        )
    return records


def load_text(path: Path) -> List[IngestRecord]:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        content = path.read_text(encoding="utf-8")
        ingested_at = datetime.now(timezone.utc).isoformat()
        return [
            {
                "id": str(path),
                "text": content,
                "metadata": _metadata_for_text(
                    path,
                    content,
                    ingested_at=ingested_at,
                ),
            }
        ]
    except UnicodeDecodeError:
        if path.suffix.lower() == ".pdf":
            return _extract_pdf_pages(path)
        raise RuntimeError(
            f"cannot decode {path} as UTF-8 text. Provide plain text or install "
            "a parser for this format."
        )


def ingest_paths(paths: Iterable[Path]) -> List[IngestRecord]:
    records: List[IngestRecord] = []
    seen_hashes: set[str] = set()
    for path in paths:
        entries = load_text(path)
        added = False
        for entry in entries:
            text = str(entry.get("text", ""))
            if not text.strip():
                continue
            metadata = entry.get("metadata", {})
            content_hash = ""
            if isinstance(metadata, dict):
                content_hash = str(metadata.get("content_hash", "")).strip()
            if content_hash and content_hash in seen_hashes:
                print(f"Warning: {path} duplicates existing content, skipping.")
                continue
            records.append(entry)
            added = True
            if content_hash:
                seen_hashes.add(content_hash)
        if not added:
            print(f"Warning: {path} produced no extractable text, skipping.")
    return records
