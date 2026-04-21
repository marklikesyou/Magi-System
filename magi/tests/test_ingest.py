from __future__ import annotations

import sys
from types import SimpleNamespace

from magi.data_pipeline.ingest import ingest_paths, load_text


class _FakePage:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakePdfReader:
    def __init__(self, _path: str) -> None:
        self.pages = [_FakePage("Alpha"), _FakePage("Beta")]


def test_load_text_extracts_pdf_pages(tmp_path, monkeypatch):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"\xff\x00\xff")
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=_FakePdfReader))

    records = load_text(pdf_path)

    assert [record["id"] for record in records] == [
        f"{pdf_path}#page-1",
        f"{pdf_path}#page-2",
    ]
    assert records[0]["text"].startswith("[Page 1]")
    assert records[0]["metadata"]["source"] == str(pdf_path)
    assert records[0]["metadata"]["page"] == 1
    assert records[0]["metadata"]["content_hash"]


def test_ingest_paths_dedupes_duplicate_content(tmp_path, capsys):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("same content", encoding="utf-8")
    second.write_text("same content", encoding="utf-8")

    records = ingest_paths([first, second])

    assert len(records) == 1
    assert records[0]["metadata"]["source"] == str(first)
    assert "duplicates existing content" in capsys.readouterr().out
