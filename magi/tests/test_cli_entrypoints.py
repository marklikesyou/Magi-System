from __future__ import annotations

import argparse
from pathlib import Path

import run_magi
from magi.app import cli


def test_command_chat_returns_nonzero_on_runtime_error(monkeypatch, tmp_path: Path, capsys) -> None:
    def fail_settings():
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "get_settings", fail_settings)

    status = cli.command_chat(
        argparse.Namespace(
            query="What happened?",
            constraints="",
            store=tmp_path / "store.json",
            verbose=False,
        )
    )

    captured = capsys.readouterr()
    assert status == 1
    assert "Error: boom" in captured.out


def test_run_magi_skips_constraints_prompt_when_query_is_provided(monkeypatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("MAGI overview", encoding="utf-8")
    captured: dict[str, argparse.Namespace] = {}

    def fail_prompt(_message: str) -> str:
        raise AssertionError("prompt_text should not be called for non-interactive constraints")

    def fake_ingest(args: argparse.Namespace) -> int:
        return 0

    def fake_chat(args: argparse.Namespace) -> int:
        captured["chat"] = args
        return 0

    monkeypatch.setattr(run_magi, "prompt_text", fail_prompt)
    monkeypatch.setattr(run_magi, "command_ingest", fake_ingest)
    monkeypatch.setattr(run_magi, "command_chat", fake_chat)

    status = run_magi.main(
        [
            "--docs",
            str(doc_path),
            "--query",
            "Summarize MAGI",
            "--store",
            str(tmp_path / "store.json"),
        ]
    )

    assert status == 0
    assert captured["chat"].constraints == ""


def test_run_magi_returns_ingest_failure_without_calling_chat(monkeypatch, tmp_path: Path) -> None:
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("MAGI overview", encoding="utf-8")

    def fake_ingest(args: argparse.Namespace) -> int:
        return 1

    def fail_chat(args: argparse.Namespace) -> int:
        raise AssertionError("command_chat should not be called when ingestion fails")

    monkeypatch.setattr(run_magi, "command_ingest", fake_ingest)
    monkeypatch.setattr(run_magi, "command_chat", fail_chat)

    status = run_magi.main(
        [
            "--docs",
            str(doc_path),
            "--query",
            "Summarize MAGI",
            "--store",
            str(tmp_path / "store.json"),
        ]
    )

    assert status == 1
