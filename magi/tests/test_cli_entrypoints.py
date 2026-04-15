from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import run_magi
from magi.app import cli
from magi.app.service import ChatSessionResult
from magi.decision.schema import FinalDecision
from magi.dspy_programs.schemas import FusionResponse


def test_command_chat_returns_nonzero_on_runtime_error(
    monkeypatch, tmp_path: Path, capsys
) -> None:
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


def test_parser_accepts_common_options_after_subcommand(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"

    args = cli.build_parser().parse_args(
        ["chat", "Summarize MAGI", "--store", str(store_path), "--json"]
    )

    assert args.store == store_path
    assert args.json is True


def test_command_chat_json_output(monkeypatch, tmp_path: Path, capsys) -> None:
    decision = FinalDecision(
        verdict="revise",
        justification="Need more evidence.",
        persona_outputs=[],
        residual_risk="medium",
    )
    fused = FusionResponse(
        verdict="revise",
        justification="Need more evidence.",
        confidence=0.5,
        final_answer="Need more evidence.",
        next_steps=["Add source documents."],
        consensus_points=[],
        disagreements=[],
        residual_risk="medium",
        risks=[],
        mitigations=[],
    )

    monkeypatch.setattr(cli, "get_settings", lambda: SimpleNamespace())
    monkeypatch.setattr(cli, "build_embedder", lambda _settings: object())
    monkeypatch.setattr(cli, "initialize_store", lambda _store, _embedder: object())
    monkeypatch.setattr(
        cli,
        "run_chat_session",
        lambda *_args, **_kwargs: ChatSessionResult(
            final_decision=decision,
            fused=fused,
            personas={},
            effective_mode="stub",
            model="",
        ),
    )

    status = cli.command_chat(
        argparse.Namespace(
            query="Summarize MAGI",
            constraints="",
            store=tmp_path / "store.json",
            verbose=True,
            json=True,
        )
    )

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["decision"]["verdict"] == "revise"
    assert payload["effective_mode"] == "stub"


def test_run_magi_skips_constraints_prompt_when_query_is_provided(
    monkeypatch, tmp_path: Path
) -> None:
    doc_path = tmp_path / "doc.txt"
    doc_path.write_text("MAGI overview", encoding="utf-8")
    captured: dict[str, argparse.Namespace] = {}

    def fail_prompt(_message: str) -> str:
        raise AssertionError(
            "prompt_text should not be called for non-interactive constraints"
        )

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


def test_run_magi_returns_ingest_failure_without_calling_chat(
    monkeypatch, tmp_path: Path
) -> None:
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
