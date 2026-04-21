from __future__ import annotations

import os

import pytest

from magi.app.service import run_chat_session
from magi.core.config import get_settings
from magi.dspy_programs.personas import clear_cache
from magi.eval.scenario_harness import ScenarioEvidence, ScenarioRetriever


def _live_smoke_enabled() -> tuple[bool, str]:
    if os.getenv("MAGI_RUN_LIVE_SMOKE", "0") != "1":
        return False, "set MAGI_RUN_LIVE_SMOKE=1 to enable provider-backed smoke test"
    if os.getenv("MAGI_FORCE_DSPY_STUB", "0") != "0":
        return False, "MAGI_FORCE_DSPY_STUB must be 0 for live smoke test"
    get_settings.cache_clear()
    settings = get_settings()
    if not (settings.openai_api_key or settings.google_api_key):
        return False, "no provider API key configured for live smoke test"
    return True, ""


def test_run_chat_session_live_smoke() -> None:
    enabled, reason = _live_smoke_enabled()
    if not enabled:
        pytest.skip(reason)

    clear_cache()
    retriever = ScenarioRetriever(
        [
            ScenarioEvidence(
                source="README",
                text="MAGI is a multi persona reasoning engine for assessing user requests against an evidence base.",
            ),
            ScenarioEvidence(
                source="README",
                text="It retrieves context from a local vector store, convenes three specialized personas, and returns a final verdict.",
            ),
        ]
    )

    result = run_chat_session(
        "Summarize MAGI in one sentence.",
        "",
        retriever,
        force_stub=False,
    )

    assert result.effective_mode == "live"
    assert result.model
    assert set(result.personas) == {"melchior", "balthasar", "casper"}
    assert result.final_decision.verdict in {"approve", "revise", "reject"}
    assert "[1]" in (result.fused.final_answer or result.final_decision.justification)
    assert "magi" in (result.fused.final_answer or result.final_decision.justification).lower()
