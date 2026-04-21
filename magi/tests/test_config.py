from __future__ import annotations

from typing import Any, cast

from magi.core.config import Settings


def test_settings_default_runtime_controls() -> None:
    settings = cast(Any, Settings)(_env_file=None)

    assert settings.provider_max_retries == 3
    assert settings.provider_retry_initial_delay == 1.0
    assert settings.provider_requests_per_minute == 0
    assert settings.approve_min_citation_hit_rate == 1.0
    assert settings.approve_min_answer_support_score == 0.2
    assert settings.require_human_review_for_approvals is True
    assert settings.decision_trace_dir == ""


def test_settings_read_runtime_controls_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MAGI_PROVIDER_MAX_RETRIES", "5")
    monkeypatch.setenv("MAGI_PROVIDER_RETRY_INITIAL_DELAY", "0.25")
    monkeypatch.setenv("MAGI_PROVIDER_REQUESTS_PER_MINUTE", "30")
    monkeypatch.setenv("MAGI_APPROVE_MIN_CITATION_HIT_RATE", "0.75")
    monkeypatch.setenv("MAGI_APPROVE_MIN_ANSWER_SUPPORT_SCORE", "0.35")
    monkeypatch.setenv("MAGI_REQUIRE_HUMAN_REVIEW_FOR_APPROVALS", "false")
    monkeypatch.setenv("MAGI_DECISION_TRACE_DIR", "/tmp/magi-traces")

    settings = cast(Any, Settings)(_env_file=None)

    assert settings.provider_max_retries == 5
    assert settings.provider_retry_initial_delay == 0.25
    assert settings.provider_requests_per_minute == 30
    assert settings.approve_min_citation_hit_rate == 0.75
    assert settings.approve_min_answer_support_score == 0.35
    assert settings.require_human_review_for_approvals is False
    assert settings.decision_trace_dir == "/tmp/magi-traces"
