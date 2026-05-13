from __future__ import annotations

from typing import Any, cast

from pathlib import Path

from magi.app.artifacts import artifact_dir
from magi.app.cli import default_store_path
from magi.core.config import Settings, get_settings, user_data_dir, user_env_file


def test_settings_default_runtime_controls() -> None:
    settings = cast(Any, Settings)(_env_file=None)

    assert settings.provider_max_retries == 3
    assert settings.provider_retry_initial_delay == 1.0
    assert settings.provider_requests_per_minute == 0
    assert settings.approve_min_citation_hit_rate == 1.0
    assert settings.approve_min_answer_support_score == 0.2
    assert settings.require_human_review_for_approvals is True
    assert settings.openai_model == "gpt-5-mini"
    assert settings.openai_fast_model == "gpt-5-mini"
    assert settings.openai_strong_model == "gpt-5.2"
    assert settings.openai_high_stakes_model == "gpt-5.2"
    assert settings.enable_model_routing is True
    assert settings.enable_responder_llm is False
    assert settings.enable_live_personas is True
    assert settings.decision_trace_dir == ""
    assert settings.run_artifact_dir == ""
    assert settings.data_dir == ""
    assert settings.profile_dir == ""


def test_settings_read_runtime_controls_from_env(monkeypatch) -> None:
    monkeypatch.setenv("MAGI_PROVIDER_MAX_RETRIES", "5")
    monkeypatch.setenv("MAGI_PROVIDER_RETRY_INITIAL_DELAY", "0.25")
    monkeypatch.setenv("MAGI_PROVIDER_REQUESTS_PER_MINUTE", "30")
    monkeypatch.setenv("MAGI_APPROVE_MIN_CITATION_HIT_RATE", "0.75")
    monkeypatch.setenv("MAGI_APPROVE_MIN_ANSWER_SUPPORT_SCORE", "0.35")
    monkeypatch.setenv("MAGI_REQUIRE_HUMAN_REVIEW_FOR_APPROVALS", "false")
    monkeypatch.setenv("MAGI_OPENAI_FAST_MODEL", "gpt-5-nano")
    monkeypatch.setenv("MAGI_OPENAI_STRONG_MODEL", "gpt-5.2")
    monkeypatch.setenv("MAGI_OPENAI_HIGH_STAKES_MODEL", "gpt-5.2-pro")
    monkeypatch.setenv("MAGI_ENABLE_MODEL_ROUTING", "false")
    monkeypatch.setenv("MAGI_ENABLE_RESPONDER_LLM", "true")
    monkeypatch.setenv("MAGI_ENABLE_LIVE_PERSONAS", "true")
    monkeypatch.setenv("MAGI_DECISION_TRACE_DIR", "/tmp/magi-traces")
    monkeypatch.setenv("MAGI_RUN_ARTIFACT_DIR", "/tmp/magi-artifacts")
    monkeypatch.setenv("MAGI_DATA_DIR", "/tmp/magi-data")
    monkeypatch.setenv("MAGI_PROFILE_DIR", "/tmp/magi-profiles")

    settings = cast(Any, Settings)(_env_file=None)

    assert settings.provider_max_retries == 5
    assert settings.provider_retry_initial_delay == 0.25
    assert settings.provider_requests_per_minute == 30
    assert settings.approve_min_citation_hit_rate == 0.75
    assert settings.approve_min_answer_support_score == 0.35
    assert settings.require_human_review_for_approvals is False
    assert settings.openai_fast_model == "gpt-5-nano"
    assert settings.openai_strong_model == "gpt-5.2"
    assert settings.openai_high_stakes_model == "gpt-5.2-pro"
    assert settings.enable_model_routing is False
    assert settings.enable_responder_llm is True
    assert settings.enable_live_personas is True
    assert settings.decision_trace_dir == "/tmp/magi-traces"
    assert settings.run_artifact_dir == "/tmp/magi-artifacts"
    assert settings.data_dir == "/tmp/magi-data"
    assert settings.profile_dir == "/tmp/magi-profiles"


def test_default_runtime_paths_use_magi_data_dir(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "magi-data"
    monkeypatch.setenv("MAGI_DATA_DIR", str(data_dir))

    settings = cast(Any, Settings)(_env_file=None)

    assert user_data_dir(settings) == data_dir
    assert default_store_path(settings) == data_dir / "storage" / "vector_store.json"
    assert artifact_dir(settings) == data_dir / "artifacts"


def test_default_runtime_paths_use_xdg_data_home(monkeypatch, tmp_path: Path) -> None:
    xdg_home = tmp_path / "xdg-data"
    monkeypatch.delenv("MAGI_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_home))

    settings = cast(Any, Settings)(_env_file=None)

    assert user_data_dir(settings) == xdg_home / "magi-system"
    assert default_store_path(settings) == (
        xdg_home / "magi-system" / "storage" / "vector_store.json"
    )
    assert artifact_dir(settings) == xdg_home / "magi-system" / "artifacts"


def test_get_settings_reads_user_config_file(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("MAGI_CONFIG_DIR", str(tmp_path / "config"))
    config_file = user_env_file()
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text('GOOGLE_API_KEY="google-test-key"\n', encoding="utf-8")

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.google_api_key == "google-test-key"
    get_settings.cache_clear()
