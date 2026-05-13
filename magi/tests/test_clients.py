from __future__ import annotations

import pytest

from magi.core import clients
from magi.core.clients import (
    GeminiClient,
    LLMClientError,
    OpenAIClient,
    build_default_client,
)
from magi.core.config import Settings


class _FakeResponse:
    output_text = '{"ok": true}'

    def model_dump(self) -> dict[str, object]:
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"ok": true}'}],
                }
            ]
        }


class _FakeResponses:
    def __init__(self, failures: int = 0) -> None:
        self.kwargs: dict[str, object] | None = None
        self.failures = failures
        self.calls = 0

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls += 1
        self.kwargs = dict(kwargs)
        if self.calls <= self.failures:
            raise RuntimeError("temporary openai failure")
        return _FakeResponse()


class _FakeRateLimiter:
    def __init__(self) -> None:
        self.calls = 0

    def acquire(self) -> None:
        self.calls += 1


def test_openai_client_uses_responses_api_and_forwards_json_schema():
    responses = _FakeResponses()
    client = object.__new__(OpenAIClient)
    client.model = "gpt-4o-mini-2024-07-18"
    client.client = type("FakeOpenAI", (), {"responses": responses})()

    response = client.complete(
        [{"role": "user", "content": "Return JSON"}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "demo", "strict": True, "schema": {}},
        },
    )

    assert responses.kwargs is not None
    assert responses.kwargs["input"] == [
        {"role": "user", "content": "Return JSON", "type": "message"}
    ]
    assert responses.kwargs["store"] is False
    assert responses.kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "demo",
            "strict": True,
            "schema": {},
        },
    }
    assert response["choices"][0]["message"]["content"] == '{"ok": true}'


class _FakeGeminiResponse:
    text = '{"ok": true}'


class _FakeGeminiModels:
    def __init__(self, failures: int = 0) -> None:
        self.kwargs: dict[str, object] | None = None
        self.failures = failures
        self.calls = 0

    def generate_content(self, **kwargs: object) -> _FakeGeminiResponse:
        self.calls += 1
        self.kwargs = dict(kwargs)
        if self.calls <= self.failures:
            raise RuntimeError("temporary gemini failure")
        return _FakeGeminiResponse()


def test_gemini_client_forwards_response_schema():
    models = _FakeGeminiModels()
    client = object.__new__(GeminiClient)
    client.model = "gemini-test"
    client.client = type("FakeGemini", (), {"models": models})()

    response = client.complete(
        [{"role": "user", "content": "Return JSON"}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}
            },
        },
    )

    assert models.kwargs is not None
    assert models.kwargs["model"] == "gemini-test"
    assert "config" in models.kwargs
    assert response["choices"][0]["message"]["content"] == '{"ok": true}'


def test_openai_client_retries_transient_failure():
    responses = _FakeResponses(failures=1)
    client = object.__new__(OpenAIClient)
    client.model = "gpt-4o-mini-2024-07-18"
    client.max_retries = 2
    client.retry_initial_delay = 0.0
    client.client = type("FakeOpenAI", (), {"responses": responses})()

    response = client.complete([{"role": "user", "content": "Return JSON"}])

    assert responses.calls == 2
    assert response["choices"]


def test_openai_client_uses_rate_limiter():
    responses = _FakeResponses()
    limiter = _FakeRateLimiter()
    client = object.__new__(OpenAIClient)
    client.model = "gpt-4o-mini-2024-07-18"
    client.max_retries = 1
    client.retry_initial_delay = 0.0
    client.rate_limiter = limiter
    client.client = type("FakeOpenAI", (), {"responses": responses})()

    client.complete([{"role": "user", "content": "Return JSON"}])

    assert limiter.calls == 1


def test_gemini_client_raises_after_retry_exhaustion():
    models = _FakeGeminiModels(failures=2)
    client = object.__new__(GeminiClient)
    client.model = "gemini-test"
    client.max_retries = 2
    client.retry_initial_delay = 0.0
    client.client = type("FakeGemini", (), {"models": models})()

    with pytest.raises(LLMClientError, match="temporary gemini failure"):
        client.complete([{"role": "user", "content": "Return JSON"}])

    assert models.calls == 2


def test_build_default_client_uses_gemini_default_when_google_only(monkeypatch):
    class FakeGeminiClient:
        def __init__(self, model: str, api_key: str, **kwargs: object) -> None:
            self.model = model
            self.api_key = api_key
            self.kwargs = kwargs

    monkeypatch.setattr(clients, "GeminiClient", FakeGeminiClient)
    settings = Settings(
        openai_api_key="",
        google_api_key="google-key",
        openai_model="openai-default",
        gemini_model="gemini-default",
        provider_max_retries=5,
        provider_retry_initial_delay=0.25,
        provider_requests_per_minute=12,
    )

    client = build_default_client(settings)

    assert isinstance(client, FakeGeminiClient)
    assert client.model == "gemini-default"
    assert client.kwargs["max_retries"] == 5
    assert client.kwargs["retry_initial_delay"] == 0.25
    assert client.kwargs["requests_per_minute"] == 12
