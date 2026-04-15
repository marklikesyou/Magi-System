from __future__ import annotations

from magi.core import clients
from magi.core.clients import GeminiClient, OpenAIClient, build_default_client
from magi.core.config import Settings


class _FakeResponse:
    def model_dump(self) -> dict[str, object]:
        return {"choices": [{"message": {"content": '{"ok": true}'}}]}


class _FakeCompletions:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def create(self, **kwargs: object) -> _FakeResponse:
        self.kwargs = dict(kwargs)
        return _FakeResponse()


def test_openai_client_forwards_response_format():
    completions = _FakeCompletions()
    client = object.__new__(OpenAIClient)
    client.model = "gpt-4o-mini-2024-07-18"
    client.client = type("FakeOpenAI", (), {"chat": type("FakeChat", (), {"completions": completions})()})()

    response = client.complete(
        [{"role": "user", "content": "Return JSON"}],
        response_format={"type": "json_schema", "json_schema": {"name": "demo", "strict": True, "schema": {}}},
    )

    assert completions.kwargs is not None
    assert completions.kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "demo", "strict": True, "schema": {}},
    }
    assert response["choices"]


class _FakeGeminiResponse:
    text = '{"ok": true}'


class _FakeGeminiModels:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def generate_content(self, **kwargs: object) -> _FakeGeminiResponse:
        self.kwargs = dict(kwargs)
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
            "json_schema": {"schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
        },
    )

    assert models.kwargs is not None
    assert models.kwargs["model"] == "gemini-test"
    assert "config" in models.kwargs
    assert response["choices"][0]["message"]["content"] == '{"ok": true}'


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
    )

    client = build_default_client(settings)

    assert isinstance(client, FakeGeminiClient)
    assert client.model == "gemini-default"
