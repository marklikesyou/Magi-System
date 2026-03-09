from __future__ import annotations

import pytest

from magi.core.clients import GeminiClient, LLMClientError, OpenAIClient


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


def test_gemini_client_rejects_response_format():
    client = object.__new__(GeminiClient)
    client.model = "gemini-test"
    client.client = object()

    with pytest.raises(LLMClientError, match="response_format"):
        client.complete(
            [{"role": "user", "content": "Return JSON"}],
            response_format={"type": "json_schema"},
        )
