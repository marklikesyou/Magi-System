from __future__ import annotations

import pytest

from magi.core.clients import LLMClient, LLMClientError
from magi.dspy_programs.runtime import _StructuredRunner
from magi.dspy_programs.schemas import MelchiorResponse


class _MalformedClient(LLMClient):
    def complete(self, messages, *, tools=None, response_format=None):  # type: ignore[override]
        return {"choices": [{"message": {"content": '{"analysis": "ok"}'}}]}


def test_structured_runner_rejects_partial_json():
    runner = _StructuredRunner(_MalformedClient(), "gpt-4o-mini-2024-07-18")
    with pytest.raises(LLMClientError, match="schema validation"):
        runner.run(
            system_prompt="Return JSON",
            user_prompt="Return JSON",
            schema_name="melchior_response",
            schema_cls=MelchiorResponse,
        )
