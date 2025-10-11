

from __future__ import annotations 

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict

from httpx import Timeout


if TYPE_CHECKING:
    from magi.core.config import Settings


class LLMMessage(TypedDict, total=False):
    role: str
    content: Any


class ToolSpec(TypedDict, total=False):
    name: str
    description: str
    parameters: Dict[str, Any]


class LLMClientError(RuntimeError):
    pass


class LLMClient:
    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


def _normalize_messages(messages: Sequence[LLMMessage]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role", "")).strip() or "user"
        content = message.get("content", "")
        payload.append({"role": role, "content": content})
    return payload


def _normalize_tools(tools: Optional[Sequence[ToolSpec]]) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    normalized: List[Dict[str, Any]] = []
    for spec in tools:
        normalized.append(
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                },
            }
        )
    return normalized


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        api_base: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAIClient") from exc

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if api_base:
            kwargs["base_url"] = api_base
        if organization:
            kwargs["organization"] = organization
        if timeout is not None:
            kwargs["timeout"] = Timeout(timeout)
        self.model = model
        self.client = OpenAI(**kwargs)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_messages = _normalize_messages(messages)
        normalized_tools = _normalize_tools(tools)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=normalized_messages,
                tools=normalized_tools,
                response_format=response_format,
            )
        except Exception as exc:  # pragma: no cover
            raise LLMClientError(str(exc)) from exc
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return json.loads(response.json()) if hasattr(response, "json") else {"response": response}


class GeminiClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        vertex: bool = False,
        project: Optional[str] = None,
        location: Optional[str] = None,
        timeout: float | None = None,
    ):
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("google-genai package is required for GeminiClient") from exc

        kwargs: Dict[str, Any] = {"api_key": api_key}
        if vertex:
            kwargs["vertexai"] = True
            if project:
                kwargs["project"] = project
            if location:
                kwargs["location"] = location
        self.model = model
        self.client = genai.Client(**kwargs)
        self.timeout = timeout

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_messages = _normalize_messages(messages)
        try:
            response = self.client.responses.generate(
                model=self.model,
                input=normalized_messages,
                tools=tools,
                response_format=response_format,
            )
        except Exception as exc:  # pragma: no cover
            raise LLMClientError(str(exc)) from exc
        if hasattr(response, "to_dict"):
            return response.to_dict()
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return {"response": response}


def build_default_client(settings: "Settings", *, model: Optional[str] = None) -> Optional[LLMClient]:
    preferred_model = model or settings.openai_model
    if settings.openai_api_key:
        return OpenAIClient(
            preferred_model,
            settings.openai_api_key,
            api_base=settings.openai_api_base or None,
            organization=settings.openai_organization or None,
            timeout=settings.openai_request_timeout,
        )
    if settings.google_api_key:
        return GeminiClient(
            settings.gemini_model if model is None else preferred_model,
            settings.google_api_key,
            vertex=settings.google_use_vertex,
            project=settings.google_project or None,
            location=settings.google_location or None,
            timeout=settings.openai_request_timeout,
        )
    return None
