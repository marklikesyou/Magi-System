from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, cast

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


def _normalize_tools(
    tools: Optional[Sequence[ToolSpec]],
) -> Optional[List[Dict[str, Any]]]:
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
        except ImportError as exc:
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
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
        }
        if normalized_tools is not None:
            kwargs["tools"] = normalized_tools
        if response_format is not None:
            kwargs["response_format"] = response_format
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise LLMClientError(str(exc)) from exc
        if hasattr(response, "model_dump"):
            return cast(Dict[str, Any], response.model_dump())
        if hasattr(response, "to_dict"):
            return cast(Dict[str, Any], response.to_dict())
        return (
            json.loads(response.json())
            if hasattr(response, "json")
            else {"response": response}
        )


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
        except ImportError as exc:
            raise RuntimeError(
                "google-genai package is required for GeminiClient"
            ) from exc

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
        del tools
        contents = "\n\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in normalized_messages
        )
        kwargs: Dict[str, Any] = {"model": self.model, "contents": contents}
        if response_format is not None:
            schema = response_format.get("json_schema", {}).get("schema", {})
            kwargs["contents"] = "\n\n".join(
                [
                    contents,
                    "Return only JSON matching this JSON Schema:",
                    json.dumps(schema, ensure_ascii=True),
                ]
            )
            try:
                from google.genai import types

                kwargs["config"] = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                )
            except Exception:
                kwargs["config"] = {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
        try:
            response = self.client.models.generate_content(**kwargs)
        except Exception as exc:
            raise LLMClientError(str(exc)) from exc
        text = getattr(response, "text", None)
        if isinstance(text, str):
            return {"choices": [{"message": {"content": text}}]}
        if hasattr(response, "to_dict"):
            return cast(Dict[str, Any], response.to_dict())
        if hasattr(response, "model_dump"):
            return cast(Dict[str, Any], response.model_dump())
        return {"response": response}


def build_default_client(
    settings: "Settings", *, model: Optional[str] = None
) -> Optional[LLMClient]:
    if settings.openai_api_key:
        preferred_model = model or settings.openai_model
        return OpenAIClient(
            preferred_model,
            settings.openai_api_key,
            api_base=settings.openai_api_base or None,
            organization=settings.openai_organization or None,
            timeout=settings.openai_request_timeout,
        )
    if settings.google_api_key:
        preferred_model = model or settings.gemini_model
        return GeminiClient(
            preferred_model,
            settings.google_api_key,
            vertex=settings.google_use_vertex,
            project=settings.google_project or None,
            location=settings.google_location or None,
            timeout=settings.openai_request_timeout,
        )
    return None
