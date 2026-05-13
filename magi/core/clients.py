from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, TypedDict, cast

from httpx import Timeout

from .utils import RateLimiter, retry_with_backoff


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


def _normalize_responses_input(
    messages: Sequence[LLMMessage],
) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    valid_roles = {"user", "assistant", "system", "developer"}
    for message in messages:
        role = str(message.get("role", "")).strip().lower() or "user"
        if role not in valid_roles:
            role = "user"
        content = message.get("content", "")
        payload.append({"role": role, "content": content, "type": "message"})
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


def _normalize_responses_tools(
    tools: Optional[Sequence[ToolSpec]],
) -> Optional[List[Dict[str, Any]]]:
    if not tools:
        return None
    normalized: List[Dict[str, Any]] = []
    for spec in tools:
        normalized.append(
            {
                "type": "function",
                "name": spec["name"],
                "description": spec.get("description", ""),
                "parameters": spec.get("parameters", {}),
                "strict": True,
            }
        )
    return normalized


def _responses_text_config(
    response_format: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if response_format is None:
        return None
    if response_format.get("type") == "json_schema":
        json_schema = response_format.get("json_schema", {})
        if not isinstance(json_schema, dict):
            raise LLMClientError("json_schema response format must be an object")
        format_payload: Dict[str, Any] = {
            "type": "json_schema",
            "name": json_schema.get("name", "magi_response"),
            "schema": json_schema.get("schema", {}),
            "strict": bool(json_schema.get("strict", True)),
        }
        description = json_schema.get("description")
        if description:
            format_payload["description"] = description
        return {"format": format_payload}
    if response_format.get("type") == "json_object":
        return {"format": {"type": "json_object"}}
    return {"format": response_format}


def _extract_responses_text(payload: Dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    parts: List[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, list):
                for content_item in content:
                    if not isinstance(content_item, dict):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        parts.append(text.strip())
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    if parts:
        return "\n".join(parts)

    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    return ""


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        api_base: str | None = None,
        organization: str | None = None,
        timeout: float | None = None,
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        requests_per_minute: int = 0,
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
        self.max_retries = max(1, int(max_retries))
        self.retry_initial_delay = max(0.0, float(retry_initial_delay))
        self.rate_limiter = RateLimiter(requests_per_minute)

    def complete(
        self,
        messages: Sequence[LLMMessage],
        *,
        tools: Optional[Sequence[ToolSpec]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        normalized_input = _normalize_responses_input(messages)
        normalized_tools = _normalize_responses_tools(tools)
        text_config = _responses_text_config(response_format)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "input": normalized_input,
            "store": False,
        }
        if normalized_tools is not None:
            kwargs["tools"] = normalized_tools
            kwargs["parallel_tool_calls"] = True
        if text_config is not None:
            kwargs["text"] = text_config
        rate_limiter = getattr(self, "rate_limiter", None)
        if rate_limiter is not None:
            rate_limiter.acquire()
        max_retries = max(1, int(getattr(self, "max_retries", 3)))
        retry_initial_delay = max(
            0.0, float(getattr(self, "retry_initial_delay", 1.0))
        )
        call = retry_with_backoff(
            max_retries=max_retries,
            initial_delay=retry_initial_delay,
            exceptions=(Exception,),
        )(lambda: self.client.responses.create(**kwargs))
        try:
            response = call()
        except Exception as exc:
            raise LLMClientError(str(exc)) from exc
        if hasattr(response, "model_dump"):
            raw = cast(Dict[str, Any], response.model_dump())
        elif hasattr(response, "to_dict"):
            raw = cast(Dict[str, Any], response.to_dict())
        elif hasattr(response, "json"):
            raw = cast(Dict[str, Any], json.loads(response.json()))
        else:
            raw = {"response": response}

        text = getattr(response, "output_text", None)
        if not isinstance(text, str) or not text.strip():
            text = _extract_responses_text(raw)
        return {
            "choices": [{"message": {"content": text or json.dumps(raw)}}],
            "response": raw,
            "text": text,
        }


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
        max_retries: int = 3,
        retry_initial_delay: float = 1.0,
        requests_per_minute: int = 0,
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
        self.max_retries = max(1, int(max_retries))
        self.retry_initial_delay = max(0.0, float(retry_initial_delay))
        self.rate_limiter = RateLimiter(requests_per_minute)

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
        rate_limiter = getattr(self, "rate_limiter", None)
        if rate_limiter is not None:
            rate_limiter.acquire()
        max_retries = max(1, int(getattr(self, "max_retries", 3)))
        retry_initial_delay = max(
            0.0, float(getattr(self, "retry_initial_delay", 1.0))
        )
        call = retry_with_backoff(
            max_retries=max_retries,
            initial_delay=retry_initial_delay,
            exceptions=(Exception,),
        )(lambda: self.client.models.generate_content(**kwargs))
        try:
            response = call()
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
            max_retries=settings.provider_max_retries,
            retry_initial_delay=settings.provider_retry_initial_delay,
            requests_per_minute=settings.provider_requests_per_minute,
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
            max_retries=settings.provider_max_retries,
            retry_initial_delay=settings.provider_retry_initial_delay,
            requests_per_minute=settings.provider_requests_per_minute,
        )
    return None
