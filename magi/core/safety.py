from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Tuple

from .config import get_settings

logger = logging.getLogger(__name__)


_DEFAULT_FORBIDDEN = {
    "ignore previous",
    "override instructions",
    "system override",
    "unfiltered output",
    "disable safety",
}

_DEFAULT_BANNED = {
    "api_key",
    "password",
    "ssn",
    "credit card",
    "confidential",
}

_INJECTION_PATTERN = re.compile(r"(?:ignore|bypass|forget)\s+(?:all\s+)?(?:previous|prior)\s+(?:directions|instructions)", re.IGNORECASE)
_CREDENTIAL_PATTERN = re.compile(r"(?:api[-_]?key|secret|token|password)\s*[:=]", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<\s*(?:script|iframe|object)", re.IGNORECASE)


@dataclass
class SafetyReport:
    flagged: bool
    reasons: List[str]
    metadata: Dict[str, Any]
    stage: Literal["input", "retrieval", "output"] = "input"
    blocked: bool = False


def moderate_text(text: str, client: Any | None) -> Tuple[bool, Dict[str, Any]]:
    if not client:
        return False, {}
    response = client.moderations.create(model="omni-moderation-latest", input=text)
    first = response.results[0]
    categories = getattr(first, "categories", {}) or {}
    if hasattr(categories, "model_dump"):
        category_values = list(categories.model_dump().values())
    elif isinstance(categories, dict):
        category_values = list(categories.values())
    else:
        category_values = []
    flagged = getattr(first, "flagged", False) or any(bool(value) for value in category_values)
    if hasattr(response, "to_dict"):
        payload = response.to_dict()
    elif hasattr(response, "model_dump"):
        payload = response.model_dump()
    else:
        payload = {"raw": response}
    return flagged, payload


def detect_prompt_injection(text: str, forbidden_markers: Iterable[str] | None = None) -> bool:
    markers = set(forbidden_markers or _DEFAULT_FORBIDDEN)
    lowered = text.lower()
    if any(marker in lowered for marker in markers):
        return True
    if _INJECTION_PATTERN.search(text):
        return True
    return False


def detect_sensitive_leak(text: str, banned_keywords: Iterable[str] | None = None) -> bool:
    markers = set(banned_keywords or _DEFAULT_BANNED)
    lowered = text.lower()
    if any(marker in lowered for marker in markers):
        return True
    if _CREDENTIAL_PATTERN.search(text):
        return True
    return False


def detect_malicious_markup(text: str) -> bool:
    return bool(_HTML_PATTERN.search(text))


def is_blocked(report: SafetyReport) -> bool:
    return report.blocked


def analyze_safety(
    text: str,
    client: Any | None = None,
    *,
    stage: Literal["input", "retrieval", "output"] = "input",
) -> SafetyReport:
    if client is None:
        settings = get_settings()
        if settings.openai_api_key:
            try:
                from openai import OpenAI
            except ImportError:
                client = None
            else:
                kwargs: Dict[str, Any] = {"api_key": settings.openai_api_key}
                if settings.openai_api_base:
                    kwargs["base_url"] = settings.openai_api_base
                if settings.openai_organization:
                    kwargs["organization"] = settings.openai_organization
                client = OpenAI(**kwargs)
    reasons: List[str] = []
    try:
        flagged_moderation, moderation_payload = moderate_text(text, client)
    except Exception:
        logger.warning("Moderation API unreachable; degrading to local-only safety checks")
        flagged_moderation, moderation_payload = False, {"error": "moderation_unavailable"}
    if detect_prompt_injection(text):
        reasons.append("prompt_injection")
    if detect_sensitive_leak(text):
        reasons.append("sensitive_leak")
    if detect_malicious_markup(text):
        reasons.append("malicious_markup")
    if flagged_moderation:
        reasons.append("provider_moderation")
    flagged = bool(reasons)
    metadata: Dict[str, Any] = {"moderation": moderation_payload}
    blocked = False
    if stage == "retrieval":
        blocked = any(reason in {"prompt_injection", "malicious_markup", "sensitive_leak"} for reason in reasons)
    elif stage == "input":
        blocked = any(reason in {"provider_moderation", "malicious_markup", "sensitive_leak"} for reason in reasons)
        if "prompt_injection" in reasons and not text.strip().lower().startswith(("explain", "describe", "summarize")):
            blocked = True
    else:
        blocked = any(reason in {"provider_moderation", "malicious_markup", "sensitive_leak"} for reason in reasons)
    return SafetyReport(flagged=flagged, reasons=reasons, metadata=metadata, stage=stage, blocked=blocked)


__all__ = [
    "SafetyReport",
    "moderate_text",
    "detect_prompt_injection",
    "detect_sensitive_leak",
    "detect_malicious_markup",
    "analyze_safety",
    "is_blocked",
]
