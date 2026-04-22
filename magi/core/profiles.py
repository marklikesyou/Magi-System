from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError, field_validator

from .routing import QueryMode

PresentationStyle = Literal[
    "standard",
    "executive_brief",
    "incident_review",
    "policy_triage",
    "vendor_review",
    "security_review",
]


class Profile(BaseModel):
    name: str
    description: str = ""
    default_constraints: str = ""
    route_mode: QueryMode | None = None
    retrieval_top_k: int | None = Field(default=None, ge=1, le=32)
    metadata_filters: dict[str, object] = Field(default_factory=dict)
    source_weights: dict[str, float] = Field(default_factory=dict)
    prompt_preamble: str = ""
    response_format_guidance: str = ""
    presentation_style: PresentationStyle = "standard"
    show_persona_perspectives: bool = True
    show_routing_rationale: bool = True
    show_cited_evidence: bool = True
    show_blocked_evidence: bool = True
    max_next_steps: int | None = Field(default=None, ge=1, le=10)
    approve_min_citation_hit_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )
    approve_min_answer_support_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name(cls, value: object) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("profile name must not be empty")
        return text

    @field_validator(
        "description",
        "default_constraints",
        "prompt_preamble",
        "response_format_guidance",
        mode="before",
    )
    @classmethod
    def _normalize_text(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("source_weights", mode="before")
    @classmethod
    def _normalize_source_weights(cls, value: object) -> dict[str, float]:
        if not isinstance(value, dict):
            return {}
        normalized: dict[str, float] = {}
        for key, raw in value.items():
            source = str(key).strip()
            if not source:
                continue
            try:
                weight = float(raw)
            except (TypeError, ValueError):
                continue
            normalized[source] = max(0.0, min(weight, 5.0))
        return normalized


@dataclass(frozen=True)
class ProfileSummary:
    name: str
    description: str
    path: Path
    source: str
    route_mode: QueryMode | None
    retrieval_top_k: int | None
    presentation_style: PresentationStyle


def builtin_profile_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "profiles"


def _search_roots(base_dir: Path | None = None) -> list[Path]:
    roots: list[Path] = []
    if base_dir is not None:
        roots.append(base_dir)
    builtin = builtin_profile_dir()
    if builtin not in roots:
        roots.append(builtin)
    return roots


def resolve_profile_path(reference: str, *, base_dir: Path | None = None) -> Path:
    raw = Path(reference).expanduser()
    if raw.exists():
        return raw.resolve()
    for root in _search_roots(base_dir):
        direct = root / reference
        if direct.exists():
            return direct.resolve()
        for suffix in (".yaml", ".yml", ".json"):
            candidate = root / f"{reference}{suffix}"
            if candidate.exists():
                return candidate.resolve()
    raise FileNotFoundError(f"profile not found: {reference}")


def _load_profile_payload(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        import json

        return json.loads(raw)
    return yaml.safe_load(raw) or {}


def load_profile(reference: str | None, *, base_dir: Path | None = None) -> Profile | None:
    if reference is None or not str(reference).strip():
        return None
    path = resolve_profile_path(reference.strip(), base_dir=base_dir)
    payload = _load_profile_payload(path)
    try:
        profile = Profile.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid profile '{path.name}': {exc}") from exc
    if not profile.name:
        raise ValueError(f"invalid profile '{path.name}': name is required")
    return profile


def list_profiles(*, base_dir: Path | None = None) -> list[ProfileSummary]:
    summaries: dict[str, ProfileSummary] = {}
    for root in _search_roots(base_dir):
        if not root.exists():
            continue
        source = "workspace" if base_dir is not None and root == base_dir else "builtin"
        for path in sorted(root.glob("*")):
            if path.suffix.lower() not in {".yaml", ".yml", ".json"}:
                continue
            try:
                profile = load_profile(path.name, base_dir=root)
            except Exception:
                continue
            if profile is None:
                continue
            summaries.setdefault(
                profile.name,
                ProfileSummary(
                name=profile.name,
                description=profile.description,
                path=path.resolve(),
                source=source,
                route_mode=profile.route_mode,
                retrieval_top_k=profile.retrieval_top_k,
                presentation_style=profile.presentation_style,
                ),
            )
    return sorted(summaries.values(), key=lambda item: item.name)


__all__ = [
    "Profile",
    "ProfileSummary",
    "PresentationStyle",
    "builtin_profile_dir",
    "list_profiles",
    "load_profile",
    "resolve_profile_path",
]
