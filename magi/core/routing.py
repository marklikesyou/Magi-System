from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from magi.core.semantic import semantic_similarity

QueryMode = Literal["summarize", "extract", "fact_check", "recommend", "decision"]
_MODE_ORDER: tuple[QueryMode, ...] = (
    "extract",
    "fact_check",
    "decision",
    "recommend",
    "summarize",
)

_ROUTE_PROFILES: dict[QueryMode, tuple[str, ...]] = {
    "summarize": (
        "summarize summary concise informational explanation of retrieved source material",
        "brief overview of what the document says with caveats",
        "operator briefing status summary current state source material evidence says",
    ),
    "extract": (
        "direct extraction of exact source wording location field clause page or line",
        "extract stated owner accountable team assigned field value from source",
    ),
    "fact_check": (
        "verify whether a factual claim is true false supported or contradicted by evidence",
        "check if the source supports the claim or proves the assertion",
    ),
    "recommend": (
        "compare options and suggest a bounded approach with tradeoffs",
        "recommend next evidence backed step action operators should take",
    ),
    "decision": (
        "approval decision about taking an action under constraints and risk",
    ),
}
_WH_INFORMATION_RE = re.compile(r"^(?:what(?:'s)?|who|when|where|why|how)\b")
_COLLECTIVE_DECISION_RE = re.compile(
    r"^(?:(?:should|could|can|do|would)\s+we\b|is\s+it\b.+\bto\b)"
)
_YES_NO_RE = re.compile(
    r"^(?:is|are|do|does|did|was|were|has|have|can|could|would|will)\b.+\?$"
)
_ASSISTANT_REQUEST_RE = re.compile(r"^(?:can|could|would)\s+you\b")
_SOURCE_LOCATION_RE = re.compile(r"\b(?:page|section|clause|line)\b")
_SUMMARY_DIRECTIVE_RE = re.compile(
    r"\b(?:summarize|summary|brief|overview|condense|explain|describe)\b"
)
_EXTRACTION_DIRECTIVE_RE = re.compile(
    r"\b(?:extract|return|identify|which|who|owner|owning|accountable)\b"
)
_CLAIM_SUPPORT_RE = re.compile(
    r"\b(?:claim|verify|contradict|prove)\b|\b(?:source|evidence)\s+support\b"
)
_RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend|recommendation|next\s+step|what\s+should\b|how\s+should\b|action\s+follows)\b"
)
_DECISION_DIRECTIVE_RE = re.compile(
    r"\b(?:approve|approval|authorize|decision|threshold|move\s+forward|proceed)\b"
)


@dataclass(frozen=True)
class RoutingDecision:
    mode: QueryMode
    rationale: str
    retrieval_top_k: int = 8
    answer_style: str = "Deliver a grounded answer with explicit uncertainty."
    scores: dict[QueryMode, int] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)


def _is_information_request(text: str) -> bool:
    lowered = text.strip().lower()
    if _COLLECTIVE_DECISION_RE.search(lowered):
        return False
    if _SOURCE_LOCATION_RE.search(lowered):
        return False
    if lowered.endswith("?") and _WH_INFORMATION_RE.search(lowered):
        return True
    return _ASSISTANT_REQUEST_RE.search(lowered) is not None


def _add_score(
    scores: dict[QueryMode, int],
    signals: list[str],
    mode: QueryMode,
    points: int,
    reason: str,
) -> None:
    scores[mode] += points
    signals.append(f"{mode}:{reason}")


def route_query(
    query: str,
    constraints: str = "",
    *,
    forced_mode: QueryMode | None = None,
) -> RoutingDecision:
    if forced_mode is not None:
        return RoutingDecision(
            mode=forced_mode,
            rationale="Explicit route override was provided.",
            retrieval_top_k=_default_top_k(forced_mode),
            answer_style=_answer_style(forced_mode),
            scores={mode: (10 if mode == forced_mode else 0) for mode in _MODE_ORDER},
            signals=[f"{forced_mode}:forced override"],
        )

    lowered = query.strip().lower()
    constrained = bool(str(constraints).strip())
    scores: dict[QueryMode, int] = {mode: 0 for mode in _MODE_ORDER}
    signals: list[str] = []

    for mode, profiles in _ROUTE_PROFILES.items():
        similarity = semantic_similarity(lowered, profiles)
        semantic_points = int(round(similarity * 12))
        if semantic_points > 0:
            _add_score(
                scores,
                signals,
                mode,
                semantic_points,
                "semantic route profile",
            )

    if _is_information_request(lowered):
        _add_score(
            scores,
            signals,
            "summarize",
            5,
            "question asks for information rather than a go/no-go decision",
        )

    if _SUMMARY_DIRECTIVE_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "summarize",
            5,
            "summary directive",
        )

    if _EXTRACTION_DIRECTIVE_RE.search(lowered) and not _SUMMARY_DIRECTIVE_RE.search(
        lowered
    ):
        _add_score(
            scores,
            signals,
            "extract",
            8,
            "field extraction directive",
        )

    if _CLAIM_SUPPORT_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "fact_check",
            5,
            "claim verification directive",
        )

    if _RECOMMENDATION_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "recommend",
            9,
            "recommendation directive",
        )

    if _DECISION_DIRECTIVE_RE.search(lowered) and not _SUMMARY_DIRECTIVE_RE.search(
        lowered
    ):
        _add_score(
            scores,
            signals,
            "decision",
            6,
            "approval decision directive",
        )

    if _SOURCE_LOCATION_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "extract",
            4,
            "source-location structure",
        )

    if constrained and _COLLECTIVE_DECISION_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "decision",
            3,
            "explicit constraints were supplied",
        )

    if _YES_NO_RE.search(lowered) and not _ASSISTANT_REQUEST_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "fact_check",
            2,
            "yes/no phrasing suggests verification",
        )

    if _COLLECTIVE_DECISION_RE.search(lowered):
        _add_score(
            scores,
            signals,
            "decision",
            2,
            "leading phrasing suggests a decision task",
        )

    if max(scores.values()) <= 0:
        scores["summarize"] = 1
        signals.append("summarize:default informational route")

    mode = max(_MODE_ORDER, key=lambda item: (scores[item], -_MODE_ORDER.index(item)))
    rationale = _build_rationale(mode, scores, signals)
    return RoutingDecision(
        mode=mode,
        rationale=rationale,
        retrieval_top_k=_default_top_k(mode, scores=score_copy(scores)),
        answer_style=_answer_style(mode),
        scores=score_copy(scores),
        signals=signals[:6],
    )


def _build_rationale(
    mode: QueryMode,
    scores: dict[QueryMode, int],
    signals: list[str],
) -> str:
    top = ", ".join(f"{label}={scores[label]}" for label in _MODE_ORDER)
    selected = [signal for signal in signals if signal.startswith(f"{mode}:")]
    if selected:
        reason = selected[0].split(":", 1)[1]
        return f"Selected {mode} because {reason}. Scores: {top}."
    return f"Selected {mode}. Scores: {top}."


def score_copy(scores: dict[QueryMode, int]) -> dict[QueryMode, int]:
    return {mode: int(value) for mode, value in scores.items()}


def _default_top_k(
    mode: QueryMode,
    *,
    scores: dict[QueryMode, int] | None = None,
) -> int:
    base = {
        "extract": 10,
        "fact_check": 9,
        "recommend": 8,
        "decision": 8,
        "summarize": 6,
    }[mode]
    if not scores:
        return base
    if mode == "extract" and scores.get("extract", 0) >= 6:
        return 12
    if mode == "fact_check" and scores.get("fact_check", 0) >= 6:
        return 10
    if mode == "decision" and scores.get("decision", 0) >= 7:
        return 9
    return base


def _answer_style(mode: QueryMode) -> str:
    if mode == "extract":
        return "Answer with direct extraction, citations, and minimal inference."
    if mode == "fact_check":
        return "Verify the claim explicitly and abstain when evidence does not directly support it."
    if mode == "recommend":
        return "Provide a bounded recommendation, assumptions, tradeoffs, and concrete guardrails."
    if mode == "decision":
        return "Return a clear decision with cited support, risks, mitigations, and explicit unknowns."
    return "Provide a concise grounded summary with citations and explicit caveats."


def mode_prompt_brief(route: RoutingDecision) -> str:
    return (
        f"Route mode: {route.mode}. "
        f"Expected answer style: {route.answer_style} "
    ).strip()


__all__ = [
    "QueryMode",
    "RoutingDecision",
    "mode_prompt_brief",
    "route_query",
]
