from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

QueryMode = Literal["summarize", "extract", "fact_check", "recommend", "decision"]
_MODE_ORDER: tuple[QueryMode, ...] = (
    "extract",
    "fact_check",
    "decision",
    "recommend",
    "summarize",
)

_SUMMARIZE_PATTERNS = (
    "summarize",
    "summary",
    "overview",
    "explain",
    "describe",
    "clarify",
    "one sentence",
    "two sentences",
    "briefly",
    "tl;dr",
)
_EXTRACT_PATTERNS = (
    "extract",
    "list ",
    "quote",
    "exact wording",
    "exact text",
    "find the",
    "which ",
    "what page",
    "what section",
    "what clause",
    "pull out",
    "show the line",
    "locate",
)
_FACT_CHECK_PATTERNS = (
    "fact check",
    "fact-check",
    "verify",
    "confirm whether",
    "is it true",
    "true or false",
    "accurate",
    "guarantee",
    "guaranteed",
    "does the evidence show",
    "does the evidence say",
    "does the source say",
    "supported by the evidence",
)
_RECOMMEND_PATTERNS = (
    "recommend",
    "recommendation",
    "best approach",
    "best option",
    "how should",
    "what should",
    "proposal",
    "plan",
    "roadmap",
    "tradeoff",
    "trade-off",
    "compare",
    "options",
    "suggest",
)
_DECISION_PATTERNS = (
    "should we",
    "approve",
    "reject",
    "revise",
    "go/no-go",
    "go no go",
    "proceed",
    "launch",
    "rollout",
    "deploy",
    "adopt",
    "move forward",
    "greenlight",
    "pilot",
    "ship",
)
_RISK_PATTERNS = (
    "risk",
    "risks",
    "mitigation",
    "mitigations",
    "guardrail",
    "guardrails",
    "blast radius",
    "impact",
)
_YES_NO_STARTERS = ("is ", "are ", "do ", "does ", "did ", "was ", "were ", "can ")
_DECISION_STARTERS = ("should ", "can we ", "do we ", "is it safe to ")


@dataclass(frozen=True)
class RoutingDecision:
    mode: QueryMode
    rationale: str
    retrieval_top_k: int = 8
    answer_style: str = "Deliver a grounded answer with explicit uncertainty."
    scores: dict[QueryMode, int] = field(default_factory=dict)
    signals: list[str] = field(default_factory=list)


def _pattern_matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    lowered = text.lower()
    return [pattern for pattern in patterns if pattern in lowered]


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

    summarize_matches = _pattern_matches(lowered, _SUMMARIZE_PATTERNS)
    if summarize_matches:
        _add_score(
            scores,
            signals,
            "summarize",
            2 + len(summarize_matches),
            f"summary markers {', '.join(summarize_matches[:3])}",
        )

    extract_matches = _pattern_matches(lowered, _EXTRACT_PATTERNS)
    if extract_matches:
        _add_score(
            scores,
            signals,
            "extract",
            2 + len(extract_matches),
            f"extraction markers {', '.join(extract_matches[:3])}",
        )

    fact_matches = _pattern_matches(lowered, _FACT_CHECK_PATTERNS)
    if fact_matches:
        _add_score(
            scores,
            signals,
            "fact_check",
            3 + len(fact_matches),
            f"verification markers {', '.join(fact_matches[:3])}",
        )

    recommend_matches = _pattern_matches(lowered, _RECOMMEND_PATTERNS)
    if recommend_matches:
        _add_score(
            scores,
            signals,
            "recommend",
            2 + len(recommend_matches),
            f"recommendation markers {', '.join(recommend_matches[:3])}",
        )

    decision_matches = _pattern_matches(lowered, _DECISION_PATTERNS)
    if decision_matches:
        _add_score(
            scores,
            signals,
            "decision",
            3 + len(decision_matches),
            f"decision markers {', '.join(decision_matches[:3])}",
        )

    risk_matches = _pattern_matches(lowered, _RISK_PATTERNS)
    if risk_matches:
        _add_score(
            scores,
            signals,
            "decision",
            2,
            f"risk markers {', '.join(risk_matches[:3])}",
        )
        _add_score(
            scores,
            signals,
            "recommend",
            1,
            "risk framing can imply a recommendation task",
        )

    if constrained:
        _add_score(
            scores,
            signals,
            "decision",
            3,
            "explicit constraints were supplied",
        )

    if lowered.startswith(_YES_NO_STARTERS) and lowered.endswith("?"):
        _add_score(
            scores,
            signals,
            "fact_check",
            2,
            "yes/no phrasing suggests verification",
        )

    if lowered.startswith(_DECISION_STARTERS):
        _add_score(
            scores,
            signals,
            "decision",
            2,
            "leading phrasing suggests a decision task",
        )

    if "page " in lowered or "section " in lowered or "clause " in lowered:
        _add_score(
            scores,
            signals,
            "extract",
            2,
            "source-location references imply extraction",
        )

    if "compare" in lowered and "option" in lowered:
        _add_score(
            scores,
            signals,
            "recommend",
            2,
            "compare/options phrasing implies recommendation synthesis",
        )

    if "one sentence" in lowered or "two sentences" in lowered:
        _add_score(
            scores,
            signals,
            "summarize",
            2,
            "brevity request implies summarization",
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
    score_text = ", ".join(
        f"{label}={route.scores.get(label, 0)}" for label in _MODE_ORDER
    )
    signal_text = "; ".join(route.signals) if route.signals else "none"
    return (
        f"Route mode: {route.mode}. "
        f"Reason: {route.rationale} "
        f"Expected answer style: {route.answer_style} "
        f"Routing signals: {signal_text}. "
        f"Route scores: {score_text}."
    ).strip()


__all__ = [
    "QueryMode",
    "RoutingDecision",
    "mode_prompt_brief",
    "route_query",
]
