from __future__ import annotations

from collections.abc import Sequence
import re

from magi.core.routing import route_query
from magi.core.text_signals import (
    INSUFFICIENT_INFORMATION_PATTERNS,
    REFUSAL_PATTERNS,
    contains_pattern,
)

from .grounding import query_support_terms

HARMFUL_PATTERNS = (
    "bypass",
    "hack",
    "exploit",
    "steal",
    "malware",
    "phish",
    "phishing",
    "weapon",
    "harm",
    "kill",
    "fraud",
    "credential",
    "credentials",
    "login token",
    "login tokens",
)
INSUFFICIENT_PATTERNS = INSUFFICIENT_INFORMATION_PATTERNS + (
    "does not contain",
    "need additional sources",
    "unclear from the evidence",
    "additional credible sources",
    "further details are needed",
    "lack of comprehensive information",
    "partial insights",
    "untrustworthy nature of the evidence",
    "verify this information from reliable sources",
)
REVISION_LEAD_PATTERNS = (
    "revise ",
    "review ",
    "determine if ",
    "verify ",
    "clarify ",
)
DECISION_DIRECTIVE_STARTERS = (
    "approve ",
    "approve:",
    "do not approve",
    "reject ",
    "reject:",
    "revise ",
    "revise:",
    "proceed ",
    "greenlight ",
    "go/no-go",
)
SYNTHESIS_TERMS = {"concise", "derived", "summary", "summarize"}
NEGATION_WORDS = {"no", "not", "never", "none", "without", "zero"}
NEGATION_PHRASES = ("n't", "no ", "not ", "never ", "without ", "zero ")
IRRELEVANCE_TERMS = {"inapplicable", "irrelevant", "unrelated", "unsupported"}
FACT_EVENT_WORDS = {
    "detect",
    "detected",
    "happen",
    "happened",
    "log",
    "logged",
    "occur",
    "occurred",
    "record",
    "recorded",
    "report",
    "reported",
}
CONTROL_REMOVAL_TERMS = {
    "automatic",
    "automated",
    "bypass",
    "disable",
    "drop",
    "omit",
    "remove",
    "removing",
    "skip",
    "without",
}
CONTROL_REQUIREMENT_TERMS = {
    "approval",
    "approver",
    "audit",
    "budget",
    "control",
    "escalation",
    "guardrail",
    "legal",
    "monitoring",
    "owner",
    "privacy",
    "review",
    "reviewer",
    "risk",
    "rollback",
    "signoff",
    "testing",
}
RECOMMENDATION_SUPPORT_PATTERNS = (
    "proposal",
    "pilot",
    "scope",
    "budget",
    "timeline",
    "control",
    "controls",
    "oversight",
    "reviewer",
    "guardrail",
    "guardrails",
    "mitigation",
    "mitigations",
    "low-risk",
    "low risk",
)
DECISION_CONTROL_PATTERNS = (
    "proposal",
    "plan",
    "pilot",
    "trial",
    "scope",
    "scopes",
    "limited",
    "read",
    "budget",
    "duration",
    "owner",
    "approval",
    "reviewer",
    "analyst",
    "legal",
    "control",
    "controls",
    "guardrail",
    "guardrails",
    "rollback",
    "audit",
    "qa sampling",
    "escalation",
    "monitoring",
    "exit plan",
    "data access",
    "questionnaire",
    "calibration",
)
DECISION_GAP_PATTERNS = (
    "lacks",
    "lack ",
    "missing",
    "does not include",
    "doesn't include",
    "not include",
    "no rollback",
    "no owner",
    "not assigned",
    "incomplete",
    "without",
)
DECISION_CRITICAL_PATTERNS = (
    "approval",
    "approvals",
    "authority",
    "budget",
    "control",
    "controls",
    "data mapping",
    "human sign-off",
    "legal",
    "monitoring",
    "owner",
    "privacy",
    "rollback",
    "risk",
    "staffing",
    "testing",
)
_RAW_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_FIELD_QUESTION_AUXILIARIES = {"is", "are", "was", "were", "do", "does", "did"}


def pattern_hits(text: str, patterns: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(1 for pattern in patterns if pattern in lowered)


def _route_mode(query: str) -> str:
    return route_query(query).mode


def is_information_request(query: str) -> bool:
    return _route_mode(query) == "summarize"


def is_informational_query(query: str) -> bool:
    return is_information_request(query)


def is_harmful_query(query: str) -> bool:
    lowered = query.lower()
    return any(token in lowered for token in HARMFUL_PATTERNS)


def is_decision_query(query: str) -> bool:
    return _route_mode(query) == "decision"


def is_fact_check_query(query: str) -> bool:
    return _route_mode(query) == "fact_check"


def is_extractive_query(query: str) -> bool:
    return _route_mode(query) == "extract"


def is_specific_detail_query(query: str) -> bool:
    if _route_mode(query) in {"extract", "fact_check"}:
        return True
    terms = query_support_terms(query)
    has_possessive = "'s" in query or "’s" in query
    raw_tokens = _RAW_TOKEN_RE.findall(query)
    has_identifier = any(any(character.isdigit() for character in token) for token in raw_tokens)
    is_compact_field_question = (
        len(terms) <= 3
        and len(raw_tokens) >= 2
        and raw_tokens[0].lower() == "what"
        and raw_tokens[1].lower() in _FIELD_QUESTION_AUXILIARIES
    )
    return has_possessive or has_identifier or is_compact_field_question


def is_field_extraction_query(query: str) -> bool:
    return is_extractive_query(query)


def wants_key_points(query: str) -> bool:
    lowered = query.lower()
    return any(
        pattern in lowered
        for pattern in (
            "key points",
            "main points",
            "bullets",
            "bullet",
            "in a hurry",
            "quick points",
        )
    )


def contains_negation(text: str) -> bool:
    lowered = text.lower()
    if any(phrase in lowered for phrase in NEGATION_PHRASES):
        return True
    return bool(NEGATION_WORDS & set(query_support_terms(text)))


def claim_content_terms(text: str) -> set[str]:
    return {
        term
        for term in query_support_terms(text)
        if term not in NEGATION_WORDS and term not in FACT_EVENT_WORDS
    }


def text_negates_query_claim(query: str, text: str) -> bool:
    if contains_negation(query) or not contains_negation(text):
        return False
    query_terms = claim_content_terms(query)
    text_terms = claim_content_terms(text)
    if not query_terms or not text_terms:
        return False
    overlap = query_terms & text_terms
    if len(query_terms) <= 2:
        return len(overlap) == len(query_terms)
    return len(overlap) / len(query_terms) >= 0.6


def sentence_limits_support(sentence: str) -> bool:
    terms = set(query_support_terms(sentence))
    return bool(terms & IRRELEVANCE_TERMS)


def query_removes_evidence_requirement(query: str, evidence_text: str) -> bool:
    query_terms = set(query_support_terms(query))
    evidence_terms = set(query_support_terms(evidence_text))
    if not (query_terms & CONTROL_REMOVAL_TERMS):
        return False
    shared_requirements = query_terms & evidence_terms & CONTROL_REQUIREMENT_TERMS
    return bool(shared_requirements)


def decision_control_hits(text: str) -> int:
    return pattern_hits(text, DECISION_CONTROL_PATTERNS)


def decision_gap_hits(text: str) -> int:
    return pattern_hits(text, DECISION_GAP_PATTERNS)


def decision_critical_hits(text: str) -> int:
    return pattern_hits(text, DECISION_CRITICAL_PATTERNS)


def recommendation_support_hits(text: str) -> int:
    return pattern_hits(text, RECOMMENDATION_SUPPORT_PATTERNS)


def signals_evidence_gap(text: str) -> bool:
    return contains_pattern(text, INSUFFICIENT_PATTERNS)


def signals_refusal(text: str) -> bool:
    return contains_pattern(text, REFUSAL_PATTERNS)


def starts_with_revision_cue(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(pattern) for pattern in REVISION_LEAD_PATTERNS)


def supports_synthesis_wording(text: str) -> bool:
    terms = set(query_support_terms(text))
    return bool(terms & SYNTHESIS_TERMS)


def looks_like_decision_directive(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(DECISION_DIRECTIVE_STARTERS)
