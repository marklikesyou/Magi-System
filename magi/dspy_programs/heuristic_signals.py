from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from contextvars import ContextVar
import re
from typing import Iterator

from magi.core.routing import QueryMode, route_query
from magi.core.semantic import semantic_similarity

from .grounding import query_support_terms

NEGATION_WORDS = {"no", "not", "never", "none", "without", "zero"}
NEGATION_PHRASES = ("n't", "no ", "not ", "never ", "without ", "zero ")
_FIELD_QUESTION_AUXILIARIES = {"is", "are", "was", "were", "do", "does", "did"}
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
CLAIM_EVENT_WINDOW = 4
HARMFUL_INTENT_PROFILES = (
    "unsafe request to evade controls steal credentials abuse systems or create harm",
    "bypass administrative controls and extract private passwords or access tokens",
)
EVIDENCE_GAP_PROFILES = (
    "answer says evidence is missing insufficient unsupported unclear or cannot determine",
    "source does not state or specify the requested detail",
)
REFUSAL_PROFILES = (
    "answer refuses unsafe disallowed assistance and declines the request",
)
REVISION_CUE_PROFILES = (
    "request revision clarification verification or more evidence before proceeding",
)
DECISION_DIRECTIVE_PROFILES = (
    "instruction to approve reject revise proceed or make a go no go decision",
)
SYNTHESIS_PROFILES = (
    "concise summary derived from available evidence",
)
KEY_POINT_PROFILES = (
    "brief key points bullets main takeaways from source material",
)
LIMITING_PROFILES = (
    "sentence says material is irrelevant unrelated inapplicable or unsupported",
)
CONTROL_REMOVAL_PROFILES = (
    "request removes disables bypasses omits or weakens required safeguards",
)
DECISION_SUPPORT_PROFILES = (
    "bounded action with defined scope responsible reviewer safeguards oversight monitoring rollback audit testing constraints",
)
DECISION_GAP_PROFILES = (
    "decision support is missing absent incomplete unresolved or not assigned",
)
DECISION_CRITICAL_PROFILES = (
    "decision requires authority ownership approval safeguards monitoring rollback testing risk privacy and resourcing",
)
RECOMMENDATION_SUPPORT_PROFILES = (
    "bounded recommendation with assumptions tradeoffs mitigations oversight and constraints",
)
_RAW_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_ACTIVE_ROUTE_MODE: ContextVar[QueryMode | None] = ContextVar(
    "magi_active_route_mode",
    default=None,
)


@contextmanager
def route_mode_context(mode: str | None) -> Iterator[None]:
    normalized = str(mode or "").strip().lower()
    active_mode = (
        normalized
        if normalized
        in {"summarize", "extract", "fact_check", "recommend", "decision"}
        else None
    )
    token = _ACTIVE_ROUTE_MODE.set(active_mode)  # type: ignore[arg-type]
    try:
        yield
    finally:
        _ACTIVE_ROUTE_MODE.reset(token)


def _profile_score(text: str, profiles: Sequence[str]) -> float:
    return semantic_similarity(text, profiles)


def _profile_hits(
    text: str,
    profiles: Sequence[str],
    *,
    threshold: float = 0.18,
    scale: int = 12,
) -> int:
    score = _profile_score(text, profiles)
    if score < threshold:
        return 0
    return max(1, round(score * scale))


def _route_mode(query: str) -> str:
    active_mode = _ACTIVE_ROUTE_MODE.get()
    if active_mode is not None:
        return active_mode
    return route_query(query).mode


def is_information_request(query: str) -> bool:
    return _route_mode(query) == "summarize"


def is_informational_query(query: str) -> bool:
    return is_information_request(query)


def is_harmful_query(query: str) -> bool:
    return _profile_score(query, HARMFUL_INTENT_PROFILES) >= 0.3


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
    return _profile_score(query, KEY_POINT_PROFILES) >= 0.18


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


def _event_scoped_claim_terms(text: str) -> tuple[set[str], bool]:
    terms = [
        term
        for term in query_support_terms(text)
        if term not in NEGATION_WORDS
    ]
    for index, term in enumerate(terms):
        if term not in FACT_EVENT_WORDS:
            continue
        start = max(0, index - CLAIM_EVENT_WINDOW)
        window = terms[start:index]
        if window:
            return set(window), True
    return claim_content_terms(text), False


def _claim_terms_match(query_terms: set[str], text_terms: set[str]) -> bool:
    if not query_terms or not text_terms:
        return False
    overlap = query_terms & text_terms
    if len(query_terms) <= 2:
        return len(overlap) == len(query_terms)
    return len(overlap) / len(query_terms) >= 0.6


def text_negates_query_claim(query: str, text: str) -> bool:
    if contains_negation(query) or not contains_negation(text):
        return False
    query_core_terms, query_scoped_to_event = _event_scoped_claim_terms(query)
    text_core_terms, text_scoped_to_event = _event_scoped_claim_terms(text)
    if query_scoped_to_event or text_scoped_to_event:
        return _claim_terms_match(query_core_terms, text_core_terms)
    query_terms = claim_content_terms(query)
    text_terms = claim_content_terms(text)
    return _claim_terms_match(query_terms, text_terms)


def sentence_limits_support(sentence: str) -> bool:
    return _profile_score(sentence, LIMITING_PROFILES) >= 0.2


def query_removes_evidence_requirement(query: str, evidence_text: str) -> bool:
    return (
        _profile_score(query, CONTROL_REMOVAL_PROFILES) >= 0.16
        and decision_critical_hits(evidence_text) > 0
    )


def decision_control_hits(text: str) -> int:
    return _profile_hits(text, DECISION_SUPPORT_PROFILES, threshold=0.12, scale=16)


def decision_gap_hits(text: str) -> int:
    return _profile_hits(text, DECISION_GAP_PROFILES, threshold=0.18)


def decision_critical_hits(text: str) -> int:
    return _profile_hits(text, DECISION_CRITICAL_PROFILES, threshold=0.15)


def recommendation_support_hits(text: str) -> int:
    return _profile_hits(text, RECOMMENDATION_SUPPORT_PROFILES, threshold=0.16, scale=16)


def signals_evidence_gap(text: str) -> bool:
    return _profile_score(text, EVIDENCE_GAP_PROFILES) >= 0.26


def signals_refusal(text: str) -> bool:
    return _profile_score(text, REFUSAL_PROFILES) >= 0.27


def starts_with_revision_cue(text: str) -> bool:
    lead = " ".join(query_support_terms(text)[:6])
    return _profile_score(lead, REVISION_CUE_PROFILES) >= 0.16


def supports_synthesis_wording(text: str) -> bool:
    return _profile_score(text, SYNTHESIS_PROFILES) >= 0.16


def looks_like_decision_directive(text: str) -> bool:
    lead = " ".join(query_support_terms(text)[:8])
    return _profile_score(lead, DECISION_DIRECTIVE_PROFILES) >= 0.16
