from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from typing import Mapping, Sequence

from .schemas import RetrievedEvidence

_QUERY_TOKEN_RE = re.compile(r"[a-z0-9]+")
_GROUNDING_STOPWORDS = {
    "additional",
    "about",
    "against",
    "also",
    "and",
    "answer",
    "answers",
    "are",
    "been",
    "citation",
    "citations",
    "current",
    "detail",
    "details",
    "evidence",
    "from",
    "fully",
    "have",
    "include",
    "information",
    "into",
    "missing",
    "more",
    "need",
    "needed",
    "query",
    "response",
    "retrieved",
    "reliable",
    "review",
    "source",
    "sources",
    "specific",
    "than",
    "that",
    "their",
    "them",
    "they",
    "this",
    "trustworthy",
    "what",
    "with",
}
_QUERY_SUPPORT_STOPWORDS = _GROUNDING_STOPWORDS | {
    "a",
    "analyze",
    "analysis",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "clarify",
    "compare",
    "describe",
    "detail",
    "details",
    "do",
    "does",
    "explain",
    "for",
    "give",
    "how",
    "i",
    "in",
    "is",
    "it",
    "latest",
    "month",
    "my",
    "next",
    "of",
    "on",
    "one",
    "outline",
    "overview",
    "our",
    "please",
    "provide",
    "review",
    "sentence",
    "sentences",
    "show",
    "should",
    "summarize",
    "summary",
    "tell",
    "the",
    "to",
    "two",
    "why",
    "your",
}


def _normalize_query_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def query_support_terms(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()
    return [
        _normalize_query_token(token)
        for token in _QUERY_TOKEN_RE.findall(normalized)
        if len(token) > 1 and token not in _QUERY_SUPPORT_STOPWORDS
    ]


def _query_support_text(text: str) -> str:
    return " ".join(query_support_terms(text))


def _character_ngram_counts(text: str, n: int = 3) -> dict[str, int]:
    normalized = _query_support_text(text)
    if not normalized:
        return {}
    padded = f"  {normalized} "
    counts: dict[str, int] = {}
    for index in range(len(padded) - n + 1):
        gram = padded[index : index + n]
        counts[gram] = counts.get(gram, 0) + 1
    return counts


def _cosine_similarity_counts(
    left: Mapping[str, int], right: Mapping[str, int]
) -> float:
    if not left or not right:
        return 0.0
    keys = set(left) | set(right)
    dot = sum(left.get(key, 0) * right.get(key, 0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _phrase_support_similarity(query: str, evidence_text: str) -> float:
    return _cosine_similarity_counts(
        _character_ngram_counts(query),
        _character_ngram_counts(evidence_text),
    )


def _token_support_similarity(query_token: str, evidence_token: str) -> float:
    if query_token == evidence_token:
        return 1.0
    return SequenceMatcher(None, query_token, evidence_token).ratio()


def _normalize_retrieval_score(score: object) -> float:
    try:
        raw = float(score) if isinstance(score, (int, float)) else float(str(score))
    except (TypeError, ValueError):
        return 0.0
    if raw <= 0.0:
        return 0.0
    if raw <= 1.0:
        return raw
    return raw / (1.0 + raw)


def _query_term_support_scores(
    query_terms: Sequence[str], evidence_text: str
) -> list[float]:
    if not query_terms:
        return []
    evidence_terms = query_support_terms(evidence_text)
    if not evidence_terms:
        return [0.0] * len(query_terms)
    scores: list[float] = []
    for query_term in query_terms:
        best = max(
            _token_support_similarity(query_term, evidence_term)
            for evidence_term in evidence_terms
        )
        scores.append(best)
    return scores


def _evidence_match_score(
    query: str, item: RetrievedEvidence
) -> tuple[float, float, float, float]:
    query_terms = query_support_terms(query)
    if not query_terms:
        return (0.0, 0.0, 0.0, _normalize_retrieval_score(item.score))
    term_scores = _query_term_support_scores(query_terms, item.text)
    coverage = sum(1 for score in term_scores if score >= 0.84) / len(query_terms)
    average_similarity = sum(term_scores) / len(query_terms)
    phrase_similarity = _phrase_support_similarity(query, item.text)
    retrieval_strength = _normalize_retrieval_score(item.score)
    support_score = (
        (0.45 * coverage)
        + (0.25 * average_similarity)
        + (0.2 * phrase_similarity)
        + (0.1 * retrieval_strength)
    )
    return (support_score, coverage, average_similarity, phrase_similarity)


def rank_supporting_evidence(
    query: str, evidence: Sequence[RetrievedEvidence]
) -> list[RetrievedEvidence]:
    ranked: list[
        tuple[tuple[float, float, float, float], int, RetrievedEvidence]
    ] = []
    for index, item in enumerate(evidence):
        ranked.append((_evidence_match_score(query, item), index, item))
    ranked.sort(
        key=lambda entry: (
            entry[0][0],
            entry[0][1],
            entry[0][2],
            entry[0][3],
            _normalize_retrieval_score(entry[2].score),
            -entry[1],
        ),
        reverse=True,
    )
    supporting: list[RetrievedEvidence] = []
    for rank, _index, item in ranked:
        if rank[0] >= 0.2 or rank[1] >= 0.34 or rank[2] >= 0.55:
            supporting.append(item)
    return supporting


def _query_evidence_coverage(
    query: str, evidence: Sequence[RetrievedEvidence]
) -> tuple[float, float, float]:
    query_terms = query_support_terms(query)
    if not query_terms:
        return 0.0, 0.0, 0.0
    best_term_scores = [0.0] * len(query_terms)
    best_support_score = 0.0
    for item in evidence:
        support_score, _coverage, _average, _phrase = _evidence_match_score(query, item)
        best_support_score = max(best_support_score, support_score)
        term_scores = _query_term_support_scores(query_terms, item.text)
        for index, term_score in enumerate(term_scores):
            best_term_scores[index] = max(best_term_scores[index], term_score)
    if not any(best_term_scores):
        return 0.0, 0.0, best_support_score
    fuzzy_coverage = (
        sum(1 for score in best_term_scores if score >= 0.84) / len(best_term_scores)
    )
    average_similarity = sum(best_term_scores) / len(best_term_scores)
    return fuzzy_coverage, average_similarity, best_support_score


def evidence_directly_addresses_query(
    query: str, evidence: Sequence[RetrievedEvidence]
) -> bool:
    if not evidence:
        return False
    query_terms = set(query_support_terms(query))
    if not query_terms:
        return True
    fuzzy_coverage, average_similarity, best_support_score = _query_evidence_coverage(
        query, evidence
    )
    if len(query_terms) == 1:
        return best_support_score >= 0.5 or average_similarity >= 0.84
    if len(query_terms) == 2:
        return fuzzy_coverage >= 0.5 and average_similarity >= 0.55
    return (
        fuzzy_coverage >= 0.6
        or best_support_score >= 0.58
        or (fuzzy_coverage >= 0.4 and average_similarity >= 0.72)
    )


def _citation_labels(evidence: Sequence[RetrievedEvidence]) -> set[str]:
    return {item.citation for item in evidence if item.citation}


def contains_evidence_citation(
    text: str, evidence: Sequence[RetrievedEvidence]
) -> bool:
    return any(citation in text for citation in _citation_labels(evidence))


def _grounding_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{4,}", text.lower())
        if token not in _GROUNDING_STOPWORDS
    }


def grounding_overlap(text: str, evidence: Sequence[RetrievedEvidence]) -> tuple[int, float]:
    answer_tokens = _grounding_tokens(text)
    if not answer_tokens:
        return 0, 0.0
    evidence_tokens: set[str] = set()
    for item in evidence:
        evidence_tokens.update(_grounding_tokens(item.text))
    if not evidence_tokens:
        return 0, 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap), len(overlap) / len(answer_tokens)


def unsupported_grounding_tokens(
    query: str, text: str, evidence: Sequence[RetrievedEvidence]
) -> set[str]:
    supported = _grounding_tokens(query)
    for item in evidence:
        supported.update(_grounding_tokens(item.text))
    return _grounding_tokens(text) - supported


def support_strength(evidence: Sequence[RetrievedEvidence]) -> float:
    if not evidence:
        return 0.0
    return min(1.0, 0.35 + 0.2 * len(evidence))


__all__ = [
    "contains_evidence_citation",
    "evidence_directly_addresses_query",
    "grounding_overlap",
    "query_support_terms",
    "rank_supporting_evidence",
    "support_strength",
    "unsupported_grounding_tokens",
]
