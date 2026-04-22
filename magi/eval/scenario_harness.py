from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError, model_validator

from magi.app.service import run_chat_session
from magi.core.rag import default_formatter
from magi.core.safety import analyze_safety, is_blocked
from magi.core.vectorstore import RetrievedChunk
from magi.dspy_programs.personas import (
    clear_cache,
    get_token_stats,
    reset_token_tracking,
)
from magi.eval.metrics import accuracy, answer_support_score, citation_hit_rate

VALID_VERDICTS = {"approve", "reject", "revise"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_PERSONAS = {"melchior", "balthasar", "casper"}
_CITATION_RE = re.compile(r"\[\d+\]")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "one",
    "or",
    "our",
    "sentence",
    "sentences",
    "should",
    "the",
    "to",
    "two",
    "what",
    "with",
    "your",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()


def _query_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [
        token
        for token in _TOKEN_RE.findall(normalized)
        if len(token) > 1 and token not in _STOPWORDS
    ]


def _query_phrases(tokens: list[str]) -> set[str]:
    return {
        " ".join(tokens[index : index + 2])
        for index in range(len(tokens) - 1)
        if tokens[index] != tokens[index + 1]
    }


class ScenarioEvidence(BaseModel):
    source: str = "scenario"
    text: str
    score: float = Field(default=1.0, ge=0.0)
    persona: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"text": data}
        return data

    @model_validator(mode="after")
    def _normalize(self) -> "ScenarioEvidence":
        source = self.source.strip() or "scenario"
        text = self.text.strip()
        if not text:
            raise ValueError("scenario evidence text must not be empty")
        persona = (self.persona or "").strip().lower()
        if persona and persona not in VALID_PERSONAS:
            raise ValueError(f"invalid persona '{self.persona}'")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "persona", persona or None)
        return self


class ScenarioChecks(BaseModel):
    required_terms_all: List[str] = Field(default_factory=list)
    required_terms_any: List[str] = Field(default_factory=list)
    required_sources_all: List[str] = Field(default_factory=list)
    required_sources_any: List[str] = Field(default_factory=list)
    forbidden_terms: List[str] = Field(default_factory=list)
    min_citations: int = Field(default=0, ge=0)
    min_citation_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    min_answer_support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    require_human_review: bool | None = None
    blocked_evidence_min: int = Field(default=0, ge=0)
    blocked_evidence_max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _normalize(self) -> "ScenarioChecks":
        for field_name in (
            "required_terms_all",
            "required_terms_any",
            "required_sources_all",
            "required_sources_any",
            "forbidden_terms",
        ):
            values = [
                str(item).strip()
                for item in getattr(self, field_name)
                if str(item).strip()
            ]
            object.__setattr__(self, field_name, values)
        if (
            self.blocked_evidence_max is not None
            and self.blocked_evidence_max < self.blocked_evidence_min
        ):
            raise ValueError("blocked_evidence_max must be >= blocked_evidence_min")
        return self


class ScenarioCase(BaseModel):
    id: str
    description: str = ""
    query: str
    constraints: str = ""
    evidence: List[ScenarioEvidence] = Field(default_factory=list)
    expected_verdict: str
    expected_residual_risk: str | None = None
    checks: ScenarioChecks = Field(default_factory=ScenarioChecks)
    tags: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "ScenarioCase":
        case_id = self.id.strip()
        if not case_id:
            raise ValueError("scenario id must not be empty")
        verdict = self.expected_verdict.strip().lower()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid expected verdict '{self.expected_verdict}'")
        risk = (self.expected_residual_risk or "").strip().lower()
        if risk and risk not in VALID_RISK_LEVELS:
            raise ValueError(
                f"invalid expected residual risk '{self.expected_residual_risk}'"
            )
        query = self.query.strip()
        if not query:
            raise ValueError("scenario query must not be empty")
        object.__setattr__(self, "id", case_id)
        object.__setattr__(self, "description", self.description.strip())
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "constraints", self.constraints.strip())
        object.__setattr__(self, "expected_verdict", verdict)
        object.__setattr__(self, "expected_residual_risk", risk or None)
        object.__setattr__(
            self, "tags", [str(tag).strip() for tag in self.tags if str(tag).strip()]
        )
        return self


class ScenarioDataset(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    cases: List[ScenarioCase] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data: Any) -> Any:
        if not data:
            return {"cases": []}
        if isinstance(data, dict):
            payload = dict(data)
            if "cases" not in payload:
                if "scenarios" in payload:
                    payload["cases"] = payload.pop("scenarios")
                elif "tasks" in payload:
                    payload["cases"] = payload.pop("tasks")
            return payload
        return data

    @model_validator(mode="after")
    def _validate(self) -> "ScenarioDataset":
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("scenario identifiers must be unique")
        if not self.cases:
            raise ValueError("scenario dataset contains no cases")
        return self


class ScenarioCheckResult(BaseModel):
    name: str
    passed: bool
    details: str


class ScenarioCaseResult(BaseModel):
    id: str
    description: str
    query: str
    tags: List[str] = Field(default_factory=list)
    passed: bool
    expected_verdict: str
    predicted_verdict: str
    verdict_match: bool
    expected_residual_risk: str | None = None
    predicted_residual_risk: str
    citation_count: int
    citation_hit_rate: float = 0.0
    answer_support_score: float = 0.0
    answer_supported: bool = False
    requires_human_review: bool = False
    review_reason: str = ""
    safe_evidence_count: int
    blocked_evidence_count: int
    retrieved_chunk_count: int = 0
    retrieved_relevant_chunk_count: int = 0
    expected_retrieval_sources: List[str] = Field(default_factory=list)
    first_expected_source_rank: int | None = None
    retrieval_source_recall: float = 0.0
    retrieved_sources: List[str] = Field(default_factory=list)
    checks: List[ScenarioCheckResult] = Field(default_factory=list)
    final_answer: str
    justification: str
    next_steps: List[str] = Field(default_factory=list)
    persona_stances: Dict[str, str] = Field(default_factory=dict)


class ScenarioSummary(BaseModel):
    total_cases: int
    passed_cases: int
    overall_score: float
    verdict_accuracy: float
    requirement_pass_rate: float
    requested_mode: str
    effective_mode: str
    model: str
    total_requirements: int
    passed_requirements: int
    retrieval_evaluable_cases: int = 0
    cases_with_retrieval_hits: int = 0
    retrieval_hit_rate: float = 0.0
    retrieval_ranked_cases: int = 0
    retrieval_top_source_accuracy: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_source_recall: float = 0.0
    answer_support_evaluable_cases: int = 0
    average_citation_hit_rate: float = 0.0
    average_answer_support_score: float = 0.0
    supported_answer_rate: float = 0.0
    token_stats: Dict[str, Any] = Field(default_factory=dict)


class ScenarioReport(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: ScenarioSummary
    cases: List[ScenarioCaseResult] = Field(default_factory=list)


@dataclass
class _ScenarioExecution:
    case_result: ScenarioCaseResult
    effective_mode: str
    resolved_model: str
    retrieval_evaluable: bool = False
    retrieval_hit: bool = False
    retrieval_ranked: bool = False
    retrieval_top_source_hit: bool = False
    retrieval_reciprocal_rank: float = 0.0
    retrieval_source_recall: float = 0.0
    answer_support_evaluable: bool = False
    citation_hit_rate: float = 0.0
    answer_support_score: float = 0.0
    answer_supported: bool = False


@dataclass
class _ScenarioCounters:
    retrieval_evaluable_cases: int = 0
    cases_with_retrieval_hits: int = 0
    retrieval_ranked_cases: int = 0
    retrieval_top_source_hits: int = 0
    retrieval_rr_total: float = 0.0
    retrieval_source_recall_total: float = 0.0
    answer_support_evaluable_cases: int = 0
    citation_hit_rate_total: float = 0.0
    answer_support_score_total: float = 0.0
    supported_answers: int = 0

    def record(self, execution: _ScenarioExecution) -> None:
        if execution.retrieval_evaluable:
            self.retrieval_evaluable_cases += 1
            if execution.retrieval_hit:
                self.cases_with_retrieval_hits += 1
        if execution.retrieval_ranked:
            self.retrieval_ranked_cases += 1
            self.retrieval_source_recall_total += execution.retrieval_source_recall
            self.retrieval_rr_total += execution.retrieval_reciprocal_rank
            if execution.retrieval_top_source_hit:
                self.retrieval_top_source_hits += 1
        if execution.answer_support_evaluable:
            self.answer_support_evaluable_cases += 1
            self.citation_hit_rate_total += execution.citation_hit_rate
            self.answer_support_score_total += execution.answer_support_score
            if execution.answer_supported:
                self.supported_answers += 1


class ScenarioRetriever:
    def __init__(self, evidence: Sequence[ScenarioEvidence]):
        self._evidence = list(evidence)

    @staticmethod
    def _score_evidence(query: str, item: ScenarioEvidence) -> tuple[float, int, float]:
        query_tokens = _query_terms(query)
        if not query_tokens:
            return (0.0, 0, item.score)
        evidence_tokens = set(_query_terms(item.text))
        token_overlap = len(evidence_tokens.intersection(query_tokens))
        coverage = token_overlap / max(1, len(set(query_tokens)))
        normalized_text = _normalize_text(item.text)
        phrase_hits = sum(
            1 for phrase in _query_phrases(query_tokens) if phrase in normalized_text
        )
        lexical_score = coverage + (0.2 * phrase_hits)
        return (lexical_score, token_overlap, item.score)

    def retrieve(
        self,
        query: str,
        *,
        persona: str | None = None,
        top_k: int = 8,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> list[RetrievedChunk]:
        del metadata_filters
        persona_key = (persona or "").strip().lower() or None
        ranked: list[tuple[tuple[float, int, float], int, ScenarioEvidence]] = []
        for index, item in enumerate(self._evidence, start=1):
            if item.persona and item.persona != persona_key:
                continue
            ranked.append((self._score_evidence(query, item), index, item))
        ranked.sort(
            key=lambda entry: (
                entry[0][0],
                entry[0][1],
                entry[0][2],
                -entry[1],
            )
            ,
            reverse=True,
        )
        chunks: list[RetrievedChunk] = []
        for rank, index, item in ranked[:top_k]:
            chunks.append(
                RetrievedChunk(
                    document_id=f"{item.source}::{index}",
                    text=item.text,
                    score=max(0.0, rank[0] + (0.01 * rank[2]) - (index * 1e-6)),
                    metadata={
                        "source": item.source,
                        "persona": item.persona or "",
                        "query_overlap": rank[1],
                    },
                )
            )
        return chunks

    def __call__(
        self,
        query: str,
        *,
        persona: str | None = None,
        top_k: int = 8,
        metadata_filters: Mapping[str, object] | None = None,
    ) -> str:
        return default_formatter(
            self.retrieve(
                query,
                persona=persona,
                top_k=top_k,
                metadata_filters=metadata_filters,
            )
        )


def load_scenario_dataset(path: Path) -> ScenarioDataset:
    if not path.exists():
        raise FileNotFoundError(f"scenario dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        return ScenarioDataset.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid scenario dataset: {exc}") from exc


def write_scenario_report(report: ScenarioReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def _evidence_safety_counts(evidence: Sequence[ScenarioEvidence]) -> tuple[int, int]:
    safe = 0
    blocked = 0
    for item in evidence:
        report = analyze_safety(item.text, client=False, stage="retrieval")
        if is_blocked(report):
            blocked += 1
        else:
            safe += 1
    return safe, blocked


def _combined_answer_text(
    final_answer: str, justification: str, next_steps: Sequence[str]
) -> str:
    return "\n".join(
        part for part in (final_answer, justification, " ".join(next_steps)) if part
    ).strip()


def _retrieval_details(chunks: Sequence[RetrievedChunk]) -> tuple[list[str], int]:
    sources: list[str] = []
    relevant = 0
    seen_sources: set[str] = set()
    for chunk in chunks:
        source = str(chunk.metadata.get("source", "")).strip()
        if source and source not in seen_sources:
            seen_sources.add(source)
            sources.append(source)
        overlap = chunk.metadata.get("query_overlap", 0)
        if isinstance(overlap, bool):
            overlap_value = int(overlap)
        elif isinstance(overlap, (int, float)):
            overlap_value = int(overlap)
        else:
            try:
                overlap_value = int(str(overlap or 0))
            except (TypeError, ValueError):
                overlap_value = 0
        if overlap_value > 0:
            relevant += 1
    return sources, relevant


def _answer_grounding(
    text: str,
    chunks: Sequence[RetrievedChunk],
) -> tuple[float, float, bool]:
    hit_rate = citation_hit_rate(text, len(chunks))
    support_score = answer_support_score(text, (chunk.text for chunk in chunks))
    supported = bool(chunks) and hit_rate > 0.0 and support_score >= 0.2
    return hit_rate, support_score, supported


def _retrieval_expectations(
    case: ScenarioCase, retrieved_sources: Sequence[str]
) -> tuple[list[str], int | None, float]:
    expected_all = [
        _normalize_text(source) for source in case.checks.required_sources_all if source
    ]
    expected_any = [
        _normalize_text(source) for source in case.checks.required_sources_any if source
    ]
    expected_union = list(dict.fromkeys(expected_all + expected_any))
    if not expected_union:
        return [], None, 0.0

    raw_expected = list(
        dict.fromkeys(
            [source for source in case.checks.required_sources_all if source]
            + [source for source in case.checks.required_sources_any if source]
        )
    )
    normalized_retrieved = [
        _normalize_text(source) for source in retrieved_sources if str(source).strip()
    ]
    first_expected_rank: int | None = None
    for index, source in enumerate(normalized_retrieved, start=1):
        if source in expected_union:
            first_expected_rank = index
            break

    recall_parts: list[float] = []
    if expected_all:
        matched_all = sum(1 for source in expected_all if source in normalized_retrieved)
        recall_parts.append(matched_all / len(expected_all))
    if expected_any:
        recall_parts.append(
            1.0 if any(source in normalized_retrieved for source in expected_any) else 0.0
        )
    source_recall = (
        sum(recall_parts) / len(recall_parts) if recall_parts else 0.0
    )
    return raw_expected, first_expected_rank, source_recall


def _evaluate_checks(
    case: ScenarioCase,
    *,
    predicted_residual_risk: str,
    combined_text: str,
    citation_count: int,
    citation_hit_rate: float,
    answer_support_score: float,
    requires_human_review: bool,
    blocked_evidence_count: int,
    retrieved_sources: Sequence[str],
) -> List[ScenarioCheckResult]:
    checks: list[ScenarioCheckResult] = []
    lowered = _normalize_text(combined_text)
    normalized_sources = {_normalize_text(source) for source in retrieved_sources}

    if case.expected_residual_risk is not None:
        passed = predicted_residual_risk == case.expected_residual_risk
        checks.append(
            ScenarioCheckResult(
                name="expected_residual_risk",
                passed=passed,
                details=(
                    f"expected {case.expected_residual_risk}, got {predicted_residual_risk}"
                    if not passed
                    else f"matched {predicted_residual_risk}"
                ),
            )
        )

    if case.checks.required_terms_all:
        missing = [
            term
            for term in case.checks.required_terms_all
            if _normalize_text(term) not in lowered
        ]
        checks.append(
            ScenarioCheckResult(
                name="required_terms_all",
                passed=not missing,
                details="all required terms present"
                if not missing
                else f"missing: {', '.join(missing)}",
            )
        )

    if case.checks.required_terms_any:
        matches = [
            term
            for term in case.checks.required_terms_any
            if _normalize_text(term) in lowered
        ]
        checks.append(
            ScenarioCheckResult(
                name="required_terms_any",
                passed=bool(matches),
                details=(
                    f"matched: {', '.join(matches)}"
                    if matches
                    else f"expected one of: {', '.join(case.checks.required_terms_any)}"
                ),
            )
        )

    if case.checks.required_sources_all:
        missing = [
            source
            for source in case.checks.required_sources_all
            if _normalize_text(source) not in normalized_sources
        ]
        checks.append(
            ScenarioCheckResult(
                name="required_sources_all",
                passed=not missing,
                details="all required sources retrieved"
                if not missing
                else f"missing sources: {', '.join(missing)}",
            )
        )

    if case.checks.required_sources_any:
        matches = [
            source
            for source in case.checks.required_sources_any
            if _normalize_text(source) in normalized_sources
        ]
        checks.append(
            ScenarioCheckResult(
                name="required_sources_any",
                passed=bool(matches),
                details=(
                    f"retrieved sources: {', '.join(matches)}"
                    if matches
                    else f"expected one of: {', '.join(case.checks.required_sources_any)}"
                ),
            )
        )

    if case.checks.forbidden_terms:
        offending = [
            term
            for term in case.checks.forbidden_terms
            if _normalize_text(term) in lowered
        ]
        checks.append(
            ScenarioCheckResult(
                name="forbidden_terms",
                passed=not offending,
                details="no forbidden terms detected"
                if not offending
                else f"found: {', '.join(offending)}",
            )
        )

    if case.checks.min_citations:
        passed = citation_count >= case.checks.min_citations
        checks.append(
            ScenarioCheckResult(
                name="min_citations",
                passed=passed,
                details=(
                    f"expected at least {case.checks.min_citations}, got {citation_count}"
                    if not passed
                    else f"found {citation_count} citation(s)"
                ),
            )
        )

    if case.checks.min_citation_hit_rate:
        passed = citation_hit_rate >= case.checks.min_citation_hit_rate
        checks.append(
            ScenarioCheckResult(
                name="min_citation_hit_rate",
                passed=passed,
                details=(
                    f"expected at least {case.checks.min_citation_hit_rate:.2f}, got {citation_hit_rate:.2f}"
                    if not passed
                    else f"citation hit rate {citation_hit_rate:.2f}"
                ),
            )
        )

    if case.checks.min_answer_support_score:
        passed = answer_support_score >= case.checks.min_answer_support_score
        checks.append(
            ScenarioCheckResult(
                name="min_answer_support_score",
                passed=passed,
                details=(
                    f"expected at least {case.checks.min_answer_support_score:.2f}, got {answer_support_score:.2f}"
                    if not passed
                    else f"answer support score {answer_support_score:.2f}"
                ),
            )
        )

    if case.checks.require_human_review is not None:
        passed = requires_human_review is case.checks.require_human_review
        checks.append(
            ScenarioCheckResult(
                name="require_human_review",
                passed=passed,
                details=(
                    f"expected require_human_review={case.checks.require_human_review}, got {requires_human_review}"
                    if not passed
                    else f"require_human_review={requires_human_review}"
                ),
            )
        )

    if case.checks.blocked_evidence_min or case.checks.blocked_evidence_max is not None:
        within_min = blocked_evidence_count >= case.checks.blocked_evidence_min
        within_max = (
            True
            if case.checks.blocked_evidence_max is None
            else blocked_evidence_count <= case.checks.blocked_evidence_max
        )
        passed = within_min and within_max
        expected = f">= {case.checks.blocked_evidence_min}"
        if case.checks.blocked_evidence_max is not None:
            expected += f" and <= {case.checks.blocked_evidence_max}"
        checks.append(
            ScenarioCheckResult(
                name="blocked_evidence_count",
                passed=passed,
                details=(
                    f"expected {expected}, got {blocked_evidence_count}"
                    if not passed
                    else f"blocked evidence count {blocked_evidence_count}"
                ),
            )
        )

    return checks


def _safe_ratio(numerator: float, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _evaluate_scenario_case(
    case: ScenarioCase,
    *,
    force_stub: bool | None,
    model: str | None,
    client: Any | None,
) -> _ScenarioExecution:
    safe_evidence_count, blocked_evidence_count = _evidence_safety_counts(case.evidence)
    retriever = ScenarioRetriever(case.evidence)
    retrieved_chunks = retriever.retrieve(case.query)
    retrieved_sources, retrieved_relevant_chunk_count = _retrieval_details(
        retrieved_chunks
    )
    (
        expected_retrieval_sources,
        first_expected_source_rank,
        retrieval_source_recall,
    ) = _retrieval_expectations(case, retrieved_sources)
    session = run_chat_session(
        case.query,
        case.constraints,
        retriever,
        force_stub=force_stub,
        model=model,
        client=client,
    )

    final_answer = session.fused.final_answer
    justification = session.fused.justification
    next_steps = [
        str(item).strip() for item in session.fused.next_steps if str(item).strip()
    ]
    combined_text = _combined_answer_text(final_answer, justification, next_steps)
    citation_count = len(_CITATION_RE.findall(combined_text))
    case_citation_hit_rate, case_answer_support_score, answer_supported = (
        _answer_grounding(combined_text, retrieved_chunks)
    )
    check_results = _evaluate_checks(
        case,
        predicted_residual_risk=session.final_decision.residual_risk,
        combined_text=combined_text,
        citation_count=citation_count,
        citation_hit_rate=case_citation_hit_rate,
        answer_support_score=case_answer_support_score,
        requires_human_review=session.final_decision.requires_human_review,
        blocked_evidence_count=blocked_evidence_count,
        retrieved_sources=retrieved_sources,
    )
    verdict_match = session.final_decision.verdict == case.expected_verdict
    passed = verdict_match and all(check.passed for check in check_results)
    persona_stances = {
        name: str(getattr(payload, "stance", "unknown")).strip().lower() or "unknown"
        for name, payload in session.personas.items()
    }

    return _ScenarioExecution(
        case_result=ScenarioCaseResult(
            id=case.id,
            description=case.description,
            query=case.query,
            tags=case.tags,
            passed=passed,
            expected_verdict=case.expected_verdict,
            predicted_verdict=session.final_decision.verdict,
            verdict_match=verdict_match,
            expected_residual_risk=case.expected_residual_risk,
            predicted_residual_risk=session.final_decision.residual_risk,
            citation_count=citation_count,
            citation_hit_rate=case_citation_hit_rate,
            answer_support_score=case_answer_support_score,
            answer_supported=answer_supported,
            requires_human_review=session.final_decision.requires_human_review,
            review_reason=session.final_decision.review_reason,
            safe_evidence_count=safe_evidence_count,
            blocked_evidence_count=blocked_evidence_count,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieved_relevant_chunk_count=retrieved_relevant_chunk_count,
            expected_retrieval_sources=expected_retrieval_sources,
            first_expected_source_rank=first_expected_source_rank,
            retrieval_source_recall=retrieval_source_recall,
            retrieved_sources=retrieved_sources,
            checks=check_results,
            final_answer=final_answer,
            justification=justification,
            next_steps=next_steps,
            persona_stances=persona_stances,
        ),
        effective_mode=session.effective_mode,
        resolved_model=session.model or "",
        retrieval_evaluable=bool(case.evidence),
        retrieval_hit=retrieved_relevant_chunk_count > 0,
        retrieval_ranked=bool(expected_retrieval_sources),
        retrieval_top_source_hit=first_expected_source_rank == 1,
        retrieval_reciprocal_rank=(
            0.0
            if first_expected_source_rank is None
            else 1.0 / first_expected_source_rank
        ),
        retrieval_source_recall=retrieval_source_recall,
        answer_support_evaluable=bool(retrieved_chunks),
        citation_hit_rate=case_citation_hit_rate,
        answer_support_score=case_answer_support_score,
        answer_supported=answer_supported,
    )


def _build_scenario_summary(
    case_results: list[ScenarioCaseResult],
    counters: _ScenarioCounters,
    *,
    requested_mode: str,
    effective_mode: str,
    resolved_model: str,
) -> ScenarioSummary:
    predictions = [item.predicted_verdict for item in case_results]
    gold = [item.expected_verdict for item in case_results]
    total_requirements = sum(len(item.checks) for item in case_results)
    passed_requirements = sum(
        sum(1 for check in item.checks if check.passed) for item in case_results
    )
    passed_cases = sum(1 for item in case_results if item.passed)
    requirement_pass_rate = (
        1.0 if total_requirements == 0 else passed_requirements / total_requirements
    )

    return ScenarioSummary(
        total_cases=len(case_results),
        passed_cases=passed_cases,
        overall_score=_safe_ratio(passed_cases, len(case_results)),
        verdict_accuracy=accuracy(predictions, gold),
        requirement_pass_rate=requirement_pass_rate,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        model=resolved_model,
        total_requirements=total_requirements,
        passed_requirements=passed_requirements,
        retrieval_evaluable_cases=counters.retrieval_evaluable_cases,
        cases_with_retrieval_hits=counters.cases_with_retrieval_hits,
        retrieval_hit_rate=_safe_ratio(
            counters.cases_with_retrieval_hits, counters.retrieval_evaluable_cases
        )
        if counters.retrieval_evaluable_cases
        else 1.0,
        retrieval_ranked_cases=counters.retrieval_ranked_cases,
        retrieval_top_source_accuracy=_safe_ratio(
            counters.retrieval_top_source_hits, counters.retrieval_ranked_cases
        ),
        retrieval_mrr=_safe_ratio(
            counters.retrieval_rr_total, counters.retrieval_ranked_cases
        ),
        retrieval_source_recall=_safe_ratio(
            counters.retrieval_source_recall_total, counters.retrieval_ranked_cases
        ),
        answer_support_evaluable_cases=counters.answer_support_evaluable_cases,
        average_citation_hit_rate=_safe_ratio(
            counters.citation_hit_rate_total, counters.answer_support_evaluable_cases
        ),
        average_answer_support_score=_safe_ratio(
            counters.answer_support_score_total,
            counters.answer_support_evaluable_cases,
        ),
        supported_answer_rate=_safe_ratio(
            counters.supported_answers, counters.answer_support_evaluable_cases
        ),
        token_stats=get_token_stats(),
    )


def run_scenario_suite(
    dataset: ScenarioDataset,
    *,
    force_stub: bool | None = None,
    model: str | None = None,
    client: Any | None = None,
    requested_mode: str = "auto",
) -> ScenarioReport:
    clear_cache()
    reset_token_tracking()

    case_results: list[ScenarioCaseResult] = []
    effective_mode = "stub"
    resolved_model = model or ""
    counters = _ScenarioCounters()

    for case in dataset.cases:
        execution = _evaluate_scenario_case(
            case,
            force_stub=force_stub,
            model=model,
            client=client,
        )
        effective_mode = execution.effective_mode
        resolved_model = execution.resolved_model or resolved_model
        case_results.append(execution.case_result)
        counters.record(execution)

    summary = _build_scenario_summary(
        case_results,
        counters,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        resolved_model=resolved_model,
    )

    metadata = dict(dataset.metadata)
    metadata["suite_type"] = "live_scenarios"
    return ScenarioReport(metadata=metadata, summary=summary, cases=case_results)


def render_scenario_report(report: ScenarioReport) -> str:
    lines = [
        "case_id\texpected\tpredicted\tpassed\tcitations\tcitation_hit_rate\tanswer_support\tsupported\tblocked\tretrieved\tretrieval_hits\tfirst_expected_rank\tsource_recall"
    ]
    for item in report.cases:
        lines.append(
            "\t".join(
                [
                    item.id,
                    item.expected_verdict,
                    item.predicted_verdict,
                    "yes" if item.passed else "no",
                    str(item.citation_count),
                    f"{item.citation_hit_rate:.2f}",
                    f"{item.answer_support_score:.2f}",
                    "yes" if item.answer_supported else "no",
                    str(item.blocked_evidence_count),
                    str(item.retrieved_chunk_count),
                    str(item.retrieved_relevant_chunk_count),
                    str(item.first_expected_source_rank or "-"),
                    f"{item.retrieval_source_recall:.2f}",
                ]
            )
        )
    lines.extend(
        [
            "",
            f"overall_score\t{report.summary.overall_score:.2%}",
            f"verdict_accuracy\t{report.summary.verdict_accuracy:.2%}",
            f"requirement_pass_rate\t{report.summary.requirement_pass_rate:.2%}",
            f"requested_mode\t{report.summary.requested_mode}",
            f"effective_mode\t{report.summary.effective_mode}",
            f"model\t{report.summary.model}",
            f"passed_cases\t{report.summary.passed_cases}/{report.summary.total_cases}",
            f"passed_requirements\t{report.summary.passed_requirements}/{report.summary.total_requirements}",
            "retrieval_hits\t"
            f"{report.summary.cases_with_retrieval_hits}/{report.summary.retrieval_evaluable_cases}",
            f"retrieval_hit_rate\t{report.summary.retrieval_hit_rate:.2%}",
            "retrieval_top_source_accuracy\t"
            f"{report.summary.retrieval_top_source_accuracy:.2%}",
            f"retrieval_mrr\t{report.summary.retrieval_mrr:.4f}",
            f"retrieval_source_recall\t{report.summary.retrieval_source_recall:.2%}",
            "answer_support_cases\t"
            f"{report.summary.answer_support_evaluable_cases}/{report.summary.total_cases}",
            f"average_citation_hit_rate\t{report.summary.average_citation_hit_rate:.2%}",
            "average_answer_support_score\t"
            f"{report.summary.average_answer_support_score:.4f}",
            f"supported_answer_rate\t{report.summary.supported_answer_rate:.2%}",
        ]
    )
    token_stats = report.summary.token_stats
    if token_stats:
        lines.append(
            "token_stats\t" + json.dumps(token_stats, ensure_ascii=True, sort_keys=True)
        )
    return "\n".join(lines)


__all__ = [
    "ScenarioCase",
    "ScenarioCaseResult",
    "ScenarioChecks",
    "ScenarioDataset",
    "ScenarioEvidence",
    "ScenarioReport",
    "ScenarioRetriever",
    "load_scenario_dataset",
    "render_scenario_report",
    "run_scenario_suite",
    "write_scenario_report",
]
