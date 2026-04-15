from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from magi.eval.metrics import accuracy

VALID_VERDICTS = {"approve", "reject", "revise"}
VALID_RISK_LEVELS = {"low", "medium", "high"}
VALID_PERSONAS = {"melchior", "balthasar", "casper"}
_CITATION_RE = re.compile(r"\[\d+\]")


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("’", "'")).strip().lower()


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
    forbidden_terms: List[str] = Field(default_factory=list)
    min_citations: int = Field(default=0, ge=0)
    blocked_evidence_min: int = Field(default=0, ge=0)
    blocked_evidence_max: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _normalize(self) -> "ScenarioChecks":
        for field_name in (
            "required_terms_all",
            "required_terms_any",
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
    safe_evidence_count: int
    blocked_evidence_count: int
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
    token_stats: Dict[str, Any] = Field(default_factory=dict)


class ScenarioReport(BaseModel):
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: ScenarioSummary
    cases: List[ScenarioCaseResult] = Field(default_factory=list)


class ScenarioRetriever:
    def __init__(self, evidence: Sequence[ScenarioEvidence]):
        self._evidence = list(evidence)

    def retrieve(
        self, query: str, *, persona: str | None = None, top_k: int = 8
    ) -> list[RetrievedChunk]:
        del query
        persona_key = (persona or "").strip().lower() or None
        chunks: list[RetrievedChunk] = []
        for index, item in enumerate(self._evidence, start=1):
            if item.persona and item.persona != persona_key:
                continue
            chunks.append(
                RetrievedChunk(
                    document_id=f"{item.source}::{index}",
                    text=item.text,
                    score=max(0.0, item.score - (index * 1e-6)),
                    metadata={"source": item.source},
                )
            )
        return chunks[:top_k]

    def __call__(
        self, query: str, *, persona: str | None = None, top_k: int = 8
    ) -> str:
        return default_formatter(self.retrieve(query, persona=persona, top_k=top_k))


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


def _evaluate_checks(
    case: ScenarioCase,
    *,
    predicted_residual_risk: str,
    combined_text: str,
    citation_count: int,
    blocked_evidence_count: int,
) -> List[ScenarioCheckResult]:
    checks: list[ScenarioCheckResult] = []
    lowered = _normalize_text(combined_text)

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


def run_scenario_suite(
    dataset: ScenarioDataset,
    *,
    force_stub: bool | None = None,
    model: str | None = None,
    requested_mode: str = "auto",
) -> ScenarioReport:
    clear_cache()
    reset_token_tracking()

    case_results: list[ScenarioCaseResult] = []
    effective_mode = "stub"
    resolved_model = model or ""

    for case in dataset.cases:
        safe_evidence_count, blocked_evidence_count = _evidence_safety_counts(
            case.evidence
        )
        session = run_chat_session(
            case.query,
            case.constraints,
            ScenarioRetriever(case.evidence),
            force_stub=force_stub,
            model=model,
        )
        effective_mode = session.effective_mode
        resolved_model = session.model or resolved_model

        final_answer = session.fused.final_answer
        justification = session.fused.justification
        next_steps = [
            str(item).strip() for item in session.fused.next_steps if str(item).strip()
        ]
        combined_text = _combined_answer_text(final_answer, justification, next_steps)
        citation_count = len(_CITATION_RE.findall(combined_text))
        check_results = _evaluate_checks(
            case,
            predicted_residual_risk=session.final_decision.residual_risk,
            combined_text=combined_text,
            citation_count=citation_count,
            blocked_evidence_count=blocked_evidence_count,
        )
        verdict_match = session.final_decision.verdict == case.expected_verdict
        passed = verdict_match and all(check.passed for check in check_results)
        persona_stances = {
            name: str(getattr(payload, "stance", "unknown")).strip().lower()
            or "unknown"
            for name, payload in session.personas.items()
        }

        case_results.append(
            ScenarioCaseResult(
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
                safe_evidence_count=safe_evidence_count,
                blocked_evidence_count=blocked_evidence_count,
                checks=check_results,
                final_answer=final_answer,
                justification=justification,
                next_steps=next_steps,
                persona_stances=persona_stances,
            )
        )

    predictions = [item.predicted_verdict for item in case_results]
    gold = [item.expected_verdict for item in case_results]
    total_requirements = sum(len(item.checks) for item in case_results)
    passed_requirements = sum(
        sum(1 for check in item.checks if check.passed) for item in case_results
    )
    requirement_pass_rate = (
        1.0 if total_requirements == 0 else passed_requirements / total_requirements
    )
    passed_cases = sum(1 for item in case_results if item.passed)

    summary = ScenarioSummary(
        total_cases=len(case_results),
        passed_cases=passed_cases,
        overall_score=(passed_cases / len(case_results)) if case_results else 0.0,
        verdict_accuracy=accuracy(predictions, gold),
        requirement_pass_rate=requirement_pass_rate,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        model=resolved_model,
        total_requirements=total_requirements,
        passed_requirements=passed_requirements,
        token_stats=get_token_stats(),
    )

    metadata = dict(dataset.metadata)
    metadata["suite_type"] = "live_scenarios"
    return ScenarioReport(metadata=metadata, summary=summary, cases=case_results)


def render_scenario_report(report: ScenarioReport) -> str:
    lines = ["case_id\texpected\tpredicted\tpassed\tcitations\tblocked"]
    for item in report.cases:
        lines.append(
            "\t".join(
                [
                    item.id,
                    item.expected_verdict,
                    item.predicted_verdict,
                    "yes" if item.passed else "no",
                    str(item.citation_count),
                    str(item.blocked_evidence_count),
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
