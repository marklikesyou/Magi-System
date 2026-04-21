from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Literal, cast

from magi.core.config import get_settings
from magi.core.utils import hash_query
from magi.decision.aggregator import resolve_verdict_with_details
from magi.decision.schema import EvidenceItem, FinalDecision, PersonaOutput
from magi.dspy_programs.runtime import MagiProgram, get_token_stats
from magi.dspy_programs.schemas import FusionResponse
from magi.eval.metrics import answer_support_score

logger = logging.getLogger(__name__)
_CITATION_LABEL_RE = re.compile(r"\[\d+\]")


@dataclass
class CitedEvidenceTrace:
    citation: str = ""
    source: str = ""
    document_id: str = ""
    text: str = ""


@dataclass
class BlockedEvidenceTrace:
    citation: str = ""
    source: str = ""
    document_id: str = ""
    text: str = ""
    safety_reasons: list[str] = field(default_factory=list)


@dataclass
class DecisionTrace:
    query_hash: str = ""
    retrieved_evidence_ids: list[str] = field(default_factory=list)
    used_evidence_ids: list[str] = field(default_factory=list)
    blocked_evidence_ids: list[str] = field(default_factory=list)
    cited_evidence_ids: list[str] = field(default_factory=list)
    retrieved_sources: list[str] = field(default_factory=list)
    cited_sources: list[str] = field(default_factory=list)
    cited_evidence: list[CitedEvidenceTrace] = field(default_factory=list)
    blocked_evidence: list[BlockedEvidenceTrace] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    persona_stances: dict[str, str] = field(default_factory=dict)
    fused_verdict: str = ""
    final_verdict: str = ""
    residual_risk: str = "medium"
    safety_outcome: str = "passed"
    cache_hit: bool = False
    citation_count: int = 0
    citation_hit_rate: float = 0.0
    answer_support_score: float = 0.0
    answer_supported: bool = False
    requires_human_review: bool = False
    review_reason: str = ""
    verdict_overridden: bool = False
    program_run_ms: float = 0.0
    decision_resolution_ms: float = 0.0
    trace_capture_ms: float = 0.0
    end_to_end_ms: float = 0.0
    effective_mode: str = "stub"
    model: str = ""
    token_stats: dict[str, Any] = field(default_factory=dict)
    decision_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatSessionResult:
    final_decision: FinalDecision
    fused: FusionResponse
    personas: dict[str, Any]
    decision_trace: DecisionTrace = field(default_factory=DecisionTrace)
    effective_mode: str = "stub"
    model: str = ""


def _evidence_items(payload: Any) -> list[EvidenceItem]:
    quotes = getattr(payload, "evidence_quotes", [])
    items: list[EvidenceItem] = []
    if not isinstance(quotes, list):
        return items
    for quote in quotes:
        text = str(quote).strip()
        if not text:
            continue
        items.append(EvidenceItem(source="retrieved", quote=text))
    return items


def _normalize_verdict_label(value: object) -> Literal["approve", "reject", "revise"] | None:
    label = str(value or "").strip().lower()
    if label in {"approve", "reject", "revise"}:
        return cast(Literal["approve", "reject", "revise"], label)
    return None


def _detail_count(details: dict[str, object], key: str) -> int:
    value = details.get(key, 0)
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return max(0, int(value))
        return max(0, int(str(value or 0)))
    except (TypeError, ValueError):
        return 0


def _decision_justification(
    fused: FusionResponse,
    verdict: Literal["approve", "reject", "revise"],
    details: dict[str, object],
) -> str:
    base = str(fused.final_answer or fused.justification or "").strip()
    fused_verdict = _normalize_verdict_label(getattr(fused, "verdict", ""))
    if fused_verdict is None or fused_verdict == verdict:
        return base

    reasons: list[str] = []
    if bool(details.get("insufficient_information")):
        reasons.append("available evidence was treated as insufficient")

    reject_votes = _detail_count(details, "reject_votes")
    revise_votes = _detail_count(details, "revise_votes")
    approve_votes = _detail_count(details, "approve_votes")
    if reject_votes:
        reasons.append("at least one persona cast a reject vote")
    elif revise_votes >= 2:
        reasons.append("most personas requested revision")
    elif approve_votes >= 2 and verdict == "approve":
        reasons.append("persona consensus favored approval")

    override = (
        f"Authoritative verdict: {verdict.upper()}. "
        f"The decision layer overrode fusion's {fused_verdict.upper()} verdict"
    )
    if reasons:
        override = f"{override} because {'; '.join(reasons)}"
    override = f"{override}."
    if not base:
        return override
    return f"{override}\n\n{base}"


def _persona_stances(personas: dict[str, Any]) -> dict[str, str]:
    stances: dict[str, str] = {}
    for name, payload in personas.items():
        stance = _normalize_verdict_label(getattr(payload, "stance", ""))
        stances[name] = stance or "unknown"
    return stances


def _runtime_retrieval_trace(
    program: MagiProgram,
) -> tuple[list[str], list[str], list[str], list[str], bool]:
    metadata = getattr(program, "last_run_metadata", {}) or {}
    safe_evidence = metadata.get("safe_evidence", [])
    blocked_evidence = metadata.get("blocked_evidence", [])
    retrieved_ids: list[str] = []
    used_ids: list[str] = []
    blocked_ids: list[str] = []
    sources: list[str] = []
    seen_sources: set[str] = set()

    def collect(items: Any, target: list[str]) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            document_id = str(item.get("document_id", "")).strip()
            if document_id:
                retrieved_ids.append(document_id)
                target.append(document_id)
            source = str(item.get("source", "")).strip()
            if source and source not in seen_sources:
                seen_sources.add(source)
                sources.append(source)

    collect(safe_evidence, used_ids)
    collect(blocked_evidence, blocked_ids)
    return (
        retrieved_ids,
        used_ids,
        blocked_ids,
        sources,
        bool(metadata.get("cache_hit", False)),
    )


def _blocked_evidence_trace(program: MagiProgram) -> list[BlockedEvidenceTrace]:
    metadata = getattr(program, "last_run_metadata", {}) or {}
    blocked_evidence = metadata.get("blocked_evidence", [])
    if not isinstance(blocked_evidence, list):
        return []
    items: list[BlockedEvidenceTrace] = []
    for item in blocked_evidence:
        if not isinstance(item, dict):
            continue
        reasons = item.get("safety_reasons", [])
        items.append(
            BlockedEvidenceTrace(
                citation=str(item.get("citation", "")).strip(),
                source=str(item.get("source", "")).strip(),
                document_id=str(item.get("document_id", "")).strip(),
                text=str(item.get("text", "")).strip(),
                safety_reasons=[
                    str(reason).strip() for reason in reasons if str(reason).strip()
                ]
                if isinstance(reasons, list)
                else [],
            )
        )
    return items


def _safety_outcome(fused: FusionResponse, blocked_evidence_ids: list[str]) -> str:
    justification = str(fused.justification or "").lower()
    if "blocked by the safety gate before reasoning" in justification:
        return "input_blocked"
    if "output safety checks detected unsafe content" in justification:
        return "output_blocked"
    if blocked_evidence_ids:
        return "retrieval_items_blocked"
    return "passed"


def _citation_count(*parts: object) -> int:
    count = 0
    for part in parts:
        if not part:
            continue
        count += str(part).count("[")
    return count


def _combined_answer_text(*parts: object) -> str:
    return "\n".join(str(part).strip() for part in parts if str(part).strip()).strip()


def _grounding_metrics(
    program: MagiProgram, text: str
) -> tuple[
    float,
    float,
    bool,
    list[str],
    list[str],
    list[CitedEvidenceTrace],
    list[str],
]:
    metadata = getattr(program, "last_run_metadata", {}) or {}
    safe_evidence = metadata.get("safe_evidence", [])
    evidence_by_citation: dict[str, dict[str, str]] = {}
    evidence_texts: list[str] = []
    if isinstance(safe_evidence, list):
        for item in safe_evidence:
            if not isinstance(item, dict):
                continue
            citation = str(item.get("citation", "")).strip()
            document_id = str(item.get("document_id", "")).strip()
            source = str(item.get("source", "")).strip()
            evidence_text = str(item.get("text", "")).strip()
            if citation:
                evidence_by_citation[citation] = {
                    "document_id": document_id,
                    "source": source,
                    "text": evidence_text,
                }
            if evidence_text:
                evidence_texts.append(evidence_text)
    citations = _CITATION_LABEL_RE.findall(text)
    cited_ids: list[str] = []
    cited_sources: list[str] = []
    cited_evidence: list[CitedEvidenceTrace] = []
    unsupported_citations: list[str] = []
    if citations and evidence_by_citation:
        seen_citations: set[str] = set()
        for citation in citations:
            evidence = evidence_by_citation.get(citation)
            if evidence is None:
                if citation not in unsupported_citations:
                    unsupported_citations.append(citation)
                continue
            document_id = evidence["document_id"]
            source = evidence["source"]
            if document_id and document_id not in cited_ids:
                cited_ids.append(document_id)
            if source and source not in cited_sources:
                cited_sources.append(source)
            if citation not in seen_citations:
                seen_citations.add(citation)
                cited_evidence.append(
                    CitedEvidenceTrace(
                        citation=citation,
                        source=source,
                        document_id=document_id,
                        text=evidence["text"],
                    )
                )
        citation_hit_rate = sum(
            1 for citation in citations if citation in evidence_by_citation
        ) / len(citations)
    else:
        citation_hit_rate = 0.0
    support_score = answer_support_score(text, evidence_texts)
    supported = bool(evidence_texts) and citation_hit_rate > 0.0 and support_score >= 0.2
    return (
        citation_hit_rate,
        support_score,
        supported,
        cited_ids,
        cited_sources,
        cited_evidence,
        unsupported_citations,
    )


def _elevate_residual_risk(value: str, floor: Literal["medium", "high"]) -> Literal["medium", "high"]:
    normalized = str(value or "").strip().lower()
    if normalized == "high" or floor == "high":
        return "high"
    return "medium"


def _apply_approve_guardrail(
    decision: FinalDecision,
    *,
    citation_hit_rate: float,
    support_score: float,
) -> tuple[FinalDecision, bool]:
    settings = get_settings()
    if decision.verdict != "approve":
        return decision, False
    citation_threshold = settings.approve_min_citation_hit_rate
    support_threshold = settings.approve_min_answer_support_score
    if (
        citation_hit_rate >= citation_threshold
        and support_score >= support_threshold
    ):
        return decision, False

    reasons: list[str] = []
    if citation_hit_rate < citation_threshold:
        reasons.append(
            f"valid retrieved citations were below the approval threshold ({citation_hit_rate:.2f} < {citation_threshold:.2f})"
        )
    if support_score < support_threshold:
        reasons.append(
            f"answer support from retrieved evidence was below the approval threshold ({support_score:.2f} < {support_threshold:.2f})"
        )
    override = "Authoritative verdict: REVISE. Approval requires cited retrieved evidence support."
    if reasons:
        override = f"{override} It was downgraded because " + "; ".join(reasons) + "."
    justification = override
    if decision.justification:
        justification = f"{override}\n\n{decision.justification}"
    return (
        FinalDecision(
            verdict="revise",
            justification=justification,
            persona_outputs=decision.persona_outputs,
            risks=decision.risks,
            mitigations=decision.mitigations,
            residual_risk=_elevate_residual_risk(decision.residual_risk, "medium"),
        ),
        True,
    )


def _apply_human_review_requirement(
    decision: FinalDecision,
) -> tuple[FinalDecision, bool]:
    settings = get_settings()
    if decision.verdict != "approve" or not settings.require_human_review_for_approvals:
        return decision, False
    review_reason = (
        decision.review_reason.strip()
        or "Approve decisions require human review until live metrics stabilize."
    )
    return (
        FinalDecision(
            verdict=decision.verdict,
            justification=decision.justification,
            persona_outputs=decision.persona_outputs,
            risks=decision.risks,
            mitigations=decision.mitigations,
            residual_risk=decision.residual_risk,
            requires_human_review=True,
            review_reason=review_reason,
        ),
        True,
    )


def _log_decision_trace(trace: DecisionTrace) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    logger.info("decision_trace %s", json.dumps(asdict(trace), default=str))


def run_chat_session(
    query: str,
    constraints: str,
    retriever: Any,
    *,
    force_stub: bool | None = None,
    model: str | None = None,
    client: Any | None = None,
) -> ChatSessionResult:
    start = perf_counter()
    program = MagiProgram(
        retriever=retriever,
        force_stub=force_stub,
        model=model,
        client=client,
    )
    program_start = perf_counter()
    fused, personas = program(query, constraints=constraints)
    program_run_ms = round((perf_counter() - program_start) * 1000.0, 3)
    normalized_personas = {str(name).lower(): payload for name, payload in personas.items()}
    persona_stances = _persona_stances(normalized_personas)
    persona_outputs: list[PersonaOutput] = []
    for name, payload in normalized_personas.items():
        persona_outputs.append(
            PersonaOutput(
                name=cast(Literal["melchior", "balthasar", "casper"], name.lower()),
                text=str(getattr(payload, "text", payload)),
                confidence=float(getattr(payload, "confidence", 0.0) or 0.0),
                evidence=_evidence_items(payload),
            )
        )
    resolution_start = perf_counter()
    verdict, decision_details = resolve_verdict_with_details(
        fused,
        normalized_personas,
        persona_outputs,
    )
    decision_resolution_ms = round((perf_counter() - resolution_start) * 1000.0, 3)
    trace_start = perf_counter()
    (
        retrieved_evidence_ids,
        used_evidence_ids,
        blocked_evidence_ids,
        retrieved_sources,
        cache_hit,
    ) = _runtime_retrieval_trace(program)
    blocked_evidence = _blocked_evidence_trace(program)
    trace_capture_ms = round((perf_counter() - trace_start) * 1000.0, 3)
    decision = FinalDecision(
        verdict=verdict,
        justification=_decision_justification(fused, verdict, decision_details),
        persona_outputs=persona_outputs,
        risks=[str(item) for item in fused.risks],
        mitigations=[str(item) for item in fused.mitigations],
        residual_risk=fused.residual_risk,
    )
    grounding_text = _combined_answer_text(
        fused.final_answer,
        fused.justification,
        decision.justification,
        " ".join(str(item).strip() for item in fused.next_steps if str(item).strip()),
    )
    (
        citation_hit_rate,
        support_score,
        answer_supported,
        cited_evidence_ids,
        cited_sources,
        cited_evidence,
        unsupported_citations,
    ) = _grounding_metrics(program, grounding_text)
    decision, approve_guardrail_triggered = _apply_approve_guardrail(
        decision,
        citation_hit_rate=citation_hit_rate,
        support_score=support_score,
    )
    decision, human_review_required = _apply_human_review_requirement(decision)
    decision_trace = DecisionTrace(
        query_hash=hash_query(query, constraints),
        retrieved_evidence_ids=retrieved_evidence_ids,
        used_evidence_ids=used_evidence_ids,
        blocked_evidence_ids=blocked_evidence_ids,
        cited_evidence_ids=cited_evidence_ids,
        retrieved_sources=retrieved_sources,
        cited_sources=cited_sources,
        cited_evidence=cited_evidence,
        blocked_evidence=blocked_evidence,
        unsupported_citations=unsupported_citations,
        persona_stances=persona_stances,
        fused_verdict=fused.verdict,
        final_verdict=decision.verdict,
        residual_risk=decision.residual_risk,
        safety_outcome=_safety_outcome(fused, blocked_evidence_ids),
        cache_hit=cache_hit,
        citation_count=_citation_count(
            fused.final_answer,
            fused.justification,
            decision.justification,
        ),
        citation_hit_rate=round(citation_hit_rate, 4),
        answer_support_score=round(support_score, 4),
        answer_supported=answer_supported,
        requires_human_review=decision.requires_human_review,
        review_reason=decision.review_reason,
        verdict_overridden=fused.verdict != decision.verdict,
        program_run_ms=program_run_ms,
        decision_resolution_ms=decision_resolution_ms,
        trace_capture_ms=trace_capture_ms,
        end_to_end_ms=round((perf_counter() - start) * 1000.0, 3),
        effective_mode=program.effective_mode,
        model=program.model_name,
        token_stats=get_token_stats(),
        decision_features={
            **dict(decision_details),
            "approve_guardrail_triggered": approve_guardrail_triggered,
            "requires_human_review": human_review_required,
            "citation_hit_rate": round(citation_hit_rate, 4),
            "answer_support_score": round(support_score, 4),
            "cited_evidence_count": len(cited_evidence_ids),
            "unsupported_citations": list(unsupported_citations),
        },
    )
    _log_decision_trace(decision_trace)
    return ChatSessionResult(
        final_decision=decision,
        fused=fused,
        personas=normalized_personas,
        decision_trace=decision_trace,
        effective_mode=program.effective_mode,
        model=program.model_name,
    )
