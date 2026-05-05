from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

from magi.app.service import ChatSessionResult, DecisionTrace
from magi.core.profiles import Profile, PresentationStyle
from magi.decision.schema import PersonaOutput


@dataclass(frozen=True)
class PresentationPolicy:
    style: PresentationStyle = "standard"
    response_format_guidance: str = ""
    show_persona_perspectives: bool = True
    show_routing_rationale: bool = True
    show_cited_evidence: bool = True
    show_blocked_evidence: bool = True
    max_next_steps: int | None = None


_PERSONA_TAG_RE = re.compile(
    r"^\[(?:APPROVE|REJECT|REVISE)\]\s*\[(?:MELCHIOR|BALTHASAR|CASPER)\]\s*",
    re.IGNORECASE,
)


def presentation_policy(profile: Profile | None) -> PresentationPolicy:
    if profile is None:
        return PresentationPolicy()
    return PresentationPolicy(
        style=profile.presentation_style,
        response_format_guidance=profile.response_format_guidance,
        show_persona_perspectives=profile.show_persona_perspectives,
        show_routing_rationale=profile.show_routing_rationale,
        show_cited_evidence=profile.show_cited_evidence,
        show_blocked_evidence=profile.show_blocked_evidence,
        max_next_steps=profile.max_next_steps,
    )


def response_format_guidance(profile: Profile | None) -> str:
    policy = presentation_policy(profile)
    if policy.response_format_guidance:
        return policy.response_format_guidance
    return _default_response_guidance(policy.style)


def _default_response_guidance(style: PresentationStyle) -> str:
    if style == "executive_brief":
        return (
            "Format for executive readers. Lead with a one-sentence takeaway, keep it compact, "
            "and separate confirmed facts from caveats."
        )
    if style == "incident_review":
        return (
            "Format as an incident note with Situation, Impact, Containment, and Unknowns. "
            "Do not overstate root cause or certainty."
        )
    if style == "policy_triage":
        return (
            "Format as a policy memo with Answer, Governing Evidence, Interpretation, and Follow-up."
        )
    if style == "vendor_review":
        return (
            "Format as a vendor review with Recommendation, Evidence, Risks, and Required Follow-up."
        )
    if style == "security_review":
        return (
            "Format as a security review with Recommendation, Guardrails, Evidence, and Residual Risk."
        )
    return ""


def format_chat_report(
    result: ChatSessionResult,
    artifact_path: Path,
    profile: Profile | None,
) -> str:
    policy = presentation_policy(profile)
    if policy.style == "executive_brief":
        return _render_executive_brief(result, artifact_path, policy)
    if policy.style == "incident_review":
        return _render_incident_review(result, artifact_path, policy)
    if policy.style == "policy_triage":
        return _render_policy_triage(result, artifact_path, policy)
    if policy.style == "vendor_review":
        return _render_vendor_review(result, artifact_path, policy)
    if policy.style == "security_review":
        return _render_security_review(result, artifact_path, policy)
    return _render_standard(result, artifact_path, policy)


def _truncate_text(text: object, limit: int = 120) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    trimmed = compact[:limit]
    if " " in trimmed:
        trimmed = trimmed.rsplit(" ", 1)[0]
    return trimmed + "..."


def _strip_persona_tags(text: str) -> str:
    return _PERSONA_TAG_RE.sub("", text).strip()


def _bullet_block(title: str, items: Iterable[object]) -> list[str]:
    lines = [str(item).strip() for item in items if str(item).strip()]
    if not lines:
        return []
    return [f"{title}:"] + [f"  - {line}" for line in lines]


def _evidence_block(trace: DecisionTrace, policy: PresentationPolicy) -> list[str]:
    lines: list[str] = []
    if policy.show_cited_evidence and trace.cited_evidence:
        lines.append("Cited Evidence:")
        for cited_item in trace.cited_evidence:
            lines.append(
                f"  - {cited_item.citation} {cited_item.source}: {_truncate_text(cited_item.text)}"
            )
        lines.append("")
    if policy.show_blocked_evidence and trace.blocked_evidence:
        lines.append("Blocked Evidence:")
        for blocked_item in trace.blocked_evidence:
            reasons = ", ".join(blocked_item.safety_reasons) or "blocked"
            lines.append(
                f"  - {blocked_item.citation} {blocked_item.source}: {reasons}"
            )
            snippet = _truncate_text(blocked_item.text)
            if snippet:
                lines.append(f"    {snippet}")
        lines.append("")
    return lines


def _persona_block(personas: Iterable[PersonaOutput]) -> list[str]:
    lines = [f"{'-' * 60}", "Persona Perspectives:", ""]
    for persona in personas:
        lines.append(
            f"  [{persona.name.title()}] (confidence {persona.confidence:.2f})"
        )
        for line in _strip_persona_tags(persona.text).splitlines():
            clean = line.strip()
            if clean:
                lines.append(f"    {clean}")
        lines.append("")
    return lines


def _trim_next_steps(
    result: ChatSessionResult, policy: PresentationPolicy
) -> list[str]:
    next_steps = [str(item).strip() for item in result.fused.next_steps if str(item).strip()]
    if policy.max_next_steps is not None:
        return next_steps[: policy.max_next_steps]
    return next_steps


def _header(result: ChatSessionResult) -> list[str]:
    decision = result.final_decision
    trace = result.decision_trace
    lines = [
        "",
        "=" * 60,
        f"Verdict: {decision.verdict.upper()}",
        f"Query Mode: {trace.query_mode.upper()}",
        f"Residual Risk: {decision.residual_risk}",
        (
            "Grounding: "
            f"citations {trace.citation_hit_rate:.2f} | "
            f"support {trace.answer_support_score:.2f}"
        ),
    ]
    if decision.requires_human_review:
        lines.append("Human Review: REQUIRED")
    if decision.abstained:
        lines.append("Decision State: ABSTAINED")
    lines.extend(["=" * 60, ""])
    return lines


def _render_standard(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    trace = result.decision_trace
    lines = _header(result)
    lines.append(decision.justification)
    lines.append("")
    if decision.requires_human_review and decision.review_reason:
        lines.append(f"Review Reason: {decision.review_reason}")
        lines.append("")
    if decision.abstained and decision.abstention_reason:
        lines.append(f"Abstention Reason: {decision.abstention_reason}")
        lines.append("")
    lines.extend(_evidence_block(trace, policy))
    if policy.show_persona_perspectives:
        lines.extend(_persona_block(decision.persona_outputs))
    lines.extend(_bullet_block("Risks", decision.risks))
    mitigations = _bullet_block("Mitigations", decision.mitigations)
    if mitigations:
        if decision.risks:
            lines.append("")
        lines.extend(mitigations)
    next_steps = _trim_next_steps(result, policy)
    next_steps_block = _bullet_block("Next Steps", next_steps)
    if next_steps_block:
        lines.append("")
        lines.extend(next_steps_block)
    lines.append("")
    lines.append(f"Artifact: {artifact_path}")
    lines.append("")
    return "\n".join(lines)


def _render_executive_brief(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    trace = result.decision_trace
    lines = [
        "",
        "=" * 60,
        "EXECUTIVE BRIEF",
        "=" * 60,
        f"Decision: {decision.verdict.upper()} | Residual Risk: {decision.residual_risk.upper()}",
        "",
        "Executive Takeaway:",
        result.fused.final_answer or decision.justification,
    ]
    evidence = _evidence_block(trace, policy)
    if evidence:
        lines.extend([""] + evidence[:-1])
    risk_block = _bullet_block("Key Risks", decision.risks)
    if risk_block:
        lines.extend([""] + risk_block)
    next_steps = _trim_next_steps(result, policy)
    if next_steps:
        lines.extend([""] + _bullet_block("Follow-up", next_steps))
    if decision.requires_human_review and decision.review_reason:
        lines.extend(["", f"Review Gate: {decision.review_reason}"])
    if policy.show_persona_perspectives:
        lines.extend([""] + _persona_block(decision.persona_outputs))
    lines.extend(["", f"Artifact: {artifact_path}", ""])
    return "\n".join(lines)


def _render_incident_review(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    lines = [
        "",
        "=" * 60,
        "INCIDENT REVIEW",
        "=" * 60,
        "",
        "Situation:",
        result.fused.final_answer or decision.justification,
        "",
        f"Impact And Risk: residual risk is {decision.residual_risk}.",
    ]
    lines.extend(_bullet_block("Observed Risks", decision.risks))
    mitigations = _bullet_block(
        "Containment And Mitigations",
        list(decision.mitigations) + _trim_next_steps(result, policy),
    )
    if mitigations:
        lines.extend([""] + mitigations)
    evidence = _evidence_block(result.decision_trace, policy)
    if evidence:
        lines.extend([""] + evidence[:-1])
    if decision.abstained and decision.abstention_reason:
        lines.extend(["", f"Unknowns: {decision.abstention_reason}"])
    if policy.show_persona_perspectives:
        lines.extend([""] + _persona_block(decision.persona_outputs))
    lines.extend(["", f"Artifact: {artifact_path}", ""])
    return "\n".join(lines)


def _render_policy_triage(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    lines = [
        "",
        "=" * 60,
        "POLICY TRIAGE",
        "=" * 60,
        "",
        "Answer:",
        result.fused.final_answer or decision.justification,
    ]
    evidence = _evidence_block(result.decision_trace, policy)
    if evidence:
        lines.extend([""] + evidence[:-1])
    lines.extend(["", "Interpretation And Guardrails:", decision.justification])
    follow_up = _trim_next_steps(result, policy)
    if follow_up or decision.review_reason:
        lines.extend([""] + _bullet_block("Required Follow-up", follow_up))
        if decision.review_reason:
            lines.append(f"  - Review gate: {decision.review_reason}")
    if policy.show_persona_perspectives:
        lines.extend([""] + _persona_block(decision.persona_outputs))
    lines.extend(["", f"Artifact: {artifact_path}", ""])
    return "\n".join(lines)


def _render_vendor_review(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    lines = [
        "",
        "=" * 60,
        "VENDOR REVIEW",
        "=" * 60,
        "",
        "Recommendation:",
        result.fused.final_answer or decision.justification,
    ]
    evidence = _evidence_block(result.decision_trace, policy)
    if evidence:
        lines.extend([""] + evidence[:-1])
    risk_block = _bullet_block("Key Risks", decision.risks)
    if risk_block:
        lines.extend([""] + risk_block)
    follow_up = _trim_next_steps(result, policy)
    controls = list(decision.mitigations) + follow_up
    controls_block = _bullet_block("Controls And Follow-up", controls)
    if controls_block:
        lines.extend([""] + controls_block)
    if policy.show_persona_perspectives:
        lines.extend([""] + _persona_block(decision.persona_outputs))
    lines.extend(["", f"Artifact: {artifact_path}", ""])
    return "\n".join(lines)


def _render_security_review(
    result: ChatSessionResult,
    artifact_path: Path,
    policy: PresentationPolicy,
) -> str:
    decision = result.final_decision
    lines = [
        "",
        "=" * 60,
        "SECURITY REVIEW",
        "=" * 60,
        "",
        "Recommendation:",
        result.fused.final_answer or decision.justification,
    ]
    guardrails = list(decision.mitigations) + _trim_next_steps(result, policy)
    guardrail_block = _bullet_block("Guardrails", guardrails)
    if guardrail_block:
        lines.extend([""] + guardrail_block)
    risk_block = _bullet_block("Risks", decision.risks)
    if risk_block:
        lines.extend([""] + risk_block)
    evidence = _evidence_block(result.decision_trace, policy)
    if evidence:
        lines.extend([""] + evidence[:-1])
    if decision.requires_human_review and decision.review_reason:
        lines.extend(["", f"Review Gate: {decision.review_reason}"])
    if policy.show_persona_perspectives:
        lines.extend([""] + _persona_block(decision.persona_outputs))
    lines.extend(["", f"Artifact: {artifact_path}", ""])
    return "\n".join(lines)


__all__ = [
    "PresentationPolicy",
    "format_chat_report",
    "presentation_policy",
    "response_format_guidance",
]
