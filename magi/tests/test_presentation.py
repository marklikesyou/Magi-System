from __future__ import annotations

from pathlib import Path

from magi.app.presentation import format_chat_report
from magi.app.service import ChatSessionResult, CitedEvidenceTrace, DecisionTrace
from magi.core.profiles import load_profile
from magi.decision.schema import FinalDecision, PersonaOutput
from magi.dspy_programs.schemas import FusionResponse


def _sample_result() -> ChatSessionResult:
    personas = [
        PersonaOutput(name="melchior", text="[APPROVE] [MELCHIOR] Evidence is sufficient.", confidence=0.8, evidence=[]),
        PersonaOutput(name="balthasar", text="[APPROVE] [BALTHASAR] Proceed with a bounded rollout.", confidence=0.7, evidence=[]),
        PersonaOutput(name="casper", text="[APPROVE] [CASPER] Risk is manageable with controls.", confidence=0.9, evidence=[]),
    ]
    decision = FinalDecision(
        verdict="approve",
        justification="Grounded recommendation with cited support.",
        persona_outputs=personas,
        risks=["Residual vendor dependency risk."],
        mitigations=["Keep a rollback plan."],
        residual_risk="medium",
        requires_human_review=True,
        review_reason="Human review remains required.",
    )
    fused = FusionResponse(
        verdict="approve",
        justification="Grounded recommendation with cited support.",
        confidence=0.8,
        final_answer="Proceed with a guarded rollout [1].",
        next_steps=["Confirm rollback ownership.", "Schedule weekly review."],
        consensus_points=[],
        disagreements=[],
        residual_risk="medium",
        risks=["Residual vendor dependency risk."],
        mitigations=["Keep a rollback plan."],
    )
    trace = DecisionTrace(
        query_hash="abc",
        query_mode="decision",
        routing_rationale="Selected decision because decision markers were strongest.",
        cited_evidence=[
            CitedEvidenceTrace(
                citation="[1]",
                source="briefing.md",
                document_id="briefing::1",
                text="The evidence supports a guarded rollout with rollback ownership.",
            )
        ],
        citation_hit_rate=1.0,
        answer_support_score=0.4,
        requires_human_review=True,
        review_reason="Human review remains required.",
    )
    return ChatSessionResult(
        final_decision=decision,
        fused=fused,
        personas={},
        decision_trace=trace,
        effective_mode="stub",
        model="",
    )


def test_standard_report_includes_personas_without_routing_debug() -> None:
    report = format_chat_report(_sample_result(), Path("/tmp/artifact.json"), None)

    assert "Verdict: APPROVE" in report
    assert "Routing Rationale:" not in report
    assert "Persona Perspectives:" in report


def test_exec_brief_report_uses_executive_layout() -> None:
    profile = load_profile("exec-brief")
    report = format_chat_report(
        _sample_result(),
        Path("/tmp/artifact.json"),
        profile,
    )

    assert profile is not None
    assert "EXECUTIVE BRIEF" in report
    assert "Executive Takeaway:" in report
    assert "Why This Route:" not in report
    assert "Persona Perspectives:" not in report
    assert "Follow-up:" in report


def test_policy_triage_report_uses_policy_sections() -> None:
    profile = load_profile("policy-triage")
    report = format_chat_report(
        _sample_result(),
        Path("/tmp/artifact.json"),
        profile,
    )

    assert profile is not None
    assert "POLICY TRIAGE" in report
    assert "Answer:" in report
    assert "Interpretation And Guardrails:" in report
    assert "Why This Route:" in report


def test_security_review_report_shows_profile_routing_rationale() -> None:
    profile = load_profile("security-review")
    report = format_chat_report(
        _sample_result(),
        Path("/tmp/artifact.json"),
        profile,
    )

    assert profile is not None
    assert "SECURITY REVIEW" in report
    assert "Why This Route:" in report
    assert "Mode: decision" in report
    assert "Selected decision because decision markers were strongest." in report
