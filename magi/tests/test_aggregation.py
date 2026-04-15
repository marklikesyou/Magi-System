from magi.decision.aggregator import (
    choose_verdict,
    majority_weighted,
    parse_vote,
    PersonaVote,
    resolve_verdict,
)
from magi.decision.schema import PersonaOutput
from magi.dspy_programs.schemas import (
    BalthasarResponse,
    CasperResponse,
    FusionResponse,
    MelchiorResponse,
)


def test_majority_weighted_prefers_high_confidence():
    votes = [
        PersonaVote(name="melchior", action="approve", confidence=0.2),
        PersonaVote(name="balthasar", action="reject", confidence=0.9),
        PersonaVote(name="casper", action="approve", confidence=0.1),
    ]
    assert majority_weighted(votes) == "reject"


def test_parse_vote_defaults_to_revise_when_no_tag():
    persona = PersonaOutput(name="melchior", text="Need more data", confidence=0.5)
    vote = parse_vote(persona)
    assert vote.action == "revise"


def test_choose_verdict_uses_tags():
    personas = [
        PersonaOutput(name="melchior", text="[APPROVE] looks good", confidence=0.7),
        PersonaOutput(name="balthasar", text="[REJECT] too risky", confidence=0.8),
        PersonaOutput(name="casper", text="[REVISE] add guardrails", confidence=0.9),
    ]
    assert choose_verdict(personas) == "revise"


def test_resolve_verdict_prefers_approve_when_two_personas_approve_and_none_reject():
    melchior = MelchiorResponse(
        text="[APPROVE] [MELCHIOR] MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        analysis="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        answer_outline=["Summarize the system."],
        confidence=0.9,
        evidence_quotes=['[1] "MAGI overview"'],
        stance="approve",
        actions=["Answer directly."],
    )
    balthasar = BalthasarResponse(
        text="[APPROVE] [BALTHASAR] Provide a concise one-sentence summary.",
        plan="Provide a concise one-sentence summary.",
        communication_plan=["Keep it short."],
        cost_estimate="low",
        confidence=0.88,
        stance="approve",
        actions=["Keep it grounded."],
    )
    casper = CasperResponse(
        text="[REVISE] [CASPER] Risk level: medium. The evidence base may become outdated over time.",
        risks=["The evidence base may become outdated over time."],
        mitigations=["Refresh documents regularly."],
        residual_risk="medium",
        confidence=0.72,
        stance="revise",
        actions=["Refresh documents regularly."],
        outstanding_questions=[],
    )
    fused = FusionResponse(
        text="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        verdict="revise",
        justification="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        confidence=0.84,
        final_answer="MAGI is a multi-persona reasoning engine grounded in retrieved evidence.",
        next_steps=["Answer directly."],
        consensus_points=["The answer is grounded in retrieved evidence."],
        disagreements=["Casper noted moderate operational caution."],
        residual_risk="medium",
        risks=casper.risks,
        mitigations=casper.mitigations,
    )
    persona_outputs = [
        PersonaOutput(
            name="melchior",
            text=melchior.text,
            confidence=melchior.confidence,
            evidence=[],
        ),
        PersonaOutput(
            name="balthasar",
            text=balthasar.text,
            confidence=balthasar.confidence,
            evidence=[],
        ),
        PersonaOutput(
            name="casper", text=casper.text, confidence=casper.confidence, evidence=[]
        ),
    ]

    verdict = resolve_verdict(
        fused,
        {"melchior": melchior, "balthasar": balthasar, "casper": casper},
        persona_outputs,
    )

    assert verdict == "approve"


def test_resolve_verdict_downgrades_insufficient_information_reject_to_revise():
    melchior = MelchiorResponse(
        text="[REJECT] [MELCHIOR] The evidence does not specify MAGI's p95 latency SLA.",
        analysis="The evidence does not specify MAGI's p95 latency SLA.",
        answer_outline=["State that the documentation does not specify the SLA."],
        confidence=0.62,
        evidence_quotes=[],
        stance="reject",
        actions=["Request additional documentation."],
    )
    balthasar = BalthasarResponse(
        text="[REJECT] [BALTHASAR] Clarify that the documentation does not specify the latency SLA.",
        plan="Clarify that the documentation does not specify the latency SLA.",
        communication_plan=["Explain what is missing."],
        cost_estimate="low",
        confidence=0.61,
        stance="reject",
        actions=["Request confirmation from operators."],
    )
    casper = CasperResponse(
        text="[REVISE] [CASPER] Risk level: medium. Outdated documentation may lead to inaccurate answers.",
        risks=["Outdated documentation may lead to inaccurate answers."],
        mitigations=["Confirm the SLA with system administrators."],
        residual_risk="medium",
        confidence=0.7,
        stance="revise",
        actions=["Confirm the SLA with system administrators."],
        outstanding_questions=["What is the current production SLA?"],
    )
    fused = FusionResponse(
        text="The current evidence does not specify MAGI's guaranteed p95 latency SLA.",
        verdict="reject",
        justification="The current evidence does not specify MAGI's guaranteed p95 latency SLA.",
        confidence=0.79,
        final_answer="The current evidence does not specify MAGI's guaranteed p95 latency SLA.",
        next_steps=["Request updated documentation."],
        consensus_points=["The available documentation is incomplete."],
        disagreements=[],
        residual_risk="medium",
        risks=casper.risks,
        mitigations=casper.mitigations,
    )
    persona_outputs = [
        PersonaOutput(
            name="melchior",
            text=melchior.text,
            confidence=melchior.confidence,
            evidence=[],
        ),
        PersonaOutput(
            name="balthasar",
            text=balthasar.text,
            confidence=balthasar.confidence,
            evidence=[],
        ),
        PersonaOutput(
            name="casper", text=casper.text, confidence=casper.confidence, evidence=[]
        ),
    ]

    verdict = resolve_verdict(
        fused,
        {"melchior": melchior, "balthasar": balthasar, "casper": casper},
        persona_outputs,
    )

    assert verdict == "revise"
