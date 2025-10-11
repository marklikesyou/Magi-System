import os

os.environ.setdefault("MAGI_FORCE_DSPY_STUB", "1")

from magi.dspy_programs.personas import MagiProgram
from magi.decision.aggregator import resolve_verdict
from magi.decision.schema import PersonaOutput, FinalDecision


def stub_retriever(query: str, persona: str | None = None, top_k: int = 5) -> str:
    persona_tag = persona or "general"
    return f"{persona_tag}: summary for {query}"


def test_magi_program_generates_decision():
    program = MagiProgram(retriever=stub_retriever)
    fused, personas = program(query="Evaluate new rollout", constraints="Budget <= 10k")
    persona_outputs = []
    for name, payload in personas.items():
        persona_outputs.append(
            PersonaOutput(
                name=name,
                text=str(getattr(payload, "text", payload)),
                confidence=float(getattr(payload, "confidence", 0.0) or 0.0),
                evidence=[],
            )
        )
    verdict = resolve_verdict(fused, personas, persona_outputs)
    assert verdict in {"approve", "reject", "revise"}
    risks = []
    fused_risks = getattr(fused, "risks", [])
    if fused_risks:
        for item in fused_risks:
            risks.append(str(item))
    mitigations = []
    fused_mitigations = getattr(fused, "mitigations", [])
    if fused_mitigations:
        for item in fused_mitigations:
            mitigations.append(str(item))
    residual_risk_value = str(getattr(fused, "residual_risk", "medium")).lower()
    decision = FinalDecision(
        verdict=verdict,
        justification=str(getattr(fused, "justification", fused)),
        persona_outputs=persona_outputs,
        risks=risks,
        mitigations=mitigations,
        residual_risk=residual_risk_value if residual_risk_value in {"low", "medium", "high"} else "medium",
    )
    assert decision.verdict == verdict
