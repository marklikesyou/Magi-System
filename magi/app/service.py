from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from magi.decision.aggregator import resolve_verdict
from magi.decision.schema import EvidenceItem, FinalDecision, PersonaOutput
from magi.dspy_programs.runtime import MagiProgram
from magi.dspy_programs.schemas import FusionResponse


@dataclass
class ChatSessionResult:
    final_decision: FinalDecision
    fused: FusionResponse
    personas: dict[str, Any]


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


def run_chat_session(query: str, constraints: str, retriever: Any) -> ChatSessionResult:
    program = MagiProgram(retriever=retriever)
    fused, personas = program(query, constraints=constraints)
    persona_outputs: list[PersonaOutput] = []
    for name, payload in personas.items():
        persona_outputs.append(
            PersonaOutput(
                name=cast(Literal["melchior", "balthasar", "casper"], name.lower()),
                text=str(getattr(payload, "text", payload)),
                confidence=float(getattr(payload, "confidence", 0.0) or 0.0),
                evidence=_evidence_items(payload),
            )
        )
    decision = FinalDecision(
        verdict=resolve_verdict(fused, personas, persona_outputs),
        justification=fused.final_answer or fused.justification,
        persona_outputs=persona_outputs,
        risks=[str(item) for item in fused.risks],
        mitigations=[str(item) for item in fused.mitigations],
        residual_risk=fused.residual_risk,
    )
    return ChatSessionResult(final_decision=decision, fused=fused, personas=personas)
