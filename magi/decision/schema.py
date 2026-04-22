from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source: str
    quote: str
    weight: float = Field(default=1.0, ge=0.0)


class PersonaOutput(BaseModel):
    name: Literal["melchior", "balthasar", "casper"]
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[EvidenceItem] = Field(default_factory=list)


class FinalDecision(BaseModel):
    verdict: Literal["approve", "reject", "revise", "abstain"]
    justification: str
    persona_outputs: List[PersonaOutput]
    risks: List[str] = Field(default_factory=list)
    mitigations: List[str] = Field(default_factory=list)
    residual_risk: Literal["low", "medium", "high"] = "medium"
    requires_human_review: bool = False
    review_reason: str = ""
    abstained: bool = False
    abstention_reason: str = ""
