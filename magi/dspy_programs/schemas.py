from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Stance = Literal["approve", "reject", "revise"]
RiskLevel = Literal["low", "medium", "high"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    def __str__(self) -> str:
        text = getattr(self, "text", "")
        if isinstance(text, str) and text:
            return text
        return super().__str__()


class RetrievedEvidence(StrictModel):
    citation: str
    source: str
    document_id: str = ""
    text: str
    score: float = Field(default=0.0, ge=0.0)
    blocked: bool = False
    safety_reasons: list[str] = Field(default_factory=list)


class MelchiorResponse(StrictModel):
    text: str = ""
    analysis: str
    answer_outline: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_quotes: list[str]
    stance: Stance
    actions: list[str]


class BalthasarResponse(StrictModel):
    text: str = ""
    plan: str
    communication_plan: list[str]
    cost_estimate: str
    confidence: float = Field(ge=0.0, le=1.0)
    stance: Stance
    actions: list[str]

    @field_validator("cost_estimate", mode="before")
    @classmethod
    def _normalize_cost(cls, value: object) -> str:
        if value is None:
            return "moderate"
        text = str(value).strip().lower()
        if not text:
            return "moderate"
        if "low" in text:
            return "low"
        if "high" in text:
            return "high"
        return "moderate"


class CasperResponse(StrictModel):
    text: str = ""
    risks: list[str]
    mitigations: list[str]
    residual_risk: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    stance: Stance
    actions: list[str]
    outstanding_questions: list[str]


class FusionResponse(StrictModel):
    text: str = ""
    verdict: Stance
    justification: str
    confidence: float = Field(ge=0.0, le=1.0)
    final_answer: str
    next_steps: list[str]
    consensus_points: list[str]
    disagreements: list[str]
    residual_risk: RiskLevel
    risks: list[str]
    mitigations: list[str]


class ResponderResponse(StrictModel):
    text: str = ""
    final_answer: str
    justification: str
    next_steps: list[str]
