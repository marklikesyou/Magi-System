from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence, cast

import json
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, ValidationError, model_validator

from magi.decision.schema import PersonaOutput

VALID_VERDICTS = {"approve", "reject", "revise"}
VALID_PERSONAS = {"melchior", "balthasar", "casper"}


class PersonaCase(BaseModel):
    text: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FusedCase(BaseModel):
    verdict: str
    justification: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_risk: str = "medium"
    risks: List[str] = Field(default_factory=list)
    mitigations: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalize(self) -> "FusedCase":
        verdict = self.verdict.strip().lower()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid fused verdict '{self.verdict}'")
        object.__setattr__(self, "verdict", verdict)
        object.__setattr__(
            self, "residual_risk", self.residual_risk.strip().lower() or "medium"
        )
        return self


class EvaluationCase(BaseModel):
    id: str
    query: str
    constraints: str = ""
    expected_verdict: str
    fused: FusedCase
    personas: Dict[str, PersonaCase]

    @model_validator(mode="after")
    def _normalize(self) -> "EvaluationCase":
        verdict = self.expected_verdict.strip().lower()
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid expected verdict '{self.expected_verdict}'")
        object.__setattr__(self, "expected_verdict", verdict)
        normalized_personas: Dict[str, PersonaCase] = {}
        for key, value in self.personas.items():
            alias = key.strip().lower()
            if alias not in VALID_PERSONAS:
                raise ValueError(f"invalid persona '{key}'")
            normalized_personas[alias] = value
        if not normalized_personas:
            raise ValueError("at least one persona is required")
        object.__setattr__(self, "personas", normalized_personas)
        object.__setattr__(self, "constraints", self.constraints or "")
        return self


class EvaluationDataset(BaseModel):
    cases: List[EvaluationCase] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    def _coerce(cls, data: Any) -> Any:
        if not data:
            return {"cases": []}
        if isinstance(data, dict) and "cases" not in data and "tasks" in data:
            payload = dict(data)
            payload["cases"] = payload.pop("tasks")
            return payload
        return data

    @model_validator(mode="after")
    def _validate(self) -> "EvaluationDataset":
        identifiers = [case.id for case in self.cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("case identifiers must be unique")
        return self

    def expected_labels(self) -> List[str]:
        return [case.expected_verdict for case in self.cases]


def build_persona_outputs(case: EvaluationCase) -> List[PersonaOutput]:
    outputs: List[PersonaOutput] = []
    for name, data in case.personas.items():
        outputs.append(
            PersonaOutput(
                name=cast(Literal["melchior", "balthasar", "casper"], name),
                text=data.text,
                confidence=data.confidence,
                evidence=[],
            )
        )
    return outputs


def load_dataset(path: Path) -> EvaluationDataset:
    if not path.exists():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    try:
        dataset = EvaluationDataset.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid dataset: {exc}") from exc
    if not dataset.cases:
        raise ValueError("dataset contains no cases")
    return dataset


def export_feature_log(features: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in features:
            handle.write(json.dumps(row, ensure_ascii=True))
            handle.write("\n")
