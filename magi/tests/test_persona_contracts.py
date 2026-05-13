from __future__ import annotations

from typing import get_args

from magi.decision.schema import PersonaOutput
from magi.dspy_programs import signatures
from magi.dspy_programs.schemas import (
    BalthasarResponse,
    CasperResponse,
    MelchiorResponse,
    RiskLevel,
    Stance,
)


PERSONA_NAMES = ("melchior", "balthasar", "casper")


def _schema_fields(model: type[object]) -> tuple[str, ...]:
    return tuple(getattr(model, "model_fields"))


def _signature_output_fields(signature_cls: type[object]) -> set[str]:
    instance = signature_cls()
    output_fields = getattr(instance, "output_fields", None) or getattr(
        signature_cls,
        "output_fields",
        {},
    )
    return set(output_fields)


def test_public_persona_output_names_are_unchanged() -> None:
    annotation = PersonaOutput.model_fields["name"].annotation

    assert get_args(annotation) == PERSONA_NAMES


def test_persona_response_schemas_are_strict_and_unchanged() -> None:
    assert MelchiorResponse.model_config["extra"] == "forbid"
    assert BalthasarResponse.model_config["extra"] == "forbid"
    assert CasperResponse.model_config["extra"] == "forbid"
    assert _schema_fields(MelchiorResponse) == (
        "text",
        "analysis",
        "answer_outline",
        "confidence",
        "evidence_quotes",
        "stance",
        "actions",
    )
    assert _schema_fields(BalthasarResponse) == (
        "text",
        "plan",
        "communication_plan",
        "cost_estimate",
        "confidence",
        "stance",
        "actions",
    )
    assert _schema_fields(CasperResponse) == (
        "text",
        "risks",
        "mitigations",
        "residual_risk",
        "confidence",
        "stance",
        "actions",
        "outstanding_questions",
    )


def test_persona_stance_and_risk_enums_are_unchanged() -> None:
    assert get_args(Stance) == ("approve", "reject", "revise")
    assert get_args(RiskLevel) == ("low", "medium", "high")


def test_dspy_persona_signature_outputs_preserve_responsibilities() -> None:
    assert {
        "analysis",
        "confidence",
        "stance",
        "actions",
    }.issubset(_signature_output_fields(signatures.AnalyzeEvidence))
    assert {
        "plan",
        "cost_estimate",
        "confidence",
        "stance",
        "actions",
    }.issubset(_signature_output_fields(signatures.StakeholderPlan))
    assert {
        "risks",
        "mitigations",
        "residual_risk",
        "confidence",
        "stance",
        "actions",
        "outstanding_questions",
    }.issubset(_signature_output_fields(signatures.EthicalRisk))
