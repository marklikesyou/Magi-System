# mypy: ignore-errors
"""DSPy evaluation helpers for MAGI.

The metrics are intentionally lightweight so they remain usable in stub mode
and under static analysis. When DSPy is available, ``create_magi_evaluator``
returns the native ``dspy.evaluate.Evaluate`` wrapper.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from .signatures import STUB_MODE

if not STUB_MODE:
    import dspy
    from dspy.evaluate import Evaluate

    class _ConsensusJudge(dspy.Signature):
        query = dspy.InputField()
        melchior_analysis = dspy.InputField()
        balthasar_plan = dspy.InputField()
        casper_risks = dspy.InputField()

        consensus_reached: bool = dspy.OutputField()
        consensus_score: float = dspy.OutputField()
        areas_of_agreement: str = dspy.OutputField()
        areas_of_conflict: str = dspy.OutputField()

    class _DecisionQualityJudge(dspy.Signature):
        query = dspy.InputField()
        verdict = dspy.InputField()
        justification = dspy.InputField()
        evidence_quality = dspy.InputField()
        plan_feasibility = dspy.InputField()
        risk_severity = dspy.InputField()

        decision_quality: float = dspy.OutputField()
        is_well_justified: bool = dspy.OutputField()
        decision_critique: str = dspy.OutputField()

    class _PersonaConsistencyJudge(dspy.Signature):
        persona_name = dspy.InputField()
        expected_traits = dspy.InputField()
        actual_output = dspy.InputField()

        consistency_score: float = dspy.OutputField()
        trait_evidence: str = dspy.OutputField()
        out_of_character: str = dspy.OutputField()

    class _RiskMitigationJudge(dspy.Signature):
        identified_risks = dspy.InputField()
        proposed_mitigations = dspy.InputField()
        residual_risk = dspy.InputField()

        coverage_score: float = dspy.OutputField()
        mitigation_quality: float = dspy.OutputField()
        gaps_identified: str = dspy.OutputField()

    class _MAGISystemJudge(dspy.Signature):
        query = dspy.InputField()
        context = dspy.InputField()
        melchior_output = dspy.InputField()
        melchior_confidence = dspy.InputField()
        balthasar_output = dspy.InputField()
        balthasar_confidence = dspy.InputField()
        casper_output = dspy.InputField()
        casper_confidence = dspy.InputField()
        final_verdict = dspy.InputField()
        final_justification = dspy.InputField()

        scientific_rigor: float = dspy.OutputField()
        strategic_feasibility: float = dspy.OutputField()
        risk_management: float = dspy.OutputField()
        decision_quality: float = dspy.OutputField()
        system_coherence: float = dspy.OutputField()
        strengths: str = dspy.OutputField()
        weaknesses: str = dspy.OutputField()
        recommendation: str = dspy.OutputField()

else:
    Evaluate = None

    class _ConsensusJudge:
        pass

    class _DecisionQualityJudge:
        pass

    class _PersonaConsistencyJudge:
        pass

    class _RiskMitigationJudge:
        pass

    class _MAGISystemJudge:
        pass


ConsensusJudge = _ConsensusJudge
DecisionQualityJudge = _DecisionQualityJudge
PersonaConsistencyJudge = _PersonaConsistencyJudge
RiskMitigationJudge = _RiskMitigationJudge
MAGISystemJudge = _MAGISystemJudge


def _persona_map(pred: Any) -> Mapping[str, Any]:
    persona_outputs = getattr(pred, "persona_outputs", {})
    if isinstance(persona_outputs, Mapping):
        return persona_outputs
    return {}


def _safe_text(value: object) -> str:
    return str(value or "").strip()


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if isinstance(value, (int, float)):
            return float(value)
        return float(str(value))
    except (TypeError, ValueError):
        return default


def consensus_metric(gold: Any, pred: Any, trace: Optional[Any] = None) -> float:
    del gold, trace
    personas = _persona_map(pred)
    stances = []
    for name in ("melchior", "balthasar", "casper"):
        payload = personas.get(name, {})
        if isinstance(payload, Mapping):
            stances.append(_safe_text(payload.get("stance")))
    stances = [stance for stance in stances if stance]
    if not stances:
        return 0.0
    return 1.0 if len(set(stances)) == 1 else 0.5


def decision_quality_metric(gold: Any, pred: Any, trace: Optional[Any] = None) -> float:
    del gold, trace
    justification = _safe_text(getattr(pred, "justification", ""))
    verdict = _safe_text(getattr(pred, "verdict", ""))
    if not verdict:
        return 0.0
    score = 0.2
    if justification:
        score += 0.4
    if "[" in justification and "]" in justification:
        score += 0.2
    if verdict in {"approve", "reject", "revise"}:
        score += 0.2
    return min(1.0, score)


def personality_consistency_metric(
    gold: Any, pred: Any, trace: Optional[Any] = None
) -> float:
    del gold, trace
    personas = _persona_map(pred)
    markers = {
        "melchior": "MELCHIOR",
        "balthasar": "BALTHASAR",
        "casper": "CASPER",
    }
    matches = 0
    for name, marker in markers.items():
        payload = personas.get(name, {})
        text = _safe_text(payload.get("text")) if isinstance(payload, Mapping) else ""
        if marker in text.upper():
            matches += 1
    return matches / len(markers)


def risk_coverage_metric(gold: Any, pred: Any, trace: Optional[Any] = None) -> float:
    del gold, trace
    personas = _persona_map(pred)
    casper = personas.get("casper", {})
    if not isinstance(casper, Mapping):
        return 0.0
    risks = _safe_text(casper.get("risks"))
    mitigations = _safe_text(casper.get("mitigations"))
    score = 0.0
    if risks:
        score += 0.5
    if mitigations:
        score += 0.5
    return score


def composite_magi_metric(gold: Any, pred: Any, trace: Optional[Any] = None) -> float:
    scores = [
        consensus_metric(gold, pred, trace),
        decision_quality_metric(gold, pred, trace),
        personality_consistency_metric(gold, pred, trace),
        risk_coverage_metric(gold, pred, trace),
    ]
    return sum(scores) / len(scores)


def comprehensive_judge_metric(
    gold: Any, pred: Any, trace: Optional[Any] = None
) -> Dict[str, float]:
    del trace
    return {
        "scientific_rigor": consensus_metric(gold, pred),
        "strategic_feasibility": personality_consistency_metric(gold, pred),
        "risk_management": risk_coverage_metric(gold, pred),
        "decision_quality": decision_quality_metric(gold, pred),
        "system_coherence": composite_magi_metric(gold, pred),
        "overall": composite_magi_metric(gold, pred),
    }


def create_magi_evaluator(
    devset: Any,
    metric: Callable[[Any, Any, Optional[Any]], Any] = comprehensive_judge_metric,
    num_threads: int = 1,
) -> Any:
    if STUB_MODE or Evaluate is None:
        raise RuntimeError(
            "DSPy evaluation requires the dspy dependency and non-stub mode."
        )
    return Evaluate(devset=devset, metric=metric, num_threads=num_threads)


def magi_optimization_metric(gold: Any, pred: Any, trace: Optional[Any] = None) -> bool:
    return composite_magi_metric(gold, pred, trace) >= 0.5


__all__ = [
    "ConsensusJudge",
    "DecisionQualityJudge",
    "PersonaConsistencyJudge",
    "RiskMitigationJudge",
    "MAGISystemJudge",
    "consensus_metric",
    "decision_quality_metric",
    "personality_consistency_metric",
    "risk_coverage_metric",
    "composite_magi_metric",
    "comprehensive_judge_metric",
    "create_magi_evaluator",
    "magi_optimization_metric",
]
