from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Tuple

from .model import get_decision_model
from .schema import PersonaOutput

logger = logging.getLogger(__name__)

Action = Literal["approve", "reject", "revise"]


@dataclass
class PersonaVote:
    name: str
    action: Action
    confidence: float
    score: float = 0.0


def _clip_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_vote(persona: PersonaOutput) -> PersonaVote:
    lowered = persona.text.lower()
    if "[approve]" in lowered:
        action: Action = "approve"
    elif "[reject]" in lowered:
        action = "reject"
    elif "[revise]" in lowered:
        action = "revise"
    else:
        action = "revise"
    return PersonaVote(
        name=persona.name,
        action=action,
        confidence=_clip_confidence(persona.confidence),
    )


def majority_weighted(votes: Iterable[PersonaVote]) -> Action:
    tally: Dict[Action, float] = {"approve": 0.0, "reject": 0.0, "revise": 0.0}
    for vote in votes:
        tally[vote.action] += _clip_confidence(vote.confidence)
    return max(tally, key=tally.get)


def choose_verdict(personas: List[PersonaOutput]) -> Action:
    votes = [parse_vote(persona) for persona in personas]
    return majority_weighted(votes)


def _get_field(source: object, key: str, default: object | None = None) -> object | None:
    if source is None:
        return default
    if hasattr(source, key):
        return getattr(source, key)
    if isinstance(source, dict):
        return source.get(key, default)
    getter = getattr(source, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            return default
    return default


def _safe_float(value: object, low: float = 0.0, high: float = 1.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return low
        return max(low, min(high, number))
    except (TypeError, ValueError):
        return low


def _normalized_length(text: object, scale: float) -> float:
    if not text:
        return 0.0
    if isinstance(text, (list, tuple, set)):
        tokens = " ".join(str(item) for item in text if item).split()
    else:
        tokens = str(text).split()
    if not tokens:
        return 0.0
    ratio = len(tokens) / scale
    return max(0.0, min(1.0, ratio))


def _risk_to_score(value: object) -> float:
    mapping = {
        "low": 1.0,
        "minimal": 0.9,
        "moderate": 0.6,
        "medium": 0.55,
        "balanced": 0.55,
        "elevated": 0.35,
        "high": 0.15,
        "critical": 0.0,
    }
    if not value:
        return 0.45
    label = str(value).strip().lower()
    if label in mapping:
        return mapping[label]
    for token, score in mapping.items():
        if token in label:
            return score
    return 0.45


def _consensus_strength(confidences: List[float]) -> float:
    if not confidences:
        return 0.0
    spread = max(confidences) - min(confidences)
    return max(0.0, min(1.0, 1.0 - spread))


def _persona_reliability(name: str, persona_obj: object) -> float:
    confidence = _safe_float(_get_field(persona_obj, "confidence"))
    base = 0.35 + 0.5 * confidence
    if name == "melchior":
        analysis = _get_field(persona_obj, "analysis") or _get_field(persona_obj, "text", "")
        coverage = _normalized_length(analysis, 220.0)
        return max(0.1, min(0.95, base * 0.55 + coverage * 0.45))
    if name == "balthasar":
        plan = _get_field(persona_obj, "plan") or _get_field(persona_obj, "text", "")
        coverage = _normalized_length(plan, 240.0)
        return max(0.1, min(0.95, base * 0.5 + coverage * 0.5))
    if name == "casper":
        mitigations = _get_field(persona_obj, "mitigations") or _get_field(persona_obj, "text", "")
        coverage = _normalized_length(mitigations, 200.0)
        risk_alignment = _risk_to_score(_get_field(persona_obj, "residual_risk"))
        return max(0.1, min(0.95, base * 0.45 + coverage * 0.3 + risk_alignment * 0.25))
    text = _get_field(persona_obj, "text", "")
    coverage = _normalized_length(text, 200.0)
    return max(0.1, min(0.95, base * 0.6 + coverage * 0.4))


def _dirichlet_vote(
    persona_outputs: List[PersonaOutput],
    persona_objects: Dict[str, object],
) -> Tuple[Dict[Action, float], Dict[str, float]]:
    counts: Dict[Action, float] = {"approve": 1.0, "reject": 1.0, "revise": 1.0}
    reliabilities: Dict[str, float] = {}
    for persona in persona_outputs:
        persona_obj = persona_objects.get(persona.name)
        reliability = _persona_reliability(persona.name, persona_obj)
        reliabilities[persona.name] = reliability
        vote = parse_vote(persona)
        counts[vote.action] += reliability * (0.8 + 0.2 * persona.confidence)
        leak = max(0.05, (1.0 - reliability) * 0.25)
        for label in counts:
            if label != vote.action:
                counts[label] += leak
    total = sum(counts.values())
    probabilities = {label: counts[label] / total for label in counts}
    return probabilities, reliabilities


def _consensus_action(
    fused: object,
    persona_objects: Dict[str, object],
    persona_outputs: List[PersonaOutput],
) -> Tuple[Action | None, Dict[Action, float], Dict[str, float], Dict[str, object]]:
    mel = persona_objects.get("melchior")
    bal = persona_objects.get("balthasar")
    cas = persona_objects.get("casper")
    if not mel or not bal or not cas:
        return None, {}, {}, {}
    mel_conf = _safe_float(_get_field(mel, "confidence"))
    bal_conf = _safe_float(_get_field(bal, "confidence"))
    cas_conf = _safe_float(_get_field(cas, "confidence"))
    confidences = [mel_conf, bal_conf, cas_conf]
    base_conf = sum(confidences) / 3.0
    consensus = _consensus_strength(confidences)
    analysis_depth = _normalized_length(_get_field(mel, "analysis"), 160.0)
    plan_depth = _normalized_length(_get_field(bal, "plan"), 160.0)
    mitigation_depth = _normalized_length(_get_field(cas, "mitigations"), 140.0)
    risk_score = _risk_to_score(_get_field(cas, "residual_risk"))
    fused_conf = _safe_float(_get_field(fused, "confidence"))
    probabilities, reliabilities = _dirichlet_vote(persona_outputs, persona_objects)
    for label in ("approve", "reject", "revise"):
        probabilities.setdefault(label, 1.0 / 3.0)
    base_safety = (risk_score * 0.6) + (mitigation_depth * 0.4)
    safety = max(risk_score, base_safety)
    safety = max(0.0, min(1.0, safety))
    evidence = (analysis_depth * 0.5) + (mel_conf * 0.5)
    strategy = (plan_depth * 0.5) + (bal_conf * 0.5)
    risk_level = max(0.0, min(1.0, 1.0 - safety))
    approve_score = (
        probabilities["approve"] * (0.6 + 0.25 * evidence + 0.15 * consensus)
        + safety * 0.35
        + fused_conf * 0.2
        - (risk_level * 0.55 + probabilities["reject"] * 0.35)
    )
    reject_score = (
        probabilities["reject"] * (0.55 + 0.25 * risk_level)
        + risk_level * 0.45
        + (1.0 - safety) * 0.3
        - (probabilities["approve"] * safety * 0.4)
    )
    revise_score = (
        probabilities["revise"] * (0.5 + 0.2 * (1.0 - consensus))
        + (1.0 - evidence) * 0.3
        + (1.0 - strategy) * 0.3
    )
    scores: Dict[Action, float] = {
        "approve": approve_score,
        "reject": reject_score,
        "revise": revise_score,
    }
    decision = max(scores, key=scores.get)
    sorted_scores = sorted(scores.values(), reverse=True)
    margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else sorted_scores[0]
    if decision == "approve" and safety < 0.4 and probabilities["approve"] < 0.4:
        decision = "revise"
    elif decision == "reject" and safety > 0.6 and probabilities["reject"] < 0.35:
        decision = "revise"
    elif decision == "revise" and margin > 0.18:
        alternate = {label: score for label, score in scores.items() if label != "revise"}
        decision = max(alternate, key=alternate.get)
    if base_conf < 0.4 and decision == "approve":
        decision = "revise"
    features: Dict[str, object] = {
        "mel_confidence": mel_conf,
        "balthasar_confidence": bal_conf,
        "casper_confidence": cas_conf,
        "base_confidence": base_conf,
        "consensus": consensus,
        "analysis_depth": analysis_depth,
        "plan_depth": plan_depth,
        "mitigation_depth": mitigation_depth,
        "risk_score": risk_score,
        "fused_confidence": fused_conf,
        "safety": safety,
        "evidence": evidence,
        "strategy": strategy,
        "risk_level": risk_level,
        "scores": scores,
        "margin": margin,
        "probabilities": probabilities,
        "reliabilities": reliabilities,
    }
    return decision, probabilities, reliabilities, features


def prepare_model_features(features: Dict[str, object]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    probabilities = features.get("probabilities", {})
    if isinstance(probabilities, dict):
        for label, value in probabilities.items():
            mapping[f"prob_{label}"] = float(value)
    reliabilities = features.get("reliabilities", {})
    if isinstance(reliabilities, dict):
        for label, value in reliabilities.items():
            mapping[f"rel_{label}"] = float(value)
    for key in (
        "safety",
        "evidence",
        "strategy",
        "risk_level",
        "base_confidence",
        "fused_confidence",
        "analysis_depth",
        "plan_depth",
        "mitigation_depth",
        "risk_score",
        "margin",
        "consensus",
    ):
        value = features.get(key)
        if value is not None:
            mapping[key] = float(value)
    return mapping


def _extract_verdict(source: object) -> Action | None:
    verdict = _get_field(source, "verdict")
    if not verdict:
        return None
    label = str(verdict).strip().lower()
    mapping: Dict[str, Action] = {
        "approve": "approve",
        "approved": "approve",
        "approval": "approve",
        "reject": "reject",
        "rejected": "reject",
        "rejection": "reject",
        "revise": "revise",
        "needs revision": "revise",
        "revision": "revise",
    }
    if label in mapping:
        return mapping[label]
    for key, value in mapping.items():
        if key in label:
            return value
    return None


def resolve_verdict_with_details(
    fused: object,
    persona_objects: Dict[str, object],
    persona_outputs: List[PersonaOutput],
) -> Tuple[Action, Dict[str, object]]:
    determined = _extract_verdict(fused)
    computed, probabilities, reliabilities, features = _consensus_action(
        fused,
        persona_objects,
        persona_outputs,
    )
    if not probabilities:
        probabilities = {"approve": 1.0 / 3.0, "reject": 1.0 / 3.0, "revise": 1.0 / 3.0}
    model_probabilities: Dict[str, float] | None = None
    combined_probabilities = dict(probabilities)
    model = get_decision_model()
    if model and features:
        model_inputs = prepare_model_features(features)
        model_probabilities = model.predict(model_inputs)
        combined_probabilities = {
            label: 0.5 * probabilities.get(label, 0.0) + 0.5 * model_probabilities.get(label, 0.0)
            for label in ("approve", "reject", "revise")
        }
        combined_choice = max(combined_probabilities, key=combined_probabilities.get)
        base_choice = computed or combined_choice
        if combined_probabilities[combined_choice] - combined_probabilities.get(base_choice, 0.0) >= 0.1:
            computed = combined_choice
        probabilities = combined_probabilities
    fused_conf = _safe_float(_get_field(fused, "confidence"))
    fused_answer = str(_get_field(fused, "final_answer") or "").strip()
    fused_steps = _get_field(fused, "next_steps") or []
    persona_stances = [str(_get_field(obj, "stance") or "").lower() for obj in persona_objects.values()]

    if determined == "revise" and computed == "reject":
        if all(stance != "reject" for stance in persona_stances if stance):
            computed = "revise"

    if fused_answer:
        if all(stance not in {"reject"} for stance in persona_stances) and (
            computed not in {"revise", "reject"}
            if computed is not None
            else determined not in {"revise", "reject"}
        ):
            computed = "approve"
            probabilities = probabilities or {"approve": 0.34, "revise": 0.33, "reject": 0.33}
            probabilities["approve"] = max(probabilities.get("approve", 0.0), 0.75)
            total = sum(probabilities.values()) or 1.0
            probabilities = {k: v / total for k, v in probabilities.items()}
            combined_probabilities = dict(probabilities)
    final: Action
    if computed:
        if not determined:
            final = computed
        elif determined == "revise" and computed != "revise":
            final = computed
        elif fused_conf < 0.6 and determined != computed:
            final = computed
        elif determined == "approve" and probabilities.get("approve", 0.0) < 0.35:
            final = computed
        else:
            final = determined
    elif determined:
        final = determined
    else:
        final = choose_verdict(persona_outputs)
    features.update(
        {
            "computed_verdict": computed,
            "fused_verdict": determined,
            "fused_confidence": fused_conf,
            "fused_residual_risk": _get_field(fused, "residual_risk"),
            "fused_final_answer": fused_answer,
            "fused_next_steps": list(fused_steps) if isinstance(fused_steps, (list, tuple, set)) else fused_steps,
            "final_verdict": final,
            "selected_probabilities": probabilities,
            "persona_names": [persona.name for persona in persona_outputs],
            "persona_stances": persona_stances,
            "persona_count": len(persona_outputs),
            "fallback_vote": determined is None and computed is None,
            "model_probabilities": model_probabilities,
            "combined_probabilities": combined_probabilities,
        }
    )
    if logger.isEnabledFor(logging.INFO):
        logger.info("decision_features %s", json.dumps(features, default=str))
    return final, features


def resolve_verdict(
    fused: object,
    persona_objects: Dict[str, object],
    persona_outputs: List[PersonaOutput],
) -> Action:
    verdict, _ = resolve_verdict_with_details(fused, persona_objects, persona_outputs)
    return verdict


__all__ = [
    "Action",
    "PersonaVote",
    "choose_verdict",
    "majority_weighted",
    "parse_vote",
    "resolve_verdict",
    "resolve_verdict_with_details",
    "prepare_model_features",
]
