from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import math

Action = str

MODEL_PATH = Path(__file__).resolve().with_name("model_weights.json")
_MODEL_CACHE: Optional["DecisionModel"] = None


def _softmax(logits: Dict[Action, float]) -> Dict[Action, float]:
    max_logit = max(logits.values()) if logits else 0.0
    exp_values = {label: math.exp(value - max_logit) for label, value in logits.items()}
    total = sum(exp_values.values()) or 1.0
    return {label: exp_values[label] / total for label in exp_values}


@dataclass
class DecisionModel:
    weights: Dict[Action, Dict[str, float]]
    bias: Dict[Action, float]
    version: int = 1

    def predict(self, features: Dict[str, float]) -> Dict[Action, float]:
        logits: Dict[Action, float] = {}
        for label, weight_map in self.weights.items():
            total = self.bias.get(label, 0.0)
            for name, weight in weight_map.items():
                total += weight * features.get(name, 0.0)
            logits[label] = total
        return _softmax(logits)


def load_model(path: Optional[Path] = None) -> Optional[DecisionModel]:
    target = path or MODEL_PATH
    if not target.exists():
        return None
    payload = json.loads(target.read_text(encoding="utf-8"))
    weights = payload.get("weights", {})
    bias = payload.get("bias", {})
    version = payload.get("version", 1)
    if not weights:
        return None
    return DecisionModel(weights=weights, bias=bias, version=version)


def get_decision_model() -> Optional[DecisionModel]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_model()
    return _MODEL_CACHE


__all__ = ["DecisionModel", "get_decision_model", "load_model"]
