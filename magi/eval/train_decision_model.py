from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from magi.decision.aggregator import prepare_model_features, resolve_verdict_with_details
from magi.decision.model import MODEL_PATH
from magi.eval.dataset import EvaluationDataset, build_persona_outputs, load_dataset

CLASSES = ["approve", "reject", "revise"]


def to_label_index(label: str) -> int:
    return CLASSES.index(label)


def softmax(values: List[float]) -> List[float]:
    offset = max(values)
    exp_values = [math.exp(value - offset) for value in values]
    total = sum(exp_values) or 1.0
    return [value / total for value in exp_values]


def collect_samples(dataset: EvaluationDataset) -> Tuple[List[List[float]], List[int], List[str], List[Dict[str, float]]]:
    feature_rows: List[Dict[str, float]] = []
    labels: List[int] = []
    heuristics: List[Dict[str, float]] = []
    for case in dataset.cases:
        persona_outputs = build_persona_outputs(case)
        _, details = resolve_verdict_with_details(case.fused, case.personas, persona_outputs)
        features = prepare_model_features(details)
        feature_rows.append(features)
        labels.append(to_label_index(case.expected_verdict))
        base = details.get("probabilities", {}) or {}
        heuristics.append({label: float(base.get(label, 0.0)) for label in CLASSES})
    feature_names = sorted({name for row in feature_rows for name in row})
    samples: List[List[float]] = []
    for row in feature_rows:
        samples.append([float(row.get(name, 0.0)) for name in feature_names])
    return samples, labels, feature_names, heuristics


def train(samples: List[List[float]], labels: List[int], epochs: int = 1200, lr: float = 0.3) -> Tuple[List[List[float]], List[float]]:
    class_count = len(CLASSES)
    feature_count = len(samples[0]) if samples else 0
    weights = [[0.0 for _ in range(feature_count)] for _ in range(class_count)]
    bias = [0.0 for _ in range(class_count)]
    if not samples:
        return weights, bias
    for epoch in range(epochs):
        grad_w = [[0.0 for _ in range(feature_count)] for _ in range(class_count)]
        grad_b = [0.0 for _ in range(class_count)]
        loss = 0.0
        for x, target in zip(samples, labels):
            logits = [bias[i] + sum(weights[i][j] * x[j] for j in range(feature_count)) for i in range(class_count)]
            probs = softmax(logits)
            for i in range(class_count):
                indicator = 1.0 if i == target else 0.0
                diff = probs[i] - indicator
                for j in range(feature_count):
                    grad_w[i][j] += diff * x[j]
                grad_b[i] += diff
            loss += -math.log(max(probs[target], 1e-9))
        scale = 1.0 / len(samples)
        for i in range(class_count):
            for j in range(feature_count):
                weights[i][j] -= lr * grad_w[i][j] * scale
            bias[i] -= lr * grad_b[i] * scale
        if loss * scale < 1e-4:
            break
    return weights, bias


def evaluate(samples: List[List[float]], labels: List[int], weights: List[List[float]], bias: List[float]) -> float:
    if not samples:
        return 0.0
    correct = 0
    for x, target in zip(samples, labels):
        logits = [bias[i] + sum(weights[i][j] * x[j] for j in range(len(x))) for i in range(len(CLASSES))]
        probs = softmax(logits)
        prediction = max(range(len(CLASSES)), key=lambda idx: probs[idx])
        if prediction == target:
            correct += 1
    return correct / len(samples)


def save_model(weights: List[List[float]], bias: List[float], feature_names: List[str], path: Path) -> None:
    payload = {
        "version": 1,
        "weights": {
            label: {feature_names[i]: weights[class_index][i] for i in range(len(feature_names))}
            for class_index, label in enumerate(CLASSES)
        },
        "bias": {label: bias[class_index] for class_index, label in enumerate(CLASSES)},
        "features": feature_names,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def split_indices(count: int, val_ratio: float = 0.3) -> Tuple[List[int], List[int]]:
    if count == 0:
        return [], []
    val_size = max(1, int(round(count * val_ratio)))
    train_size = max(1, count - val_size)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, count))
    if not val_indices:
        val_indices = [train_indices.pop()]
    return train_indices, val_indices


def model_predict(weights: List[List[float]], bias: List[float], sample: List[float]) -> Dict[str, float]:
    logits = [bias[i] + sum(weights[i][j] * sample[j] for j in range(len(sample))) for i in range(len(CLASSES))]
    probs = softmax(logits)
    return {CLASSES[i]: probs[i] for i in range(len(CLASSES))}


def evaluate_split(indices: List[int], weights: List[List[float]], bias: List[float], samples: List[List[float]], labels: List[int], heuristics: List[Dict[str, float]]) -> Tuple[float, float, float]:
    if not indices:
        return 0.0, 0.0, 0.0
    base_correct = 0
    model_correct = 0
    blended_correct = 0
    for idx in indices:
        label = CLASSES[labels[idx]]
        base_probs = heuristics[idx]
        base_choice = max(base_probs, key=base_probs.get)
        if base_choice == label:
            base_correct += 1
        model_probs = model_predict(weights, bias, samples[idx])
        model_choice = max(model_probs, key=model_probs.get)
        if model_choice == label:
            model_correct += 1
        blended_probs = {cls: 0.5 * base_probs.get(cls, 0.0) + 0.5 * model_probs.get(cls, 0.0) for cls in CLASSES}
        blended_choice = max(blended_probs, key=blended_probs.get)
        if blended_choice == label:
            blended_correct += 1
    total = len(indices)
    return base_correct / total, model_correct / total, blended_correct / total


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MAGI decision model from labeled cases.")
    parser.add_argument(
        "--cases",
        type=Path,
        required=True,
        help="Path to YAML file containing evaluation cases.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=MODEL_PATH,
        help="Destination for trained model weights (JSON).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.3,
        help="Validation split ratio for hold-out reporting (0-1).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1200,
        help="Maximum training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="Learning rate for logistic regression.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = load_dataset(args.cases)
    samples, labels, feature_names, heuristics = collect_samples(dataset)
    train_idx, val_idx = split_indices(len(samples), val_ratio=args.val_ratio)
    train_samples = [samples[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    weights, bias = train(train_samples, train_labels, epochs=args.epochs, lr=args.learning_rate)
    train_base, train_model, train_blended = evaluate_split(train_idx, weights, bias, samples, labels, heuristics)
    val_base, val_model, val_blended = evaluate_split(val_idx, weights, bias, samples, labels, heuristics)
    print(f"train_heuristic_accuracy\t{train_base:.2%}")
    print(f"train_model_accuracy\t{train_model:.2%}")
    print(f"train_blended_accuracy\t{train_blended:.2%}")
    print(f"val_heuristic_accuracy\t{val_base:.2%}")
    print(f"val_model_accuracy\t{val_model:.2%}")
    print(f"val_blended_accuracy\t{val_blended:.2%}")
    final_weights, final_bias = train(samples, labels, epochs=args.epochs, lr=args.learning_rate)
    save_model(final_weights, final_bias, feature_names, args.model_out)
    print(f"model_saved\t{args.model_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
