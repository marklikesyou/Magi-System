"""Comprehensive evaluation metrics for the MAGI persona decision system.

Provides accuracy, per-class precision/recall/F1, confusion matrix,
a formatted classification report, and Wilson-score confidence intervals.

No external dependencies -- all math is implemented directly.
"""

from __future__ import annotations

import math
import re
from typing import Dict, Iterable, List, Optional, Tuple, cast

DEFAULT_LABELS: List[str] = ["approve", "reject", "revise"]


_Z_VALUES: Dict[float, float] = {
    0.90: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}
_CITATION_RE = re.compile(r"\[(\d+)\]")
_GROUNDING_STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "been",
    "being",
    "but",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "that",
    "than",
    "their",
    "them",
    "they",
    "this",
    "those",
    "through",
    "were",
    "when",
    "with",
    "would",
}


def _resolve_labels(
    predictions: List[str],
    references: List[str],
    labels: Optional[List[str]] = None,
) -> List[str]:
    """Return an ordered label list. Falls back to DEFAULT_LABELS."""
    if labels is not None:
        return list(labels)

    seen = sorted(set(predictions) | set(references))
    return seen if seen else list(DEFAULT_LABELS)


def _safe_div(numerator: float, denominator: float) -> float:
    """Division that returns 0.0 when *denominator* is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    """Simple accuracy: fraction of predictions that match the references."""
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(refs)} references"
        )
    if not preds:
        return 0.0
    matches = sum(1 for pred, ref in zip(preds, refs) if pred == ref)
    return matches / len(preds)


def citation_hit_rate(text: str, retrieved_chunk_count: int) -> float:
    """Share of bracket citations that point at retrieved chunk indices."""
    citations = [int(match.group(1)) for match in _CITATION_RE.finditer(text)]
    if not citations:
        return 0.0
    if retrieved_chunk_count <= 0:
        return 0.0
    valid = sum(1 for citation in citations if 1 <= citation <= retrieved_chunk_count)
    return valid / len(citations)


def answer_support_score(answer_text: str, evidence_texts: Iterable[str]) -> float:
    """Lexical overlap between answer tokens and retrieved evidence tokens."""

    def tokens(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]{4,}", text.lower())
            if token not in _GROUNDING_STOPWORDS
        }

    answer_tokens = tokens(answer_text)
    if not answer_tokens:
        return 0.0
    evidence_tokens: set[str] = set()
    for item in evidence_texts:
        evidence_tokens.update(tokens(item))
    if not evidence_tokens:
        return 0.0
    overlap = answer_tokens & evidence_tokens
    return len(overlap) / len(answer_tokens)


def precision_recall_f1(
    predictions: Iterable[str],
    references: Iterable[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Per-class precision, recall, and F1 scores.

    Returns a dict mapping each label to its metrics, plus ``macro_f1``
    and ``weighted_f1`` summary scores.

    Example return value::

        {
            "approve": {"precision": 0.9, "recall": 0.85, "f1": 0.87},
            "reject":  {"precision": 0.8, "recall": 0.9,  "f1": 0.85},
            "revise":  {"precision": 0.7, "recall": 0.75, "f1": 0.72},
            "macro_f1":    0.81,
            "weighted_f1": 0.82,
        }
    """
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(refs)} references"
        )
    resolved_labels = _resolve_labels(preds, refs, labels)

    tp: Dict[str, int] = {lbl: 0 for lbl in resolved_labels}
    fp: Dict[str, int] = {lbl: 0 for lbl in resolved_labels}
    fn: Dict[str, int] = {lbl: 0 for lbl in resolved_labels}
    support: Dict[str, int] = {lbl: 0 for lbl in resolved_labels}

    for pred, ref in zip(preds, refs):
        if ref in support:
            support[ref] += 1
        if pred == ref and pred in tp:
            tp[pred] += 1
        else:
            if pred in fp:
                fp[pred] += 1
            if ref in fn:
                fn[ref] += 1

    result: Dict[str, object] = {}
    f1_scores: List[float] = []
    total_support = sum(support.values())

    for lbl in resolved_labels:
        prec = _safe_div(tp[lbl], tp[lbl] + fp[lbl])
        rec = _safe_div(tp[lbl], tp[lbl] + fn[lbl])
        f1 = _safe_div(2 * prec * rec, prec + rec)
        result[lbl] = {"precision": prec, "recall": rec, "f1": f1}
        f1_scores.append(f1)

    result["macro_f1"] = _safe_div(sum(f1_scores), len(f1_scores))

    weighted_sum = sum(
        f1_scores[i] * support[lbl] for i, lbl in enumerate(resolved_labels)
    )
    result["weighted_f1"] = _safe_div(weighted_sum, total_support)

    return result


def confusion_matrix(
    predictions: Iterable[str],
    references: Iterable[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, int]]:
    """Confusion matrix as a nested dict.

    Rows are the *true* labels (references), columns are the *predicted*
    labels.  ``matrix[true_label][predicted_label]`` gives the count.

    Example::

        {
            "approve": {"approve": 10, "reject": 1, "revise": 2},
            "reject":  {"approve": 0,  "reject": 8, "revise": 1},
            "revise":  {"approve": 1,  "reject": 0, "revise": 7},
        }
    """
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(refs)} references"
        )
    resolved_labels = _resolve_labels(preds, refs, labels)

    matrix: Dict[str, Dict[str, int]] = {
        true_lbl: {pred_lbl: 0 for pred_lbl in resolved_labels}
        for true_lbl in resolved_labels
    }

    for pred, ref in zip(preds, refs):
        if ref in matrix and pred in matrix[ref]:
            matrix[ref][pred] += 1

    return matrix


def classification_report(
    predictions: Iterable[str],
    references: Iterable[str],
    labels: Optional[List[str]] = None,
) -> str:
    """Formatted text report combining per-class metrics, averages, and accuracy.

    Mimics the style of sklearn's ``classification_report`` without the
    dependency.
    """
    preds = list(predictions)
    refs = list(references)
    if len(preds) != len(refs):
        raise ValueError(
            f"Length mismatch: {len(preds)} predictions vs {len(refs)} references"
        )
    resolved_labels = _resolve_labels(preds, refs, labels)

    metrics = precision_recall_f1(preds, refs, resolved_labels)
    acc = accuracy(preds, refs)
    total = len(preds)

    support: Dict[str, int] = {lbl: 0 for lbl in resolved_labels}
    for ref in refs:
        if ref in support:
            support[ref] += 1

    max_lbl_len = max((len(lbl) for lbl in resolved_labels), default=12)
    max_lbl_len = max(max_lbl_len, len("weighted avg"))
    header_label = " " * max_lbl_len
    header = f"{header_label}  {'precision':>9}  {'recall':>9}  {'f1-score':>9}  {'support':>9}"
    separator = "-" * len(header)

    lines: List[str] = [header, separator]

    for lbl in resolved_labels:
        m = cast(Dict[str, float], metrics[lbl])
        prec = m["precision"]
        rec = m["recall"]
        f1 = m["f1"]
        sup = support[lbl]
        lines.append(
            f"{lbl:<{max_lbl_len}}  {prec:>9.4f}  {rec:>9.4f}  {f1:>9.4f}  {sup:>9d}"
        )

    lines.append(separator)

    macro_f1 = cast(float, metrics["macro_f1"])

    macro_prec = _safe_div(
        sum(
            cast(Dict[str, float], metrics[lbl])["precision"] for lbl in resolved_labels
        ),
        len(resolved_labels),
    )
    macro_rec = _safe_div(
        sum(cast(Dict[str, float], metrics[lbl])["recall"] for lbl in resolved_labels),
        len(resolved_labels),
    )
    lines.append(
        f"{'macro avg':<{max_lbl_len}}  {macro_prec:>9.4f}  {macro_rec:>9.4f}  {macro_f1:>9.4f}  {total:>9d}"
    )

    weighted_f1 = cast(float, metrics["weighted_f1"])
    total_support = sum(support.values())
    weighted_prec = _safe_div(
        sum(
            cast(Dict[str, float], metrics[lbl])["precision"] * support[lbl]
            for lbl in resolved_labels
        ),
        total_support,
    )
    weighted_rec = _safe_div(
        sum(
            cast(Dict[str, float], metrics[lbl])["recall"] * support[lbl]
            for lbl in resolved_labels
        ),
        total_support,
    )
    lines.append(
        f"{'weighted avg':<{max_lbl_len}}  {weighted_prec:>9.4f}  {weighted_rec:>9.4f}  {weighted_f1:>9.4f}  {total:>9d}"
    )

    lines.append(separator)
    lines.append(
        f"{'accuracy':<{max_lbl_len}}  {' ':>9}  {' ':>9}  {acc:>9.4f}  {total:>9d}"
    )

    return "\n".join(lines)


def confidence_interval(
    score: float,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score interval for a proportion.

    More accurate than the normal approximation for small sample sizes.

    Parameters
    ----------
    score : float
        Observed proportion (e.g. accuracy), in [0, 1].
    n : int
        Number of observations.
    confidence : float
        Desired confidence level.  Supported values: 0.90, 0.95, 0.99.

    Returns
    -------
    (lower, upper) : Tuple[float, float]
        Lower and upper bounds of the interval, clipped to [0, 1].
    """
    if n <= 0:
        return (0.0, 0.0)

    z = _Z_VALUES.get(confidence)
    if z is None:
        raise ValueError(
            f"Unsupported confidence level {confidence}. "
            f"Choose from {sorted(_Z_VALUES.keys())}."
        )

    z2 = z * z
    denominator = 1 + z2 / n
    centre = (score + z2 / (2 * n)) / denominator
    margin = (z / denominator) * math.sqrt((score * (1 - score)) / n + z2 / (4 * n * n))

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (lower, upper)
