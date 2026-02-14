"""Tests for magi.eval.metrics - evaluation metric helpers."""

from __future__ import annotations

import pytest

from magi.eval.metrics import (
    accuracy,
    classification_report,
    confidence_interval,
    confusion_matrix,
    precision_recall_f1,
)


def test_accuracy_perfect():
    """All predictions matching references yields 1.0."""
    preds = ["a", "b", "c"]
    refs = ["a", "b", "c"]
    assert accuracy(preds, refs) == 1.0


def test_accuracy_none():
    """No predictions matching references yields 0.0."""
    preds = ["x", "y", "z"]
    refs = ["a", "b", "c"]
    assert accuracy(preds, refs) == 0.0


def test_accuracy_empty():
    """Empty prediction and reference lists yield 0.0."""
    assert accuracy([], []) == 0.0


def test_accuracy_partial():
    """Mixed correct/incorrect predictions return the correct fraction."""
    preds = ["a", "wrong", "c", "wrong"]
    refs = ["a", "b", "c", "d"]
    assert accuracy(preds, refs) == 0.5





def test_accuracy_length_mismatch_raises():
    """accuracy raises ValueError when lists differ in length."""
    with pytest.raises(ValueError, match="Length mismatch"):
        accuracy(["a", "b"], ["a"])


def test_precision_recall_f1_length_mismatch_raises():
    """precision_recall_f1 raises ValueError when lists differ in length."""
    with pytest.raises(ValueError, match="Length mismatch"):
        precision_recall_f1(["approve", "reject"], ["approve"])


def test_confusion_matrix_length_mismatch_raises():
    """confusion_matrix raises ValueError when lists differ in length."""
    with pytest.raises(ValueError, match="Length mismatch"):
        confusion_matrix(["approve"], ["approve", "reject"])


def test_classification_report_length_mismatch_raises():
    """classification_report raises ValueError when lists differ in length."""
    with pytest.raises(ValueError, match="Length mismatch"):
        classification_report(["approve", "reject"], ["approve"])





def test_precision_recall_f1_perfect():
    """Perfect predictions produce all 1.0 scores."""
    preds = ["approve", "reject", "revise"]
    refs = ["approve", "reject", "revise"]
    result = precision_recall_f1(preds, refs)
    assert result["macro_f1"] == 1.0
    for label in ("approve", "reject", "revise"):
        assert result[label]["precision"] == 1.0
        assert result[label]["recall"] == 1.0


def test_precision_recall_f1_empty():
    """Empty inputs return zeroed metrics."""
    result = precision_recall_f1([], [])
    assert result["macro_f1"] == 0.0





def test_confusion_matrix_correct():
    """Confusion matrix diagonal holds all counts for perfect predictions."""
    preds = ["approve", "reject", "revise", "approve"]
    refs = ["approve", "reject", "revise", "approve"]
    matrix = confusion_matrix(preds, refs)
    assert matrix["approve"]["approve"] == 2
    assert matrix["reject"]["reject"] == 1
    assert matrix["revise"]["revise"] == 1
    assert matrix["approve"]["reject"] == 0





def test_confidence_interval_bounds():
    """Wilson interval returns valid [0,1] bounds."""
    lo, hi = confidence_interval(0.8, 100, confidence=0.95)
    assert 0.0 <= lo < 0.8
    assert 0.8 < hi <= 1.0


def test_confidence_interval_perfect():
    """Perfect score still has non-trivial upper bound."""
    lo, hi = confidence_interval(1.0, 20, confidence=0.95)
    assert lo > 0.8
    assert hi == 1.0





def test_classification_report_returns_string():
    """Report returns a non-empty formatted string."""
    preds = ["approve", "reject", "approve"]
    refs = ["approve", "reject", "revise"]
    report = classification_report(preds, refs)
    assert isinstance(report, str)
    assert "precision" in report.lower()
    assert "approve" in report
