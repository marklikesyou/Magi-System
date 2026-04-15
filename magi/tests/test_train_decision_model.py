from __future__ import annotations

import pytest

from magi.eval.train_decision_model import split_indices


def test_split_indices_allows_no_validation_split() -> None:
    train_idx, val_idx = split_indices(3, val_ratio=0.0, seed=1)

    assert sorted(train_idx) == [0, 1, 2]
    assert val_idx == []


def test_split_indices_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="val_ratio"):
        split_indices(3, val_ratio=1.0)

    with pytest.raises(ValueError, match="val_ratio"):
        split_indices(3, val_ratio=-0.1)
