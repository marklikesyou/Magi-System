import textwrap
from pathlib import Path

import pytest

from magi.eval.dataset import build_persona_outputs, load_dataset


def test_load_dataset_supports_tasks_alias(tmp_path: Path):
    payload = textwrap.dedent(
        """
        tasks:
          - id: demo
            query: Should we launch?
            expected_verdict: APPROVE
            fused:
              verdict: approve
              justification: Ready
              confidence: 0.6
            personas:
              Melchior:
                text: ok
                confidence: 0.7
        """
    ).strip()
    dataset_path = tmp_path / "tasks.yaml"
    dataset_path.write_text(payload, encoding="utf-8")
    dataset = load_dataset(dataset_path)
    assert dataset.cases[0].expected_verdict == "approve"
    outputs = build_persona_outputs(dataset.cases[0])
    assert outputs[0].name == "melchior"


def test_load_dataset_rejects_duplicate_ids(tmp_path: Path):
    payload = textwrap.dedent(
        """
        cases:
          - id: duplicate
            query: One
            expected_verdict: approve
            fused:
              verdict: approve
              justification: ok
              confidence: 0.5
            personas:
              melchior:
                text: ok
                confidence: 0.5
          - id: duplicate
            query: Two
            expected_verdict: reject
            fused:
              verdict: reject
              justification: bad
              confidence: 0.5
            personas:
              melchior:
                text: bad
                confidence: 0.5
        """
    ).strip()
    dataset_path = tmp_path / "cases.yaml"
    dataset_path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(dataset_path)
