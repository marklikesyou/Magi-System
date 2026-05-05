from __future__ import annotations

from typing import Any

from magi.dspy_programs import optimization


def test_create_optimization_pipeline_dispatches_gepa(monkeypatch) -> None:
    calls: dict[str, Any] = {}

    def fake_gepa_program(
        program: object,
        trainset: list[Any],
        metric_fn: object,
        *,
        valset: list[Any] | None = None,
        max_metric_calls: int | None = None,
    ) -> str:
        calls["program"] = program
        calls["trainset"] = trainset
        calls["metric_fn"] = metric_fn
        calls["valset"] = valset
        calls["max_metric_calls"] = max_metric_calls
        return "optimized-gepa-program"

    monkeypatch.setattr(optimization, "STUB_MODE", False)
    monkeypatch.setattr(optimization, "gepa_program", fake_gepa_program, raising=False)

    trainset = [object()]
    valset = [object()]
    result = optimization.create_optimization_pipeline(
        retriever=lambda query: "",
        trainset=trainset,
        valset=valset,
        optimization_method="gepa",
        max_metric_calls=11,
    )

    assert result == "optimized-gepa-program"
    assert calls["trainset"] == trainset
    assert calls["valset"] == valset
    assert calls["max_metric_calls"] == 11
