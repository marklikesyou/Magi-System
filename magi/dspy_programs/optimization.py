"""DSPy optimization helpers for MAGI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from .compile import bootstrap_program, compile_program
from .evaluation import (
    comprehensive_judge_metric,
    create_magi_evaluator,
    magi_optimization_metric,
)
from .personas import MagiProgram
from .signatures import STUB_MODE


def _require_dspy() -> None:
    if STUB_MODE:
        raise RuntimeError(
            "DSPy optimization requires the dspy dependency and non-stub mode."
        )


@dataclass
class MAGIOptimizer:
    retriever: Callable[[str], str]
    trainset: list[Any]
    valset: list[Any] = field(default_factory=list)
    metric: Callable[[Any, Any, Optional[Any]], Any] = magi_optimization_metric

    def __post_init__(self) -> None:
        self.base_program = MagiProgram(self.retriever)

    def optimize_with_bootstrap(
        self,
        *,
        shots: int = 6,
        metric: Optional[Callable[[Any, Any, Optional[Any]], Any]] = None,
    ) -> MagiProgram:
        _require_dspy()
        return bootstrap_program(
            self.base_program, self.trainset, metric or self.metric, shots=shots
        )

    def optimize_with_random_search(
        self,
        *,
        shots: int = 8,
        metric: Optional[Callable[[Any, Any, Optional[Any]], Any]] = None,
    ) -> MagiProgram:
        _require_dspy()
        return bootstrap_program(
            self.base_program, self.trainset, metric or self.metric, shots=shots
        )

    def optimize_with_mipro(
        self,
        *,
        heavy: bool = False,
        metric: Optional[Callable[[Any, Any, Optional[Any]], Any]] = None,
    ) -> MagiProgram:
        _require_dspy()
        return compile_program(
            self.base_program, self.trainset, metric or self.metric, heavy=heavy
        )

    def optimize_signatures(
        self,
        *,
        heavy: bool = False,
        metric: Optional[Callable[[Any, Any, Optional[Any]], Any]] = None,
    ) -> MagiProgram:
        _require_dspy()
        return compile_program(
            self.base_program, self.trainset, metric or self.metric, heavy=heavy
        )

    def evaluate_optimization(
        self,
        optimized_program: MagiProgram,
        *,
        testset: Optional[Sequence[Any]] = None,
    ) -> Mapping[str, Any]:
        _require_dspy()
        dataset = list(testset or self.valset or self.trainset)
        evaluator = create_magi_evaluator(dataset, metric=comprehensive_judge_metric)
        base_scores = evaluator(self.base_program)
        optimized_scores = evaluator(optimized_program)
        return {
            "base": base_scores,
            "optimized": optimized_scores,
        }


@dataclass
class AdaptiveMAGI:
    retriever: Callable[[str], str]
    max_history_size: int = 50
    adaptation_threshold: int = 10

    def __post_init__(self) -> None:
        self.base_program = MagiProgram(self.retriever)
        self.feedback_history: list[dict[str, Any]] = []
        self.adaptation_count = 0

    def __call__(
        self, query: str, constraints: str = "", collect_feedback: bool = True
    ) -> tuple[Any, dict[str, Any]]:
        return self.forward(
            query, constraints=constraints, collect_feedback=collect_feedback
        )

    def forward(
        self, query: str, constraints: str = "", collect_feedback: bool = True
    ) -> tuple[Any, dict[str, Any]]:
        fused, personas = self.base_program(query, constraints)
        if collect_feedback:
            self.feedback_history.append(
                {
                    "query": query,
                    "constraints": constraints,
                    "fused": fused,
                    "personas": personas,
                }
            )
            self.feedback_history = self.feedback_history[-self.max_history_size :]
            if len(self.feedback_history) >= self.adaptation_threshold:
                self.adaptation_count += 1
        return fused, personas


def create_optimization_pipeline(
    retriever: Callable[[str], str],
    trainset: list[Any],
    valset: Optional[list[Any]] = None,
    optimization_method: str = "mipro",
    **kwargs: Any,
) -> MagiProgram:
    optimizer = MAGIOptimizer(
        retriever=retriever, trainset=trainset, valset=valset or []
    )
    method = optimization_method.strip().lower()
    if method == "bootstrap":
        return optimizer.optimize_with_bootstrap(**kwargs)
    if method in {"random-search", "random_search"}:
        return optimizer.optimize_with_random_search(**kwargs)
    if method in {"signature", "signatures"}:
        return optimizer.optimize_signatures(**kwargs)
    return optimizer.optimize_with_mipro(**kwargs)


__all__ = ["MAGIOptimizer", "AdaptiveMAGI", "create_optimization_pipeline"]
