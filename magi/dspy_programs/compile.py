from __future__ import annotations

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, MIPROv2
except ImportError:
    dspy = None
    BootstrapFewShot = None
    MIPROv2 = None


def compile_program(program, trainset, metric_fn, *, heavy=False):
    if dspy is None or MIPROv2 is None:
        raise RuntimeError(
            "dspy is required for compilation. Install with: pip install 'magi-system[dspy]'"
        )
    optimizer = MIPROv2(metric=metric_fn, auto="heavy" if heavy else "medium")
    return optimizer.compile(program, trainset=trainset)


def bootstrap_program(program, trainset, metric_fn, *, shots=6):
    if dspy is None or BootstrapFewShot is None:
        raise RuntimeError(
            "dspy is required for bootstrapping. Install with: pip install 'magi-system[dspy]'"
        )
    optimizer = BootstrapFewShot(metric=metric_fn, max_bootstrapped_demos=shots)
    return optimizer.compile(program, trainset=trainset)


def gepa_program(
    program,
    trainset,
    metric_fn,
    *,
    valset=None,
    max_metric_calls=None,
    reflection_lm=None,
    track_stats=False,
    **kwargs,
):
    if dspy is None or not hasattr(dspy, "GEPA"):
        raise RuntimeError(
            "dspy.GEPA is required for GEPA optimization. Install with: pip install 'magi-system[dspy]'"
        )
    optimizer_kwargs = {
        "metric": metric_fn,
        "track_stats": track_stats,
        **kwargs,
    }
    if max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = max_metric_calls
    if reflection_lm is not None:
        optimizer_kwargs["reflection_lm"] = reflection_lm
    optimizer = dspy.GEPA(**optimizer_kwargs)
    compile_kwargs = {"student": program, "trainset": trainset}
    if valset is not None:
        compile_kwargs["valset"] = valset
    return optimizer.compile(**compile_kwargs)
