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
