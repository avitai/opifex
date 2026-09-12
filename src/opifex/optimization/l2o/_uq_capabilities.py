"""UQ capability declarations for the learn-to-optimize surfaces.

``opifex.optimization.l2o`` meta-trains learned optimisers and benchmarks them
against a tuned classical baseline on held-out tasks. The benchmark reports
measured point summaries (mean learning curves, per-task speedups, the median
speedup and the fraction of tasks that reach the target loss). Neither the
engine nor the learned optimisers produce a predictive distribution, an interval
or a posterior, so both are declared ``UNSUPPORTED``. Registration is explicit:
importing the package registers nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


if TYPE_CHECKING:
    from opifex.uncertainty.registry import UQRegistry


_L2O_ENGINE_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "L2OEngine meta-trains, applies and benchmarks a learned optimiser. Its "
        "benchmark returns measured point summaries over held-out tasks (mean "
        "learning curves, per-task and median speedup, fraction of tasks reaching "
        "the target loss) and no predictive distribution or interval."
    ),
)

_LEARNED_OPTIMIZER_CAPABILITY = UQCapability(
    default_strategy=DefaultStrategy.UNSUPPORTED,
    source_package="opifex",
    notes=(
        "LearnedOptimizer and its MLP, Adafactor-feature and learnable-SGD "
        "implementations map gradients to parameter updates. They carry no "
        "posterior over the optimisee or over their own meta-parameters."
    ),
)


L2O_CAPABILITIES: dict[str, UQCapability] = {
    "l2o:L2OEngine": _L2O_ENGINE_CAPABILITY,
    "l2o:LearnedOptimizer": _LEARNED_OPTIMIZER_CAPABILITY,
}


def register_l2o_capabilities(registry: UQRegistry) -> None:
    """Register the learn-to-optimize capabilities in ``registry``.

    Names already present are kept, so repeated calls are safe.

    Args:
        registry: Target registry, usually the shared :class:`UQRegistry` singleton.
    """
    for name, capability in L2O_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


__all__ = ["L2O_CAPABILITIES", "register_l2o_capabilities"]
