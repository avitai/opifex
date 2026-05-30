"""Bayesian neural network components with uncertainty quantification."""

from opifex.neural.bayesian._uq_capabilities import BAYESIAN_MODEL_CAPABILITIES
from opifex.neural.bayesian.calibration_tools import (
    CalibrationTools,
    IsotonicRegression,
    PlattScaling,
    TemperatureScaling,
)
from opifex.neural.bayesian.variational_framework import (
    AmortizedVariationalFramework,
    MeanFieldGaussian,
    PriorConfig,
    UncertaintyEncoder,
    VariationalConfig,
)
from opifex.uncertainty.registry import UQRegistry


def register_bayesian_capabilities(registry: UQRegistry) -> None:
    """Register the Task 7.2 Bayesian model capabilities into ``registry``.

    Explicit registration — called from a composition root rather than at
    import time (Rule 13: no mutable side effects on ``import``). The shared
    singleton :class:`UQRegistry` is populated with the ``ProbabilisticPINN``
    and ``MultiFidelityPINN`` model declarations.

    Idempotent: names already present are skipped, so repeated calls (and the
    re-entrancy of :func:`bayesian_uq_registry`) never trip CalibraX's
    duplicate-registration rejection.

    Args:
        registry: Target :class:`UQRegistry`. Almost always the singleton
            instance, but any registry is accepted for test isolation.
    """
    for name, capability in BAYESIAN_MODEL_CAPABILITIES.items():
        if name not in registry:
            registry.register(name, capability)


def bayesian_uq_registry() -> UQRegistry:
    """Return the shared singleton ``UQRegistry`` with bayesian models registered.

    Lazy composition-root accessor: callers that need the registry already
    holding the Task 7.2 model capabilities use this instead of relying on an
    import-time side effect. Registration is idempotent, so this is safe to
    call repeatedly.
    """
    registry = UQRegistry()
    register_bayesian_capabilities(registry)
    return registry


__all__ = [
    "BAYESIAN_MODEL_CAPABILITIES",
    "AmortizedVariationalFramework",
    "CalibrationTools",
    "IsotonicRegression",
    "MeanFieldGaussian",
    "PlattScaling",
    "PriorConfig",
    "TemperatureScaling",
    "UncertaintyEncoder",
    "VariationalConfig",
    "bayesian_uq_registry",
    "register_bayesian_capabilities",
]
