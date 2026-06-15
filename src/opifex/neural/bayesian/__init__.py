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


# UQ capability registration — Task 7.2. The singleton :class:`UQRegistry`
# is shared with every other surface (operators, solvers, subpackages).
# Guarded by ``name not in registry`` so re-imports during test sessions
# don't trigger CalibraX's duplicate-rejection (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in BAYESIAN_MODEL_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


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
]
