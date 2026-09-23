"""Equation-discovery surfaces: SINDy family + symbolic regression.

Top-level package re-exports :class:`SymbolicRegressor` for convenience;
the SINDy family is available under :mod:`opifex.discovery.sindy`. The UQ
capability declarations of both are collected into
:data:`opifex.capabilities.UQ_CAPABILITIES`.
"""

from opifex.discovery._uq_capabilities import DISCOVERY_CAPABILITIES
from opifex.discovery.symbolic import SymbolicRegressionConfig, SymbolicRegressor


__all__ = [
    "DISCOVERY_CAPABILITIES",
    "SymbolicRegressionConfig",
    "SymbolicRegressor",
]
