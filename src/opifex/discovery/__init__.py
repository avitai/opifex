"""Equation-discovery surfaces: SINDy family + symbolic regression.

Top-level package re-exports :class:`SymbolicRegressor` for convenience;
the SINDy family is available under :mod:`opifex.discovery.sindy`. UQ
capability declarations for both subsurfaces are registered into the
singleton :class:`UQRegistry` at import time (Task 7.5).
"""

from opifex.discovery._uq_capabilities import DISCOVERY_CAPABILITIES
from opifex.discovery.symbolic import SymbolicRegressionConfig, SymbolicRegressor
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.5. Guarded against duplicate
# registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in DISCOVERY_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "DISCOVERY_CAPABILITIES",
    "SymbolicRegressionConfig",
    "SymbolicRegressor",
]
