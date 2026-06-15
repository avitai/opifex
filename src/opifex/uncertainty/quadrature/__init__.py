"""Bayesian-quadrature primitives.

Functional building blocks for integrating a function ``f`` against a
probability measure ``π`` with calibrated uncertainty in the integral.
The initial slice provides only the no-GP baseline
(:func:`bayesian_monte_carlo`) plus a typed result container
(:class:`IntegralEstimate`); advanced methods (WSABI-L, vanilla BQ,
SOBER, FFBQ) will land as additional modules and re-export from this
package's namespace.

References
----------
* Rasmussen, C. E. & Ghahramani, Z. 2003 — *Bayesian Monte Carlo*,
  NeurIPS 16.
* Briol, F.-X. et al. 2019 — *Probabilistic Integration: A Role in
  Statistical Computation?*, Statistical Science 34(1).
"""

from __future__ import annotations

from opifex.uncertainty.quadrature._specs import (
    EmukitQuadratureAdapterSpec,
    FFBQAdapterSpec,
    SOBERAdapterSpec,
    VanillaBayesianQuadratureAdapterSpec,
    WSABILAdapterSpec,
)
from opifex.uncertainty.quadrature._uq_capabilities import QUADRATURE_CAPABILITIES
from opifex.uncertainty.quadrature.bayesian_monte_carlo import (
    bayesian_monte_carlo,
    IntegralEstimate,
)
from opifex.uncertainty.quadrature.measures import GaussianMeasure, LebesgueMeasure
from opifex.uncertainty.registry import UQRegistry


# UQ capability registration — Task 7.2. Singleton :class:`UQRegistry`
# guarded against duplicate registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in QUADRATURE_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "QUADRATURE_CAPABILITIES",
    "EmukitQuadratureAdapterSpec",
    "FFBQAdapterSpec",
    "GaussianMeasure",
    "IntegralEstimate",
    "LebesgueMeasure",
    "SOBERAdapterSpec",
    "VanillaBayesianQuadratureAdapterSpec",
    "WSABILAdapterSpec",
    "bayesian_monte_carlo",
]
