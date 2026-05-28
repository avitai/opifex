"""Simulation-Based Inference subsystem (Task 8.2).

Subpackage modules:

* :mod:`opifex.uncertainty.sbi.simulators` — :class:`Simulator` static
  container + :func:`sample_joint` joint-sampling helper.
* :mod:`opifex.uncertainty.sbi.posterior_estimation` —
  :class:`NeuralPosteriorEstimator` (NPE) fitting ``q(theta | x)``.
* :mod:`opifex.uncertainty.sbi.likelihood_estimation` —
  :class:`NeuralLikelihoodEstimator` (NLE) fitting ``q(x | theta)`` +
  posterior MCMC via the BlackJAX backend.
* :mod:`opifex.uncertainty.sbi.ratio_estimation` —
  :class:`NeuralRatioEstimator` (NRE) fitting ``log r(theta, x)`` +
  posterior MCMC via the BlackJAX backend.
* :mod:`opifex.uncertainty.sbi.diagnostics` — Simulation-Based Calibration
  (SBC) + expected posterior contraction.

Default density-estimator backend wraps Artifex's NNX-native flows
(``ConditionalRealNVP``); optional backends (``bijx`` / ``sbiax`` /
``flowMC``) raise :class:`ImportError` with the canonical install hint
when not present. MCMC sampling for NLE / NRE routes through
:class:`opifex.uncertainty.inference_backends.BlackJAXBackend`.

References (read-only):

* Greenberg, Nonnenmacher, Macke (2019) — APT/NPE, ``arXiv:1905.07488``.
* Papamakarios, Sterratt, Murray (2019) — NLE, ``arXiv:1805.07226``.
* Hermans, Begy, Louppe (2020) — NRE, ``arXiv:1903.04057``.
* Talts, Betancourt, Simpson, Vehtari, Gelman (2018) — SBC,
  ``arXiv:1804.06788``.
"""

from __future__ import annotations

from opifex.uncertainty.registry import UQRegistry
from opifex.uncertainty.sbi._uq_capabilities import SBI_CAPABILITIES
from opifex.uncertainty.sbi.diagnostics import (
    expected_posterior_contraction,
    PosteriorContractionResult,
    SBCResult,
    simulation_based_calibration,
)
from opifex.uncertainty.sbi.likelihood_estimation import (
    NeuralLikelihoodEstimator,
    NLEState,
)
from opifex.uncertainty.sbi.posterior_estimation import (
    NeuralPosteriorEstimator,
    NPEState,
)
from opifex.uncertainty.sbi.ratio_estimation import (
    NeuralRatioEstimator,
    NREState,
)
from opifex.uncertainty.sbi.simulators import (
    sample_joint,
    Simulator,
)


# Idempotent capability registration (Rule 13 — no mutable side effects
# beyond constants + idempotent registry seeding). The :class:`UQRegistry`
# is a singleton; guard against double-registration on repeat imports.
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in SBI_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "NLEState",
    "NPEState",
    "NREState",
    "NeuralLikelihoodEstimator",
    "NeuralPosteriorEstimator",
    "NeuralRatioEstimator",
    "PosteriorContractionResult",
    "SBCResult",
    "Simulator",
    "expected_posterior_contraction",
    "sample_joint",
    "simulation_based_calibration",
]
