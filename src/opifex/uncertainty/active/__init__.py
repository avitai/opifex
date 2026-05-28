"""Active Learning and Bayesian Experimental Design subsystem (Task 8.3).

Subpackage modules:

* :mod:`opifex.uncertainty.active.acquisition` — pure JAX BALD / EI /
  Log-EI / UCB / LCB / PI single-point acquisition kernels operating on
  :class:`~opifex.uncertainty.types.PredictiveDistribution` objects and a
  named-strategy :func:`acquire` dispatcher.
* :mod:`opifex.uncertainty.active.batch_active` — BatchBALD greedy
  joint-MI maximisation, reparameterised Monte-Carlo batch EI, and the
  q-EHVI multi-objective acquisition.
* :mod:`opifex.uncertainty.active.experimental_design` —
  :func:`expected_information_gain` (linear-Gaussian closed form plus
  Monte-Carlo nested-sampling fallback) and the
  :func:`bayesian_experimental_design_loop` BO loop driver.
* :mod:`opifex.uncertainty.active.pinn_acquisition` — PINN residual-norm
  acquisition that ranks candidates by PDE residual magnitude and exposes
  residual + uncertainty metadata.

All RNG-dependent acquisitions route through
:func:`artifex.generative_models.core.rng.extract_rng_key` with named
streams ``"active_acquire"``, ``"active_bald"``, ``"active_eig"``.

Container patterns (GUIDE_ALIGNMENT §5a):

* :class:`AcquisitionStrategy` — :class:`~enum.StrEnum` of strategy names.
* :class:`ActiveLearningConfig` — pattern (A): frozen, slotted,
  keyword-only dataclass.
* :class:`AcquiredBatch` — pattern (B): ``flax.struct.dataclass`` that
  carries ``jax.Array`` indices + scores through the batch loop.

Primary acquisition-function reference: ``trieste`` (TensorFlow original
ported to JAX). Each kernel docstring cites the trieste source line it
was ported from.
"""

from __future__ import annotations

from opifex.uncertainty.active.acquisition import (
    AcquiredBatch,
    AcquisitionStrategy,
    ActiveLearningConfig,
    acquire,
    bald,
    expected_improvement,
    log_expected_improvement,
    lower_confidence_bound,
    probability_of_improvement,
    upper_confidence_bound,
)
from opifex.uncertainty.active.batch_active import (
    batch_bald,
    batch_mc_expected_improvement,
    q_expected_hypervolume_improvement,
)
from opifex.uncertainty.active.experimental_design import (
    BayesianExperimentalDesignResult,
    bayesian_experimental_design_loop,
    expected_information_gain,
)
from opifex.uncertainty.active.pinn_acquisition import pinn_residual_acquisition


__all__ = [
    "AcquiredBatch",
    "AcquisitionStrategy",
    "ActiveLearningConfig",
    "BayesianExperimentalDesignResult",
    "acquire",
    "bald",
    "batch_bald",
    "batch_mc_expected_improvement",
    "bayesian_experimental_design_loop",
    "expected_improvement",
    "expected_information_gain",
    "log_expected_improvement",
    "lower_confidence_bound",
    "pinn_residual_acquisition",
    "probability_of_improvement",
    "q_expected_hypervolume_improvement",
    "upper_confidence_bound",
]
