"""State-space math primitives for opifex Kalman filtering and smoothing.

Provides the math layer for ``opifex.uncertainty.assimilation`` (Task 6.7
applied data-assimilation layer). Pure JAX; no NNX imports.

The sibling repositories ``/mnt/ssd2/Works/{bayesnewton,kalman-jax,markovflow,
ComputationAwareKalman.jl}`` serve as reference implementations only —
opifex never carries them as runtime dependencies. Algorithms here are
JAX-native and cite the sibling repo line-by-line.

References
----------
* Kalman 1960 — *A New Approach to Linear Filtering and Prediction Problems*.
* Rauch, Tung, Striebel 1965 — *Maximum Likelihood Estimates of Linear
  Dynamic Systems*, AIAA J.
* Särkkä 2013 — *Bayesian Filtering and Smoothing*.
* Pförtner, Wenger, Cockayne, Hennig arXiv:2405.08971 — *Compute-Aware
  Kalman Filtering and Smoothing* (primary CAKF/CAKS reference).
"""

from __future__ import annotations

from opifex.uncertainty.registry import UQRegistry
from opifex.uncertainty.statespace._uq_capabilities import STATESPACE_CAPABILITIES
from opifex.uncertainty.statespace.cakf import (
    cakf_predict,
    cakf_smooth,
    cakf_step,
    cakf_update,
    LowRankDowndatedMatrix,
)
from opifex.uncertainty.statespace.diagonal_ek1 import diagonal_ek1_step
from opifex.uncertainty.statespace.kalman import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_predict,
    kalman_smoother,
    kalman_update,
)
from opifex.uncertainty.statespace.kernels import (
    cosine_kernel,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    matern72_kernel,
    periodic_kernel,
    quasi_periodic_matern12_kernel,
    StateSpaceKernel,
)
from opifex.uncertainty.statespace.lti_sde import (
    discretize_lti_sde,
    process_noise_covariance,
    state_transition_matrix,
)
from opifex.uncertainty.statespace.parallel import (
    kalman_filter_parallel,
    kalman_smoother_parallel,
)
from opifex.uncertainty.statespace.sqrt_kalman import (
    sqrt_kalman_predict,
    sqrt_kalman_update,
)


# UQ capability registration — Task 7.2. Singleton :class:`UQRegistry`
# guarded against duplicate registration on repeat imports (Rule 13).
_uq_registry: UQRegistry = UQRegistry()
for _name, _capability in STATESPACE_CAPABILITIES.items():
    if _name not in _uq_registry:
        _uq_registry.register(_name, _capability)


__all__ = [
    "STATESPACE_CAPABILITIES",
    "LowRankDowndatedMatrix",
    "StateSpaceKernel",
    "cakf_predict",
    "cakf_smooth",
    "cakf_step",
    "cakf_update",
    "cosine_kernel",
    "diagonal_ek1_step",
    "discretize_lti_sde",
    "kalman_filter",
    "kalman_filter_parallel",
    "kalman_log_likelihood",
    "kalman_predict",
    "kalman_smoother",
    "kalman_smoother_parallel",
    "kalman_update",
    "matern12_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "matern72_kernel",
    "periodic_kernel",
    "process_noise_covariance",
    "quasi_periodic_matern12_kernel",
    "sqrt_kalman_predict",
    "sqrt_kalman_update",
    "state_transition_matrix",
]
