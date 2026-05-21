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

from opifex.uncertainty.statespace.kalman import (
    kalman_filter,
    kalman_log_likelihood,
    kalman_predict,
    kalman_smoother,
    kalman_update,
)
from opifex.uncertainty.statespace.lti_sde import discretize_lti_sde
from opifex.uncertainty.statespace.parallel import (
    kalman_filter_parallel,
    kalman_smoother_parallel,
)
from opifex.uncertainty.statespace.sqrt_kalman import (
    sqrt_kalman_predict,
    sqrt_kalman_update,
)


__all__ = [
    "discretize_lti_sde",
    "kalman_filter",
    "kalman_filter_parallel",
    "kalman_log_likelihood",
    "kalman_predict",
    "kalman_smoother",
    "kalman_smoother_parallel",
    "kalman_update",
    "sqrt_kalman_predict",
    "sqrt_kalman_update",
]
