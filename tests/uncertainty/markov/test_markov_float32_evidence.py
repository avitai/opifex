"""Float32 Markov GP evidence on a dense time grid.

A Matern-7/2 prior on 1000 points spaced 1e-3 lengthscales apart has process-noise components far
below float32 resolution relative to the stationary covariance. The float32 fit must still reach
the exact log marginal likelihood of the dense GP, computed in float64 from Rasmussen & Williams
(2006, eq. 4.16), to within 1e-2 nats. On this grid and data the Stillfjord-Tronarp process noise
measured 1.1e-4 nats, the closed-form increment it replaced 6.0e-2 nats, and the stationary
identity ``P_inf - A P_inf A^T`` of opifex 0.2.5 3.4 nats.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from opifex.uncertainty.markov import fit_gaussian_markov_laplace_gp
from opifex.uncertainty.statespace import matern72_kernel


_NUM_POINTS = 1000
_SPACING = 1e-3
_NOISE_VARIANCE = 0.1
_EVIDENCE_TOLERANCE = 1e-2


def _half_integer_matern(distance: np.ndarray, order: int) -> np.ndarray:
    """Return the unit Matern-(order + 1/2) covariance, Rasmussen & Williams (2006) eq. (4.16)."""
    scaled = math.sqrt(2 * order + 1) * distance
    total = sum(
        math.factorial(order + i)
        / (math.factorial(i) * math.factorial(order - i))
        * (2.0 * scaled) ** (order - i)
        for i in range(order + 1)
    )
    return np.exp(-scaled) * math.factorial(order) / math.factorial(2 * order) * total


def test_float32_evidence_on_a_dense_grid_matches_the_dense_gp() -> None:
    """The float32 Markov evidence is within 1e-2 nats of the exact float64 dense GP."""
    rng = np.random.default_rng(7)
    times = _SPACING * np.arange(_NUM_POINTS, dtype=np.float64)
    signal = np.sin(2.0 * math.pi * 3.0 * times / times[-1])
    observations = signal + math.sqrt(_NOISE_VARIANCE) * rng.standard_normal(_NUM_POINTS)

    gram = _half_integer_matern(np.abs(times[:, None] - times[None, :]), 3)
    factor = cho_factor(gram + _NOISE_VARIANCE * np.eye(_NUM_POINTS), lower=True)
    exact = float(
        -0.5 * observations @ cho_solve(factor, observations)
        - np.sum(np.log(np.diag(factor[0])))
        - 0.5 * _NUM_POINTS * math.log(2.0 * math.pi)
    )

    state = fit_gaussian_markov_laplace_gp(
        times=jnp.asarray(times, dtype=jnp.float32),
        observations=jnp.asarray(observations, dtype=jnp.float32),
        state_space_kernel=matern72_kernel(variance=1.0, lengthscale=1.0),
        noise_std=math.sqrt(_NOISE_VARIANCE),
    )
    error = abs(float(state.log_marginal_likelihood) - exact)
    assert error <= _EVIDENCE_TOLERANCE, (float(state.log_marginal_likelihood), exact, error)
