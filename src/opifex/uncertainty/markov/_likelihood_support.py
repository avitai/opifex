r"""Shared support helpers for the Markov-GP inference paths.

Two pieces of logic are byte-identical across the four inference paths
(Laplace / VI / PEP / PL) and are centralised here (Rule 1 — DRY):

* :func:`latent_variance` — unwraps the latent variance that every
  ``predict_markov_*_gp`` always sets, raising :class:`RuntimeError` (not an
  ``assert``, which ``python -O`` strips) when it is ``None``.
* :func:`interpolate_smoothed_state` — the state-space interpolation that
  propagates a smoothed state trajectory from the training grid to held-out
  test times. The four ``predict_markov_*_gp`` paths share this exact
  ``predict_one`` body; only the metadata stamped on the returned
  :class:`PredictiveDistribution` differs.

References
----------
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP §9 (state-space GP
  interpolation via the SDE transition matrix).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import StateSpaceKernel  # noqa: TC001 — runtime use
from opifex.uncertainty.types import PredictiveDistribution  # noqa: TC001 — eager per convention


# Variance floor shared by every Markov predict path: clip marginal latent
# variances away from zero to keep the downstream response maps well-defined.
_PSEUDO_NOISE_FLOOR: float = 1e-6


def latent_variance(predictive: PredictiveDistribution) -> jax.Array:
    """Return the latent variance set by every ``predict_markov_*_gp``.

    Args:
        predictive: Latent predictive returned by a Markov predict path.

    Returns:
        The marginal latent variance array.

    Raises:
        RuntimeError: When ``predictive.variance`` is ``None`` — the predict
            paths always populate it, so a ``None`` signals a contract
            violation. Uses an explicit raise (not ``assert``) so the guard
            survives ``python -O``.
    """
    if predictive.variance is None:
        raise RuntimeError("predict_markov_*_gp returned a PredictiveDistribution with no variance")
    return predictive.variance


def interpolate_smoothed_state(
    *,
    state_space_kernel: StateSpaceKernel,
    times_train: jax.Array,
    smoothed_state_means: jax.Array,
    smoothed_state_covs: jax.Array,
    times_test: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""Interpolate the smoothed state to ``times_test`` and return latent moments.

    For each test time ``t*``, locate the latest training time ``t_k <= t*`` and
    propagate the smoothed state at ``t_k`` forward by ``dt = t* - t_k`` using
    the SDE transition matrix. For test times before the first training time,
    propagate from the stationary prior (zero mean, stationary covariance).

    Shared verbatim by ``predict_markov_{laplace,vi,pep,pl}_gp`` — the inference
    algorithm differs but this smoothed-state interpolation is identical.

    Args:
        state_space_kernel: The fitted :class:`StateSpaceKernel` (provides the
            measurement operator, stationary covariance, and transition map).
        times_train: ``(n,)`` strictly-increasing training time stamps.
        smoothed_state_means: ``(n, state_dim)`` smoothed full-state trajectory.
        smoothed_state_covs: ``(n, state_dim, state_dim)`` smoothed full-state
            covariances.
        times_test: ``(m,)`` test time stamps (any order).

    Returns:
        ``(test_means, test_variances)`` — the latent ``f(t*)`` marginal mean
        and variance at each test time, variances clipped at
        :data:`_PSEUDO_NOISE_FLOOR`.
    """
    observation_matrix = state_space_kernel.measurement
    stationary_cov = state_space_kernel.stationary_cov
    state_dim = state_space_kernel.state_dim

    # For each test time t*, find the closest preceding training index via
    # right-sided searchsorted; use index 0 (with mean=0, cov=P_inf) when t*
    # precedes the first training point.
    bucket_indices = jnp.searchsorted(times_train, times_test, side="right") - 1

    def predict_one(test_time: jax.Array, bucket_index: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return the predictive mean and variance at one test time via SDE interpolation."""
        is_before_first = bucket_index < 0
        clipped_index = jnp.maximum(bucket_index, 0)
        anchor_mean = jnp.where(
            is_before_first, jnp.zeros(state_dim), smoothed_state_means[clipped_index]
        )
        anchor_cov = jnp.where(is_before_first, stationary_cov, smoothed_state_covs[clipped_index])
        anchor_time = jnp.where(is_before_first, test_time, times_train[clipped_index])
        delta = test_time - anchor_time
        transition_matrix = state_space_kernel.state_transition(delta)
        process_noise = stationary_cov - transition_matrix @ stationary_cov @ transition_matrix.T
        predicted_state_mean = transition_matrix @ anchor_mean
        predicted_state_cov = transition_matrix @ anchor_cov @ transition_matrix.T + process_noise
        latent_mean = (observation_matrix @ predicted_state_mean).squeeze(-1)
        latent_var = (observation_matrix @ predicted_state_cov @ observation_matrix.T).squeeze()
        return latent_mean, latent_var

    test_means, test_variances = jax.vmap(predict_one)(times_test, bucket_indices)
    test_variances = jnp.clip(test_variances, a_min=_PSEUDO_NOISE_FLOOR)
    return test_means, test_variances


__all__ = [
    "interpolate_smoothed_state",
    "latent_variance",
]
