r"""Tests for the parallel-scan Kalman primitives.

The parallel filter and smoother run in :math:`O(\log N)` parallel depth
via :func:`jax.lax.associative_scan` instead of the :math:`O(N)` sequential
``jax.lax.scan``. They produce numerically identical results to the
sequential reference.

Canonical reference (line-by-line port):
* ``../bayesnewton/bayesnewton/ops.py`` — ``parallel_filtering_element_``
  (line 183), ``parallel_filtering_operator`` (line 204),
  ``_parallel_kf`` (line 237), ``parallel_smoothing_element`` (line 319),
  ``parallel_smoothing_operator`` (line 329), ``_parallel_rts`` (line 338).

References
----------
* Särkkä & García-Fernández 2021 — *Temporal parallelization of Bayesian
  smoothers*, IEEE TAC arXiv:1905.13002.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.statespace import (
    kalman_filter,
    kalman_filter_parallel,
    kalman_smoother,
    kalman_smoother_parallel,
)


def _setup_problem(*, key: jax.Array, num_steps: int, state_dim: int) -> dict[str, jax.Array]:
    """Build a small linear-Gaussian state-space problem for equivalence tests."""
    key_transition, key_observations = jax.random.split(key)
    transition = jnp.eye(state_dim) + 0.05 * jax.random.normal(
        key_transition, (state_dim, state_dim)
    )
    process_noise = 0.1 * jnp.eye(state_dim)
    observation_matrix = jnp.eye(state_dim)
    observation_cov = 0.2 * jnp.eye(state_dim)
    observations = jax.random.normal(key_observations, (num_steps, state_dim))
    transitions = jnp.broadcast_to(transition, (num_steps, state_dim, state_dim))
    process_noises = jnp.broadcast_to(process_noise, (num_steps, state_dim, state_dim))
    observation_covs = jnp.broadcast_to(observation_cov, (num_steps, state_dim, state_dim))
    return {
        "transitions": transitions,
        "process_noises": process_noises,
        "observations": observations,
        "observation_matrix": observation_matrix,
        "observation_covs": observation_covs,
        "initial_mean": jnp.zeros(state_dim),
        "initial_cov": jnp.eye(state_dim),
    }


def test_kalman_filter_parallel_matches_sequential() -> None:
    """Parallel filter posterior equals sequential filter posterior."""
    problem = _setup_problem(key=jax.random.PRNGKey(0), num_steps=8, state_dim=2)
    seq_means, seq_covs = kalman_filter(**problem)
    par_means, par_covs = kalman_filter_parallel(**problem)
    assert seq_means.shape == par_means.shape
    assert seq_covs.shape == par_covs.shape
    assert jnp.allclose(seq_means, par_means, atol=1e-4)
    assert jnp.allclose(seq_covs, par_covs, atol=1e-4)


def test_kalman_smoother_parallel_matches_sequential() -> None:
    """Parallel smoother posterior equals sequential smoother posterior."""
    problem = _setup_problem(key=jax.random.PRNGKey(1), num_steps=8, state_dim=2)
    filter_means, filter_covs = kalman_filter(**problem)
    seq_means, seq_covs = kalman_smoother(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    par_means, par_covs = kalman_smoother_parallel(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    assert jnp.allclose(seq_means, par_means, atol=1e-4)
    assert jnp.allclose(seq_covs, par_covs, atol=1e-4)


def test_kalman_filter_parallel_handles_single_step() -> None:
    """Parallel filter is degenerate-safe on ``N = 1``."""
    problem = _setup_problem(key=jax.random.PRNGKey(2), num_steps=1, state_dim=2)
    par_means, par_covs = kalman_filter_parallel(**problem)
    seq_means, seq_covs = kalman_filter(**problem)
    assert par_means.shape == (1, 2)
    assert par_covs.shape == (1, 2, 2)
    assert jnp.allclose(par_means, seq_means, atol=1e-5)
    assert jnp.allclose(par_covs, seq_covs, atol=1e-5)


def test_kalman_filter_parallel_jit_compatible() -> None:
    """Parallel filter compiles under ``jax.jit`` end-to-end."""
    problem = _setup_problem(key=jax.random.PRNGKey(3), num_steps=4, state_dim=2)
    jitted = jax.jit(kalman_filter_parallel)
    means, covs = jitted(**problem)
    assert jnp.all(jnp.isfinite(means))
    assert jnp.all(jnp.isfinite(covs))


def test_kalman_smoother_parallel_recovers_filter_at_final_step() -> None:
    """At the final step the smoother covariance equals the filter covariance.

    The smoother conditions on observations through ``N``, so at step ``N``
    no future information is added — parallel and sequential implementations
    must both preserve this property.
    """
    problem = _setup_problem(key=jax.random.PRNGKey(4), num_steps=6, state_dim=2)
    filter_means, filter_covs = kalman_filter(**problem)
    smoother_means, smoother_covs = kalman_smoother_parallel(
        filter_means=filter_means,
        filter_covs=filter_covs,
        transitions=problem["transitions"],
        process_noises=problem["process_noises"],
    )
    assert jnp.allclose(smoother_means[-1], filter_means[-1], atol=1e-5)
    assert jnp.allclose(smoother_covs[-1], filter_covs[-1], atol=1e-4)
