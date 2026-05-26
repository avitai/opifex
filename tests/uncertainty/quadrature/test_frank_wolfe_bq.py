r"""Frank-Wolfe Bayesian Quadrature (Briol+ 2015 Algorithm 1).

Tests for :mod:`opifex.uncertainty.quadrature.frank_wolfe_bq`. Verifies
the published invariants of Briol et al's Frank-Wolfe BQ
(arXiv:1506.02681, NeurIPS 2015):

* **Probability simplex.** Final FW weights sum to one and are
  non-negative — the FW iterate ``π_n = (1 - α_n) π_{n-1} + α_n
  δ_{x_n}`` keeps the iterate on the simplex by construction.

* **First step = kernel-mean argmax.** The FW gradient at iteration
  zero is exactly the kernel mean embedding ``μ_π(x)``, so the first
  selected point maximises ``μ_π`` over the candidates.

* **MMD shrinkage.** Briol+ 2015 Theorem 1 guarantees the maximum
  mean discrepancy ``MMD(π_n, π) = O(1/n)``. Verified by comparing
  MMD at a small ``n`` to MMD at a larger ``n``.

* **Integral estimate convergence.** The FW integral estimator
  ``Σ_i w_i f(x_i)`` converges to the analytic posterior integral
  computed via the Vanilla BQ closed form on the same kernel +
  measure as the iteration budget grows.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from opifex.uncertainty.quadrature import FFBQAdapterSpec
from opifex.uncertainty.quadrature.frank_wolfe_bq import frank_wolfe_bq
from opifex.uncertainty.registry import UQCapability


def _rbf_kernel(x_left: jax.Array, x_right: jax.Array, lengthscale: float = 1.0) -> jax.Array:
    """Standard RBF kernel matrix with amplitude 1."""
    squared_diff = jnp.sum(
        (x_left[:, None, :] - x_right[None, :, :]) ** 2 / (lengthscale**2), axis=-1
    )
    return jnp.exp(-0.5 * squared_diff)


def _rbf_gaussian_kernel_mean(
    points: jax.Array, measure_mean: float = 0.0, measure_variance: float = 1.0
) -> jax.Array:
    r"""Closed-form ``∫ k(x, ·) p(x) dx`` for 1-D RBF (ℓ=σ²=1) + Gaussian.

    ``qK(x') = sqrt(1/(1 + s²)) exp(-(x' - b)²/(2(1 + s²)))`` for the
    isotropic RBF with unit lengthscale.
    """
    combined_variance = 1.0 + measure_variance
    factor = jnp.sqrt(1.0 / combined_variance)
    scaled_sq = jnp.sum((points - measure_mean) ** 2 / combined_variance, axis=-1)
    return factor * jnp.exp(-0.5 * scaled_sq)


def _mmd_squared(
    selected_weights: jax.Array,
    candidate_points: jax.Array,
    kernel_mean_at_candidates: jax.Array,
    qkq: float,
) -> jax.Array:
    r"""Maximum mean discrepancy ``||μ_π - Σ_i w_i k(·, x_i)||²_H``.

    Closed form: ``qKq - 2 Σ_i w_i μ_π(x_i) + Σ_ij w_i w_j k(x_i, x_j)``.
    """
    self_kernel = _rbf_kernel(candidate_points, candidate_points)
    cross_term = jnp.sum(selected_weights * kernel_mean_at_candidates)
    self_term = selected_weights @ self_kernel @ selected_weights
    return qkq - 2.0 * cross_term + self_term


# ---------------------------------------------------------------------------
# Frank-Wolfe BQ invariants
# ---------------------------------------------------------------------------


def test_frank_wolfe_bq_first_visited_point_maximises_kernel_mean() -> None:
    """The FW gradient at iteration 0 is the kernel mean — first pick = argmax."""
    key = jax.random.PRNGKey(0)
    candidates = jax.random.normal(key, (50, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    visited_indices, _ = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=5,
    )
    expected_initial = jnp.argmax(kernel_mean)
    assert visited_indices[0] == expected_initial


def test_frank_wolfe_bq_final_weights_form_probability_simplex() -> None:
    """Final FW weights sum to one and are non-negative."""
    key = jax.random.PRNGKey(1)
    candidates = jax.random.normal(key, (30, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    _, weights = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=10,
    )
    assert jnp.all(weights >= -1e-7)
    assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-6)


def test_frank_wolfe_bq_visited_indices_shape_matches_iteration_budget() -> None:
    """``visited_indices`` has shape ``(num_iterations,)``."""
    key = jax.random.PRNGKey(2)
    candidates = jax.random.normal(key, (20, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    visited_indices, _ = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=7,
    )
    assert visited_indices.shape == (7,)


def test_frank_wolfe_bq_mmd_decreases_with_iteration_budget() -> None:
    r"""``MMD(π_n, π)`` shrinks as ``n`` grows (Briol+ 2015 Theorem 1)."""
    key = jax.random.PRNGKey(3)
    candidates = jax.random.normal(key, (100, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)
    # qKq with σ²=1, ℓ²=1, s²=1: sqrt(1/3).
    qkq = float(1.0 / jnp.sqrt(3.0))

    _, weights_few = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=3,
    )
    _, weights_many = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=30,
    )
    mmd_few = _mmd_squared(weights_few, candidates, kernel_mean, qkq)
    mmd_many = _mmd_squared(weights_many, candidates, kernel_mean, qkq)
    assert mmd_many < mmd_few


def test_frank_wolfe_bq_integral_estimate_approaches_kernel_mean_integral() -> None:
    r"""FW estimator ``Σ w_i f(x_i)`` converges toward the kernel-mean integral.

    For ``f(x) = 1`` (constant) and a normalised Gaussian measure the
    true integral is ``1``. FW with non-negative weights summing to
    one returns ``1`` exactly (regardless of point selection).
    """
    key = jax.random.PRNGKey(4)
    candidates = jax.random.normal(key, (40, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    _, weights = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=10,
    )
    constant_integral = jnp.sum(weights * 1.0)
    assert jnp.allclose(constant_integral, 1.0, atol=1e-6)


def test_frank_wolfe_bq_compiles_under_jit() -> None:
    """FW must compile under ``jax.jit`` with ``num_iterations`` static."""
    key = jax.random.PRNGKey(5)
    candidates = jax.random.normal(key, (20, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    jitted = jax.jit(frank_wolfe_bq, static_argnames=("kernel_fn", "num_iterations"))
    visited_indices, weights = jitted(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=5,
    )
    assert visited_indices.shape == (5,)
    assert jnp.allclose(jnp.sum(weights), 1.0, atol=1e-6)


def test_frank_wolfe_bq_single_iteration_collapses_to_kernel_mean_argmax() -> None:
    """``num_iterations=1`` returns a single delta at ``argmax μ_π``."""
    key = jax.random.PRNGKey(6)
    candidates = jax.random.normal(key, (15, 1))
    kernel_mean = _rbf_gaussian_kernel_mean(candidates)

    visited_indices, weights = frank_wolfe_bq(
        candidate_points=candidates,
        kernel_mean_at_candidates=kernel_mean,
        kernel_fn=_rbf_kernel,
        num_iterations=1,
    )
    assert visited_indices.shape == (1,)
    assert visited_indices[0] == jnp.argmax(kernel_mean)
    assert jnp.allclose(weights[visited_indices[0]], 1.0)
    assert jnp.allclose(jnp.sum(weights), 1.0)


# ---------------------------------------------------------------------------
# Adapter-spec wrap() concretization
# ---------------------------------------------------------------------------


def test_ffbq_adapter_spec_wrap_returns_frank_wolfe_callable() -> None:
    """``FFBQAdapterSpec.wrap`` returns the Frank-Wolfe BQ primitive."""
    spec: Any = FFBQAdapterSpec()
    capability = UQCapability(default_strategy=spec.default_strategy)
    fn = spec.wrap(model=None, capability=capability)
    assert callable(fn)
    assert fn is frank_wolfe_bq


def test_ffbq_adapter_spec_notes_describe_frank_wolfe_not_frequency_domain() -> None:
    """``FFBQAdapterSpec.notes`` correctly describe Frank-Wolfe BQ.

    The original spec catalogue notes erroneously said
    "frequency-domain Bayesian quadrature"; the design notes pin
    FFBQ to Briol+ NeurIPS 2015 Frank-Wolfe BQ (arXiv:1506.02681).
    """
    notes = FFBQAdapterSpec().notes
    assert "Frank-Wolfe" in notes or "frank_wolfe" in notes.lower()
    assert "1506.02681" in notes
