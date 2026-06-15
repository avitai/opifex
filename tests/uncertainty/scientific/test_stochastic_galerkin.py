"""Tests for the Stochastic-Galerkin / Stochastic-Collocation surrogates (Task 8.4).

The plan splits responsibilities:

* :class:`StochasticGalerkinSurrogate` — least-squares ``fit`` against a
  caller-supplied model under a Monte-Carlo + orthonormal-PCE basis.
* :class:`StochasticCollocationSurrogate` — Smolyak sparse-grid
  collocation; error is monotonically non-increasing in the sparse-grid
  level for smooth integrands.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.scientific.polynomial_chaos import (
    PolynomialChaosBasis,
    smolyak_sparse_grid,
    StochasticCollocationSurrogate,
    StochasticGalerkinSurrogate,
    tensor_grid_gauss_hermite,
)
from opifex.uncertainty.scientific.stochastic_galerkin import (
    evaluate_galerkin_surrogate,
    fit_collocation_surrogate,
    fit_galerkin_surrogate,
)


def _smooth_target(x: jax.Array) -> jax.Array:
    """Smooth scalar target: f(xi) = sin(xi) * exp(-0.25 * xi**2).

    The exponential damping keeps the integrand well-behaved under the
    Gaussian weight, so Gauss-Hermite + Smolyak refinement should drive
    the surrogate error to zero monotonically.
    """
    xi = x[..., 0]
    return jnp.sin(xi) * jnp.exp(-0.25 * xi**2)


def test_stochastic_galerkin_fit_is_deterministic_given_a_fixed_key() -> None:
    """Plan exit criterion: ``StochasticGalerkinSurrogate.fit(...)``
    returns identical coefficients on two calls with the same key.
    """
    basis = PolynomialChaosBasis(family="hermite", order=4, coefficients=jnp.zeros((5,)))
    rngs = {"sg_samples": jax.random.PRNGKey(0)}
    surrogate_a = fit_galerkin_surrogate(
        model=_smooth_target, basis=basis, num_samples=512, rngs=rngs
    )
    surrogate_b = fit_galerkin_surrogate(
        model=_smooth_target, basis=basis, num_samples=512, rngs=rngs
    )
    assert isinstance(surrogate_a, StochasticGalerkinSurrogate)
    assert jnp.allclose(surrogate_a.coefficients, surrogate_b.coefficients)


def test_stochastic_galerkin_evaluate_recovers_smooth_target() -> None:
    basis = PolynomialChaosBasis(family="hermite", order=6, coefficients=jnp.zeros((7,)))
    rngs = {"sg_samples": jax.random.PRNGKey(0)}
    surrogate = fit_galerkin_surrogate(
        model=_smooth_target, basis=basis, num_samples=8192, rngs=rngs
    )
    grid = jnp.linspace(-2.0, 2.0, 17)[:, None]
    predicted = evaluate_galerkin_surrogate(surrogate=surrogate, x=grid)
    target = _smooth_target(grid)
    rel = jnp.linalg.norm(predicted - target) / jnp.linalg.norm(target)
    # Monte-Carlo Galerkin regression error is O(1/sqrt(N)) on a finite-
    # order basis; 12 % tolerance covers stochastic sampling noise at
    # N=8192 with order=6 on the chosen smooth target.
    assert float(rel) < 0.12


def test_stochastic_collocation_error_is_monotonically_non_increasing() -> None:
    """Plan exit criterion: sparse-grid Smolyak refinement gives monotone
    error decay on a smooth target.

    On Gauss-Hermite nodes, the surrogate is the Lagrange interpolant at
    the sparse-grid nodes; the L2 error at a dense test grid must be
    non-increasing as the sparse-grid level rises.
    """
    test_grid = jnp.linspace(-2.0, 2.0, 33)[:, None]
    target = _smooth_target(test_grid)

    errors: list[float] = []
    for level in (1, 2, 3, 4):
        grid = smolyak_sparse_grid(level=level, num_dims=1, family="hermite")
        surrogate = fit_collocation_surrogate(model=_smooth_target, sparse_grid=grid, rngs={})
        assert isinstance(surrogate, StochasticCollocationSurrogate)
        predicted = surrogate.evaluate(test_grid)
        err = float(jnp.linalg.norm(predicted - target))
        errors.append(err)

    # Tolerate a tiny numerical jitter (<= 1e-10 absolute) but require
    # non-increasing behaviour overall.
    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i] + 1e-10, f"Collocation error not monotone: {errors}"


def test_tensor_grid_gauss_hermite_returns_known_count() -> None:
    """For order ``n``, a 1-D Gauss-Hermite tensor grid has ``n`` nodes;
    for ``d`` dims, it has ``n**d`` nodes.
    """
    grid_1d = tensor_grid_gauss_hermite(order=4, num_dims=1)
    assert grid_1d.nodes.shape == (4, 1)
    assert grid_1d.weights.shape == (4,)
    assert jnp.allclose(jnp.sum(grid_1d.weights), 1.0, atol=1e-6)

    grid_2d = tensor_grid_gauss_hermite(order=3, num_dims=2)
    assert grid_2d.nodes.shape == (9, 2)
    assert jnp.allclose(jnp.sum(grid_2d.weights), 1.0, atol=1e-6)


def test_stochastic_galerkin_surrogate_traces_under_jit() -> None:
    """Pattern (B) ``StochasticGalerkinSurrogate`` round-trips a ``jit``."""
    basis = PolynomialChaosBasis(family="hermite", order=3, coefficients=jnp.zeros((4,)))
    rngs = {"sg_samples": jax.random.PRNGKey(0)}
    surrogate = fit_galerkin_surrogate(
        model=_smooth_target, basis=basis, num_samples=256, rngs=rngs
    )

    @jax.jit
    def evaluate_jit(s: StochasticGalerkinSurrogate, x: jax.Array) -> jax.Array:
        return s.evaluate(x)

    grid = jnp.linspace(-1.5, 1.5, 9)[:, None]
    out = evaluate_jit(surrogate, grid)
    direct = surrogate.evaluate(grid)
    assert jnp.allclose(out, direct)


def test_stochastic_collocation_surrogate_traces_under_jit() -> None:
    """``StochasticCollocationSurrogate.evaluate`` must compile under ``jit``."""
    grid = smolyak_sparse_grid(level=2, num_dims=1, family="hermite")
    surrogate = fit_collocation_surrogate(model=_smooth_target, sparse_grid=grid, rngs={})

    @jax.jit
    def evaluate_jit(s: StochasticCollocationSurrogate, x: jax.Array) -> jax.Array:
        return s.evaluate(x)

    test_grid = jnp.linspace(-1.5, 1.5, 9)[:, None]
    out = evaluate_jit(surrogate, test_grid)
    direct = surrogate.evaluate(test_grid)
    assert jnp.allclose(out, direct)
