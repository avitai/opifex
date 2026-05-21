"""Tests for the empirical Fisher information primitives.

The empirical Fisher matrix at parameters ``θ`` for a per-sample loss
``ℓ_i(θ) = L(f(θ, x_i), t_i)`` is

    F(θ) = (1/N) Σ_i ∇_θ ℓ_i(θ) ∇_θ ℓ_i(θ)^T,

i.e., the outer product of per-sample gradients. The diagonal is the
standard cheap curvature proxy used in optimisers (Adam-like) and
post-hoc Laplace approximations (Kunstner+ 2019 critique notwithstanding).

Canonical reference:
* ``../bayesian-torch`` and Daxberger Laplace package — empirical-Fisher
  formulation as a positive-semidefinite curvature estimate.
* The diagonal estimator follows from the standard ``jax.vmap`` /
  ``jax.grad`` per-sample-gradient recipe.

References
----------
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* Kunstner, F., Hennig, P., Balles, L. 2019 — *Limitations of the
  empirical Fisher approximation for natural gradient descent*,
  arXiv:1905.12558.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature import empirical_fisher_diagonal


def test_empirical_fisher_diagonal_matches_squared_sum_of_persample_gradients() -> None:
    """``F_ii = (1/N) Σ_n (∂ℓ_n/∂θ_i)²``."""
    rng = jax.random.PRNGKey(0)
    key_x, key_t, key_p = jax.random.split(rng, 3)
    inputs = jax.random.normal(key_x, (4, 2))
    targets = jax.random.normal(key_t, (4,))
    parameters = jax.random.normal(key_p, (2,))

    def model(theta: jax.Array, batch: jax.Array) -> jax.Array:
        return batch @ theta

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (model(theta, x) - t) ** 2

    diagonal = empirical_fisher_diagonal(per_sample_loss, parameters, inputs, targets)

    per_sample_grads = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))(
        parameters, inputs, targets
    )
    expected = jnp.mean(per_sample_grads**2, axis=0)
    assert jnp.allclose(diagonal, expected, atol=1e-6)


def test_empirical_fisher_diagonal_is_nonnegative() -> None:
    """The empirical Fisher diagonal is non-negative entry-wise."""
    rng = jax.random.PRNGKey(1)
    key_x, key_t, key_p = jax.random.split(rng, 3)
    inputs = jax.random.normal(key_x, (8, 3))
    targets = jax.random.normal(key_t, (8,))
    parameters = jax.random.normal(key_p, (3,))

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    diagonal = empirical_fisher_diagonal(per_sample_loss, parameters, inputs, targets)
    assert jnp.all(diagonal >= 0.0)


def test_empirical_fisher_diagonal_zero_at_minimiser() -> None:
    """At the OLS optimum, per-sample gradients have zero mean — for an
    over-determined problem, ``∂ℓ/∂θ`` is non-zero per sample but the
    diagonal still reflects the residual squared-gradient scale.
    """
    # Synthetic data where the linear model has a known minimiser.
    rng = jax.random.PRNGKey(2)
    key_x, key_w = jax.random.split(rng)
    inputs = jax.random.normal(key_x, (20, 2))
    true_params = jax.random.normal(key_w, (2,))
    targets = inputs @ true_params  # zero-residual case

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    diagonal = empirical_fisher_diagonal(per_sample_loss, true_params, inputs, targets)
    assert jnp.allclose(diagonal, jnp.zeros_like(diagonal), atol=1e-6)


def test_empirical_fisher_diagonal_is_jit_compatible() -> None:
    """``empirical_fisher_diagonal`` works under ``jax.jit``."""
    inputs = jnp.eye(3)
    targets = jnp.zeros(3)
    parameters = jnp.asarray([1.0, 2.0, 3.0])

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    @jax.jit
    def call(theta: jax.Array) -> jax.Array:
        return empirical_fisher_diagonal(per_sample_loss, theta, inputs, targets)

    diagonal = call(parameters)
    assert diagonal.shape == parameters.shape
    assert jnp.all(jnp.isfinite(diagonal))
