"""Tests for the diagonal Laplace posterior approximation.

A diagonal Laplace approximation at a MAP point ``θ*`` is
``θ | data ~ N(θ*, diag(prior_precision + observation_precision)⁻¹)``,
where ``observation_precision`` is the empirical Fisher diagonal at
``θ*``. This is the canonical post-hoc UQ wrapper for a deterministic
model (Daxberger 2021).

Canonical reference:
* ``../bayesian-torch`` and Daxberger's Laplace package use the same
  diagonal-precision formula; opifex computes
  ``observation_precision`` via :func:`empirical_fisher_diagonal` and
  adds a scalar prior precision.

References
----------
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806.
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.curvature import diagonal_laplace_posterior


def test_diagonal_laplace_posterior_precision_decomposes_into_prior_and_fisher() -> None:
    """``precision = prior_precision + empirical_fisher_diagonal``."""
    rng = jax.random.PRNGKey(0)
    key_x, key_t, key_p = jax.random.split(rng, 3)
    inputs = jax.random.normal(key_x, (4, 2))
    targets = jax.random.normal(key_t, (4,))
    map_estimate = jax.random.normal(key_p, (2,))
    prior_precision = 0.5

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    posterior = diagonal_laplace_posterior(
        per_sample_loss=per_sample_loss,
        map_estimate=map_estimate,
        inputs=inputs,
        targets=targets,
        prior_precision=prior_precision,
    )
    per_sample_grads = jax.vmap(jax.grad(per_sample_loss), in_axes=(None, 0, 0))(
        map_estimate, inputs, targets
    )
    expected_precision = prior_precision + jnp.mean(per_sample_grads**2, axis=0)
    assert jnp.allclose(posterior.precision_diagonal, expected_precision, atol=1e-6)
    assert jnp.allclose(posterior.mean, map_estimate)


def test_diagonal_laplace_posterior_variance_is_positive() -> None:
    """The posterior diagonal variance ``1 / precision`` is strictly positive."""
    rng = jax.random.PRNGKey(1)
    inputs = jax.random.normal(rng, (8, 3))
    targets = jnp.zeros(8)
    map_estimate = jnp.zeros(3)

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    posterior = diagonal_laplace_posterior(
        per_sample_loss=per_sample_loss,
        map_estimate=map_estimate,
        inputs=inputs,
        targets=targets,
        prior_precision=1.0,
    )
    variance = 1.0 / posterior.precision_diagonal
    assert jnp.all(variance > 0.0)


def test_diagonal_laplace_posterior_collapses_to_prior_with_no_data_likelihood() -> None:
    """At a zero-residual minimiser, the posterior equals the prior."""
    inputs = jnp.eye(2)
    map_estimate = jnp.zeros(2)
    targets = jnp.zeros(2)  # zero-residual
    prior_precision = 2.0

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    posterior = diagonal_laplace_posterior(
        per_sample_loss=per_sample_loss,
        map_estimate=map_estimate,
        inputs=inputs,
        targets=targets,
        prior_precision=prior_precision,
    )
    assert jnp.allclose(
        posterior.precision_diagonal, jnp.full_like(map_estimate, prior_precision), atol=1e-6
    )


def test_diagonal_laplace_posterior_is_jit_compatible() -> None:
    """The full pipeline compiles under ``jax.jit``."""
    inputs = jnp.eye(3)
    targets = jnp.zeros(3)
    map_estimate = jnp.asarray([1.0, 2.0, 3.0])

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    @jax.jit
    def fit(theta: jax.Array) -> jax.Array:
        posterior = diagonal_laplace_posterior(
            per_sample_loss=per_sample_loss,
            map_estimate=theta,
            inputs=inputs,
            targets=targets,
            prior_precision=1.0,
        )
        return posterior.precision_diagonal

    precision = fit(map_estimate)
    assert precision.shape == map_estimate.shape
    assert jnp.all(jnp.isfinite(precision))


def test_diagonal_laplace_posterior_rejects_nonpositive_prior_precision() -> None:
    """Prior precision must be strictly positive — zero is invalid."""
    import pytest

    def per_sample_loss(theta: jax.Array, x: jax.Array, t: jax.Array) -> jax.Array:
        return 0.5 * (x @ theta - t) ** 2

    with pytest.raises(ValueError, match="prior_precision must be positive"):
        diagonal_laplace_posterior(
            per_sample_loss=per_sample_loss,
            map_estimate=jnp.zeros(2),
            inputs=jnp.eye(2),
            targets=jnp.zeros(2),
            prior_precision=0.0,
        )
