"""Tests for the Morris elementary-effects screening (Task 6.4).

Reference: Morris, M. D. (1991), "Factorial sampling plans for
preliminary computational experiments", Technometrics 33(2), pp.
161–174; with the improved-trajectory recommendation from Campolongo,
F., Cariboni, J., Saltelli, A. (2007), Environmental Modelling &
Software 22, pp. 1509–1518.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.sensitivity import morris_screening, MorrisResult


def _influential_vs_constant_model(x: jax.Array) -> jax.Array:
    """``f(x) = 10 * x0 + x1`` where ``x2`` is the constant dimension.

    Morris should rank ``mu_star[0] > mu_star[1] > mu_star[2]``.
    """
    return 10.0 * x[..., 0] + x[..., 1] + 0.0 * x[..., 2]


def test_morris_ranks_influential_dimension_above_constant() -> None:
    """Plan exit criterion 3: ``mu_star`` ranks influential > constant."""
    result = morris_screening(
        _influential_vs_constant_model,
        num_trajectories=32,
        num_levels=6,
        lower=jnp.zeros(3),
        upper=jnp.ones(3),
        rng_key=jax.random.PRNGKey(0),
    )

    assert isinstance(result, MorrisResult)
    assert result.mu_star.shape == (3,)
    assert result.mu_star[0] > result.mu_star[1]
    assert result.mu_star[1] > result.mu_star[2]
    assert jnp.allclose(result.mu_star[2], 0.0, atol=1e-6)


def test_morris_mu_star_is_nonnegative_and_finite() -> None:
    """``mu_star`` is the mean of absolute elementary effects."""
    result = morris_screening(
        _influential_vs_constant_model,
        num_trajectories=16,
        num_levels=4,
        lower=jnp.zeros(3),
        upper=jnp.ones(3),
        rng_key=jax.random.PRNGKey(1),
    )
    assert jnp.all(jnp.isfinite(result.mu_star))
    assert jnp.all(result.mu_star >= 0.0)
    assert jnp.all(result.sigma >= 0.0)


def test_morris_invalid_sample_shape_raises_value_error() -> None:
    """Plan exit criterion 4: invalid shapes raise ``ValueError``."""
    with pytest.raises(ValueError, match="lower.*upper"):
        morris_screening(
            _influential_vs_constant_model,
            num_trajectories=4,
            num_levels=4,
            lower=jnp.zeros(3),
            upper=jnp.ones(4),  # mismatched dim
            rng_key=jax.random.PRNGKey(0),
        )

    with pytest.raises(ValueError, match="num_trajectories"):
        morris_screening(
            _influential_vs_constant_model,
            num_trajectories=0,
            num_levels=4,
            lower=jnp.zeros(3),
            upper=jnp.ones(3),
            rng_key=jax.random.PRNGKey(0),
        )

    with pytest.raises(ValueError, match="num_levels"):
        morris_screening(
            _influential_vs_constant_model,
            num_trajectories=4,
            num_levels=1,  # must be ≥ 2
            lower=jnp.zeros(3),
            upper=jnp.ones(3),
            rng_key=jax.random.PRNGKey(0),
        )


def test_morris_jit_compatible() -> None:
    """JAX-transform compatibility — required by Task 6.4 exit criterion."""
    lower = jnp.zeros(3)
    upper = jnp.ones(3)

    def mu_star_only(key: jax.Array) -> jax.Array:
        return morris_screening(
            _influential_vs_constant_model,
            num_trajectories=8,
            num_levels=4,
            lower=lower,
            upper=upper,
            rng_key=key,
        ).mu_star

    rng_key = jax.random.PRNGKey(2)
    jit_result = jax.jit(mu_star_only)(rng_key)
    eager_result = mu_star_only(rng_key)
    assert jnp.allclose(jit_result, eager_result)
