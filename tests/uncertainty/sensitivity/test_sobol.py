"""Tests for the Sobol global-sensitivity utilities (Task 6.4).

The estimator follows the Saltelli (2002) pick-freeze scheme as the
canonical reference; sample-matrix construction and the per-index
formulas match the description in Saltelli, A. (2002), "Making best
use of model evaluations to compute sensitivity indices",
Computer Physics Communications 145, pp. 280–297.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.sensitivity import sobol_indices, SobolResult


def _linear_model(x: jax.Array) -> jax.Array:
    """``f(x1, x2) = x1 + 2 * x2`` — analytic Sobol ground truth.

    With independent uniform inputs on ``[0, 1]``, the variance budget
    is ``V[Y] = 1/12 + 4/12 = 5/12``, so the closed-form first-order
    indices are ``S1 = 1/5`` and ``S2 = 4/5``.
    """
    return x[..., 0] + 2.0 * x[..., 1]


def test_sobol_first_order_ranks_x2_above_x1_on_linear_model() -> None:
    """Plan exit criterion 1: ``f = x1 + 2 x2`` must rank x2 > x1."""
    rng_key = jax.random.PRNGKey(0)
    result = sobol_indices(
        _linear_model,
        num_samples=4096,
        lower=jnp.zeros(2),
        upper=jnp.ones(2),
        rng_key=rng_key,
    )

    assert isinstance(result, SobolResult)
    assert result.first_order.shape == (2,)
    assert result.first_order[1] > result.first_order[0]
    # Loose tolerance because Monte Carlo: theoretical S1=1/5, S2=4/5.
    assert jnp.allclose(result.first_order, jnp.array([0.2, 0.8]), atol=0.1)


def test_sobol_total_order_is_nonnegative_and_finite() -> None:
    """Plan exit criterion 2: total-order indices ≥ 0 and finite."""
    rng_key = jax.random.PRNGKey(1)
    result = sobol_indices(
        _linear_model,
        num_samples=2048,
        lower=jnp.zeros(2),
        upper=jnp.ones(2),
        rng_key=rng_key,
    )
    assert result.total_order.shape == (2,)
    # The clipped Jansen estimator guarantees non-negativity at the API
    # boundary; we also verify finiteness.
    assert jnp.all(jnp.isfinite(result.total_order))
    assert jnp.all(result.total_order >= 0.0)


def test_sobol_first_order_sums_to_one_for_additive_model() -> None:
    """Additive ``f = x1 + 2 x2`` has no interactions, so ``ΣSi ≈ 1``."""
    rng_key = jax.random.PRNGKey(2)
    result = sobol_indices(
        _linear_model,
        num_samples=4096,
        lower=jnp.zeros(2),
        upper=jnp.ones(2),
        rng_key=rng_key,
    )
    assert jnp.allclose(jnp.sum(result.first_order), 1.0, atol=0.15)


def test_sobol_invalid_sample_shape_raises_value_error() -> None:
    """Plan exit criterion 4: invalid shapes must raise ``ValueError``."""
    with pytest.raises(ValueError, match=r"lower.*upper"):
        sobol_indices(
            _linear_model,
            num_samples=64,
            lower=jnp.zeros(2),
            upper=jnp.ones(3),  # mismatched dim
            rng_key=jax.random.PRNGKey(0),
        )

    with pytest.raises(ValueError, match="num_samples"):
        sobol_indices(
            _linear_model,
            num_samples=0,
            lower=jnp.zeros(2),
            upper=jnp.ones(2),
            rng_key=jax.random.PRNGKey(0),
        )


def test_sobol_indices_jit_compatible() -> None:
    """JAX-transform compatibility — required by Task 6.4 exit criterion."""
    rng_key = jax.random.PRNGKey(3)
    lower = jnp.zeros(2)
    upper = jnp.ones(2)

    def first_order_only(key: jax.Array) -> jax.Array:
        return sobol_indices(
            _linear_model,
            num_samples=512,
            lower=lower,
            upper=upper,
            rng_key=key,
        ).first_order

    jit_result = jax.jit(first_order_only)(rng_key)
    eager_result = first_order_only(rng_key)
    assert jnp.allclose(jit_result, eager_result)
