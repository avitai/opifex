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
import numpy as np
import pytest

from opifex.uncertainty.sensitivity import morris_screening, MorrisResult
from opifex.uncertainty.sensitivity.morris import _build_trajectory


# One float32 addition of a base value and delta, e.g. float32(1/3) + float32(2/3) = 1.0000001.
_ROUNDING = 4.0 * float(np.finfo(np.float32).eps)


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
    with pytest.raises(ValueError, match=r"lower.*upper"):
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


def _trajectories(num_levels: int, dim: int = 5, count: int = 256) -> np.ndarray:
    """Unit-cube points of ``count`` trajectories, shape ``(count, dim + 1, dim)``."""
    keys = jax.random.split(jax.random.PRNGKey(num_levels), count)
    points, _ = jax.vmap(lambda key: _build_trajectory(key, dim, num_levels))(keys)
    return np.asarray(points, dtype=np.float64)


@pytest.mark.parametrize("num_levels", [2, 4, 5, 6, 8])
def test_morris_trajectories_stay_in_the_unit_box(num_levels: int) -> None:
    """Every trajectory point lies in ``[0, 1]^d``.

    Morris (1991), p. 164: the base value ``x*`` takes values in ``{0, 1/(p-1), ..., 1 - Delta}``
    and each coordinate of the orientation ``B*`` is ``x*_i`` or ``x*_i + Delta``, so no point
    leaves the region of interest.
    """
    points = _trajectories(num_levels)
    assert points.min() >= -_ROUNDING, points.min()
    assert points.max() <= 1.0 + _ROUNDING, points.max()


@pytest.mark.parametrize("num_levels", [2, 4, 5, 6, 8])
def test_morris_coordinates_take_the_base_value_or_the_base_value_plus_delta(
    num_levels: int,
) -> None:
    """Along a trajectory each coordinate takes exactly ``x*_i`` and ``x*_i + Delta``.

    In ``B* = (J x* + (Delta/2)[(2B - J) D* + J]) P*`` (Morris 1991, p. 164) a coordinate whose sign
    in ``D*`` is ``-1`` starts at ``x*_i + Delta`` and steps down to ``x*_i``; one with ``+1``
    starts at ``x*_i`` and steps up. The lower value is a base value, at most ``1 - Delta``.
    """
    delta = num_levels / (2.0 * (num_levels - 1))
    points = _trajectories(num_levels)
    lows = points.min(axis=1)
    highs = points.max(axis=1)
    np.testing.assert_allclose(highs - lows, delta, rtol=0.0, atol=_ROUNDING)
    assert lows.min() >= -_ROUNDING, lows.min()
    assert lows.max() <= 1.0 - delta + _ROUNDING, lows.max()


def test_morris_screening_is_finite_for_a_model_defined_only_on_the_box() -> None:
    """A model undefined below zero screens to finite statistics.

    ``f(x) = sum_i sqrt(x_i)`` is increasing in every input, so every elementary effect is positive
    and ``mu_star`` equals ``mu``.
    """
    result = morris_screening(
        lambda x: jnp.sum(jnp.sqrt(x), axis=-1),
        num_trajectories=64,
        num_levels=4,
        lower=jnp.zeros(5),
        upper=jnp.ones(5),
        rng_key=jax.random.PRNGKey(7),
    )
    assert bool(jnp.all(jnp.isfinite(result.mu_star)))
    assert bool(jnp.all(jnp.isfinite(result.sigma)))
    assert bool(jnp.all(result.mu > 0.0))
    assert bool(jnp.allclose(result.mu_star, result.mu))
