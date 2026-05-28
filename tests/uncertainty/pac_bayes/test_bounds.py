"""Tests for pure-JAX PAC-Bayes bound formulas.

The bound functions in :mod:`opifex.uncertainty.pac_bayes.bounds` are pure
JAX kernels: they must (1) be differentiable, (2) compose with
``jax.jit`` / ``jax.grad`` / ``jax.vmap``, and (3) satisfy two universal
monotonicity properties documented by Alquier (2024, §3) — the population-risk
bound is non-decreasing in the empirical risk and non-increasing in the
sample size, for any fixed prior and confidence ``delta``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.pac_bayes.bounds import (
    catoni_bound,
    kl_bernoulli,
    mcallester_bound,
    quadratic_bound,
)


_BOUNDS = (mcallester_bound, catoni_bound, quadratic_bound)


# ---- ValueError gating ------------------------------------------------------


@pytest.mark.parametrize("bound_fn", _BOUNDS)
@pytest.mark.parametrize("bad_delta", [0.0, -0.1, 1.0, 1.5])
def test_bound_rejects_delta_outside_open_unit_interval(bound_fn, bad_delta: float) -> None:
    with pytest.raises(ValueError, match=r"delta"):
        bound_fn(jnp.asarray(0.1), jnp.asarray(1.0), 100, bad_delta)


def test_catoni_rejects_non_positive_beta() -> None:
    with pytest.raises(ValueError, match=r"beta"):
        catoni_bound(jnp.asarray(0.1), jnp.asarray(1.0), 100, 0.05, beta=0.0)


# ---- finiteness on synthetic small inputs -----------------------------------


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_returns_finite_scalar_on_small_synthetic_inputs(bound_fn) -> None:
    value = bound_fn(jnp.asarray(0.05), jnp.asarray(2.0), 64, 0.05)
    assert value.shape == ()
    assert bool(jnp.isfinite(value))


# ---- monotonicity properties ------------------------------------------------


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_is_non_decreasing_in_empirical_risk(bound_fn) -> None:
    lower = bound_fn(jnp.asarray(0.05), jnp.asarray(2.0), 200, 0.05)
    higher = bound_fn(jnp.asarray(0.20), jnp.asarray(2.0), 200, 0.05)
    assert float(higher) >= float(lower)


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_is_non_increasing_in_dataset_size(bound_fn) -> None:
    small_n = bound_fn(jnp.asarray(0.05), jnp.asarray(2.0), 50, 0.05)
    large_n = bound_fn(jnp.asarray(0.05), jnp.asarray(2.0), 5000, 0.05)
    assert float(large_n) <= float(small_n)


# ---- canonical-formula spot checks ------------------------------------------


def test_mcallester_matches_canonical_formula() -> None:
    """``R + sqrt((KL + log(2 sqrt(n)/delta)) / (2 n))``."""
    risk = 0.1
    kl = 2.5
    n = 1000
    delta = 0.05
    expected = risk + jnp.sqrt((kl + jnp.log(2.0 * jnp.sqrt(n) / delta)) / (2.0 * n))
    result = mcallester_bound(jnp.asarray(risk), jnp.asarray(kl), n, delta)
    assert float(result) == pytest.approx(float(expected), rel=1e-5)


def test_catoni_matches_canonical_formula() -> None:
    """``(beta R + (KL + log(1/delta))/n) / (1 - exp(-beta))``."""
    risk = 0.1
    kl = 2.5
    n = 1000
    delta = 0.05
    beta = 1.0
    expected = (beta * risk + (kl + jnp.log(1.0 / delta)) / n) / (1.0 - jnp.exp(-beta))
    result = catoni_bound(jnp.asarray(risk), jnp.asarray(kl), n, delta, beta=beta)
    assert float(result) == pytest.approx(float(expected), rel=1e-5)


def test_kl_bernoulli_is_zero_when_arguments_match() -> None:
    """``kl_bernoulli(p, p) == 0`` for any ``p in (0, 1)``."""
    assert float(kl_bernoulli(jnp.asarray(0.3), jnp.asarray(0.3))) == pytest.approx(0.0, abs=1e-6)
    assert float(kl_bernoulli(jnp.asarray(0.5), jnp.asarray(0.5))) == pytest.approx(0.0, abs=1e-6)


def test_quadratic_bound_satisfies_kl_inversion_identity() -> None:
    """The quadratic bound must solve ``kl_bernoulli(R, B) ≈ (KL + log(2 sqrt(n)/delta)) / n``."""
    risk = jnp.asarray(0.05)
    kl = jnp.asarray(2.0)
    n = 500
    delta = 0.05
    bound = quadratic_bound(risk, kl, n, delta)
    rhs = (float(kl) + float(jnp.log(2.0 * jnp.sqrt(n) / delta))) / n
    lhs = float(kl_bernoulli(risk, bound))
    assert lhs == pytest.approx(rhs, abs=1e-4)


# ---- transform compatibility ------------------------------------------------


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_is_jit_compatible(bound_fn) -> None:
    jitted = jax.jit(lambda r, k: bound_fn(r, k, 256, 0.05))
    value = jitted(jnp.asarray(0.05), jnp.asarray(1.5))
    assert bool(jnp.isfinite(value))


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_is_grad_compatible(bound_fn) -> None:
    """``d/dR`` of every bound must be finite and non-negative around ``R = 0.05``."""

    def f(risk: jax.Array) -> jax.Array:
        return bound_fn(risk, jnp.asarray(1.5), 256, 0.05)

    g = jax.grad(f)(jnp.asarray(0.05))
    assert bool(jnp.isfinite(g))
    assert float(g) >= 0.0


@pytest.mark.parametrize("bound_fn", _BOUNDS)
def test_bound_is_vmap_compatible(bound_fn) -> None:
    """Vectorising over a batch of empirical risks yields the same shape."""
    risks = jnp.linspace(0.01, 0.5, 5)

    def f(risk: jax.Array) -> jax.Array:
        return bound_fn(risk, jnp.asarray(1.5), 256, 0.05)

    values = jax.vmap(f)(risks)
    assert values.shape == (5,)
    assert bool(jnp.all(jnp.isfinite(values)))
