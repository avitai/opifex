"""Tests for shared Gaussian (z-score) normalization (`opifex.core.normalization`).

TDD: defines `GaussianNormalizer`, the single tested helper that replaces the
per-field z-score normalization hand-rolled across the neural-operator examples.
Statistics are fit on the training field and applied to train/test, and
predictions are mapped back with ``denormalize`` — input and output fields get
their own normalizer (they have different scales).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.normalization import GaussianNormalizer


def test_normalizer_is_jit_grad_vmap_compatible() -> None:
    """As a flax.struct pytree it must pass through jit/grad/vmap (used in pipelines)."""
    x = jnp.arange(24.0).reshape(6, 4)
    norm = GaussianNormalizer.fit(x)
    # the normalizer is a pytree argument to a jitted function
    assert jnp.isfinite(jax.jit(lambda n, v: n.normalize(v).sum())(norm, x))
    grad = jax.grad(lambda v: norm.normalize(v).sum())(x)  # differentiable
    assert jnp.all(jnp.isfinite(grad))
    assert jax.vmap(norm.normalize)(x).shape == x.shape


def test_fit_normalize_gives_zero_mean_unit_std() -> None:
    """Normalizing the fitted data yields ~zero mean and ~unit std."""
    x = jnp.linspace(-3.0, 5.0, 200).reshape(40, 5)
    norm = GaussianNormalizer.fit(x)
    z = norm.normalize(x)
    assert float(jnp.mean(z)) == pytest.approx(0.0, abs=1e-5)
    assert float(jnp.std(z)) == pytest.approx(1.0, rel=1e-4)


def test_normalize_matches_z_score_formula() -> None:
    """normalize(x) == (x - mean) / std."""
    x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    norm = GaussianNormalizer.fit(x)
    expected = (x - jnp.mean(x)) / jnp.std(x)
    assert jnp.allclose(norm.normalize(x), expected, atol=1e-5)


def test_denormalize_is_inverse_of_normalize() -> None:
    """denormalize round-trips normalize back to the original values."""
    x = jnp.arange(24.0).reshape(6, 4)
    norm = GaussianNormalizer.fit(x)
    assert jnp.allclose(norm.denormalize(norm.normalize(x)), x, atol=1e-4)


def test_fit_on_train_applies_to_test() -> None:
    """Stats are fit once (on train) and reused for test data."""
    x_train = jnp.array([[0.0, 10.0], [20.0, 30.0]])
    x_test = jnp.array([[40.0, 50.0]])
    norm = GaussianNormalizer.fit(x_train)
    # test uses the TRAIN mean/std, so its normalized values exceed unit scale
    z_test = norm.normalize(x_test)
    assert jnp.allclose(z_test, (x_test - norm.mean) / norm.std, atol=1e-5)


def test_constant_field_does_not_divide_by_zero() -> None:
    """A zero-variance field is regularised, not NaN/Inf."""
    x = jnp.full((4, 3), 7.0)
    norm = GaussianNormalizer.fit(x)
    assert jnp.all(jnp.isfinite(norm.normalize(x)))


def test_is_a_frozen_pytree_dataclass() -> None:
    """The normalizer is an immutable container of scalar stats."""
    norm = GaussianNormalizer.fit(jnp.arange(10.0))
    with pytest.raises((AttributeError, TypeError)):
        norm.mean = 1.0  # type: ignore[misc]
