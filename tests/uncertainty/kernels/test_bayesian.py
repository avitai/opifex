"""Canonical Bayesian-kernel KL helper tests.

These tests pin two binding rules:

1. ``diagonal_gaussian_kl`` MUST delegate to Artifex
   ``gaussian_kl_divergence`` for the N(0,1) prior; the Opifex helper only adds
   the parametric ``(prior_mean, prior_std)`` correction otherwise.
2. The kernel module is pure JAX — no ``flax.nnx`` imports
   (enforced by the container-pattern boundary audit).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.losses.divergence import (
    gaussian_kl_divergence as artifex_gaussian_kl,
)

from opifex.uncertainty.kernels.bayesian import (
    diagonal_gaussian_kl,
    sample_diagonal_gaussian,
)


def test_diagonal_gaussian_kl_delegates_to_artifex_for_standard_normal_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For prior_mean=0, prior_std=1, the helper MUST delegate to Artifex."""
    calls: list[int] = []

    def spy(
        mean: jax.Array,
        logvar: jax.Array,
        reduction: str = "mean",
        weights: jax.Array | None = None,
        axis: int | tuple[int, ...] | None = None,
    ) -> jax.Array:
        calls.append(mean.size)
        return artifex_gaussian_kl(mean, logvar, reduction, weights, axis)

    monkeypatch.setattr("opifex.uncertainty.kernels.bayesian.artifex_gaussian_kl_divergence", spy)
    mean = jnp.array([[0.5, -0.2]])
    logvar = jnp.array([[0.0, 0.0]])
    result = diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=1.0)
    assert calls, "Artifex gaussian_kl_divergence must be invoked for N(0,1) prior"
    assert jnp.isfinite(result)


def test_diagonal_gaussian_kl_matches_hand_computed_value_for_standard_normal_prior() -> None:
    """KL(N(1,1) || N(0,1)) = 0.5."""
    mean = jnp.array([[1.0]])
    logvar = jnp.array([[0.0]])
    result = diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=1.0)
    assert float(result) == pytest.approx(0.5, rel=1e-5)


def test_diagonal_gaussian_kl_matches_hand_computed_value_for_parametric_prior() -> None:
    """KL(N(1,1) || N(2, 9)) = log(3) + 2/18 - 0.5."""
    mean = jnp.array([[1.0]])
    logvar = jnp.array([[0.0]])
    expected = float(jnp.log(jnp.array(3.0)) + 2.0 / 18.0 - 0.5)
    result = diagonal_gaussian_kl(mean, logvar, prior_mean=2.0, prior_std=3.0)
    assert float(result) == pytest.approx(expected, rel=1e-5)


def test_diagonal_gaussian_kl_rejects_non_positive_prior_std() -> None:
    mean = jnp.array([[0.0]])
    logvar = jnp.array([[0.0]])
    with pytest.raises(ValueError, match=r"prior_std"):
        diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=0.0)
    with pytest.raises(ValueError, match=r"prior_std"):
        diagonal_gaussian_kl(mean, logvar, prior_mean=0.0, prior_std=-1.0)


def test_diagonal_gaussian_kl_is_jit_compatible() -> None:
    mean = jnp.array([[0.5, -0.2]])
    logvar = jnp.array([[0.1, -0.05]])
    jitted = jax.jit(lambda m, lv: diagonal_gaussian_kl(m, lv, prior_mean=0.0, prior_std=1.0))
    assert jnp.isfinite(jitted(mean, logvar))


def test_diagonal_gaussian_kl_is_grad_compatible() -> None:
    mean = jnp.array([[0.5, -0.2]])
    logvar = jnp.array([[0.1, -0.05]])
    grads = jax.grad(
        lambda m: jnp.sum(diagonal_gaussian_kl(m, logvar, prior_mean=0.0, prior_std=1.0))
    )(mean)
    assert grads.shape == mean.shape
    assert jnp.all(jnp.isfinite(grads))


def test_sample_diagonal_gaussian_is_deterministic_for_fixed_key() -> None:
    mean = jnp.zeros((4, 3))
    logvar = jnp.zeros((4, 3))
    key = jax.random.PRNGKey(0)
    a = sample_diagonal_gaussian(mean, logvar, key)
    b = sample_diagonal_gaussian(mean, logvar, key)
    assert jnp.array_equal(a, b)


def test_sample_diagonal_gaussian_changes_with_different_keys() -> None:
    mean = jnp.zeros((4, 3))
    logvar = jnp.zeros((4, 3))
    a = sample_diagonal_gaussian(mean, logvar, jax.random.PRNGKey(0))
    b = sample_diagonal_gaussian(mean, logvar, jax.random.PRNGKey(1))
    assert not jnp.array_equal(a, b)
