"""Diagonal Gaussian prior log-density tests.

The diagonal-Gaussian prior log density MUST share the same (mean, log-variance)
parameterization as :func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl`
so the same posterior parameters can be plugged into either helper.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.priors import (
    diagonal_gaussian_log_prior,
    PriorSpec,
)


def test_diagonal_gaussian_log_prior_matches_hand_computed_value() -> None:
    # log N(theta=0; mean=0, std=1) = -0.5 * log(2*pi)
    theta = jnp.array([0.0])
    expected = float(-0.5 * jnp.log(2 * jnp.pi))
    result = float(diagonal_gaussian_log_prior(theta, prior_mean=0.0, prior_std=1.0))
    assert result == pytest.approx(expected, rel=1e-5)


def test_diagonal_gaussian_log_prior_sums_over_features() -> None:
    theta = jnp.zeros((4,))
    expected = float(4 * -0.5 * jnp.log(2 * jnp.pi))
    result = float(diagonal_gaussian_log_prior(theta, prior_mean=0.0, prior_std=1.0))
    assert result == pytest.approx(expected, rel=1e-5)


def test_diagonal_gaussian_log_prior_rejects_non_positive_std() -> None:
    theta = jnp.zeros(1)
    with pytest.raises(ValueError, match=r"prior_std"):
        diagonal_gaussian_log_prior(theta, prior_mean=0.0, prior_std=0.0)
    with pytest.raises(ValueError, match=r"prior_std"):
        diagonal_gaussian_log_prior(theta, prior_mean=0.0, prior_std=-1.0)


def test_diagonal_gaussian_log_prior_is_jit_and_grad_compatible() -> None:
    theta = jnp.array([0.5, -0.3])
    jitted = jax.jit(lambda t: diagonal_gaussian_log_prior(t, prior_mean=0.0, prior_std=1.0))
    assert jnp.isfinite(jitted(theta))
    grads = jax.grad(jitted)(theta)
    assert grads.shape == theta.shape


def test_prior_spec_is_pattern_a_frozen_dataclass() -> None:
    spec = PriorSpec(
        name="diagonal_gaussian",
        family="continuous",
        parameter_names=("prior_mean", "prior_std"),
    )
    assert dataclasses.is_dataclass(PriorSpec)
    assert hasattr(PriorSpec, "__slots__")
    assert not hasattr(spec, "__dict__")
    assert hash(spec) is not None
