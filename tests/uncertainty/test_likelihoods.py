"""Phase 1 Task 1.6 — likelihood log-density tests.

Backend-neutral likelihood log-density helpers. Container patterns:

* Likelihood spec metadata (``LikelihoodSpec``) → pattern (A) per
  GUIDE_ALIGNMENT §5a.
* Combined log-density containers carrying array values → pattern (B).
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.likelihoods import (
    gaussian_log_likelihood,
    heteroscedastic_gaussian_log_likelihood,
    laplace_log_likelihood,
    LikelihoodSpec,
    mixture_log_likelihood,
    student_t_log_likelihood,
)


def test_gaussian_log_likelihood_matches_hand_computed_value() -> None:
    # log N(y=1; mu=0, sigma=1) = -0.5 * (log(2*pi) + 1)
    y = jnp.array([1.0])
    mean = jnp.array([0.0])
    scale = jnp.array([1.0])
    expected = float(-0.5 * (jnp.log(2 * jnp.pi) + 1.0))
    result = float(gaussian_log_likelihood(y, mean=mean, scale=scale)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_gaussian_log_likelihood_rejects_non_positive_scale() -> None:
    y = jnp.zeros(1)
    mean = jnp.zeros(1)
    with pytest.raises(ValueError, match=r"scale"):
        gaussian_log_likelihood(y, mean=mean, scale=jnp.array([0.0]))
    with pytest.raises(ValueError, match=r"scale"):
        gaussian_log_likelihood(y, mean=mean, scale=jnp.array([-1.0]))


def test_heteroscedastic_gaussian_log_likelihood_allows_per_point_scale() -> None:
    y = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 2.5])
    scale = jnp.array([1.0, 0.5])
    result = heteroscedastic_gaussian_log_likelihood(y, mean=mean, scale=scale)
    expected_0 = float(-0.5 * (jnp.log(2 * jnp.pi) + 1.0))
    expected_1 = float(-0.5 * (jnp.log(2 * jnp.pi) + jnp.log(0.25) + (2.0 - 2.5) ** 2 / 0.25))
    assert float(jnp.sum(result)) == pytest.approx(expected_0 + expected_1, rel=1e-5)


def test_laplace_log_likelihood_matches_closed_form() -> None:
    # log Laplace(y=0; mu=0, b=1) = -log(2)
    y = jnp.array([0.0])
    location = jnp.array([0.0])
    scale = jnp.array([1.0])
    expected = float(-jnp.log(jnp.array(2.0)))
    result = float(laplace_log_likelihood(y, location=location, scale=scale)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_student_t_log_likelihood_rejects_non_positive_df() -> None:
    y = jnp.zeros(1)
    location = jnp.zeros(1)
    scale = jnp.ones(1)
    with pytest.raises(ValueError, match=r"df"):
        student_t_log_likelihood(y, df=0.0, location=location, scale=scale)
    with pytest.raises(ValueError, match=r"df"):
        student_t_log_likelihood(y, df=-1.0, location=location, scale=scale)


def test_student_t_log_likelihood_approaches_gaussian_for_large_df() -> None:
    # df=1000 is "large enough" without hitting float32 gammaln precision loss
    # (df=1e6 collapses the gammaln difference to its float32 nearest representable value).
    y = jnp.array([0.3])
    location = jnp.array([0.0])
    scale = jnp.array([1.0])
    student = float(student_t_log_likelihood(y, df=1000.0, location=location, scale=scale)[0])
    gaussian = float(gaussian_log_likelihood(y, mean=location, scale=scale)[0])
    assert student == pytest.approx(gaussian, abs=2e-3)


def test_mixture_log_likelihood_rejects_invalid_weights() -> None:
    y = jnp.array([0.0])
    weights = jnp.array([0.5, 0.4])  # does not sum to 1
    means = jnp.array([0.0, 1.0])
    scales = jnp.array([1.0, 1.0])
    with pytest.raises(ValueError, match=r"weight"):
        mixture_log_likelihood(y, weights=weights, means=means, scales=scales)


def test_mixture_log_likelihood_matches_single_component_gaussian() -> None:
    y = jnp.array([0.5])
    weights = jnp.array([1.0])
    means = jnp.array([0.0])
    scales = jnp.array([1.0])
    expected = float(gaussian_log_likelihood(y, mean=means, scale=scales)[0])
    result = float(mixture_log_likelihood(y, weights=weights, means=means, scales=scales)[0])
    assert result == pytest.approx(expected, rel=1e-5)


def test_gaussian_log_likelihood_is_jit_and_grad_compatible() -> None:
    y = jnp.array([1.0, 2.0])
    mean = jnp.array([0.0, 1.5])
    scale = jnp.array([1.0, 0.5])
    jitted = jax.jit(lambda m: jnp.sum(gaussian_log_likelihood(y, mean=m, scale=scale)))
    assert jnp.isfinite(jitted(mean))
    grads = jax.grad(jitted)(mean)
    assert grads.shape == mean.shape
    assert jnp.all(jnp.isfinite(grads))


def test_likelihood_spec_is_pattern_a_frozen_dataclass() -> None:
    spec = LikelihoodSpec(
        name="gaussian",
        family="continuous",
        parameter_names=("mean", "scale"),
        supports_heteroscedastic=False,
    )
    assert dataclasses.is_dataclass(LikelihoodSpec)
    assert hasattr(LikelihoodSpec, "__slots__")
    assert not hasattr(spec, "__dict__")
    assert hash(spec) is not None
