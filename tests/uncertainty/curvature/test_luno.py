"""Tests for Linearized Neural Operator (LUNO) predictive posterior.

LUNO (Magnani et al. 2024, arXiv:2406.04317) treats a trained neural
operator as a *linearised* function-valued Gaussian Process. Given a
trained model ``f(x; θ)`` with MAP estimate ``θ*`` and diagonal Laplace
posterior ``θ | data ~ N(θ*, Σ)`` (with ``Σ = diag(1 / precision)``),
the first-order Taylor expansion around ``θ*`` yields the predictive
moments

    μ(x) = f(x; θ*),
    Σ_pred(x, x') = J_θ f(x; θ*) · Σ · J_θ f(x'; θ*)^T,

where ``J_θ`` is the Jacobian of the network output with respect to the
parameter vector. For a linear model ``f(θ, x) = x · θ`` this collapses
to the closed-form ``Var(x) = Σ_i x_i² / precision_i``, which makes
calibration testing trivial against a known analytic posterior.

Canonical reference:
* tinygp ``Transform(Kernel)`` pattern at
  ``../tinygp/src/tinygp/transforms.py:23`` shows the linearised-kernel
  shape; opifex implements the JAX-native version because tinygp is an
  adapter-only optional backend, not a runtime dependency.

References
----------
* Magnani, E. et al. 2024 — *Linearised neural operators for function
  uncertainty quantification*, arXiv:2406.04317 (PRIMARY).
* Daxberger, E. et al. 2021 — *Laplace Redux — Effortless Bayesian Deep
  Learning*, arXiv:2106.14806 (parameter-space Laplace posterior).
* MacKay, D. J. C. 1992 — *A practical Bayesian framework for
  backpropagation networks*, Neural Computation 4(3).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.curvature import (
    DiagonalLaplacePosterior,
    linearized_neural_operator_posterior,
)
from opifex.uncertainty.types import PredictiveDistribution


def _linear_model(parameters: jax.Array, x: jax.Array) -> jax.Array:
    """``f(θ, x) = x · θ`` — Jacobian wrt θ is ``x`` exactly."""
    return x @ parameters


def test_predictive_mean_equals_model_at_map_estimate() -> None:
    """For a linearised model, the predictive mean is the deterministic forward."""
    map_estimate = jnp.asarray([1.0, -2.0, 0.5])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.asarray([1.0, 2.0, 4.0]),
    )
    inputs = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

    predictive = linearized_neural_operator_posterior(
        model_fn=_linear_model,
        laplace_posterior=posterior,
        x=inputs,
    )
    assert isinstance(predictive, PredictiveDistribution)
    assert jnp.allclose(predictive.mean, _linear_model(map_estimate, inputs), atol=1e-6)


def test_predictive_variance_matches_closed_form_for_linear_model() -> None:
    r"""``Var(x) = \sum_i x_i^2 / precision_i`` for ``f(θ, x) = x · θ``."""
    precision = jnp.asarray([1.0, 2.0, 4.0])
    map_estimate = jnp.zeros(3)
    posterior = DiagonalLaplacePosterior(mean=map_estimate, precision_diagonal=precision)
    inputs = jax.random.normal(jax.random.PRNGKey(1), (4, 3))

    predictive = linearized_neural_operator_posterior(
        model_fn=_linear_model,
        laplace_posterior=posterior,
        x=inputs,
    )
    expected = jnp.sum(inputs**2 / precision, axis=-1)
    assert predictive.variance is not None
    assert jnp.allclose(predictive.variance, expected, atol=1e-6)


def test_predictive_collapses_to_deterministic_in_infinite_precision_limit() -> None:
    """As precision → ∞, the predictive variance → 0 and only the mean survives."""
    map_estimate = jnp.asarray([0.3, -0.4])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.full_like(map_estimate, 1e12),
    )
    inputs = jnp.asarray([[1.0, 2.0], [3.0, -1.0]])

    predictive = linearized_neural_operator_posterior(
        model_fn=_linear_model,
        laplace_posterior=posterior,
        x=inputs,
    )
    assert predictive.variance is not None
    assert jnp.all(predictive.variance < 1e-6)
    assert jnp.allclose(predictive.mean, _linear_model(map_estimate, inputs), atol=1e-6)


def test_predictive_is_jit_compatible() -> None:
    """The LUNO posterior compiles under ``jax.jit``."""
    map_estimate = jnp.asarray([1.0, 2.0])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.asarray([1.0, 1.0]),
    )
    inputs = jnp.eye(2)

    @jax.jit
    def predict(theta: jax.Array, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        post = DiagonalLaplacePosterior(mean=theta, precision_diagonal=posterior.precision_diagonal)
        result = linearized_neural_operator_posterior(
            model_fn=_linear_model, laplace_posterior=post, x=x
        )
        assert result.variance is not None
        return result.mean, result.variance

    mean, variance = predict(map_estimate, inputs)
    assert mean.shape == (2,)
    assert variance.shape == (2,)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(variance))


def test_predictive_is_vmap_compatible_across_separate_batches() -> None:
    """``jax.vmap`` over a leading batch axis of inputs should map cleanly."""
    map_estimate = jnp.asarray([0.5, -0.25])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.asarray([1.0, 2.0]),
    )
    batched_inputs = jax.random.normal(jax.random.PRNGKey(2), (3, 4, 2))

    def predict_one(x: jax.Array) -> jax.Array:
        out = linearized_neural_operator_posterior(
            model_fn=_linear_model, laplace_posterior=posterior, x=x
        )
        assert out.variance is not None
        return out.variance

    variances = jax.vmap(predict_one)(batched_inputs)
    assert variances.shape == (3, 4)
    assert jnp.all(jnp.isfinite(variances))


def test_predictive_metadata_advertises_laplace_source() -> None:
    """The metadata tuple identifies the source method as LUNO/laplace."""
    posterior = DiagonalLaplacePosterior(
        mean=jnp.zeros(2),
        precision_diagonal=jnp.ones(2),
    )
    predictive = linearized_neural_operator_posterior(
        model_fn=_linear_model,
        laplace_posterior=posterior,
        x=jnp.eye(2),
    )
    keys = [k for k, _ in predictive.metadata]
    assert "method" in keys
    assert "source_package" in keys


def test_predictive_rejects_shape_mismatched_posterior() -> None:
    """The posterior precision must match the parameter count of ``model_fn``."""
    posterior = DiagonalLaplacePosterior(
        mean=jnp.zeros(3),  # 3 parameters
        precision_diagonal=jnp.ones(3),
    )
    inputs = jax.random.normal(jax.random.PRNGKey(3), (4, 2))  # 2-feature inputs

    with pytest.raises((TypeError, ValueError)):
        linearized_neural_operator_posterior(
            model_fn=_linear_model, laplace_posterior=posterior, x=inputs
        )
