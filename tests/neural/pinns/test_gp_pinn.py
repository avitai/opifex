"""Tests for the GP-PINN function-valued GP predictive.

A GP-PINN treats a trained PINN as a function-valued Gaussian Process
via the *linearised Laplace* equivalence (Immer, Korzepa, Bauer 2021,
AISTATS, arXiv:2008.08400 §3 — "Improving predictions of Bayesian
neural nets via local linearisation"). Given a PINN forward
``f(x; θ)`` and a diagonal Laplace posterior ``θ ~ N(θ*, Σ)`` with
``Σ = diag(1 / precision)``, the predictive moments are

    μ(x)             = f(x; θ*),
    Var(x)           = J_θ f(x; θ*) · Σ · J_θ f(x; θ*)^T
                     = Σ_i (∂f/∂θ_i)^2 / precision_i.

The math is identical to LUNO (Task 10.1) — what differs is the
*context*: a PINN forward consumes spatial / spatio-temporal
coordinates rather than spectral fields, and the GP equivalence is
advertised through one of the Task 6.3 GP adapter specs (``GPJaxAdapterSpec``,
``TinygpAdapterSpec``, …) for capability bookkeeping.

References
----------
* Immer, A., Korzepa, M., Bauer, M. 2021 — *Improving predictions of
  Bayesian neural nets via local linearisation*, AISTATS,
  arXiv:2008.08400 (PRIMARY — NN-as-GP linearised-Laplace equivalence).
* Daxberger, E. et al. 2021 — *Laplace Redux*, arXiv:2106.14806
  (parameter-space diagonal Laplace posterior).
* Karniadakis, G. et al. 2022 — *Physics-informed machine learning*,
  Nat. Rev. Phys. (GP-PINN motivation).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.neural.pinns.gp_pinn import gp_pinn_predictive_posterior
from opifex.uncertainty.adapters.gp import TinygpAdapterSpec
from opifex.uncertainty.curvature import (
    DiagonalLaplacePosterior,
    linearized_neural_operator_posterior,
)
from opifex.uncertainty.types import PredictiveDistribution


def _linear_pinn(parameters: jax.Array, x: jax.Array) -> jax.Array:
    """Toy "PINN" ``f(θ, x) = x · θ`` — Jacobian wrt θ is ``x`` exactly."""
    return x @ parameters


def test_predictive_mean_equals_pinn_forward_at_map_estimate() -> None:
    """``μ(x) = f(x; θ*)`` exactly for the linear toy PINN."""
    map_estimate = jnp.asarray([1.0, -2.0, 0.5])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.asarray([1.0, 2.0, 4.0]),
    )
    coordinates = jax.random.normal(jax.random.PRNGKey(0), (5, 3))

    predictive = gp_pinn_predictive_posterior(
        pinn_forward=_linear_pinn,
        laplace_posterior=posterior,
        coordinates=coordinates,
        gp_adapter_spec=TinygpAdapterSpec(),
    )
    assert isinstance(predictive, PredictiveDistribution)
    assert jnp.allclose(predictive.mean, _linear_pinn(map_estimate, coordinates), atol=1e-6)


def test_predictive_variance_matches_closed_form_linearised_laplace() -> None:
    r"""``Var(x) = Σ_i x_i^2 / precision_i`` for ``f(θ, x) = x · θ``."""
    precision = jnp.asarray([1.0, 2.0, 4.0])
    posterior = DiagonalLaplacePosterior(mean=jnp.zeros(3), precision_diagonal=precision)
    coordinates = jax.random.normal(jax.random.PRNGKey(1), (4, 3))

    predictive = gp_pinn_predictive_posterior(
        pinn_forward=_linear_pinn,
        laplace_posterior=posterior,
        coordinates=coordinates,
        gp_adapter_spec=TinygpAdapterSpec(),
    )
    expected = jnp.sum(coordinates**2 / precision, axis=-1)
    assert predictive.variance is not None
    assert jnp.allclose(predictive.variance, expected, atol=1e-6)


def test_predictive_agrees_with_luno_on_the_same_inputs() -> None:
    """GP-PINN delegates to LUNO; predictive moments must match exactly."""
    map_estimate = jnp.asarray([1.0, -2.0])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.asarray([1.0, 2.0]),
    )
    coordinates = jax.random.normal(jax.random.PRNGKey(2), (3, 2))

    luno = linearized_neural_operator_posterior(
        model_fn=_linear_pinn,
        laplace_posterior=posterior,
        x=coordinates,
    )
    gp_pinn = gp_pinn_predictive_posterior(
        pinn_forward=_linear_pinn,
        laplace_posterior=posterior,
        coordinates=coordinates,
        gp_adapter_spec=TinygpAdapterSpec(),
    )
    assert luno.variance is not None
    assert gp_pinn.variance is not None
    assert jnp.allclose(gp_pinn.mean, luno.mean, atol=1e-6)
    assert jnp.allclose(gp_pinn.variance, luno.variance, atol=1e-6)


def test_predictive_metadata_advertises_gp_adapter_spec() -> None:
    """The chosen GP adapter spec's source package + family tags are recorded."""
    spec = TinygpAdapterSpec()
    posterior = DiagonalLaplacePosterior(
        mean=jnp.zeros(2),
        precision_diagonal=jnp.ones(2),
    )
    predictive = gp_pinn_predictive_posterior(
        pinn_forward=_linear_pinn,
        laplace_posterior=posterior,
        coordinates=jnp.eye(2),
        gp_adapter_spec=spec,
    )
    metadata = dict(predictive.metadata)
    assert metadata.get("method") is not None
    assert metadata.get("source_package") is not None
    assert metadata.get("gp_adapter_source_package") == spec.source_package
    assert metadata.get("estimator") == "gp_pinn_linearized_laplace"


def test_predictive_is_jit_compatible() -> None:
    """The GP-PINN predictive compiles under ``jax.jit``."""
    map_estimate = jnp.asarray([1.0, 2.0])
    posterior = DiagonalLaplacePosterior(
        mean=map_estimate,
        precision_diagonal=jnp.ones(2),
    )
    spec = TinygpAdapterSpec()
    coords = jnp.eye(2)

    @jax.jit
    def predict(theta: jax.Array) -> tuple[jax.Array, jax.Array]:
        post = DiagonalLaplacePosterior(mean=theta, precision_diagonal=posterior.precision_diagonal)
        result = gp_pinn_predictive_posterior(
            pinn_forward=_linear_pinn,
            laplace_posterior=post,
            coordinates=coords,
            gp_adapter_spec=spec,
        )
        assert result.variance is not None
        return result.mean, result.variance

    mean, variance = predict(map_estimate)
    assert jnp.all(jnp.isfinite(mean))
    assert jnp.all(jnp.isfinite(variance))


def test_rejects_non_gp_adapter_spec() -> None:
    """Non-GP adapter specs must be rejected with an actionable message."""

    class _NotAGPAdapterSpec:
        source_package = "opifex.uncertainty.adapters.model"

    posterior = DiagonalLaplacePosterior(mean=jnp.zeros(2), precision_diagonal=jnp.ones(2))
    with pytest.raises(TypeError, match="GP adapter spec"):
        gp_pinn_predictive_posterior(
            pinn_forward=_linear_pinn,
            laplace_posterior=posterior,
            coordinates=jnp.eye(2),
            gp_adapter_spec=_NotAGPAdapterSpec(),  # type: ignore[arg-type]
        )
