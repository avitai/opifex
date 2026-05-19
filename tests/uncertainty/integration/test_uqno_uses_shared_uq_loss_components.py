"""Pin ``UQNO`` integration against the shared UQ surface.

The Phase 3 migration replaced UQNO's local Bayesian layer copies + the
``predict_with_uncertainty`` dict + the manual KL assembly with the
shared platform surface:

* :class:`opifex.uncertainty.layers.bayesian.BayesianLinear` and
  :class:`~opifex.uncertainty.layers.bayesian.BayesianSpectralConvolution`
  replace UQNO's local copies;
* :class:`~opifex.uncertainty.types.PredictiveDistribution` is the
  return of :meth:`predict_distribution`;
* :class:`~opifex.uncertainty.objectives.UQLossComponents` /
  :class:`~opifex.uncertainty.objectives.ObjectiveConfig` drive
  ``loss_components`` and ``negative_elbo``; and
* every stochastic UQNO method takes caller-owned ``nnx.Rngs`` at the
  boundary — no hidden ``nnx.Rngs(0)`` fallback, no sample counters.

This file pins those four contracts and the JAX/NNX transform
compatibility that the matching example
(``examples/uncertainty/uqno_darcy.py``) exercises.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jax import random

from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)
from opifex.uncertainty.layers.bayesian import (
    BayesianLinear as SharedBayesianLinear,
    BayesianSpectralConvolution as SharedBayesianSpectralConvolution,
)
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution


def _make_uqno() -> UncertaintyQuantificationNeuralOperator:
    return UncertaintyQuantificationNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=8,
        modes=(2, 2),
        num_layers=1,
        rngs=nnx.Rngs(0),
    )


def _make_objective() -> ObjectiveConfig:
    return ObjectiveConfig(
        kl_weight=1e-4,
        dataset_size=None,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=1.0,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )


def test_uqno_imports_shared_bayesian_layer_classes() -> None:
    """UQNO's `BayesianLinear` / `BayesianSpectralConvolution` are the shared ones."""
    from opifex.neural.operators.specialized import uqno as uqno_module

    assert uqno_module.BayesianLinear is SharedBayesianLinear
    assert uqno_module.BayesianSpectralConvolution is SharedBayesianSpectralConvolution


def test_uqno_loss_components_is_uq_loss_components() -> None:
    """`loss_components` emits a `UQLossComponents` with data + aggregated KL."""
    model = _make_uqno()
    key = random.PRNGKey(0)
    k_x, k_y = random.split(key)
    batch = {
        "x": random.normal(k_x, (2, 4, 4, 1)),
        "y": random.normal(k_y, (2, 4, 4, 1)),
    }
    components = model.loss_components(batch, rngs=nnx.Rngs(sample=7), objective=_make_objective())

    assert isinstance(components, UQLossComponents)
    assert components.data is not None
    assert components.kl is not None
    assert float(components.kl) == jax.numpy.asarray(model.kl_divergence()).item()


def test_uqno_negative_elbo_is_jit_and_grad_compatible() -> None:
    """`negative_elbo` composes with `nnx.jit` + `nnx.value_and_grad`.

    Mirrors the canonical Flax NNX pattern: ``rngs`` is passed as a
    traced argument, not closed over (closure would raise
    ``TraceContextError`` when the inner RngStream is advanced across
    trace levels).
    """
    model = _make_uqno()
    objective = _make_objective()
    key = random.PRNGKey(0)
    k_x, k_y = random.split(key)
    batch = {
        "x": random.normal(k_x, (2, 4, 4, 1)),
        "y": random.normal(k_y, (2, 4, 4, 1)),
    }

    @nnx.jit
    def loss_step(m: UncertaintyQuantificationNeuralOperator, rngs: nnx.Rngs) -> jax.Array:
        def loss_fn(m: UncertaintyQuantificationNeuralOperator, rngs: nnx.Rngs) -> jax.Array:
            return m.negative_elbo(batch, rngs=rngs, objective=objective).total

        loss, _ = nnx.value_and_grad(loss_fn)(m, rngs)
        return loss

    total = loss_step(model, nnx.Rngs(sample=11))
    assert isinstance(total, jax.Array)
    assert total.shape == ()
    assert bool(jnp.all(jnp.isfinite(total)))


def test_uqno_predict_distribution_returns_predictive_distribution_with_spatial_axes() -> None:
    """`predict_distribution` returns a `PredictiveDistribution` with spatial metadata."""
    model = _make_uqno()
    x = jnp.ones((1, 4, 4, 1))
    dist = model.predict_distribution(x, rngs=nnx.Rngs(sample=3), num_samples=4)

    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (1, 4, 4, 1)
    assert dist.epistemic is not None
    assert tuple(dist.metadata_dict()["spatial_axes"]) == (1, 2)


def test_uqno_predict_distribution_epistemic_is_nonnegative_for_meaningful_n_samples() -> None:
    """Signal captured from the `uqno_darcy` example: epistemic variance ≥ 0.

    Pre-migration UQNO conflated per-layer weight-posterior std (mislabeled
    "aleatoric") with the MC posterior epistemic estimate; the migration
    derives epistemic from MC sample variance, which is non-negative by
    construction. This is a lightweight pin so that contract regression
    is caught even when the larger example tutorial does not run in CI.
    """
    model = _make_uqno()
    x = jnp.ones((2, 4, 4, 1))
    dist = model.predict_distribution(x, rngs=nnx.Rngs(sample=5), num_samples=8)

    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic >= 0.0))
