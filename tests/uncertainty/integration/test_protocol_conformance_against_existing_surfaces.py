"""Protocol conformance tests for existing Opifex UQ surfaces.

Treats the structural UQ protocols as an executable specification for the
existing Bayesian / UQNO surfaces. ``xfail`` markers identify gaps where a
surface does not yet implement the protocol; when the gap closes the test
flips to ``XPASS`` and the marker should be removed in the same commit.

Tests that are NOT marked ``xfail`` MUST pass today — they pin behavior that
already exists.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.bayesian.variational_framework import MeanFieldGaussian
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)
from opifex.uncertainty.protocols import (
    UncertaintyAwareModule,
    VariationalModule,
)


def _make_uqno() -> UncertaintyQuantificationNeuralOperator:
    return UncertaintyQuantificationNeuralOperator(
        in_channels=2,
        out_channels=1,
        hidden_channels=8,
        modes=(2, 2),
        num_layers=2,
        rngs=nnx.Rngs(0),
    )


def test_mean_field_gaussian_has_kl_divergence_method() -> None:
    """Sanity: pre-existing ``MeanFieldGaussian`` exposes ``kl_divergence(...)``."""
    layer = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
    assert callable(layer.kl_divergence)
    kl_value = float(layer.kl_divergence(prior_mean=0.0, prior_std=1.0))
    assert jnp.isfinite(kl_value)


@pytest.mark.xfail(
    reason="MeanFieldGaussian does not yet expose predict_distribution(...)",
    strict=True,
)
def test_mean_field_gaussian_conforms_to_uncertainty_aware_module() -> None:
    """``MeanFieldGaussian`` must add ``predict_distribution`` to conform."""
    layer: Any = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
    assert isinstance(layer, UncertaintyAwareModule)


@pytest.mark.xfail(
    reason=(
        "Bayesian dense layer does not yet expose loss_components / negative_elbo "
        "/ predict_distribution surfaces."
    ),
    strict=True,
)
def test_mean_field_gaussian_conforms_to_variational_module() -> None:
    """``MeanFieldGaussian`` must add the loss_components + negative_elbo surface."""
    layer: Any = MeanFieldGaussian(num_params=4, rngs=nnx.Rngs(0))
    assert isinstance(layer, VariationalModule)


def test_uqno_predict_distribution_returns_predictive_distribution() -> None:
    """Post-migration: UQNO exposes the shared ``predict_distribution`` surface."""
    from opifex.uncertainty.types import PredictiveDistribution

    model = _make_uqno()
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 8, 2))
    dist = model.predict_distribution(x, rngs=nnx.Rngs(sample=1), num_samples=2)
    assert isinstance(dist, PredictiveDistribution)
    assert dist.mean.shape == (1, 8, 8, 1)


def test_uqno_conforms_to_uncertainty_aware_module() -> None:
    """UQNO satisfies :class:`UncertaintyAwareModule` via ``predict_distribution``."""
    model: Any = _make_uqno()
    assert isinstance(model, UncertaintyAwareModule)
