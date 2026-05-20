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
    from opifex.neural.operators.fno.base import FourierNeuralOperator
    from opifex.neural.operators.specialized.uqno import (
        UQNOBaseSolutionOperator,
        UQNOResidualOperator,
    )

    return UncertaintyQuantificationNeuralOperator(
        base=UQNOBaseSolutionOperator(
            FourierNeuralOperator(
                in_channels=2,
                out_channels=1,
                hidden_channels=8,
                modes=2,
                num_layers=2,
                rngs=nnx.Rngs(0),
            )
        ),
        residual=UQNOResidualOperator(
            FourierNeuralOperator(
                in_channels=2,
                out_channels=1,
                hidden_channels=8,
                modes=2,
                num_layers=2,
                rngs=nnx.Rngs(1),
            )
        ),
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


def test_uqno_predict_with_bands_returns_predictive_distribution_with_interval() -> None:
    """Conformal UQNO exposes ``predict_with_bands`` returning a bounded ``PredictiveDistribution``."""
    from opifex.uncertainty.types import PredictionInterval, PredictiveDistribution

    model = _make_uqno()
    x_calib = jax.random.normal(jax.random.PRNGKey(0), (8, 2, 8, 8))
    y_calib = model.predict_base(x_calib) + 0.1 * jax.random.normal(
        jax.random.PRNGKey(1), (8, 1, 8, 8)
    )
    model = model.with_calibrator(model.calibrate(x_calib, y_calib, alpha=0.1, delta=0.1))
    dist = model.predict_with_bands(jax.random.normal(jax.random.PRNGKey(2), (1, 2, 8, 8)))
    assert isinstance(dist, PredictiveDistribution)
    assert isinstance(dist.interval, PredictionInterval)
    assert dist.mean.shape == (1, 1, 8, 8)


def test_uqno_does_not_claim_uncertainty_aware_module_native_bayesian_surface() -> None:
    """Conformal UQNO is honest: it does NOT expose ``predict_distribution``.

    The conformal three-stage operator carries an
    ``FNOConformalAdapterSpec`` capability declaration (adapter-mediated
    UQ, ``native_bayesian=False``) — not the protocol surface used by
    Bayesian / variational modules.
    """
    model: Any = _make_uqno()
    assert not hasattr(model, "predict_distribution")
    assert not isinstance(model, UncertaintyAwareModule)
