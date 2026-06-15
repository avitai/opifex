"""Tests for the evidential (NIG) distribution adapter."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters.base import DistributionAdapterProtocol
from opifex.uncertainty.adapters.evidential import EvidentialAdapter
from opifex.uncertainty.evidential import (
    aleatoric_variance,
    epistemic_variance,
    NIGParams,
    positive_evidential_params,
)
from opifex.uncertainty.types import PredictiveDistribution


def _params() -> NIGParams:
    """Valid NIG parameters (alpha > 1) from a raw 4-vector."""
    return positive_evidential_params(jnp.array([0.5, 0.3, 1.2, 0.7]))


class TestEvidentialAdapter:
    def test_satisfies_distribution_adapter_protocol(self) -> None:
        assert isinstance(EvidentialAdapter(), DistributionAdapterProtocol)

    def test_returns_predictive_distribution_with_mean_gamma(self) -> None:
        params = _params()
        pred = EvidentialAdapter().from_distribution(params)
        assert isinstance(pred, PredictiveDistribution)
        assert jnp.allclose(pred.mean, params.gamma)

    def test_variance_decomposition_matches_primitive(self) -> None:
        params = _params()
        pred = EvidentialAdapter().from_distribution(params)
        assert pred.aleatoric is not None
        assert pred.epistemic is not None
        assert pred.total_uncertainty is not None
        assert jnp.allclose(pred.aleatoric, aleatoric_variance(params))
        assert jnp.allclose(pred.epistemic, epistemic_variance(params))
        assert jnp.allclose(pred.total_uncertainty, pred.aleatoric + pred.epistemic)

    def test_attaches_method_and_source_package_provenance(self) -> None:
        metadata = dict(EvidentialAdapter().from_distribution(_params()).metadata)
        assert metadata["method"] == "deep_evidential_regression"
        assert metadata["source_package"] == "opifex"

    def test_jit_safe(self) -> None:
        adapter = EvidentialAdapter()
        params = _params()
        jitted = jax.jit(lambda p: adapter.from_distribution(p).mean)
        assert jnp.allclose(jitted(params), params.gamma)
