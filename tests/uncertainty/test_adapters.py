"""Tests for distribution adapter / model adapter protocols.

Sibling-package targets:

* Artifex ``Distribution`` (``../artifex/src/artifex/generative_models/core/
  distributions/base.py``) is the primary adapter target. Distrax-like is
  secondary. TFP / FlowJAX / bijx / GPJax / NumPyro are exposed through
  unsupported-backend metadata.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.adapters import (
    DistributionAdapterProtocol,
    DistributionAdapterSpec,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


if TYPE_CHECKING:
    from flax import nnx


class _FakeArtifexLikeDistribution:
    """Minimal stand-in for ``artifex.generative_models.core.distributions.Normal``."""

    def sample(
        self,
        sample_shape: tuple[int, ...] = (),
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        del rngs
        return jnp.zeros(sample_shape or (1,))

    def log_prob(self, x: jax.Array) -> jax.Array:
        return jnp.zeros_like(x)

    def mean(self) -> jax.Array:
        return jnp.zeros(1)

    def variance(self) -> jax.Array:
        return jnp.ones(1)


class _ToyDistributionAdapter:
    def from_distribution(self, distribution: Any) -> PredictiveDistribution:
        mean = distribution.mean()
        variance = distribution.variance()
        return PredictiveDistribution(mean=mean, variance=variance)


class _ToyModelAdapter:
    def wrap(
        self,
        model: Any,
        capability: UQCapability,
    ) -> dict[str, Any]:
        del model
        return {
            "strategy": capability.default_strategy.value,
            "source_package": capability.source_package,
        }


def test_distribution_adapter_protocol_accepts_conforming_class() -> None:
    instance: Any = _ToyDistributionAdapter()
    assert isinstance(instance, DistributionAdapterProtocol)


def test_distribution_adapter_round_trips_artifex_like_distribution() -> None:
    adapter = _ToyDistributionAdapter()
    distribution = _FakeArtifexLikeDistribution()
    result = adapter.from_distribution(distribution)
    assert isinstance(result, PredictiveDistribution)
    assert result.mean.shape == (1,)
    assert result.variance is not None and result.variance.shape == (1,)


def test_model_uncertainty_adapter_protocol_accepts_conforming_class() -> None:
    instance: Any = _ToyModelAdapter()
    assert isinstance(instance, ModelUncertaintyAdapterProtocol)


def test_model_uncertainty_adapter_records_strategy_and_source_package() -> None:
    adapter = _ToyModelAdapter()
    capability = UQCapability(
        native_bayesian=True,
        default_strategy=DefaultStrategy.BAYESIAN,
        source_package="artifex",
    )
    wrapped = adapter.wrap(model=object(), capability=capability)
    assert wrapped["strategy"] == "bayesian"
    assert wrapped["source_package"] == "artifex"


def test_distribution_adapter_spec_resolution_order_is_pinned_tuple() -> None:
    spec = DistributionAdapterSpec(
        name="multi_backend",
        primary_target="artifex_distribution",
        resolution_order=(
            "artifex_distribution",
            "distrax",
            "tfp_substrate",
            "bijx",
            "flowjax",
            "gpjax",
            "numpyro",
        ),
    )
    assert spec.resolution_order[0] == "artifex_distribution"
    assert spec.resolution_order[1] == "distrax"
    assert hash(spec) == hash(spec)
