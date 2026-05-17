"""Phase 1 Task 1.3 — UQ protocol structural-conformance tests.

Sibling Reuse Gate:

* CalibraX `core/protocols.py` exposes ``BenchmarkProtocol``/``MetricProtocol``/
  ``DatasetProtocol`` — none have a ``predict_distribution`` / ``kl_divergence``
  / ``calibrate`` / ``fit`` surface for UQ. No match.
* Artifex ``GenerativeModelProtocol`` covers ``__call__`` + ``generate`` for
  *generation*, not for *uncertainty quantification*. No match.
* Datarax has no calibrator/conformalizer protocols.

Local implementation justified. These protocols are structural typing surfaces
only — no inheritance required.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.protocols import (
    Calibrator,
    Conformalizer,
    UncertaintyAwareModule,
    UncertaintyEstimator,
    VariationalModule,
)
from opifex.uncertainty.types import PredictiveDistribution


if TYPE_CHECKING:
    from flax import nnx


class _FakeUncertaintyAware:
    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PredictiveDistribution:
        del rngs
        return PredictiveDistribution(mean=x)


class _FakeVariational:
    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PredictiveDistribution:
        del rngs
        return PredictiveDistribution(mean=x)

    def kl_divergence(self) -> jax.Array:
        return jnp.array(0.5)

    def loss_components(
        self,
        batch: tuple[jax.Array, jax.Array],
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,
    ) -> UQLossComponents:
        del rngs
        x, _ = batch
        return UQLossComponents.from_components(
            config=config, data=jnp.mean(x), kl=self.kl_divergence()
        )

    def negative_elbo(
        self,
        batch: tuple[jax.Array, jax.Array],
        *,
        config: ObjectiveConfig,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        components = self.loss_components(batch, config=config, rngs=rngs)
        return components.total


class _FakeCalibrator:
    def fit(
        self,
        predictions: jax.Array,
        targets: jax.Array,
    ) -> dict[str, jax.Array]:
        return {"temperature": jnp.mean(predictions - targets)}


class _FakeConformalizer:
    def calibrate(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        *,
        alpha: float,
    ) -> dict[str, jax.Array | float]:
        return {"threshold": float(jnp.mean(predictions - targets)), "alpha": alpha}


class _FakeUncertaintyEstimator:
    def fit(
        self,
        x: jax.Array,
        y: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> None:
        del x, y, rngs

    def predict_distribution(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> PredictiveDistribution:
        del rngs
        return PredictiveDistribution(mean=x)


def test_uncertainty_aware_module_protocol_runtime_check_accepts_conforming_class() -> None:
    instance: Any = _FakeUncertaintyAware()
    assert isinstance(instance, UncertaintyAwareModule)


def test_uncertainty_aware_module_protocol_rejects_non_conforming_class() -> None:
    class NoPredict:
        pass

    instance: Any = NoPredict()
    assert not isinstance(instance, UncertaintyAwareModule)


def test_variational_module_protocol_runtime_check_accepts_conforming_class() -> None:
    instance: Any = _FakeVariational()
    assert isinstance(instance, VariationalModule)
    assert isinstance(instance, UncertaintyAwareModule)


def test_variational_module_returns_uq_loss_components() -> None:
    instance = _FakeVariational()
    config = ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=10,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=1.0,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )
    batch = (jnp.array([1.0, 2.0, 3.0]), jnp.array([0.0, 0.0, 0.0]))
    components = instance.loss_components(batch, config=config)
    assert isinstance(components, UQLossComponents)
    assert jnp.isfinite(components.total)


def test_variational_module_negative_elbo_is_scalar() -> None:
    instance = _FakeVariational()
    config = ObjectiveConfig(
        kl_weight=1.0,
        dataset_size=10,
        physics_weight=1.0,
        data_weight=1.0,
        boundary_weight=1.0,
        initial_condition_weight=1.0,
        regularization_weight=1.0,
        calibration_weight=1.0,
        conformal_weight=1.0,
        pac_bayes_weight=1.0,
    )
    batch = (jnp.array([1.0]), jnp.array([0.0]))
    elbo = instance.negative_elbo(batch, config=config)
    assert jnp.isfinite(elbo)
    assert elbo.shape == ()


def test_calibrator_protocol_runtime_check_accepts_conforming_class() -> None:
    instance: Any = _FakeCalibrator()
    assert isinstance(instance, Calibrator)


def test_conformalizer_protocol_runtime_check_accepts_conforming_class() -> None:
    instance: Any = _FakeConformalizer()
    assert isinstance(instance, Conformalizer)


def test_uncertainty_estimator_protocol_runtime_check_accepts_conforming_class() -> None:
    instance: Any = _FakeUncertaintyEstimator()
    assert isinstance(instance, UncertaintyEstimator)


def test_protocols_do_not_require_inheritance() -> None:
    """Structural typing — no base class required."""
    estimator: Any = _FakeUncertaintyEstimator()
    assert UncertaintyEstimator not in type(estimator).__mro__


def test_predict_distribution_returns_predictive_distribution() -> None:
    instance: UncertaintyAwareModule = _FakeUncertaintyAware()
    result = instance.predict_distribution(jnp.array([1.0, 2.0]))
    assert isinstance(result, PredictiveDistribution)


def test_conformalizer_calibrate_records_alpha_in_output() -> None:
    instance = _FakeConformalizer()
    out = instance.calibrate(jnp.zeros(4), jnp.zeros(4), alpha=0.1)
    assert out["alpha"] == 0.1
