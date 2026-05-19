"""Task 3.4: UQNO shared-objective contract tests.

The migration moves UQNO from a plain-dict ``__call__`` + a
``predict_with_uncertainty(key=...)`` MC helper + a free-floating
``kl_divergence`` onto the shared platform surface:

* :class:`opifex.uncertainty.objectives.UQLossComponents` /
  :class:`~opifex.uncertainty.objectives.ObjectiveConfig` for the training
  loss decomposition;
* :class:`opifex.uncertainty.types.PredictiveDistribution` for
  ``predict_distribution`` (with function-space metadata when the input
  carries spatial axes);
* ``negative_elbo`` mirroring the Task 3.2 ``ProbabilisticPINN``
  convention; and
* caller-owned ``nnx.Rngs`` at every stochastic method boundary —
  no hidden ``nnx.Rngs(0)`` fallback, no sample counters.

The tests in this file pin the contract before the migration so we can
watch them go RED → GREEN.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Mapping

from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents
from opifex.uncertainty.types import PredictiveDistribution


def _make_uqno(rngs: nnx.Rngs) -> UncertaintyQuantificationNeuralOperator:
    return UncertaintyQuantificationNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=4,
        modes=(2, 2),
        num_layers=1,
        rngs=rngs,
    )


def _make_objective(**overrides: float | str | None) -> ObjectiveConfig:
    base: dict[str, float | str | None] = {
        "kl_weight": 1.0,
        "dataset_size": 16,
        "physics_weight": 1.0,
        "data_weight": 1.0,
        "boundary_weight": 1.0,
        "initial_condition_weight": 1.0,
        "regularization_weight": 1.0,
        "calibration_weight": 1.0,
        "conformal_weight": 1.0,
        "pac_bayes_weight": 1.0,
    }
    base.update(overrides)
    return ObjectiveConfig(**base)  # type: ignore[arg-type]


def _make_batch(seed: int = 0) -> Mapping[str, jax.Array]:
    key = jax.random.PRNGKey(seed)
    k_x, k_y = jax.random.split(key)
    return {
        "x": jax.random.normal(k_x, (2, 4, 4, 1)),
        "y": jax.random.normal(k_y, (2, 4, 4, 1)),
    }


class TestUqnoPredictDistribution:
    """`predict_distribution` returns a :class:`PredictiveDistribution`."""

    def test_returns_predictive_distribution(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        x = jnp.ones((1, 4, 4, 1))
        dist = uqno.predict_distribution(x, rngs=nnx.Rngs(sample=7), num_samples=4)
        assert isinstance(dist, PredictiveDistribution)
        assert dist.mean.shape == (1, 4, 4, 1)

    def test_includes_function_space_metadata_for_spatial_input(self) -> None:
        """Spatial axes are advertised in the metadata so function-space callers know."""
        uqno = _make_uqno(nnx.Rngs(0))
        x = jnp.ones((1, 4, 4, 1))
        dist = uqno.predict_distribution(x, rngs=nnx.Rngs(sample=7), num_samples=2)
        meta = dist.metadata_dict()
        assert "spatial_axes" in meta
        assert tuple(meta["spatial_axes"]) == (1, 2)

    def test_populates_epistemic_variance_from_mc_samples(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        x = jnp.ones((1, 4, 4, 1))
        dist = uqno.predict_distribution(x, rngs=nnx.Rngs(sample=7), num_samples=4)
        assert dist.epistemic is not None
        assert dist.epistemic.shape == dist.mean.shape
        assert bool(jnp.all(dist.epistemic >= 0.0))

    def test_requires_rngs_at_method_boundary(self) -> None:
        """No hidden ``nnx.Rngs(0)`` fallback — the call must fail without ``rngs``."""
        uqno = _make_uqno(nnx.Rngs(0))
        x = jnp.ones((1, 4, 4, 1))
        with pytest.raises(TypeError):
            uqno.predict_distribution(x, num_samples=2)  # type: ignore[call-arg]


class TestUqnoLossComponents:
    """`loss_components` returns a shared :class:`UQLossComponents`."""

    def test_returns_uq_loss_components(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        components = uqno.loss_components(
            _make_batch(), rngs=nnx.Rngs(sample=7), objective=_make_objective()
        )
        assert isinstance(components, UQLossComponents)
        assert jnp.isfinite(components.total)

    def test_populates_data_and_kl_components(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        components = uqno.loss_components(
            _make_batch(), rngs=nnx.Rngs(sample=7), objective=_make_objective()
        )
        assert components.data is not None
        assert components.kl is not None
        assert bool(jnp.all(jnp.isfinite(components.kl)))

    def test_kl_aggregates_shared_bayesian_layer_kls(self) -> None:
        """``kl_divergence()`` aggregates each shared Bayesian layer's KL."""
        uqno = _make_uqno(nnx.Rngs(0))
        components = uqno.loss_components(
            _make_batch(), rngs=nnx.Rngs(sample=7), objective=_make_objective(kl_weight=1.0)
        )
        assert components.kl is not None
        assert float(components.kl) == pytest.approx(
            float(uqno.kl_divergence()), rel=1e-6, abs=1e-6
        )


class TestUqnoNegativeElbo:
    """`negative_elbo` sets the ``negative_elbo`` slot without weighting it again."""

    def test_returns_uq_loss_components_with_negative_elbo_populated(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        components = uqno.negative_elbo(
            _make_batch(), rngs=nnx.Rngs(sample=7), objective=_make_objective()
        )
        assert isinstance(components, UQLossComponents)
        assert components.negative_elbo is not None

    def test_negative_elbo_matches_total(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        components = uqno.negative_elbo(
            _make_batch(), rngs=nnx.Rngs(sample=7), objective=_make_objective()
        )
        assert components.negative_elbo is not None
        assert float(components.negative_elbo) == pytest.approx(
            float(components.total), rel=1e-6, abs=1e-6
        )


class TestUqnoUsesSharedBayesianLayers:
    """UQNO imports the shared Bayesian layers — no local duplicates."""

    def test_uqno_layers_use_shared_bayesian_spectral_convolution(self) -> None:
        from opifex.neural.operators.specialized import uqno as uqno_module
        from opifex.uncertainty.layers.bayesian import (
            BayesianLinear as SharedBayesianLinear,
            BayesianSpectralConvolution as SharedBayesianSpectralConvolution,
        )

        # The names exposed from the UQNO module must resolve to the shared
        # surface (so any local copy has been deleted, not just shadowed).
        assert uqno_module.BayesianLinear is SharedBayesianLinear
        assert uqno_module.BayesianSpectralConvolution is SharedBayesianSpectralConvolution

    def test_uqno_module_does_not_define_local_sample_counters(self) -> None:
        """Counter-based hidden RNG paths are gone."""
        import inspect

        from opifex.neural.operators.specialized import uqno as uqno_module

        source = inspect.getsource(uqno_module)
        assert "_sample_counter" not in source
        # The local counter-keyed RNG fallbacks must be gone too.
        assert "jax.random.PRNGKey(counter" not in source


class TestUqnoTransformCompatibility:
    """JAX/NNX transforms must remain compatible with the new contract."""

    def test_negative_elbo_traces_under_nnx_jit_with_explicit_rngs(self) -> None:
        uqno = _make_uqno(nnx.Rngs(0))
        objective = _make_objective()
        batch = _make_batch()

        @nnx.jit
        def step(model: UncertaintyQuantificationNeuralOperator, rngs: nnx.Rngs) -> jax.Array:
            return model.negative_elbo(batch, rngs=rngs, objective=objective).total

        out = step(uqno, nnx.Rngs(sample=11))
        assert isinstance(out, jax.Array)
        assert out.shape == ()
        assert bool(jnp.all(jnp.isfinite(out)))
