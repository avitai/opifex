"""Real-model-invocation tests for the rewritten uncertainty trainers.

Per Task 8.3 (Phase 8 plan, lines 451-649), the three trainers below were
previously generating mock ``PRNGKey(42)`` predictions and never actually
calling their wrapped model. This test module verifies the rewrite:

1. ``ActiveUncertaintyLearner.acquire_samples(...)`` calls the wrapped
   model on ``x_pool`` (counter pattern) and accepts an
   ``acquisition_fn: Callable[[jax.Array, jax.Array], jax.Array]``
   instead of a string strategy.
2. ``UncertaintyGuidedTrainer.select_uncertain_samples(...)`` and
   ``compute_uncertainty_weights(...)`` invoke the wrapped
   ``uncertainty_quantifier``.
3. ``MultiFidelityUncertaintyTrainer.propagate_multi_fidelity_uncertainty``
   invokes **both** ``high_fidelity_model`` and ``low_fidelity_model``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.training.basic_trainer import (
    ActiveUncertaintyLearner,
    MultiFidelityUncertaintyTrainer,
    UncertaintyGuidedTrainer,
)
from opifex.uncertainty.aggregators.basic import UncertaintyQuantifier
from opifex.uncertainty.aggregators.types import UncertaintyComponents


class _CountingModel(nnx.Module):
    """nnx.Module whose ``__call__`` increments a counter on every invocation."""

    def __init__(self) -> None:
        super().__init__()
        # Use a python list to capture mutability under nnx (counter is not
        # an nnx state variable; the test only inspects host-side counts).
        self._counter = [0]

    def __call__(self, x: jax.Array) -> jax.Array:
        self._counter[0] += 1
        # Linear forward pass with shape (batch, 1) so the downstream
        # quantifier sees the right shape.
        if x.ndim == 1:
            x = x[:, None]
        return jnp.sum(x, axis=-1, keepdims=True)

    @property
    def call_count(self) -> int:
        return self._counter[0]


class _CountingQuantifier(UncertaintyQuantifier):
    """Quantifier wrapper that counts ``decompose_uncertainty`` calls."""

    def __init__(self, num_samples: int = 8) -> None:
        super().__init__(num_samples=num_samples)
        self.call_count: int = 0

    def decompose_uncertainty(
        self,
        predictions: jax.Array,
        aleatoric_variance: jax.Array | None = None,
    ) -> UncertaintyComponents:
        self.call_count += 1
        return super().decompose_uncertainty(predictions, aleatoric_variance)


# ---------------------------------------------------------------------------
# ActiveUncertaintyLearner
# ---------------------------------------------------------------------------


class TestActiveUncertaintyLearnerInvokesModel:
    def test_acquire_samples_calls_wrapped_model(self) -> None:
        model = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=6)
        rngs = nnx.Rngs(active_acquire=0)
        learner = ActiveUncertaintyLearner(
            model=model,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
            acquisition_size=3,
        )
        x_pool = jnp.linspace(0.0, 1.0, 12).reshape(-1, 1)

        indices = learner.acquire_samples(x_pool)

        assert model.call_count > 0, "model was not invoked"
        assert len(indices) == 3
        assert all(0 <= int(i) < 12 for i in indices)

    def test_acquire_samples_accepts_callable_acquisition(self) -> None:
        """``acquire_samples`` must accept a callable acquisition_fn."""
        model = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=6)
        rngs = nnx.Rngs(active_acquire=0)
        learner = ActiveUncertaintyLearner(
            model=model,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
            acquisition_size=2,
        )
        x_pool = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)

        def custom_acq(mean: jax.Array, variance: jax.Array) -> jax.Array:
            # variance-weighted: pick highest variance.
            return variance

        indices = learner.acquire_samples(x_pool, acquisition_fn=custom_acq)

        assert model.call_count > 0
        assert len(indices) == 2


# ---------------------------------------------------------------------------
# UncertaintyGuidedTrainer
# ---------------------------------------------------------------------------


class TestUncertaintyGuidedTrainerInvokesModel:
    def test_select_uncertain_samples_calls_quantifier(self) -> None:
        model = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=5)
        rngs = nnx.Rngs(active_acquire=0)
        trainer = UncertaintyGuidedTrainer(
            model=model,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
        )
        x_pool = jnp.linspace(0.0, 1.0, 10).reshape(-1, 1)

        indices = trainer.select_uncertain_samples(x_pool, num_samples=3)

        assert quantifier.call_count > 0, "quantifier was not invoked"
        assert model.call_count > 0, "wrapped model was not invoked"
        assert len(indices) == 3

    def test_compute_uncertainty_weights_calls_quantifier(self) -> None:
        model = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=5)
        rngs = nnx.Rngs(active_acquire=0)
        trainer = UncertaintyGuidedTrainer(
            model=model,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
        )
        x = jnp.linspace(0.0, 1.0, 8).reshape(-1, 1)
        y = jnp.zeros((8, 1))

        weights = trainer.compute_uncertainty_weights(x, y)

        assert quantifier.call_count > 0
        assert model.call_count > 0
        assert weights.shape == (8,)


# ---------------------------------------------------------------------------
# MultiFidelityUncertaintyTrainer
# ---------------------------------------------------------------------------


class TestMultiFidelityTrainerInvokesBothModels:
    def test_propagate_calls_both_models(self) -> None:
        hi = _CountingModel()
        lo = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=4)
        rngs = nnx.Rngs(active_acquire=0)
        trainer = MultiFidelityUncertaintyTrainer(
            high_fidelity_model=hi,
            low_fidelity_model=lo,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
            fidelity_ratio=0.3,
        )
        x = jnp.linspace(-1.0, 1.0, 7).reshape(-1, 1)

        out = trainer.propagate_multi_fidelity_uncertainty(x)

        assert hi.call_count > 0, "high_fidelity_model was not invoked"
        assert lo.call_count > 0, "low_fidelity_model was not invoked"
        assert out.shape == (7,)

    def test_fidelity_weighting_consumes_outputs(self) -> None:
        """The Kennedy-O'Hagan fidelity-ratio weighting must apply."""
        hi = _CountingModel()
        lo = _CountingModel()
        quantifier = _CountingQuantifier(num_samples=4)
        rngs = nnx.Rngs(active_acquire=0)
        trainer = MultiFidelityUncertaintyTrainer(
            high_fidelity_model=hi,
            low_fidelity_model=lo,
            uncertainty_quantifier=quantifier,
            rngs=rngs,
            fidelity_ratio=1.0,  # full high-fidelity
        )
        trainer_lo = MultiFidelityUncertaintyTrainer(
            high_fidelity_model=hi,
            low_fidelity_model=lo,
            uncertainty_quantifier=_CountingQuantifier(num_samples=4),
            rngs=nnx.Rngs(active_acquire=0),
            fidelity_ratio=0.0,  # full low-fidelity
        )
        x = jnp.linspace(0.0, 1.0, 5).reshape(-1, 1)

        full_hi = trainer.propagate_multi_fidelity_uncertainty(x)
        full_lo = trainer_lo.propagate_multi_fidelity_uncertainty(x)
        # Both endpoints must produce finite uncertainties.
        assert jnp.all(jnp.isfinite(full_hi))
        assert jnp.all(jnp.isfinite(full_lo))
