"""Tests for Training Strategies and Components in Unified Solver.

This module tests the TrainingStrategy protocol and concrete implementations
like AdaptiveLossBalancing and CurriculumRegularization.
"""

from typing import Protocol

import jax.numpy as jnp
from flax import nnx

from opifex.core.solver.strategies import (
    AdaptiveLossBalancing,
    CurriculumRegularization,
    TrainingCallback,
    TrainingStrategy,
)


class TestStrategyProtocols:
    """Test protocol definitions."""

    def test_training_component_protocol(self):
        """Verify TrainingCallback protocol definition."""
        # A component should be callable or have specific lifecycle methods
        # For now, we assume it needs at least an update or on_step method
        assert issubclass(TrainingCallback, Protocol)

    def test_training_strategy_protocol(self):
        """Verify TrainingStrategy protocol definition."""
        # A strategy should be able to configure training
        assert issubclass(TrainingStrategy, Protocol)


class TestAdaptiveLossBalancing:
    """Test Adaptive Loss Balancing strategy."""

    def test_initialization(self):
        """Test initialization with config."""
        strategy = AdaptiveLossBalancing(loss_keys=["pde", "bc"], alpha=0.5)
        assert isinstance(strategy, TrainingCallback)
        # Weights should be initialized eagerly
        assert len(strategy.weights) == 2

    def test_weight_update(self):
        """Test weight update logic."""
        strategy = AdaptiveLossBalancing(loss_keys=["pde", "bc"], alpha=0.5)
        # Simulate loss components
        losses = {"pde": 1.0, "bc": 0.01}
        strategy.update(losses)
        # Weights should remain populated
        assert len(strategy.weights) == 2

        # Iterate to check values without knowing key order or accessing by key
        # This avoids potential AttributeError with nnx.Dict string keys in some envs
        weights = list(strategy.weights.values())  # Returns variables
        # We expect one small weight (pde, ~1.0) and one large weight (bc, ~100.0)
        vals = sorted([float(w.value) for w in weights])

        # Inverse magnitude logic: 100.0 > 1.0
        assert vals[1] > vals[0] * 50


class TestCurriculumRegularization:
    """Test Curriculum Regularization strategy.

    The callback anneals a bound loss-term weight (an ``nnx.Variable`` shared
    with the trainer's ``loss_fn``) along a linear schedule. The behavioural
    contract is that ``on_epoch_begin`` actually writes the scheduled value into
    that weight and that the write is visible across an ``nnx.jit`` boundary.
    """

    def test_initialization(self):
        """Bind to a loss-term weight Variable and expose schedule bounds."""
        weight = nnx.Variable(jnp.array(0.0))
        strategy = CurriculumRegularization(target_weight=weight)
        assert isinstance(strategy, TrainingCallback)
        assert strategy.start_val == 0.0

    def test_parameter_progression(self):
        """Linear schedule (clamped) matches optax.linear_schedule."""
        weight = nnx.Variable(jnp.array(0.0))
        strategy = CurriculumRegularization(
            target_weight=weight, start_val=0.0, end_val=1.0, total_epochs=10
        )
        assert strategy.get_value(epoch=0) == 0.0
        assert strategy.get_value(epoch=5) == 0.5
        assert strategy.get_value(epoch=10) == 1.0
        assert strategy.get_value(epoch=20) == 1.0  # clamped past transition

    def test_on_epoch_begin_applies_schedule_to_target(self):
        """on_epoch_begin must WRITE the scheduled value into the bound weight."""
        weight = nnx.Variable(jnp.array(0.0))
        strategy = CurriculumRegularization(
            target_weight=weight, start_val=0.0, end_val=1.0, total_epochs=10
        )
        strategy.on_epoch_begin(epoch=5, state=None)
        assert float(weight.value) == 0.5
        strategy.on_epoch_begin(epoch=10, state=None)
        assert float(weight.value) == 1.0

    def test_weight_update_visible_across_jit_boundary(self):
        """Eager per-epoch update is picked up by a subsequent nnx.jit step.

        This pins the transform-compatibility contract: mutating the bound
        ``nnx.Variable`` outside the jitted step (as the trainer does in its
        epoch loop) is reflected inside ``nnx.jit`` without recompilation.
        """
        weight = nnx.Variable(jnp.array(0.0))
        strategy = CurriculumRegularization(
            target_weight=weight, start_val=0.0, end_val=1.0, total_epochs=10
        )

        @nnx.jit
        def weighted_loss(weight_var: nnx.Variable, x: jnp.ndarray) -> jnp.ndarray:
            return weight_var.value * jnp.sum(x**2)

        x = jnp.array([1.0, 2.0])  # sum(x**2) == 5.0

        strategy.on_epoch_begin(epoch=0, state=None)  # weight -> 0.0
        loss_start = weighted_loss(weight, x)
        strategy.on_epoch_begin(epoch=10, state=None)  # weight -> 1.0
        loss_end = weighted_loss(weight, x)

        assert float(loss_start) == 0.0
        assert float(loss_end) == 5.0
