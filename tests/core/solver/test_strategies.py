"""Tests for Training Strategies and Components in Unified Solver.

This module tests the TrainingStrategy protocol and concrete implementations
like AdaptiveLossBalancing and CurriculumRegularization.
"""

from typing import Protocol

from opifex.core.solver.strategies import (
    AdaptiveLossBalancing,
    CurriculumRegularization,
    TrainingComponent,
    TrainingStrategy,
)


class TestStrategyProtocols:
    """Test protocol definitions."""

    def test_training_component_protocol(self):
        """Verify TrainingComponent protocol definition."""
        # A component should be callable or have specific lifecycle methods
        # For now, we assume it needs at least an update or on_step method
        assert issubclass(TrainingComponent, Protocol)

    def test_training_strategy_protocol(self):
        """Verify TrainingStrategy protocol definition."""
        # A strategy should be able to configure training
        assert issubclass(TrainingStrategy, Protocol)


class TestAdaptiveLossBalancing:
    """Test Adaptive Loss Balancing strategy."""

    def test_initialization(self):
        """Test initialization with config."""
        strategy = AdaptiveLossBalancing(loss_keys=["pde", "bc"], alpha=0.5)
        assert isinstance(strategy, TrainingComponent)
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
    """Test Curriculum Regularization strategy."""

    def test_initialization(self):
        """Test initialization."""
        strategy = CurriculumRegularization()
        assert isinstance(strategy, TrainingComponent)
        assert strategy.start_val == 0.0

    def test_parameter_progression(self):
        """Test regularization parameter progression over epochs."""
        strategy = CurriculumRegularization(start_val=0.0, end_val=1.0, total_epochs=10)
        val_0 = strategy.get_value(epoch=0)
        val_5 = strategy.get_value(epoch=5)
        val_10 = strategy.get_value(epoch=10)

        assert val_0 == 0.0
        assert val_5 == 0.5
        assert val_10 == 1.0

    def test_on_epoch_begin_coverage(self):
        """Verify on_epoch_begin runs (coverage check)."""
        strategy = CurriculumRegularization()
        # Should run without error and hit the logic
        strategy.on_epoch_begin(0, None)
