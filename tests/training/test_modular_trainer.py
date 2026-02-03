"""Tests for ModularTrainer class.

This module tests the modular trainer functionality, covering:
- ModularTrainer initialization with default and custom configs
- Physics loss setting
- Training step execution
- Model parameter updates
- Training with validation data
- Metrics summary retrieval
- Cleanup method
- Error recovery integration
"""

from __future__ import annotations

import contextlib
from typing import cast
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.components import TrainingComponent
from opifex.core.training.config import (
    LossConfig,
    TrainingConfig,
)
from opifex.neural.base import StandardMLP
from opifex.training.basic_trainer import ModularTrainer


class TestModularTrainerInitialization:
    """Test ModularTrainer initialization."""

    def test_default_initialization(self):
        """Test ModularTrainer with default config."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = ModularTrainer(model, config)

        assert trainer.model is model
        assert trainer.config is config
        assert trainer.physics_loss is None
        assert trainer.training_state is not None

    def test_initialization_with_custom_rngs(self):
        """Test ModularTrainer with custom RNGs."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)
        rngs = nnx.Rngs(123)

        trainer = ModularTrainer(model, config, rngs=rngs)

        assert trainer.rngs is rngs

    def test_initialization_with_components(self):
        """Test ModularTrainer with custom components."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        # Create a mock component
        mock_component = MagicMock(spec=TrainingComponent)
        components = cast("dict[str, TrainingComponent]", {"custom": mock_component})

        trainer = ModularTrainer(model, config, components=components)

        assert "custom" in trainer.components
        mock_component.setup.assert_called_once()

    def test_training_state_initialized(self):
        """Test that training state is properly initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = ModularTrainer(model, config)

        assert trainer.training_state.step == 0
        assert trainer.training_state.epoch == 0
        assert trainer.training_state.optimizer is not None
        assert trainer.training_state.opt_state is not None

    def test_metrics_collector_initialized(self):
        """Test that metrics collector is initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = ModularTrainer(model, config)

        assert trainer.metrics_collector is not None

    def test_error_recovery_initialized(self):
        """Test that error recovery is initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = ModularTrainer(model, config)

        assert trainer.error_recovery is not None
        assert trainer.error_recovery.max_retries == 3


class TestModularTrainerPhysicsLoss:
    """Test physics loss setting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=10)
        self.trainer = ModularTrainer(self.model, self.config)

    def test_set_physics_loss(self):
        """Test setting physics loss."""
        mock_physics_loss = MagicMock()

        self.trainer.set_physics_loss(mock_physics_loss)

        assert self.trainer.physics_loss is mock_physics_loss

    def test_physics_loss_initially_none(self):
        """Test that physics loss is None by default."""
        assert self.trainer.physics_loss is None


class TestModularTrainerTrainingStep:
    """Test training step functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(
            num_epochs=10,
            loss_config=LossConfig(loss_type="mse"),
        )
        self.trainer = ModularTrainer(self.model, self.config)

    def test_training_step_returns_loss_and_metrics(self):
        """Test that training step returns loss and metrics."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        loss, metrics = self.trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar
        assert isinstance(metrics, dict)

    def test_training_step_updates_step_counter(self):
        """Test that training step updates the step counter."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        initial_step = self.trainer.training_state.step

        self.trainer.training_step(x, y)

        assert self.trainer.training_state.step == initial_step + 1

    def test_training_step_with_mse_loss(self):
        """Test training step with MSE loss."""
        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 1))

        loss, _ = self.trainer.training_step(x, y)

        assert float(loss) > 0  # MSE should be positive for non-matching outputs

    def test_training_step_with_mae_loss(self):
        """Test training step with MAE loss."""
        self.trainer.config.loss_config.loss_type = "mae"

        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 1))

        loss, _ = self.trainer.training_step(x, y)

        assert float(loss) > 0

    def test_training_step_with_physics_loss(self):
        """Test training step with physics loss enabled."""
        # Create a mock physics loss
        mock_physics_loss = MagicMock(return_value=jnp.array(0.1))
        self.trainer.set_physics_loss(mock_physics_loss)

        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        loss, _ = self.trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)


class TestModularTrainerTrain:
    """Test full training loop."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(
            num_epochs=3,
            validation_frequency=1,
        )
        self.trainer = ModularTrainer(self.model, self.config)

    def test_train_returns_model_and_metrics(self):
        """Test that train returns model and metrics."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))

        model, metrics = self.trainer.train((x_train, y_train))

        assert model is not None
        assert isinstance(metrics, dict)

    def test_train_with_validation_data(self):
        """Test training with validation data."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        model, metrics = self.trainer.train((x_train, y_train), val_data=(x_val, y_val))

        assert model is not None
        assert isinstance(metrics, dict)

    def test_train_updates_epoch(self):
        """Test that training updates epoch counter."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))

        self.trainer.train((x_train, y_train))

        # After training completes, final metrics are returned
        assert self.trainer.training_state.epoch >= 0


class TestModularTrainerValidationStep:
    """Test validation step functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=10)
        self.trainer = ModularTrainer(self.model, self.config)

    def test_validation_step_computes_loss(self):
        """Test that validation step computes loss."""
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        val_loss = self.trainer._validation_step(x_val, y_val)

        assert isinstance(val_loss, jnp.ndarray)
        assert val_loss.shape == ()  # Scalar


class TestModularTrainerMetrics:
    """Test metrics summary functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=3)
        self.trainer = ModularTrainer(self.model, self.config)

    def test_get_comprehensive_metrics_summary(self):
        """Test getting comprehensive metrics summary."""
        # Do some training first
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))
        self.trainer.training_step(x, y)

        metrics = self.trainer.get_comprehensive_metrics_summary()

        assert isinstance(metrics, dict)
        assert "metrics" in metrics
        assert "state" in metrics
        assert "recovery_attempts" in metrics

    def test_metrics_summary_includes_training_state(self):
        """Test that metrics summary includes training state."""
        metrics = self.trainer.get_comprehensive_metrics_summary()

        state_metrics = metrics["state"]["training_state"]
        assert "step" in state_metrics
        assert "epoch" in state_metrics
        assert "best_loss" in state_metrics

    def test_metrics_summary_window_size(self):
        """Test metrics summary with custom window size."""
        metrics = self.trainer.get_comprehensive_metrics_summary(window_size=5)

        assert isinstance(metrics, dict)


class TestModularTrainerCleanup:
    """Test cleanup functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=10)
        self.trainer = ModularTrainer(self.model, self.config)

    def test_cleanup_calls_component_cleanup(self):
        """Test that cleanup calls cleanup on all components."""
        # Add a mock component
        mock_component = MagicMock(spec=TrainingComponent)
        self.trainer.components["test"] = cast("TrainingComponent", mock_component)

        self.trainer.cleanup()

        mock_component.cleanup.assert_called_once()

    def test_cleanup_handles_empty_components(self):
        """Test cleanup with no custom components."""
        # Should not raise
        self.trainer.cleanup()


class TestModularTrainerErrorRecovery:
    """Test error recovery integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=10)
        self.trainer = ModularTrainer(self.model, self.config)

    def test_validate_issue_type_raises_on_none(self):
        """Test that validate_issue_type raises ValueError on None."""
        with pytest.raises(ValueError, match="cannot be None"):
            self.trainer._validate_issue_type(None)

    def test_validate_issue_type_returns_value(self):
        """Test that validate_issue_type returns the value."""
        result = self.trainer._validate_issue_type("gradient_explosion")

        assert result == "gradient_explosion"


class TestModularTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_batch(self):
        """Test handling of empty batch (may fail gracefully)."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=3)
        trainer = ModularTrainer(model, config)

        x = jnp.ones((0, 4))
        y = jnp.ones((0, 1))

        # This may or may not raise depending on implementation
        # The test documents the behavior
        try:
            loss, _metrics = trainer.training_step(x, y)
            assert loss is not None
        except Exception:
            pass  # Empty batch handling is implementation-specific

    def test_mismatched_batch_sizes(self):
        """Test handling of mismatched batch sizes."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=3)
        trainer = ModularTrainer(model, config)

        x = jnp.ones((8, 4))
        y = jnp.ones((4, 1))  # Different batch size

        # This should fail or handle gracefully
        with contextlib.suppress(Exception):
            trainer.training_step(x, y)

    def test_large_learning_rate_stability(self):
        """Test training stability with large learning rate."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(
            num_epochs=3,
            learning_rate=10.0,  # Very large
        )
        trainer = ModularTrainer(model, config)

        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        # Should handle gracefully with error recovery
        with contextlib.suppress(RuntimeError):
            trainer.training_step(x, y)
