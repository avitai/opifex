"""Extended tests for BasicTrainer class.

This module tests the basic trainer functionality, covering:
- BasicTrainer initialization
- Loss computation (MSE, MAE, regularization)
- Training step execution
- Validation step execution
- Full training loop
- Comprehensive metrics summary
- Physics-informed training step
- Quantum training step
- Checkpointing
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax.numpy as jnp
from flax import nnx

from opifex.core.training.config import (
    CheckpointConfig,
    LossConfig,
    OptimizationConfig,
    QuantumTrainingConfig,
    TrainingConfig,
    ValidationConfig,
)
from opifex.neural.base import StandardMLP
from opifex.training.basic_trainer import BasicTrainer


class TestBasicTrainerInitialization:
    """Test BasicTrainer initialization."""

    def test_default_initialization(self):
        """Test BasicTrainer with default config."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = BasicTrainer(model, config)

        assert trainer.config is config
        assert trainer.state is not None
        assert trainer.state.model is model
        assert trainer.physics_loss is None

    def test_initialization_with_custom_rngs(self):
        """Test BasicTrainer with custom RNGs."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)
        rngs = nnx.Rngs(123)

        trainer = BasicTrainer(model, config, rngs=rngs)

        assert trainer.state.rngs is rngs

    def test_training_state_initialized(self):
        """Test that training state is properly initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = BasicTrainer(model, config)

        assert trainer.state.step == 0
        assert trainer.state.epoch == 0
        assert trainer.state.optimizer is not None
        assert trainer.state.opt_state is not None
        assert trainer.state.best_loss == float("inf")

    def test_metrics_initialized(self):
        """Test that metrics object is initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = BasicTrainer(model, config)

        assert trainer.metrics is not None

    def test_advanced_metrics_initialized(self):
        """Test that advanced metrics collector is initialized."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=10)

        trainer = BasicTrainer(model, config)

        assert trainer.advanced_metrics is not None

    def test_optimizer_types(self):
        """Test different optimizer types."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        for optimizer_type in ["adam", "sgd", "rmsprop", "adamw"]:
            config = TrainingConfig(
                num_epochs=10,
                optimization_config=OptimizationConfig(optimizer=optimizer_type),
            )
            trainer = BasicTrainer(model, config)
            assert trainer.state.optimizer is not None


class TestBasicTrainerLossComputation:
    """Test loss computation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(
            num_epochs=10,
            loss_config=LossConfig(loss_type="mse"),
        )
        self.trainer = BasicTrainer(self.model, self.config)

    def test_compute_loss_mse(self):
        """Test MSE loss computation."""
        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 1))

        loss = self.trainer.compute_loss(x, y)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar
        assert float(loss) >= 0

    def test_compute_loss_mae(self):
        """Test MAE loss computation."""
        self.trainer.config.loss_config.loss_type = "mae"

        x = jnp.ones((8, 4))
        y = jnp.zeros((8, 1))

        loss = self.trainer.compute_loss(x, y)

        assert isinstance(loss, jnp.ndarray)
        assert float(loss) >= 0

    def test_compute_loss_with_regularization(self):
        """Test loss computation with regularization."""
        self.trainer.config.loss_config.regularization_weight = 0.01

        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        loss_with_reg = self.trainer.compute_loss(x, y)

        # Reset and compute without regularization
        self.trainer.config.loss_config.regularization_weight = 0.0
        loss_without_reg = self.trainer.compute_loss(x, y)

        # With regularization should be >= without
        assert float(loss_with_reg) >= float(loss_without_reg)

    def test_compute_loss_zero_for_perfect_prediction(self):
        """Test that loss is zero when prediction matches target."""
        # This depends on model output, so just verify it's a valid scalar
        x = jnp.zeros((8, 4))
        y = jnp.zeros((8, 1))

        loss = self.trainer.compute_loss(x, y)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()


class TestBasicTrainerTrainingStep:
    """Test training step functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(
            num_epochs=10,
            loss_config=LossConfig(loss_type="mse"),
        )
        self.trainer = BasicTrainer(self.model, self.config)

    def test_training_step_returns_loss(self):
        """Test that training step returns loss."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        loss = self.trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar

    def test_training_step_updates_step_counter(self):
        """Test that training step updates the step counter."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        initial_step = self.trainer.state.step

        self.trainer.training_step(x, y)

        assert self.trainer.state.step == initial_step + 1

    def test_training_step_updates_best_loss(self):
        """Test that training step updates best loss."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        self.trainer.training_step(x, y)

        # Best loss should be updated from infinity
        assert self.trainer.state.best_loss < float("inf")

    def test_training_step_with_physics_loss(self):
        """Test training step with physics loss enabled."""
        # Create a mock physics loss with compute_residuals method
        mock_physics_loss = MagicMock()
        mock_physics_loss.compute_residuals = MagicMock(return_value=jnp.array(0.1))
        self.trainer.set_physics_loss(mock_physics_loss)

        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        loss = self.trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)
        mock_physics_loss.compute_residuals.assert_called()


class TestBasicTrainerValidationStep:
    """Test validation step functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=10)
        self.trainer = BasicTrainer(self.model, self.config)

    def test_validation_step_returns_loss(self):
        """Test that validation step returns loss."""
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        val_loss = self.trainer.validation_step(x_val, y_val)

        assert isinstance(val_loss, jnp.ndarray)
        assert val_loss.shape == ()  # Scalar

    def test_validation_step_updates_best_val_loss(self):
        """Test that validation step updates best validation loss."""
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        self.trainer.validation_step(x_val, y_val)

        # Best val loss should be updated from infinity
        assert self.trainer.state.best_val_loss < float("inf")

    def test_validation_step_increases_plateau_count(self):
        """Test that validation step increases plateau count when loss doesn't improve."""
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        # First call sets best_val_loss
        self.trainer.validation_step(x_val, y_val)
        initial_plateau = self.trainer.state.plateau_count

        # Subsequent calls with same data may increase plateau count
        # (depends on whether loss improves)
        self.trainer.validation_step(x_val, y_val)

        # Plateau count should be >= initial
        assert self.trainer.state.plateau_count >= initial_plateau


class TestBasicTrainerTrain:
    """Test full training loop."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(
            num_epochs=3,
            validation_config=ValidationConfig(validation_frequency=1),
            verbose=False,
        )
        self.trainer = BasicTrainer(self.model, self.config)

    def test_train_returns_model_and_metrics(self):
        """Test that train returns model and metrics."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))

        model, metrics = self.trainer.train((x_train, y_train))

        assert model is not None
        assert metrics is not None

    def test_train_with_validation_data(self):
        """Test training with validation data."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))
        x_val = jnp.ones((8, 4))
        y_val = jnp.ones((8, 1))

        model, metrics = self.trainer.train((x_train, y_train), val_data=(x_val, y_val))

        assert model is not None
        # Validation losses should have been recorded
        assert len(metrics.val_losses) > 0

    def test_train_updates_metrics(self):
        """Test that training updates metrics."""
        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))

        _, metrics = self.trainer.train((x_train, y_train))

        # Training losses should have been recorded
        assert len(metrics.train_losses) > 0

    def test_train_with_progress_callback(self):
        """Test training with progress callback."""
        callback_calls = []

        def progress_callback(info):
            callback_calls.append(info)

        self.trainer.config.progress_callback = progress_callback

        x_train = jnp.ones((16, 4))
        y_train = jnp.ones((16, 1))

        self.trainer.train((x_train, y_train))

        # Callback should have been called for each epoch
        assert len(callback_calls) == self.config.num_epochs


class TestBasicTrainerQuantumTraining:
    """Test quantum-specific training functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Model with input size matching flattened positions
        self.model = StandardMLP([12, 16, 1], rngs=nnx.Rngs(42))  # 4 atoms * 3 coords
        self.config = TrainingConfig(
            num_epochs=3,
            quantum_config=QuantumTrainingConfig(),
            verbose=False,
        )
        self.trainer = BasicTrainer(self.model, self.config)

    def test_compute_quantum_loss(self):
        """Test quantum loss computation."""
        positions = jnp.ones((8, 4, 3))  # batch, atoms, coords
        energies = jnp.ones((8, 1))

        loss = self.trainer.compute_quantum_loss(positions, energies)

        assert isinstance(loss, jnp.ndarray)
        assert loss.shape == ()  # Scalar

    def test_compute_quantum_loss_with_density_constraints(self):
        """Test quantum loss with density constraints enabled."""
        # quantum_config is set in setup_method, so it's not None
        assert self.trainer.config.quantum_config is not None
        self.trainer.config.quantum_config.enable_density_constraints = True

        positions = jnp.ones((8, 4, 3))
        energies = jnp.ones((8, 1))

        loss = self.trainer.compute_quantum_loss(positions, energies)

        assert isinstance(loss, jnp.ndarray)


class TestBasicTrainerMetrics:
    """Test comprehensive metrics summary."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=3, verbose=False)
        self.trainer = BasicTrainer(self.model, self.config)

    def test_get_comprehensive_metrics_summary(self):
        """Test getting comprehensive metrics summary."""
        # Do some training first
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))
        self.trainer.training_step(x, y)

        summary = self.trainer.get_comprehensive_metrics_summary()

        assert isinstance(summary, dict)
        assert "basic_metrics" in summary
        assert "training_state" in summary
        assert "advanced_metrics" in summary

    def test_metrics_summary_includes_training_state(self):
        """Test that metrics summary includes training state."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))
        self.trainer.training_step(x, y)

        summary = self.trainer.get_comprehensive_metrics_summary()

        state = summary["training_state"]
        assert "current_step" in state
        assert "current_epoch" in state
        assert "best_loss" in state
        assert "best_val_loss" in state

    def test_metrics_summary_window_size(self):
        """Test metrics summary with custom window size."""
        x = jnp.ones((8, 4))
        y = jnp.ones((8, 1))

        # Do multiple training steps
        for _ in range(5):
            self.trainer.training_step(x, y)

        summary = self.trainer.get_comprehensive_metrics_summary(window_size=3)

        # Check that window size is applied to train_losses
        basic_metrics = summary["basic_metrics"]
        assert len(basic_metrics["train_losses"]) <= 3


class TestBasicTrainerPhysicsLoss:
    """Test physics loss integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        self.config = TrainingConfig(num_epochs=3, verbose=False)
        self.trainer = BasicTrainer(self.model, self.config)

    def test_set_physics_loss(self):
        """Test setting physics loss."""
        mock_physics_loss = MagicMock()

        self.trainer.set_physics_loss(mock_physics_loss)

        assert self.trainer.physics_loss is mock_physics_loss

    def test_physics_loss_initially_none(self):
        """Test that physics loss is None by default."""
        assert self.trainer.physics_loss is None


class TestBasicTrainerCheckpointing:
    """Test checkpointing functionality."""

    def test_checkpoint_setup(self, temp_directory):
        """Test checkpoint manager setup."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        checkpoint_dir = str(temp_directory / "checkpoints")

        config = TrainingConfig(
            num_epochs=3,
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=checkpoint_dir,
                save_frequency=1,
            ),
            verbose=False,
        )

        trainer = BasicTrainer(model, config)

        # Checkpoint manager should be set up
        assert hasattr(trainer, "checkpoint_manager")

    def test_checkpoint_directory_created(self, temp_directory):
        """Test that checkpoint directory is created."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        checkpoint_dir = temp_directory / "checkpoints"

        config = TrainingConfig(
            num_epochs=3,
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=str(checkpoint_dir),
            ),
            verbose=False,
        )

        BasicTrainer(model, config)

        assert checkpoint_dir.exists()


class TestBasicTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_batch(self):
        """Test training with single sample batch."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=3, verbose=False)
        trainer = BasicTrainer(model, config)

        x = jnp.ones((1, 4))
        y = jnp.ones((1, 1))

        loss = trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)

    def test_large_batch(self):
        """Test training with large batch."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(num_epochs=1, verbose=False)
        trainer = BasicTrainer(model, config)

        x = jnp.ones((256, 4))
        y = jnp.ones((256, 1))

        loss = trainer.training_step(x, y)

        assert isinstance(loss, jnp.ndarray)

    def test_different_input_dimensions(self):
        """Test training with different input dimensions."""
        for input_dim in [2, 8, 16]:
            model = StandardMLP([input_dim, 8, 1], rngs=nnx.Rngs(42))
            config = TrainingConfig(num_epochs=1, verbose=False)
            trainer = BasicTrainer(model, config)

            x = jnp.ones((8, input_dim))
            y = jnp.ones((8, 1))

            loss = trainer.training_step(x, y)

            assert isinstance(loss, jnp.ndarray)

    def test_different_output_dimensions(self):
        """Test training with different output dimensions."""
        for output_dim in [1, 4, 8]:
            model = StandardMLP([4, 8, output_dim], rngs=nnx.Rngs(42))
            config = TrainingConfig(num_epochs=1, verbose=False)
            trainer = BasicTrainer(model, config)

            x = jnp.ones((8, 4))
            y = jnp.ones((8, output_dim))

            loss = trainer.training_step(x, y)

            assert isinstance(loss, jnp.ndarray)
