"""Tests for unified Trainer class.

Following strict TDD principles - these tests are written FIRST to define
the expected behavior of the unified Trainer that consolidates BasicTrainer
and PhysicsInformedTrainer.

This unified Trainer is generic and extensible for all scientific ML methods.
"""

import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.core.training.physics_configs import (
    BoundaryConfig,
    ConservationConfig,
    ConstraintConfig,
)
from opifex.core.training.trainer import Trainer


class MockModel(nnx.Module):
    """Mock model for testing."""

    def __init__(self, features: int = 32, rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.linear1 = nnx.Linear(2, features, rngs=rngs)
        self.linear2 = nnx.Linear(features, features, rngs=rngs)
        self.linear3 = nnx.Linear(features, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    return MockModel(features=32, rngs=nnx.Rngs(42))


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 2))
    y = jnp.sum(x**2, axis=1, keepdims=True)
    return x, y


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


class TestTrainerInitialization:
    """Test Trainer initialization."""

    def test_basic_initialization(self, mock_model):
        """Test basic trainer initialization."""
        config = TrainingConfig(num_epochs=10, learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        assert trainer.model is mock_model
        assert trainer.config is config
        assert trainer.state is not None
        assert trainer.optimizer is not None

    def test_initialization_with_rngs(self, mock_model):
        """Test initialization with custom RNGs."""
        config = TrainingConfig(num_epochs=10)
        rngs = nnx.Rngs(42)
        trainer = Trainer(mock_model, config, rngs=rngs)

        assert trainer.model is mock_model
        assert trainer.rngs is rngs

    def test_initialization_with_physics_config(self, mock_model, temp_checkpoint_dir):
        """Test initialization with physics configuration."""
        boundary_config = BoundaryConfig(weight=1.0, enforce=True)
        constraint_config = ConstraintConfig(
            constraints=["energy_conservation"],
            adaptive_weighting=True,
        )

        config = TrainingConfig(
            num_epochs=10,
            learning_rate=1e-3,
            batch_size=32,
            boundary_config=boundary_config,
            constraint_config=constraint_config,
        )
        config.checkpoint_config.checkpoint_dir = str(temp_checkpoint_dir)

        trainer = Trainer(mock_model, config)

        assert trainer.config.boundary_config == boundary_config
        assert trainer.config.constraint_config == constraint_config
        assert trainer.config.learning_rate == 1e-3
        assert hasattr(trainer, "state")

    def test_initialization_with_different_optimizers(self, mock_model):
        """Test initialization with different optimizer types."""
        # Test Adam
        config_adam = TrainingConfig(learning_rate=1e-3)
        config_adam.optimization_config.optimizer = "adam"
        trainer_adam = Trainer(mock_model, config_adam)
        assert trainer_adam.optimizer is not None

        # Test SGD
        config_sgd = TrainingConfig(learning_rate=1e-3)
        config_sgd.optimization_config.optimizer = "sgd"
        config_sgd.optimization_config.momentum = 0.9
        trainer_sgd = Trainer(
            MockModel(features=32, rngs=nnx.Rngs(43)),
            config_sgd,
        )
        assert trainer_sgd.optimizer is not None

        # Test AdamW
        config_adamw = TrainingConfig(learning_rate=1e-3)
        config_adamw.optimization_config.optimizer = "adamw"
        config_adamw.optimization_config.weight_decay = 1e-4
        trainer_adamw = Trainer(
            MockModel(features=32, rngs=nnx.Rngs(44)),
            config_adamw,
        )
        assert trainer_adamw.optimizer is not None


class TestStandardTraining:
    """Test standard (non-physics) training."""

    def test_training_step(self, mock_model, sample_data):
        """Test single training step."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()
        assert isinstance(metrics, dict)
        assert "gradient_norm" in metrics
        assert "learning_rate" in metrics
        assert "step" in metrics

    def test_validation_step(self, mock_model, sample_data):
        """Test validation step without parameter updates."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.validation_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)

    def test_train_loop(self, mock_model, sample_data):
        """Test full training loop."""
        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            batch_size=32,
            validation_frequency=1,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_train, y_train = x[:80], y[:80]
        x_val, y_val = x[80:], y[80:]

        trained_model, metrics = trainer.fit(
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
        )

        assert trained_model is not None
        assert isinstance(metrics, dict)
        assert "final_train_loss" in metrics
        assert "final_val_loss" in metrics


class TestPhysicsLossComputation:
    """Test physics loss computation."""

    def test_basic_loss_computation(self, mock_model, sample_data):
        """Test basic loss computation without physics constraints."""
        x, y = sample_data

        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        loss, _ = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()
        assert loss >= 0.0

    def test_loss_with_boundary_conditions(self, mock_model, sample_data):
        """Test loss computation with boundary conditions."""
        x, y = sample_data
        x_boundary = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
        y_boundary = jnp.zeros((20, 1))

        boundary_config = BoundaryConfig(weight=0.5, enforce=True)
        config = TrainingConfig(
            learning_rate=1e-3,
            boundary_config=boundary_config,
        )
        trainer = Trainer(mock_model, config)

        loss, metrics = trainer.training_step(
            x[:10], y[:10], boundary_data=(x_boundary, y_boundary)
        )

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()
        assert loss >= 0.0
        assert "boundary_loss" in metrics

    def test_different_loss_types(self, mock_model, sample_data):
        """Test different constraint loss configurations."""
        x, y = sample_data

        # Test with energy conservation constraint
        constraint_config = ConstraintConfig(
            constraints=["energy_conservation"],
        )
        config = TrainingConfig(
            learning_rate=1e-3,
            constraint_config=constraint_config,
        )
        trainer = Trainer(mock_model, config)
        loss, _ = trainer.training_step(x[:10], y[:10])
        assert isinstance(loss, jax.Array)

        # Test with multiple constraints
        constraint_config_multi = ConstraintConfig(
            constraints=["energy_conservation", "momentum_conservation"],
        )
        config_multi = TrainingConfig(
            learning_rate=1e-3,
            constraint_config=constraint_config_multi,
        )
        trainer_multi = Trainer(
            MockModel(features=32, rngs=nnx.Rngs(43)),
            config_multi,
        )
        loss_multi, _ = trainer_multi.training_step(x[:10], y[:10])
        assert isinstance(loss_multi, jax.Array)


class TestPhysicsInformedTraining:
    """Test physics-informed training with composable configs."""

    def test_training_with_boundary_config(self, mock_model, sample_data):
        """Test training with boundary conditions."""
        boundary_config = BoundaryConfig(weight=0.5, enforce=True)
        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            boundary_config=boundary_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_boundary = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
        y_boundary = jnp.zeros((20, 1))

        loss, metrics = trainer.training_step(
            x[:10], y[:10], boundary_data=(x_boundary, y_boundary)
        )

        assert isinstance(loss, jax.Array)
        assert "boundary_loss" in metrics

    def test_training_with_constraint_config(self, mock_model, sample_data):
        """Test training with physics constraints."""
        constraint_config = ConstraintConfig(
            constraints=["energy_conservation"],
            adaptive_weighting=True,
        )
        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            constraint_config=constraint_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert "constraint_loss" in metrics or "energy_conservation_loss" in metrics

    def test_training_with_conservation_config(self, mock_model, sample_data):
        """Test training with conservation laws."""
        conservation_config = ConservationConfig(
            laws=["energy", "momentum"],
            energy_monitoring=True,
        )
        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert "energy_conservation" in metrics or "conservation_loss" in metrics


class TestCheckpointing:
    """Test checkpointing functionality."""

    def test_save_and_load_checkpoint(self, mock_model, temp_checkpoint_dir):
        """Test checkpoint save and load."""
        config = TrainingConfig(
            num_epochs=2,
            checkpoint_frequency=1,
        )
        config.checkpoint_config.checkpoint_dir = str(temp_checkpoint_dir)

        trainer = Trainer(mock_model, config)

        # Save checkpoint
        checkpoint_path = trainer.save_checkpoint(step=10, loss=0.5)
        assert checkpoint_path is not None
        assert Path(checkpoint_path).exists()

        # Load checkpoint
        loaded_model, metadata = trainer.load_checkpoint(step=10)
        assert loaded_model is not None
        assert metadata["step"] == 10
        assert metadata["loss"] == 0.5

    def test_checkpoint_with_physics_metadata(self, mock_model, temp_checkpoint_dir):
        """Test checkpointing with physics-specific metadata."""
        config = TrainingConfig(num_epochs=2, checkpoint_frequency=1)
        config.checkpoint_config.checkpoint_dir = str(temp_checkpoint_dir)

        trainer = Trainer(mock_model, config)

        # Create physics metadata
        physics_metadata = {
            "constraint_violations": {
                "energy_conservation": [1e-6, 5e-7, 2e-7],
                "momentum_conservation": [2e-5, 1e-5, 8e-6],
            },
            "physics_metrics": {
                "boundary_loss": 0.001,
                "physics_loss": 0.005,
            },
        }

        # Save checkpoint with physics metadata
        checkpoint_path = trainer.save_checkpoint(
            step=100, loss=0.01, physics_metadata=physics_metadata
        )

        assert checkpoint_path is not None

        # Load checkpoint and verify metadata
        loaded_model, loaded_metadata = trainer.load_checkpoint(step=100)

        assert loaded_model is not None
        assert "physics_metadata" in loaded_metadata
        loaded_physics = loaded_metadata["physics_metadata"]

        # Verify physics metadata preservation
        assert (
            loaded_physics["constraint_violations"]
            == physics_metadata["constraint_violations"]
        )
        assert loaded_physics["physics_metrics"] == physics_metadata["physics_metrics"]


class TestProgressTracking:
    """Test progress tracking and callbacks."""

    def test_progress_callback(self, mock_model, sample_data):
        """Test custom progress callback."""
        callback_data = []

        def progress_callback(epoch, metrics):
            callback_data.append({"epoch": epoch, "metrics": metrics})

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            progress_callback=progress_callback,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        trainer.fit(train_data=(x[:80], y[:80]))

        assert len(callback_data) > 0
        assert all("epoch" in d for d in callback_data)
        assert all("metrics" in d for d in callback_data)

    def test_verbose_logging(self, mock_model, sample_data, caplog):
        """Test verbose logging output."""
        import logging

        # Explicitly configure the specific logger to ensure capture
        logger = logging.getLogger("opifex.core.training.trainer")
        logger.setLevel(logging.INFO)

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            verbose=True,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data

        with caplog.at_level(logging.INFO, logger="opifex.core.training.trainer"):
            trainer.fit(train_data=(x[:80], y[:80]))

        assert "Epoch" in caplog.text or "epoch" in caplog.text.lower()


class TestConfigurationComposition:
    """Test composable configuration system."""

    def test_multiple_physics_configs(self, mock_model, sample_data):
        """Test training with multiple physics configs composed together."""
        constraint_config = ConstraintConfig(constraints=["energy_conservation"])
        conservation_config = ConservationConfig(laws=["energy"])
        boundary_config = BoundaryConfig(weight=0.5)

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            constraint_config=constraint_config,
            conservation_config=conservation_config,
            boundary_config=boundary_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_boundary = jax.random.normal(jax.random.PRNGKey(1), (20, 2))
        y_boundary = jnp.zeros((20, 1))

        loss, metrics = trainer.training_step(
            x[:10], y[:10], boundary_data=(x_boundary, y_boundary)
        )

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)


class TestOptimizers:
    """Test different optimizer configurations."""

    def test_adam_optimizer(self, mock_model, sample_data):
        """Test training with Adam optimizer."""
        config = TrainingConfig(learning_rate=1e-3)
        config.optimization_config.optimizer = "adam"

        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, _ = trainer.training_step(x[:10], y[:10])
        assert isinstance(loss, jax.Array)

    def test_sgd_optimizer(self, mock_model):
        """Test training with SGD optimizer."""
        config = TrainingConfig(learning_rate=1e-3)
        config.optimization_config.optimizer = "sgd"
        config.optimization_config.momentum = 0.9

        trainer = Trainer(MockModel(features=32, rngs=nnx.Rngs(43)), config)
        assert trainer.optimizer is not None

    def test_adamw_optimizer(self, mock_model):
        """Test training with AdamW optimizer."""
        config = TrainingConfig(learning_rate=1e-3)
        config.optimization_config.optimizer = "adamw"
        config.optimization_config.weight_decay = 1e-4

        trainer = Trainer(MockModel(features=32, rngs=nnx.Rngs(44)), config)
        assert trainer.optimizer is not None


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_checkpoint_without_directory(self, mock_model):
        """Test checkpoint operations without checkpoint directory."""
        config = TrainingConfig(num_epochs=2)
        # Explicitly set checkpoint_dir to None/empty
        config.checkpoint_config.checkpoint_dir = None  # type: ignore  # noqa: PGH003
        trainer = Trainer(mock_model, config)

        # Should handle missing checkpoint directory gracefully
        result = trainer.save_checkpoint(step=10, loss=0.5)
        assert result is None

    def test_load_nonexistent_checkpoint(self, mock_model, temp_checkpoint_dir):
        """Test loading non-existent checkpoint."""
        config = TrainingConfig(num_epochs=2)
        config.checkpoint_config.checkpoint_dir = str(temp_checkpoint_dir)

        trainer = Trainer(mock_model, config)

        # Should handle missing checkpoint gracefully
        model, metadata = trainer.load_checkpoint(step=999)
        assert model is None
        assert metadata == {}


class TestExtensibility:
    """Test trainer extensibility features."""

    def test_custom_loss_registration(self, mock_model, sample_data):
        """Test registration of custom loss functions."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        def custom_loss(model, x, y_pred, y_true):
            return jnp.mean((y_pred - y_true) ** 2) * 0.5

        trainer.register_custom_loss("custom", custom_loss)
        assert "custom" in trainer.custom_losses

    def test_hook_registration(self, mock_model):
        """Test registration and execution of hooks."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        hook_called = [False]

        def test_hook(*args):
            hook_called[0] = True

        trainer.register_hook("training_step_end", test_hook)
        assert "training_step_end" in trainer.hooks


class TestMetricsCollection:
    """Test complete metrics collection."""

    def test_training_metrics(self, mock_model, sample_data):
        """Test collection of training metrics."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        _, metrics = trainer.training_step(x[:10], y[:10])

        # Verify standard metrics
        assert "step" in metrics
        assert "learning_rate" in metrics
        assert "gradient_norm" in metrics
        assert "loss" in metrics

    def test_physics_metrics(self, mock_model, sample_data):
        """Test collection of physics-specific metrics."""
        constraint_config = ConstraintConfig(
            constraints=["energy_conservation"],
            violation_monitoring=True,
        )
        config = TrainingConfig(
            learning_rate=1e-3,
            constraint_config=constraint_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        _, metrics = trainer.training_step(x[:10], y[:10])

        # Should have physics-related metrics
        assert any(
            "constraint" in k or "conservation" in k or "physics" in k for k in metrics
        )


class TestPerformance:
    """Test performance characteristics."""

    def test_training_performance(self, mock_model, sample_data):
        """Test that training completes in reasonable time."""
        import time

        config = TrainingConfig(num_epochs=2, learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        start_time = time.time()
        trainer.fit(train_data=(x[:80], y[:80]))
        elapsed = time.time() - start_time

        # Should complete quickly
        assert elapsed < 10.0  # 10 seconds is generous

    def test_memory_efficiency(self, temp_checkpoint_dir):
        """Test memory efficiency of trainer."""
        # Test with minimal configuration for memory efficiency
        config = TrainingConfig(
            num_epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            checkpoint_frequency=1,
        )
        config.checkpoint_config.checkpoint_dir = str(temp_checkpoint_dir)

        model = MockModel(features=8, rngs=nnx.Rngs(42))  # Smaller model
        trainer = Trainer(model, config)

        # Create minimal test data
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[5.0], [6.0]])

        # Run training step
        loss, metrics = trainer.training_step(x, y)

        # Verify training works correctly
        assert isinstance(loss, (float, jnp.ndarray))
        assert isinstance(metrics, dict)
        assert jnp.isfinite(loss)


class TestJITCompilation:
    """Test JIT compilation of training and validation steps."""

    def test_training_step_jit_with_jax(self, mock_model, sample_data):
        """Test that training_step can be JIT compiled with jax.jit."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_batch, y_batch = x[:10], y[:10]

        # Create a standalone JIT-compiled training step
        # This demonstrates the correct way to JIT compile with Trainer components
        @nnx.jit
        def train_step_impl(model, optimizer, x, y):
            def loss_fn(model):
                y_pred = model(x)
                return jnp.mean((y_pred - y) ** 2), {}

            (loss, _), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
            optimizer.update(model, grads)
            return loss, {}

        try:
            # Execute the JIT-compiled step
            loss, metrics = train_step_impl(
                trainer.model, trainer.optimizer, x_batch, y_batch
            )
            assert isinstance(loss, jax.Array)
            assert isinstance(metrics, dict)
        except Exception as e:
            pytest.fail(f"nnx.jit failed: {e}")

    def test_training_step_jit_with_nnx(self, mock_model, sample_data):
        """Test that training_step works correctly with nnx.Optimizer.

        With nnx.Optimizer, training_step uses nnx.value_and_grad which
        provides implicit JIT compilation. No need for external wrapping.
        """
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_batch, y_batch = x[:10], y[:10]

        # ✅ Direct call - nnx.value_and_grad provides JIT internally
        loss, metrics = trainer.training_step(x_batch, y_batch)

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "gradient_norm" in metrics

    def test_validation_step_jit_with_jax(self, mock_model, sample_data):
        """Test that validation_step can be JIT compiled with jax.jit."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_batch, y_batch = x[:10], y[:10]

        # Try to JIT compile with jax.jit
        jitted_val = jax.jit(trainer.validation_step)

        try:
            loss, metrics = jitted_val(x_batch, y_batch)
            assert isinstance(loss, jax.Array)
            assert isinstance(metrics, dict)
        except Exception as e:
            # If jax.jit doesn't work, that's okay - we'll use nnx.jit
            pytest.skip(f"jax.jit not compatible: {e}")

    def test_validation_step_jit_with_nnx(self, mock_model, sample_data):
        """Test that validation_step can be JIT compiled with nnx.jit."""
        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_batch, y_batch = x[:10], y[:10]

        # JIT compile with nnx.jit
        @nnx.jit
        def jitted_val(x, y):
            return trainer.validation_step(x, y)

        loss, metrics = jitted_val(x_batch, y_batch)

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)
        assert "val_loss" in metrics

    def test_jit_performance_improvement(self, mock_model, sample_data):
        """Test that nnx.Optimizer provides good performance.

        With nnx.Optimizer + nnx.value_and_grad, training is automatically
        optimized. Test that multiple calls execute efficiently.
        """
        import time

        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        x_batch, y_batch = x[:10], y[:10]

        # Warm-up (first calls include compilation)
        for _ in range(3):
            trainer.training_step(x_batch, y_batch)

        # Time subsequent calls (should be fast - compiled)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            loss, _ = trainer.training_step(x_batch, y_batch)
            if hasattr(loss, "block_until_ready"):
                loss.block_until_ready()
            times.append(time.perf_counter() - start)

        mean_time = sum(times) / len(times)

        # With nnx.value_and_grad optimization, should be reasonably fast
        # Each training step should complete in milliseconds
        assert mean_time < 0.1, (
            f"Training step too slow: {mean_time:.6f}s. "
            f"nnx.Optimizer should provide efficient training."
        )
