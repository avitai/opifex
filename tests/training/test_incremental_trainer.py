"""Tests for IncrementalTrainer - Test-Driven Development.

This module contains comprehensive tests for the IncrementalTrainer class,
written first to define the expected behavior before implementation.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.training.incremental_trainer import IncrementalTrainer


class TestIncrementalTrainerInitialization:
    """Test IncrementalTrainer initialization."""

    def test_incremental_trainer_initialization(self):
        """Test that IncrementalTrainer initializes correctly."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )

        # Test
        trainer = IncrementalTrainer(model, rngs)

        # Verify
        assert isinstance(trainer, IncrementalTrainer)
        assert trainer.model is model
        assert trainer.current_modes == (4, 4)  # Default modes
        assert trainer.variance_threshold == 1.0  # Default threshold

    def test_incremental_trainer_initialization_with_custom_modes(self):
        """Test IncrementalTrainer initialization with custom mode configuration."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        # Test
        trainer = IncrementalTrainer(model, rngs)

        # Verify
        assert trainer.current_modes == (8, 8)
        assert trainer.variance_threshold == 1.0

    def test_incremental_trainer_initialization_invalid_model(self):
        """Test IncrementalTrainer initialization with invalid model."""
        # Setup
        rngs = nnx.Rngs(0)
        invalid_model = "not a model"

        # Test & Verify
        with pytest.raises(TypeError):
            IncrementalTrainer(invalid_model, rngs)  # type: ignore[arg-type]

    def test_incremental_trainer_initialization_invalid_rngs(self):
        """Test IncrementalTrainer initialization with invalid rngs."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )

        # Test & Verify
        with pytest.raises(TypeError):
            IncrementalTrainer(model, "invalid rngs")  # type: ignore[arg-type]


class TestIncrementalTrainerModeExpansion:
    """Test mode expansion functionality."""

    def test_should_expand_modes_with_high_variance(self):
        """Test that high gradient variance triggers mode expansion."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)
        trainer.variance_threshold = 0.1  # Lower threshold for testing

        # Create mock gradients with very high variance
        high_variance_grads = {
            "layer1": jnp.array([10.0, 0.1, 9.0, 0.2]),  # Very high variance
            "layer2": jnp.array([8.0, 0.05, 9.5, 0.15]),  # Very high variance
        }

        # Test
        should_expand = trainer.should_expand_modes(high_variance_grads)

        # Verify
        assert should_expand is True

    def test_should_expand_modes_with_low_variance(self):
        """Test that low gradient variance does not trigger mode expansion."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Create mock gradients with low variance
        low_variance_grads = {
            "layer1": jnp.array([0.5, 0.51, 0.49, 0.52]),  # Low variance
            "layer2": jnp.array([0.3, 0.31, 0.29, 0.32]),  # Low variance
        }

        # Test
        should_expand = trainer.should_expand_modes(low_variance_grads)

        # Verify
        assert should_expand is False

    def test_expand_modes_increases_model_complexity(self):
        """Test that expanding modes increases model complexity."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)
        initial_modes = trainer.current_modes

        # Test
        trainer.expand_modes((8, 8))

        # Verify
        assert trainer.current_modes == (8, 8)
        assert trainer.current_modes[0] > initial_modes[0]
        assert trainer.current_modes[1] > initial_modes[1]

    def test_expand_modes_invalid_modes_raises_error(self):
        """Test that invalid mode expansion raises appropriate errors."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Test & Verify
        with pytest.raises(ValueError, match=r".*greater than.*"):
            trainer.expand_modes((1, 1))  # Smaller than current modes

        with pytest.raises(ValueError, match=r".*negative.*"):
            trainer.expand_modes((-1, 2))  # Negative modes

    def test_expand_modes_same_modes_no_change(self):
        """Test that expanding to same modes doesn't change anything."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)
        initial_modes = trainer.current_modes

        # Test
        trainer.expand_modes(initial_modes)

        # Verify
        assert trainer.current_modes == initial_modes


class TestIncrementalTrainerTrainingStep:
    """Test training step functionality."""

    def test_train_step_returns_loss(self):
        """Test that train_step returns a valid loss value."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Create sample data
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.zeros((2, 1, 8, 8))

        # Test
        loss = trainer.train_step(x, y)

        # Verify
        assert isinstance(loss, (float, jnp.ndarray))
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_train_step_updates_model_parameters(self):
        """Test that train_step updates model parameters."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Store initial parameters (capture values, not variables)
        initial_state = nnx.state(model, nnx.Param)
        initial_params = jax.tree.map(lambda x: x, initial_state)

        # Create sample data
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.zeros((2, 1, 8, 8))  # Different from input to create loss

        # Test
        _ = trainer.train_step(x, y)

        # Verify parameters changed
        updated_state = nnx.state(model, nnx.Param)
        updated_params = jax.tree.map(lambda x: x, updated_state)

        # Check that at least one parameter changed
        params_changed = False

        def compare_params(initial, updated):
            """Recursively compare parameter trees."""
            initial_flat = jax.tree_util.tree_leaves(initial)
            updated_flat = jax.tree_util.tree_leaves(updated)

            for init_param, upd_param in zip(initial_flat, updated_flat, strict=False):
                if not jnp.allclose(init_param, upd_param, atol=1e-6):
                    return True
            return False

        params_changed = compare_params(initial_params, updated_params)
        assert params_changed, "Model parameters should have been updated"

    def test_train_step_with_invalid_input_shapes(self):
        """Test that train_step handles invalid input shapes appropriately."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Test with mismatched shapes
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((3, 1, 8, 8))  # Different batch size

        # Test & Verify (should handle gracefully or raise appropriate error)
        with pytest.raises((ValueError, TypeError)):
            trainer.train_step(x, y)

    def test_train_step_with_different_batch_sizes(self):
        """Test train_step with different batch sizes."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Test with different batch sizes
        x_small = jnp.ones((1, 1, 8, 8))
        y_small = jnp.zeros((1, 1, 8, 8))
        x_large = jnp.ones((4, 1, 8, 8))
        y_large = jnp.zeros((4, 1, 8, 8))

        # Test
        loss_small = trainer.train_step(x_small, y_small)
        loss_large = trainer.train_step(x_large, y_large)

        # Verify
        assert jnp.isfinite(loss_small)
        assert jnp.isfinite(loss_large)
        # Losses might be different due to batch size effects


class TestIncrementalTrainerGradientAnalysis:
    """Test gradient analysis functionality."""

    def test_gradient_variance_analysis_computation(self):
        """Test gradient variance analysis computation."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Create mock gradients
        mock_grads = {
            "layer1": jnp.array([1.0, 0.5, 0.2, 0.8]),
            "layer2": jnp.array([[0.3, 0.7], [0.9, 0.1]]),
        }

        # Test
        variance = trainer._compute_gradient_variance(mock_grads)

        # Verify
        assert isinstance(variance, (float, jnp.ndarray))
        assert jnp.isfinite(variance)
        assert variance >= 0.0

    def test_gradient_variance_threshold_logic(self):
        """Test gradient variance threshold logic."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)
        trainer.variance_threshold = 0.5  # Set custom threshold

        # Test with high variance (above threshold)
        high_variance = 2.0
        assert trainer._should_expand_based_on_variance(high_variance) is True

        # Test with low variance (below threshold)
        low_variance = 0.1
        assert trainer._should_expand_based_on_variance(low_variance) is False

    def test_gradient_analysis_with_empty_gradients(self):
        """Test gradient analysis with empty or zero gradients."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Test with empty gradients
        empty_grads = {}
        variance = trainer._compute_gradient_variance(empty_grads)
        assert variance == 0.0

        # Test with zero gradients
        zero_grads = {"layer": jnp.zeros((4,))}
        variance = trainer._compute_gradient_variance(zero_grads)
        assert variance == 0.0


class TestIncrementalTrainerIntegration:
    """Test integration of all IncrementalTrainer functionality."""

    def test_full_incremental_training_cycle(self):
        """Test a complete incremental training cycle."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=2,  # Start small
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)
        trainer.variance_threshold = 0.01  # Low threshold for testing

        # Training data
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.zeros((2, 1, 8, 8))

        # Initial training steps
        initial_modes = trainer.current_modes
        loss1 = trainer.train_step(x, y)
        loss2 = trainer.train_step(x, y)

        # Verify training progresses
        assert jnp.isfinite(loss1)
        assert jnp.isfinite(loss2)

        # Test mode expansion (manual since we can't predict exact gradients)
        trainer.expand_modes((4, 4))
        assert trainer.current_modes != initial_modes

    def test_multiple_mode_expansions(self):
        """Test multiple sequential mode expansions."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=2,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Sequential expansions
        trainer.expand_modes((4, 4))
        assert trainer.current_modes == (4, 4)

        trainer.expand_modes((8, 8))
        assert trainer.current_modes == (8, 8)

        trainer.expand_modes((16, 16))
        assert trainer.current_modes == (16, 16)

    def test_training_with_mode_expansion_sequence(self):
        """Test training with a planned sequence of mode expansions."""
        # Setup
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=2,
            num_layers=2,
            rngs=rngs,
        )
        trainer = IncrementalTrainer(model, rngs)

        # Training data
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.zeros((2, 1, 8, 8))

        # Planned training sequence
        mode_sequence = [(2, 2), (4, 4), (8, 8)]
        losses = []

        for modes in mode_sequence:
            if modes != trainer.current_modes:
                trainer.expand_modes(modes)

            # Train for a few steps at this mode level
            for _ in range(3):
                loss = trainer.train_step(x, y)
                losses.append(loss)

        # Verify all losses are finite
        assert all(jnp.isfinite(loss) for loss in losses)
        assert len(losses) == 9  # 3 modes × 3 steps each
