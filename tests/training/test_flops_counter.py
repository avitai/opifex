"""
Test suite for FlopsCounter class

This module contains comprehensive tests for the FlopsCounter class,
covering initialization, forward pass profiling, backward pass profiling,
model comparison, and training step profiling functionality.
"""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.training.flops_counter import FlopsCounter


class TestFlopsCounterInitialization:
    """Test FlopsCounter initialization functionality."""

    def test_flops_counter_initialization(self):
        """Test basic FlopsCounter initialization."""
        # Test default initialization
        counter = FlopsCounter()
        assert counter.enable_profiling is True

        # Test disabled profiling
        counter_disabled = FlopsCounter(enable_profiling=False)
        assert counter_disabled.enable_profiling is False


class TestFlopsCounterForwardPass:
    """Test forward pass FLOPS counting functionality."""

    def test_count_forward_flops_basic(self):
        """Test basic forward FLOPS counting."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test
        flops_info = counter.count_forward_flops(model, x)

        # Verify
        assert isinstance(flops_info, dict)
        assert "total_flops" in flops_info
        assert "timing" in flops_info
        assert "input_shape" in flops_info
        assert "input_size" in flops_info
        assert flops_info["total_flops"] > 0
        assert flops_info["timing"] >= 0.0

    def test_count_forward_flops_different_input_sizes(self):
        """Test forward FLOPS counting with different input sizes."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x_small = jnp.ones((1, 1, 4, 4))
        x_large = jnp.ones((4, 1, 16, 16))

        # Test
        flops_small = counter.count_forward_flops(model, x_small)
        flops_large = counter.count_forward_flops(model, x_large)

        # Verify
        assert flops_small["total_flops"] < flops_large["total_flops"]

    def test_count_forward_flops_different_models(self):
        """Test forward FLOPS counting with different model complexities."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        simple_model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=2,
            num_layers=1,
            rngs=rngs,
        )
        complex_model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=64,
            modes=8,
            num_layers=4,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test
        simple_flops = counter.count_forward_flops(simple_model, x)
        complex_flops = counter.count_forward_flops(complex_model, x)

        # Verify
        assert simple_flops["total_flops"] < complex_flops["total_flops"]

    def test_count_forward_flops_invalid_model(self):
        """Test forward FLOPS counting with invalid model."""
        # Setup
        counter = FlopsCounter()
        invalid_model = "not a model"
        x = jnp.ones((2, 1, 8, 8))

        # Test & Verify
        with pytest.raises(TypeError):
            counter.count_forward_flops(invalid_model, x)  # type: ignore[arg-type]

    def test_count_forward_flops_invalid_input(self):
        """Test forward FLOPS counting with invalid input."""
        # Setup
        counter = FlopsCounter()
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
        with pytest.raises(AttributeError):
            counter.count_forward_flops(model, "invalid input")  # type: ignore[arg-type]


class TestFlopsCounterBackwardPass:
    """Test backward pass FLOPS counting functionality."""

    def test_count_backward_flops_basic(self):
        """Test basic backward FLOPS counting."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        flops_info = counter.count_backward_flops(model, x, y)

        # Verify
        assert isinstance(flops_info, dict)
        assert "total_flops" in flops_info
        assert "forward_flops" in flops_info
        assert "backward_flops" in flops_info
        assert "timing" in flops_info
        assert flops_info["total_flops"] > 0
        assert flops_info["backward_flops"] > flops_info["forward_flops"]

    def test_count_backward_flops_includes_forward_flops(self):
        """Test that backward FLOPS includes forward FLOPS."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        forward_flops = counter.count_forward_flops(model, x)
        backward_flops = counter.count_backward_flops(model, x, y)

        # Verify
        assert backward_flops["forward_flops"] == forward_flops["total_flops"]
        assert backward_flops["total_flops"] > forward_flops["total_flops"]

    def test_count_backward_flops_different_input_sizes(self):
        """Test backward FLOPS counting with different input sizes."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x_small = jnp.ones((1, 1, 4, 4))
        y_small = jnp.ones((1, 1, 4, 4))
        x_large = jnp.ones((4, 1, 16, 16))
        y_large = jnp.ones((4, 1, 16, 16))

        # Test
        flops_small = counter.count_backward_flops(model, x_small, y_small)
        flops_large = counter.count_backward_flops(model, x_large, y_large)

        # Verify
        assert flops_small["total_flops"] < flops_large["total_flops"]

    def test_count_backward_flops_mismatched_shapes(self):
        """Test backward FLOPS counting with mismatched input/target shapes."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 4, 4))  # Different shape

        # Test - should work (just counting FLOPs, not executing)
        flops_info = counter.count_backward_flops(model, x, y)

        # Verify
        assert flops_info["total_flops"] > 0

    def test_count_backward_flops_invalid_inputs(self):
        """Test backward FLOPS counting with invalid inputs."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test & Verify
        with pytest.raises(AttributeError):
            counter.count_backward_flops(model, x, "invalid target")  # type: ignore[arg-type]


class TestFlopsCounterModelComparison:
    """Test model comparison functionality."""

    def test_compare_models_basic(self):
        """Test basic model comparison functionality."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model1 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=2,
            num_layers=1,
            rngs=rngs,
        )
        model2 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        models = [model1, model2]
        x = jnp.ones((2, 1, 8, 8))

        # Test
        comparison = counter.compare_models(models, x)  # type: ignore[arg-type]

        # Verify
        assert isinstance(comparison, dict)
        assert "models" in comparison
        assert "comparison" in comparison
        assert len(comparison["models"]) == 2
        assert (
            comparison["models"][0]["total_flops"]
            < comparison["models"][1]["total_flops"]
        )

    def test_compare_models_multiple_models(self):
        """Test model comparison with multiple models."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        models = []
        for i in range(3):
            model = FourierNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=16 * (i + 1),
                modes=2 * (i + 1),
                num_layers=i + 1,
                rngs=rngs,
            )
            models.append(model)
        x = jnp.ones((2, 1, 8, 8))

        # Test
        comparison = counter.compare_models(models, x)  # type: ignore[arg-type]

        # Verify
        assert len(comparison["models"]) == 3
        # Models should have increasing FLOPS
        flops = [m["total_flops"] for m in comparison["models"]]
        assert flops[0] < flops[1] < flops[2]

    def test_compare_models_includes_relative_metrics(self):
        """Test that model comparison includes relative metrics."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model1 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=16,
            modes=2,
            num_layers=1,
            rngs=rngs,
        )
        model2 = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        models = [model1, model2]
        x = jnp.ones((2, 1, 8, 8))

        # Test
        comparison = counter.compare_models(models, x)  # type: ignore[arg-type]

        # Verify
        assert "relative_flops" in comparison["models"][0]
        assert "relative_flops" in comparison["models"][1]
        assert comparison["models"][0]["relative_flops"] == 1.0  # Base model

    def test_compare_models_empty_list(self):
        """Test model comparison with empty model list."""
        # Setup
        counter = FlopsCounter()
        x = jnp.ones((2, 1, 8, 8))

        # Test & Verify
        with pytest.raises(ValueError, match=r".*empty.*"):
            counter.compare_models([], x)

    def test_compare_models_single_model(self):
        """Test model comparison with single model."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test
        comparison = counter.compare_models([model], x)

        # Verify
        assert len(comparison["models"]) == 1
        assert comparison["models"][0]["total_flops"] > 0

    def test_compare_models_invalid_model_in_list(self):
        """Test model comparison with invalid model in list."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        valid_model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        models = [valid_model, "invalid model"]
        x = jnp.ones((2, 1, 8, 8))

        # Test & Verify
        with pytest.raises(TypeError):
            counter.compare_models(models, x)


class TestFlopsCounterTrainingStepProfiling:
    """Test training step profiling functionality."""

    def test_profile_training_step_basic(self):
        """Test basic training step profiling."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        profile = counter.profile_training_step(model, x, y)

        # Verify
        assert isinstance(profile, dict)
        assert "total_flops" in profile
        assert "forward_flops" in profile
        assert "backward_flops" in profile
        assert "update_flops" in profile
        assert "timing" in profile
        assert profile["total_flops"] > 0

    def test_profile_training_step_breakdown(self):
        """Test training step profiling includes breakdown."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        profile = counter.profile_training_step(model, x, y)

        # Verify
        assert "breakdown" in profile
        breakdown = profile["breakdown"]
        assert "forward_percentage" in breakdown
        assert "backward_percentage" in breakdown
        assert "update_percentage" in breakdown

        # Percentages should sum to ~100%
        total_percentage = (
            breakdown["forward_percentage"]
            + breakdown["backward_percentage"]
            + breakdown["update_percentage"]
        )
        assert abs(total_percentage - 100.0) < 1.0

    def test_profile_training_step_different_batch_sizes(self):
        """Test training step profiling with different batch sizes."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x_small = jnp.ones((1, 1, 8, 8))
        y_small = jnp.ones((1, 1, 8, 8))
        x_large = jnp.ones((8, 1, 8, 8))
        y_large = jnp.ones((8, 1, 8, 8))

        # Test
        profile_small = counter.profile_training_step(model, x_small, y_small)
        profile_large = counter.profile_training_step(model, x_large, y_large)

        # Verify
        assert profile_small["total_flops"] < profile_large["total_flops"]

    def test_profile_training_step_invalid_inputs(self):
        """Test training step profiling with invalid inputs."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test & Verify
        with pytest.raises(AttributeError):
            counter.profile_training_step(model, x, "invalid target")  # type: ignore[arg-type]


class TestFlopsCounterIntegration:
    """Test integration and consistency across different methods."""

    def test_forward_backward_consistency(self):
        """Test consistency between forward and backward FLOPS."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        forward_flops = counter.count_forward_flops(model, x)
        backward_flops = counter.count_backward_flops(model, x, y)

        # Verify
        assert backward_flops["forward_flops"] == forward_flops["total_flops"]

    def test_training_step_vs_individual_counts(self):
        """Test training step FLOPS vs individual forward/backward counts."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))
        y = jnp.ones((2, 1, 8, 8))

        # Test
        forward_flops = counter.count_forward_flops(model, x)
        backward_flops = counter.count_backward_flops(model, x, y)
        training_profile = counter.profile_training_step(model, x, y)

        # Verify
        assert training_profile["forward_flops"] == forward_flops["total_flops"]
        assert training_profile["backward_flops"] == backward_flops["backward_flops"]

    def test_model_comparison_vs_individual_counts(self):
        """Test model comparison vs individual FLOPS counts."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test
        individual_flops = counter.count_forward_flops(model, x)
        comparison = counter.compare_models([model], x)

        # Verify
        assert individual_flops["total_flops"] == comparison["models"][0]["total_flops"]

    def test_flops_counter_with_different_precisions(self):
        """Test FLOPS counter with different floating point precisions."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x_32 = jnp.ones((2, 1, 8, 8), dtype=jnp.float32)
        x_64 = jnp.ones((2, 1, 8, 8), dtype=jnp.float64)

        # Test
        flops_32 = counter.count_forward_flops(model, x_32)
        flops_64 = counter.count_forward_flops(model, x_64)

        # Verify (FLOPS should be similar regardless of precision)
        assert abs(flops_32["total_flops"] - flops_64["total_flops"]) < 100

    def test_flops_counter_reproducibility(self):
        """Test that FLOPS counting is reproducible."""
        # Setup
        counter = FlopsCounter()
        rngs = nnx.Rngs(0)
        model = FourierNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            modes=4,
            num_layers=2,
            rngs=rngs,
        )
        x = jnp.ones((2, 1, 8, 8))

        # Test
        flops_1 = counter.count_forward_flops(model, x)
        flops_2 = counter.count_forward_flops(model, x)

        # Verify
        assert flops_1["total_flops"] == flops_2["total_flops"]
