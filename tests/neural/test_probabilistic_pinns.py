"""Comprehensive tests for Probabilistic Physics-Informed Neural Networks."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from jax import random

from opifex.neural.bayesian.probabilistic_pinns import (
    MultiFidelityPINN,
    ProbabilisticPINN,
    RobustPINNOptimizer,
)


def assert_array_shape(obj, expected_shape, name="array"):
    """Type-safe shape assertion helper."""
    if isinstance(obj, jax.Array):
        assert obj.shape == expected_shape, (
            f"{name} shape mismatch: expected {expected_shape}, got {obj.shape}"
        )
    elif isinstance(obj, dict) and "prediction" in obj:
        pred = obj["prediction"]
        if isinstance(pred, jax.Array):
            assert pred.shape == expected_shape, f"{name} prediction shape mismatch"
        else:
            pytest.fail(f"Expected Array in prediction, got {type(pred)}")
    else:
        pytest.fail(f"Expected Array or dict with prediction, got {type(obj)}")


class TestProbabilisticPINN:
    """Test ProbabilisticPINN class."""

    def test_probabilistic_pinn_init_default(self):
        """Test ProbabilisticPINN initialization with default parameters."""
        pinn = ProbabilisticPINN(input_dim=2)

        assert pinn.physics_loss_weight == 1.0
        assert pinn.uncertainty_weight == 0.1
        assert hasattr(pinn, "layers")
        assert hasattr(pinn, "output_layer")
        assert hasattr(pinn, "use_bayesian")
        # assert hasattr(pinn, "physics_priors") # Removed as it was None anyway

    def test_probabilistic_pinn_init_custom(self):
        """Test ProbabilisticPINN initialization with custom parameters."""
        key = random.PRNGKey(123)
        rngs = nnx.Rngs(params=key, dropout=random.split(key)[0])

        pinn = ProbabilisticPINN(
            input_dim=3,
            hidden_dims=(128, 64, 32),
            physics_loss_weight=2.0,
            uncertainty_weight=0.2,
            rngs=rngs,
        )

        assert pinn.physics_loss_weight == 2.0
        assert pinn.uncertainty_weight == 0.2

    def test_probabilistic_pinn_forward_pass(self):
        """Test forward pass through ProbabilisticPINN."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        # Create test input
        key = random.PRNGKey(42)
        x = random.normal(key, (10, 2))

        # Forward pass
        output = pinn(x)

        assert output.shape[0] == 10  # Batch size
        assert len(output.shape) == 2  # 2D output

    def test_predict_with_uncertainty(self):
        """Test uncertainty quantification prediction."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        # Create test input
        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        # Predict with uncertainty
        result = pinn.predict_with_uncertainty(x, num_samples=10)

        assert "mean" in result
        assert "std" in result
        assert "confidence_95_lower" in result
        assert "confidence_95_upper" in result

        # Check shapes
        assert result["mean"].shape[0] == 5
        assert result["std"].shape[0] == 5
        assert result["confidence_95_lower"].shape[0] == 5
        assert result["confidence_95_upper"].shape[0] == 5

    def test_physics_loss_basic(self):
        """Test basic physics loss computation."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        # Create test input
        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        # Simple PDE residual function (heat equation)
        def pde_residual_fn(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)  # Simplified residual

        # Compute physics loss
        loss = pinn.physics_loss(x, pde_residual_fn)

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()  # Scalar loss

    def test_physics_loss_with_boundary_conditions(self):
        """Test physics loss with boundary conditions."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        def pde_residual_fn(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)

        boundary_conditions = {"dirichlet": {"value": 0.0, "location": "boundary"}}

        loss = pinn.physics_loss(x, pde_residual_fn, boundary_conditions)

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()

    def test_robust_loss(self):
        """Test robust loss computation."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))
        y_true = random.normal(key, (5, 16))  # Match output dimension

        loss = pinn.robust_loss(x, y_true, noise_scale=0.01)

        assert isinstance(loss, jax.Array)
        assert loss.shape == ()

    def test_robust_loss_different_noise_scales(self):
        """Test robust loss with different noise scales."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))
        y_true = random.normal(key, (5, 16))

        loss1 = pinn.robust_loss(x, y_true, noise_scale=0.001)
        loss2 = pinn.robust_loss(x, y_true, noise_scale=0.1)

        # Higher noise should generally lead to higher loss
        assert isinstance(loss1, jax.Array)
        assert isinstance(loss2, jax.Array)


class TestMultiFidelityPINN:
    """Test MultiFidelityPINN class."""

    def test_multifidelity_pinn_init_default(self):
        """Test MultiFidelityPINN initialization with default parameters."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        assert hasattr(mf_pinn, "low_fidelity_layers")
        assert hasattr(mf_pinn, "high_fidelity_networks")
        # assert hasattr(mf_pinn, "fusion_network") # Removed as unused

    def test_multifidelity_pinn_init_custom(self):
        """Test MultiFidelityPINN initialization with custom parameters."""
        rngs = nnx.Rngs(42)

        mf_pinn = MultiFidelityPINN(
            input_dim=3,
            low_fidelity_dims=(16, 16),
            high_fidelity_dims=(64, 32),
            fusion_dims=(24,),
            rngs=rngs,
        )

        assert hasattr(mf_pinn, "low_fidelity_layers")
        assert hasattr(mf_pinn, "high_fidelity_networks")
        # assert hasattr(mf_pinn, "fusion_network")

    def test_multifidelity_forward_low_fidelity(self):
        """Test forward pass with low fidelity."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        output = mf_pinn(x, fidelity_level="low")

        assert isinstance(output, dict)
        assert "low_fidelity_pred" in output
        pred = output["low_fidelity_pred"]
        # Type-safe shape assertion
        if isinstance(pred, jax.Array):
            assert pred.shape[0] == 5
            assert len(pred.shape) == 2
        else:
            pytest.fail(f"Expected jax.Array, got {type(pred)}")

    def test_multifidelity_forward_high_fidelity(self):
        """Test forward pass with high fidelity."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        output = mf_pinn(x, fidelity_level="high")

        assert isinstance(output, dict)
        assert "prediction" in output or "high_fidelity_pred" in output

    def test_multifidelity_forward_fusion(self):
        """Test forward pass with adaptive fidelity."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        output = mf_pinn(x, fidelity_level="adaptive")

        assert isinstance(output, dict)
        assert "prediction" in output
        pred = output["prediction"]
        # Type-safe shape assertion
        if isinstance(pred, jax.Array):
            assert pred.shape[0] == 5
            assert len(pred.shape) == 2
        else:
            pytest.fail(f"Expected jax.Array, got {type(pred)}")

    def test_fusion_prediction(self):
        """Test internal fusion prediction method."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (3, 2))

        # Test private method through public interface
        output = mf_pinn._fusion_prediction(x)

        assert output.shape[0] == 3
        assert len(output.shape) == 2

    def test_adaptive_prediction(self):
        """Test adaptive prediction based on uncertainty."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (3, 2))

        predictions, uncertainties, info = mf_pinn.adaptive_prediction(
            x, uncertainty_threshold=0.1
        )

        assert isinstance(predictions, jax.Array)
        assert isinstance(uncertainties, jax.Array)
        assert predictions.shape[0] == 3
        assert uncertainties.shape[0] == 3
        assert isinstance(info, dict)
        assert "high_fidelity_count" in info
        assert "low_fidelity_count" in info

    def test_adaptive_prediction_different_thresholds(self):
        """Test adaptive prediction with different uncertainty thresholds."""
        mf_pinn = MultiFidelityPINN(input_dim=2)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        # Low threshold (more high-fidelity)
        _, _, info1 = mf_pinn.adaptive_prediction(x, uncertainty_threshold=0.01)

        # High threshold (more low-fidelity)
        _, _, info2 = mf_pinn.adaptive_prediction(x, uncertainty_threshold=1.0)

        # With higher threshold, should use more low-fidelity
        assert info2["low_fidelity_count"] >= info1["low_fidelity_count"]


class TestRobustPINNOptimizer:
    """Test RobustPINNOptimizer class."""

    def test_robust_optimizer_init(self):
        """Test RobustPINNOptimizer initialization."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(
            model=pinn, learning_rate=1e-3, robustness_weight=0.1
        )

        assert optimizer.model == pinn
        assert optimizer.learning_rate == 1e-3
        assert optimizer.robustness_weight == 0.1

    def test_robust_optimizer_default_values(self):
        """Test RobustPINNOptimizer with default values."""
        pinn = ProbabilisticPINN(input_dim=2)
        optimizer = RobustPINNOptimizer(model=pinn)

        assert optimizer.learning_rate == 1e-3
        assert optimizer.robustness_weight == 0.1

    def test_compute_loss_components(self):
        """Test computation of loss components."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))
        y_true = random.normal(key, (5, 16))

        def simple_pde_residual(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)

        loss_components = optimizer.compute_loss_components(
            x, y_true, simple_pde_residual
        )

        assert isinstance(loss_components, dict)
        assert "data_loss" in loss_components
        assert "physics_loss" in loss_components
        assert "robustness_penalty" in loss_components
        assert "total_loss" in loss_components

    def test_compute_loss_components_with_boundary_conditions(self):
        """Test loss components with boundary conditions."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))
        y_true = random.normal(key, (5, 16))

        def simple_pde_residual(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)

        boundary_conditions = {"type": "dirichlet", "value": 0.0}

        loss_components = optimizer.compute_loss_components(
            x, y_true, simple_pde_residual, boundary_conditions
        )

        assert "total_loss" in loss_components
        assert isinstance(loss_components["total_loss"], jax.Array)

    def test_compute_robustness_penalty(self):
        """Test robustness penalty computation."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        penalty = optimizer._compute_robustness_penalty(x, noise_scale=0.01)

        assert isinstance(penalty, jax.Array)
        assert penalty.shape == ()

    def test_robustness_penalty_different_noise_scales(self):
        """Test robustness penalty with different noise scales."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        penalty1 = optimizer._compute_robustness_penalty(x, noise_scale=0.001)
        penalty2 = optimizer._compute_robustness_penalty(x, noise_scale=0.1)

        assert isinstance(penalty1, jax.Array)
        assert isinstance(penalty2, jax.Array)

    def test_uncertainty_guided_sampling(self):
        """Test uncertainty-guided sampling."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x_candidates = random.normal(key, (20, 2))

        selected_points = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=5, uncertainty_threshold=0.1
        )

        assert selected_points.shape[0] == 5
        assert selected_points.shape[1] == 2

    def test_uncertainty_guided_sampling_different_thresholds(self):
        """Test uncertainty-guided sampling with different thresholds."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x_candidates = random.normal(key, (50, 2))

        # Test with different thresholds
        points1 = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, uncertainty_threshold=0.01
        )
        points2 = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, uncertainty_threshold=1.0
        )

        assert points1.shape == (10, 2)
        assert points2.shape == (10, 2)


class TestIntegration:
    """Integration tests for probabilistic PINNs."""

    def test_complete_training_workflow(self):
        """Test complete training workflow with probabilistic PINN."""
        # Initialize PINN
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn, learning_rate=1e-3)

        # Generate training data
        key = random.PRNGKey(42)
        x_train = random.normal(key, (10, 2))
        y_train = random.normal(key, (10, 16))

        # Define PDE residual
        def heat_equation_residual(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)

        # Compute loss components
        loss_components = optimizer.compute_loss_components(
            x_train, y_train, heat_equation_residual
        )

        # Check all components are present
        assert "data_loss" in loss_components
        assert "physics_loss" in loss_components
        assert "robustness_penalty" in loss_components
        assert "total_loss" in loss_components

        # Test uncertainty prediction
        uncertainty_result = pinn.predict_with_uncertainty(x_train, num_samples=10)
        assert "mean" in uncertainty_result
        assert "std" in uncertainty_result

    def test_multifidelity_adaptive_workflow(self):
        """Test multifidelity PINN with adaptive prediction."""
        # Initialize multi-fidelity PINN
        mf_pinn = MultiFidelityPINN(
            input_dim=2,
            low_fidelity_dims=(16, 8),
            high_fidelity_dims=(64, 32),
            fusion_dims=(24,),
        )

        # Generate test data
        key = random.PRNGKey(42)
        x_test = random.normal(key, (15, 2))

        # Test different fidelity modes
        low_pred = mf_pinn(x_test, fidelity_level="low")
        high_pred = mf_pinn(x_test, fidelity_level="high")
        adaptive_pred = mf_pinn(x_test, fidelity_level="adaptive")

        # FIXED: Compare the actual prediction arrays from dicts
        low_array = low_pred["low_fidelity_pred"]
        high_array = high_pred.get("high_fidelity_pred", high_pred.get("prediction"))
        adaptive_array = adaptive_pred["prediction"]

        assert low_array.shape == adaptive_array.shape
        assert isinstance(high_array, jax.Array)

        # Test adaptive prediction
        adaptive_pred_result, uncertainties, info = mf_pinn.adaptive_prediction(
            x_test, uncertainty_threshold=0.1
        )

        # Type-safe shape assertions
        assert_array_shape(adaptive_pred_result, (15, 1), "adaptive_pred_result")
        assert_array_shape(uncertainties, (15, 1), "uncertainties")
        assert info["high_fidelity_count"] + info["low_fidelity_count"] == 15

    def test_robust_optimization_with_uncertainty_sampling(self):
        """Test robust optimization with uncertainty-guided sampling."""
        # Initialize system
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn, robustness_weight=0.2)

        # Generate large candidate set
        key = random.PRNGKey(42)
        x_candidates = random.normal(key, (100, 2))
        y_train = random.normal(key, (10, 16))

        # Select high-uncertainty points
        selected_points = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, uncertainty_threshold=0.1
        )

        # Define physics residual
        def simple_residual(x_input, predictions):
            return jnp.mean(predictions**2)

        # Compute loss on selected points
        loss_components = optimizer.compute_loss_components(
            selected_points, y_train, simple_residual
        )

        assert isinstance(loss_components["total_loss"], jax.Array)
        assert loss_components["total_loss"].shape == ()

    def test_error_handling_edge_cases(self):
        """Test error handling for edge cases."""
        # Test with very small input dimensions
        pinn = ProbabilisticPINN(input_dim=1, hidden_dims=(8,))

        key = random.PRNGKey(42)
        x = random.normal(key, (3, 1))

        # Should handle small dimensions gracefully
        output = pinn(x)
        assert output.shape[0] == 3

        # Test uncertainty prediction with few samples
        uncertainty_result = pinn.predict_with_uncertainty(x, num_samples=2)
        assert "mean" in uncertainty_result
        assert "std" in uncertainty_result

    def test_physics_loss_various_boundary_conditions(self):
        """Test physics loss with various boundary condition types."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))

        key = random.PRNGKey(42)
        x = random.normal(key, (5, 2))

        def simple_residual(x_input, predictions):
            return jnp.sum(predictions**2, axis=-1)

        # Test with different boundary condition formats
        bc_types = [
            None,
            {"type": "dirichlet"},
            {"type": "neumann", "value": 1.0},
            {"mixed": True, "values": [0.0, 1.0]},
        ]

        for bc in bc_types:
            loss = pinn.physics_loss(x, simple_residual, bc)
            assert isinstance(loss, jax.Array)
            assert loss.shape == ()
