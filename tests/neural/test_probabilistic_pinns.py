"""Full tests for Probabilistic Physics-Informed Neural Networks."""

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
from opifex.uncertainty.objectives import ObjectiveConfig, UQLossComponents


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

        predictions, uncertainties, info = mf_pinn.adaptive_prediction(x, uncertainty_threshold=0.1)

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
        """RobustPINNOptimizer wraps the PINN; no instance-stored weights or rngs."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        assert optimizer.model is pinn

    def test_compute_loss_components_returns_uq_components(self):
        """compute_loss_components returns UQLossComponents from a batch."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )

        assert isinstance(components, UQLossComponents)
        assert components.data is not None
        assert components.physics_residual is not None
        assert components.regularization is not None
        assert components.total.shape == ()

    def test_compute_loss_components_with_boundary_conditions(self):
        """Boundary-condition batches populate the boundary component."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(with_bc=True),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )

        assert components.boundary is not None
        assert isinstance(components.total, jax.Array)

    def test_compute_robustness_penalty(self):
        """The internal robustness penalty is a finite scalar."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        x = random.normal(random.PRNGKey(42), (5, 2))

        penalty = optimizer._compute_robustness_penalty(x, noise_scale=0.01, rngs=nnx.Rngs(11))

        assert isinstance(penalty, jax.Array)
        assert penalty.shape == ()

    def test_robustness_penalty_different_noise_scales(self):
        """The penalty stays finite across noise scales."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        x = random.normal(random.PRNGKey(42), (5, 2))

        penalty1 = optimizer._compute_robustness_penalty(x, noise_scale=0.001, rngs=nnx.Rngs(11))
        penalty2 = optimizer._compute_robustness_penalty(x, noise_scale=0.1, rngs=nnx.Rngs(11))

        assert isinstance(penalty1, jax.Array)
        assert isinstance(penalty2, jax.Array)

    def test_uncertainty_guided_sampling(self):
        """Uncertainty-guided sampling returns the requested number of points."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        x_candidates = random.normal(random.PRNGKey(42), (20, 2))

        selected_points = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=5, rngs=nnx.Rngs(13), uncertainty_threshold=0.1
        )

        assert selected_points.shape == (5, 2)

    def test_uncertainty_guided_sampling_different_thresholds(self):
        """Threshold variation preserves output shape."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)
        x_candidates = random.normal(random.PRNGKey(42), (50, 2))

        points1 = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, rngs=nnx.Rngs(13), uncertainty_threshold=0.01
        )
        points2 = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, rngs=nnx.Rngs(13), uncertainty_threshold=1.0
        )

        assert points1.shape == (10, 2)
        assert points2.shape == (10, 2)


def _make_robust_objective(**overrides: float | str | None) -> ObjectiveConfig:
    base: dict[str, float | str | None] = {
        "kl_weight": 1.0,
        "dataset_size": 32,
        "physics_weight": 1.0,
        "data_weight": 1.0,
        "boundary_weight": 1.0,
        "initial_condition_weight": 1.0,
        "regularization_weight": 0.1,
        "calibration_weight": 1.0,
        "conformal_weight": 1.0,
        "pac_bayes_weight": 1.0,
    }
    base.update(overrides)
    return ObjectiveConfig(**base)  # type: ignore[arg-type]


def _make_robust_batch(
    *,
    seed: int = 42,
    n: int = 5,
    with_residual: bool = True,
    with_bc: bool = False,
) -> dict:
    key = random.PRNGKey(seed)
    k_x, k_y = random.split(key)
    batch: dict = {
        "x": random.normal(k_x, (n, 2)),
        "y_true": random.normal(k_y, (n, 16)),
    }
    if with_residual:
        batch["pde_residual_fn"] = lambda _x_in, y_p: jnp.sum(y_p**2, axis=-1)
    if with_bc:
        batch["boundary_conditions"] = {"type": "dirichlet", "value": 0.0}
    return batch


class TestRobustPINNOptimizerSharedObjective:
    """Task 3.3: ``RobustPINNOptimizer.compute_loss_components`` returns
    :class:`UQLossComponents` built from a shared :class:`ObjectiveConfig`,
    requires caller-owned ``rngs`` at the method boundary, and never falls
    back to a hidden fixed key for uncertainty-guided sampling."""

    def test_compute_loss_components_returns_uq_loss_components(self) -> None:
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        batch = _make_robust_batch()
        objective = _make_robust_objective()

        components = optimizer.compute_loss_components(batch, rngs=nnx.Rngs(7), objective=objective)
        assert isinstance(components, UQLossComponents)
        assert jnp.isfinite(components.total)
        assert components.total.shape == ()

    def test_compute_loss_components_preserves_data_term(self) -> None:
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )
        assert components.data is not None
        assert jnp.isfinite(components.data)

    def test_compute_loss_components_preserves_physics_term(self) -> None:
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(with_residual=True),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )
        assert components.physics_residual is not None
        assert jnp.isfinite(components.physics_residual)

    def test_compute_loss_components_preserves_regularization_term(self) -> None:
        """The robustness penalty maps to the ``regularization`` slot."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )
        assert components.regularization is not None
        assert jnp.isfinite(components.regularization)

    def test_compute_loss_components_preserves_kl_term(self) -> None:
        """``ProbabilisticPINN.kl_divergence`` is threaded into the KL slot."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )
        assert components.kl is not None
        expected_kl = float(pinn.kl_divergence())
        assert float(components.kl) == pytest.approx(expected_kl, rel=1e-6, abs=1e-6)

    def test_compute_loss_components_preserves_boundary_term(self) -> None:
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        components = optimizer.compute_loss_components(
            _make_robust_batch(with_bc=True),
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(),
        )
        assert components.boundary is not None
        assert jnp.isfinite(components.boundary)

    def test_compute_loss_components_requires_rngs_at_method_boundary(self) -> None:
        """``rngs`` is required keyword-only; no hidden fallback to instance state."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        with pytest.raises(TypeError):
            optimizer.compute_loss_components(  # type: ignore[call-arg]
                _make_robust_batch(), objective=_make_robust_objective()
            )

    def test_compute_loss_components_total_scales_with_objective_weights(self) -> None:
        """``total`` scales with ``ObjectiveConfig`` weights (shared semantics)."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        batch = _make_robust_batch()
        cmp_low = optimizer.compute_loss_components(
            batch,
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(
                data_weight=0.1, physics_weight=0.1, regularization_weight=0.0, kl_weight=0.0
            ),
        )
        cmp_high = optimizer.compute_loss_components(
            batch,
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(
                data_weight=10.0, physics_weight=10.0, regularization_weight=10.0, kl_weight=0.0
            ),
        )
        assert float(cmp_high.total) > float(cmp_low.total)

    def test_uncertainty_guided_sampling_requires_rngs_no_hidden_fixed_key(self) -> None:
        """No hidden ``nnx.Rngs(0)`` fallback — ``rngs`` must be supplied per call."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16), rngs=nnx.Rngs(0))
        optimizer = RobustPINNOptimizer(model=pinn)
        x_candidates = random.normal(random.PRNGKey(11), (20, 2))
        with pytest.raises(TypeError):
            optimizer.uncertainty_guided_sampling(  # type: ignore[call-arg]
                x_candidates, num_samples=5
            )


class TestIntegration:
    """Integration tests for probabilistic PINNs."""

    def test_complete_training_workflow(self):
        """End-to-end training step builds a finite ``UQLossComponents.total``."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x_train = random.normal(key, (10, 2))
        y_train = random.normal(key, (10, 16))

        batch = {
            "x": x_train,
            "y_true": y_train,
            "pde_residual_fn": lambda _x, predictions: jnp.sum(predictions**2, axis=-1),
        }
        components = optimizer.compute_loss_components(
            batch, rngs=nnx.Rngs(7), objective=_make_robust_objective()
        )

        assert isinstance(components, UQLossComponents)
        assert components.data is not None
        assert components.physics_residual is not None
        assert components.regularization is not None
        assert jnp.isfinite(components.total)

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
        """Uncertainty-guided sampling composes with the shared loss surface."""
        pinn = ProbabilisticPINN(input_dim=2, hidden_dims=(32, 16))
        optimizer = RobustPINNOptimizer(model=pinn)

        key = random.PRNGKey(42)
        x_candidates = random.normal(key, (100, 2))
        y_train = random.normal(key, (10, 16))

        selected_points = optimizer.uncertainty_guided_sampling(
            x_candidates, num_samples=10, rngs=nnx.Rngs(13), uncertainty_threshold=0.1
        )

        batch = {
            "x": selected_points,
            "y_true": y_train,
            "pde_residual_fn": lambda _x, predictions: jnp.mean(predictions**2),
        }
        components = optimizer.compute_loss_components(
            batch,
            rngs=nnx.Rngs(7),
            objective=_make_robust_objective(regularization_weight=0.2),
        )

        assert isinstance(components.total, jax.Array)
        assert components.total.shape == ()

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
