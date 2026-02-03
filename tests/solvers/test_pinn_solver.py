"""Tests for PINNSolver.

This module tests the Physics-Informed Neural Network (PINN) solver implementation,
verifying the high-level API for solving PDEs with physics-informed neural networks.

The solver uses a composition pattern:
- PINNConfig composes PhysicsLossConfig for loss weights
- Residual functions are passed directly (factory functions like poisson_residual())
- No string keys or hidden registries
"""

import jax.numpy as jnp
from flax import nnx

from opifex.core.physics.losses import PhysicsLossConfig
from opifex.geometry import Interval
from opifex.geometry.csg import Rectangle
from opifex.neural.pinns import create_poisson_pinn, SimplePINN
from opifex.solvers.pinn import (
    PINNConfig,
    PINNResult,
    PINNSolver,
    poisson_residual,
)


class TestPINNConfig:
    """Test PINNConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PINNConfig()
        assert config.n_interior == 100
        assert config.n_boundary == 50
        assert config.num_iterations == 2000
        assert config.learning_rate == 1e-3
        assert config.print_every == 500
        assert config.seed == 42
        # Check composed PhysicsLossConfig defaults
        assert config.loss_config.physics_loss_weight == 1.0
        assert config.loss_config.boundary_loss_weight == 100.0
        assert config.loss_config.data_loss_weight == 0.0

    def test_custom_values(self):
        """Test custom configuration values."""
        custom_loss_config = PhysicsLossConfig(
            physics_loss_weight=2.0,
            boundary_loss_weight=50.0,
            data_loss_weight=1.0,
        )
        config = PINNConfig(
            n_interior=200,
            n_boundary=100,
            num_iterations=5000,
            learning_rate=1e-4,
            print_every=100,
            seed=123,
            loss_config=custom_loss_config,
        )
        assert config.n_interior == 200
        assert config.n_boundary == 100
        assert config.num_iterations == 5000
        assert config.learning_rate == 1e-4
        assert config.print_every == 100
        assert config.seed == 123
        assert config.loss_config.physics_loss_weight == 2.0
        assert config.loss_config.boundary_loss_weight == 50.0
        assert config.loss_config.data_loss_weight == 1.0


class TestPINNResult:
    """Test PINNResult dataclass."""

    def test_result_fields(self):
        """Test that PINNResult has expected fields."""
        model = SimplePINN(
            input_dim=1, output_dim=1, hidden_dims=[10], rngs=nnx.Rngs(0)
        )
        result = PINNResult(
            model=model,
            losses=[1.0, 0.5, 0.1],
            final_loss=0.1,
            training_time=1.5,
            metrics={"test_key": "test_value"},
        )
        assert result.model is model
        assert result.losses == [1.0, 0.5, 0.1]
        assert result.final_loss == 0.1
        assert result.training_time == 1.5
        assert result.metrics == {"test_key": "test_value"}


class TestPINNSolver:
    """Test PINNSolver implementation."""

    def test_initialization(self):
        """Test solver initialization with model."""
        model = SimplePINN(
            input_dim=2, output_dim=1, hidden_dims=[32], rngs=nnx.Rngs(0)
        )
        solver = PINNSolver(model=model)
        assert solver.model is model

    def test_solve_poisson_1d(self):
        """Test solve on 1D Poisson equation.

        Solves: -u''(x) = pi^2 * sin(pi*x) on [-1, 1]
        BCs: u(-1) = u(1) = 0
        Exact solution: u(x) = sin(pi*x)

        Uses poisson_residual() factory which handles the standard
        Poisson equation: -∇²u = f.
        """
        # Setup
        geometry = Interval(-1.0, 1.0)
        pinn = create_poisson_pinn(
            spatial_dim=1, hidden_dims=[50, 50, 50], rngs=nnx.Rngs(42)
        )

        # Source term f(x) = pi^2 * sin(pi*x)
        def source_fn(x):
            return jnp.pi**2 * jnp.sin(jnp.pi * x)

        # Create residual function using factory
        residual_fn = poisson_residual(source_fn)

        def bc_fn(x):
            return jnp.zeros_like(x[..., 0])

        # Solve with minimal iterations for test speed
        solver = PINNSolver(pinn)
        config = PINNConfig(
            n_interior=50,
            n_boundary=2,
            num_iterations=100,
            print_every=0,  # Suppress output
            seed=42,
        )
        result = solver.solve(geometry, residual_fn, bc_fn, config)

        # Verify result structure
        assert isinstance(result, PINNResult)
        assert result.model is pinn
        assert len(result.losses) == 100
        assert result.final_loss == result.losses[-1]
        assert result.training_time > 0
        assert "n_iterations" in result.metrics
        assert result.metrics["n_iterations"] == 100

    def test_solve_reduces_loss(self):
        """Test that solve actually reduces the loss."""
        geometry = Interval(-1.0, 1.0)
        pinn = create_poisson_pinn(
            spatial_dim=1, hidden_dims=[32, 32], rngs=nnx.Rngs(42)
        )

        def source_fn(x):
            return jnp.pi**2 * jnp.sin(jnp.pi * x)

        residual_fn = poisson_residual(source_fn)

        def bc_fn(x):
            return jnp.zeros_like(x[..., 0])

        solver = PINNSolver(pinn)
        config = PINNConfig(
            n_interior=50,
            n_boundary=2,
            num_iterations=200,
            print_every=0,
            seed=42,
        )
        result = solver.solve(geometry, residual_fn, bc_fn, config)

        # Loss should decrease during training
        assert result.final_loss < result.metrics["initial_loss"]

    def test_solve_with_custom_residual(self):
        """Test solve with user-provided residual function."""
        geometry = Interval(0.0, 1.0)
        model = SimplePINN(
            input_dim=1, output_dim=1, hidden_dims=[32], rngs=nnx.Rngs(0)
        )

        # Simple residual: u(x) - x^2 = 0 (trivial PDE where solution is x^2)
        def custom_residual_fn(model, x):
            u = model(x).squeeze(-1)
            target = x[..., 0] ** 2
            return u - target

        def bc_fn(x):
            # u(0) = 0, u(1) = 1
            return x[..., 0] ** 2

        solver = PINNSolver(model)
        config = PINNConfig(
            n_interior=50,
            n_boundary=10,
            num_iterations=50,
            print_every=0,
            seed=42,
        )
        result = solver.solve(geometry, custom_residual_fn, bc_fn, config)

        # Verify result structure
        assert isinstance(result, PINNResult)
        assert len(result.losses) == 50
        assert result.training_time > 0

    def test_solve_poisson_2d(self):
        """Test solve on 2D domain."""
        geometry = Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0)
        pinn = create_poisson_pinn(
            spatial_dim=2, hidden_dims=[32, 32], rngs=nnx.Rngs(42)
        )

        def source_fn(x):
            # f(x,y) = 2*pi^2 * sin(pi*x) * sin(pi*y)
            return (
                2
                * jnp.pi**2
                * jnp.sin(jnp.pi * x[..., 0])
                * jnp.sin(jnp.pi * x[..., 1])
            )

        residual_fn = poisson_residual(source_fn)

        def bc_fn(x):
            return jnp.zeros_like(x[..., 0])

        solver = PINNSolver(pinn)
        config = PINNConfig(
            n_interior=100,
            n_boundary=40,
            num_iterations=50,
            print_every=0,
            seed=42,
        )
        result = solver.solve(geometry, residual_fn, bc_fn, config)

        assert isinstance(result, PINNResult)
        assert len(result.losses) == 50

    def test_metrics_contain_expected_keys(self):
        """Test that result metrics contain all expected keys."""
        geometry = Interval(0.0, 1.0)
        model = SimplePINN(
            input_dim=1, output_dim=1, hidden_dims=[16], rngs=nnx.Rngs(0)
        )

        def residual_fn(model, x):
            return model(x).squeeze(-1)

        def bc_fn(x):
            return jnp.zeros_like(x[..., 0])

        solver = PINNSolver(model)
        config = PINNConfig(
            n_interior=30,
            n_boundary=10,
            num_iterations=5,
            print_every=0,
        )
        result = solver.solve(geometry, residual_fn, bc_fn, config)

        expected_keys = [
            "final_loss",
            "initial_loss",
            "training_time",
            "n_iterations",
            "n_interior",
            "n_boundary",
        ]
        for key in expected_keys:
            assert key in result.metrics, f"Missing key: {key}"

        assert result.metrics["n_interior"] == 30
        assert result.metrics["n_boundary"] == 10
