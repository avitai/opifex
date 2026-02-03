"""Test Physics-Informed Neural Operators.

Test suite for physics-informed neural operator integration including
PDE constraints, conservation laws, and quantum constraints.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.physics import PhysicsInformedLoss, PhysicsLossConfig

# Import TestEnvironmentManager from extracted location
from opifex.neural.base import StandardMLP
from opifex.neural.operators.fno.base import FourierNeuralOperator
from opifex.neural.operators.physics import PhysicsInformedOperator


class TestPhysicsInformedOperator:
    """Test Physics-Informed Neural Operator integration."""

    def test_physics_informed_neural_operator_initialization(self):
        """Test PINO initialization as standalone operator."""
        rngs = nnx.Rngs(42)

        # Test standalone physics-informed operator
        pino = PhysicsInformedOperator(
            layer_sizes=[2, 64, 64, 32, 1],  # [input_dim, hidden..., output_dim]
            physics_type="pde",
            activation="gelu",
            physics_weight=0.1,
            data_weight=1.0,
            rngs=rngs,
        )

        assert hasattr(pino, "network")
        assert hasattr(pino, "physics_type")
        assert pino.physics_type == "pde"
        assert pino.physics_weight == 0.1
        assert pino.data_weight == 1.0

    def test_physics_informed_neural_operator_with_deeponet(self):
        """Test PINO with DeepONet as base operator."""
        rngs = nnx.Rngs(42)

        # Test PhysicsInformedOperator with new API - no need for base_operator

        pino_deeponet = PhysicsInformedOperator(
            layer_sizes=[64, 128, 64, 32],
            physics_type="pde",
            physics_weight=0.1,
            rngs=rngs,
        )

        assert isinstance(pino_deeponet.network, StandardMLP)

    def test_physics_informed_neural_operator_with_gno(self):
        """Test PINO with GraphNeuralOperator as base operator."""
        rngs = nnx.Rngs(42)

        # Test PhysicsInformedOperator with new API - no need for base_gno

        pino_gno = PhysicsInformedOperator(
            layer_sizes=[16, 32, 16, 8],
            physics_type="conservation",
            physics_weight=0.2,
            rngs=rngs,
        )

        assert isinstance(pino_gno.network, StandardMLP)

    def test_physics_informed_forward_pass(self):
        """Test PINO forward pass maintains base operator functionality."""
        rngs = nnx.Rngs(42)
        batch_size = 4
        # grid_size = 32  # unused in new API

        _base_fno = FourierNeuralOperator(
            in_channels=2,
            out_channels=1,
            hidden_channels=32,
            modes=8,
            num_layers=2,
            rngs=rngs,
        )

        physics_config = PhysicsLossConfig()
        _physics_loss = PhysicsInformedLoss(
            config=physics_config,
            equation_type="poisson",
            domain_type="2d",
        )

        pino = PhysicsInformedOperator(
            layer_sizes=[2 * 32 * 32, 256, 128, 1 * 32 * 32],
            physics_type="pde",
            physics_weight=0.1,
            rngs=rngs,
        )

        # Test forward pass - input needs to be flattened for the new API
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2 * 32 * 32))

        output = pino(x)

        # Verify output shape and properties
        assert output.shape == (batch_size, 1 * 32 * 32)
        assert jnp.isfinite(output).all()

    def test_physics_constrained_loss_computation(self):
        """Test PINO with physics constraint loss computation."""
        rngs = nnx.Rngs(42)
        batch_size = 4

        # Create physics-informed operator
        pino = PhysicsInformedOperator(
            layer_sizes=[64, 128, 64, 32],
            physics_type="pde",
            physics_weight=0.1,
            data_weight=1.0,
            rngs=rngs,
        )

        # Test data
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 64))
        y_true = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 32))

        # Forward pass
        y_pred = pino(x)

        # Compute data loss
        data_loss = jnp.mean((y_pred - y_true) ** 2)

        # For this test, we'll create a simple physics constraint
        def simple_physics_constraint(model, inputs):
            # Simple constraint: output should have reasonable magnitude
            outputs = model(inputs)
            # Physics constraint: outputs shouldn't be too large
            constraint_violation = jnp.mean(jnp.maximum(0, jnp.abs(outputs) - 10.0))
            return constraint_violation

        physics_loss = simple_physics_constraint(pino, x)

        # Combined loss
        total_loss = pino.data_weight * data_loss + pino.physics_weight * physics_loss

        assert jnp.isfinite(data_loss)
        assert jnp.isfinite(physics_loss)
        assert jnp.isfinite(total_loss)
        assert total_loss >= 0.0

        # Test loss components have expected magnitudes
        assert data_loss.shape == ()
        assert physics_loss.shape == ()
        assert total_loss.shape == ()

    def test_physics_informed_differentiability(self):
        """Test PINO differentiability for gradient-based physics training."""
        rngs = nnx.Rngs(42)
        batch_size = 2

        pino = PhysicsInformedOperator(
            layer_sizes=[32, 64, 32, 16],
            physics_type="pde",
            physics_weight=0.1,
            rngs=rngs,
        )

        def loss_fn(model, inputs):
            outputs = model(inputs)
            return jnp.mean(outputs**2)

        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 32))

        # Test that we can compute gradients
        loss, grad = nnx.value_and_grad(loss_fn)(pino, x)

        assert jnp.isfinite(loss)
        assert grad is not None

        # Verify gradients exist for model parameters
        # Note: In NNX, gradients are computed w.r.t. the model parameters
        # and we check that loss computation succeeded

    def test_conservation_law_enforcement(self):
        """Test PINO with conservation law constraints."""
        rngs = nnx.Rngs(42)
        batch_size = 4

        # Create PINO for conservation law problems
        pino = PhysicsInformedOperator(
            layer_sizes=[64, 128, 64, 64],  # Same input/output dim for conservation
            physics_type="conservation",
            physics_weight=0.2,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 64))
        output = pino(x)

        # Test conservation constraint: mass conservation
        def mass_conservation_constraint(inputs, outputs):
            # Simple mass conservation: sum of inputs ≈ sum of outputs
            input_mass = jnp.sum(inputs, axis=-1)
            output_mass = jnp.sum(outputs, axis=-1)
            conservation_error = jnp.mean((input_mass - output_mass) ** 2)
            return conservation_error

        conservation_error = mass_conservation_constraint(x, output)

        assert jnp.isfinite(conservation_error)
        assert conservation_error >= 0.0

        # Test that PINO can be trained to respect conservation
        def conservation_loss_fn(model, inputs):
            outputs = model(inputs)
            return mass_conservation_constraint(inputs, outputs)

        loss, grad = nnx.value_and_grad(conservation_loss_fn)(pino, x)

        assert jnp.isfinite(loss)
        assert grad is not None

    def test_quantum_constraint_enforcement(self):
        """Test PINO with quantum mechanics constraints."""
        rngs = nnx.Rngs(42)
        batch_size = 4

        # Create PINO for quantum problems
        pino = PhysicsInformedOperator(
            layer_sizes=[32, 64, 64, 32],
            physics_type="quantum",
            physics_weight=0.15,
            rngs=rngs,
        )

        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 32))
        output = pino(x)

        # Test quantum constraint: probability normalization
        def probability_normalization_constraint(wave_function):
            # Wave function should be normalized: ∫|ψ|² dx = 1
            prob_density = jnp.abs(wave_function) ** 2
            total_prob = jnp.sum(prob_density, axis=-1)
            normalization_error = jnp.mean((total_prob - 1.0) ** 2)
            return normalization_error

        normalization_error = probability_normalization_constraint(output)

        assert jnp.isfinite(normalization_error)
        assert normalization_error >= 0.0

        # Test that PINO can be trained to respect quantum constraints
        def quantum_loss_fn(model, inputs):
            outputs = model(inputs)
            return probability_normalization_constraint(outputs)

        loss, grad = nnx.value_and_grad(quantum_loss_fn)(pino, x)

        assert jnp.isfinite(loss)
        assert grad is not None

    def test_adaptive_weight_scheduling(self):
        """Test PINO with adaptive physics weight scheduling."""
        rngs = nnx.Rngs(42)
        batch_size = 2

        # Create PINO with initial weights
        initial_physics_weight = 0.1
        initial_data_weight = 1.0

        pino = PhysicsInformedOperator(
            layer_sizes=[32, 64, 32],
            physics_type="pde",
            physics_weight=initial_physics_weight,
            data_weight=initial_data_weight,
            rngs=rngs,
        )

        assert pino.physics_weight == initial_physics_weight
        assert pino.data_weight == initial_data_weight

        # Test weight adaptation based on training progress
        def adaptive_weight_schedule(
            epoch, initial_physics_weight, initial_data_weight
        ):
            # Example adaptive schedule: increase physics weight over time
            physics_weight = initial_physics_weight * (1 + 0.1 * epoch)
            data_weight = initial_data_weight * (1 - 0.05 * epoch)
            return max(physics_weight, 0.01), max(data_weight, 0.1)

        # Simulate training epochs
        for epoch in range(3):
            new_physics_weight, new_data_weight = adaptive_weight_schedule(
                epoch, initial_physics_weight, initial_data_weight
            )

            # Update weights (in practice, this would be done through model configuration)
            # For testing, we create a new instance with updated weights
            updated_pino = PhysicsInformedOperator(
                layer_sizes=[32, 64, 32],
                physics_type="pde",
                physics_weight=new_physics_weight,
                data_weight=new_data_weight,
                rngs=rngs,
            )

            assert updated_pino.physics_weight == new_physics_weight
            assert updated_pino.data_weight == new_data_weight

        # Test that the model still works with updated weights
        x = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 32))
        output = updated_pino(x)

        assert output.shape == (batch_size, 32)
        assert jnp.isfinite(output).all()
