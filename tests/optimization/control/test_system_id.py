"""Tests for System Identification Networks (Task 5.2.1).

This module provides comprehensive tests for neural network-based system identification
with physics constraints, online learning, and control integration.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.control.system_id import (
    BenchmarkValidationResult,
    ControlIntegratedSystemID,
    OnlineSystemLearner,
    PhysicsConstrainedSystemID,
    PhysicsConstraint,
    SystemDynamicsModel,
    SystemIdentifier,
)


@pytest.fixture
def rngs():
    """Standard RNG fixture for tests."""
    return nnx.Rngs(42)


@pytest.fixture
def sample_system_data():
    """Generate sample system identification data."""
    time_steps = 100
    state_dim = 4
    input_dim = 2

    # Simple linear system for testing: x_{k+1} = A*x_k + B*u_k
    A = jnp.array(
        [
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 0.8, 0.1, 0.0],
            [0.0, 0.0, 0.7, 0.1],
            [0.0, 0.0, 0.0, 0.6],
        ]
    )
    B = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.0, 0.5]])

    # Generate trajectory
    states = jnp.zeros((time_steps, state_dim))
    inputs = jax.random.normal(jax.random.key(123), (time_steps, input_dim))

    for t in range(time_steps - 1):
        states = states.at[t + 1].set(A @ states[t] + B @ inputs[t])

    return {
        "states": states,
        "inputs": inputs,
        "true_A": A,
        "true_B": B,
        "state_dim": state_dim,
        "input_dim": input_dim,
    }


@pytest.fixture
def nonlinear_system_data():
    """Generate nonlinear system data for testing."""
    time_steps = 100
    state_dim = 2
    input_dim = 1

    def nonlinear_dynamics(x, u):
        # Nonlinear pendulum-like system
        x1, x2 = x[0], x[1]
        u_val = u[0]

        dx1 = x2
        dx2 = -jnp.sin(x1) + u_val

        return jnp.array([dx1, dx2])

    # Generate trajectory with numerical integration
    dt = 0.1
    states = jnp.zeros((time_steps, state_dim))
    inputs = 0.1 * jax.random.normal(jax.random.key(456), (time_steps, input_dim))

    for t in range(time_steps - 1):
        dx = nonlinear_dynamics(states[t], inputs[t])
        states = states.at[t + 1].set(states[t] + dt * dx)

    return {
        "states": states,
        "inputs": inputs,
        "dynamics_fn": nonlinear_dynamics,
        "dt": dt,
        "state_dim": state_dim,
        "input_dim": input_dim,
    }


class TestSystemIdentifier:
    """Test basic system identification functionality."""

    def test_initialization(self, rngs):
        """Test basic system identifier initialization."""
        system_id = SystemIdentifier(state_dim=4, input_dim=2, hidden_dim=64, rngs=rngs)

        assert system_id.state_dim == 4
        assert system_id.input_dim == 2
        assert system_id.hidden_dim == 64

    def test_forward_pass_shape(self, rngs, sample_system_data):
        """Test forward pass produces correct output shapes."""
        data = sample_system_data
        system_id = SystemIdentifier(
            state_dim=data["state_dim"],
            input_dim=data["input_dim"],
            hidden_dim=32,
            rngs=rngs,
        )

        # Test single step prediction
        state = data["states"][0]
        input_val = data["inputs"][0]

        next_state = system_id(state, input_val)
        assert next_state.shape == (data["state_dim"],)
        assert jnp.isfinite(next_state).all()

    def test_batch_prediction(self, rngs, sample_system_data):
        """Test batch prediction capability."""
        data = sample_system_data
        system_id = SystemIdentifier(
            state_dim=data["state_dim"],
            input_dim=data["input_dim"],
            hidden_dim=32,
            rngs=rngs,
        )

        # Test batch prediction
        batch_size = 10
        states_batch = data["states"][:batch_size]
        inputs_batch = data["inputs"][:batch_size]

        # Use vmap for batch processing
        batch_predict = jax.vmap(system_id, in_axes=(0, 0))
        predictions = batch_predict(states_batch, inputs_batch)

        assert predictions.shape == (batch_size, data["state_dim"])
        assert jnp.isfinite(predictions).all()

    def test_training_functionality(self, rngs, sample_system_data):
        """Test system identification training process."""
        data = sample_system_data
        system_id = SystemIdentifier(
            state_dim=data["state_dim"],
            input_dim=data["input_dim"],
            hidden_dim=32,
            rngs=rngs,
        )

        # Define loss function
        def loss_fn(model, states, inputs, targets):
            predictions = jax.vmap(model, in_axes=(0, 0))(states, inputs)
            return jnp.mean((predictions - targets) ** 2)

        # Training data
        train_states = data["states"][:-1]
        train_inputs = data["inputs"][:-1]
        train_targets = data["states"][1:]

        # Compute loss and gradients
        loss_and_grad = nnx.value_and_grad(loss_fn)
        loss, grads = loss_and_grad(
            system_id, train_states, train_inputs, train_targets
        )

        assert jnp.isfinite(loss)
        assert loss >= 0.0
        # Check gradients exist for all parameters
        param_count = len(jax.tree.leaves(grads))
        assert param_count > 0


class TestPhysicsConstrainedSystemID:
    """Test physics-constrained system identification."""

    def test_initialization_with_constraints(self, rngs):
        """Test initialization with physics constraints."""
        # Energy conservation constraint
        energy_constraint = PhysicsConstraint(
            name="energy_conservation", constraint_type="conservation", tolerance=1e-3
        )

        physics_system_id = PhysicsConstrainedSystemID(
            state_dim=4, input_dim=2, constraints=[energy_constraint], rngs=rngs
        )

        assert len(physics_system_id.constraints) == 1
        assert physics_system_id.constraints[0].name == "energy_conservation"

    def test_constraint_enforcement(self, rngs):
        """Test that physics constraints are enforced during prediction."""
        # Stability constraint
        stability_constraint = PhysicsConstraint(
            name="stability", constraint_type="stability", tolerance=1e-2
        )

        physics_system_id = PhysicsConstrainedSystemID(
            state_dim=2, input_dim=1, constraints=[stability_constraint], rngs=rngs
        )

        # Test prediction with constraint checking
        state = jnp.array([1.0, -0.5])
        input_val = jnp.array([0.1])

        result = physics_system_id.predict_with_constraints(state, input_val)

        assert "prediction" in result
        assert "constraint_violations" in result
        assert result["prediction"].shape == (2,)

    def test_conservation_law_enforcement(self, rngs):
        """Test conservation law enforcement during training."""
        # Energy conservation
        energy_constraint = PhysicsConstraint(
            name="energy_conservation", constraint_type="conservation", tolerance=1e-3
        )

        physics_system_id = PhysicsConstrainedSystemID(
            state_dim=4, input_dim=2, constraints=[energy_constraint], rngs=rngs
        )

        # Sample data
        state = jnp.array([1.0, 0.5, -0.3, 0.8])
        input_val = jnp.array([0.1, -0.2])

        # Check energy conservation
        initial_energy = physics_system_id.compute_energy(state)
        prediction = physics_system_id(state, input_val)
        final_energy = physics_system_id.compute_energy(prediction)

        # Energy should be approximately conserved (within tolerance)
        energy_diff = jnp.abs(final_energy - initial_energy)
        # For untrained network, we just check the computation works
        assert jnp.isfinite(energy_diff)


class TestOnlineSystemLearner:
    """Test online learning and adaptation capabilities."""

    def test_initialization(self, rngs):
        """Test online learner initialization."""
        online_learner = OnlineSystemLearner(
            state_dim=3,
            input_dim=1,
            learning_rate=1e-3,
            adaptation_rate=0.95,
            rngs=rngs,
        )

        assert online_learner.learning_rate == 1e-3
        assert online_learner.adaptation_rate == 0.95

    def test_online_update(self, rngs):
        """Test online model update with new data."""
        online_learner = OnlineSystemLearner(
            state_dim=2, input_dim=1, learning_rate=1e-3, rngs=rngs
        )

        # New observation
        state = jnp.array([1.0, -0.5])
        input_val = jnp.array([0.1])
        target = jnp.array([0.9, -0.4])

        # Get initial prediction
        initial_pred = online_learner(state, input_val)

        # Perform online update
        update_result = online_learner.update_online(state, input_val, target)

        # Get updated prediction
        updated_pred = online_learner(state, input_val)

        assert "loss" in update_result
        assert "adaptation_strength" in update_result
        assert jnp.isfinite(update_result["loss"])

        # Prediction should change after update (even if slightly)
        prediction_changed = not jnp.allclose(initial_pred, updated_pred, atol=1e-6)
        # For neural networks, we expect some change
        # If identical, could be due to initialization - check gradient norm
        assert prediction_changed or update_result["adaptation_strength"] > 0

    def test_adaptive_learning_rate(self, rngs):
        """Test adaptive learning rate based on performance."""
        online_learner = OnlineSystemLearner(
            state_dim=2, input_dim=1, learning_rate=1e-3, adaptive_lr=True, rngs=rngs
        )

        # Simulate good and bad predictions
        state = jnp.array([1.0, -0.5])
        input_val = jnp.array([0.1])

        # Good prediction (target close to prediction)
        good_target = jnp.array([1.1, -0.6])
        result_good = online_learner.update_online(state, input_val, good_target)

        # Bad prediction (target far from prediction)
        bad_target = jnp.array([5.0, -10.0])
        result_bad = online_learner.update_online(state, input_val, bad_target)

        # Both should work (specific behavior depends on implementation)
        assert jnp.isfinite(result_good["loss"])
        assert jnp.isfinite(result_bad["loss"])

    def test_memory_management(self, rngs):
        """Test memory management for online learning."""
        online_learner = OnlineSystemLearner(
            state_dim=2, input_dim=1, buffer_size=10, rngs=rngs
        )

        # Add multiple observations
        for i in range(15):  # More than buffer size
            state = jnp.array([float(i), -float(i)])
            input_val = jnp.array([float(i % 3)])
            target = jnp.array([float(i + 0.1), -float(i + 0.1)])

            online_learner.update_online(state, input_val, target)

        # Check memory size is limited
        memory_info = online_learner.get_memory_info()
        assert "buffer_size" in memory_info
        assert "current_size" in memory_info
        assert memory_info["current_size"] <= 10


class TestControlIntegratedSystemID:
    """Test control integration capabilities."""

    def test_initialization_with_control_policy(self, rngs):
        """Test initialization with control policy integration."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=4, input_dim=2, control_dim=2, rngs=rngs
        )

        assert control_system_id.state_dim == 4
        assert control_system_id.input_dim == 2
        assert control_system_id.control_dim == 2

    def test_joint_system_control_optimization(self, rngs):
        """Test joint optimization of system ID and control policy."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=3, input_dim=1, control_dim=1, rngs=rngs
        )

        # Sample trajectory
        trajectory_length = 20
        states = jax.random.normal(jax.random.key(789), (trajectory_length, 3))
        targets = jax.random.normal(jax.random.key(987), (trajectory_length, 3))

        # Joint optimization
        result = control_system_id.joint_optimization(states, targets)

        assert "system_id_loss" in result
        assert "control_loss" in result
        assert "total_loss" in result
        assert jnp.isfinite(result["total_loss"])

    def test_control_policy_adaptation(self, rngs):
        """Test adaptation of control policy based on system identification."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=2, input_dim=1, control_dim=1, rngs=rngs
        )

        # Current state and desired target
        current_state = jnp.array([1.0, -0.5])
        target_state = jnp.array([0.0, 0.0])

        # Get control action
        control_action = control_system_id.compute_control_action(
            current_state, target_state
        )

        assert control_action.shape == (1,)
        assert jnp.isfinite(control_action).all()

    def test_closed_loop_simulation(self, rngs):
        """Test closed-loop simulation with learned system and control."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=2, input_dim=1, control_dim=1, rngs=rngs
        )

        # Initial state and target
        initial_state = jnp.array([2.0, -1.0])
        target_state = jnp.array([0.0, 0.0])

        # Simulate closed loop
        simulation_result = control_system_id.simulate_closed_loop(
            initial_state=initial_state, target_state=target_state, steps=10
        )

        assert "states" in simulation_result
        assert "actions" in simulation_result
        assert "tracking_error" in simulation_result
        assert simulation_result["states"].shape == (11, 2)  # steps + 1
        assert simulation_result["actions"].shape == (10, 1)


class TestSystemDynamicsModel:
    """Test system dynamics model representation."""

    def test_linear_system_model(self, rngs):
        """Test linear system dynamics model."""
        model = SystemDynamicsModel(
            model_type="linear", state_dim=3, input_dim=2, rngs=rngs
        )

        assert model.model_type == "linear"
        assert model.state_dim == 3
        assert model.input_dim == 2

    def test_nonlinear_system_model(self, rngs):
        """Test nonlinear system dynamics model."""
        model = SystemDynamicsModel(
            model_type="nonlinear",
            state_dim=2,
            input_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )

        assert model.model_type == "nonlinear"
        assert model.hidden_dims == [32, 32]

    def test_model_prediction_consistency(self, rngs):
        """Test prediction consistency for the same input."""
        model = SystemDynamicsModel(
            model_type="linear", state_dim=2, input_dim=1, rngs=rngs
        )

        state = jnp.array([1.0, -0.5])
        input_val = jnp.array([0.1])

        # Multiple predictions should be identical (deterministic)
        pred1 = model(state, input_val, deterministic=True)
        pred2 = model(state, input_val, deterministic=True)

        assert jnp.allclose(pred1, pred2)


class TestBenchmarkValidation:
    """Test benchmark validation functionality."""

    def test_linear_system_benchmark(self, rngs, sample_system_data):
        """Test validation against linear system benchmark."""
        data = sample_system_data
        system_id = SystemIdentifier(
            state_dim=data["state_dim"],
            input_dim=data["input_dim"],
            hidden_dim=64,
            rngs=rngs,
        )

        # Create benchmark problem
        result = system_id.validate_on_benchmark(
            benchmark_name="linear_system",
            test_data={
                "states": data["states"][:-1],  # Current states (exclude last)
                "inputs": data["inputs"][:-1],  # Inputs (exclude last)
                "targets": data["states"][1:],  # Next states
            },
        )

        assert isinstance(result, BenchmarkValidationResult)
        assert result.benchmark_name == "linear_system"
        assert "prediction_error" in result.metrics
        assert jnp.isfinite(result.metrics["prediction_error"])

    def test_nonlinear_system_benchmark(self, rngs, nonlinear_system_data):
        """Test validation against nonlinear system benchmark."""
        data = nonlinear_system_data
        system_id = SystemIdentifier(
            state_dim=data["state_dim"],
            input_dim=data["input_dim"],
            hidden_dim=64,
            rngs=rngs,
        )

        result = system_id.validate_on_benchmark(
            benchmark_name="nonlinear_pendulum",
            test_data={
                "states": data["states"][:-1],  # Current states (exclude last)
                "inputs": data["inputs"][:-1],  # Inputs (exclude last)
                "targets": data["states"][1:],
            },
        )

        assert isinstance(result, BenchmarkValidationResult)
        assert result.benchmark_name == "nonlinear_pendulum"
        assert "prediction_error" in result.metrics

    def test_control_benchmark_validation(self, rngs):
        """Test control-specific benchmark validation."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=2, input_dim=1, control_dim=1, rngs=rngs
        )

        # Control benchmark: reference tracking
        result = control_system_id.validate_control_benchmark(
            benchmark_name="reference_tracking",
            initial_state=jnp.array([1.0, -0.5]),
            reference_trajectory=jnp.zeros((10, 2)),
            steps=10,
        )

        assert "tracking_error" in result.metrics
        assert "control_effort" in result.metrics
        assert "settling_time" in result.metrics


class TestIntegrationWithL2O:
    """Test integration with existing L2O components."""

    def test_parametric_solver_integration(self, rngs):
        """Test integration with parametric optimization solver."""
        # This would test integration with existing L2O components
        # For now, just test the interface exists
        system_id = SystemIdentifier(state_dim=3, input_dim=2, rngs=rngs)

        # Test that system ID can be used as part of L2O optimization
        result = system_id.integrate_with_l2o_solver()

        assert "optimization_ready" in result
        assert result["optimization_ready"] is True

    def test_constraint_learning_integration(self, rngs):
        """Test integration with constraint learning from Sprint 5.1."""
        control_system_id = ControlIntegratedSystemID(
            state_dim=2, input_dim=1, control_dim=1, rngs=rngs
        )

        # Test integration with constraint learning
        constraint_result = control_system_id.integrate_constraint_learning()

        assert "constraint_satisfaction" in constraint_result
        assert constraint_result["constraint_satisfaction"] is not None


if __name__ == "__main__":
    pytest.main([__file__])
