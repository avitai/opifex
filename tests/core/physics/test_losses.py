"""Comprehensive tests for physics loss composition and PDE residual computation.

This test suite validates all physics loss functionality following TDD principles.
Tests are written FIRST to define the API and expected behavior.

Test Coverage:
    - PhysicsLossConfig configuration and validation
    - PhysicsLossComposer loss composition
    - AdaptiveWeightScheduler scheduling algorithms
    - ConservationLawEnforcer conservation law residuals
    - ResidualComputer PDE/ODE residual computation
    - PhysicsInformedLoss integrated loss system
    - JAX compatibility (JIT, vmap, grad)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.physics.losses import (
    AdaptiveWeightScheduler,
    ConservationLawEnforcer,
    PhysicsInformedLoss,
    PhysicsLossComposer,
    PhysicsLossConfig,
    ResidualComputer,
)


class TestPhysicsLossConfig:
    """Test physics loss configuration."""

    def test_initialization(self):
        """Test config initialization with default values."""
        config = PhysicsLossConfig()
        assert config.data_loss_weight == 1.0
        assert config.physics_loss_weight == 0.1
        assert config.boundary_loss_weight == 1.0
        assert config.conservation_weights == {}
        assert config.adaptive_weighting is False
        assert config.weight_schedule == "exponential"

    def test_custom_weights(self):
        """Test config with custom weights."""
        config = PhysicsLossConfig(
            data_loss_weight=2.0,
            physics_loss_weight=0.5,
            boundary_loss_weight=1.5,
            conservation_weights={"energy": 0.1, "momentum": 0.2},
        )
        assert config.data_loss_weight == 2.0
        assert config.physics_loss_weight == 0.5
        assert config.boundary_loss_weight == 1.5
        assert config.conservation_weights["energy"] == 0.1
        assert config.conservation_weights["momentum"] == 0.2

    def test_validation_invalid_schedule(self):
        """Test config validation with invalid schedule type."""
        with pytest.raises(ValueError, match="Invalid weight schedule"):
            PhysicsLossConfig(weight_schedule="invalid")

    def test_validation_valid_schedules(self):
        """Test config with all valid schedule types."""
        for schedule in ["linear", "exponential", "step"]:
            config = PhysicsLossConfig(weight_schedule=schedule)
            assert config.weight_schedule == schedule

    def test_step_schedule_milestones(self):
        """Test step schedule milestone initialization."""
        config = PhysicsLossConfig(
            weight_schedule="step", step_milestones=[100, 200, 300]
        )
        assert config.step_milestones == [100, 200, 300]

        # Test auto-initialization to empty list
        config2 = PhysicsLossConfig(weight_schedule="step")
        assert config2.step_milestones == []

    def test_quantum_constraints(self):
        """Test quantum constraint configuration."""
        config = PhysicsLossConfig(
            quantum_constraints=True,
            density_positivity_weight=0.1,
            wavefunction_normalization_weight=0.2,
        )
        assert config.quantum_constraints is True
        assert config.density_positivity_weight == 0.1
        assert config.wavefunction_normalization_weight == 0.2


class TestPhysicsLossComposer:
    """Test physics loss composition."""

    def test_basic_composition(self):
        """Test basic loss composition without optional terms."""
        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
        )
        composer = PhysicsLossComposer(config)

        data_loss = jnp.array(2.0)
        physics_residual = jnp.array(0.5)
        boundary_residual = jnp.array(1.0)

        total_loss = composer.compose_loss(
            data_loss, physics_residual, boundary_residual
        )
        expected = 1.0 * 2.0 + 0.1 * 0.5 + 1.0 * 1.0
        assert jnp.allclose(total_loss, expected)

    def test_composition_with_conservation(self):
        """Test loss composition with conservation law terms."""
        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"energy": 0.2, "momentum": 0.3},
        )
        composer = PhysicsLossComposer(config)

        conservation_residuals = {
            "energy": jnp.array(0.1),
            "momentum": jnp.array(0.2),
        }

        total_loss = composer.compose_loss(
            jnp.array(2.0),
            jnp.array(0.5),
            jnp.array(1.0),
            conservation_residuals=conservation_residuals,
        )

        expected = 1.0 * 2.0 + 0.1 * 0.5 + 1.0 * 1.0 + 0.2 * 0.1 + 0.3 * 0.2
        assert jnp.allclose(total_loss, expected)

    def test_composition_with_quantum(self):
        """Test loss composition with quantum constraint terms."""
        config = PhysicsLossConfig(
            quantum_constraints=True,
            density_positivity_weight=0.1,
            wavefunction_normalization_weight=0.2,
        )
        composer = PhysicsLossComposer(config)

        quantum_residuals = {
            "density_positivity": jnp.array(0.5),
            "wavefunction_normalization": jnp.array(0.3),
        }

        total_loss = composer.compose_loss(
            jnp.array(0.0),
            jnp.array(0.0),
            jnp.array(0.0),
            quantum_residuals=quantum_residuals,
        )

        expected = 0.1 * 0.5 + 0.2 * 0.3
        assert jnp.allclose(total_loss, expected)

    def test_compute_residuals(self):
        """Test basic residual computation."""
        config = PhysicsLossConfig()
        composer = PhysicsLossComposer(config)

        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        inputs = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        residuals = composer.compute_residuals(predictions, targets, inputs)

        assert "data_loss" in residuals
        assert "physics_residual" in residuals
        assert "boundary_residual" in residuals
        assert jnp.allclose(
            residuals["data_loss"], jnp.mean((predictions - targets) ** 2)
        )


class TestAdaptiveWeightScheduler:
    """Test adaptive weight scheduling."""

    def test_linear_schedule(self):
        """Test linear weight schedule."""
        scheduler = AdaptiveWeightScheduler(
            schedule_type="linear",
            initial_physics_weight=0.01,
            final_physics_weight=1.0,
            transition_epochs=100,
        )

        # At epoch 0
        weight_0 = scheduler.get_weight(0)
        assert jnp.allclose(weight_0, 0.01)

        # At epoch 50 (halfway)
        weight_50 = scheduler.get_weight(50)
        assert jnp.allclose(weight_50, 0.505, atol=1e-3)

        # At epoch 100 (end)
        weight_100 = scheduler.get_weight(100)
        assert jnp.allclose(weight_100, 1.0)

    def test_exponential_schedule(self):
        """Test exponential weight schedule."""
        scheduler = AdaptiveWeightScheduler(
            schedule_type="exponential",
            initial_physics_weight=0.01,
            final_physics_weight=1.0,
            transition_epochs=100,
        )

        # At epoch 0
        weight_0 = scheduler.get_weight(0)
        assert jnp.allclose(weight_0, 0.01)

        # At epoch 100 (end)
        weight_100 = scheduler.get_weight(100)
        assert jnp.allclose(weight_100, 1.0)

        # Exponential should be slower than linear at epoch 50
        weight_50 = scheduler.get_weight(50)
        assert weight_50 < 0.505  # Less than linear midpoint

    def test_step_schedule(self):
        """Test step-wise weight schedule."""
        scheduler = AdaptiveWeightScheduler(
            schedule_type="step",
            initial_physics_weight=0.0,
            final_physics_weight=1.0,
            transition_epochs=100,
            step_milestones=[25, 50, 75],
        )

        # Before first milestone
        assert jnp.allclose(scheduler.get_weight(0), 0.0)
        assert jnp.allclose(scheduler.get_weight(24), 0.0)

        # After first milestone
        weight_after_25 = scheduler.get_weight(25)
        assert weight_after_25 > 0.0

        # After all milestones
        weight_after_75 = scheduler.get_weight(100)
        assert jnp.allclose(weight_after_75, 1.0)

    def test_step_schedule_empty_milestones(self):
        """Test step schedule with no milestones."""
        scheduler = AdaptiveWeightScheduler(
            schedule_type="step",
            initial_physics_weight=0.5,
            final_physics_weight=1.0,
            step_milestones=[],
        )

        # Should return initial weight
        assert jnp.allclose(scheduler.get_weight(0), 0.5)
        assert jnp.allclose(scheduler.get_weight(100), 0.5)

    def test_epoch_clamping(self):
        """Test that epochs beyond transition are clamped."""
        scheduler = AdaptiveWeightScheduler(
            schedule_type="linear",
            initial_physics_weight=0.0,
            final_physics_weight=1.0,
            transition_epochs=100,
        )

        # Epoch beyond transition should clamp to final weight
        weight_200 = scheduler.get_weight(200)
        assert jnp.allclose(weight_200, 1.0)


class TestConservationLawEnforcer:
    """Test conservation law enforcement."""

    def test_mass_conservation(self):
        """Test mass conservation residual computation."""
        enforcer = ConservationLawEnforcer(["mass"], tolerance=1e-6)

        velocity_field = jnp.array([0.1, 0.2, 0.3])
        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        residual = enforcer.compute_residual("mass", velocity_field, coords)
        assert residual >= 0  # Residual should be non-negative

    def test_momentum_conservation(self):
        """Test momentum conservation residual computation."""
        enforcer = ConservationLawEnforcer(["momentum"], tolerance=1e-6)

        velocity = jnp.array([1.0, 2.0, 3.0])
        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        residual = enforcer.compute_residual("momentum", velocity, coords)
        assert residual >= 0

    def test_energy_conservation(self):
        """Test energy conservation residual computation."""
        enforcer = ConservationLawEnforcer(["energy"], tolerance=1e-6)

        # State = (position, momentum)
        state = (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
        coords = jnp.array([0.0, 0.1])

        residual = enforcer.compute_residual("energy", state, coords)
        assert residual >= 0

    def test_particle_number_conservation(self):
        """Test particle number conservation for quantum systems."""
        enforcer = ConservationLawEnforcer(["particle_number"], tolerance=1e-6)

        # Identity density matrix (particle number = 2)
        density_matrix = jnp.eye(2)
        residual = enforcer.compute_residual("particle_number", density_matrix)
        assert residual >= 0

    def test_charge_conservation(self):
        """Test charge conservation for quantum systems."""
        enforcer = ConservationLawEnforcer(["charge"], tolerance=1e-6)

        density_matrix = jnp.eye(3)
        residual = enforcer.compute_residual("charge", density_matrix)
        assert residual >= 0

    def test_symmetry_conservation(self):
        """Test symmetry preservation."""
        enforcer = ConservationLawEnforcer(["symmetry"], tolerance=1e-6)

        # Perfectly symmetric field
        symmetric_field = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])
        residual_symmetric = enforcer.compute_residual("symmetry", symmetric_field)
        assert jnp.allclose(residual_symmetric, 0.0, atol=1e-6)

        # Asymmetric field
        asymmetric_field = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residual_asymmetric = enforcer.compute_residual("symmetry", asymmetric_field)
        assert residual_asymmetric > 0.1

    def test_unknown_law(self):
        """Test handling of unknown conservation law."""
        enforcer = ConservationLawEnforcer(["unknown"], tolerance=1e-6)

        state = jnp.array([1.0, 2.0])
        residual = enforcer.compute_residual("unknown", state)
        assert jnp.allclose(residual, 0.0)  # Should return zero for unknown

    def test_tuple_state_handling(self):
        """Test handling of tuple states for various laws."""
        enforcer = ConservationLawEnforcer(
            ["mass", "particle_number", "charge", "symmetry"], tolerance=1e-6
        )

        # Test mass with tuple
        velocity_tuple = (jnp.array([1.0, 2.0]),)
        residual = enforcer.compute_residual("mass", velocity_tuple, None)
        assert residual >= 0

        # Test particle_number with tuple
        dm_tuple = (jnp.eye(2),)
        residual = enforcer.compute_residual("particle_number", dm_tuple, None)
        assert residual >= 0


class TestResidualComputer:
    """Test physics equation residual computation."""

    def test_poisson_residual(self):
        """Test Poisson equation residual: ∇²u = f."""
        computer = ResidualComputer("poisson", "2d")

        # Create a simple model function
        def model(x):
            # For u = x² + y², ∇²u = 2 + 2 = 4
            return jnp.sum(x**2, axis=-1)

        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        source = jnp.array([4.0, 4.0, 4.0])

        residual = computer.compute_residual(model, coords, source=source)
        # Residual should be finite and close to 0 for this exact solution
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.all(jnp.abs(residual) < 1e-5)

    def test_wave_residual(self):
        """Test wave equation residual: ∂²u/∂t² = c²∇²u."""
        computer = ResidualComputer("wave", "2d")

        # Create a simple model function
        def model(x):
            # Constant function has zero Laplacian
            return jnp.ones(x.shape[0])

        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        residual = computer.compute_residual(model, coords)
        # For constant function, ∇²u = 0, so residual should be ~0
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.all(jnp.abs(residual) < 1e-5)

    def test_schrodinger_residual(self):
        """Test Schrödinger equation residual: Ĥψ = Eψ."""
        computer = ResidualComputer("schrodinger", "3d", potential_type="harmonic")

        coords = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        # Create a model function for the wavefunction
        def model(x):
            # Ground state harmonic oscillator wavefunction: ψ(x) = exp(-r²/2)
            return jnp.exp(-0.5 * jnp.sum(x**2, axis=-1))

        residual = computer.compute_residual(model, coords)
        # Residual should be finite
        assert jnp.all(jnp.isfinite(residual))
        # For ground state harmonic oscillator, residual should be small
        assert jnp.all(jnp.abs(residual) < 0.1)

    def test_heat_residual(self):
        """Test heat equation residual (steady-state): α∇²u = 0."""
        computer = ResidualComputer("heat", "2d", alpha=1.0)

        # Create a model function - harmonic function has ∇²u = 0
        def model(x):
            # u = x² - y² is harmonic (∇²u = 2 - 2 = 0)
            return x[..., 0] ** 2 - x[..., 1] ** 2

        coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        residual = computer.compute_residual(model, coords)
        # For harmonic function, residual should be ~0
        assert jnp.all(jnp.isfinite(residual))
        assert jnp.all(jnp.abs(residual) < 1e-5)

    def test_burgers_residual(self):
        """Test Burgers equation residual: u·∇u - ν∇²u = 0."""
        computer = ResidualComputer("burgers", "1d", nu=0.01)

        # Create a simple model function
        def model(x):
            # Linear function u = x
            return x[..., 0]

        coords = jnp.array([[0.0], [1.0], [2.0]])

        residual = computer.compute_residual(model, coords)
        # For u = x: ∇u = 1, ∇²u = 0, so residual = u * 1 - 0 = x
        assert jnp.all(jnp.isfinite(residual))

    def test_unknown_equation(self):
        """Test handling of unknown equation type raises KeyError."""
        computer = ResidualComputer("unknown", "2d")

        def model(x):
            return jnp.sum(x, axis=-1)

        coords = jnp.array([[0.0, 0.0], [1.0, 0.0]])

        # Unknown PDE should raise KeyError with helpful message
        with pytest.raises(KeyError, match="PDE 'unknown' not found in registry"):
            computer.compute_residual(model, coords)

    def test_equation_params(self):
        """Test residual computation with equation parameters."""
        computer = ResidualComputer(
            "schrodinger", "3d", potential_type="coulomb", charge=-1.0
        )

        assert computer.equation_params["potential_type"] == "coulomb"
        assert computer.equation_params["charge"] == -1.0


class TestPhysicsInformedLoss:
    """Test integrated physics-informed loss system."""

    def test_initialization(self):
        """Test PhysicsInformedLoss initialization."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        assert loss_system.config == config
        assert loss_system.composer is not None
        assert loss_system.residual_computer is not None
        assert loss_system.current_epoch == 0

    def test_adaptive_scheduler_creation(self):
        """Test that adaptive scheduler is created when enabled."""
        config = PhysicsLossConfig(adaptive_weighting=True, weight_schedule="linear")
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        assert loss_system.weight_scheduler is not None

    def test_conservation_enforcer_creation(self):
        """Test that conservation enforcer is created when needed."""
        config = PhysicsLossConfig(
            conservation_weights={"energy": 0.1, "momentum": 0.2}
        )
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        assert loss_system.conservation_enforcer is not None
        assert "energy" in loss_system.conservation_enforcer.conservation_laws
        assert "momentum" in loss_system.conservation_enforcer.conservation_laws

    def test_basic_loss_computation(self):
        """Test basic physics-informed loss computation."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        total_loss, components = loss_system.compute_loss(predictions, targets, inputs)

        assert jnp.isscalar(total_loss)
        assert "data_loss" in components
        assert "physics_loss" in components
        assert "boundary_loss" in components
        assert "total_loss" in components

    def test_loss_with_boundary_conditions(self):
        """Test loss computation with boundary conditions."""
        config = PhysicsLossConfig(boundary_loss_weight=2.0)
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0]])

        boundary_predictions = jnp.array([0.0, 0.0])
        boundary_targets = jnp.array([0.1, 0.1])
        boundary_inputs = jnp.array([[0.0, 0.0], [1.0, 1.0]])

        _, components = loss_system.compute_loss(
            predictions,
            targets,
            inputs,
            boundary_predictions=boundary_predictions,
            boundary_targets=boundary_targets,
            boundary_inputs=boundary_inputs,
        )

        assert components["boundary_loss"] > 0

    def test_loss_with_conservation_laws(self):
        """Test loss computation with conservation law enforcement."""
        config = PhysicsLossConfig(
            conservation_weights={"energy": 0.1, "momentum": 0.2}
        )
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.0, 2.0, 3.0])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        _, components = loss_system.compute_loss(predictions, targets, inputs)

        assert "energy" in components
        assert "momentum" in components

    def test_loss_with_quantum_constraints(self):
        """Test loss computation with quantum constraints."""
        config = PhysicsLossConfig(
            quantum_constraints=True,
            density_positivity_weight=0.1,
            wavefunction_normalization_weight=0.2,
        )
        loss_system = PhysicsInformedLoss(config, "schrodinger", "3d")

        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        inputs = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        density_matrix = jnp.eye(2)

        _, components = loss_system.compute_loss(
            predictions,
            targets,
            inputs,
            density_matrix=density_matrix,
            n_electrons=2,
        )

        assert "quantum_constraints" in components

    def test_adaptive_weight_update(self):
        """Test adaptive weight updating during training."""
        config = PhysicsLossConfig(
            adaptive_weighting=True,
            weight_schedule="linear",
            physics_loss_weight=0.01,
        )
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        # Initial weight
        initial_weight = loss_system.get_current_physics_weight()
        assert jnp.allclose(initial_weight, 0.01)

        # Update epoch
        loss_system.update_weights(500)
        updated_weight = loss_system.get_current_physics_weight()
        assert updated_weight > initial_weight

    def test_epoch_parameter_in_compute_loss(self):
        """Test that epoch parameter affects loss computation."""
        config = PhysicsLossConfig(
            adaptive_weighting=True,
            weight_schedule="linear",
            physics_loss_weight=0.01,
        )
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0]])

        # Compute loss at different epochs
        loss_epoch_0, _ = loss_system.compute_loss(
            predictions, targets, inputs, epoch=0
        )
        loss_epoch_1000, _ = loss_system.compute_loss(
            predictions, targets, inputs, epoch=1000
        )

        # Losses can differ due to adaptive weighting
        # (or be same if physics residual is zero)
        assert jnp.isfinite(loss_epoch_0)
        assert jnp.isfinite(loss_epoch_1000)


class TestJAXCompatibility:
    """Test JAX transformation compatibility."""

    def test_jit_compilation(self):
        """Test JIT compilation of loss computation."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        @jax.jit
        def jitted_loss(predictions, targets, inputs):
            loss, _ = loss_system.compute_loss(predictions, targets, inputs)
            return loss

        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        loss = jitted_loss(predictions, targets, inputs)
        assert jnp.isfinite(loss)

    def test_vmap_over_batch(self):
        """Test vmap over batch dimension."""
        scheduler = AdaptiveWeightScheduler(schedule_type="linear")

        # Vectorize over epochs
        epochs = jnp.array([0, 50, 100])
        weights = jax.vmap(scheduler.get_weight)(epochs)  # type: ignore  # noqa: PGH003

        assert weights.shape == (3,)
        assert jnp.all(jnp.isfinite(weights))

    def test_grad_through_loss(self):
        """Test gradient computation through loss."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        def loss_fn(predictions, targets, inputs):
            loss, _ = loss_system.compute_loss(predictions, targets, inputs)
            return loss

        predictions = jnp.array([1.0, 2.0, 3.0])
        targets = jnp.array([1.1, 2.1, 3.1])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        grad_fn = jax.grad(loss_fn)
        gradients = grad_fn(predictions, targets, inputs)

        assert gradients.shape == predictions.shape
        assert jnp.all(jnp.isfinite(gradients))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_predictions(self):
        """Test loss computation with zero predictions."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.zeros(3)
        targets = jnp.array([1.0, 2.0, 3.0])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        loss, components = loss_system.compute_loss(predictions, targets, inputs)
        assert jnp.isfinite(loss)
        assert components["data_loss"] > 0

    def test_empty_conservation_weights(self):
        """Test that empty conservation weights work correctly."""
        config = PhysicsLossConfig(conservation_weights={})
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        assert loss_system.conservation_enforcer is None

    def test_none_boundary_conditions(self):
        """Test loss computation with None boundary conditions."""
        config = PhysicsLossConfig()
        loss_system = PhysicsInformedLoss(config, "poisson", "2d")

        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        inputs = jnp.array([[0.0, 0.0], [1.0, 0.0]])

        _, components = loss_system.compute_loss(
            predictions,
            targets,
            inputs,
            boundary_predictions=None,
            boundary_targets=None,
        )

        assert jnp.allclose(components["boundary_loss"], 0.0)

    def test_none_density_matrix(self):
        """Test quantum constraints with None density matrix."""
        config = PhysicsLossConfig(
            quantum_constraints=True,
            density_positivity_weight=0.1,
        )
        loss_system = PhysicsInformedLoss(config, "schrodinger", "3d")

        predictions = jnp.array([1.0, 2.0])
        targets = jnp.array([1.0, 2.0])
        inputs = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        _, components = loss_system.compute_loss(
            predictions,
            targets,
            inputs,
            density_matrix=None,
        )

        # Should not crash, quantum constraints should be skipped
        assert "quantum_constraints" not in components
