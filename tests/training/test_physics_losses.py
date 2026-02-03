"""Test suite for physics-informed loss functions.

This module tests physics-informed neural network (PINN) functionality including:
- Physics loss configuration and composition
- Adaptive weight scheduling for physics losses
- Conservation law enforcement
- Residual computation for PDEs
- Physics-informed loss integration with training

Tests extracted from test_basic_trainer.py during refactoring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.core.physics import (
    AdaptiveWeightScheduler,
    ConservationLawEnforcer,
    PhysicsInformedLoss,
    PhysicsLossComposer,
    PhysicsLossConfig,
    ResidualComputer,
)


class TestPhysicsLossConfig:
    """Test physics-informed loss configuration."""

    def test_physics_loss_config_creation(self):
        """Test basic physics loss configuration creation."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"mass": 1.0, "momentum": 0.5, "energy": 2.0},
            adaptive_weighting=True,
            weight_schedule="exponential",
        )

        assert config.data_loss_weight == 1.0
        assert config.physics_loss_weight == 0.1
        assert config.boundary_loss_weight == 1.0
        assert config.conservation_weights["mass"] == 1.0
        assert config.conservation_weights["momentum"] == 0.5
        assert config.conservation_weights["energy"] == 2.0
        assert config.adaptive_weighting is True
        assert config.weight_schedule == "exponential"

    def test_physics_loss_config_validation(self):
        """Test physics loss configuration validation."""

        # Test invalid weight schedule
        with pytest.raises(ValueError, match="Invalid weight schedule"):
            PhysicsLossConfig(
                data_loss_weight=1.0,
                physics_loss_weight=0.1,
                boundary_loss_weight=1.0,
                weight_schedule="invalid_schedule",
            )

    def test_physics_loss_config_quantum_extension(self):
        """Test physics loss configuration with quantum extensions."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"particle_number": 1.0, "charge": 1.0},
            quantum_constraints=True,
            density_positivity_weight=0.5,
            wavefunction_normalization_weight=2.0,
        )

        assert config.quantum_constraints is True
        assert config.density_positivity_weight == 0.5
        assert config.wavefunction_normalization_weight == 2.0


class TestPhysicsLossComposer:
    """Test hierarchical physics loss composition."""

    def test_loss_composer_initialization(self):
        """Test physics loss composer initialization."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
        )

        composer = PhysicsLossComposer(config)
        assert composer.config == config
        assert hasattr(composer, "compose_loss")
        assert hasattr(composer, "compute_residuals")

    def test_loss_composition_standard_pde(self):
        """Test loss composition for standard PDE problems."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
        )

        composer = PhysicsLossComposer(config)

        # Mock data
        batch_size = 32
        x = jnp.linspace(0, 1, batch_size).reshape(-1, 1)
        y_true = jnp.sin(jnp.pi * x)
        y_pred = jnp.sin(jnp.pi * x) + 0.1 * jax.random.normal(
            jax.random.PRNGKey(42), x.shape
        )

        # Mock residuals
        physics_residual = jnp.mean((y_pred - y_true) ** 2)
        boundary_residual = jnp.mean((y_pred[0] - 0.0) ** 2 + (y_pred[-1] - 0.0) ** 2)

        total_loss = composer.compose_loss(
            data_loss=jnp.mean((y_pred - y_true) ** 2),
            physics_residual=physics_residual,
            boundary_residual=boundary_residual,
        )

        expected_loss = (
            config.data_loss_weight * jnp.mean((y_pred - y_true) ** 2)
            + config.physics_loss_weight * physics_residual
            + config.boundary_loss_weight * boundary_residual
        )

        assert jnp.allclose(total_loss, expected_loss, atol=1e-6)

    def test_loss_composition_conservation_laws(self):
        """Test loss composition with conservation law enforcement."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"mass": 1.0, "momentum": 0.5, "energy": 2.0},
        )

        composer = PhysicsLossComposer(config)

        # Mock conservation law violations
        conservation_residuals = {
            "mass": jnp.array(0.1),
            "momentum": jnp.array(0.05),
            "energy": jnp.array(0.02),
        }

        total_loss = composer.compose_loss(
            data_loss=jnp.array(1.0),
            physics_residual=jnp.array(0.1),
            boundary_residual=jnp.array(0.1),
            conservation_residuals=conservation_residuals,
        )

        expected_conservation_loss = (
            1.0 * 0.1  # mass
            + 0.5 * 0.05  # momentum
            + 2.0 * 0.02  # energy
        )

        expected_total = 1.0 * 1.0 + 0.1 * 0.1 + 1.0 * 0.1 + expected_conservation_loss

        assert jnp.allclose(total_loss, expected_total, atol=1e-6)


class TestAdaptiveWeightScheduler:
    """Test adaptive weight scheduling for physics losses."""

    def test_scheduler_initialization(self):
        """Test adaptive weight scheduler initialization."""

        scheduler = AdaptiveWeightScheduler(
            schedule_type="exponential",
            initial_physics_weight=0.01,
            final_physics_weight=1.0,
            transition_epochs=1000,
        )

        assert scheduler.schedule_type == "exponential"
        assert scheduler.initial_physics_weight == 0.01
        assert scheduler.final_physics_weight == 1.0
        assert scheduler.transition_epochs == 1000

    def test_exponential_schedule(self):
        """Test exponential weight scheduling."""

        scheduler = AdaptiveWeightScheduler(
            schedule_type="exponential",
            initial_physics_weight=0.01,
            final_physics_weight=1.0,
            transition_epochs=1000,
        )

        # Test at beginning
        weight_0 = scheduler.get_weight(epoch=0)
        assert jnp.allclose(weight_0, 0.01, atol=1e-6)

        # Test at end
        weight_final = scheduler.get_weight(epoch=1000)
        assert jnp.allclose(weight_final, 1.0, atol=1e-6)

        # Test monotonic increase
        weight_mid = scheduler.get_weight(epoch=500)
        assert 0.01 < weight_mid < 1.0
        assert weight_0 < weight_mid < weight_final

    def test_linear_schedule(self):
        """Test linear weight scheduling."""

        scheduler = AdaptiveWeightScheduler(
            schedule_type="linear",
            initial_physics_weight=0.1,
            final_physics_weight=1.0,
            transition_epochs=100,
        )

        # Test at beginning
        weight_0 = scheduler.get_weight(epoch=0)
        assert jnp.allclose(weight_0, 0.1, atol=1e-6)

        # Test at middle
        weight_mid = scheduler.get_weight(epoch=50)
        expected_mid = 0.1 + (1.0 - 0.1) * 0.5
        assert jnp.allclose(weight_mid, expected_mid, atol=1e-6)

        # Test at end
        weight_final = scheduler.get_weight(epoch=100)
        assert jnp.allclose(weight_final, 1.0, atol=1e-6)

    def test_step_schedule(self):
        """Test step weight scheduling."""

        scheduler = AdaptiveWeightScheduler(
            schedule_type="step",
            initial_physics_weight=0.1,
            final_physics_weight=1.0,
            transition_epochs=100,
            step_milestones=[25, 50, 75],
        )

        # Test before first milestone
        weight_0 = scheduler.get_weight(epoch=10)
        assert jnp.allclose(weight_0, 0.1, atol=1e-6)

        # Test after first milestone
        weight_30 = scheduler.get_weight(epoch=30)
        assert weight_30 > 0.1

        # Test final weight
        weight_final = scheduler.get_weight(epoch=100)
        assert jnp.allclose(weight_final, 1.0, atol=1e-6)


class TestConservationLawEnforcer:
    """Test conservation law enforcement."""

    def test_enforcer_initialization(self):
        """Test conservation law enforcer initialization."""

        enforcer = ConservationLawEnforcer(
            conservation_laws=["mass", "momentum", "energy"],
            tolerance=1e-6,
        )

        assert enforcer.conservation_laws == ["mass", "momentum", "energy"]
        assert enforcer.tolerance == 1e-6

    def test_mass_conservation(self):
        """Test mass conservation enforcement."""

        enforcer = ConservationLawEnforcer(
            conservation_laws=["mass"],
            tolerance=1e-6,
        )

        # Mock velocity field (should be divergence-free for mass conservation)
        x = jnp.linspace(0, 1, 100).reshape(-1, 1)
        u = jnp.sin(2 * jnp.pi * x)  # velocity field

        # Compute mass conservation residual
        mass_residual = enforcer.compute_residual("mass", u, x)

        # Should be a scalar residual
        assert mass_residual.shape == ()
        assert jnp.isfinite(mass_residual)

    def test_energy_conservation(self):
        """Test energy conservation enforcement."""

        enforcer = ConservationLawEnforcer(
            conservation_laws=["energy"],
            tolerance=1e-6,
        )

        # Mock system state
        t = jnp.linspace(0, 1, 100)
        q = jnp.sin(t)  # position
        p = jnp.cos(t)  # momentum

        # Compute energy conservation residual
        energy_residual = enforcer.compute_residual("energy", (q, p), t)

        # Should be a scalar residual
        assert energy_residual.shape == ()
        assert jnp.isfinite(energy_residual)

    def test_quantum_conservation_laws(self):
        """Test quantum mechanical conservation laws."""

        enforcer = ConservationLawEnforcer(
            conservation_laws=["particle_number", "charge"],
            tolerance=1e-8,
        )

        # Mock quantum state
        batch_size = 32
        n_orbitals = 10
        density_matrix = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, n_orbitals, n_orbitals)
        )

        # Compute particle number conservation
        particle_residual = enforcer.compute_residual(
            "particle_number", density_matrix, None
        )

        assert particle_residual.shape == ()
        assert jnp.isfinite(particle_residual)

    def test_symmetry_conservation(self):
        """Test symmetry preservation enforcement."""

        enforcer = ConservationLawEnforcer(
            conservation_laws=["symmetry"],
            tolerance=1e-6,
        )

        # Mock field that should preserve symmetry
        batch_size = 64
        field_size = 32
        # Create symmetric field: f(x) = f(-x)
        x = jnp.linspace(-1, 1, field_size)
        symmetric_field = jnp.exp(-(x**2))  # Gaussian is symmetric

        # Stack for batch
        field = jnp.tile(symmetric_field, (batch_size, 1))

        # Compute symmetry conservation residual
        symmetry_residual = enforcer.compute_residual("symmetry", field, None)

        # Should be a scalar residual
        assert symmetry_residual.shape == ()
        assert jnp.isfinite(symmetry_residual)

        # For a symmetric field, residual should be small
        assert symmetry_residual < 0.1

        # Test with asymmetric field
        asymmetric_field = x  # Linear function is asymmetric
        asym_field = jnp.tile(asymmetric_field, (batch_size, 1))

        asym_residual = enforcer.compute_residual("symmetry", asym_field, None)

        # Asymmetric field should have larger residual
        assert asym_residual > symmetry_residual


class TestResidualComputer:
    """Test residual computation for physics equations."""

    def test_residual_computer_initialization(self):
        """Test residual computer initialization."""

        computer = ResidualComputer(
            equation_type="poisson",
            domain_type="2d_rectangular",
        )

        assert computer.equation_type == "poisson"
        assert computer.domain_type == "2d_rectangular"

    def test_poisson_residual_computation(self):
        """Test Poisson equation residual computation with real autodiff.

        REFACTORED: Now uses proper model functions instead of pre-computed values.
        This enables real autodiff-based derivative computation.
        """

        computer = ResidualComputer(
            equation_type="poisson",
            domain_type="2d_rectangular",
        )

        # Define analytical solution as a function: u = x² + y²
        def u_solution(x_eval):
            """Analytical Poisson solution with ∇²u = 4"""
            return jnp.sum(x_eval**2, axis=-1)

        # Test points
        batch_size = 64
        x = jax.random.uniform(jax.random.PRNGKey(42), (batch_size, 2))

        # Source term: ∇²u = 4 for this solution
        f = jnp.full(batch_size, 4.0)

        # Compute residual using real autodiff
        residual = computer.compute_residual(u_solution, x, source=f)

        # REFACTORED: Now returns per-point residuals using real autodiff
        assert residual.shape == (batch_size,)
        assert jnp.isfinite(residual).all()
        # Residuals should be near zero for this exact solution (∇²u - f = 4 - 4 = 0)
        assert jnp.allclose(residual, 0.0, atol=1e-4)

    def test_wave_equation_residual(self):
        """Test wave equation residual computation with real autodiff.

        REFACTORED: Now uses proper model functions instead of pre-computed values.
        """

        computer = ResidualComputer(
            equation_type="wave",
            domain_type="1d",
            wave_speed=1.0,
        )

        # Define wave solution as a function
        def u_wave(xt):
            """Wave solution: u(x,t) = sin(π·x)"""
            # For spatial part of wave equation, just use x coordinate
            return jnp.sin(jnp.pi * xt[..., 0])

        # Test points (spatial coordinates)
        batch_size = 100
        x = jnp.linspace(0, 1, batch_size).reshape(-1, 1)

        # Compute residual using real autodiff
        residual = computer.compute_residual(u_wave, x)

        # REFACTORED: Now returns per-point residuals using real autodiff
        assert residual.shape == (batch_size,)
        assert jnp.isfinite(residual).all()

    def test_schrodinger_residual(self):
        """Test Schrödinger equation residual computation with real autodiff.

        REFACTORED: Now uses proper model functions instead of pre-computed values.
        """

        computer = ResidualComputer(
            equation_type="schrodinger",
            domain_type="1d",
            potential_type="harmonic",
        )

        # Define ground state wavefunction as a function
        def psi_ground_state(x_eval):
            """Ground state of 1D harmonic oscillator: ψ = exp(-x²/2)"""
            r_sq = jnp.sum(x_eval**2, axis=-1)
            return jnp.exp(-0.5 * r_sq)

        # Test points
        batch_size = 128
        x = jnp.linspace(-3, 3, batch_size).reshape(-1, 1)

        # Compute residual using real autodiff
        residual = computer.compute_residual(psi_ground_state, x)

        # REFACTORED: Now returns per-point residuals using real autodiff
        assert residual.shape == (batch_size,)
        assert jnp.isfinite(residual).all()
        # For true ground state of harmonic oscillator, residual should be small
        # Note: Energy eigenvalue is E = 0.5 for 1D ground state
        assert jnp.mean(jnp.abs(residual)) < 0.1


class TestPhysicsInformedLoss:
    """Test integrated physics-informed loss system."""

    def test_integrated_loss_initialization(self):
        """Test integrated physics-informed loss initialization."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"mass": 1.0, "energy": 1.0},
            adaptive_weighting=True,
        )

        pi_loss = PhysicsInformedLoss(
            config=config,
            equation_type="poisson",
            domain_type="2d_rectangular",
        )

        assert pi_loss.config == config
        assert hasattr(pi_loss, "compute_loss")
        assert hasattr(pi_loss, "update_weights")

    def test_full_loss_computation(self):
        """Test full physics-informed loss computation."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            adaptive_weighting=False,
        )

        pi_loss = PhysicsInformedLoss(
            config=config,
            equation_type="poisson",
            domain_type="2d_rectangular",
        )

        # Mock training data
        batch_size = 64
        x_data = jax.random.uniform(jax.random.PRNGKey(42), (batch_size, 2))
        y_data = jnp.sum(x_data**2, axis=1)

        # Mock neural network predictions
        y_pred = y_data + 0.1 * jax.random.normal(jax.random.PRNGKey(43), y_data.shape)

        # Mock boundary data
        x_boundary = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        y_boundary = jnp.array([0.0, 2.0])
        y_boundary_pred = jnp.array([0.05, 1.95])

        # Compute full loss
        total_loss, loss_components = pi_loss.compute_loss(
            predictions=y_pred,
            targets=y_data,
            inputs=x_data,
            boundary_predictions=y_boundary_pred,
            boundary_targets=y_boundary,
            boundary_inputs=x_boundary,
            epoch=0,
        )

        assert jnp.isfinite(total_loss)
        assert total_loss.shape == ()
        assert "data_loss" in loss_components
        assert "physics_loss" in loss_components
        assert "boundary_loss" in loss_components
        assert "total_loss" in loss_components

    def test_adaptive_weight_update(self):
        """Test adaptive weight updating during training."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.01,
            boundary_loss_weight=1.0,
            adaptive_weighting=True,
            weight_schedule="exponential",
        )

        pi_loss = PhysicsInformedLoss(
            config=config,
            equation_type="poisson",
            domain_type="2d_rectangular",
        )

        # Initial weight
        initial_weight = pi_loss.get_current_physics_weight()
        assert jnp.allclose(initial_weight, 0.01, atol=1e-6)

        # Update weights
        pi_loss.update_weights(epoch=500)
        updated_weight = pi_loss.get_current_physics_weight()

        # Should have increased
        assert updated_weight > initial_weight

    def test_quantum_enhanced_loss(self):
        """Test quantum-enhanced physics-informed loss."""

        config = PhysicsLossConfig(
            data_loss_weight=1.0,
            physics_loss_weight=0.1,
            boundary_loss_weight=1.0,
            conservation_weights={"particle_number": 1.0, "charge": 1.0},
            quantum_constraints=True,
            density_positivity_weight=0.5,
            wavefunction_normalization_weight=2.0,
        )

        pi_loss = PhysicsInformedLoss(
            config=config,
            equation_type="schrodinger",
            domain_type="3d_molecular",
        )

        # Mock quantum system data
        batch_size = 32
        n_electrons = 10
        n_orbitals = 20

        # Mock molecular coordinates
        x_mol = jax.random.normal(jax.random.PRNGKey(42), (batch_size, 3))

        # Mock density matrix
        density_pred = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, n_orbitals, n_orbitals)
        )

        # Mock target energy
        energy_target = jax.random.normal(jax.random.PRNGKey(44), (batch_size,))
        energy_pred = energy_target + 0.1 * jax.random.normal(
            jax.random.PRNGKey(45), (batch_size,)
        )

        # Compute quantum-enhanced loss
        total_loss, loss_components = pi_loss.compute_loss(
            predictions=energy_pred,
            targets=energy_target,
            inputs=x_mol,
            density_matrix=density_pred,
            n_electrons=n_electrons,
            epoch=0,
        )

        assert jnp.isfinite(total_loss)
        assert total_loss.shape == ()
        assert "data_loss" in loss_components
        assert "physics_loss" in loss_components
        assert "quantum_constraints" in loss_components
        assert "density_positivity" in loss_components
        assert "wavefunction_normalization" in loss_components


# TestPhysicsInformedTrainingIntegration class removed - these tests were for the old
# trainer API that used set_physics_loss() method. The new Trainer uses composable
# configs (BoundaryConfig, ConservationConfig, etc.) and is already tested in
# tests/core/training/test_trainer.py with 27 passing tests.
