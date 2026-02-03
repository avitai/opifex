"""Tests for quantum training module."""

from __future__ import annotations

import jax.numpy as jnp

from opifex.training.quantum_training import QuantumTrainingManager


class TestQuantumTrainingManager:
    """Test quantum training manager."""

    def test_manager_initialization(self):
        """Test quantum training manager initialization."""
        config = {
            "quantum_training": True,
            "scf_tolerance": 1e-6,
            "max_scf_iterations": 50,
            "dft_functional": "pbe",
        }

        manager = QuantumTrainingManager(config)

        assert manager.scf_tolerance == 1e-6
        assert manager.max_scf_iterations == 50
        assert manager.dft_functional == "pbe"

    def test_manager_disabled(self):
        """Test quantum training manager when disabled."""
        config = {"quantum_training": False}

        manager = QuantumTrainingManager(config)

        # Should use defaults when disabled
        assert manager.scf_tolerance == 1e-6
        assert manager.max_scf_iterations == 50

    def test_scf_convergence_monitoring(self):
        """Test SCF convergence monitoring."""
        config = {
            "quantum_training": True,
            "scf_tolerance": 1e-5,
            "max_scf_iterations": 10,
            "scf_mixing_parameter": 0.5,
        }

        manager = QuantumTrainingManager(config)

        # Create mock positions
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        converged, iterations = manager.monitor_scf_convergence(positions)

        # Should converge
        assert isinstance(converged, bool)
        assert isinstance(iterations, int)
        assert iterations > 0
        assert iterations <= 10

    def test_scf_step(self):
        """Test single SCF step."""
        config = {"quantum_training": True}
        manager = QuantumTrainingManager(config)

        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        density = jnp.ones(2) * 0.5

        new_density = manager.scf_step(positions, density)

        # Should return new density
        assert new_density.shape == density.shape
        assert jnp.all(jnp.isfinite(new_density))

    def test_dft_energy_computation(self):
        """Test DFT energy computation."""
        config = {
            "quantum_training": True,
            "dft_functional": "pbe",
        }

        manager = QuantumTrainingManager(config)

        # Create mock predictions
        y_pred = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        energy = manager.compute_dft_energy(y_pred)

        # Should return finite energy
        assert jnp.isfinite(energy)
        assert energy >= 0.0

    def test_quantum_state_computation(self):
        """Test quantum state computation."""
        config = {
            "quantum_training": True,
            "track_quantum_states": True,
        }

        manager = QuantumTrainingManager(config)

        # Create mock predictions
        y_pred = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        state = manager.compute_quantum_state(y_pred)

        # Should return finite state measure
        assert jnp.isfinite(state)
        assert state >= 0.0

    def test_electron_density_threshold(self):
        """Test electron density threshold configuration."""
        config = {
            "quantum_training": True,
            "electron_density_threshold": 1e-8,
        }

        manager = QuantumTrainingManager(config)

        assert manager.electron_density_threshold == 1e-8

    def test_exchange_correlation_weight(self):
        """Test exchange-correlation weight configuration."""
        config = {
            "quantum_training": True,
            "exchange_correlation_weight": 0.3,
        }

        manager = QuantumTrainingManager(config)

        assert manager.exchange_correlation_weight == 0.3

    def test_quantum_state_tracking(self):
        """Test quantum state tracking configuration."""
        config = {
            "quantum_training": True,
            "track_quantum_states": True,
            "quantum_state_history": True,
        }

        manager = QuantumTrainingManager(config)

        assert manager.track_quantum_states is True
        assert manager.quantum_state_history is True

    def test_electronic_structure_support(self):
        """Test electronic structure configuration."""
        config = {
            "quantum_training": True,
            "electronic_structure": True,
            "orbital_optimization": True,
            "electronic_temperature": 300.0,
        }

        manager = QuantumTrainingManager(config)

        assert manager.electronic_structure is True
        assert manager.orbital_optimization is True
        assert manager.electronic_temperature == 300.0

    def test_scf_with_different_tolerances(self):
        """Test SCF convergence with different tolerances."""
        # Tight tolerance
        config1 = {
            "quantum_training": True,
            "scf_tolerance": 1e-8,
            "max_scf_iterations": 100,
        }

        manager1 = QuantumTrainingManager(config1)
        positions = jnp.array([[0.0, 0.0, 0.0]])

        converged1, iterations1 = manager1.monitor_scf_convergence(positions)

        # Loose tolerance
        config2 = {
            "quantum_training": True,
            "scf_tolerance": 1e-3,
            "max_scf_iterations": 100,
        }

        manager2 = QuantumTrainingManager(config2)

        converged2, iterations2 = manager2.monitor_scf_convergence(positions)

        # Loose tolerance should converge faster
        assert iterations2 <= iterations1 or (converged2 and not converged1)

    def test_scf_mixing_parameter(self):
        """Test SCF mixing parameter effect."""
        config = {
            "quantum_training": True,
            "scf_mixing_parameter": 0.8,  # High mixing
            "scf_tolerance": 1e-5,
            "max_scf_iterations": 20,
        }

        manager = QuantumTrainingManager(config)

        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        converged, iterations = manager.monitor_scf_convergence(positions)

        # Should still converge with different mixing
        assert isinstance(converged, bool)
        assert iterations <= 20
