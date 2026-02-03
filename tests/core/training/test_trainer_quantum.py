"""Tests for quantum-specific training features with unified Trainer.

This module tests quantum training capabilities following the domain-agnostic
trainer architecture. Tests verify that quantum configs can be composed with
the unified Trainer without requiring domain-specific trainer methods.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.core.training.physics_configs import (
    DFTConfig,
    ElectronicStructureConfig,
    SCFConfig,
)
from opifex.core.training.trainer import Trainer
from opifex.training.quantum_training import QuantumTrainingManager


class MockModel(nnx.Module):
    """Mock model for quantum testing."""

    def __init__(self, features: int = 32, rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.linear1 = nnx.Linear(2, features, rngs=rngs)
        self.linear2 = nnx.Linear(features, features, rngs=rngs)
        self.linear3 = nnx.Linear(features, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        x = nnx.relu(self.linear2(x))
        return self.linear3(x)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    return MockModel(features=32, rngs=nnx.Rngs(42))


@pytest.fixture
def sample_data():
    """Generate sample training data."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 2))
    y = jnp.sum(x**2, axis=1, keepdims=True)
    return x, y


class TestQuantumCapabilities:
    """Test quantum training capabilities."""

    def test_quantum_manager_scf_convergence(self):
        """Test SCF convergence monitoring using QuantumTrainingManager."""
        # Use QuantumTrainingManager directly with dict config
        config = {
            "quantum_training": True,
            "max_scf_iterations": 100,
            "scf_tolerance": 1e-6,
        }
        manager = QuantumTrainingManager(config)

        positions = jax.random.normal(jax.random.PRNGKey(0), (5, 3, 3))
        converged, iterations = manager.monitor_scf_convergence(positions)

        assert isinstance(converged, bool)
        assert isinstance(iterations, int)
        assert iterations >= 0
        assert iterations <= 100

    def test_scf_config_composition(self, mock_model, sample_data):
        """Test SCF configuration composition with trainer."""
        scf_config = SCFConfig(
            max_iterations=50,
            tolerance=1e-6,
        )

        config = TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            scf_config=scf_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)

    def test_dft_config_composition(self, mock_model):
        """Test DFT configuration composition."""
        dft_config = DFTConfig(
            functional="pbe",
            electron_density_threshold=1e-8,
        )

        config = TrainingConfig(
            num_epochs=5,
            learning_rate=1e-3,
            dft_config=dft_config,
        )
        trainer = Trainer(mock_model, config)

        assert trainer.config.dft_config == dft_config
        assert trainer.config.dft_config is not None
        assert trainer.config.dft_config.functional == "pbe"

    def test_scf_convergence_with_different_tolerances(self):
        """Test SCF convergence with varying tolerance settings."""
        # Strict tolerance
        config_strict = {
            "quantum_training": True,
            "max_scf_iterations": 100,
            "scf_tolerance": 1e-8,
        }
        manager_strict = QuantumTrainingManager(config_strict)

        positions = jax.random.normal(jax.random.PRNGKey(42), (3, 3, 3))
        converged, iterations = manager_strict.monitor_scf_convergence(positions)

        assert isinstance(converged, bool)
        assert iterations >= 0

    def test_electronic_structure_config(self, mock_model):
        """Test electronic structure configuration."""
        electronic_config = ElectronicStructureConfig(
            enabled=True,
            orbital_optimization=True,
            temperature=300.0,  # Correct parameter name
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            electronic_structure_config=electronic_config,
        )
        trainer = Trainer(mock_model, config)

        assert trainer.config.electronic_structure_config == electronic_config
        assert trainer.config.electronic_structure_config is not None
        assert trainer.config.electronic_structure_config.temperature == 300.0

    def test_combined_quantum_configs(self, mock_model, sample_data):
        """Test combining multiple quantum configs."""
        scf_config = SCFConfig(max_iterations=50, tolerance=1e-6)
        dft_config = DFTConfig(functional="pbe")
        electronic_config = ElectronicStructureConfig(orbital_optimization=True)

        config = TrainingConfig(
            learning_rate=1e-3,
            scf_config=scf_config,
            dft_config=dft_config,
            electronic_structure_config=electronic_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(loss, jax.Array)
        assert isinstance(metrics, dict)


class TestQuantumTrainingEnhancements:
    """Test quantum training enhancements for real SCF and DFT integration."""

    def test_real_scf_convergence(self):
        """Test actual self-consistent field integration."""
        config = {
            "quantum_training": True,
            "scf_tolerance": 1e-6,
            "max_scf_iterations": 50,
            "scf_mixing_parameter": 0.5,
        }
        manager = QuantumTrainingManager(config)

        # Test SCF convergence with realistic positions
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])  # Simple molecule

        converged, iterations = manager.monitor_scf_convergence(positions)

        # Verify SCF convergence monitoring
        assert isinstance(converged, bool)
        assert isinstance(iterations, int)
        assert iterations > 0
        assert iterations <= 50

    def test_dft_integration(self, mock_model, sample_data):
        """Test density functional theory training capabilities."""
        dft_config = DFTConfig(
            functional="pbe",
            electron_density_threshold=1e-8,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            dft_config=dft_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data
        loss, _ = trainer.training_step(x[:10], y[:10])

        # Verify DFT integration
        assert isinstance(loss, jax.Array)
        assert loss > 0

    def test_electronic_structure_support(self, mock_model):
        """Test molecular electronic structure optimization."""
        electronic_config = ElectronicStructureConfig(
            enabled=True,
            orbital_optimization=True,
            temperature=300.0,  # Kelvin - correct parameter name
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            electronic_structure_config=electronic_config,
        )
        trainer = Trainer(mock_model, config)

        # Test electronic structure optimization
        # Create molecular system data with correct dimensions for mock model
        molecular_data = jnp.array([[1.0, 0.0], [0.0, 1.0]])  # 2 atoms, 2 features each
        electronic_energy = jnp.array([[-1.5]])  # Hartree

        loss, _ = trainer.training_step(molecular_data, electronic_energy)

        # Verify electronic structure support
        assert isinstance(loss, jax.Array)
        assert loss > 0

    def test_dft_functional_configuration(self):
        """Test DFT functional configuration options."""
        # Test PBE functional
        config_pbe = {"quantum_training": True, "dft_functional": "pbe"}
        manager_pbe = QuantumTrainingManager(config_pbe)
        assert manager_pbe.dft_functional == "pbe"

        # Test B3LYP functional
        config_b3lyp = {"quantum_training": True, "dft_functional": "b3lyp"}
        manager_b3lyp = QuantumTrainingManager(config_b3lyp)
        assert manager_b3lyp.dft_functional == "b3lyp"

    def test_scf_mixing_parameter(self):
        """Test SCF mixing parameter configuration."""
        config = {
            "quantum_training": True,
            "scf_mixing_parameter": 0.7,
            "max_scf_iterations": 20,
        }
        manager = QuantumTrainingManager(config)

        assert manager.scf_mixing_parameter == 0.7
        assert manager.max_scf_iterations == 20


class TestAdvancedQuantumTraining:
    """Test advanced quantum training workflows."""

    def test_quantum_configuration_composition(self, mock_model):
        """Test quantum-specific configuration composition."""
        dft_config = DFTConfig(functional="PBE")
        scf_config = SCFConfig(max_iterations=10, tolerance=1e-4)

        config = TrainingConfig(
            learning_rate=1e-3,
            dft_config=dft_config,
            scf_config=scf_config,
        )
        trainer = Trainer(mock_model, config)

        # Verify configuration
        assert trainer.config.dft_config is not None
        assert trainer.config.dft_config.functional == "PBE"
        assert trainer.config.scf_config is not None
        assert trainer.config.scf_config.max_iterations == 10
        assert trainer.config.scf_config.tolerance == 1e-4

    def test_custom_dft_workflow(self, mock_model, sample_data):
        """Test custom DFT workflow integration."""
        dft_config = DFTConfig(functional="pbe")
        scf_config = SCFConfig(max_iterations=20)

        config = TrainingConfig(
            learning_rate=1e-3,
            dft_config=dft_config,
            scf_config=scf_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data

        # Execute a training step with the DFT workflow
        loss, metrics = trainer.training_step(x[:10], y[:10])
        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)

    def test_quantum_manager_dft_energy(self):
        """Test DFT energy computation via QuantumTrainingManager."""
        config = {"quantum_training": True, "dft_functional": "pbe"}
        manager = QuantumTrainingManager(config)

        # Test DFT energy computation
        y_pred = jnp.array([[1.0], [2.0], [3.0]])
        dft_energy = manager.compute_dft_energy(y_pred)

        assert isinstance(dft_energy, jax.Array)
        assert dft_energy > 0

    def test_quantum_ml_workflow(self, mock_model, sample_data):
        """Test quantum machine learning workflow with specialized components."""
        dft_config = DFTConfig(functional="pbe")
        scf_config = SCFConfig(tolerance=1e-6)

        config = TrainingConfig(
            learning_rate=1e-3,
            dft_config=dft_config,
            scf_config=scf_config,
        )
        trainer = Trainer(mock_model, config)

        x, y = sample_data

        # Execute a training step with the quantum ML workflow
        loss, metrics = trainer.training_step(x[:10], y[:10])
        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)
