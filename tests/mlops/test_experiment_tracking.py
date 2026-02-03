"""Tests for MLOps experiment tracking functionality."""

from unittest.mock import Mock

import pytest


class TestPhysicsMetadata:
    """Test physics metadata class."""

    def test_initialization_default(self):
        """Test default initialization."""
        from opifex.mlops.experiment import PhysicsMetadata

        metadata = PhysicsMetadata()
        assert metadata.pde_type is None
        assert metadata.dimensionality is None
        assert metadata.boundary_conditions is None
        assert metadata.domain_bounds is None
        assert metadata.time_horizon is None
        assert metadata.material_properties is None
        assert metadata.system_parameters is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        from opifex.mlops.experiment import PhysicsMetadata

        metadata = PhysicsMetadata(
            pde_type="navier_stokes",
            dimensionality=2,
            boundary_conditions=["dirichlet"],
            domain_bounds=(0.0, 1.0, 0.0, 1.0),
            time_horizon=1.0,
            material_properties={"viscosity": 0.01},
            system_parameters={"reynolds_number": 1000},
        )
        assert metadata.pde_type == "navier_stokes"
        assert metadata.dimensionality == 2
        assert metadata.boundary_conditions == ["dirichlet"]
        assert metadata.domain_bounds == (0.0, 1.0, 0.0, 1.0)
        assert metadata.time_horizon == 1.0
        assert metadata.material_properties is not None
        assert metadata.material_properties["viscosity"] == 0.01
        assert metadata.system_parameters is not None
        assert metadata.system_parameters["reynolds_number"] == 1000


class TestExperimentConfig:
    """Test experiment configuration."""

    def test_initialization_default(self):
        """Test default initialization."""
        from opifex.mlops.experiment import ExperimentConfig, Framework, PhysicsDomain

        config = ExperimentConfig(
            name="test_experiment",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
        )
        assert config.name == "test_experiment"
        assert config.physics_domain == PhysicsDomain.NEURAL_OPERATORS
        assert config.framework == Framework.JAX
        assert config.description is None
        assert config.tags == []
        assert config.backend == "auto"
        assert config.research_group is None

    def test_initialization_custom(self):
        """Test custom initialization."""
        from opifex.mlops.experiment import (
            ExperimentConfig,
            Framework,
            PhysicsDomain,
            PhysicsMetadata,
        )

        metadata = PhysicsMetadata(pde_type="navier_stokes")
        config = ExperimentConfig(
            name="fluid_dynamics_experiment",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
            description="Test fluid dynamics experiment",
            tags=["neural_operator", "fluid_dynamics"],
            physics_metadata=metadata,
            backend="mlflow",
            research_group="opifex_team",
            project_id="fluid_sim_2024",
            random_seed=42,
        )
        assert config.name == "fluid_dynamics_experiment"
        assert config.physics_domain == PhysicsDomain.NEURAL_OPERATORS
        assert config.framework == Framework.JAX
        assert config.description == "Test fluid dynamics experiment"
        assert config.tags == ["neural_operator", "fluid_dynamics"]
        assert config.physics_metadata == metadata
        assert config.backend == "mlflow"
        assert config.research_group == "opifex_team"
        assert config.project_id == "fluid_sim_2024"
        assert config.random_seed == 42


class TestEnums:
    """Test enum definitions."""

    def test_physics_domain_enum(self):
        """Test PhysicsDomain enum."""
        from opifex.mlops.experiment import PhysicsDomain

        assert PhysicsDomain.NEURAL_OPERATORS.value == "neural-operators"
        assert PhysicsDomain.L2O.value == "l2o"
        assert PhysicsDomain.NEURAL_DFT.value == "neural-dft"
        assert PhysicsDomain.PINN.value == "pinn"
        assert PhysicsDomain.QUANTUM_COMPUTING.value == "quantum-computing"

    def test_framework_enum(self):
        """Test Framework enum."""
        from opifex.mlops.experiment import Framework

        assert Framework.JAX.value == "jax"
        assert Framework.PYTORCH.value == "pytorch"
        assert Framework.TENSORFLOW.value == "tensorflow"


class TestExperimentTracker:
    """Test experiment tracker factory."""

    def test_initialization(self):
        """Test ExperimentTracker initialization."""
        from opifex.mlops.experiment import ExperimentTracker

        tracker = ExperimentTracker()
        assert tracker.default_backend == "auto"
        assert tracker._backend_registry == {}

    def test_register_backend(self):
        """Test backend registration."""
        from opifex.mlops.experiment import ExperimentTracker

        tracker = ExperimentTracker()
        mock_backend_class = Mock()

        tracker.register_backend("test_backend", mock_backend_class)

        assert "test_backend" in tracker._backend_registry
        assert tracker._backend_registry["test_backend"] == mock_backend_class


class TestMLOpsImports:
    """Test MLOps import functionality."""

    def test_basic_imports(self):
        """Test that basic MLOps components can be imported."""
        from opifex.mlops import (
            ExperimentConfig,
            ExperimentTracker,
            Framework,
            MLFLOW_AVAILABLE,
            PhysicsDomain,
            PhysicsMetadata,
        )

        # Test that classes can be instantiated
        metadata = PhysicsMetadata()
        assert metadata is not None

        config = ExperimentConfig(
            name="test",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
        )
        assert config is not None

        tracker = ExperimentTracker()
        assert tracker is not None

        # Test that MLFLOW_AVAILABLE is a boolean
        assert isinstance(MLFLOW_AVAILABLE, bool)

    def test_metadata_serialization(self):
        """Test that metadata can be converted to dict."""
        from opifex.mlops.experiment import PhysicsMetadata

        metadata = PhysicsMetadata(pde_type="navier_stokes", dimensionality=2)
        metadata_dict = metadata.__dict__
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["pde_type"] == "navier_stokes"
        assert metadata_dict["dimensionality"] == 2

    def test_experiment_config_with_metadata(self):
        """Test experiment config with physics metadata."""
        from opifex.mlops.experiment import (
            ExperimentConfig,
            Framework,
            PhysicsDomain,
            PhysicsMetadata,
        )

        metadata = PhysicsMetadata(pde_type="navier_stokes")
        config = ExperimentConfig(
            name="test_experiment",
            physics_domain=PhysicsDomain.NEURAL_OPERATORS,
            framework=Framework.JAX,
            physics_metadata=metadata,
        )
        assert config.physics_metadata == metadata
        assert config.physics_metadata is not None
        assert config.physics_metadata.pde_type == "navier_stokes"


if __name__ == "__main__":
    pytest.main([__file__])
