"""Tests for composable physics configuration classes.

Following strict TDD principles - these tests are written FIRST to define
the expected behavior of the composable physics configuration system.
"""

from opifex.core.training.physics_configs import (
    BoundaryConfig,
    ConservationConfig,
    ConstraintConfig,
    DFTConfig,
    ElectronicStructureConfig,
    LoggingConfig,
    MetricsTrackingConfig,
    MultiScaleConfig,
    PerformanceConfig,
    SCFConfig,
)


class TestConstraintConfig:
    """Test ConstraintConfig for physics constraint enforcement."""

    def test_default_initialization(self):
        """Test ConstraintConfig with default values."""
        config = ConstraintConfig()

        assert config.constraints == []
        assert config.weights == {}
        assert config.adaptive_weighting is False
        assert config.adaptation_rate == 0.1
        assert config.violation_threshold == 0.01
        assert config.violation_monitoring is False

    def test_custom_initialization(self):
        """Test ConstraintConfig with custom values."""
        config = ConstraintConfig(
            constraints=["energy_conservation", "momentum_conservation"],
            weights={"energy_conservation": 0.6, "momentum_conservation": 0.4},
            adaptive_weighting=True,
            adaptation_rate=0.2,
            violation_threshold=0.05,
            violation_monitoring=True,
        )

        assert config.constraints == ["energy_conservation", "momentum_conservation"]
        assert config.weights == {
            "energy_conservation": 0.6,
            "momentum_conservation": 0.4,
        }
        assert config.adaptive_weighting is True
        assert config.adaptation_rate == 0.2
        assert config.violation_threshold == 0.05
        assert config.violation_monitoring is True

    def test_constraint_weights_validation(self):
        """Test that constraint weights sum validation works."""
        # Valid weights (don't need to sum to 1)
        config = ConstraintConfig(
            constraints=["energy_conservation"],
            weights={"energy_conservation": 0.5},
        )
        assert config.weights == {"energy_conservation": 0.5}


class TestConservationConfig:
    """Test ConservationConfig for conservation law enforcement."""

    def test_default_initialization(self):
        """Test ConservationConfig with default values."""
        config = ConservationConfig()

        assert config.laws == []
        assert config.energy_tolerance == 1e-6
        assert config.energy_monitoring is False
        assert config.momentum_tolerance == 1e-5
        assert config.momentum_components == []
        assert config.symmetry_tolerance == 1e-6
        assert config.symmetry_groups == []
        assert config.particle_tolerance == 1e-4
        assert config.target_particle_number == 0.0

    def test_custom_initialization(self):
        """Test ConservationConfig with custom values."""
        config = ConservationConfig(
            laws=["energy", "momentum", "symmetry"],
            energy_tolerance=1e-7,
            energy_monitoring=True,
            momentum_tolerance=1e-6,
            momentum_components=["x", "y", "z"],
            symmetry_tolerance=1e-7,
            symmetry_groups=["point_group"],
            particle_tolerance=1e-5,
            target_particle_number=10.0,
        )

        assert config.laws == ["energy", "momentum", "symmetry"]
        assert config.energy_tolerance == 1e-7
        assert config.energy_monitoring is True
        assert config.momentum_tolerance == 1e-6
        assert config.momentum_components == ["x", "y", "z"]
        assert config.symmetry_tolerance == 1e-7
        assert config.symmetry_groups == ["point_group"]
        assert config.particle_tolerance == 1e-5
        assert config.target_particle_number == 10.0


class TestMultiScaleConfig:
    """Test MultiScaleConfig for multi-scale physics integration."""

    def test_default_initialization(self):
        """Test MultiScaleConfig with default values."""
        config = MultiScaleConfig()

        assert config.scales == []
        assert config.weights == {}
        assert config.coupling is False

    def test_custom_initialization(self):
        """Test MultiScaleConfig with custom values."""
        config = MultiScaleConfig(
            scales=["molecular", "atomic", "electronic"],
            weights={"molecular": 0.4, "atomic": 0.4, "electronic": 0.2},
            coupling=True,
        )

        assert config.scales == ["molecular", "atomic", "electronic"]
        assert config.weights == {"molecular": 0.4, "atomic": 0.4, "electronic": 0.2}
        assert config.coupling is True


class TestBoundaryConfig:
    """Test BoundaryConfig for boundary conditions."""

    def test_default_initialization(self):
        """Test BoundaryConfig with default values."""
        config = BoundaryConfig()

        assert config.weight == 0.0
        assert config.enforce is False

    def test_custom_initialization(self):
        """Test BoundaryConfig with custom values."""
        config = BoundaryConfig(weight=0.5, enforce=True)

        assert config.weight == 0.5
        assert config.enforce is True


class TestDFTConfig:
    """Test DFTConfig for density functional theory."""

    def test_default_initialization(self):
        """Test DFTConfig with default values."""
        config = DFTConfig()

        assert config.functional == "pbe"
        assert config.electron_density_threshold == 1e-8
        assert config.exchange_correlation_weight == 0.3

    def test_custom_initialization(self):
        """Test DFTConfig with custom values."""
        config = DFTConfig(
            functional="pbe0",
            electron_density_threshold=1e-9,
            exchange_correlation_weight=0.4,
        )

        assert config.functional == "pbe0"
        assert config.electron_density_threshold == 1e-9
        assert config.exchange_correlation_weight == 0.4


class TestSCFConfig:
    """Test SCFConfig for self-consistent field calculations."""

    def test_default_initialization(self):
        """Test SCFConfig with default values."""
        config = SCFConfig()

        assert config.max_iterations == 50
        assert config.tolerance == 1e-6
        assert config.mixing_parameter == 0.5

    def test_custom_initialization(self):
        """Test SCFConfig with custom values."""
        config = SCFConfig(max_iterations=100, tolerance=1e-7, mixing_parameter=0.7)

        assert config.max_iterations == 100
        assert config.tolerance == 1e-7
        assert config.mixing_parameter == 0.7


class TestElectronicStructureConfig:
    """Test ElectronicStructureConfig for electronic structure optimization."""

    def test_default_initialization(self):
        """Test ElectronicStructureConfig with default values."""
        config = ElectronicStructureConfig()

        assert config.enabled is False
        assert config.orbital_optimization is False
        assert config.temperature == 300.0

    def test_custom_initialization(self):
        """Test ElectronicStructureConfig with custom values."""
        config = ElectronicStructureConfig(
            enabled=True, orbital_optimization=True, temperature=500.0
        )

        assert config.enabled is True
        assert config.orbital_optimization is True
        assert config.temperature == 500.0


class TestMetricsTrackingConfig:
    """Test MetricsTrackingConfig for physics metrics tracking."""

    def test_default_initialization(self):
        """Test MetricsTrackingConfig with default values."""
        config = MetricsTrackingConfig()

        assert config.detailed is True
        assert config.physics_metrics == ["chemical_accuracy"]
        assert config.quantum_states is False
        assert config.state_history is False

    def test_custom_initialization(self):
        """Test MetricsTrackingConfig with custom values."""
        config = MetricsTrackingConfig(
            detailed=False,
            physics_metrics=["chemical_accuracy", "scf_convergence"],
            quantum_states=True,
            state_history=True,
        )

        assert config.detailed is False
        assert config.physics_metrics == ["chemical_accuracy", "scf_convergence"]
        assert config.quantum_states is True
        assert config.state_history is True


class TestLoggingConfig:
    """Test LoggingConfig for physics-specific logging."""

    def test_default_initialization(self):
        """Test LoggingConfig with default values."""
        config = LoggingConfig()

        assert config.backends == ["tensorboard"]
        assert config.level == "INFO"
        assert config.frequency == 1
        assert config.real_time is False
        assert config.alert_thresholds == {}

    def test_custom_initialization(self):
        """Test LoggingConfig with custom values."""
        config = LoggingConfig(
            backends=["tensorboard", "wandb"],
            level="DEBUG",
            frequency=10,
            real_time=True,
            alert_thresholds={"loss": 1.0, "violation": 0.01},
        )

        assert config.backends == ["tensorboard", "wandb"]
        assert config.level == "DEBUG"
        assert config.frequency == 10
        assert config.real_time is True
        assert config.alert_thresholds == {"loss": 1.0, "violation": 0.01}


class TestPerformanceConfig:
    """Test PerformanceConfig for performance analytics."""

    def test_default_initialization(self):
        """Test PerformanceConfig with default values."""
        config = PerformanceConfig()

        assert config.analytics is False
        assert config.timing is False
        assert config.memory is False
        assert config.convergence is False

    def test_custom_initialization(self):
        """Test PerformanceConfig with custom values."""
        config = PerformanceConfig(
            analytics=True, timing=True, memory=True, convergence=True
        )

        assert config.analytics is True
        assert config.timing is True
        assert config.memory is True
        assert config.convergence is True


class TestConfigComposition:
    """Test composition of physics configs with TrainingConfig."""

    def test_trainingconfig_accepts_physics_configs(self):
        """Test that TrainingConfig can compose with physics configs."""
        from opifex.core.training.config import TrainingConfig

        # Create physics configs
        constraint_config = ConstraintConfig(
            constraints=["energy_conservation"], adaptive_weighting=True
        )
        conservation_config = ConservationConfig(
            laws=["energy"], energy_monitoring=True
        )
        boundary_config = BoundaryConfig(weight=0.5, enforce=True)

        # Compose into TrainingConfig
        config = TrainingConfig(
            num_epochs=100,
            learning_rate=1e-3,
            constraint_config=constraint_config,
            conservation_config=conservation_config,
            boundary_config=boundary_config,
        )

        # Verify composition
        assert config.constraint_config is constraint_config
        assert config.conservation_config is conservation_config
        assert config.boundary_config is boundary_config
        assert config.constraint_config.constraints == ["energy_conservation"]  # pyright: ignore[reportOptionalMemberAccess]
        assert config.conservation_config.laws == ["energy"]  # pyright: ignore[reportOptionalMemberAccess]
        assert config.boundary_config.weight == 0.5  # pyright: ignore[reportOptionalMemberAccess]

    def test_trainingconfig_accepts_quantum_configs(self):
        """Test that TrainingConfig can compose with quantum configs."""
        from opifex.core.training.config import QuantumTrainingConfig, TrainingConfig

        # Create quantum configs
        quantum_config = QuantumTrainingConfig(chemical_accuracy_target=1e-3)
        dft_config = DFTConfig(functional="pbe0")
        scf_config = SCFConfig(max_iterations=100)
        electronic_config = ElectronicStructureConfig(enabled=True)

        # Compose into TrainingConfig
        config = TrainingConfig(
            num_epochs=100,
            quantum_config=quantum_config,
            dft_config=dft_config,
            scf_config=scf_config,
            electronic_structure_config=electronic_config,
        )

        # Verify composition
        assert config.quantum_config is quantum_config
        assert config.dft_config is dft_config
        assert config.scf_config is scf_config
        assert config.electronic_structure_config is electronic_config

    def test_trainingconfig_optional_physics_configs(self):
        """Test that all physics configs are optional."""
        from opifex.core.training.config import TrainingConfig

        # Create TrainingConfig without physics configs
        config = TrainingConfig(num_epochs=100, learning_rate=1e-3)

        # Verify all physics configs are None
        assert config.constraint_config is None
        assert config.conservation_config is None
        assert config.multiscale_config is None
        assert config.boundary_config is None
        assert config.dft_config is None
        assert config.scf_config is None
        assert config.electronic_structure_config is None
        assert config.metrics_tracking_config is None
        assert config.logging_config is None
        assert config.performance_config is None
