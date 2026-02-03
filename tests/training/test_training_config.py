"""Tests for training configuration module."""

from __future__ import annotations

from opifex.core.training.config import (
    CheckpointConfig,
    LossConfig,
    OptimizationConfig,
    QuantumTrainingConfig,
    TrainingConfig,
    ValidationConfig,
)


class TestQuantumTrainingConfig:
    """Test quantum training configuration."""

    def test_default_initialization(self):
        """Test default quantum config initialization."""
        config = QuantumTrainingConfig()

        assert config.chemical_accuracy_target == 1e-3
        assert config.scf_max_iterations == 100
        assert config.scf_tolerance == 1e-6
        assert config.enable_symmetry_enforcement is True
        assert config.enable_density_constraints is True
        assert config.enable_energy_conservation is True

    def test_custom_initialization(self):
        """Test custom quantum config initialization."""
        config = QuantumTrainingConfig(
            chemical_accuracy_target=5e-4,
            scf_max_iterations=200,
            scf_tolerance=1e-8,
            enable_symmetry_enforcement=False,
        )

        assert config.chemical_accuracy_target == 5e-4
        assert config.scf_max_iterations == 200
        assert config.scf_tolerance == 1e-8
        assert config.enable_symmetry_enforcement is False
        assert config.enable_density_constraints is True


class TestLossConfig:
    """Test loss configuration."""

    def test_default_initialization(self):
        """Test default loss config initialization."""
        config = LossConfig()

        assert config.loss_type == "mse"
        assert config.physics_weight == 1.0
        assert config.boundary_weight == 1.0
        assert config.quantum_constraint_weight == 1.0
        assert config.density_constraint_weight == 1.0
        assert config.regularization_weight == 0.0

    def test_custom_initialization(self):
        """Test custom loss config initialization."""
        config = LossConfig(
            loss_type="mae",
            physics_weight=0.5,
            regularization_weight=0.01,
        )

        assert config.loss_type == "mae"
        assert config.physics_weight == 0.5
        assert config.regularization_weight == 0.01

    def test_loss_types(self):
        """Test different loss types."""
        for loss_type in ["mse", "mae", "quantum_energy"]:
            config = LossConfig(loss_type=loss_type)
            assert config.loss_type == loss_type


class TestOptimizationConfig:
    """Test optimization configuration."""

    def test_default_initialization(self):
        """Test default optimization config initialization."""
        config = OptimizationConfig()

        assert config.optimizer == "adam"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0
        assert config.momentum == 0.9
        assert config.eps == 1e-8
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999

    def test_adam_config(self):
        """Test Adam optimizer configuration."""
        config = OptimizationConfig(
            optimizer="adam",
            learning_rate=1e-4,
            beta1=0.95,
            beta2=0.9999,
        )

        assert config.optimizer == "adam"
        assert config.learning_rate == 1e-4
        assert config.beta1 == 0.95
        assert config.beta2 == 0.9999

    def test_sgd_config(self):
        """Test SGD optimizer configuration."""
        config = OptimizationConfig(
            optimizer="sgd",
            learning_rate=0.01,
            momentum=0.95,
        )

        assert config.optimizer == "sgd"
        assert config.learning_rate == 0.01
        assert config.momentum == 0.95


class TestValidationConfig:
    """Test validation configuration."""

    def test_default_initialization(self):
        """Test default validation config initialization."""
        config = ValidationConfig()

        assert config.validation_frequency == 10
        assert config.early_stopping_patience == 20
        assert config.early_stopping_min_delta == 1e-6
        assert config.compute_val_metrics is True

    def test_custom_initialization(self):
        """Test custom validation config initialization."""
        config = ValidationConfig(
            validation_frequency=5,
            early_stopping_patience=50,
            compute_val_metrics=False,
        )

        assert config.validation_frequency == 5
        assert config.early_stopping_patience == 50
        assert config.compute_val_metrics is False


class TestCheckpointConfig:
    """Test checkpoint configuration."""

    def test_default_initialization(self):
        """Test default checkpoint config initialization."""
        config = CheckpointConfig()

        assert config.checkpoint_dir == "./checkpoints"
        assert config.save_frequency == 50
        assert config.max_to_keep == 5
        assert config.save_best_only is False

    def test_custom_initialization(self):
        """Test custom checkpoint config initialization."""
        config = CheckpointConfig(
            checkpoint_dir="/tmp/my_checkpoints",  # noqa: S108
            save_frequency=10,
            max_to_keep=3,
            save_best_only=True,
        )

        assert config.checkpoint_dir == "/tmp/my_checkpoints"  # noqa: S108
        assert config.save_frequency == 10
        assert config.max_to_keep == 3
        assert config.save_best_only is True


class TestTrainingConfig:
    """Test main training configuration."""

    def test_default_initialization(self):
        """Test default training config initialization."""
        config = TrainingConfig()

        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.validation_frequency == 10
        assert config.checkpoint_frequency == 50
        assert config.verbose is True
        assert config.progress_callback is None

        # Check sub-configs are initialized
        assert isinstance(config.loss_config, LossConfig)
        assert isinstance(config.optimization_config, OptimizationConfig)
        assert isinstance(config.validation_config, ValidationConfig)
        assert isinstance(config.checkpoint_config, CheckpointConfig)
        assert config.quantum_config is None

    def test_custom_initialization(self):
        """Test custom training config initialization."""
        config = TrainingConfig(
            num_epochs=200,
            batch_size=64,
            learning_rate=1e-4,
            validation_frequency=5,
            checkpoint_frequency=25,
            verbose=False,
        )

        assert config.num_epochs == 200
        assert config.batch_size == 64
        assert config.learning_rate == 1e-4
        assert config.validation_frequency == 5
        assert config.checkpoint_frequency == 25
        assert config.verbose is False

    def test_post_init_updates_sub_configs(self):
        """Test __post_init__ updates sub-configs with main values."""
        config = TrainingConfig(
            learning_rate=5e-4,
            validation_frequency=15,
            checkpoint_frequency=30,
        )

        # Check that sub-configs were updated
        assert config.optimization_config.learning_rate == 5e-4
        assert config.validation_config.validation_frequency == 15
        assert config.checkpoint_config.save_frequency == 30

    def test_with_quantum_config(self):
        """Test training config with quantum configuration."""
        quantum_config = QuantumTrainingConfig(
            chemical_accuracy_target=5e-4,
            scf_max_iterations=150,
        )

        config = TrainingConfig(
            num_epochs=50,
            quantum_config=quantum_config,
        )

        assert config.quantum_config is not None
        assert config.quantum_config.chemical_accuracy_target == 5e-4
        assert config.quantum_config.scf_max_iterations == 150

    def test_with_custom_sub_configs(self):
        """Test training config with custom sub-configurations."""
        loss_config = LossConfig(loss_type="mae", physics_weight=0.5)
        opt_config = OptimizationConfig(optimizer="sgd", momentum=0.95)

        config = TrainingConfig(
            num_epochs=50,
            loss_config=loss_config,
            optimization_config=opt_config,
        )

        assert config.loss_config.loss_type == "mae"
        assert config.loss_config.physics_weight == 0.5
        assert config.optimization_config.optimizer == "sgd"
        assert config.optimization_config.momentum == 0.95

    def test_progress_callback_assignment(self):
        """Test assigning progress callback."""

        def dummy_callback(info):
            pass

        config = TrainingConfig(progress_callback=dummy_callback)

        assert config.progress_callback is dummy_callback

    def test_immutable_defaults(self):
        """Test that default factory creates new instances."""
        config1 = TrainingConfig()
        config2 = TrainingConfig()

        # Ensure different instances
        assert config1.loss_config is not config2.loss_config
        assert config1.optimization_config is not config2.optimization_config
        assert config1.validation_config is not config2.validation_config
        assert config1.checkpoint_config is not config2.checkpoint_config

    def test_validation_frequency_sync(self):
        """Test validation frequency synchronization between configs."""
        config = TrainingConfig(validation_frequency=7)

        assert config.validation_frequency == 7
        assert config.validation_config.validation_frequency == 7

    def test_checkpoint_frequency_sync(self):
        """Test checkpoint frequency synchronization between configs."""
        config = TrainingConfig(checkpoint_frequency=33)

        assert config.checkpoint_frequency == 33
        assert config.checkpoint_config.save_frequency == 33
