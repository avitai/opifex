"""Tests for opifex.core.training.config module.

This test suite follows strict TDD principles:
1. Tests are written FIRST to define the API
2. Implementation comes AFTER tests are defined
3. Tests define expected behavior, not accommodate implementation
4. Comprehensive coverage of all config types and edge cases
"""

from __future__ import annotations

import pytest

from opifex.core.training.config import (
    CheckpointConfig,
    LossConfig,
    MetaOptimizerConfig,
    OptimizationConfig,
    QuantumTrainingConfig,
    TrainingConfig,
    ValidationConfig,
)


class TestTrainingConfig:
    """Test cases for TrainingConfig dataclass."""

    def test_training_config_defaults(self):
        """Test TrainingConfig creation with default values."""
        config = TrainingConfig()

        assert config.num_epochs == 100
        assert config.batch_size == 32
        assert config.learning_rate == 1e-3
        assert config.validation_frequency == 10
        assert config.checkpoint_frequency == 50
        assert config.progress_callback is None
        assert config.verbose is True

    def test_training_config_custom_values(self):
        """Test TrainingConfig creation with custom values."""
        config = TrainingConfig(
            num_epochs=200,
            batch_size=64,
            learning_rate=5e-4,
            validation_frequency=5,
            checkpoint_frequency=25,
            verbose=False,
        )

        assert config.num_epochs == 200
        assert config.batch_size == 64
        assert config.learning_rate == 5e-4
        assert config.validation_frequency == 5
        assert config.checkpoint_frequency == 25
        assert config.verbose is False

    def test_training_config_sub_configs(self):
        """Test TrainingConfig has proper sub-configuration instances."""
        config = TrainingConfig()

        assert isinstance(config.loss_config, LossConfig)
        assert isinstance(config.optimization_config, OptimizationConfig)
        assert isinstance(config.validation_config, ValidationConfig)
        assert isinstance(config.checkpoint_config, CheckpointConfig)
        assert config.quantum_config is None

    def test_training_config_post_init_sync(self):
        """Test that __post_init__ syncs values to sub-configs."""
        config = TrainingConfig(
            learning_rate=5e-4,
            validation_frequency=20,
            checkpoint_frequency=100,
        )

        # Check that sub-configs are synchronized
        assert config.optimization_config.learning_rate == 5e-4
        assert config.validation_config.validation_frequency == 20
        assert config.checkpoint_config.save_frequency == 100

    def test_training_config_with_quantum_config(self):
        """Test TrainingConfig with quantum configuration."""
        quantum_cfg = QuantumTrainingConfig(
            chemical_accuracy_target=2e-3,
            scf_max_iterations=50,
        )
        config = TrainingConfig(quantum_config=quantum_cfg)

        assert config.quantum_config is not None
        assert isinstance(config.quantum_config, QuantumTrainingConfig)
        assert config.quantum_config.chemical_accuracy_target == 2e-3
        assert config.quantum_config.scf_max_iterations == 50


class TestLossConfig:
    """Test cases for LossConfig dataclass."""

    def test_loss_config_defaults(self):
        """Test LossConfig creation with default values."""
        config = LossConfig()

        assert config.loss_type == "mse"
        assert config.physics_weight == 1.0
        assert config.boundary_weight == 1.0
        assert config.quantum_constraint_weight == 1.0
        assert config.density_constraint_weight == 1.0
        assert config.regularization_weight == 0.0

    def test_loss_config_custom_values(self):
        """Test LossConfig creation with custom values."""
        config = LossConfig(
            loss_type="mae",
            physics_weight=2.0,
            boundary_weight=0.5,
            regularization_weight=1e-4,
        )

        assert config.loss_type == "mae"
        assert config.physics_weight == 2.0
        assert config.boundary_weight == 0.5
        assert config.regularization_weight == 1e-4

    def test_loss_config_quantum_weights(self):
        """Test LossConfig with quantum-specific weights."""
        config = LossConfig(
            quantum_constraint_weight=3.0,
            density_constraint_weight=2.5,
        )

        assert config.quantum_constraint_weight == 3.0
        assert config.density_constraint_weight == 2.5


class TestOptimizationConfig:
    """Test cases for OptimizationConfig dataclass."""

    def test_optimization_config_defaults(self):
        """Test OptimizationConfig creation with default values."""
        config = OptimizationConfig()

        assert config.optimizer == "adam"
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 0.0
        assert config.momentum == 0.9
        assert config.eps == 1e-8
        assert config.beta1 == 0.9
        assert config.beta2 == 0.999

    def test_optimization_config_adam_custom(self):
        """Test OptimizationConfig for Adam optimizer with custom params."""
        config = OptimizationConfig(
            optimizer="adam",
            learning_rate=5e-4,
            beta1=0.95,
            beta2=0.9999,
            eps=1e-7,
        )

        assert config.optimizer == "adam"
        assert config.learning_rate == 5e-4
        assert config.beta1 == 0.95
        assert config.beta2 == 0.9999
        assert config.eps == 1e-7

    def test_optimization_config_sgd_with_momentum(self):
        """Test OptimizationConfig for SGD with momentum."""
        config = OptimizationConfig(
            optimizer="sgd",
            learning_rate=1e-2,
            momentum=0.95,
        )

        assert config.optimizer == "sgd"
        assert config.learning_rate == 1e-2
        assert config.momentum == 0.95

    def test_optimization_config_adamw_with_decay(self):
        """Test OptimizationConfig for AdamW with weight decay."""
        config = OptimizationConfig(
            optimizer="adamw",
            learning_rate=3e-4,
            weight_decay=0.01,
        )

        assert config.optimizer == "adamw"
        assert config.learning_rate == 3e-4
        assert config.weight_decay == 0.01


class TestValidationConfig:
    """Test cases for ValidationConfig dataclass."""

    def test_validation_config_defaults(self):
        """Test ValidationConfig creation with default values."""
        config = ValidationConfig()

        assert config.validation_frequency == 10
        assert config.early_stopping_patience == 20
        assert config.early_stopping_min_delta == 1e-6
        assert config.compute_val_metrics is True

    def test_validation_config_custom_values(self):
        """Test ValidationConfig creation with custom values."""
        config = ValidationConfig(
            validation_frequency=5,
            early_stopping_patience=50,
            early_stopping_min_delta=1e-5,
            compute_val_metrics=False,
        )

        assert config.validation_frequency == 5
        assert config.early_stopping_patience == 50
        assert config.early_stopping_min_delta == 1e-5
        assert config.compute_val_metrics is False


class TestCheckpointConfig:
    """Test cases for CheckpointConfig dataclass."""

    def test_checkpoint_config_defaults(self):
        """Test CheckpointConfig creation with default values."""
        config = CheckpointConfig()

        assert config.checkpoint_dir == "./checkpoints"
        assert config.save_frequency == 50
        assert config.max_to_keep == 5
        assert config.save_best_only is False

    def test_checkpoint_config_custom_values(self):
        """Test CheckpointConfig creation with custom values."""
        config = CheckpointConfig(
            checkpoint_dir="/tmp/my_checkpoints",  # noqa: S108
            save_frequency=100,
            max_to_keep=10,
            save_best_only=True,
        )

        assert config.checkpoint_dir == "/tmp/my_checkpoints"  # noqa: S108
        assert config.save_frequency == 100
        assert config.max_to_keep == 10
        assert config.save_best_only is True


class TestQuantumTrainingConfig:
    """Test cases for QuantumTrainingConfig dataclass."""

    def test_quantum_config_defaults(self):
        """Test QuantumTrainingConfig creation with default values."""
        config = QuantumTrainingConfig()

        assert config.chemical_accuracy_target == 1e-3
        assert config.scf_max_iterations == 100
        assert config.scf_tolerance == 1e-6
        assert config.enable_symmetry_enforcement is True
        assert config.enable_density_constraints is True
        assert config.enable_energy_conservation is True

    def test_quantum_config_custom_values(self):
        """Test QuantumTrainingConfig creation with custom values."""
        config = QuantumTrainingConfig(
            chemical_accuracy_target=5e-4,
            scf_max_iterations=200,
            scf_tolerance=1e-7,
            enable_symmetry_enforcement=False,
            enable_density_constraints=False,
            enable_energy_conservation=False,
        )

        assert config.chemical_accuracy_target == 5e-4
        assert config.scf_max_iterations == 200
        assert config.scf_tolerance == 1e-7
        assert config.enable_symmetry_enforcement is False
        assert config.enable_density_constraints is False
        assert config.enable_energy_conservation is False


class TestMetaOptimizerConfig:
    """Test cases for MetaOptimizerConfig dataclass."""

    def test_meta_optimizer_config_defaults(self):
        """Test MetaOptimizerConfig creation with default values."""
        config = MetaOptimizerConfig()

        assert config.meta_algorithm == "l2o"
        assert config.base_optimizer == "adam"
        assert config.meta_learning_rate == 1e-4
        assert config.adaptation_steps == 10
        assert config.warm_start_strategy == "previous_params"
        assert config.performance_tracking is True
        assert config.memory_efficient is True
        assert config.quantum_aware is False
        assert config.scf_adaptation is False
        assert config.energy_convergence_tracking is False
        assert config.chemical_accuracy_target == 1e-3

    def test_meta_optimizer_config_custom_values(self):
        """Test MetaOptimizerConfig creation with custom values."""
        config = MetaOptimizerConfig(
            meta_algorithm="adaptive_lr",
            base_optimizer="sgd",
            meta_learning_rate=5e-5,
            adaptation_steps=20,
            warm_start_strategy="similar_problems",
            performance_tracking=False,
            memory_efficient=False,
        )

        assert config.meta_algorithm == "adaptive_lr"
        assert config.base_optimizer == "sgd"
        assert config.meta_learning_rate == 5e-5
        assert config.adaptation_steps == 20
        assert config.warm_start_strategy == "similar_problems"
        assert config.performance_tracking is False
        assert config.memory_efficient is False

    def test_meta_optimizer_config_quantum_aware(self):
        """Test MetaOptimizerConfig with quantum-aware settings."""
        config = MetaOptimizerConfig(
            quantum_aware=True,
            scf_adaptation=True,
            energy_convergence_tracking=True,
            chemical_accuracy_target=2e-3,
        )

        assert config.quantum_aware is True
        assert config.scf_adaptation is True
        assert config.energy_convergence_tracking is True
        assert config.chemical_accuracy_target == 2e-3

    def test_meta_optimizer_config_validation_invalid_algorithm(self):
        """Test MetaOptimizerConfig validation for invalid algorithm."""
        with pytest.raises(ValueError, match="Invalid meta algorithm"):
            MetaOptimizerConfig(meta_algorithm="invalid_algo")

    def test_meta_optimizer_config_validation_invalid_optimizer(self):
        """Test MetaOptimizerConfig validation for invalid base optimizer."""
        with pytest.raises(ValueError, match="Invalid base optimizer"):
            MetaOptimizerConfig(base_optimizer="invalid_opt")

    def test_meta_optimizer_config_valid_algorithms(self):
        """Test MetaOptimizerConfig accepts all valid algorithms."""
        for algo in ["l2o", "adaptive_lr", "warm_start"]:
            config = MetaOptimizerConfig(meta_algorithm=algo)
            assert config.meta_algorithm == algo

    def test_meta_optimizer_config_valid_optimizers(self):
        """Test MetaOptimizerConfig accepts all valid base optimizers."""
        for opt in ["adam", "sgd", "rmsprop", "adamw"]:
            config = MetaOptimizerConfig(base_optimizer=opt)
            assert config.base_optimizer == opt


class TestConfigIntegration:
    """Integration tests for configuration classes."""

    def test_training_config_full_composition(self):
        """Test TrainingConfig with all sub-configs specified."""
        loss_cfg = LossConfig(loss_type="mae", physics_weight=2.0)
        opt_cfg = OptimizationConfig(optimizer="adamw", learning_rate=5e-4)
        val_cfg = ValidationConfig(validation_frequency=5)
        ckpt_cfg = CheckpointConfig(max_to_keep=10)
        quantum_cfg = QuantumTrainingConfig(scf_max_iterations=50)

        config = TrainingConfig(
            num_epochs=200,
            batch_size=64,
            learning_rate=5e-4,  # Should sync to opt_cfg
            validation_frequency=5,  # Should sync to val_cfg
            checkpoint_frequency=100,  # Should sync to ckpt_cfg
            loss_config=loss_cfg,
            optimization_config=opt_cfg,
            validation_config=val_cfg,
            checkpoint_config=ckpt_cfg,
            quantum_config=quantum_cfg,
        )

        # Verify main config
        assert config.num_epochs == 200
        assert config.batch_size == 64

        # Verify sub-configs are properly assigned
        assert config.loss_config.loss_type == "mae"
        assert config.optimization_config.optimizer == "adamw"
        assert config.validation_config.validation_frequency == 5
        assert config.checkpoint_config.max_to_keep == 10
        assert config.quantum_config.scf_max_iterations == 50  # pyright: ignore[reportOptionalMemberAccess]

        # Verify synchronization (post_init should update sub-configs)
        assert config.optimization_config.learning_rate == 5e-4
        assert config.validation_config.validation_frequency == 5
        assert config.checkpoint_config.save_frequency == 100

    def test_config_immutability_check(self):
        """Test that configs can be modified after creation (dataclasses are mutable)."""
        config = TrainingConfig()

        # Dataclasses are mutable by default
        config.num_epochs = 500
        assert config.num_epochs == 500

        config.learning_rate = 1e-4
        assert config.learning_rate == 1e-4


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_training_config_zero_epochs(self):
        """Test TrainingConfig with zero epochs (edge case)."""
        config = TrainingConfig(num_epochs=0)
        assert config.num_epochs == 0

    def test_training_config_negative_batch_size_allowed(self):
        """Test that negative batch size is not validated (design choice)."""
        # No validation in dataclass - responsibility of user/trainer
        config = TrainingConfig(batch_size=-1)
        assert config.batch_size == -1

    def test_optimization_config_zero_learning_rate(self):
        """Test OptimizationConfig with zero learning rate."""
        config = OptimizationConfig(learning_rate=0.0)
        assert config.learning_rate == 0.0

    def test_validation_config_zero_frequency(self):
        """Test ValidationConfig with zero validation frequency."""
        config = ValidationConfig(validation_frequency=0)
        assert config.validation_frequency == 0

    def test_checkpoint_config_empty_dir(self):
        """Test CheckpointConfig with empty directory string."""
        config = CheckpointConfig(checkpoint_dir="")
        assert config.checkpoint_dir == ""

    def test_quantum_config_extreme_values(self):
        """Test QuantumTrainingConfig with extreme values."""
        config = QuantumTrainingConfig(
            chemical_accuracy_target=1e-10,
            scf_max_iterations=10000,
            scf_tolerance=1e-12,
        )

        assert config.chemical_accuracy_target == 1e-10
        assert config.scf_max_iterations == 10000
        assert config.scf_tolerance == 1e-12

    def test_meta_optimizer_config_extreme_adaptation_steps(self):
        """Test MetaOptimizerConfig with extreme adaptation steps."""
        config = MetaOptimizerConfig(adaptation_steps=1000)
        assert config.adaptation_steps == 1000

    def test_loss_config_all_zero_weights(self):
        """Test LossConfig with all zero weights (edge case)."""
        config = LossConfig(
            physics_weight=0.0,
            boundary_weight=0.0,
            quantum_constraint_weight=0.0,
            density_constraint_weight=0.0,
            regularization_weight=0.0,
        )

        assert config.physics_weight == 0.0
        assert config.boundary_weight == 0.0
        assert config.quantum_constraint_weight == 0.0
        assert config.density_constraint_weight == 0.0
        assert config.regularization_weight == 0.0
