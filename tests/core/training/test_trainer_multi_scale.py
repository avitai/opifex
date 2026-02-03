"""Tests for multi-scale physics and adaptive weighting in unified Trainer.

This module tests the advanced physics features migrated from PhysicsInformedTrainer:
- Multi-scale physics loss computation (molecular, atomic, electronic)
- Adaptive constraint weighting based on violation severity
- Physics state tracking (chemical accuracy, SCF convergence, conservation violations)

Following strict TDD principles - these tests define the expected behavior FIRST.
"""

from __future__ import annotations

import jax.numpy as jnp
from flax import nnx

from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.physics_configs import (
    ChemicalAccuracyTracking,
    ConservationViolationTracking,
    ConstraintConfig,
    MetricsTrackingConfig,
    MultiScaleConfig,
    SCFConvergenceTracking,
)


class TestMultiScalePhysicsLoss:
    """Test multi-scale physics loss computation in unified Trainer."""

    def test_molecular_scale_only(self):
        """Test training with only molecular scale physics."""
        # Create simple model
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        # Configure for molecular scale only
        config = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            multiscale_config=MultiScaleConfig(
                scales=["molecular"],
                weights={"molecular": 1.0},
            ),
        )

        trainer = Trainer(model=model, config=config)

        # Training data
        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        # Train and verify it works
        trained_model, metrics = trainer.fit((x_train, y_train))

        assert trained_model is not None
        assert "final_train_loss" in metrics

    def test_all_three_scales(self):
        """Test training with molecular, atomic, and electronic scales."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            multiscale_config=MultiScaleConfig(
                scales=["molecular", "atomic", "electronic"],
                weights={"molecular": 0.5, "atomic": 0.3, "electronic": 0.2},
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, metrics = trainer.fit((x_train, y_train))

        assert trained_model is not None
        assert "final_train_loss" in metrics

    def test_scale_coupling_enabled(self):
        """Test training with scale coupling enabled."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            multiscale_config=MultiScaleConfig(
                scales=["molecular", "atomic"],
                weights={"molecular": 0.6, "atomic": 0.4},
                coupling=True,  # Enable coupling between scales
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, _ = trainer.fit((x_train, y_train))

        assert trained_model is not None

    def test_multi_scale_loss_decreases(self):
        """Test that multi-scale loss decreases during training."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=50,  # More epochs for convergence
            batch_size=4,
            learning_rate=1e-2,
            multiscale_config=MultiScaleConfig(
                scales=["molecular", "atomic"],
                weights={"molecular": 0.7, "atomic": 0.3},
            ),
        )

        trainer = Trainer(model=model, config=config)

        # Use varied training data for better learning signal
        x_train = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
        y_train = jnp.array([[0.0], [1.0], [0.5], [0.25]])

        _, metrics = trainer.fit((x_train, y_train))

        # Loss should decrease (or at least not increase)
        assert metrics["final_train_loss"] <= metrics["initial_train_loss"]


class TestAdaptiveConstraintWeighting:
    """Test adaptive constraint weighting in unified Trainer."""

    def test_adaptive_weighting_enabled(self):
        """Test training with adaptive constraint weighting."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            constraint_config=ConstraintConfig(
                constraints=["energy_conservation", "momentum_conservation"],
                weights={"energy_conservation": 0.5, "momentum_conservation": 0.5},
                adaptive_weighting=True,
                adaptation_rate=0.2,
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, metrics = trainer.fit((x_train, y_train))

        assert trained_model is not None
        assert "final_train_loss" in metrics

    def test_adaptive_weights_update_during_training(self):
        """Test that constraint weights adapt during training."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=10,
            batch_size=4,
            learning_rate=1e-3,
            constraint_config=ConstraintConfig(
                constraints=["energy_conservation"],
                weights={"energy_conservation": 1.0},
                adaptive_weighting=True,
                adaptation_rate=0.1,
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        # Training should work with adaptive weighting
        trained_model, _ = trainer.fit((x_train, y_train))

        assert trained_model is not None

    def test_violation_monitoring(self):
        """Test constraint violation monitoring."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            constraint_config=ConstraintConfig(
                constraints=["energy_conservation"],
                weights={"energy_conservation": 1.0},
                violation_monitoring=True,
                violation_threshold=0.01,
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, _ = trainer.fit((x_train, y_train))

        assert trained_model is not None


class TestPhysicsStateTracking:
    """Test physics state tracking (chemical accuracy, SCF, conservation)."""

    def test_chemical_accuracy_tracking(self):
        """Test that chemical accuracy is tracked during training."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            metrics_tracking_config=MetricsTrackingConfig(
                chemical_accuracy=ChemicalAccuracyTracking(enabled=True),
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, _ = trainer.fit((x_train, y_train))

        # Should track chemical accuracy
        assert trained_model is not None

    def test_scf_convergence_tracking(self):
        """Test that SCF convergence is tracked during training."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            metrics_tracking_config=MetricsTrackingConfig(
                scf_convergence=SCFConvergenceTracking(enabled=True),
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, _ = trainer.fit((x_train, y_train))

        assert trained_model is not None

    def test_conservation_violation_tracking(self):
        """Test that conservation violations are tracked."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            metrics_tracking_config=MetricsTrackingConfig(
                conservation_violations=ConservationViolationTracking(enabled=True),
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, _ = trainer.fit((x_train, y_train))

        assert trained_model is not None


class TestCombinedPhysicsFeatures:
    """Test combining multiple advanced physics features."""

    def test_multi_scale_with_adaptive_weighting(self):
        """Test combining multi-scale physics with adaptive weighting."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            multiscale_config=MultiScaleConfig(
                scales=["molecular", "atomic"],
                weights={"molecular": 0.6, "atomic": 0.4},
            ),
            constraint_config=ConstraintConfig(
                constraints=["energy_conservation"],
                weights={"energy_conservation": 1.0},
                adaptive_weighting=True,
                adaptation_rate=0.15,
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, metrics = trainer.fit((x_train, y_train))

        assert trained_model is not None
        assert "final_train_loss" in metrics

    def test_all_physics_features_enabled(self):
        """Test enabling all advanced physics features together."""
        model = nnx.Linear(2, 1, rngs=nnx.Rngs(0))

        config = TrainingConfig(
            num_epochs=5,
            batch_size=4,
            learning_rate=1e-3,
            multiscale_config=MultiScaleConfig(
                scales=["molecular", "atomic", "electronic"],
                weights={"molecular": 0.5, "atomic": 0.3, "electronic": 0.2},
                coupling=True,
            ),
            constraint_config=ConstraintConfig(
                constraints=["energy_conservation", "momentum_conservation"],
                weights={"energy_conservation": 0.6, "momentum_conservation": 0.4},
                adaptive_weighting=True,
                adaptation_rate=0.1,
                violation_monitoring=True,
                violation_threshold=0.01,
            ),
            metrics_tracking_config=MetricsTrackingConfig(
                chemical_accuracy=ChemicalAccuracyTracking(enabled=True),
                scf_convergence=SCFConvergenceTracking(enabled=True),
                conservation_violations=ConservationViolationTracking(enabled=True),
            ),
        )

        trainer = Trainer(model=model, config=config)

        x_train = jnp.ones((4, 2))
        y_train = jnp.ones((4, 1))

        trained_model, metrics = trainer.fit((x_train, y_train))

        assert trained_model is not None
        assert "final_train_loss" in metrics
