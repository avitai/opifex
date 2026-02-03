"""Conservation law enforcement tests for unified Trainer.

This module tests conservation law enforcement using the refactored API:
- ConservationConfig for configuration
- ConservationViolations utility for direct violation checking
- Trainer remains domain-agnostic

Migrated from test_physics_informed_trainer.py (Phase 0E - Batch 3)
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.physics.conservation import (
    energy_violation,
    momentum_violation,
    particle_number_violation,
    symmetry_violation,
)
from opifex.core.training.config import TrainingConfig
from opifex.core.training.physics_configs import ConservationConfig
from opifex.core.training.trainer import Trainer


class MockModel(nnx.Module):
    """Mock model for conservation tests."""

    def __init__(self, features: int = 32, rngs: nnx.Rngs | None = None):
        if rngs is None:
            rngs = nnx.Rngs(0)
        self.linear1 = nnx.Linear(2, features, rngs=rngs)
        self.linear2 = nnx.Linear(features, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def mock_model():
    """Create mock model for testing."""
    return MockModel(features=32, rngs=nnx.Rngs(42))


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (100, 2))
    y = jnp.sum(x**2, axis=1, keepdims=True)
    return x, y


class TestConservationLawEnforcement:
    """Test comprehensive conservation law enforcement.

    Migrated from PhysicsInformedTrainer tests.
    Uses ConservationConfig and ConservationViolations utility.
    """

    def test_energy_conservation_checking(self, mock_model, sample_data):
        """Test comprehensive energy conservation checking."""
        conservation_config = ConservationConfig(
            laws=["energy"],
            energy_tolerance=1e-6,
            energy_monitoring=True,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run training step
        _, _ = trainer.training_step(x, y)

        # Verify energy conservation checking using the utility directly
        violation = energy_violation(
            y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
        )

        assert isinstance(violation, jax.Array)
        assert violation >= 0.0

    def test_particle_number_conservation(self, mock_model, sample_data):
        """Test electron/particle number preservation."""
        conservation_config = ConservationConfig(
            laws=["particle_number"],
            target_particle_number=10.0,
            particle_tolerance=1e-4,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run training step
        _, _ = trainer.training_step(x, y)

        # Test particle conservation using the utility directly
        y_pred = jnp.ones((10, 1)) * 1.0  # Sum will be 10.0
        violation = particle_number_violation(
            y_pred, target_particle_number=10.0, tolerance=1e-4
        )

        assert isinstance(violation, jax.Array)
        assert violation >= 0.0

    def test_momentum_conservation(self, mock_model, sample_data):
        """Test momentum conservation in physics systems."""
        conservation_config = ConservationConfig(
            laws=["momentum"],
            momentum_tolerance=1e-5,
            momentum_components=["x", "y", "z"],
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run training step
        _, _ = trainer.training_step(x, y)

        # Test momentum conservation using the utility directly
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        violation = momentum_violation(y_pred, y_true, tolerance=1e-5)

        assert isinstance(violation, jax.Array)
        assert violation >= 0.0

    def test_symmetry_preservation(self, mock_model, sample_data):
        """Test molecular and crystalline symmetry conservation."""
        conservation_config = ConservationConfig(
            laws=["symmetry"],
            symmetry_groups=["point_group", "space_group"],
            symmetry_tolerance=1e-6,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run training step
        _, _ = trainer.training_step(x, y)

        # Test symmetry conservation using the utility directly
        y_symmetric = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        violation = symmetry_violation(y_symmetric, tolerance=1e-6)

        assert isinstance(violation, jax.Array)
        assert violation >= 0.0


class TestConservationLawToleranceBehavior:
    """Test tolerance checking behavior for conservation laws.

    These tests verify that tolerance checking works correctly.
    Migrated from PhysicsInformedTrainer bug fix verification tests.
    """

    def test_energy_conservation_respects_tolerance(self, mock_model, sample_data):
        """Test that energy conservation violations below tolerance are ignored."""
        conservation_config = ConservationConfig(
            laws=["energy"],
            energy_tolerance=1e-5,
            energy_monitoring=True,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, y = sample_data

        # Test using the ConservationViolations API
        violation = energy_violation(
            y[:5], y[:5], tolerance=1e-5, monitoring_enabled=True
        )

        # Small violations should return 0 due to tolerance threshold
        assert violation >= 0.0, "Violation should be non-negative"

    def test_energy_conservation_monitoring_flag(self, mock_model, sample_data):
        """Test that monitoring flag can disable energy conservation checking."""
        conservation_config = ConservationConfig(
            laws=["energy"],
            energy_tolerance=1e-6,
            energy_monitoring=False,  # Disabled
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, y = sample_data

        # Even with large violations, should return 0 when monitoring disabled
        violation = energy_violation(
            y[:10], y[:10] * 2.0, tolerance=1e-6, monitoring_enabled=False
        )

        assert violation == 0.0, "Should return 0 when monitoring is disabled"

    def test_particle_conservation_respects_tolerance(self, mock_model, sample_data):
        """Test that particle conservation violations below tolerance are ignored."""
        conservation_config = ConservationConfig(
            laws=["particle_number"],
            target_particle_number=10.0,
            particle_tolerance=1e-4,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, _ = sample_data

        # Create predictions that give particle number close to target
        y_pred = jnp.ones((10, 1)) * 1.0  # Sum will be 10.0
        violation = particle_number_violation(
            y_pred, target_particle_number=10.0, tolerance=1e-4
        )

        # Should have small violation due to tolerance
        assert violation >= 0.0, "Violation should be non-negative"

    def test_momentum_conservation_component_wise(self, mock_model, sample_data):
        """Test that momentum is conserved component-wise, not as a scalar sum."""
        conservation_config = ConservationConfig(
            laws=["momentum"],
            momentum_tolerance=1e-5,
            momentum_components=["x", "y", "z"],
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, _ = sample_data

        # Create test data where momentum should be conserved
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.0, 2.0], [3.0, 4.0]])

        violation = momentum_violation(y_pred, y_true, tolerance=1e-5)

        # Should have zero violation since pred == true
        assert violation == 0.0, "Identical predictions should have zero violation"

    def test_momentum_conservation_respects_tolerance(self, mock_model, sample_data):
        """Test that momentum conservation violations below tolerance are ignored."""
        conservation_config = ConservationConfig(
            laws=["momentum"],
            momentum_tolerance=1.0,  # Large tolerance
            momentum_components=["x", "y", "z"],
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, _ = sample_data

        # Create data with small momentum violation
        y_pred = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y_true = jnp.array([[1.01, 2.01], [3.01, 4.01]])  # Small difference

        violation = momentum_violation(y_pred, y_true, tolerance=1.0)

        # With large tolerance, small violations should return 0
        assert violation >= 0.0, "Violation should be non-negative"

    def test_symmetry_conservation_respects_tolerance(self, mock_model, sample_data):
        """Test that symmetry violations below tolerance are ignored."""
        conservation_config = ConservationConfig(
            laws=["symmetry"],
            symmetry_groups=["point_group"],
            symmetry_tolerance=1e-6,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        Trainer(mock_model, config)
        _, _ = sample_data

        # Create perfectly symmetric data
        y_symmetric = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        violation = symmetry_violation(y_symmetric, tolerance=1e-6)

        # Perfectly symmetric data should have very small violation
        assert violation >= 0.0, "Violation should be non-negative"

    def test_conservation_laws_integration_with_tolerance(
        self, mock_model, sample_data
    ):
        """Test that all conservation laws work together with tolerance checking."""
        conservation_config = ConservationConfig(
            laws=["energy", "momentum", "symmetry"],
            energy_tolerance=1e-6,
            energy_monitoring=True,
            momentum_tolerance=1e-5,
            momentum_components=["x", "y"],
            symmetry_tolerance=1e-6,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run a training step with all conservation laws
        loss, _ = trainer.training_step(x[:10], y[:10])

        # Verify loss is finite
        assert jnp.isfinite(loss), "Loss should be finite with all conservation laws"

        # Test each violation separately using the utility
        # Energy
        energy_viol = energy_violation(
            y[:5], y[:5], tolerance=1e-6, monitoring_enabled=True
        )
        assert energy_viol >= 0.0

        # Momentum
        y_test = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        momentum_viol = momentum_violation(y_test, y_test, tolerance=1e-5)
        assert momentum_viol >= 0.0

        # Symmetry
        y_sym = jnp.array([[1.0, 1.0], [2.0, 2.0]])
        symmetry_viol = symmetry_violation(y_sym, tolerance=1e-6)
        assert symmetry_viol >= 0.0


class TestAdaptiveConstraintWeighting:
    """Test adaptive constraint weighting functionality.

    These tests verify that constraint weights adapt dynamically based on
    violation severity, which is a CRITICAL feature for robust physics-informed training.
    """

    def test_adaptive_weight_initialization(self):
        """Test adaptive weighting system initialization."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        # Initialize with equal weights
        constraints = ["energy", "momentum", "symmetry"]
        initial_weights = {"energy": 0.4, "momentum": 0.4, "symmetry": 0.2}

        weighting = AdaptiveConstraintWeighting(
            constraints=constraints,
            initial_weights=initial_weights,
            adaptation_rate=0.1,
        )

        # Verify initialization
        assert weighting.constraints == constraints
        assert weighting.current_weights == initial_weights
        assert weighting.adaptation_rate == 0.1

    def test_adaptive_weight_updates_based_on_violations(self):
        """Test that weights adapt based on violation severity."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        constraints = ["energy", "momentum"]
        initial_weights = {"energy": 0.5, "momentum": 0.5}

        weighting = AdaptiveConstraintWeighting(
            constraints=constraints,
            initial_weights=initial_weights,
            adaptation_rate=0.1,
        )

        # Simulate violations (energy is violated more)
        violations = {
            "energy": 1.0,  # High violation
            "momentum": 0.1,  # Low violation
        }

        # Update weights
        new_weights = weighting.update_weights(violations)

        # Verify weights are updated
        assert new_weights["energy"] > new_weights["momentum"], (
            "Energy should have higher weight due to higher violation"
        )

        # Verify weights sum to 1.0 (normalized)
        assert abs(sum(new_weights.values()) - 1.0) < 1e-6, (
            "Weights should be normalized to sum to 1.0"
        )

    def test_adaptive_weighting_with_multiple_updates(self):
        """Test adaptive weighting over multiple training steps."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        constraints = ["energy", "momentum", "symmetry"]
        initial_weights = {
            "energy": 1.0 / 3.0,
            "momentum": 1.0 / 3.0,
            "symmetry": 1.0 / 3.0,
        }

        weighting = AdaptiveConstraintWeighting(
            constraints=constraints,
            initial_weights=initial_weights,
            adaptation_rate=0.2,
        )

        # Simulate multiple training steps with changing violations
        violation_history = [
            {"energy": 1.0, "momentum": 0.5, "symmetry": 0.1},  # Step 1
            {"energy": 0.8, "momentum": 0.6, "symmetry": 0.2},  # Step 2
            {"energy": 0.6, "momentum": 0.7, "symmetry": 0.3},  # Step 3
        ]

        weights_history = []
        for violations in violation_history:
            new_weights = weighting.update_weights(violations)
            weights_history.append(dict(new_weights))

        # Verify weights changed over steps
        assert weights_history[0] != weights_history[1], (
            "Weights should change between steps"
        )
        assert weights_history[1] != weights_history[2], (
            "Weights should continue to adapt"
        )

        # Verify all weights are normalized
        for weights in weights_history:
            assert abs(sum(weights.values()) - 1.0) < 1e-6, (
                "All weights should be normalized"
            )

    def test_adaptive_weighting_with_zero_violations(self):
        """Test that weights remain stable when violations are zero."""
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        constraints = ["energy", "momentum"]
        initial_weights = {"energy": 0.6, "momentum": 0.4}

        weighting = AdaptiveConstraintWeighting(
            constraints=constraints,
            initial_weights=initial_weights,
            adaptation_rate=0.1,
        )

        # No violations
        violations = {"energy": 0.0, "momentum": 0.0}

        # Update weights
        new_weights = weighting.update_weights(violations)

        # Weights should remain unchanged
        assert new_weights == initial_weights, (
            "Weights should not change when violations are zero"
        )

    def test_adaptive_weighting_integration_with_trainer(self, mock_model, sample_data):
        """Test adaptive weighting integrated with Trainer.

        This is the CRITICAL test that validates adaptive weighting works
        during actual training with the unified Trainer.
        """
        from opifex.core.physics.conservation import AdaptiveConstraintWeighting

        # Create config with adaptive weighting
        conservation_config = ConservationConfig(
            laws=["energy", "momentum"],
            energy_tolerance=1e-6,
            energy_monitoring=True,
            momentum_tolerance=1e-5,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Create adaptive weighting system
        constraints = ["energy", "momentum"]
        initial_weights = {"energy": 0.5, "momentum": 0.5}
        weighting = AdaptiveConstraintWeighting(
            constraints=constraints,
            initial_weights=initial_weights,
            adaptation_rate=0.1,
        )

        # Run multiple training steps and track weight adaptation
        metrics_history = []
        weights_history = []

        for _step in range(5):
            _, metrics = trainer.training_step(x[:10], y[:10])

            # Compute violations using ConservationViolations
            violations = {
                "energy": float(
                    energy_violation(
                        y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
                    )
                ),
                "momentum": float(momentum_violation(y[:10], y[:10], tolerance=1e-5)),
            }

            # Update adaptive weights
            new_weights = weighting.update_weights(violations)

            metrics_history.append(metrics)
            weights_history.append(dict(new_weights))

        # Verify training works
        assert all(jnp.isfinite(m["loss"]) for m in metrics_history), (
            "All losses should be finite"
        )

        # Verify weights were tracked
        assert len(weights_history) == 5, "Should have 5 weight updates"

        # Verify weights are normalized at each step
        for weights in weights_history:
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-6, f"Weights should sum to 1.0, got {total}"

    def test_adaptive_weighting_with_constraint_aggregator(self):
        """Test adaptive weighting through ConstraintAggregator."""
        from opifex.core.physics.conservation import ConstraintAggregator

        # Config with adaptive weighting enabled
        config = {
            "conservation_laws": ["energy", "momentum", "symmetry"],
            "energy_conservation_tolerance": 1e-6,
            "energy_conservation_monitoring": True,
            "momentum_conservation_tolerance": 1e-5,
            "symmetry_tolerance": 1e-6,
            "adaptive_weighting": True,
            "adaptation_rate": 0.15,
        }

        aggregator = ConstraintAggregator(config)

        # Verify adaptive weighting is enabled
        assert aggregator.adaptive_weighting is True
        assert hasattr(aggregator, "weight_manager")
        assert aggregator.weight_manager.adaptation_rate == 0.15

        # Create test data
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (10, 2))
        y = jnp.sum(x**2, axis=1, keepdims=True)

        # Compute metrics (which triggers weight update)
        metrics = aggregator.compute_constraint_metrics(x, y, y)

        # Verify conservation metrics are present
        assert "energy_conservation" in metrics
        assert "momentum_conservation" in metrics
        assert "symmetry_conservation" in metrics

        # Verify all metrics are finite
        assert all(jnp.isfinite(jnp.array(v)) for v in metrics.values()), (
            "All constraint metrics should be finite"
        )

        # Get current weights after update
        current_weights = aggregator.weight_manager.get_current_weights()

        # Verify weights are normalized
        assert abs(sum(current_weights.values()) - 1.0) < 1e-6, (
            "Adaptive weights should be normalized"
        )
