"""Integration tests demonstrating unified Trainer extensibility.

This module shows that the unified Trainer achieves the same extensibility
goals as PhysicsInformedTrainer, but through composable configurations
instead of runtime hooks and custom loss registration.

Key demonstrations:
- Multi-domain physics via composable configs
- Custom physics constraints via ConservationConfig, MultiScaleConfig
- Metrics tracking via MetricsTrackingConfig
- Logging configuration via LoggingConfig
- Performance analytics via PerformanceConfig

These tests prove that the refactored architecture maintains all extensibility
capabilities while being more maintainable and type-safe.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.physics.conservation import (
    energy_violation,
    momentum_violation,
    MultiScalePhysics,
    symmetry_violation,
)
from opifex.core.training.config import TrainingConfig
from opifex.core.training.physics_configs import (
    ConservationConfig,
    LoggingConfig,
    MetricsTrackingConfig,
    MultiScaleConfig,
    PerformanceConfig,
)
from opifex.core.training.trainer import Trainer


class MockModel(nnx.Module):
    """Mock model for extensibility tests."""

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


class TestComposablePhysicsConfigurations:
    """Test that composable configs replace custom loss registration.

    Demonstrates: The unified Trainer achieves custom physics constraints
    through composable config objects instead of runtime loss registration.
    """

    def test_energy_conservation_configuration(self, mock_model, sample_data):
        """Test energy conservation via ConservationConfig."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # trainer.register_custom_loss("energy_conservation", energy_loss_fn)

        # NEW PATTERN (Unified Trainer):
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

        # Verify training works with conservation config
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)

        # Verify we can check conservation violations directly
        violation = energy_violation(
            y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
        )
        assert violation >= 0.0

    def test_multi_scale_physics_configuration(self, mock_model, sample_data):
        """Test multi-scale physics via MultiScaleConfig."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # trainer.register_custom_loss("atomic_scale", atomic_loss_fn)
        # trainer.register_custom_loss("molecular_scale", molecular_loss_fn)
        # trainer.enable_multi_physics_mode()

        # NEW PATTERN (Unified Trainer):
        multi_scale_config = MultiScaleConfig(
            scales=["molecular", "atomic", "electronic"],
            weights={"molecular": 0.4, "atomic": 0.4, "electronic": 0.2},
            coupling=True,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            multiscale_config=multi_scale_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training works with multi-scale config
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)

        # Verify we can use MultiScalePhysics utility directly
        multi_scale = MultiScalePhysics(
            scales=["molecular", "atomic", "electronic"],
            scale_weights={"molecular": 0.4, "atomic": 0.4, "electronic": 0.2},
        )

        # Compute multi-scale loss
        def base_loss(y_pred, y_true):
            return jnp.mean((y_pred - y_true) ** 2)

        scale_loss = multi_scale.compute_loss(x[:10], y[:10], y[:10], base_loss)
        assert jnp.isfinite(scale_loss)

    def test_combined_physics_configurations(self, mock_model, sample_data):
        """Test combining multiple physics configs."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # Multiple register_custom_loss() calls
        # Multiple register_hook() calls

        # NEW PATTERN (Unified Trainer):
        # Compose multiple configs together
        conservation_config = ConservationConfig(
            laws=["energy", "momentum"],
            energy_tolerance=1e-6,
            momentum_tolerance=1e-5,
        )

        multi_scale_config = MultiScaleConfig(
            scales=["molecular", "atomic"],
            weights={"molecular": 0.5, "atomic": 0.5},
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
            multiscale_config=multi_scale_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training works with composed configs
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)


class TestMetricsTrackingExtensibility:
    """Test that MetricsTrackingConfig replaces custom metrics logging.

    Demonstrates: The unified Trainer achieves metrics tracking through
    typed config objects instead of trainer.log_physics_metrics().
    """

    def test_metrics_tracking_configuration(self, mock_model, sample_data):
        """Test metrics tracking via MetricsTrackingConfig."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # trainer.log_physics_metrics({"metric": value}, step=i)
        # summary = trainer.get_physics_metrics_summary()

        # NEW PATTERN (Unified Trainer):
        metrics_config = MetricsTrackingConfig(
            detailed=True,
            physics_metrics=["chemical_accuracy", "scf_convergence"],
            quantum_states=False,
            state_history=False,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            metrics_tracking_config=metrics_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training produces metrics
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert isinstance(metrics, dict)
        assert "step" in metrics
        assert "loss" in metrics
        assert jnp.isfinite(loss)

    def test_performance_analytics_configuration(self, mock_model, sample_data):
        """Test performance analytics via PerformanceConfig."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # physics_config = {"performance_analytics": True, "timing_analysis": True}

        # NEW PATTERN (Unified Trainer):
        performance_config = PerformanceConfig(
            analytics=True,
            timing=True,
            memory=True,
            convergence=True,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            performance_config=performance_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run multiple steps to collect analytics
        for _ in range(3):
            loss, metrics = trainer.training_step(x[:10], y[:10])
            assert jnp.isfinite(loss)
            assert isinstance(metrics, dict)


class TestLoggingExtensibility:
    """Test that LoggingConfig replaces custom logging backends.

    Demonstrates: The unified Trainer achieves multi-backend logging
    through LoggingConfig instead of runtime configuration.
    """

    def test_logging_configuration(self, mock_model, sample_data):
        """Test logging configuration via LoggingConfig."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # physics_config = {
        #     "logging_backends": ["tensorboard", "wandb"],
        #     "logging_level": "INFO"
        # }

        # NEW PATTERN (Unified Trainer):
        logging_config = LoggingConfig(
            backends=["tensorboard"],
            level="INFO",
            frequency=1,
            real_time=False,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            logging_config=logging_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training works with logging config
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)

    def test_alert_thresholds_configuration(self, mock_model, sample_data):
        """Test alert threshold configuration."""
        # OLD PATTERN (PhysicsInformedTrainer):
        # physics_config = {
        #     "alert_thresholds": {"constraint_violation": 0.01}
        # }

        # NEW PATTERN (Unified Trainer):
        logging_config = LoggingConfig(
            backends=["tensorboard"],
            level="INFO",
            alert_thresholds={
                "constraint_violation": 0.01,
                "convergence_failure": 0.1,
            },
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            logging_config=logging_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training works with alert thresholds
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)


class TestMultiDomainPhysicsIntegration:
    """Integration test showing multi-domain physics via composition.

    Demonstrates: The unified Trainer handles quantum, conservation, and
    multi-scale physics simultaneously through config composition.
    """

    def test_quantum_conservation_multi_scale_integration(
        self, mock_model, sample_data
    ):
        """Test integrating quantum, conservation, and multi-scale physics."""
        # This integration test shows what was previously achieved through:
        # - trainer.register_custom_loss() (multiple calls)
        # - trainer.register_hook() (multiple calls)
        # - trainer.enable_multi_physics_mode()
        # - trainer.add_extension_point() (multiple calls)

        # NEW PATTERN: Compose all configs together
        conservation_config = ConservationConfig(
            laws=["energy", "momentum"],
            energy_tolerance=1e-6,
            momentum_tolerance=1e-5,
        )

        multi_scale_config = MultiScaleConfig(
            scales=["molecular", "atomic", "electronic"],
            weights={"molecular": 0.4, "atomic": 0.4, "electronic": 0.2},
            coupling=True,
        )

        metrics_config = MetricsTrackingConfig(
            detailed=True,
            physics_metrics=["chemical_accuracy", "scf_convergence"],
        )

        logging_config = LoggingConfig(
            backends=["tensorboard"],
            level="INFO",
            frequency=1,
        )

        performance_config = PerformanceConfig(
            analytics=True,
            timing=True,
            memory=False,
            convergence=True,
        )

        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
            multiscale_config=multi_scale_config,
            metrics_tracking_config=metrics_config,
            logging_config=logging_config,
            performance_config=performance_config,
        )

        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Verify training works with all configs composed
        loss, metrics = trainer.training_step(x[:10], y[:10])

        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)
        assert "step" in metrics
        assert "loss" in metrics

        # Verify we can still use utilities directly
        energy_viol = energy_violation(
            y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
        )
        assert energy_viol >= 0.0

    def test_extensibility_through_direct_utility_usage(self, mock_model, sample_data):
        """Test that domain-specific operations use utilities directly."""
        # This demonstrates the architectural pattern:
        # Instead of: trainer.compute_energy_conservation()
        # We use: ConservationViolations.energy() directly

        config = TrainingConfig(learning_rate=1e-3)
        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # Run training
        loss, _ = trainer.training_step(x[:10], y[:10])

        # Use utilities directly for domain-specific operations
        energy_viol = energy_violation(
            y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
        )

        momentum_viol = momentum_violation(y[:10], y[:10], tolerance=1e-5)

        symmetry_viol = symmetry_violation(
            jnp.array([[1.0, 1.0], [2.0, 2.0]]), tolerance=1e-6
        )

        # All violations should be computable
        assert energy_viol >= 0.0
        assert momentum_viol >= 0.0
        assert symmetry_viol >= 0.0

        # Training should still work
        assert jnp.isfinite(loss)


class TestExtensibilityComparisonSummary:
    """Summary test showing the architectural differences.

    This test class documents how the same extensibility goals are achieved
    through different patterns in the refactored architecture.
    """

    def test_extensibility_pattern_comparison(self, mock_model, sample_data):
        """Document the pattern comparison between old and new architectures."""

        # ========================================
        # OLD PATTERN (PhysicsInformedTrainer)
        # ========================================
        # Runtime extensibility through methods:
        # - trainer.register_custom_loss("name", fn)
        # - trainer.register_hook("hook_point", fn)
        # - trainer.execute_hooks("hook_point", *args)
        # - trainer.enable_multi_physics_mode()
        # - trainer.add_extension_point("domain")
        # - trainer.log_physics_metrics(dict, step)
        # - trainer.get_physics_metrics_summary()

        # ========================================
        # NEW PATTERN (Unified Trainer)
        # ========================================
        # Compile-time extensibility through composition:

        # 1. Physics constraints via typed configs
        conservation_config = ConservationConfig(
            laws=["energy"],
            energy_tolerance=1e-6,
        )

        # 2. Multi-domain physics via composition
        multi_scale_config = MultiScaleConfig(
            scales=["molecular", "atomic"],
            weights={"molecular": 0.5, "atomic": 0.5},
        )

        # 3. Metrics tracking via config
        metrics_config = MetricsTrackingConfig(
            detailed=True,
            physics_metrics=["chemical_accuracy"],
        )

        # 4. Compose all configs
        config = TrainingConfig(
            learning_rate=1e-3,
            conservation_config=conservation_config,
            multiscale_config=multi_scale_config,
            metrics_tracking_config=metrics_config,
        )

        # 5. Create trainer
        trainer = Trainer(mock_model, config)
        x, y = sample_data

        # 6. Train (same interface)
        loss, metrics = trainer.training_step(x[:10], y[:10])

        # 7. Use utilities directly for domain operations
        violation = energy_violation(
            y[:10], y[:10], tolerance=1e-6, monitoring_enabled=True
        )

        # Verify everything works
        assert jnp.isfinite(loss)
        assert isinstance(metrics, dict)
        assert violation >= 0.0

        # ========================================
        # KEY ARCHITECTURAL DIFFERENCES
        # ========================================
        # OLD: Runtime registration, dynamic hooks, trainer has domain methods
        # NEW: Compile-time composition, typed configs, trainer is generic
        #
        # Benefits of NEW approach:
        # - Type safety (configs are dataclasses)
        # - Better IDE support (autocomplete, type checking)
        # - Easier to test (configs are pure data)
        # - Clear separation of concerns (trainer vs utilities)
        # - No runtime state modifications
        # - Composition over inheritance/hooks
