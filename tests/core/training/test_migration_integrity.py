import pytest


def test_monitoring_migration():
    """Verify metrics and flops are available in core and re-exported legacy."""
    # This will fail until migration is complete
    try:
        from opifex.core.training.monitoring import flops, metrics
    except ImportError:
        pytest.fail("Failed to import monitoring modules from core")

    # Verify integrity of migrated classes
    assert hasattr(metrics, "TrainingMetrics")
    assert hasattr(metrics, "AdvancedMetricsCollector")
    assert hasattr(flops, "FlopsCounter")


def test_components_migration():
    """Verify adaptive sampling and recovery are available in core."""
    try:
        from opifex.core.training.components import (
            adaptive_sampling,
            checkpoint_manager,
            orbax_manager,
            recovery,
        )
    except ImportError:
        pytest.fail("Failed to import components from core")

    assert hasattr(adaptive_sampling, "RADSampler")
    assert hasattr(recovery, "ErrorRecoveryManager")
    assert hasattr(checkpoint_manager, "CheckpointManager")
    assert hasattr(orbax_manager, "OrbaxCheckpointManager")


def test_strategies_migration():
    """Verify strategies are available in core."""
    try:
        from opifex.core.training.strategies import (
            incremental_trainer,
            mixed_precision,
            quantum,
        )
    except ImportError:
        pytest.fail("Failed to import strategies from core")

    assert hasattr(mixed_precision, "MixedPrecisionTrainer")
    assert hasattr(incremental_trainer, "IncrementalTrainer")
    assert hasattr(quantum, "QuantumTrainingManager")


def test_legacy_shim_imports():
    """Verify that old imports still work via shim re-exports."""
    # These imports should NOT fail, but should now point to objects defined in core
    try:
        from opifex.training import metrics, mixed_precision, recovery
    except ImportError:
        pytest.fail("Legacy shim imports failed")

    # Verify they are actually the SAME objects (identity check)
    from opifex.core.training.components import recovery as core_recovery
    from opifex.core.training.monitoring import metrics as core_metrics
    from opifex.core.training.strategies import mixed_precision as core_mp

    assert metrics.TrainingMetrics is core_metrics.TrainingMetrics
    assert mixed_precision.MixedPrecisionTrainer is core_mp.MixedPrecisionTrainer
    assert recovery.ErrorRecoveryManager is core_recovery.ErrorRecoveryManager
