import pytest


def test_monitoring_migration():
    """Verify the monitoring metrics are available in core."""
    # This will fail until migration is complete
    try:
        from opifex.core.training.monitoring import metrics
    except ImportError:
        pytest.fail("Failed to import monitoring modules from core")

    # Verify integrity of migrated classes; FLOP counting is calibrax's
    # (opifex.benchmarking.profiling re-exports it), not a monitoring module.
    assert hasattr(metrics, "TrainingMetrics")
    assert hasattr(metrics, "AdvancedMetricsCollector")


def test_components_migration():
    """Verify adaptive sampling and recovery are available in core."""
    try:
        from opifex.core.training.components import (
            adaptive_sampling,
            checkpoint_store,
            recovery,
        )
    except ImportError:
        pytest.fail("Failed to import components from core")

    assert hasattr(adaptive_sampling, "RADSampler")
    assert hasattr(recovery, "ErrorRecoveryManager")
    assert hasattr(checkpoint_store, "CheckpointStore")
    assert hasattr(checkpoint_store, "OrbaxCheckpointStore")


def test_strategies_migration():
    """Verify strategies are available in core."""
    try:
        from opifex.core.training.strategies import (
            incremental_trainer,
            mixed_precision,
        )
    except ImportError:
        pytest.fail("Failed to import strategies from core")

    assert hasattr(mixed_precision, "MixedPrecisionTrainer")
    assert hasattr(incremental_trainer, "IncrementalTrainer")


def test_legacy_shims_are_gone():
    """The 15 ``opifex.training.*`` shim modules were deleted per Rule 1.

    Importing them must now ``ModuleNotFoundError`` — anything still trying
    to reach those legacy paths needs to migrate to the canonical
    ``opifex.core.training.*`` locations. This guards against the shims
    silently reappearing through stray autoformatter / merge artefacts.
    """
    legacy_shims = (
        "opifex.training.metrics",
        "opifex.training.mixed_precision",
        "opifex.training.recovery",
        "opifex.training.adaptive_sampling",
        "opifex.training.checkpoint_manager",
        "opifex.training.orbax_checkpoint_manager",
        "opifex.training.incremental_trainer",
        "opifex.training.quantum_training",
        "opifex.training.flops_counter",
        "opifex.training.utils",
        "opifex.training.multilevel",
    )
    import importlib

    for shim in legacy_shims:
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(shim)
