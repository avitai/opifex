"""Tests for the structured logging context managers.

Regression coverage for the frozen ``LogContext`` mutation bug: the public
``training_context`` / ``inference_context`` context managers must not crash
when they update the immutable logging context.
"""

import logging

from opifex.deployment.monitoring.logging import LogContext, StructuredLogger


def _make_logger() -> StructuredLogger:
    """Build a StructuredLogger with a hermetic handler (no log files)."""
    return StructuredLogger(name="test-logging", handlers=[logging.NullHandler()])


def test_training_context_updates_frozen_context_without_crashing() -> None:
    """training_context must mutate the frozen LogContext and restore it."""
    logger = _make_logger()
    assert logger.context.model_name is None

    with logger.training_context(
        model_name="resnet", epoch=3, batch_size=16, experiment_id="exp-42"
    ):
        logger.info("inside training context")
        # Inside the block the immutable context has been rebuilt with new values.
        assert logger.context.model_name == "resnet"
        assert logger.context.experiment_id == "exp-42"

    # On exit the original context fields are restored.
    assert logger.context.model_name is None
    assert logger.context.experiment_id is None


def test_inference_context_updates_frozen_context_without_crashing() -> None:
    """inference_context must mutate the frozen LogContext and restore it."""
    logger = _make_logger()
    assert logger.context.model_name is None

    with logger.inference_context(model_name="bert", batch_size=8):
        logger.info("inside inference context")
        assert logger.context.model_name == "bert"

    assert logger.context.model_name is None


def test_log_context_stays_frozen() -> None:
    """The fix must keep LogContext immutable (frozen dataclass invariant)."""
    context = LogContext()
    import pytest

    with pytest.raises((AttributeError, TypeError)):
        context.model_name = "mutated"  # type: ignore[misc]
