"""Tests for BaseCallback coverage."""

from opifex.core.training.components.base import BaseCallback


class TestBaseCallback:
    def test_default_methods(self):
        """Verify that default methods run without error (no-ops)."""
        callback = BaseCallback()
        # Should do nothing
        callback.on_epoch_begin(0, None)
        callback.on_epoch_end(0, None, {})
        callback.on_batch_begin(0, None)
        callback.on_batch_end(0, None, 0.0, {})
