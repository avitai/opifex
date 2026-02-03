"""Tests for BaseComponent coverage."""

from opifex.core.training.components.base import BaseComponent


class TestBaseComponent:
    def test_default_methods(self):
        """Verify that default methods run without error (no-ops)."""
        component = BaseComponent()
        # Should do nothing
        component.on_epoch_begin(0, None)
        component.on_epoch_end(0, None, {})
        component.on_batch_begin(0, None)
        component.on_batch_end(0, None, 0.0, {})
