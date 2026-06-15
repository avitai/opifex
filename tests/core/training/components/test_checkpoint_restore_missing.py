"""Tests for :meth:`CheckpointComponent.restore_checkpoint` error handling.

These tests pin the fix for the silent-wrong-result finding: restoring a
step that was never saved must raise (matching the documented
``Raises: ValueError``) instead of returning a fabricated empty
checkpoint with ``model_state == {}``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import pytest
from flax import nnx

from opifex.core.training.components import CheckpointComponent


class _SimpleModel(nnx.Module):
    """Minimal NNX model providing real state to checkpoint."""

    def __init__(self, *, rngs: nnx.Rngs) -> None:
        self.linear = nnx.Linear(10, 5, rngs=rngs)


def _make_training_state(step: int) -> Any:
    """Build a lightweight training-state stub at ``step``."""
    state = Mock()
    state.step = step
    state.loss = 1.0
    return state


@pytest.fixture
def component(tmp_path) -> CheckpointComponent:
    """Checkpoint component that saves on every step."""
    return CheckpointComponent(config={"checkpoint_dir": str(tmp_path), "save_frequency": 1})


def test_restore_missing_checkpoint_raises(component) -> None:
    """Restoring a never-saved step raises ValueError, not an empty mock."""
    model = _SimpleModel(rngs=nnx.Rngs(0))
    component.setup(model, _make_training_state(0))

    # Store real checkpoints at steps 1 and 2.
    for step in (1, 2):
        component.step(model, _make_training_state(step))

    with pytest.raises(ValueError, match="99"):
        component.restore_checkpoint(step=99)


def test_restore_existing_checkpoint_returns_real_state(component) -> None:
    """Restoring a saved step returns the genuine stored checkpoint."""
    model = _SimpleModel(rngs=nnx.Rngs(0))
    component.setup(model, _make_training_state(0))
    component.step(model, _make_training_state(7))

    restored = component.restore_checkpoint(step=7)

    assert restored["step"] == 7
    # Real state, not the fabricated empty-dict mock.
    assert isinstance(restored["model_state"], nnx.State)
    assert restored["model_state"] != {}
