"""Regression tests for latent crashes in BasicTrainer.

Covers two defects on otherwise-untested fallback paths:

- Task 12.0.9: ``_physics_informed_training_step`` raised ``KeyError`` on the
  no-physics fallback branch because it unconditionally read the
  ``"physics_loss"`` / ``"boundary_loss"`` keys, which the fallback
  ``loss_components`` dict never populated.
- Task 12.0.10: ``create_progress_bar_callback`` returned a two-positional-arg
  callback ``(epoch, metrics)`` while ``train`` invokes the callback with a
  single ``progress_info`` dict, raising ``TypeError`` when the shipped factory
  was wired into ``TrainingConfig.progress_callback``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.neural.base import StandardMLP
from opifex.training.basic_trainer import (
    BasicTrainer,
    create_progress_bar_callback,
)


def _make_trainer(num_epochs: int = 1) -> BasicTrainer:
    """Build a minimal trainer over a tiny MLP for fast, focused tests."""
    model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
    config = TrainingConfig(num_epochs=num_epochs, verbose=False)
    return BasicTrainer(model, config)


class TestPhysicsInformedStepNoPhysicsFallback:
    """Task 12.0.9 — fallback path when ``physics_loss`` is ``None``."""

    def test_step_completes_without_keyerror_when_physics_loss_none(self) -> None:
        """Driving the physics step with no physics loss must not KeyError."""
        trainer = _make_trainer()
        assert trainer.physics_loss is None  # exercise the fallback branch

        x_train = jnp.ones((8, 4))
        y_train = jnp.ones((8, 1))
        boundary_data = (jnp.ones((4, 4)), jnp.ones((4, 1)))

        loss_value = trainer._physics_informed_training_step(x_train, y_train, boundary_data)

        # Step completed end-to-end: scalar loss returned and counter advanced.
        assert jnp.ndim(loss_value) == 0
        assert trainer.state.step == 1
        # No physics/boundary components are recorded on the fallback path.
        assert trainer.metrics.physics_losses == []
        assert trainer.metrics.boundary_losses == []


class TestProgressCallbackContract:
    """Task 12.0.10 — single-dict progress-callback contract."""

    def test_factory_callback_accepts_single_progress_dict(self) -> None:
        """The shipped factory callback must accept one ``progress_info`` dict."""
        callback = create_progress_bar_callback("Test", total_epochs=1)

        progress_info: dict[str, Any] = {
            "epoch": 0,
            "total_epochs": 1,
            "train_loss": 0.5,
            "val_loss": None,
            "step": 1,
            "best_loss": 0.5,
            "best_val_loss": float("inf"),
        }

        # Single positional dict argument — must not raise TypeError.
        callback(progress_info)

    def test_train_invokes_factory_callback_without_typeerror(self) -> None:
        """``train`` must drive the shipped factory callback with a dict."""
        received: list[dict[str, Any]] = []
        factory_callback = create_progress_bar_callback("Test", total_epochs=1)

        def recording_callback(progress_info: dict[str, Any]) -> None:
            received.append(progress_info)
            factory_callback(progress_info)

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        config = TrainingConfig(
            num_epochs=1,
            verbose=False,
            progress_callback=recording_callback,
        )
        trainer = BasicTrainer(model, config)

        x_train = jnp.ones((8, 4))
        y_train = jnp.ones((8, 1))
        trainer.train((x_train, y_train))

        assert len(received) == 1
        assert isinstance(received[0], dict)
        assert received[0]["epoch"] == 0
        assert received[0]["total_epochs"] == 1


if __name__ == "__main__":  # pragma: no cover - manual invocation aid
    pytest.main([__file__, "-vv", "--no-cov"])
