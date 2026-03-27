"""Tests for error recovery and training stability management."""

from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import pytest

from opifex.core.training.components.recovery import ErrorRecoveryManager


@dataclass
class MockTrainingState:
    """Minimal training state for testing recovery."""

    step: int = 0
    loss: float = 0.0
    recovery_state: dict[str, Any] = field(default_factory=dict)


class TestErrorRecoveryManager:
    """Tests for ErrorRecoveryManager."""

    def test_default_config(self):
        """Default config has sensible thresholds."""
        mgr = ErrorRecoveryManager()
        assert mgr.max_retries == 3
        assert mgr.gradient_clip_threshold == 10.0
        assert mgr.loss_explosion_threshold == 1e6

    def test_custom_config(self):
        """Custom config overrides defaults."""
        mgr = ErrorRecoveryManager({"max_retries": 5, "loss_explosion_threshold": 100.0})
        assert mgr.max_retries == 5
        assert mgr.loss_explosion_threshold == 100.0


class TestStabilityChecks:
    """Tests for training stability detection."""

    def test_stable_training(self):
        """Normal loss and gradients are reported stable."""
        mgr = ErrorRecoveryManager()
        grads = {"w": jnp.ones((4, 4)) * 0.1}
        state = MockTrainingState(step=10, loss=0.5)

        is_stable, issue = mgr.check_training_stability(0.5, grads, state)
        assert is_stable is True
        assert issue is None

    def test_detects_loss_explosion(self):
        """Loss above threshold triggers instability."""
        mgr = ErrorRecoveryManager({"loss_explosion_threshold": 100.0})
        grads = {"w": jnp.ones((4,))}
        state = MockTrainingState()

        is_stable, issue = mgr.check_training_stability(200.0, grads, state)
        assert is_stable is False
        assert issue == "loss_explosion"

    def test_detects_nan_loss(self):
        """NaN loss triggers instability."""
        mgr = ErrorRecoveryManager()
        grads = {"w": jnp.ones((4,))}
        state = MockTrainingState()

        is_stable, issue = mgr.check_training_stability(float("nan"), grads, state)
        assert is_stable is False
        assert issue == "nan_loss"

    def test_detects_gradient_explosion(self):
        """Large gradient norm triggers instability."""
        mgr = ErrorRecoveryManager({"gradient_clip_threshold": 1.0})
        grads = {"w": jnp.ones((4,)) * 100.0}
        state = MockTrainingState()

        is_stable, issue = mgr.check_training_stability(0.5, grads, state)
        assert is_stable is False
        assert issue == "gradient_explosion"

    def test_detects_nan_gradients(self):
        """NaN in gradients triggers instability."""
        mgr = ErrorRecoveryManager()
        grads = {"w": jnp.array([1.0, float("nan"), 3.0])}
        state = MockTrainingState()

        is_stable, issue = mgr.check_training_stability(0.5, grads, state)
        assert is_stable is False
        assert issue == "nan_gradients"


class TestGradientClipping:
    """Tests for gradient clipping."""

    def test_clips_large_gradients(self):
        """Gradients exceeding threshold are scaled down."""
        mgr = ErrorRecoveryManager({"gradient_clip_threshold": 1.0})
        grads = {"w": jnp.ones((4,)) * 10.0}

        clipped = mgr.apply_gradient_clipping(grads)
        grad_norm = jnp.sqrt(jnp.sum(clipped["w"] ** 2))
        assert float(grad_norm) == pytest.approx(1.0, abs=0.01)

    def test_preserves_small_gradients(self):
        """Gradients below threshold are unchanged."""
        mgr = ErrorRecoveryManager({"gradient_clip_threshold": 100.0})
        grads = {"w": jnp.ones((4,)) * 0.1}

        clipped = mgr.apply_gradient_clipping(grads)
        assert jnp.allclose(clipped["w"], grads["w"])


class TestRecovery:
    """Tests for recovery from instability."""

    def test_restores_last_stable_state(self):
        """Recovery returns last stable state."""
        mgr = ErrorRecoveryManager()
        stable = MockTrainingState(step=50, loss=0.1)
        mgr.last_stable_state = stable

        unstable = MockTrainingState(step=55, loss=1e8)
        recovered = mgr.recover_from_instability("loss_explosion", unstable)
        assert recovered.step == 50

    def test_exceeds_max_retries_raises(self):
        """Exceeding max retries raises RuntimeError."""
        mgr = ErrorRecoveryManager({"max_retries": 2})
        mgr.recovery_attempts = 2
        state = MockTrainingState()

        with pytest.raises(RuntimeError, match="Maximum recovery attempts"):
            mgr.recover_from_instability("nan_loss", state)

    def test_update_stable_resets_attempts(self):
        """Updating stable state resets recovery counter."""
        mgr = ErrorRecoveryManager()
        mgr.recovery_attempts = 2
        mgr.update_stable_state(MockTrainingState(step=100))
        assert mgr.recovery_attempts == 0
        assert mgr.last_stable_state.step == 100
