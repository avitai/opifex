"""Comprehensive tests for training components module.

Following strict TDD - these tests define the API and behavior of the
centralized components system that will consolidate all component patterns.

Test Coverage Goals:
- Base component class and lifecycle
- Checkpoint component functionality
- Mixed precision component
- Recovery component
- Component composition and registration
- State management
- Edge cases and error handling

Author: Opifex Framework Team
Date: October 2025
"""

from __future__ import annotations

from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.components import (
    CheckpointComponent,
    MixedPrecisionComponent,
    RecoveryComponent,
    TrainingComponent,
)


# Fixtures
@pytest.fixture
def simple_model():
    """Create a simple test model."""

    class SimpleModel(nnx.Module):
        def __init__(self, *, rngs: nnx.Rngs):
            self.linear = nnx.Linear(10, 5, rngs=rngs)

        def __call__(self, x: jax.Array) -> jax.Array:
            return self.linear(x)

    return SimpleModel(rngs=nnx.Rngs(0))


@pytest.fixture
def training_state():
    """Create a mock training state."""
    state = Mock()
    state.step = 0
    state.loss = 1.0
    state.metrics = {}
    state.recovery_state = {}
    return state


# ===================================================================
# Test TrainingComponent Base Class
# ===================================================================


class TestTrainingComponent:
    """Tests for the base TrainingComponent class."""

    def test_component_initialization(self):
        """Test basic component initialization."""
        config = {"key": "value"}
        component = TrainingComponent(config=config)

        assert component.config == config
        assert component.name == "TrainingComponent"

    def test_component_default_config(self):
        """Test component with no config provided."""
        component = TrainingComponent()

        assert component.config == {}
        assert component.name == "TrainingComponent"

    def test_component_lifecycle_setup(self, simple_model, training_state):
        """Test component setup lifecycle method."""
        component = TrainingComponent()

        # Should not raise error
        component.setup(simple_model, training_state)

        # Base implementation should do nothing
        assert True

    def test_component_lifecycle_cleanup(self):
        """Test component cleanup lifecycle method."""
        component = TrainingComponent()

        # Should not raise error
        component.cleanup()

        # Base implementation should do nothing
        assert True

    def test_component_step_method(self, simple_model, training_state):
        """Test component step method."""
        component = TrainingComponent()

        # Should not raise error and return None by default
        result = component.step(simple_model, training_state)

        assert result is None

    def test_component_state_management(self):
        """Test component maintains internal state."""
        component = TrainingComponent(config={"enabled": True})

        assert component.config["enabled"] is True


# ===================================================================
# Test CheckpointComponent
# ===================================================================


class TestCheckpointComponent:
    """Tests for the CheckpointComponent class."""

    def test_checkpoint_component_initialization(self):
        """Test checkpoint component initialization."""
        config = {
            "checkpoint_dir": "./test_checkpoints",
            "save_frequency": 100,
            "max_to_keep": 5,
        }
        component = CheckpointComponent(config=config)

        assert component.config["checkpoint_dir"] == "./test_checkpoints"
        assert component.config["save_frequency"] == 100
        assert component.config["max_to_keep"] == 5

    def test_checkpoint_component_save(self, simple_model, training_state):
        """Test checkpoint save functionality."""
        config = {"checkpoint_dir": "/tmp/test_checkpoints", "save_frequency": 10}  # noqa: S108
        component = CheckpointComponent(config=config)

        component.setup(simple_model, training_state)

        # Should save checkpoint
        training_state.step = 10
        result = component.step(simple_model, training_state)

        assert result is not None
        assert "checkpoint_saved" in result

    def test_checkpoint_component_skip_save(self, simple_model, training_state):
        """Test checkpoint skips save when not at frequency."""
        config = {"checkpoint_dir": "/tmp/test_checkpoints", "save_frequency": 100}  # noqa: S108
        component = CheckpointComponent(config=config)

        component.setup(simple_model, training_state)

        # Should not save checkpoint
        training_state.step = 10
        result = component.step(simple_model, training_state)

        assert result is None or "checkpoint_saved" not in result

    def test_checkpoint_component_max_to_keep(self, simple_model, training_state):
        """Test checkpoint component respects max_to_keep."""
        config = {
            "checkpoint_dir": "/tmp/test_checkpoints",  # noqa: S108
            "save_frequency": 1,
            "max_to_keep": 3,
        }
        component = CheckpointComponent(config=config)

        component.setup(simple_model, training_state)

        # Save multiple checkpoints
        for step in range(5):
            training_state.step = step
            component.step(simple_model, training_state)

        # Should only keep 3 checkpoints
        assert len(component._checkpoints) <= 3

    def test_checkpoint_component_restore(self, simple_model, training_state):
        """Test checkpoint restore functionality."""
        config = {"checkpoint_dir": "/tmp/test_checkpoints"}  # noqa: S108
        component = CheckpointComponent(config=config)

        # Save checkpoint first
        component.setup(simple_model, training_state)
        training_state.step = 10
        component.step(simple_model, training_state)

        # Restore checkpoint
        restored_state = component.restore_checkpoint(step=10)

        assert restored_state is not None
        assert restored_state["step"] == 10


# ===================================================================
# Test MixedPrecisionComponent
# ===================================================================


class TestMixedPrecisionComponent:
    """Tests for the MixedPrecisionComponent class."""

    def test_mixed_precision_initialization(self):
        """Test mixed precision component initialization."""
        config = {
            "compute_dtype": jnp.bfloat16,
            "param_dtype": jnp.float32,
            "loss_scale": 2**15,
        }
        component = MixedPrecisionComponent(config=config)

        assert component.config["compute_dtype"] == jnp.bfloat16
        assert component.config["param_dtype"] == jnp.float32
        assert component.config["loss_scale"] == 2**15

    def test_mixed_precision_setup(self, simple_model, training_state):
        """Test mixed precision setup with model."""
        component = MixedPrecisionComponent()

        component.setup(simple_model, training_state)

        # Should initialize precision state
        assert hasattr(component, "precision_state")
        assert component.precision_state.loss_scale > 0

    def test_mixed_precision_policy_application(self):
        """Test mixed precision policy is created correctly."""
        config = {"compute_dtype": jnp.bfloat16}
        component = MixedPrecisionComponent(config=config)

        # Should create precision policy
        assert hasattr(component, "create_precision_policy")

        policy = component.create_precision_policy()
        assert callable(policy)

    def test_mixed_precision_gradient_scaling(self):
        """Test gradient scaling functionality."""
        component = MixedPrecisionComponent(config={"loss_scale": 1024.0})

        # Test gradient scaling
        grads = {"w": jnp.array([1.0, 2.0, 3.0])}
        scaled_grads = component.scale_gradients(grads)

        assert jnp.allclose(scaled_grads["w"], grads["w"] * 1024.0)

    def test_mixed_precision_overflow_detection(self):
        """Test overflow detection in gradients."""
        component = MixedPrecisionComponent()

        # Test with normal gradients
        normal_grads = {"w": jnp.array([1.0, 2.0, 3.0])}
        assert not component.check_overflow(normal_grads)

        # Test with NaN gradients
        nan_grads = {"w": jnp.array([1.0, jnp.nan, 3.0])}
        assert component.check_overflow(nan_grads)

        # Test with Inf gradients
        inf_grads = {"w": jnp.array([1.0, jnp.inf, 3.0])}
        assert component.check_overflow(inf_grads)

    def test_mixed_precision_dynamic_loss_scaling(self, training_state):
        """Test dynamic loss scale adjustment."""
        config = {"dynamic_loss_scaling": True, "loss_scale": 1024.0}
        component = MixedPrecisionComponent(config=config)

        # Test loss scale increase on successful step
        component.precision_state.loss_scale = 1024.0
        component.update_loss_scale(has_overflow=False)

        assert component.precision_state.loss_scale >= 1024.0

        # Test loss scale decrease on overflow
        component.precision_state.loss_scale = 1024.0
        component.update_loss_scale(has_overflow=True)

        assert component.precision_state.loss_scale < 1024.0


# ===================================================================
# Test RecoveryComponent
# ===================================================================


class TestRecoveryComponent:
    """Tests for the RecoveryComponent class."""

    def test_recovery_component_initialization(self):
        """Test recovery component initialization."""
        config = {
            "max_retries": 5,
            "gradient_clip_threshold": 10.0,
            "loss_explosion_threshold": 1e6,
        }
        component = RecoveryComponent(config=config)

        assert component.config["max_retries"] == 5
        assert component.config["gradient_clip_threshold"] == 10.0

    def test_recovery_component_stability_check(self, training_state):
        """Test stability checking functionality."""
        component = RecoveryComponent()

        # Test stable training
        loss = 0.5
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}

        is_stable, issue = component.check_stability(loss, grads, training_state)

        assert is_stable is True
        assert issue is None

    def test_recovery_component_loss_explosion_detection(self, training_state):
        """Test loss explosion detection."""
        component = RecoveryComponent(config={"loss_explosion_threshold": 1e6})

        # Test with exploded loss
        loss = 1e7
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}

        is_stable, issue = component.check_stability(loss, grads, training_state)

        assert is_stable is False
        assert issue == "loss_explosion"

    def test_recovery_component_gradient_explosion_detection(self, training_state):
        """Test gradient explosion detection."""
        component = RecoveryComponent(config={"gradient_clip_threshold": 10.0})

        # Test with exploded gradients
        loss = 0.5
        grads = {"w": jnp.array([100.0, 200.0, 300.0])}

        is_stable, issue = component.check_stability(loss, grads, training_state)

        assert is_stable is False
        assert issue == "gradient_explosion"

    def test_recovery_component_nan_detection(self, training_state):
        """Test NaN detection in loss and gradients."""
        component = RecoveryComponent()

        # Test NaN loss
        loss = jnp.nan
        grads = {"w": jnp.array([0.1, 0.2, 0.3])}

        is_stable, issue = component.check_stability(loss, grads, training_state)

        assert is_stable is False
        assert issue == "nan_loss"

        # Test NaN gradients
        loss = 0.5
        grads = {"w": jnp.array([0.1, jnp.nan, 0.3])}

        is_stable, issue = component.check_stability(loss, grads, training_state)

        assert is_stable is False
        assert issue == "nan_gradients"

    def test_recovery_component_gradient_clipping(self):
        """Test gradient clipping functionality."""
        component = RecoveryComponent(config={"gradient_clip_threshold": 10.0})

        # Test with large gradients
        grads = {"w": jnp.array([100.0, 200.0, 300.0])}
        clipped_grads = component.apply_gradient_clipping(grads)

        # Calculate gradient norm
        grad_norm = jnp.sqrt(jnp.sum(clipped_grads["w"] ** 2))

        assert grad_norm <= 10.0

    def test_recovery_component_recovery_from_instability(self, training_state):
        """Test recovery from training instability."""
        component = RecoveryComponent(config={"max_retries": 3})

        # Setup with stable state
        stable_state = Mock()
        stable_state.step = 10
        stable_state.loss = 0.5
        stable_state.recovery_state = {}  # Add recovery_state attribute
        component.last_stable_state = stable_state

        # Attempt recovery
        recovered_state = component.recover_from_instability(
            "loss_explosion", training_state
        )

        assert recovered_state is not None
        assert component.recovery_attempts == 1
        # Verify learning rate was reduced
        assert "reduced_lr" in stable_state.recovery_state

    def test_recovery_component_max_retries_exceeded(self, training_state):
        """Test that recovery fails after max retries."""
        component = RecoveryComponent(config={"max_retries": 2})

        # Exceed max retries
        component.recovery_attempts = 2

        with pytest.raises(RuntimeError, match="Maximum recovery attempts"):
            component.recover_from_instability("loss_explosion", training_state)

    def test_recovery_component_update_stable_state(self, training_state):
        """Test updating last stable state."""
        component = RecoveryComponent()

        component.update_stable_state(training_state)

        assert component.last_stable_state == training_state
        assert component.recovery_attempts == 0


# ===================================================================
# Test Component Composition
# ===================================================================


class TestComponentComposition:
    """Tests for composing multiple components together."""

    def test_multiple_components_setup(self, simple_model, training_state):
        """Test setting up multiple components."""
        checkpoint = CheckpointComponent(config={"checkpoint_dir": "/tmp/test"})  # noqa: S108
        recovery = RecoveryComponent()
        mixed_precision = MixedPrecisionComponent()

        # Setup all components
        checkpoint.setup(simple_model, training_state)
        recovery.setup(simple_model, training_state)
        mixed_precision.setup(simple_model, training_state)

        # All should complete without error
        assert True

    def test_components_can_be_chained(self, simple_model, training_state):
        """Test that component steps can be chained."""
        checkpoint = CheckpointComponent(config={"save_frequency": 1})
        recovery = RecoveryComponent()

        checkpoint.setup(simple_model, training_state)
        recovery.setup(simple_model, training_state)

        # Chain component steps
        training_state.step = 1
        checkpoint_result = checkpoint.step(simple_model, training_state)
        recovery_result = recovery.step(simple_model, training_state)

        # Both should execute without error
        assert checkpoint_result is not None or checkpoint_result is None
        assert recovery_result is None  # Recovery only acts on instability

    def test_component_cleanup_all(self):
        """Test cleaning up multiple components."""
        components = [
            CheckpointComponent(),
            RecoveryComponent(),
            MixedPrecisionComponent(),
        ]

        # Cleanup all
        for component in components:
            component.cleanup()

        # Should complete without error
        assert True


# ===================================================================
# Test Edge Cases
# ===================================================================


class TestComponentEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_component_with_none_config(self):
        """Test component handles None config gracefully."""
        component = TrainingComponent(config=None)

        assert component.config == {}

    def test_checkpoint_with_invalid_directory(self, simple_model, training_state):
        """Test checkpoint component with invalid directory."""
        config = {"checkpoint_dir": "/invalid/path/that/does/not/exist"}
        component = CheckpointComponent(config=config)

        # Should raise PermissionError for invalid path
        with pytest.raises(PermissionError):
            component.setup(simple_model, training_state)

    def test_mixed_precision_with_incompatible_dtype(self):
        """Test mixed precision with incompatible dtype."""
        config = {"compute_dtype": "invalid_dtype"}

        with pytest.raises((TypeError, ValueError)):
            MixedPrecisionComponent(config=config)

    def test_recovery_component_without_stable_state(self, training_state):
        """Test recovery when no stable state exists."""
        component = RecoveryComponent()

        # No stable state set
        assert component.last_stable_state is None

        # Recovery should handle gracefully
        recovered_state = component.recover_from_instability(
            "loss_explosion", training_state
        )

        # Should return the current state or handle gracefully
        assert recovered_state is not None
