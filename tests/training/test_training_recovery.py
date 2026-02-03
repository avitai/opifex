"""Tests for error recovery module."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from opifex.neural.base import StandardMLP
from opifex.training.metrics import TrainingState
from opifex.training.recovery import ErrorRecoveryManager


class TestErrorRecoveryManager:
    """Test error recovery manager."""

    def test_initialization(self):
        """Test error recovery manager initialization."""
        config = {
            "max_retries": 5,
            "checkpoint_on_error": True,
            "gradient_clip_threshold": 15.0,
            "loss_explosion_threshold": 1e8,
        }
        manager = ErrorRecoveryManager(config)

        assert manager.max_retries == 5
        assert manager.checkpoint_on_error is True
        assert manager.gradient_clip_threshold == 15.0
        assert manager.loss_explosion_threshold == 1e8
        assert manager.recovery_attempts == 0
        assert manager.last_stable_state is None

    def test_default_initialization(self):
        """Test error recovery manager with default config."""
        manager = ErrorRecoveryManager()

        assert manager.max_retries == 3
        assert manager.checkpoint_on_error is True
        assert manager.gradient_clip_threshold == 10.0
        assert manager.loss_explosion_threshold == 1e6
        assert manager.recovery_attempts == 0

    def test_setup(self):
        """Test setup method."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        manager.setup(model, training_state)

        assert manager.last_stable_state is not None
        assert manager.recovery_attempts == 0

    def test_check_training_stability_loss_explosion(self):
        """Test training stability check - loss explosion."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Create normal gradients
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)

        # Test with exploded loss
        is_stable, issue = manager.check_training_stability(
            loss=1e7,  # Above default threshold of 1e6
            grads=grads,
            training_state=training_state,
        )

        assert not is_stable
        assert issue == "loss_explosion"

    def test_check_training_stability_gradient_explosion(self):
        """Test training stability check - gradient explosion."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Create large gradients (norm > 10.0)
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 20.0, params)

        is_stable, issue = manager.check_training_stability(
            loss=0.5,
            grads=grads,
            training_state=training_state,
        )

        assert not is_stable
        assert issue == "gradient_explosion"

    def test_check_training_stability_nan_loss(self):
        """Test training stability check - NaN loss."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)

        is_stable, issue = manager.check_training_stability(
            loss=float("nan"),
            grads=grads,
            training_state=training_state,
        )

        assert not is_stable
        assert issue == "nan_loss"

    def test_check_training_stability_nan_gradients(self):
        """Test training stability check - NaN gradients."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Create gradients with NaN values
        grads = jax.tree_util.tree_map(
            lambda x: jnp.ones_like(x) * float("nan"), params
        )

        is_stable, issue = manager.check_training_stability(
            loss=0.5,
            grads=grads,
            training_state=training_state,
        )

        assert not is_stable
        assert issue == "nan_gradients"

    def test_check_training_stability_stable(self):
        """Test training stability check - stable training."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Create normal gradients
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)

        is_stable, issue = manager.check_training_stability(
            loss=0.5,
            grads=grads,
            training_state=training_state,
        )

        assert is_stable
        assert issue is None

    def test_apply_gradient_clipping(self):
        """Test gradient clipping."""
        manager = ErrorRecoveryManager({"gradient_clip_threshold": 1.0})
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        params = nnx.to_tree(nnx.state(model, nnx.Param))

        # Create large gradients
        large_grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 10.0, params)

        # Compute original norm
        original_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(large_grads))
        )

        # Apply clipping
        clipped_grads = manager.apply_gradient_clipping(large_grads)

        # Compute clipped norm
        clipped_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(clipped_grads))
        )

        # Clipped norm should be less than original and close to threshold
        assert clipped_norm < original_norm
        assert clipped_norm <= manager.gradient_clip_threshold

    def test_apply_gradient_clipping_no_clip_needed(self):
        """Test gradient clipping when gradients are already small."""
        manager = ErrorRecoveryManager({"gradient_clip_threshold": 10.0})
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        params = nnx.to_tree(nnx.state(model, nnx.Param))

        # Create small gradients
        small_grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, params)

        # Apply clipping
        clipped_grads = manager.apply_gradient_clipping(small_grads)

        # Gradients should be unchanged
        for orig, clipped in zip(
            jax.tree_util.tree_leaves(small_grads),
            jax.tree_util.tree_leaves(clipped_grads),
            strict=False,
        ):
            assert jnp.allclose(orig, clipped)

    def test_recover_from_instability_loss_explosion(self):
        """Test recovery from loss explosion."""
        manager = ErrorRecoveryManager({"learning_rate": 1e-3})
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Set last stable state
        manager.last_stable_state = training_state

        # Attempt recovery
        recovered_state = manager.recover_from_instability(
            "loss_explosion", training_state
        )

        assert recovered_state is not None
        assert manager.recovery_attempts == 1
        # Learning rate should be reduced in recovery state
        assert "reduced_lr" in recovered_state.recovery_state

    def test_recover_from_instability_nan_loss(self):
        """Test recovery from NaN loss."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Set last stable state
        manager.last_stable_state = training_state

        # Attempt recovery
        recovered_state = manager.recover_from_instability("nan_loss", training_state)

        assert recovered_state is not None
        assert manager.recovery_attempts == 1
        assert "reinitialized" in recovered_state.recovery_state

    def test_recover_from_instability_max_retries_exceeded(self):
        """Test that recovery fails after max retries."""
        manager = ErrorRecoveryManager({"max_retries": 2})
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        manager.last_stable_state = training_state

        # Attempt recovery multiple times
        manager.recover_from_instability("loss_explosion", training_state)
        manager.recover_from_instability("loss_explosion", training_state)

        # Third attempt should raise error
        with pytest.raises(RuntimeError, match="Maximum recovery attempts"):
            manager.recover_from_instability("loss_explosion", training_state)

    def test_update_stable_state(self):
        """Test updating stable state."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            step=100,
        )

        # Simulate some recovery attempts
        manager.recovery_attempts = 3

        # Update stable state
        manager.update_stable_state(training_state)

        assert manager.last_stable_state is not None
        assert manager.last_stable_state.step == 100
        # Recovery attempts should be reset
        assert manager.recovery_attempts == 0

    def test_cleanup(self):
        """Test cleanup method."""
        manager = ErrorRecoveryManager()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        training_state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        manager.setup(model, training_state)
        assert manager.last_stable_state is not None

        # Cleanup should work without errors
        manager.cleanup()
