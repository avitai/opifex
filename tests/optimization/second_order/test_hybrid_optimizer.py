"""Tests for hybrid Adam→L-BFGS optimizer.

TDD: These tests define the expected behavior for the HybridOptimizer.
"""

import jax
import jax.numpy as jnp
import optax

from opifex.optimization.second_order.config import (
    HybridOptimizerConfig,
    LBFGSConfig,
    SwitchCriterion,
)
from opifex.optimization.second_order.hybrid_optimizer import (
    HybridOptimizer,
    HybridOptimizerState,
)


class TestHybridOptimizerInit:
    """Test hybrid optimizer initialization."""

    def test_creates_with_default_config(self):
        """Should create with default configuration."""
        config = HybridOptimizerConfig()
        optimizer = HybridOptimizer(config)
        assert optimizer is not None
        assert optimizer.config == config

    def test_creates_with_custom_config(self):
        """Should create with custom configuration."""
        config = HybridOptimizerConfig(
            first_order_steps=500,
            adam_learning_rate=1e-4,
            lbfgs_config=LBFGSConfig(memory_size=5),
        )
        optimizer = HybridOptimizer(config)
        assert optimizer.config.first_order_steps == 500
        assert optimizer.config.adam_learning_rate == 1e-4

    def test_init_creates_valid_state(self):
        """init() should create valid optimizer state."""
        config = HybridOptimizerConfig()
        optimizer = HybridOptimizer(config)
        params = {"w": jnp.ones(10), "b": jnp.zeros(5)}

        state = optimizer.init(params)

        assert isinstance(state, HybridOptimizerState)
        assert state.step_count == 0
        assert state.using_lbfgs is False
        assert state.switched is False
        assert state.loss_history.shape == (config.loss_history_window,)


class TestHybridOptimizerUpdate:
    """Test hybrid optimizer update step."""

    def test_update_with_adam_initially(self):
        """Should use Adam optimizer initially."""
        config = HybridOptimizerConfig(first_order_steps=100)
        optimizer = HybridOptimizer(config)
        params = jnp.array([1.0, 2.0, 3.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        _updates, new_state = optimizer.update(grads, state, params, loss=loss)

        assert new_state.step_count == 1
        assert new_state.using_lbfgs is False
        assert new_state.switched is False

    def test_update_tracks_loss_history(self):
        """Should track loss history for variance computation."""
        config = HybridOptimizerConfig(loss_history_window=10)
        optimizer = HybridOptimizer(config)
        params = jnp.array([1.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # Run a few steps
        for _ in range(5):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(grads, state, params, loss=loss)
            params = optax.apply_updates(params, updates)

        # Loss history should have finite values at the end
        assert jnp.isfinite(state.loss_history[-1])

    def test_update_increments_step_count(self):
        """Step count should increment with each update."""
        config = HybridOptimizerConfig()
        optimizer = HybridOptimizer(config)
        params = jnp.ones(5)
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        for i in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(grads, state, params, loss=loss)
            params = optax.apply_updates(params, updates)
            assert state.step_count == i + 1


class TestSwitchingBehavior:
    """Test optimizer switching from Adam to L-BFGS."""

    def test_switch_on_epoch_criterion(self):
        """Should switch to L-BFGS after first_order_steps with EPOCH criterion."""
        config = HybridOptimizerConfig(
            first_order_steps=5,
            switch_criterion=SwitchCriterion.EPOCH,
        )
        optimizer = HybridOptimizer(config)
        params = jnp.array([1.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # Run up to first_order_steps
        for _ in range(5):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(
                grads, state, params, loss=loss, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)

        # Should not have switched yet (step 5 is first check)
        assert state.using_lbfgs is False

        # One more step should trigger switch
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, state = optimizer.update(
            grads, state, params, loss=loss, value_fn=loss_fn
        )

        assert state.switched is True
        assert state.using_lbfgs is True

    def test_switch_persists_after_triggering(self):
        """Once switched to L-BFGS, should stay on L-BFGS."""
        config = HybridOptimizerConfig(
            first_order_steps=2,
            switch_criterion=SwitchCriterion.EPOCH,
        )
        optimizer = HybridOptimizer(config)
        params = jnp.array([5.0, -3.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # Run until switch
        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(
                grads, state, params, loss=loss, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)

        # Should be using L-BFGS and stay on it
        assert state.switched is True
        assert state.using_lbfgs is True

    def test_no_switch_before_first_order_steps(self):
        """Should not switch before first_order_steps even with low variance."""
        config = HybridOptimizerConfig(
            first_order_steps=100,
            switch_criterion=SwitchCriterion.LOSS_VARIANCE,
            loss_variance_threshold=1.0,  # Very high threshold
        )
        optimizer = HybridOptimizer(config)
        params = jnp.ones(3)
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # Run fewer than first_order_steps
        for _ in range(50):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(grads, state, params, loss=loss)
            params = optax.apply_updates(params, updates)

        # Should still be on Adam
        assert state.using_lbfgs is False
        assert state.switched is False


class TestConvergence:
    """Test convergence behavior of hybrid optimizer."""

    def test_converges_on_quadratic(self):
        """Hybrid optimizer should converge on quadratic function."""
        config = HybridOptimizerConfig(
            first_order_steps=20,
            switch_criterion=SwitchCriterion.EPOCH,
            adam_learning_rate=0.1,
        )
        optimizer = HybridOptimizer(config)
        params = jnp.array([5.0, -3.0, 2.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return 0.5 * jnp.sum(p**2)

        initial_loss = loss_fn(params)

        # Run optimization
        for _ in range(50):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(
                grads, state, params, loss=loss, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)

        final_loss = loss_fn(params)

        # Should make significant progress
        assert final_loss < initial_loss * 0.1

    def test_adam_phase_makes_progress(self):
        """Adam phase should make progress before switch."""
        config = HybridOptimizerConfig(
            first_order_steps=50,
            switch_criterion=SwitchCriterion.EPOCH,
            adam_learning_rate=0.01,
        )
        optimizer = HybridOptimizer(config)
        params = jnp.array([5.0, -3.0])
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        initial_loss = loss_fn(params)

        # Run only Adam phase
        for _ in range(50):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(grads, state, params, loss=loss)
            params = optax.apply_updates(params, updates)

        assert state.using_lbfgs is False
        final_loss = loss_fn(params)
        assert final_loss < initial_loss


class TestGradientNormSwitching:
    """Test gradient norm based switching."""

    def test_switch_on_low_gradient_norm(self):
        """Should switch when gradient norm drops below threshold."""
        config = HybridOptimizerConfig(
            first_order_steps=5,
            switch_criterion=SwitchCriterion.GRADIENT_NORM,
            gradient_norm_threshold=100.0,  # High threshold to trigger early
        )
        optimizer = HybridOptimizer(config)
        params = jnp.array([0.1, 0.1])  # Near optimum, small gradients
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        # Run optimization
        for _ in range(10):
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, state = optimizer.update(
                grads, state, params, loss=loss, value_fn=loss_fn
            )
            params = optax.apply_updates(params, updates)

        # Should have switched due to low gradient norm
        assert state.switched is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_handles_none_loss(self):
        """Should handle None loss value gracefully."""
        config = HybridOptimizerConfig(first_order_steps=100)
        optimizer = HybridOptimizer(config)
        params = jnp.ones(3)
        state = optimizer.init(params)

        grads = jnp.ones(3) * 0.1
        # Should not crash with None loss
        _updates, new_state = optimizer.update(grads, state, params, loss=None)
        assert new_state is not None

    def test_value_alias_for_loss(self):
        """Should accept 'value' as alias for 'loss' parameter."""
        config = HybridOptimizerConfig()
        optimizer = HybridOptimizer(config)
        params = jnp.ones(3)
        state = optimizer.init(params)

        def loss_fn(p):
            return jnp.sum(p**2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        # Use value= instead of loss=
        _updates, new_state = optimizer.update(grads, state, params, value=loss)
        assert new_state.step_count == 1
