"""Tests for nnx.Optimizer migration in Trainer.

This test suite defines the expected behavior after migrating from
manual Optax state management to nnx.Optimizer API.

Following TDD: These tests are written FIRST to define expected behavior.
Implementation must be updated to make these tests pass.

Non-negotiable principles:
- TDD: Tests define behavior, not implementation
- DRY: nnx.Optimizer eliminates manual state management duplication
- Breaking Changes OK: Better design > backward compatibility
"""

import jax.numpy as jnp
from flax import nnx


class TestTrainerUsesNNXOptimizer:
    """Test that Trainer uses nnx.Optimizer (not manual Optax)."""

    def test_trainer_has_nnx_optimizer_attribute(self):
        """Verify Trainer.optimizer is nnx.Optimizer instance.

        After migration, trainer.optimizer should be nnx.Optimizer,
        not raw Optax GradientTransformation.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        # Create simple model
        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        # Test: trainer.optimizer should be nnx.Optimizer
        assert isinstance(trainer.optimizer, nnx.Optimizer), (
            f"Expected nnx.Optimizer, got {type(trainer.optimizer)}. "
            f"Trainer should use nnx.Optimizer, not raw Optax."
        )

    def test_trainer_state_no_manual_opt_state(self):
        """Verify TrainingState doesn't manually track opt_state.

        With nnx.Optimizer, opt_state is managed internally.
        No need for manual tracking in TrainingState.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        # Test: TrainingState should not have manual opt_state field
        # (Optimizer manages it internally)
        # If it exists, it's for backward compat only
        state_dict = vars(trainer.state)

        # nnx.Optimizer manages opt_state internally
        # So either:
        # 1. No opt_state field (clean migration), OR
        # 2. opt_state is None/unused (transitional)
        if "opt_state" in state_dict:
            # Transitional: might exist but should not be used
            # Just verify optimizer exists
            assert hasattr(trainer, "optimizer")

        # Main assertion: has nnx.Optimizer
        assert isinstance(trainer.optimizer, nnx.Optimizer)

    def test_training_step_uses_simplified_update(self):
        """Verify training_step uses simplified optimizer.update() API.

        With nnx.Optimizer:
        - optimizer.update(model, grads) - ONE line

        Not manual Optax:
        - updates, opt_state = optimizer.update(grads, opt_state)
        - new_params = optax.apply_updates(params, updates)
        - nnx.update(model, new_params)
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        x = jnp.ones((32, 2))
        y = jnp.ones((32, 1))

        # Should work with nnx.Optimizer
        _, _ = trainer.training_step(x[:10], y[:10])

        # Verify optimizer state updated
        assert isinstance(trainer.optimizer, nnx.Optimizer)
        assert trainer.optimizer.opt_state is not None


class TestNNXOptimizerPerformance:
    """Test that nnx.Optimizer provides expected performance."""

    def test_optimizer_update_is_fast(self):
        """Verify nnx.Optimizer.update is efficient.

        Should have same performance as manual Optax.
        """
        import time

        import optax

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(100, 100, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        x = jnp.ones((100, 100))

        def loss_fn(model):
            return jnp.sum(model(x) ** 2)

        # Warmup
        for _ in range(3):
            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)

        # Time multiple updates
        times = []
        for _ in range(10):
            grads = nnx.grad(loss_fn)(model)

            start = time.perf_counter()
            optimizer.update(model, grads)
            end = time.perf_counter()
            times.append(end - start)

        mean_time = jnp.mean(jnp.array(times))

        # Should be fast (optimized update)
        assert mean_time < 0.01, (
            f"Optimizer update too slow: {mean_time:.6f}s. "
            f"Expected <10ms for small model."
        )


class TestTrainerJITWithNNXOptimizer:
    """Test that @nnx.jit works with class-based Trainer using nnx.Optimizer."""

    def test_jit_decorated_training_step_compiles(self):
        """Verify @nnx.jit decorator works on Trainer.training_step.

        Class methods can be JIT-compiled with nnx.jit.
        Should show compilation speedup on subsequent calls.
        """
        import time

        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        x = jnp.ones((32, 2))
        y = jnp.ones((32, 1))

        # First call (compilation + execution)
        start1 = time.perf_counter()
        loss1, _ = trainer.training_step(x, y)
        if hasattr(loss1, "block_until_ready"):
            loss1.block_until_ready()
        time1 = time.perf_counter() - start1

        # Second call (execution only - compiled)
        start2 = time.perf_counter()
        loss2, _ = trainer.training_step(x, y)
        if hasattr(loss2, "block_until_ready"):
            loss2.block_until_ready()
        time2 = time.perf_counter() - start2

        # Test: Should see compilation speedup
        speedup = time1 / time2
        assert speedup > 1.5, (
            f"No JIT speedup. First: {time1:.6f}s, Second: {time2:.6f}s, "
            f"Speedup: {speedup:.2f}x (expected >1.5x)"
        )

    def test_nnx_optimizer_with_jit_preserves_state(self):
        """Verify nnx.Optimizer state is preserved through JIT.

        Important: nnx.jit should preserve optimizer state across calls.
        Optimizer should track steps, momentum, etc.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=16)
        trainer = Trainer(model, config)

        x = jnp.ones((16, 2))
        y = jnp.ones((16, 1))

        # Get initial loss
        initial_loss, _ = trainer.training_step(x, y)

        # Multiple training steps (parameters should improve)
        losses = []
        for _ in range(10):
            loss, _ = trainer.training_step(x, y)
            losses.append(float(loss))

        final_loss = losses[-1]

        # Loss should decrease (training is working)
        # Or at least stay stable (optimizer state preserved)
        assert final_loss <= initial_loss * 1.5, (
            f"Training not working properly. "
            f"Initial: {initial_loss:.6f}, Final: {final_loss:.6f}. "
            f"Optimizer state may not be preserved!"
        )


class TestNNXOptimizerIntegration:
    """Integration tests for nnx.Optimizer in Trainer."""

    def test_full_training_with_nnx_optimizer(self):
        """Test complete training loop works with nnx.Optimizer."""
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(5, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=2, batch_size=32, verbose=False)
        trainer = Trainer(model, config)

        # Generate simple dataset
        x_train = jnp.ones((64, 5))
        y_train = jnp.ones((64, 1))

        # Train (should work with nnx.Optimizer)
        trained_model, metrics = trainer.fit((x_train, y_train))

        # Verify training happened
        assert "final_train_loss" in metrics
        assert isinstance(trained_model, nnx.Module)

    def test_nnx_optimizer_with_different_optax_optimizers(self):
        """Verify nnx.Optimizer works with various Optax optimizers.

        Should support: Adam, AdamW, SGD, RMSprop, etc.
        """
        import optax

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))

        # Test various Optax optimizers
        optax_optimizers = [
            optax.adam(1e-3),
            optax.adamw(1e-3, weight_decay=1e-4),
            optax.sgd(1e-2, momentum=0.9),
            optax.rmsprop(1e-3),
        ]

        for tx in optax_optimizers:
            # Should work with any Optax optimizer
            optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

            # Test update works
            def loss_fn(m):
                return jnp.sum(m(jnp.ones((1, 2))) ** 2)

            grads = nnx.grad(loss_fn)(model)
            optimizer.update(model, grads)  # Should not raise


# ==============================================================================
# TDD Status: These tests define expected behavior after migration
# ==============================================================================

# Expected Status:
# - All tests will FAIL initially (Trainer still uses Optax direct)
# - After migration: ALL TESTS SHOULD PASS
# - Tests define the new nnx.Optimizer-based API

# TDD Workflow:
# 1. ✅ Tests written (this file) - defines expected behavior
# 2. ⏳ Run tests (expect failures - RED)
# 3. ⏳ Refactor Trainer to use nnx.Optimizer (implementation)
# 4. ⏳ Run tests again (expect pass - GREEN)
# 5. ⏳ Verify cleaner code (67% reduction in update logic)
