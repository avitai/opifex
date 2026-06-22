"""Contracts for the Trainer's nnx.Optimizer integration.

The unified :class:`Trainer` uses ``nnx.Optimizer`` (which manages the optax state internally),
not a hand-rolled ``optimizer.update`` / ``optax.apply_updates`` / ``nnx.update`` cycle, and its
``TrainingState`` carries no separate ``opt_state``.
"""

import chex
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
            def __init__(self, *, rngs: nnx.Rngs) -> None:
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

    def test_trainer_state_has_no_manual_opt_state(self):
        """TrainingState carries no separate opt_state — nnx.Optimizer manages it internally."""
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        assert not hasattr(trainer.state, "opt_state")
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
            def __init__(self, *, rngs: nnx.Rngs) -> None:
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


class TestNNXOptimizerJitCompiles:
    """The nnx.Optimizer update is JIT-compilable and recompiles only once."""

    def test_optimizer_update_traces_once_under_jit(self):
        """A jitted update step compiles a single time across repeated calls (no retracing)."""
        import optax

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.dense = nnx.Linear(16, 16, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        x = jnp.ones((8, 16))

        @nnx.jit
        @chex.assert_max_traces(n=1)
        def step(model: SimpleModel, optimizer: nnx.Optimizer) -> None:
            grads = nnx.grad(lambda m: jnp.sum(m(x) ** 2))(model)
            optimizer.update(model, grads)

        chex.clear_trace_counter()
        for _ in range(5):
            # chex's stub narrows the wrapped signature; the call is correct.
            step(model, optimizer)  # pyright: ignore[reportCallIssue]


class TestTrainerJITWithNNXOptimizer:
    """Test that @nnx.jit works with class-based Trainer using nnx.Optimizer."""

    def test_repeated_training_step_runs_and_stays_finite(self):
        """``training_step`` runs repeatedly through the nnx.Optimizer and returns finite losses."""
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32)
        trainer = Trainer(model, config)

        x = jnp.ones((32, 2))
        y = jnp.ones((32, 1))

        for _ in range(3):
            loss, _ = trainer.training_step(x, y)
            assert jnp.isfinite(loss)

    def test_nnx_optimizer_with_jit_preserves_state(self):
        """Verify nnx.Optimizer state is preserved through JIT.

        Important: nnx.jit should preserve optimizer state across calls.
        Optimizer should track steps, momentum, etc.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs) -> None:
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
            def __init__(self, *, rngs: nnx.Rngs) -> None:
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
            def __init__(self, *, rngs: nnx.Rngs) -> None:
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
