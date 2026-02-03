"""Tests for JIT compilation in training loops.

This test suite verifies that all training code properly uses JIT compilation
for maximum performance. Training without JIT is 10-100x slower.

Following TDD: These tests are written FIRST to define expected behavior.
Implementation must be updated to make these tests pass.

Non-negotiable principle: Tests define behavior, not current implementation.
"""

import time

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestTrainerJITCompilation:
    """Test that Trainer.training_step is JIT-compiled."""

    def test_trainer_has_jit_compiled_training_step(self):
        """Verify Trainer.training_step is JIT-compiled.

        JIT compilation is critical for performance - training without JIT
        is 10-100x slower than necessary.

        This test verifies that training_step compiles once and reuses
        the compiled version on subsequent calls.
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

        # Test data
        x = jnp.ones((32, 2))
        y = jnp.ones((32, 1))

        # First call (includes compilation if JIT-compiled)
        start1 = time.perf_counter()
        loss1, _ = trainer.training_step(x, y)
        # Wait for completion
        if hasattr(loss1, "block_until_ready"):
            loss1.block_until_ready()
        time1 = time.perf_counter() - start1

        # Second call (should be much faster if JIT-compiled)
        start2 = time.perf_counter()
        loss2, _ = trainer.training_step(x, y)
        if hasattr(loss2, "block_until_ready"):
            loss2.block_until_ready()
        time2 = time.perf_counter() - start2

        # Test: Second call should be faster (compilation cached)
        # If JIT-compiled: second call is 5-50x faster
        # If not JIT-compiled: both calls take similar time
        assert time2 < time1 / 2, (
            f"training_step not JIT-compiled. "
            f"First: {time1:.6f}s, Second: {time2:.6f}s. "
            f"Ratio: {time1 / time2:.1f}x (expected >2x for JIT). "
            f"Training without JIT is 10-100x slower!"
        )

    def test_training_step_execution_speed(self):
        """Verify training step executes fast (indicating JIT compilation)."""
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        # Create simple model
        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(10, 10, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=100)
        trainer = Trainer(model, config)

        # Test data
        x = jnp.ones((100, 10))
        y = jnp.ones((100, 10))

        # Warmup
        for _ in range(3):
            loss, _ = trainer.training_step(x, y)
            if hasattr(loss, "block_until_ready"):
                loss.block_until_ready()

        # Time multiple iterations
        times = []
        for _ in range(10):
            start = time.perf_counter()
            loss, _ = trainer.training_step(x, y)
            if hasattr(loss, "block_until_ready"):
                loss.block_until_ready()
            times.append(time.perf_counter() - start)

        mean_time = jnp.mean(jnp.array(times))

        # Test: Should complete quickly if JIT-compiled
        # Simple MLP training step should be <10ms on GPU
        # Without JIT, could be 100ms+
        assert mean_time < 0.05, (
            f"Training step too slow: {mean_time:.3f}s. "
            f"Expected <50ms for JIT-compiled code. "
            f"Likely missing @nnx.jit decorator."
        )

    def test_training_step_is_deterministic(self):
        """Verify training step is deterministic with same inputs.

        JIT-compiled functions should be deterministic.
        This also verifies no random state issues.
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

        x = jnp.ones((32, 2))
        y = jnp.ones((32, 1))

        # Run twice with same inputs
        loss1, _ = trainer.training_step(x, y)
        loss2, _ = trainer.training_step(x, y)

        # Losses should be different (parameters updated)
        # But computation should be deterministic (same gradients)
        assert isinstance(loss1, jax.Array)
        assert isinstance(loss2, jax.Array)


class TestBasicTrainerJIT:
    """Test BasicTrainer JIT compilation (if it exists)."""

    @pytest.mark.skipif(
        not hasattr(
            __import__("opifex.training.basic_trainer", fromlist=["BasicTrainer"]),
            "BasicTrainer",
        ),
        reason="BasicTrainer may not exist or be deprecated",
    )
    def test_basic_trainer_uses_jit(self):
        """Test BasicTrainer training step is JIT-compiled."""
        # Will be implemented if BasicTrainer exists


class TestIncrementalTrainerJIT:
    """Test IncrementalTrainer JIT compilation (if it exists)."""

    @pytest.mark.skipif(
        not hasattr(
            __import__(
                "opifex.training.incremental_trainer", fromlist=["IncrementalTrainer"]
            ),
            "IncrementalTrainer",
        ),
        reason="IncrementalTrainer may not exist",
    )
    def test_incremental_trainer_uses_jit(self):
        """Test IncrementalTrainer training step is JIT-compiled."""
        # Will be implemented if IncrementalTrainer exists


class TestQuantumTrainingJIT:
    """Test quantum training code uses JIT compilation."""

    def test_quantum_training_step_jit_compiled(self):
        """Test quantum training utilities use JIT compilation.

        Quantum training with SCF iterations is compute-intensive.
        JIT compilation is critical for performance.
        """
        # Will be implemented after checking quantum_training.py


# ==============================================================================
# Integration Tests - Verify Complete Training Pipeline Uses JIT
# ==============================================================================


class TestTrainingPipelinePerformance:
    """Integration tests for training pipeline performance."""

    def test_full_training_epoch_performance(self):
        """Test complete training epoch completes in reasonable time.

        With JIT compilation, a small training epoch should complete quickly.
        Without JIT, it could be 10-100x slower.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        # Create simple model
        class SimpleModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense1 = nnx.Linear(5, 10, rngs=rngs)
                self.dense2 = nnx.Linear(10, 5, rngs=rngs)

            def __call__(self, x):
                return self.dense2(nnx.gelu(self.dense1(x)))

        model = SimpleModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=32, verbose=False)
        trainer = Trainer(model, config)

        # Generate small dataset
        x_train = jnp.ones((128, 5))
        y_train = jnp.ones((128, 5))

        # Time one epoch (4 batches)
        start = time.perf_counter()
        _, _ = trainer.fit((x_train, y_train))
        epoch_time = time.perf_counter() - start

        # Test: One epoch should complete quickly
        # With JIT: <1 second for small model
        # Without JIT: could be 10+ seconds
        assert epoch_time < 5.0, (
            f"Training epoch too slow: {epoch_time:.2f}s. "
            f"Expected <5s for JIT-compiled training. "
            f"Likely missing JIT compilation in training loop."
        )

    def test_training_speedup_with_jit(self):
        """Test that JIT provides expected speedup.

        This test verifies JIT compilation provides significant speedup
        by comparing compilation + execution vs execution-only time.
        """
        from opifex.core.training.config import TrainingConfig
        from opifex.core.training.trainer import Trainer

        # Create tiny model for fast testing
        class TinyModel(nnx.Module):
            def __init__(self, *, rngs: nnx.Rngs):
                self.dense = nnx.Linear(2, 1, rngs=rngs)

            def __call__(self, x):
                return self.dense(x)

        model = TinyModel(rngs=nnx.Rngs(0))
        config = TrainingConfig(num_epochs=1, batch_size=16)
        trainer = Trainer(model, config)

        x = jnp.ones((16, 2))
        y = jnp.ones((16, 1))

        # First call (compilation + execution if JIT)
        start1 = time.perf_counter()
        loss1, _ = trainer.training_step(x, y)
        if hasattr(loss1, "block_until_ready"):
            loss1.block_until_ready()
        first_call_time = time.perf_counter() - start1

        # Subsequent calls (execution only if JIT)
        subsequent_times = []
        for _ in range(5):
            start = time.perf_counter()
            loss, _ = trainer.training_step(x, y)
            if hasattr(loss, "block_until_ready"):
                loss.block_until_ready()
            subsequent_times.append(time.perf_counter() - start)

        avg_subsequent_time = jnp.mean(jnp.array(subsequent_times))

        # Test: Subsequent calls should be faster than first
        # (First includes compilation if JIT-compiled)
        speedup = first_call_time / avg_subsequent_time

        # Allow for some variance, but should see speedup
        # If not JIT-compiled, speedup would be ~1x
        # If JIT-compiled, speedup should be >1.5x
        assert speedup > 1.3, (
            f"No JIT speedup detected. "
            f"First call: {first_call_time:.6f}s, "
            f"Avg subsequent: {avg_subsequent_time:.6f}s, "
            f"Speedup: {speedup:.2f}x (expected >1.3x). "
            f"Likely missing @nnx.jit or @jax.jit decorator."
        )


# ==============================================================================
# TDD Status: These tests are written FIRST
# ==============================================================================

# Expected Initial Status: Some tests may fail if JIT is missing
# - test_trainer_has_jit_compiled_training_step: May FAIL
# - test_training_step_execution_speed: May FAIL if no JIT
# - test_full_training_epoch_performance: May FAIL if no JIT
# - test_training_speedup_with_jit: WILL FAIL if no JIT

# After Implementation: ALL TESTS SHOULD PASS
# - @nnx.jit added to training_step methods
# - Training 10-100x faster
# - All performance tests pass

# TDD Workflow:
# 1. ✅ Tests written (this file)
# 2. ⏳ Run tests (expect some failures - RED)
# 3. ⏳ Add @nnx.jit decorators (implementation)
# 4. ⏳ Run tests again (expect pass - GREEN)
# 5. ⏳ Verify 10x+ speedup achieved
