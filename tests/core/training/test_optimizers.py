"""Tests for optimizer creation module.

Following strict TDD - these tests define the API and expected behavior
BEFORE implementation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
import pytest

from opifex.core.training.optimizers import (
    create_adam,
    create_adamw,
    create_optimizer,
    create_rmsprop,
    create_schedule,
    create_sgd,
    OptimizerConfig,
    with_gradient_clipping,
    with_schedule,
)


class TestOptimizerCreation:
    """Test basic optimizer creation functions."""

    def test_create_adam_default_params(self):
        """Test Adam optimizer with default parameters."""
        optimizer = create_adam()

        # Verify it's a valid optax optimizer
        assert isinstance(optimizer, optax.GradientTransformation)

        # Test initialization
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_adam_custom_params(self):
        """Test Adam optimizer with custom parameters."""
        optimizer = create_adam(learning_rate=0.01, b1=0.95, b2=0.99, eps=1e-7)

        # Verify it works
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, new_state = optimizer.update(grads, opt_state, params)
        assert updates is not None
        assert new_state is not None

    def test_create_adamw_default_params(self):
        """Test AdamW optimizer with default parameters."""
        optimizer = create_adamw()

        # Verify it's a valid optax optimizer
        assert isinstance(optimizer, optax.GradientTransformation)

        # Test with parameters
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_adamw_with_weight_decay(self):
        """Test AdamW with custom weight decay."""
        optimizer = create_adamw(learning_rate=0.001, weight_decay=0.01)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates is not None

    def test_create_sgd_default_params(self):
        """Test SGD optimizer with default parameters."""
        optimizer = create_sgd()

        assert isinstance(optimizer, optax.GradientTransformation)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_sgd_with_momentum(self):
        """Test SGD with momentum."""
        optimizer = create_sgd(learning_rate=0.1, momentum=0.9)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates is not None

    def test_create_rmsprop_default_params(self):
        """Test RMSprop optimizer with default parameters."""
        optimizer = create_rmsprop()

        assert isinstance(optimizer, optax.GradientTransformation)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        assert opt_state is not None

    def test_create_rmsprop_custom_params(self):
        """Test RMSprop with custom parameters."""
        optimizer = create_rmsprop(learning_rate=0.01, eps=1e-7, decay=0.95)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates is not None


class TestScheduleCreation:
    """Test learning rate schedule creation."""

    def test_create_constant_schedule(self):
        """Test constant learning rate schedule."""
        schedule = create_schedule(schedule_type="constant", init_value=0.001)

        assert callable(schedule)
        assert schedule(0) == 0.001
        assert schedule(100) == 0.001
        assert schedule(1000) == 0.001

    def test_create_cosine_schedule(self):
        """Test cosine decay schedule."""
        schedule = create_schedule(
            schedule_type="cosine", init_value=0.01, decay_steps=1000, alpha=0.1
        )

        assert callable(schedule)
        assert schedule(0) == pytest.approx(0.01)
        # At end of schedule, should be close to alpha * init_value
        assert schedule(1000) <= 0.01
        assert schedule(1000) >= 0.001  # alpha * init_value

    def test_create_exponential_schedule(self):
        """Test exponential decay schedule."""
        schedule = create_schedule(
            schedule_type="exponential",
            init_value=0.1,
            transition_steps=100,
            decay_rate=0.96,
        )

        assert callable(schedule)
        assert schedule(0) == 0.1
        # Learning rate should decay
        assert schedule(100) < 0.1
        assert schedule(200) < schedule(100)

    def test_create_linear_schedule(self):
        """Test linear decay schedule."""
        schedule = create_schedule(
            schedule_type="linear",
            init_value=0.01,
            end_value=0.001,
            transition_steps=1000,
        )

        assert callable(schedule)
        assert schedule(0) == 0.01
        assert schedule(1000) == pytest.approx(0.001)
        # Halfway point should be halfway between
        assert schedule(500) == pytest.approx(0.0055)

    def test_create_step_schedule(self):
        """Test step decay schedule."""
        boundaries = [100, 200, 300]
        values = [0.1, 0.01, 0.001, 0.0001]

        schedule = create_schedule(
            schedule_type="step", boundaries_and_values=(boundaries, values)
        )

        assert callable(schedule)
        assert schedule(0) == pytest.approx(0.1)
        assert schedule(99) == pytest.approx(0.1)
        assert schedule(100) == pytest.approx(0.01)
        assert schedule(199) == pytest.approx(0.01)
        assert schedule(200) == pytest.approx(0.001)
        assert schedule(300) == pytest.approx(0.0001)

    def test_create_warmup_cosine_schedule(self):
        """Test warmup with cosine decay schedule."""
        schedule = create_schedule(
            schedule_type="warmup_cosine",
            init_value=0.0,
            peak_value=0.01,
            warmup_steps=100,
            decay_steps=1000,
        )

        assert callable(schedule)
        assert schedule(0) == 0.0
        # After warmup, should be at peak
        assert schedule(100) == pytest.approx(0.01)
        # Should decay after warmup
        assert schedule(500) < 0.01

    def test_invalid_schedule_type_raises_error(self):
        """Test that invalid schedule type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown schedule type"):
            create_schedule(schedule_type="invalid_schedule")


class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_with_gradient_clipping_global_norm(self):
        """Test adding global norm gradient clipping."""
        base_optimizer = optax.adam(0.001)
        clipped_optimizer = with_gradient_clipping(base_optimizer, max_norm=1.0)

        assert isinstance(clipped_optimizer, optax.GradientTransformation)

        # Test with large gradients
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = clipped_optimizer.init(params)
        large_grads = jnp.array([10.0, 20.0, 30.0])  # Norm > 1.0

        updates, _ = clipped_optimizer.update(large_grads, opt_state, params)
        # Updates should be clipped
        # Convert updates to array tree for norm computation
        updates_flat, _ = jax.tree_util.tree_flatten(updates)
        update_norm = jnp.sqrt(
            jnp.sum(jnp.array([jnp.sum(u**2) for u in updates_flat]))
        )
        assert (
            update_norm <= 2.0
        )  # Should be clipped (accounting for optimizer scaling)

    def test_with_gradient_clipping_by_value(self):
        """Test adding value-based gradient clipping."""
        base_optimizer = optax.adam(0.001)
        clipped_optimizer = with_gradient_clipping(
            base_optimizer, clip_type="by_value", max_value=0.5
        )

        assert isinstance(clipped_optimizer, optax.GradientTransformation)

        # Test with large gradients
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = clipped_optimizer.init(params)
        large_grads = jnp.array([10.0, -20.0, 30.0])

        updates, _ = clipped_optimizer.update(large_grads, opt_state, params)
        # Each update element should be clipped in absolute value
        assert jnp.all(jnp.abs(updates) <= 1.0)  # type: ignore  # noqa: PGH003


class TestScheduleIntegration:
    """Test integrating schedules with optimizers."""

    def test_with_schedule_cosine(self):
        """Test applying cosine schedule to optimizer."""
        base_optimizer = optax.adam(1.0)  # Placeholder LR
        schedule = optax.cosine_decay_schedule(0.01, 1000)
        scheduled_optimizer = with_schedule(base_optimizer, schedule)

        assert isinstance(scheduled_optimizer, optax.GradientTransformation)

        # Verify it works through multiple steps
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = scheduled_optimizer.init(params)

        for _ in range(10):
            grads = jnp.array([0.1, 0.2, 0.3])
            updates, opt_state = scheduled_optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    def test_with_schedule_exponential(self):
        """Test applying exponential schedule to optimizer."""
        base_optimizer = optax.sgd(1.0)
        schedule = optax.exponential_decay(0.1, 100, 0.96)
        scheduled_optimizer = with_schedule(base_optimizer, schedule)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = scheduled_optimizer.init(params)

        grads = jnp.array([0.1, 0.2, 0.3])
        updates, opt_state = scheduled_optimizer.update(grads, opt_state, params)
        assert updates is not None


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass."""

    def test_optimizer_config_creation_defaults(self):
        """Test creating OptimizerConfig with default values."""
        config = OptimizerConfig()

        assert config.optimizer_type == "adam"
        assert config.learning_rate == 1e-3
        assert config.schedule_type is None
        assert config.gradient_clip is None

    def test_optimizer_config_creation_custom(self):
        """Test creating OptimizerConfig with custom values."""
        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=0.01,
            weight_decay=0.001,
            schedule_type="cosine",
            decay_steps=5000,
            gradient_clip=1.0,
        )

        assert config.optimizer_type == "adamw"
        assert config.learning_rate == 0.01
        assert config.weight_decay == 0.001
        assert config.schedule_type == "cosine"
        assert config.decay_steps == 5000
        assert config.gradient_clip == 1.0

    def test_optimizer_config_validation(self):
        """Test OptimizerConfig validation."""
        # Invalid optimizer type
        config = OptimizerConfig(optimizer_type="invalid")
        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(config)

    def test_optimizer_config_sgd_with_momentum(self):
        """Test OptimizerConfig for SGD with momentum."""
        config = OptimizerConfig(optimizer_type="sgd", learning_rate=0.1, momentum=0.9)

        assert config.optimizer_type == "sgd"
        assert config.momentum == 0.9


class TestCreateOptimizer:
    """Test the main create_optimizer function."""

    def test_create_optimizer_adam(self):
        """Test creating Adam optimizer from config."""
        config = OptimizerConfig(optimizer_type="adam", learning_rate=0.001)
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

        # Verify it works
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])
        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates is not None

    def test_create_optimizer_adamw(self):
        """Test creating AdamW optimizer from config."""
        config = OptimizerConfig(
            optimizer_type="adamw", learning_rate=0.001, weight_decay=0.01
        )
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_sgd(self):
        """Test creating SGD optimizer from config."""
        config = OptimizerConfig(optimizer_type="sgd", learning_rate=0.1, momentum=0.9)
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_rmsprop(self):
        """Test creating RMSprop optimizer from config."""
        config = OptimizerConfig(optimizer_type="rmsprop", learning_rate=0.01)
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

    def test_create_optimizer_with_schedule(self):
        """Test creating optimizer with learning rate schedule."""
        config = OptimizerConfig(
            optimizer_type="adam",
            learning_rate=0.01,
            schedule_type="cosine",
            decay_steps=1000,
        )
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

        # Run through multiple steps
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)

        for _ in range(50):
            grads = jnp.array([0.1, 0.2, 0.3])
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    def test_create_optimizer_with_gradient_clipping(self):
        """Test creating optimizer with gradient clipping."""
        config = OptimizerConfig(
            optimizer_type="adam", learning_rate=0.001, gradient_clip=1.0
        )
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

        # Test with large gradients
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        large_grads = jnp.array([100.0, 200.0, 300.0])

        updates, _ = optimizer.update(large_grads, opt_state, params)
        assert updates is not None

    def test_create_optimizer_with_schedule_and_clipping(self):
        """Test creating optimizer with both schedule and clipping."""
        config = OptimizerConfig(
            optimizer_type="adam",
            learning_rate=0.01,
            schedule_type="cosine",
            decay_steps=1000,
            gradient_clip=1.0,
        )
        optimizer = create_optimizer(config)

        assert isinstance(optimizer, optax.GradientTransformation)

        # Test with large gradients over multiple steps
        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)

        for _ in range(10):
            large_grads = jnp.array([10.0, 20.0, 30.0])
            updates, opt_state = optimizer.update(large_grads, opt_state, params)
            params = optax.apply_updates(params, updates)

    def test_create_optimizer_invalid_type(self):
        """Test that invalid optimizer type raises ValueError."""
        config = OptimizerConfig(optimizer_type="invalid_optimizer")

        with pytest.raises(ValueError, match="Unknown optimizer type"):
            create_optimizer(config)


class TestJAXCompatibility:
    """Test JAX transformations compatibility."""

    def test_jit_compatible(self):
        """Test that optimizers are JIT compatible."""
        optimizer = create_adam(learning_rate=0.001)

        @jax.jit
        def update_fn(params, grads, opt_state):
            updates, new_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        new_params, new_state = update_fn(params, grads, opt_state)
        assert new_params is not None
        assert new_state is not None

    def test_vmap_compatible(self):
        """Test that optimizer updates can be vmapped."""
        optimizer = create_sgd(learning_rate=0.1)

        def single_update(params, grads, opt_state):
            updates, _ = optimizer.update(grads, opt_state, params)
            return updates

        # Batch of parameters and gradients
        batch_params = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        batch_grads = jnp.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        # Initialize optimizer state for each
        opt_states = jax.vmap(optimizer.init)(batch_params)

        # Vmap the update
        batch_updates = jax.vmap(single_update)(batch_params, batch_grads, opt_states)
        assert batch_updates.shape == batch_params.shape  # type: ignore  # noqa: PGH003

    def test_optimizer_with_schedule_jit(self):
        """Test JIT with scheduled optimizer."""
        config = OptimizerConfig(
            optimizer_type="adam",
            learning_rate=0.01,
            schedule_type="cosine",
            decay_steps=1000,
        )
        optimizer = create_optimizer(config)

        @jax.jit
        def train_step(params, grads, opt_state):
            updates, new_state = optimizer.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return new_params, new_state

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        # Run multiple steps
        for _ in range(10):
            params, opt_state = train_step(params, grads, opt_state)

        assert params is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_learning_rate(self):
        """Test optimizer with zero learning rate."""
        optimizer = create_adam(learning_rate=0.0)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        # Updates should be zero
        assert jnp.allclose(updates, 0.0)  # type: ignore  # noqa: PGH003

    def test_negative_learning_rate(self):
        """Test that negative learning rate works (gradient ascent)."""
        optimizer = create_adam(learning_rate=-0.001)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        # Should work but move in opposite direction
        assert updates is not None

    def test_very_small_eps(self):
        """Test optimizer with very small epsilon."""
        optimizer = create_adam(eps=1e-12)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([0.1, 0.2, 0.3])

        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates is not None

    def test_extreme_gradient_clipping(self):
        """Test with very small clip value."""
        config = OptimizerConfig(
            optimizer_type="adam", learning_rate=0.001, gradient_clip=1e-6
        )
        optimizer = create_optimizer(config)

        params = jnp.array([1.0, 2.0, 3.0])
        opt_state = optimizer.init(params)
        grads = jnp.array([100.0, 200.0, 300.0])

        updates, _ = optimizer.update(grads, opt_state, params)
        # Updates should be heavily clipped
        assert jnp.max(jnp.abs(updates)) < 1.0  # type: ignore  # noqa: PGH003

    def test_empty_schedule_boundaries(self):
        """Test step schedule with minimal boundaries."""
        boundaries = []
        values = [0.1]

        schedule = create_schedule(
            schedule_type="step", boundaries_and_values=(boundaries, values)
        )

        assert callable(schedule)
        assert schedule(0) == 0.1
        assert schedule(1000) == 0.1

    def test_single_parameter_optimization(self):
        """Test optimizer with single parameter."""
        optimizer = create_adam()

        param = jnp.array(1.0)
        opt_state = optimizer.init(param)
        grad = jnp.array(0.1)

        update, _ = optimizer.update(grad, opt_state, param)
        assert update is not None

    def test_high_dimensional_parameters(self):
        """Test optimizer with high-dimensional parameters."""
        optimizer = create_adam()

        params = jnp.ones((100, 100))
        opt_state = optimizer.init(params)
        grads = jnp.ones((100, 100)) * 0.1

        updates, _ = optimizer.update(grads, opt_state, params)
        assert updates.shape == params.shape  # type: ignore  # noqa: PGH003
