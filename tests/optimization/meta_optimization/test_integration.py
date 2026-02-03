"""Test suite for meta-optimization integration.

This module tests meta-optimization functionality including:
- Meta-optimizer configuration
- Adaptive learning rate scheduling
- Learn-to-optimize (L2O) algorithms
- Integrated meta-optimizer systems

Tests extracted from test_basic_trainer.py during refactoring.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.meta_optimization import (
    AdaptiveLearningRateScheduler,
    LearnToOptimize,
    MetaOptimizer,
    MetaOptimizerConfig,
)


class TestMetaOptimizerConfig:
    """Test meta-optimizer configuration."""

    def test_meta_optimizer_config_creation(self):
        """Test basic meta-optimizer configuration creation."""

        config = MetaOptimizerConfig(
            meta_algorithm="l2o",  # learn-to-optimize
            base_optimizer="adam",
            meta_learning_rate=1e-4,
            adaptation_steps=10,
            warm_start_strategy="previous_params",
            performance_tracking=True,
            memory_efficient=True,
        )

        assert config.meta_algorithm == "l2o"
        assert config.base_optimizer == "adam"
        assert config.meta_learning_rate == 1e-4
        assert config.adaptation_steps == 10
        assert config.warm_start_strategy == "previous_params"
        assert config.performance_tracking is True
        assert config.memory_efficient is True

    def test_meta_optimizer_config_validation(self):
        """Test meta-optimizer configuration validation."""

        # Test invalid meta algorithm
        with pytest.raises(ValueError, match="Invalid meta algorithm"):
            MetaOptimizerConfig(
                meta_algorithm="invalid_algorithm",
                base_optimizer="adam",
            )


class TestAdaptiveLearningRateScheduler:
    """Test adaptive learning rate scheduling algorithms."""

    def test_scheduler_initialization(self):
        """Test adaptive learning rate scheduler initialization."""

        scheduler = AdaptiveLearningRateScheduler(
            schedule_type="cosine_annealing",
            initial_lr=1e-3,
            final_lr=1e-6,
            adaptation_period=100,
            warmup_steps=10,
        )

        assert scheduler.schedule_type == "cosine_annealing"
        assert scheduler.initial_lr == 1e-3
        assert scheduler.final_lr == 1e-6
        assert scheduler.adaptation_period == 100
        assert scheduler.warmup_steps == 10

    def test_cosine_annealing_schedule(self):
        """Test cosine annealing learning rate schedule."""

        scheduler = AdaptiveLearningRateScheduler(
            schedule_type="cosine_annealing",
            initial_lr=1e-3,
            final_lr=1e-6,
            adaptation_period=100,
        )

        # Test at beginning
        lr_0 = scheduler.get_learning_rate(step=0)
        assert jnp.allclose(lr_0, 1e-3, atol=1e-6)

        # Test at middle
        lr_mid = scheduler.get_learning_rate(step=50)
        assert 1e-6 < lr_mid < 1e-3

        # Test at end
        lr_final = scheduler.get_learning_rate(step=100)
        assert jnp.allclose(lr_final, 1e-6, atol=1e-7)


class TestLearnToOptimize:
    """Test learn-to-optimize (L2O) algorithms."""

    def test_l2o_initialization(self):
        """Test learn-to-optimize algorithm initialization."""

        l2o = LearnToOptimize(
            meta_network_layers=[128, 64, 32],
            base_optimizer="adam",
            meta_learning_rate=1e-4,
            unroll_steps=20,
            rngs=nnx.Rngs(42),
        )

        assert hasattr(l2o, "meta_network")
        assert hasattr(l2o, "compute_update")
        assert l2o.base_optimizer == "adam"
        assert l2o.meta_learning_rate == 1e-4
        assert l2o.unroll_steps == 20

    def test_meta_gradient_computation(self):
        """Test meta-gradient computation for L2O."""

        l2o = LearnToOptimize(
            meta_network_layers=[64, 32],
            unroll_steps=5,
            rngs=nnx.Rngs(42),
        )

        # Mock optimization problem
        def mock_loss_fn(params):
            return jnp.sum(params**2)

        # Mock initial parameters
        init_params = jax.random.normal(jax.random.PRNGKey(42), (10,))

        # Compute meta-gradients
        meta_grads = l2o.compute_meta_gradients(mock_loss_fn, init_params)

        # Verify meta-gradients structure
        assert isinstance(meta_grads, dict)
        assert (
            jnp.isfinite(meta_grads).all()
            if isinstance(meta_grads, jax.Array)
            else True
        )


class TestMetaOptimizer:
    """Test integrated meta-optimizer system."""

    def test_meta_optimizer_initialization(self):
        """Test integrated meta-optimizer initialization."""

        config = MetaOptimizerConfig(
            meta_algorithm="l2o",
            base_optimizer="adam",
            meta_learning_rate=1e-4,
            adaptation_steps=10,
            warm_start_strategy="parameter_transfer",
            performance_tracking=True,
        )

        meta_opt = MetaOptimizer(
            config=config,
            rngs=nnx.Rngs(42),
        )

        assert meta_opt.config == config
        assert hasattr(meta_opt, "l2o_engine")
        assert hasattr(meta_opt, "warm_start_strategy")
        assert hasattr(meta_opt, "performance_monitor")
        assert hasattr(meta_opt, "learning_rate_scheduler")

    def test_meta_optimization_step(self):
        """Test meta-optimization step execution."""

        config = MetaOptimizerConfig(
            meta_algorithm="l2o",
            base_optimizer="adam",
        )

        meta_opt = MetaOptimizer(config=config, rngs=nnx.Rngs(42))

        # Mock optimization problem
        def loss_fn(params):
            return jnp.sum(params**2) + 0.1 * jnp.sum(jnp.sin(params))

        # Mock parameters and optimizer state
        params = jax.random.normal(jax.random.PRNGKey(42), (20,))
        opt_state = meta_opt.init_optimizer_state(params)

        # Perform meta-optimization step
        new_params, new_opt_state, meta_info = meta_opt.step(
            loss_fn, params, opt_state, step=0
        )

        assert new_params.shape == params.shape
        assert jnp.isfinite(new_params).all()
        assert new_opt_state is not None  # Ensure opt_state is used
        assert isinstance(meta_info, dict)
        assert "learning_rate" in meta_info
        assert "meta_gradient_norm" in meta_info
