"""Tests for the integrated MetaOptimizer system."""

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.training.config import MetaOptimizerConfig
from opifex.optimization.meta_optimization.meta_optimizer import MetaOptimizer


@pytest.fixture
def default_config():
    """Standard MetaOptimizerConfig for testing."""
    return MetaOptimizerConfig(
        meta_algorithm="adaptive_lr",
        base_optimizer="adam",
        meta_learning_rate=1e-3,
        performance_tracking=False,
    )


@pytest.fixture
def l2o_config():
    """L2O-enabled MetaOptimizerConfig."""
    return MetaOptimizerConfig(
        meta_algorithm="l2o",
        base_optimizer="adam",
        meta_learning_rate=1e-3,
        performance_tracking=True,
    )


class TestMetaOptimizerInit:
    """Tests for MetaOptimizer initialization."""

    def test_creates_with_default_config(self, default_config):
        """Initializes without errors using adaptive_lr algorithm."""
        opt = MetaOptimizer(default_config, rngs=nnx.Rngs(0))
        assert opt.current_step == 0
        assert opt.l2o_engine is None
        assert opt.base_optimizer is not None

    def test_creates_with_l2o(self, l2o_config):
        """Initializes L2O engine when algorithm is 'l2o'."""
        opt = MetaOptimizer(l2o_config, rngs=nnx.Rngs(0))
        assert opt.l2o_engine is not None
        assert opt.performance_monitor is not None


class TestMetaOptimizerStep:
    """Tests for optimization step."""

    def test_standard_step_updates_params(self, default_config):
        """Standard step produces updated parameters."""
        opt = MetaOptimizer(default_config, rngs=nnx.Rngs(0))
        params = jnp.ones(4)
        opt_state = opt.init_optimizer_state(params)

        loss_fn = lambda p: jnp.sum(p**2)
        new_params, _new_state, info = opt.step(loss_fn, params, opt_state, step=0)

        assert new_params.shape == params.shape
        assert "loss" in info
        assert "gradient_norm" in info
        assert "learning_rate" in info

    def test_l2o_step_updates_params(self, l2o_config):
        """L2O step produces updated parameters."""
        opt = MetaOptimizer(l2o_config, rngs=nnx.Rngs(0))
        params = jnp.ones(4)
        opt_state = opt.init_optimizer_state(params)

        loss_fn = lambda p: jnp.sum(p**2)
        new_params, _new_state, info = opt.step(loss_fn, params, opt_state, step=0)

        assert new_params.shape == params.shape
        assert "meta_gradient_norm" in info

    def test_step_reduces_loss(self, default_config):
        """Multiple steps reduce loss on a simple quadratic."""
        opt = MetaOptimizer(default_config, rngs=nnx.Rngs(0))
        params = jnp.array([5.0, -3.0, 2.0])
        opt_state = opt.init_optimizer_state(params)

        loss_fn = lambda p: jnp.sum(p**2)
        initial_loss = float(loss_fn(params))

        for step_i in range(10):
            params, opt_state, _ = opt.step(loss_fn, params, opt_state, step=step_i)

        final_loss = float(loss_fn(params))
        assert final_loss < initial_loss


class TestWarmStarting:
    """Tests for warm-start parameter retrieval."""

    def test_store_and_retrieve(self, default_config):
        """Stored optimization results can be retrieved."""
        opt = MetaOptimizer(default_config, rngs=nnx.Rngs(0))
        params = jnp.array([1.0, 2.0, 3.0])
        features = jnp.array([0.5, 0.5])

        opt.store_optimization_result(params, features)
        warm_params = opt.get_warm_start_params(features, target_shape=(3,))

        assert warm_params.shape == (3,)

    def test_no_history_returns_random(self, default_config):
        """Without stored results, returns random initialization."""
        opt = MetaOptimizer(default_config, rngs=nnx.Rngs(0))
        warm_params = opt.get_warm_start_params(
            jnp.array([1.0]),
            target_shape=(4,),
        )
        assert warm_params.shape == (4,)


class TestBaseOptimizer:
    """Tests for base optimizer creation."""

    def test_adam_optimizer(self):
        """Adam optimizer is created by default."""
        config = MetaOptimizerConfig(base_optimizer="adam", meta_learning_rate=1e-3)
        opt = MetaOptimizer(config, rngs=nnx.Rngs(0))
        params = jnp.ones(2)
        state = opt.init_optimizer_state(params)
        assert state is not None

    def test_sgd_optimizer(self):
        """SGD optimizer variant works."""
        config = MetaOptimizerConfig(base_optimizer="sgd", meta_learning_rate=0.01)
        opt = MetaOptimizer(config, rngs=nnx.Rngs(0))
        params = jnp.ones(2)
        state = opt.init_optimizer_state(params)
        assert state is not None
