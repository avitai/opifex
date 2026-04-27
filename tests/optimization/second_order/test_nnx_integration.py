"""Tests for FLAX NNX integration with second-order optimizers.

TDD: These tests define the expected behavior for NNX integration helpers.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.optimization.second_order.config import (
    HybridOptimizerConfig,
    LBFGSConfig,
    SwitchCriterion,
)
from opifex.optimization.second_order.nnx_integration import (
    _coerce_loss_to_parameter_dtype,
    create_nnx_lbfgs_optimizer,
    NNXHybridOptimizer,
    NNXSecondOrderOptimizer,
)


def _all_leaves_finite(tree) -> bool:
    """Return True when every array leaf in a PyTree is finite."""
    return all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


class SimpleModel(nnx.Module):
    """Simple test model for optimization."""

    def __init__(self, in_features: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(in_features, out_features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class MLPModel(nnx.Module):
    """MLP for more complex optimization tests."""

    def __init__(self, in_features: int, hidden: int, out_features: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features, hidden, rngs=rngs)
        self.linear2 = nnx.Linear(hidden, out_features, rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        return self.linear2(x)


class TestCreateNNXLBFGSOptimizer:
    """Test L-BFGS optimizer creation for NNX models."""

    def test_creates_valid_optimizer(self):
        """Should create optimizer that works with NNX models."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = create_nnx_lbfgs_optimizer(model)
        assert optimizer is not None

    def test_creates_with_custom_config(self):
        """Should accept custom L-BFGS config."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        config = LBFGSConfig(memory_size=5)
        optimizer = create_nnx_lbfgs_optimizer(model, config)
        assert optimizer is not None


class TestNNXSecondOrderOptimizer:
    """Test NNXSecondOrderOptimizer class."""

    def test_init_with_model(self):
        """Should initialize with NNX model."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXSecondOrderOptimizer(model)
        assert optimizer.model is model

    def test_init_with_custom_config(self):
        """Should accept custom configuration."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        config = LBFGSConfig(memory_size=20)
        optimizer = NNXSecondOrderOptimizer(model, config)
        assert optimizer.config.memory_size == 20

    def test_step_reduces_loss(self):
        """Single step should reduce loss on simple problem."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXSecondOrderOptimizer(model)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[1.0], [2.0]])

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        initial_loss = loss_fn(model)
        optimizer.step(loss_fn)
        final_loss = loss_fn(model)

        # L-BFGS should make progress
        assert final_loss <= initial_loss

    def test_multiple_steps_converge(self):
        """Multiple steps should converge toward optimum."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXSecondOrderOptimizer(model, LBFGSConfig(memory_size=10))

        # Simple linear regression problem
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[1.0], [2.0], [3.0]])

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        initial_loss = loss_fn(model)

        # Run several steps
        for _ in range(10):
            optimizer.step(loss_fn)

        final_loss = loss_fn(model)
        assert final_loss < initial_loss * 0.5

    def test_updates_model_in_place(self):
        """Should update model parameters in place."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXSecondOrderOptimizer(model)

        # Get initial weight by accessing the linear layer directly
        initial_weight = model.linear.kernel.value.copy()

        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[5.0]])

        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)

        optimizer.step(loss_fn)

        # Get updated weight
        new_weight = model.linear.kernel.value

        # Parameters should have changed
        assert not jnp.allclose(initial_weight, new_weight)

    def test_step_keeps_default_float32_loss_with_x64_inputs(self):
        """x64-enabled inputs should not promote default NNX optimizer losses."""
        with jax.enable_x64(True):
            model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
            optimizer = NNXSecondOrderOptimizer(model)

            x = jnp.array([[1.0, 2.0]])
            y = jnp.array([[5.0]])

            def loss_fn(model):
                return jnp.mean((model(x) - y) ** 2)

            loss = optimizer.step(loss_fn)

        assert loss.dtype == jnp.float32
        assert jnp.isfinite(loss)

    def test_loss_dtype_coercion_is_jittable_and_differentiable(self):
        """Loss dtype coercion should preserve JIT tracing and gradients."""
        with jax.enable_x64(True):
            x = jnp.array(2.0)

            def loss_fn(weight):
                loss = (weight * x - 1.0) ** 2
                return _coerce_loss_to_parameter_dtype(loss, jnp.dtype(jnp.float32))

            value, grad = jax.jit(jax.value_and_grad(loss_fn))(jnp.asarray(0.25, dtype=jnp.float32))

        assert value.dtype == jnp.float32
        assert grad.dtype == jnp.float32
        assert jnp.isfinite(value)
        assert jnp.isfinite(grad)

    def test_nnx_functional_loss_is_jittable_and_differentiable_with_x64_inputs(self):
        """NNX functional loss should remain traceable after dtype coercion."""
        with jax.enable_x64(True):
            model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
            optimizer = NNXSecondOrderOptimizer(model)

            x = jnp.array([[1.0, 2.0]])
            y = jnp.array([[5.0]])

            def loss_fn(model):
                return jnp.mean((model(x) - y) ** 2)

            def functional_loss(params):
                model = nnx.merge(optimizer._graphdef, params)
                return _coerce_loss_to_parameter_dtype(loss_fn(model), optimizer._loss_dtype)

            value, grads = jax.jit(jax.value_and_grad(functional_loss))(optimizer._params)

        grad_leaves = jax.tree_util.tree_leaves(grads)
        assert value.dtype == jnp.float32
        assert {leaf.dtype for leaf in grad_leaves} == {jnp.dtype(jnp.float32)}
        assert jnp.isfinite(value)
        assert _all_leaves_finite(grads)


class TestNNXHybridOptimizer:
    """Test NNXHybridOptimizer class."""

    def test_init_with_model(self):
        """Should initialize with NNX model."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXHybridOptimizer(model)
        assert optimizer.model is model

    def test_init_with_custom_config(self):
        """Should accept custom configuration."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        config = HybridOptimizerConfig(first_order_steps=50)
        optimizer = NNXHybridOptimizer(model, config)
        assert optimizer.config.first_order_steps == 50

    def test_step_reduces_loss(self):
        """Single step should reduce loss."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXHybridOptimizer(model)

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        y = jnp.array([[1.0], [2.0]])

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        initial_loss = loss_fn(model)
        optimizer.step(loss_fn)
        final_loss = loss_fn(model)

        assert final_loss <= initial_loss

    def test_starts_with_adam(self):
        """Should start with Adam optimizer."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        config = HybridOptimizerConfig(first_order_steps=100)
        optimizer = NNXHybridOptimizer(model, config)

        assert optimizer.is_using_lbfgs is False

    def test_switches_to_lbfgs(self):
        """Should switch to L-BFGS after first_order_steps."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        config = HybridOptimizerConfig(
            first_order_steps=5,
            switch_criterion=SwitchCriterion.EPOCH,
        )
        optimizer = NNXHybridOptimizer(model, config)

        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[1.0]])

        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)

        # Run steps until switch
        for _ in range(10):
            optimizer.step(loss_fn)

        assert optimizer.is_using_lbfgs is True

    def test_convergence_on_mlp(self):
        """Should converge on MLP training problem."""
        model = MLPModel(2, 8, 1, rngs=nnx.Rngs(0))
        config = HybridOptimizerConfig(
            first_order_steps=20,
            switch_criterion=SwitchCriterion.EPOCH,
            adam_learning_rate=0.01,
        )
        optimizer = NNXHybridOptimizer(model, config)

        # Simple regression data
        x = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y = jnp.array([[0.0], [1.0], [1.0], [0.0]])  # XOR-like

        def loss_fn(model):
            pred = model(x)
            return jnp.mean((pred - y) ** 2)

        initial_loss = loss_fn(model)

        for _ in range(50):
            optimizer.step(loss_fn)

        final_loss = loss_fn(model)
        assert final_loss < initial_loss

    def test_step_count_tracking(self):
        """Should track step count correctly."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXHybridOptimizer(model)

        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[1.0]])

        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)

        for i in range(5):
            optimizer.step(loss_fn)
            assert optimizer.step_count == i + 1


class TestJITCompatibility:
    """Test JIT compatibility of NNX integration."""

    def test_nnx_optimizer_step_not_jitted(self):
        """NNX optimizer step should work (uses internal JIT)."""
        model = SimpleModel(2, 1, rngs=nnx.Rngs(0))
        optimizer = NNXSecondOrderOptimizer(model)

        x = jnp.array([[1.0, 2.0]])
        y = jnp.array([[1.0]])

        def loss_fn(model):
            return jnp.mean((model(x) - y) ** 2)

        # Should complete without error
        loss = optimizer.step(loss_fn)
        assert jnp.isfinite(loss)
