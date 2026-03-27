"""Tests for the Learn-to-Optimize neural learner."""

import jax.numpy as jnp
from flax import nnx

from opifex.optimization.meta_optimization.neural_learner import LearnToOptimize


class TestLearnToOptimizeInit:
    """Tests for LearnToOptimize initialization."""

    def test_default_init(self):
        """Initializes with default layer sizes."""
        l2o = LearnToOptimize(rngs=nnx.Rngs(0))
        assert l2o.meta_network_layers == [128, 64, 32]
        assert l2o.base_optimizer == "adam"
        assert l2o.quantum_aware is False

    def test_custom_layers(self):
        """Initializes with custom layer sizes."""
        l2o = LearnToOptimize(meta_network_layers=[64, 32], rngs=nnx.Rngs(0))
        assert l2o.meta_network_layers == [64, 32]

    def test_is_nnx_module(self):
        """LearnToOptimize is an nnx.Module."""
        l2o = LearnToOptimize(rngs=nnx.Rngs(0))
        assert isinstance(l2o, nnx.Module)


class TestComputeUpdate:
    """Tests for compute_update."""

    def test_returns_update_array(self):
        """Compute update returns an array."""
        l2o = LearnToOptimize(meta_network_layers=[16, 8], rngs=nnx.Rngs(0))
        gradient = jnp.ones(4)
        prev_updates = jnp.zeros((0, 4))

        update = l2o.compute_update(gradient, prev_updates)
        assert update.ndim >= 1

    def test_with_loss_history(self):
        """Compute update works with loss history provided."""
        l2o = LearnToOptimize(meta_network_layers=[16, 8], rngs=nnx.Rngs(0))
        gradient = jnp.ones(4)
        prev_updates = jnp.zeros((0, 4))
        loss_history = jnp.array([1.0, 0.8, 0.6])

        update = l2o.compute_update(gradient, prev_updates, loss_history)
        assert update.ndim >= 1

    def test_with_previous_updates(self):
        """Compute update incorporates previous update history."""
        l2o = LearnToOptimize(meta_network_layers=[16, 8], rngs=nnx.Rngs(0))
        gradient = jnp.ones(4)
        prev_updates = jnp.ones((3, 4)) * 0.1

        update = l2o.compute_update(gradient, prev_updates)
        assert update.ndim >= 1


class TestAdaptiveStepSize:
    """Tests for adaptive step size mode."""

    def test_adaptive_mode_init(self):
        """Initializes in adaptive step size mode."""
        l2o = LearnToOptimize(
            meta_network_layers=[16, 8],
            adaptive_step_size=True,
            rngs=nnx.Rngs(0),
        )
        assert l2o.adaptive_step_size is True

    def test_adaptive_returns_update(self):
        """Adaptive mode returns a parameter update."""
        l2o = LearnToOptimize(
            meta_network_layers=[16, 8],
            adaptive_step_size=True,
            rngs=nnx.Rngs(0),
        )
        gradient = jnp.ones(4)
        prev_updates = jnp.zeros((0, 4))

        update = l2o.compute_update(gradient, prev_updates)
        assert update.ndim >= 1


class TestQuantumAware:
    """Tests for quantum-aware mode."""

    def test_quantum_mode_init(self):
        """Initializes in quantum-aware mode."""
        l2o = LearnToOptimize(quantum_aware=True, rngs=nnx.Rngs(0))
        assert l2o.quantum_aware is True
        assert l2o.scf_integration is False

    def test_scf_integration_init(self):
        """SCF integration can be enabled."""
        l2o = LearnToOptimize(quantum_aware=True, scf_integration=True, rngs=nnx.Rngs(0))
        assert l2o.scf_integration is True
