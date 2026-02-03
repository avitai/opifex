"""Test Multipole Graph Neural Operator (MGNO).

Test suite for MGNO implementation with graph-based processing
and multipole interactions.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.mgno import MultipoleGraphNeuralOperator


class TestMultipoleGraphNeuralOperator:
    """Test suite for Multipole Graph Neural Operator."""

    def setup_method(self):
        """Setup for each test method with GPU/CPU backend detection."""
        self.backend = jax.default_backend()
        print(f"Running MultipoleGraphNeuralOperator tests on {self.backend}")

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_mgno_initialization(self, rngs):
        """Test MGNO initialization with GPU/CPU compatibility."""
        mgno = MultipoleGraphNeuralOperator(
            in_features=8,
            out_features=3,
            hidden_features=32,
            num_layers=2,
            max_degree=2,
            rngs=rngs,
        )

        assert mgno.in_features == 8
        assert mgno.out_features == 3
        assert mgno.hidden_features == 32
        assert mgno.num_layers == 2
        assert hasattr(mgno, "mgno_layers")

    def test_mgno_forward_pass(self, rngs, rng_key):
        """Test MGNO forward pass with graph data."""
        mgno = MultipoleGraphNeuralOperator(
            in_features=4,
            out_features=3,
            hidden_features=32,
            num_layers=2,
            max_degree=2,
            rngs=rngs,
        )

        # Create graph data
        batch_size = 2
        num_nodes = 64
        features = jax.random.normal(rng_key, (batch_size, num_nodes, 4))
        positions = jax.random.normal(rng_key, (batch_size, num_nodes, 3))

        output = mgno(features, positions)

        expected_shape = (batch_size, num_nodes, 3)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

    def test_mgno_multipole_interactions(self, rngs, rng_key):
        """Test MGNO with different node configurations."""
        mgno = MultipoleGraphNeuralOperator(
            in_features=4,
            out_features=3,
            hidden_features=32,
            num_layers=2,
            max_degree=2,
            rngs=rngs,
        )

        # Test with different numbers of nodes
        for num_nodes in [32, 64]:
            features = jax.random.normal(rng_key, (2, num_nodes, 4))
            positions = jax.random.normal(rng_key, (2, num_nodes, 3))

            output = mgno(features, positions)
            expected_shape = (2, num_nodes, 3)
            assert output.shape == expected_shape

            # Check that output is finite
            assert jnp.all(jnp.isfinite(output))

    def test_mgno_different_max_degree(self, rngs, rng_key):
        """Test MGNO with different max_degree values."""
        for max_degree in [1, 2, 3]:
            mgno = MultipoleGraphNeuralOperator(
                in_features=4,
                out_features=2,
                hidden_features=24,
                num_layers=1,
                max_degree=max_degree,
                rngs=rngs,
            )

            features = jax.random.normal(rng_key, (2, 32, 4))
            positions = jax.random.normal(rng_key, (2, 32, 3))

            output = mgno(features, positions)
            expected_shape = (2, 32, 2)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

    def test_mgno_differentiability(self, rngs, rng_key):
        """Test MGNO differentiability with GPU/CPU compatibility."""
        mgno = MultipoleGraphNeuralOperator(
            in_features=4,
            out_features=2,
            hidden_features=16,
            num_layers=1,
            max_degree=2,
            rngs=rngs,
        )

        def loss_fn(model, features, positions):
            return jnp.sum(model(features, positions) ** 2)

        features = jax.random.normal(rng_key, (2, 16, 4))
        positions = jax.random.normal(rng_key, (2, 16, 3))

        grads = nnx.grad(loss_fn)(mgno, features, positions)

        assert grads is not None
        # Check that at least some gradients are non-zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)
