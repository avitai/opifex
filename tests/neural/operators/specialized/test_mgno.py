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
        grad_norms = [jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

    def test_mgno_forward_is_jit_and_vmap_compatible(self, rngs, rng_key):
        """Forward pass must trace under jit/vmap (no data-dependent Python branching).

        The pre-fix implementation used ``if jnp.any(jnp.isnan(...))`` control flow,
        which concretises a traced array and raises ``TracerBoolConversionError``
        under ``nnx.jit``/``jax.vmap``. Every operator must support these transforms,
        and the transformed result must match the eager result on valid input.
        """
        mgno = MultipoleGraphNeuralOperator(
            in_features=4,
            out_features=3,
            hidden_features=16,
            num_layers=2,
            max_degree=2,
            rngs=rngs,
        )

        batch_size, num_nodes = 2, 32
        k_feat, k_pos = jax.random.split(rng_key)
        features = jax.random.normal(k_feat, (batch_size, num_nodes, 4))
        positions = jax.random.normal(k_pos, (batch_size, num_nodes, 3))

        eager_out = mgno(features, positions)

        @nnx.jit
        def jitted_forward(model, feats, pos):
            return model(feats, pos)

        # Must not raise TracerBoolConversionError.
        jit_out = jitted_forward(mgno, features, positions)
        assert jit_out.shape == (batch_size, num_nodes, 3)
        assert jnp.all(jnp.isfinite(jit_out))
        # atol=1e-4 tolerates GPU kernel-fusion reassociation between the fused
        # (jit) and unfused (eager) execution paths; tight enough to prove the
        # jitted path computes the same function.
        assert jnp.allclose(eager_out, jit_out, atol=1e-4)

        # vmap over an explicit leading sample axis (per-sample unbatched forward).
        graphdef, state = nnx.split(mgno)

        def forward_single(single_feats, single_pos):
            model = nnx.merge(graphdef, state)
            return model(single_feats[None, ...], single_pos[None, ...])[0]

        vmapped_out = jax.vmap(forward_single)(features, positions)
        assert vmapped_out.shape == (batch_size, num_nodes, 3)
        assert jnp.all(jnp.isfinite(vmapped_out))
        # vmapped single-sample path matches the natively batched eager path.
        assert jnp.allclose(eager_out, vmapped_out, atol=1e-5)

    def test_mgno_does_not_silently_zero_fill_on_failure(self, rngs, rng_key):
        """Failures must propagate, never be swallowed into an all-zero prediction.

        The pre-fix implementation wrapped each layer and the output projection in
        ``try/except ... -> continue`` / ``-> jnp.zeros(...)``, so a genuinely failing
        layer was silently skipped and a failing projection returned a valid-looking
        all-zero output. After the fail-fast fix the error must propagate instead.
        """
        mgno = MultipoleGraphNeuralOperator(
            in_features=4,
            out_features=3,
            hidden_features=16,
            num_layers=2,
            max_degree=2,
            rngs=rngs,
        )

        batch_size, num_nodes = 2, 32
        k_feat, k_pos = jax.random.split(rng_key)
        features = jax.random.normal(k_feat, (batch_size, num_nodes, 4))
        positions = jax.random.normal(k_pos, (batch_size, num_nodes, 3))

        # Sanity: a valid forward pass yields a non-zero, finite prediction (not a
        # zero-fill fallback).
        valid_out = mgno(features, positions)
        assert jnp.all(jnp.isfinite(valid_out))
        assert jnp.any(valid_out != 0.0)

        # Force a layer to raise. With the per-layer try/except removed the error
        # must propagate rather than being swallowed and the layer skipped.
        sentinel = "mgno-layer-deliberate-failure"

        def _raise(*_args, **_kwargs):
            raise RuntimeError(sentinel)

        original_call = type(mgno.mgno_layers[0]).__call__
        try:
            type(mgno.mgno_layers[0]).__call__ = _raise  # type: ignore[method-assign]
            with pytest.raises(RuntimeError, match=sentinel):
                mgno(features, positions)
        finally:
            type(mgno.mgno_layers[0]).__call__ = original_call  # type: ignore[method-assign]
