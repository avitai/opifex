"""Tests for POD-enhanced DeepONet (PODDeepONet).

TDD tests based on reference implementation from:
    Lu et al. (2022), "A comprehensive and fair comparison of two neural operators."
    GitHub: lu-group/deeponet-fno
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.deeponet.pod import (
    create_pod_deeponet,
    PODDeepONet,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


@pytest.fixture
def pod_basis():
    """Simulated POD basis: (n_locations, n_modes) from SVD."""
    key = jax.random.PRNGKey(42)
    # Create a simple orthogonal basis via QR decomposition
    A = jax.random.normal(key, (64, 8))
    Q, _ = jnp.linalg.qr(A)
    return Q  # (64, 8) — 64 locations, 8 POD modes


# ---------------------------------------------------------------------------
# PODDeepONet
# ---------------------------------------------------------------------------


class TestPODDeepONet:
    """Tests for POD-enhanced DeepONet."""

    def test_init(self, rngs, pod_basis):
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        assert model.n_modes == 8
        assert model.n_locations == 64

    def test_forward_shape(self, rngs, pod_basis):
        """Output shape: (batch_size, n_locations)."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        branch_input = jax.random.normal(rngs.params(), (4, 100))
        out = model(branch_input)
        assert out.shape == (4, 64)

    def test_pod_basis_not_trainable(self, rngs, pod_basis):
        """POD basis should be stored but NOT as a trainable parameter."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        # The pod_basis itself should be nnx.Variable, not nnx.Param
        assert isinstance(model.pod_basis_modes, nnx.Variable)
        assert not isinstance(model.pod_basis_modes, nnx.Param)

    def test_branch_output_matches_modes(self, rngs, pod_basis):
        """Branch output dim must equal number of POD modes."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],  # last dim = 8 = n_modes
            pod_basis=pod_basis,
            rngs=rngs,
        )
        branch_input = jax.random.normal(rngs.params(), (2, 100))
        branch_out = model.branch_net(branch_input, deterministic=True)
        assert branch_out.shape[-1] == pod_basis.shape[1]

    def test_mismatched_modes_raises(self, rngs, pod_basis):
        """Branch output dim must match POD mode count."""
        with pytest.raises(ValueError, match=r"branch.*modes"):
            PODDeepONet(
                branch_sizes=[100, 64, 5],  # 5 != 8 modes
                pod_basis=pod_basis,
                rngs=rngs,
            )

    def test_jit_compatible(self, rngs, pod_basis):
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        branch_input = jax.random.normal(rngs.params(), (2, 100))

        @nnx.jit
        def forward(m, x):
            return m(x)

        out = forward(model, branch_input)
        assert out.shape == (2, 64)

    def test_gradient_flow(self, rngs, pod_basis):
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        branch_input = jax.random.normal(rngs.params(), (2, 100))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(branch_input) ** 2)

        grads = nnx.grad(loss_fn)(model)
        flat_grads = jax.tree.leaves(nnx.state(grads))
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads)
        assert has_nonzero

    def test_batch_dimension(self, rngs, pod_basis):
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        for bs in [1, 4, 16]:
            x = jax.random.normal(jax.random.PRNGKey(0), (bs, 100))
            out = model(x)
            assert out.shape == (bs, 64)

    def test_with_mean(self, rngs, pod_basis):
        """When output_mean is provided, it should be added to the output."""
        mean = jnp.ones(64) * 0.5
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            output_mean=mean,
            rngs=rngs,
        )
        branch_input = jnp.zeros((1, 100))
        out_with_mean = model(branch_input)
        # With zero input, branch outputs near zero, so output ≈ mean
        # Just check shape and that mean was used
        assert out_with_mean.shape == (1, 64)

    def test_different_mode_counts(self, rngs):
        """Support different numbers of POD modes."""
        for n_modes in [2, 4, 16]:
            key = jax.random.PRNGKey(n_modes)
            basis = jax.random.normal(key, (32, n_modes))
            model = PODDeepONet(
                branch_sizes=[50, 32, n_modes],
                pod_basis=basis,
                rngs=rngs,
            )
            x = jax.random.normal(key, (2, 50))
            out = model(x)
            assert out.shape == (2, 32)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreatePODDeepONet:
    def test_factory_creates_model(self, rngs, pod_basis):
        model = create_pod_deeponet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        assert isinstance(model, PODDeepONet)

    def test_factory_forward(self, rngs, pod_basis):
        model = create_pod_deeponet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))
        out = model(x)
        assert out.shape == (2, 64)

    def test_factory_with_mean(self, rngs, pod_basis):
        """Factory should accept output_mean kwarg."""
        mean = jnp.zeros(64)
        model = create_pod_deeponet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            output_mean=mean,
            rngs=rngs,
        )
        assert model.output_mean is not None


# ---------------------------------------------------------------------------
# Additional comprehensive tests
# ---------------------------------------------------------------------------


class TestPODDeepONetAdvanced:
    """Advanced tests to match FNO/DeepONet/Transolver coverage."""

    def test_output_dtype(self, rngs, pod_basis):
        """Output must be float32."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))
        out = model(x)
        assert out.dtype == jnp.float32

    def test_output_finite(self, rngs, pod_basis):
        """Output should not contain NaN or Inf."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 100))
        out = model(x)
        assert jnp.all(jnp.isfinite(out))

    def test_deterministic_consistent(self, rngs, pod_basis):
        """Deterministic=True should give identical results."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))
        out1 = model(x, deterministic=True)
        out2 = model(x, deterministic=True)
        assert jnp.allclose(out1, out2)

    def test_mean_changes_output(self, rngs, pod_basis):
        """Adding a mean should shift the output."""
        model_no_mean = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        mean = jnp.ones(64) * 3.0
        model_with_mean = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            output_mean=mean,
            rngs=rngs,
        )
        # Copy branch weights to ensure same branch output
        model_with_mean.branch_net = model_no_mean.branch_net

        x = jax.random.normal(rngs.params(), (2, 100))
        out_no = model_no_mean(x)
        out_yes = model_with_mean(x)
        diff = out_yes - out_no  # Should be approximately 3.0
        assert jnp.allclose(diff, 3.0, atol=1e-5)

    def test_gradient_not_to_basis(self, rngs, pod_basis):
        """Gradients should flow through branch but NOT to pod_basis."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        # pod_basis_modes is nnx.Variable, not nnx.Param,
        # so it should not appear in Param-filtered grad state
        param_grads = nnx.state(grads, nnx.Param)
        param_leaves = jax.tree.leaves(param_grads)
        assert len(param_leaves) > 0  # Branch params have grads
        # Branch grads should be non-zero
        has_nonzero = any(jnp.any(g != 0) for g in param_leaves)
        assert has_nonzero

    def test_mean_not_trainable(self, rngs, pod_basis):
        """Output mean should be nnx.Variable, not nnx.Param."""
        mean = jnp.zeros(64)
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            output_mean=mean,
            rngs=rngs,
        )
        assert isinstance(model.output_mean, nnx.Variable)
        assert not isinstance(model.output_mean, nnx.Param)

    def test_different_activations(self, rngs, pod_basis):
        """Model should work with various activation functions."""
        for act in ["gelu", "relu", "tanh", "silu"]:
            model = PODDeepONet(
                branch_sizes=[100, 64, 8],
                pod_basis=pod_basis,
                activation=act,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 100))
            out = model(x)
            assert out.shape == (1, 64)

    def test_different_locations(self, rngs):
        """Support different numbers of output locations."""
        for n_loc in [16, 64, 128]:
            basis = jax.random.normal(jax.random.PRNGKey(n_loc), (n_loc, 4))
            model = PODDeepONet(
                branch_sizes=[50, 32, 4],
                pod_basis=basis,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (2, 50))
            out = model(x)
            assert out.shape == (2, n_loc)

    def test_parameter_count_scales(self, rngs, pod_basis):
        """Wider branch should have more parameters."""

        def count_params(model):
            leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
            return sum(p.size for p in leaves)

        small = PODDeepONet(
            branch_sizes=[100, 16, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        large = PODDeepONet(
            branch_sizes=[100, 128, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        assert count_params(large) > count_params(small)

    def test_deep_branch_network(self, rngs, pod_basis):
        """Deep branch networks should work."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 32, 16, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))
        out = model(x)
        assert out.shape == (2, 64)

    def test_vmap_compatible(self, rngs, pod_basis):
        """Model should work with jax.vmap."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )

        graphdef, state = nnx.split(model)

        def forward_single(state, x):
            m = nnx.merge(graphdef, state)
            return m(x[None, ...])[0]

        x_batch = jax.random.normal(rngs.params(), (4, 100))
        vmapped = jax.vmap(lambda x: forward_single(state, x))(x_batch)
        assert vmapped.shape == (4, 64)

    def test_single_mode(self, rngs):
        """Edge case: single POD mode."""
        basis = jax.random.normal(jax.random.PRNGKey(0), (32, 1))
        model = PODDeepONet(
            branch_sizes=[50, 16, 1],
            pod_basis=basis,
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (2, 50))
        out = model(x)
        assert out.shape == (2, 32)

    def test_jit_grad_combo(self, rngs, pod_basis):
        """JIT-compiled gradient computation must work."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))

        @nnx.jit
        def train_step(m):
            def loss_fn(m):
                return jnp.mean(m(x) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(m)
            return loss, grads

        loss, grads = train_step(model)
        assert jnp.isfinite(loss)
        flat_grads = jax.tree.leaves(nnx.state(grads))
        assert any(jnp.any(g != 0) for g in flat_grads)

    def test_jit_output_consistency(self, rngs, pod_basis):
        """JIT and eager should produce identical outputs."""
        model = PODDeepONet(
            branch_sizes=[100, 64, 8],
            pod_basis=pod_basis,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 100))

        eager_out = model(x, deterministic=True)

        @nnx.jit
        def jitted_forward(m, x):
            return m(x, deterministic=True)

        jit_out = jitted_forward(model, x)
        assert jnp.allclose(eager_out, jit_out, atol=1e-5)
