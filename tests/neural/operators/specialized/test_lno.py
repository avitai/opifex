"""Tests for Laplace Neural Operator (LNO).

TDD tests based on reference implementation from:
    Cao, Q., Goswami, S., & Karniadakis, G. E. (2023)
    GitHub: qianyingcao/Laplace-Neural-Operator
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.lno import (
    create_lno,
    LaplaceLayer,
    LaplaceLayerConfig,
    LaplaceNeuralOperator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# ---------------------------------------------------------------------------
# LaplaceLayerConfig
# ---------------------------------------------------------------------------


class TestLaplaceLayerConfig:
    """Config validation tests."""

    def test_default_config(self):
        cfg = LaplaceLayerConfig(in_channels=4, out_channels=4)
        assert cfg.in_channels == 4
        assert cfg.out_channels == 4
        assert cfg.num_poles == 16

    def test_config_frozen(self):
        cfg = LaplaceLayerConfig(in_channels=4, out_channels=4)
        with pytest.raises(AttributeError):
            cfg.in_channels = 8  # type: ignore[misc]

    def test_config_validation_positive(self):
        with pytest.raises(ValueError, match="in_channels"):
            LaplaceLayerConfig(in_channels=0, out_channels=4)
        with pytest.raises(ValueError, match="out_channels"):
            LaplaceLayerConfig(in_channels=4, out_channels=-1)
        with pytest.raises(ValueError, match="num_poles"):
            LaplaceLayerConfig(in_channels=4, out_channels=4, num_poles=0)


# ---------------------------------------------------------------------------
# LaplaceLayer (core pole-residue layer)
# ---------------------------------------------------------------------------


class TestLaplaceLayer:
    """Tests for the core Laplace layer (PR module in reference)."""

    def test_init(self, rngs):
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        assert layer.in_channels == 4
        assert layer.out_channels == 4
        assert layer.num_poles == 8

    def test_forward_shape(self, rngs):
        """Output shape must match (B, C_out, N)."""
        layer = LaplaceLayer(in_channels=4, out_channels=8, num_poles=16, rngs=rngs)
        x = jax.random.normal(rngs.params(), (2, 4, 64))
        out = layer(x)
        assert out.shape == (2, 8, 64)

    def test_output_dtype(self, rngs):
        """Output must be real-valued float32, not complex."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        x = jax.random.normal(rngs.params(), (2, 4, 128))
        out = layer(x)
        assert out.dtype == jnp.float32

    def test_different_sequence_lengths(self, rngs):
        """Layer should work with any spatial resolution."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        for n in [32, 64, 128, 256]:
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 4, n))
            out = layer(x)
            assert out.shape == (1, 4, n)

    def test_has_complex_weights(self, rngs):
        """Reference uses complex-valued poles and residues."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        # Must have both real and imaginary parts for poles
        assert hasattr(layer, "weights_pole")
        assert hasattr(layer, "weights_residue")

    def test_skip_connection(self, rngs):
        """Must include local linear skip connection (like FNO)."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        assert hasattr(layer, "local_linear")

    def test_channel_mixing(self, rngs):
        """Different in/out channels should work."""
        layer = LaplaceLayer(in_channels=2, out_channels=8, num_poles=4, rngs=rngs)
        x = jax.random.normal(rngs.params(), (1, 2, 64))
        out = layer(x)
        assert out.shape == (1, 8, 64)

    def test_jit_compatible(self, rngs):
        """LaplaceLayer must work under nnx.jit."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        x = jax.random.normal(rngs.params(), (2, 4, 64))

        @nnx.jit
        def forward(m, x):
            return m(x)

        out = forward(layer, x)
        assert out.shape == (2, 4, 64)
        # JIT result should match eager result
        out_eager = layer(x)
        assert jnp.allclose(out, out_eager, atol=1e-5)


# ---------------------------------------------------------------------------
# LaplaceNeuralOperator (full model)
# ---------------------------------------------------------------------------


class TestLaplaceNeuralOperator:
    """Tests for the full LNO model."""

    def test_init(self, rngs):
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=4,
            rngs=rngs,
        )
        assert model.in_channels == 1
        assert model.out_channels == 1
        assert model.num_layers == 2

    def test_forward_shape(self, rngs):
        """Output shape must match (B, C_out, N)."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1, 128))
        out = model(x)
        assert out.shape == (4, 1, 128)

    def test_multi_channel(self, rngs):
        """Multiple input/output channels."""
        model = LaplaceNeuralOperator(
            in_channels=3,
            out_channels=2,
            hidden_channels=16,
            num_layers=2,
            num_poles=8,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 3, 64))
        out = model(x)
        assert out.shape == (2, 2, 64)

    def test_jit_compatible(self, rngs):
        """Must work under jax.jit."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))

        @nnx.jit
        def forward(m, x):
            return m(x)

        out = forward(model, x)
        assert out.shape == (2, 1, 64)

    def test_gradient_flow(self, rngs):
        """Gradients must flow through the full model."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        # Check at least one gradient is non-zero
        flat_grads = jax.tree.leaves(nnx.state(grads))
        has_nonzero = any(jnp.any(g != 0) for g in flat_grads)
        assert has_nonzero

    def test_batch_dimension(self, rngs):
        """Different batch sizes should work."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        for bs in [1, 4, 8]:
            x = jax.random.normal(jax.random.PRNGKey(0), (bs, 1, 64))
            out = model(x)
            assert out.shape[0] == bs

    def test_resolution_invariance(self, rngs):
        """LNO should handle different spatial resolutions."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        for n in [32, 64, 128]:
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, n))
            out = model(x)
            assert out.shape == (1, 1, n)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateLNO:
    """Tests for factory function."""

    def test_factory_creates_model(self, rngs):
        model = create_lno(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=4,
            rngs=rngs,
        )
        assert isinstance(model, LaplaceNeuralOperator)

    def test_factory_forward(self, rngs):
        model = create_lno(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))
        out = model(x)
        assert out.shape == (2, 1, 64)


# ---------------------------------------------------------------------------
# Additional comprehensive tests
# ---------------------------------------------------------------------------


class TestLaplaceLayerAdvanced:
    """Advanced tests for LaplaceLayer to match FNO/Transolver coverage."""

    def test_weight_shapes(self, rngs):
        """Verify weight tensor shapes match (in, out, poles)."""
        layer = LaplaceLayer(in_channels=3, out_channels=5, num_poles=7, rngs=rngs)
        assert layer.weights_pole.shape == (3, 5, 7)
        assert layer.weights_pole_imag.shape == (3, 5, 7)
        assert layer.weights_residue.shape == (3, 5, 7)
        assert layer.weights_residue_imag.shape == (3, 5, 7)

    def test_gradient_flows_to_poles_and_residues(self, rngs):
        """Gradients must reach all four complex weight components."""
        layer = LaplaceLayer(in_channels=2, out_channels=2, num_poles=4, rngs=rngs)
        x = jax.random.normal(rngs.params(), (1, 2, 32))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(layer)
        # All four weight components should have non-zero gradients
        for name in [
            "weights_pole",
            "weights_pole_imag",
            "weights_residue",
            "weights_residue_imag",
        ]:
            g = getattr(grads, name)
            assert jnp.any(g != 0), f"No gradient for {name}"

    def test_skip_connection_contributes(self, rngs):
        """Verify skip connection is non-trivially contributing."""
        layer = LaplaceLayer(in_channels=4, out_channels=4, num_poles=8, rngs=rngs)
        x = jax.random.normal(rngs.params(), (1, 4, 64))

        # Full output
        full_out = layer(x)

        # Skip only (manual)
        skip = layer.local_linear(x.transpose(0, 2, 1)).transpose(0, 2, 1)

        # Skip should be different from full output (spectral part adds)
        assert not jnp.allclose(full_out, skip, atol=1e-6)

    def test_single_pole(self, rngs):
        """Edge case: single pole should still work."""
        layer = LaplaceLayer(in_channels=2, out_channels=2, num_poles=1, rngs=rngs)
        x = jax.random.normal(rngs.params(), (1, 2, 32))
        out = layer(x)
        assert out.shape == (1, 2, 32)
        assert out.dtype == jnp.float32

    def test_large_num_poles(self, rngs):
        """Large number of poles should work without numerical issues."""
        layer = LaplaceLayer(in_channels=2, out_channels=2, num_poles=64, rngs=rngs)
        x = jax.random.normal(rngs.params(), (1, 2, 32))
        out = layer(x)
        assert out.shape == (1, 2, 32)
        assert jnp.all(jnp.isfinite(out))


class TestLaplaceNeuralOperatorAdvanced:
    """Advanced tests for the full LNO model."""

    def test_deterministic_mode_consistent(self, rngs):
        """Deterministic=True should produce identical outputs."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))
        out1 = model(x, deterministic=True)
        out2 = model(x, deterministic=True)
        assert jnp.allclose(out1, out2)

    def test_output_dtype(self, rngs):
        """Full model output must be real-valued float32."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))
        out = model(x)
        assert out.dtype == jnp.float32

    def test_output_finite(self, rngs):
        """Output should not contain NaN or Inf."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=8,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1, 128))
        out = model(x)
        assert jnp.all(jnp.isfinite(out))

    def test_multi_layer_stacking(self, rngs):
        """Multiple Laplace layers should compose correctly."""
        for n_layers in [1, 2, 4]:
            model = LaplaceNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                num_layers=n_layers,
                num_poles=4,
                rngs=rngs,
            )
            assert len(model.laplace_layers) == n_layers
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 32))
            out = model(x)
            assert out.shape == (1, 1, 32)

    def test_different_activations(self, rngs):
        """Model should work with various activation functions."""
        for act in ["gelu", "relu", "tanh", "silu"]:
            model = LaplaceNeuralOperator(
                in_channels=1,
                out_channels=1,
                hidden_channels=8,
                num_layers=1,
                num_poles=4,
                activation=act,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 32))
            out = model(x)
            assert out.shape == (1, 1, 32)

    def test_multi_output_channels(self, rngs):
        """Multiple output channels should work."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=5,
            hidden_channels=16,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))
        out = model(x)
        assert out.shape == (2, 5, 64)

    def test_parameter_count_scales(self, rngs):
        """Wider hidden channels should have more parameters."""

        def count_params(model):
            leaves = jax.tree.leaves(nnx.state(model, nnx.Param))
            return sum(p.size for p in leaves)

        small = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        large = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=32,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        assert count_params(large) > count_params(small)

    def test_single_sample_batch(self, rngs):
        """Single sample batch should work."""
        model = LaplaceNeuralOperator(
            in_channels=2,
            out_channels=3,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (1, 2, 32))
        out = model(x)
        assert out.shape == (1, 3, 32)

    def test_jit_grad_combo(self, rngs):
        """JIT-compiled gradient computation must work."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))

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

    def test_jit_output_consistency(self, rngs):
        """JIT and eager should produce identical outputs."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (2, 1, 64))

        eager_out = model(x, deterministic=True)

        @nnx.jit
        def jitted_forward(m, x):
            return m(x, deterministic=True)

        jit_out = jitted_forward(model, x)
        assert jnp.allclose(eager_out, jit_out, atol=1e-5)

    def test_vmap_compatible(self, rngs):
        """Model should work with jax.vmap over batch dim."""
        model = LaplaceNeuralOperator(
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=1,
            num_poles=4,
            rngs=rngs,
        )

        graphdef, state = nnx.split(model)

        def forward_single(state, x):
            m = nnx.merge(graphdef, state)
            return m(x[None, ...])[0]

        # vmap over multiple samples
        x_batch = jax.random.normal(rngs.params(), (4, 1, 64))
        vmapped = jax.vmap(lambda x: forward_single(state, x))(x_batch)
        assert vmapped.shape == (4, 1, 64)
