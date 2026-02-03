"""Tests for CliffordFNO — Clifford Fourier Neural Operator.

TDD tests written before implementation. Covers config
validation, CliffordFourierBlock, CliffordFNO forward passes
for 2D and 3D, JIT, gradient flow, and factory function.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.clifford.clifford_fno import (
    CliffordFNO,
    CliffordFNOConfig,
    CliffordFourierBlock,
    create_clifford_fno,
)


# ---------------------------------------------------------------------------
# CliffordFNOConfig
# ---------------------------------------------------------------------------


class TestCliffordFNOConfig:
    """Validate CliffordFNOConfig frozen dataclass."""

    def test_default_3d(self):
        cfg = CliffordFNOConfig()
        assert cfg.metric == (1, 1, 1)
        assert cfg.num_layers == 4
        assert cfg.hidden_channels == 20

    def test_custom_2d(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            modes=(12, 12),
            hidden_channels=32,
        )
        assert cfg.metric == (1, 1)
        assert cfg.n_blades == 4  # 2^2

    def test_n_blades_3d(self):
        cfg = CliffordFNOConfig(metric=(1, 1, 1))
        assert cfg.n_blades == 8  # 2^3

    def test_frozen(self):
        cfg = CliffordFNOConfig()
        with pytest.raises(AttributeError):
            cfg.num_layers = 10  # type: ignore[misc]

    def test_invalid_modes_length(self):
        """modes tuple length must match metric dimension."""
        with pytest.raises(ValueError, match="modes"):
            CliffordFNOConfig(
                metric=(1, 1, 1),
                modes=(12, 12),  # 2 modes for 3D
            )

    def test_invalid_hidden_channels(self):
        with pytest.raises(ValueError, match="hidden_channels"):
            CliffordFNOConfig(hidden_channels=0)


# ---------------------------------------------------------------------------
# CliffordFourierBlock
# ---------------------------------------------------------------------------


class TestCliffordFourierBlock:
    """Test single CliffordFNO block."""

    def test_2d_block_shape(self):
        """2D block preserves spatial and blade dims."""
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            hidden_channels=8,
            modes=(4, 4),
        )
        block = CliffordFourierBlock(
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        # (B, H, W, C, n_blades=4)
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 8, 4),
        )
        y = block(x)
        assert y.shape == x.shape

    def test_3d_block_shape(self):
        """3D block preserves spatial and blade dims."""
        cfg = CliffordFNOConfig(
            metric=(1, 1, 1),
            hidden_channels=4,
            modes=(3, 3, 3),
        )
        block = CliffordFourierBlock(
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        # (B, D, H, W, C, n_blades=8)
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 6, 6, 6, 4, 8),
        )
        y = block(x)
        assert y.shape == x.shape

    def test_2d_block_finite(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            hidden_channels=8,
            modes=(4, 4),
        )
        block = CliffordFourierBlock(
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 8, 4),
        )
        y = block(x)
        assert jnp.all(jnp.isfinite(y))


# ---------------------------------------------------------------------------
# CliffordFNO
# ---------------------------------------------------------------------------


class TestCliffordFNO:
    """Test full CliffordFNO operator."""

    def test_2d_forward_shape(self):
        """2D FNO: (B, H, W, C_in) → (B, H, W, C_out)."""
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=3,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        # Input: (B, H, W, C_in)
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 3),
        )
        y = model(x)
        assert y.shape == (2, 8, 8, 1)

    def test_3d_forward_shape(self):
        """3D FNO: (B, D, H, W, C_in) → (B, D, H, W, C_out)."""
        cfg = CliffordFNOConfig(
            metric=(1, 1, 1),
            in_channels=1,
            out_channels=1,
            hidden_channels=4,
            num_layers=2,
            modes=(3, 3, 3),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 6, 6, 6, 1),
        )
        y = model(x)
        assert y.shape == (2, 6, 6, 6, 1)

    def test_2d_finite(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 1),
        )
        y = model(x)
        assert jnp.all(jnp.isfinite(y))

    def test_deterministic_kwarg(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 1),
        )
        y = model(x, deterministic=True)
        assert y.shape == (2, 8, 8, 1)

    def test_jit_compatible(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 1),
        )

        @nnx.jit
        def fwd(m, x):
            return m(x)

        y = fwd(model, x)
        assert y.shape == (2, 8, 8, 1)

    def test_gradient_flow(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=1,
            out_channels=1,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = CliffordFNO(config=cfg, rngs=nnx.Rngs(0))
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 1),
        )

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = nnx.grad(loss_fn)(model, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateCliffordFNO:
    """Test factory function."""

    def test_creates_model(self):
        model = create_clifford_fno(rngs=nnx.Rngs(0))
        assert isinstance(model, CliffordFNO)

    def test_custom_config(self):
        cfg = CliffordFNOConfig(
            metric=(1, 1),
            in_channels=2,
            out_channels=3,
            hidden_channels=8,
            num_layers=2,
            modes=(4, 4),
        )
        model = create_clifford_fno(
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (2, 8, 8, 2),
        )
        y = model(x)
        assert y.shape == (2, 8, 8, 3)
