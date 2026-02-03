"""Tests for physics-informed KAN wrappers.

TDD tests written before implementation. Covers PIKANConfig
validation, PIKAN / SincKAN forward passes, JIT, gradient
flow, grid updates, and factory function.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.kan.pikan import (
    create_pikan,
    PIKAN,
    PIKANConfig,
    SincKAN,
)


# ---------------------------------------------------------------------------
# PIKANConfig
# ---------------------------------------------------------------------------


class TestPIKANConfig:
    """Validate PIKANConfig frozen dataclass."""

    def test_default_values(self):
        cfg = PIKANConfig()
        assert cfg.n_layers == 4
        assert cfg.hidden_dim == 32
        assert cfg.kan_type == "dense"
        assert cfg.k == 3
        assert cfg.grid_intervals == 5

    def test_custom_values(self):
        cfg = PIKANConfig(
            n_layers=6,
            hidden_dim=64,
            kan_type="chebyshev",
        )
        assert cfg.n_layers == 6
        assert cfg.hidden_dim == 64
        assert cfg.kan_type == "chebyshev"

    def test_frozen(self):
        cfg = PIKANConfig()
        with pytest.raises(AttributeError):
            cfg.n_layers = 10  # type: ignore[misc]

    def test_invalid_n_layers(self):
        with pytest.raises(ValueError, match="n_layers"):
            PIKANConfig(n_layers=0)

    def test_invalid_kan_type(self):
        with pytest.raises(ValueError, match="kan_type"):
            PIKANConfig(kan_type="invalid_type")


# ---------------------------------------------------------------------------
# PIKAN
# ---------------------------------------------------------------------------


class TestPIKAN:
    """Test Physics-Informed KAN network."""

    @pytest.fixture
    def default_pikan(self):
        cfg = PIKANConfig(
            n_layers=2,
            hidden_dim=16,
            kan_type="dense",
            grid_intervals=3,
        )
        return PIKAN(
            in_dim=2,
            out_dim=1,
            config=cfg,
            rngs=nnx.Rngs(0),
        )

    def test_init(self, default_pikan):
        assert default_pikan is not None

    def test_forward_shape(self, default_pikan):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
        y = default_pikan(x)
        assert y.shape == (8, 1)

    def test_forward_finite(self, default_pikan):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
        y = default_pikan(x)
        assert jnp.all(jnp.isfinite(y))

    def test_deterministic_flag(self, default_pikan):
        """deterministic kwarg accepted without error."""
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))
        y1 = default_pikan(x, deterministic=True)
        y2 = default_pikan(x, deterministic=False)
        assert y1.shape == y2.shape

    def test_different_kan_types(self):
        """PIKAN should accept multiple Artifex KAN variants."""
        for kan_type in ("dense", "efficient", "chebyshev"):
            cfg = PIKANConfig(
                n_layers=2,
                hidden_dim=16,
                kan_type=kan_type,
                grid_intervals=3,
            )
            model = PIKAN(
                in_dim=2,
                out_dim=1,
                config=cfg,
                rngs=nnx.Rngs(0),
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))
            y = model(x)
            assert y.shape == (4, 1), f"Failed for kan_type={kan_type}"

    def test_multi_output(self):
        """Support multiple output dimensions."""
        cfg = PIKANConfig(n_layers=2, hidden_dim=16)
        model = PIKAN(
            in_dim=3,
            out_dim=5,
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
        y = model(x)
        assert y.shape == (4, 5)

    def test_jit_compatible(self, default_pikan):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))

        @nnx.jit
        def fwd(m, x):
            return m(x)

        y = fwd(default_pikan, x)
        assert y.shape == (4, 1)

    def test_gradient_flow(self, default_pikan):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = nnx.grad(loss_fn)(default_pikan, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)

    def test_update_grids(self, default_pikan):
        """Grid refinement should not crash."""
        x = jax.random.normal(jax.random.PRNGKey(0), (16, 2))
        default_pikan.update_grids(x, new_intervals=5)
        y = default_pikan(x)
        assert y.shape == (16, 1)


# ---------------------------------------------------------------------------
# SincKAN
# ---------------------------------------------------------------------------


class TestSincKAN:
    """Test sinc-interpolation KAN for singular PDEs."""

    @pytest.fixture
    def default_sinckan(self):
        cfg = PIKANConfig(
            n_layers=2,
            hidden_dim=16,
            kan_type="sine",
        )
        return SincKAN(
            in_dim=2,
            out_dim=1,
            config=cfg,
            rngs=nnx.Rngs(0),
        )

    def test_init(self, default_sinckan):
        assert default_sinckan is not None

    def test_forward_shape(self, default_sinckan):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
        y = default_sinckan(x)
        assert y.shape == (8, 1)

    def test_finite_output(self, default_sinckan):
        x = jax.random.normal(jax.random.PRNGKey(0), (8, 2))
        y = default_sinckan(x)
        assert jnp.all(jnp.isfinite(y))

    def test_jit_compatible(self, default_sinckan):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))

        @nnx.jit
        def fwd(m, x):
            return m(x)

        y = fwd(default_sinckan, x)
        assert y.shape == (4, 1)

    def test_gradient_flow(self, default_sinckan):
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))

        def loss_fn(model, x):
            return jnp.mean(model(x) ** 2)

        grads = nnx.grad(loss_fn)(default_sinckan, x)
        leaves = jax.tree_util.tree_leaves(grads)
        assert len(leaves) > 0
        assert all(jnp.all(jnp.isfinite(g)) for g in leaves)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreatePikan:
    """Test factory function."""

    def test_creates_pikan(self):
        model = create_pikan(
            in_dim=2,
            out_dim=1,
            rngs=nnx.Rngs(0),
        )
        assert isinstance(model, PIKAN)

    def test_creates_sinckan(self):
        model = create_pikan(
            in_dim=2,
            out_dim=1,
            kan_type="sine",
            rngs=nnx.Rngs(0),
        )
        assert isinstance(model, SincKAN)

    def test_custom_config(self):
        cfg = PIKANConfig(
            n_layers=3,
            hidden_dim=24,
            kan_type="chebyshev",
        )
        model = create_pikan(
            in_dim=2,
            out_dim=1,
            config=cfg,
            rngs=nnx.Rngs(0),
        )
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 2))
        y = model(x)
        assert y.shape == (4, 1)
