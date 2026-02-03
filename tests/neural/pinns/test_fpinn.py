"""Tests for fPINN (Fractional Physics-Informed Neural Network).

TDD tests based on:
    Pang et al. "fPINNs: Fractional Physics-Informed Neural Networks"
    (2019, SIAM Journal on Scientific Computing)

Key ideas:
    - Caputo fractional derivative via L1 discretization scheme
    - Operational matrix for efficient fractional derivative computation
    - Applications: anomalous diffusion, fractional PDEs
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.pinns.fpinn import (
    caputo_derivative_l1,
    create_fpinn,
    FPINNConfig,
    FractionalPINN,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# ---------------------------------------------------------------------------
# Caputo derivative
# ---------------------------------------------------------------------------


class TestCaputoDerivative:
    """Test Caputo fractional derivative via L1 scheme."""

    def test_output_shape(self):
        """Output shape should match input."""
        t = jnp.linspace(0, 1, 20)
        f_vals = t**2
        result = caputo_derivative_l1(f_vals, t, alpha=0.5)
        assert result.shape == f_vals.shape

    def test_integer_order_matches_derivative(self):
        """alpha=1.0 should approximate the standard derivative."""
        t = jnp.linspace(0.01, 1, 50)
        f_vals = t**2
        # Caputo d^1/dt^1 (t^2) = 2t
        result = caputo_derivative_l1(f_vals, t, alpha=0.99)
        expected = 2 * t
        # L1 scheme has some error, check rough agreement
        assert jnp.allclose(result[5:], expected[5:], atol=0.5)

    def test_constant_function_zero(self):
        """Caputo derivative of a constant should be approximately zero."""
        t = jnp.linspace(0.01, 1, 30)
        f_vals = jnp.ones_like(t) * 3.0
        result = caputo_derivative_l1(f_vals, t, alpha=0.5)
        assert jnp.allclose(result, 0.0, atol=0.1)

    def test_fractional_order(self):
        """0 < alpha < 1 should produce finite non-zero results."""
        t = jnp.linspace(0.01, 1, 30)
        f_vals = t**2
        result = caputo_derivative_l1(f_vals, t, alpha=0.5)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.any(result != 0)

    def test_different_alpha_values(self):
        """Different alpha should produce different results."""
        t = jnp.linspace(0.01, 1, 30)
        f_vals = t**2
        r1 = caputo_derivative_l1(f_vals, t, alpha=0.3)
        r2 = caputo_derivative_l1(f_vals, t, alpha=0.7)
        assert not jnp.allclose(r1, r2)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestFPINNConfig:
    def test_default_config(self):
        cfg = FPINNConfig()
        assert 0 < cfg.alpha <= 2.0
        assert cfg.n_discretization > 0

    def test_config_frozen(self):
        cfg = FPINNConfig()
        with pytest.raises(AttributeError):
            cfg.alpha = 0.9  # type: ignore[misc]

    def test_custom_config(self):
        cfg = FPINNConfig(alpha=0.7, n_discretization=50)
        assert cfg.alpha == 0.7
        assert cfg.n_discretization == 50


# ---------------------------------------------------------------------------
# FractionalPINN
# ---------------------------------------------------------------------------


class TestFractionalPINN:
    """Tests for the fPINN module."""

    def test_init(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        assert model.input_dim == 2
        assert model.output_dim == 1

    def test_forward_shape(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (8, 2))
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_dtype(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).dtype == jnp.float32

    def test_output_finite(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert jnp.all(jnp.isfinite(model(x)))

    def test_fractional_residual(self, rngs):
        """fractional_residual should return scalar loss."""
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            config=FPINNConfig(alpha=0.5, n_discretization=20),
            rngs=rngs,
        )
        # Collocation points: (x, t) in [0,1]^2
        x = jax.random.uniform(rngs.params(), (8, 2))
        t_grid = jnp.linspace(0.01, 1.0, 20)

        loss = model.fractional_residual(x, t_grid)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_jit_compatible(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))

        @nnx.jit
        def forward(m, x):
            return m(x)

        out = forward(model, x)
        assert out.shape == (4, 1)
        eager_out = model(x)
        assert jnp.allclose(out, eager_out, atol=1e-5)

    def test_jit_grad_combo(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))

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

    def test_gradient_flow(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        flat_grads = jax.tree.leaves(nnx.state(grads))
        assert any(jnp.any(g != 0) for g in flat_grads)

    def test_batch_dimension(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        for bs in [1, 4, 16]:
            x = jax.random.normal(jax.random.PRNGKey(bs), (bs, 2))
            assert model(x).shape == (bs, 1)

    def test_multi_output(self, rngs):
        model = FractionalPINN(
            input_dim=3,
            output_dim=2,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 3))
        assert model(x).shape == (4, 2)

    def test_different_activations(self, rngs):
        for act in [jnp.tanh, nnx.relu, nnx.gelu]:
            model = FractionalPINN(
                input_dim=2,
                output_dim=1,
                hidden_dims=[16, 16],
                activation=act,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
            assert model(x).shape == (2, 1)

    def test_different_alpha(self, rngs):
        """Different fractional orders should work."""
        for alpha in [0.3, 0.5, 0.8, 1.5]:
            model = FractionalPINN(
                input_dim=2,
                output_dim=1,
                hidden_dims=[16, 16],
                config=FPINNConfig(alpha=alpha),
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
            assert model(x).shape == (2, 1)

    def test_deterministic_consistent(self, rngs):
        model = FractionalPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        out1 = model(x)
        out2 = model(x)
        assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateFPINN:
    def test_factory_creates_model(self, rngs):
        model = create_fpinn(
            input_dim=2,
            output_dim=1,
            rngs=rngs,
        )
        assert isinstance(model, FractionalPINN)

    def test_factory_forward(self, rngs):
        model = create_fpinn(
            input_dim=2,
            output_dim=1,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).shape == (4, 1)

    def test_factory_custom_config(self, rngs):
        cfg = FPINNConfig(alpha=0.7)
        model = create_fpinn(
            input_dim=2,
            output_dim=1,
            config=cfg,
            rngs=rngs,
        )
        assert model.config.alpha == 0.7
