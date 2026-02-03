"""Tests for VPINN (Variational Physics-Informed Neural Network).

TDD tests based on:
    Kharazmi et al. "hp-VPINNs: Variational Physics-Informed Neural Networks
    With Domain Decomposition" (2021, Computer Methods in Applied Mechanics
    and Engineering)

Key ideas:
    - Weak form PDE: multiply by test functions, integrate by parts
    - Gauss-Legendre quadrature for numerical integration
    - Test functions: Legendre polynomials or similar basis
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.pinns.vpinn import (
    create_vpinn,
    gauss_legendre_quadrature,
    VPINN,
    VPINNConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------


class TestGaussLegendreQuadrature:
    """Test Gauss-Legendre quadrature utility."""

    def test_returns_points_and_weights(self):
        points, weights = gauss_legendre_quadrature(n=5)
        assert points.shape == (5,)
        assert weights.shape == (5,)

    def test_points_in_range(self):
        """Quadrature points should be in [-1, 1]."""
        points, _ = gauss_legendre_quadrature(n=10)
        assert jnp.all(points >= -1.0)
        assert jnp.all(points <= 1.0)

    def test_weights_positive(self):
        _, weights = gauss_legendre_quadrature(n=8)
        assert jnp.all(weights > 0)

    def test_integrates_constant(self):
        """Integral of 1 over [-1,1] should be 2."""
        points, weights = gauss_legendre_quadrature(n=3)
        integral = jnp.sum(weights * jnp.ones_like(points))
        assert jnp.allclose(integral, 2.0, atol=1e-6)

    def test_integrates_polynomial(self):
        """Integral of x^2 over [-1,1] should be 2/3."""
        points, weights = gauss_legendre_quadrature(n=5)
        integral = jnp.sum(weights * points**2)
        assert jnp.allclose(integral, 2.0 / 3.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestVPINNConfig:
    def test_default_config(self):
        cfg = VPINNConfig()
        assert cfg.n_test_functions > 0
        assert cfg.n_quadrature_points > 0

    def test_config_frozen(self):
        cfg = VPINNConfig()
        with pytest.raises(AttributeError):
            cfg.n_test_functions = 999  # type: ignore[misc]

    def test_custom_config(self):
        cfg = VPINNConfig(n_test_functions=10, n_quadrature_points=20)
        assert cfg.n_test_functions == 10
        assert cfg.n_quadrature_points == 20


# ---------------------------------------------------------------------------
# VPINN
# ---------------------------------------------------------------------------


class TestVPINN:
    """Tests for the VPINN module."""

    def test_init(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        assert model.input_dim == 1
        assert model.output_dim == 1

    def test_forward_shape(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (8, 1))
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_dtype(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))
        assert model(x).dtype == jnp.float32

    def test_output_finite(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))
        assert jnp.all(jnp.isfinite(model(x)))

    def test_variational_residual(self, rngs):
        """variational_residual should return per-test-function residuals."""
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            config=VPINNConfig(n_test_functions=5, n_quadrature_points=10),
            rngs=rngs,
        )

        # Simple diffusion operator: -u''(x)
        def pde_lhs_fn(model, x):
            """Return -u''(x) for 1D input."""

            def u_scalar(xi_scalar):
                """Evaluate u at scalar xi."""
                return model(jnp.array([[xi_scalar]]))[0, 0]

            def d2u_at(xi):
                return jax.grad(jax.grad(u_scalar))(xi[0])

            d2u = jax.vmap(d2u_at)(x)
            return -d2u  # (n_quad,)

        domain = (-1.0, 1.0)
        residuals = model.variational_residual(pde_lhs_fn, domain=domain)
        assert residuals.shape == (5,)  # one per test function
        assert jnp.all(jnp.isfinite(residuals))

    def test_variational_loss(self, rngs):
        """variational_loss should return scalar."""
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            config=VPINNConfig(n_test_functions=3, n_quadrature_points=8),
            rngs=rngs,
        )

        def pde_lhs_fn(model, x):
            def u_scalar(xi_scalar):
                return model(jnp.array([[xi_scalar]]))[0, 0]

            def d2u_at(xi):
                return jax.grad(jax.grad(u_scalar))(xi[0])

            d2u = jax.vmap(d2u_at)(x)
            return -d2u

        loss = model.variational_loss(pde_lhs_fn, domain=(-1.0, 1.0))
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_jit_compatible(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))

        @nnx.jit
        def forward(m, x):
            return m(x)

        out = forward(model, x)
        assert out.shape == (4, 1)
        eager_out = model(x)
        assert jnp.allclose(out, eager_out, atol=1e-5)

    def test_jit_grad_combo(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))

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
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))

        @nnx.jit
        def loss_fn(m):
            return jnp.mean(m(x) ** 2)

        grads = nnx.grad(loss_fn)(model)
        flat_grads = jax.tree.leaves(nnx.state(grads))
        assert any(jnp.any(g != 0) for g in flat_grads)

    def test_batch_dimension(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        for bs in [1, 4, 16]:
            x = jax.random.normal(jax.random.PRNGKey(bs), (bs, 1))
            assert model(x).shape == (bs, 1)

    def test_2d_input(self, rngs):
        """VPINN should work with 2D spatial input."""
        model = VPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).shape == (4, 1)

    def test_multi_output(self, rngs):
        model = VPINN(
            input_dim=2,
            output_dim=3,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).shape == (4, 3)

    def test_different_activations(self, rngs):
        for act in [jnp.tanh, nnx.relu, nnx.gelu]:
            model = VPINN(
                input_dim=1,
                output_dim=1,
                hidden_dims=[16, 16],
                activation=act,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (2, 1))
            assert model(x).shape == (2, 1)

    def test_deterministic_consistent(self, rngs):
        model = VPINN(
            input_dim=1,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))
        out1 = model(x)
        out2 = model(x)
        assert jnp.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateVPINN:
    def test_factory_creates_model(self, rngs):
        model = create_vpinn(
            input_dim=1,
            output_dim=1,
            rngs=rngs,
        )
        assert isinstance(model, VPINN)

    def test_factory_forward(self, rngs):
        model = create_vpinn(
            input_dim=1,
            output_dim=1,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 1))
        assert model(x).shape == (4, 1)

    def test_factory_custom_config(self, rngs):
        cfg = VPINNConfig(n_test_functions=8)
        model = create_vpinn(
            input_dim=1,
            output_dim=1,
            config=cfg,
            rngs=rngs,
        )
        assert model.config.n_test_functions == 8
