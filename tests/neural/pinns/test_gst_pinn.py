"""Tests for gST-PINN (Gradient-enhanced Self-Training PINN).

TDD tests based on:
    "Gradient Enhanced Self-Training Physics-Informed Neural Network"
    (2023 International Conference on Machine Learning and Computer Application)

Key ideas:
    - gPINN component: adds ||grad(PDE_residual)||^2 to loss
    - Self-training component: pseudo-labels from high-confidence predictions
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.pinns.gst_pinn import (
    create_gst_pinn,
    GradientEnhancedPINN,
    GSTConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rngs():
    return nnx.Rngs(params=0, dropout=1)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestGSTConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        cfg = GSTConfig()
        assert cfg.gradient_weight > 0
        assert cfg.pseudo_label_threshold > 0

    def test_config_frozen(self):
        cfg = GSTConfig()
        with pytest.raises(AttributeError):
            cfg.gradient_weight = 999.0  # type: ignore[misc]

    def test_custom_config(self):
        cfg = GSTConfig(gradient_weight=0.5, pseudo_label_threshold=0.01)
        assert cfg.gradient_weight == 0.5
        assert cfg.pseudo_label_threshold == 0.01


# ---------------------------------------------------------------------------
# GradientEnhancedPINN
# ---------------------------------------------------------------------------


class TestGradientEnhancedPINN:
    """Tests for the gST-PINN module."""

    def test_init(self, rngs):
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        assert model.input_dim == 2
        assert model.output_dim == 1

    def test_forward_shape(self, rngs):
        """Output shape: (batch, output_dim)."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (8, 2))
        out = model(x)
        assert out.shape == (8, 1)

    def test_output_dtype(self, rngs):
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).dtype == jnp.float32

    def test_output_finite(self, rngs):
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert jnp.all(jnp.isfinite(model(x)))

    def test_gradient_enhanced_loss(self, rngs):
        """gradient_enhanced_loss returns scalar with PDE residual gradient."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )

        # Simple residual function: Laplacian(u) - f
        def pde_residual_fn(model, x):
            """Compute PDE residual at points x."""

            def u_fn(xi):
                return model(xi[None, :])[0, 0]

            laplacian = jax.vmap(lambda xi: jnp.trace(jax.hessian(u_fn)(xi)))(x)
            return laplacian  # (batch,)

        x = jax.random.normal(rngs.params(), (8, 2))
        loss = model.gradient_enhanced_loss(x, pde_residual_fn)
        assert loss.shape == ()  # scalar
        assert jnp.isfinite(loss)
        assert loss >= 0  # loss should be non-negative

    def test_gradient_of_residual_contributes(self, rngs):
        """The gradient-enhanced term should make loss > pure residual MSE."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            config=GSTConfig(gradient_weight=1.0),
            rngs=rngs,
        )

        def pde_residual_fn(model, x):
            def u_fn(xi):
                return model(xi[None, :])[0, 0]

            laplacian = jax.vmap(lambda xi: jnp.trace(jax.hessian(u_fn)(xi)))(x)
            return laplacian

        x = jax.random.normal(rngs.params(), (4, 2))
        residuals = pde_residual_fn(model, x)
        pure_mse = jnp.mean(residuals**2)
        grad_loss = model.gradient_enhanced_loss(x, pde_residual_fn)
        # gradient_enhanced_loss >= pure MSE since it adds gradient norm
        assert grad_loss >= pure_mse - 1e-6

    def test_pseudo_label_generation(self, rngs):
        """generate_pseudo_labels returns labels + mask."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            config=GSTConfig(pseudo_label_threshold=100.0),  # loose
            rngs=rngs,
        )

        def pde_residual_fn(model, x):
            def u_fn(xi):
                return model(xi[None, :])[0, 0]

            return jax.vmap(lambda xi: jnp.trace(jax.hessian(u_fn)(xi)))(x)

        x = jax.random.normal(rngs.params(), (8, 2))
        labels, mask = model.generate_pseudo_labels(x, pde_residual_fn)
        assert labels.shape == (8, 1)
        assert mask.shape == (8,)
        assert mask.dtype == jnp.bool_

    def test_self_training_loss(self, rngs):
        """self_training_loss should be a scalar."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            config=GSTConfig(pseudo_label_threshold=100.0),
            rngs=rngs,
        )

        def pde_residual_fn(model, x):
            def u_fn(xi):
                return model(xi[None, :])[0, 0]

            return jax.vmap(lambda xi: jnp.trace(jax.hessian(u_fn)(xi)))(x)

        x = jax.random.normal(rngs.params(), (4, 2))
        loss = model.self_training_loss(x, pde_residual_fn)
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_jit_compatible(self, rngs):
        model = GradientEnhancedPINN(
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
        # Consistency
        eager_out = model(x)
        assert jnp.allclose(out, eager_out, atol=1e-5)

    def test_jit_grad_combo(self, rngs):
        """JIT + grad should work together."""
        model = GradientEnhancedPINN(
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
        model = GradientEnhancedPINN(
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
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        for bs in [1, 4, 16]:
            x = jax.random.normal(jax.random.PRNGKey(bs), (bs, 2))
            assert model(x).shape == (bs, 1)

    def test_multi_output(self, rngs):
        model = GradientEnhancedPINN(
            input_dim=3,
            output_dim=3,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 3))
        assert model(x).shape == (4, 3)

    def test_different_activations(self, rngs):
        for act in [jnp.tanh, nnx.relu, nnx.gelu]:
            model = GradientEnhancedPINN(
                input_dim=2,
                output_dim=1,
                hidden_dims=[16, 16],
                activation=act,
                rngs=rngs,
            )
            x = jax.random.normal(jax.random.PRNGKey(0), (2, 2))
            assert model(x).shape == (2, 1)

    def test_compute_derivatives(self, rngs):
        """compute_derivatives should return gradient dict."""
        model = GradientEnhancedPINN(
            input_dim=2,
            output_dim=1,
            hidden_dims=[32, 32],
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        derivs = model.compute_derivatives(x, order=1)
        assert "grad" in derivs
        assert derivs["grad"].shape == (4, 2)

    def test_deterministic_consistent(self, rngs):
        model = GradientEnhancedPINN(
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
# Factory function
# ---------------------------------------------------------------------------


class TestCreateGSTPINN:
    def test_factory_creates_model(self, rngs):
        model = create_gst_pinn(
            input_dim=2,
            output_dim=1,
            rngs=rngs,
        )
        assert isinstance(model, GradientEnhancedPINN)

    def test_factory_forward(self, rngs):
        model = create_gst_pinn(
            input_dim=2,
            output_dim=1,
            rngs=rngs,
        )
        x = jax.random.normal(rngs.params(), (4, 2))
        assert model(x).shape == (4, 1)

    def test_factory_custom_config(self, rngs):
        cfg = GSTConfig(gradient_weight=0.5)
        model = create_gst_pinn(
            input_dim=2,
            output_dim=1,
            config=cfg,
            rngs=rngs,
        )
        assert model.config.gradient_weight == 0.5
