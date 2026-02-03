"""Integration test for create_navier_stokes_pinn factory (Sprint 1 E.3)."""

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.neural.pinns.multi_scale import create_navier_stokes_pinn


class TestNavierStokesFactory:
    """Validate that the NS PINN factory produces a viable model."""

    def test_factory_creates_model(self) -> None:
        """Factory returns a MultiScalePINN without error."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(spatial_dim=2, rngs=rngs)
        assert model is not None

    def test_forward_pass_shapes(self) -> None:
        """Forward pass produces correct output shape for 2D NS."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(spatial_dim=2, rngs=rngs)
        # Input: (batch, x, y, t) = (batch, 3)
        x = jax.random.normal(jax.random.PRNGKey(42), (16, 3))
        y = model(x)
        # Output: (batch, u, v, p) = (batch, 3)
        assert y.shape == (16, 3)

    def test_forward_pass_3d(self) -> None:
        """Forward pass works for 3D NS."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(spatial_dim=3, rngs=rngs)
        # Input: (batch, x, y, z, t) = (batch, 4)
        x = jax.random.normal(jax.random.PRNGKey(42), (8, 4))
        y = model(x)
        # Output: (batch, u, v, w, p) = (batch, 4)
        assert y.shape == (8, 4)

    def test_gradient_flow(self) -> None:
        """Gradients flow through the model without NaN."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(spatial_dim=2, rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(42), (16, 3))
        y_true = jax.random.normal(jax.random.PRNGKey(43), (16, 3))

        def loss_fn(model: nnx.Module) -> jax.Array:
            y_pred = model(x)  # pyright: ignore[reportCallIssue]
            return jnp.mean((y_pred - y_true) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        assert jnp.isfinite(loss)
        for g in jax.tree.leaves(grads):
            assert jnp.all(jnp.isfinite(g)), "NaN gradient detected"

    def test_short_training_loop(self) -> None:
        """Model loss decreases over a short training loop."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(
            spatial_dim=2,
            scales=[1, 2],
            hidden_dims=[32, 16],
            rngs=rngs,
        )
        x = jax.random.normal(jax.random.PRNGKey(42), (32, 3))
        y_true = jnp.zeros((32, 3))  # Simple zero target

        optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

        def loss_fn(model: nnx.Module) -> jax.Array:
            y_pred = model(x)  # pyright: ignore[reportCallIssue]
            return jnp.mean((y_pred - y_true) ** 2)

        initial_loss, _ = nnx.value_and_grad(loss_fn)(model)
        for _ in range(20):
            _, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        final_loss = loss_fn(model)
        assert float(final_loss) < float(initial_loss), (
            f"Loss did not decrease: {float(initial_loss):.6f} -> {float(final_loss):.6f}"
        )

    def test_custom_scales(self) -> None:
        """Factory accepts custom scales."""
        rngs = nnx.Rngs(0)
        model = create_navier_stokes_pinn(spatial_dim=2, scales=[1, 3, 7], rngs=rngs)
        x = jax.random.normal(jax.random.PRNGKey(42), (4, 3))
        y = model(x)
        assert y.shape == (4, 3)
