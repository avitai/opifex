"""Tests for XPINN (Extended Physics-Informed Neural Network).

TDD: These tests define the expected behavior for XPINN with interface conditions.
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
from opifex.neural.pinns.domain_decomposition.xpinn import XPINN, XPINNConfig


class TestXPINNConfig:
    """Test XPINN configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        config = XPINNConfig()
        assert config.continuity_weight > 0
        assert config.flux_weight >= 0
        assert config.residual_weight > 0

    def test_custom_weights(self):
        """Should accept custom interface weights."""
        config = XPINNConfig(
            continuity_weight=10.0,
            flux_weight=5.0,
            residual_weight=1.0,
        )
        assert config.continuity_weight == 10.0
        assert config.flux_weight == 5.0


class TestXPINNCreation:
    """Test XPINN model creation."""

    def test_create_xpinn(self):
        """Should create XPINN with subdomains and interfaces."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16, 16],
            rngs=nnx.Rngs(0),
        )

        assert model is not None
        assert len(model.subdomains) == 2

    def test_create_with_custom_config(self):
        """Should accept custom XPINN configuration."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = []
        config = XPINNConfig(continuity_weight=5.0)

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            config=config,
            rngs=nnx.Rngs(0),
        )

        assert model.config.continuity_weight == 5.0


class TestXPINNForward:
    """Test XPINN forward pass."""

    def test_forward_pass(self):
        """Should compute forward pass correctly."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        y = model(x)

        assert y.shape == (2, 1)
        assert jnp.isfinite(y).all()


class TestXPINNLosses:
    """Test XPINN loss computations."""

    def test_continuity_loss(self):
        """Should compute interface continuity loss."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        loss = model.compute_continuity_loss()
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_flux_loss(self):
        """Should compute interface flux continuity loss."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        loss = model.compute_flux_loss()
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0

    def test_total_interface_loss(self):
        """Should compute weighted total interface loss."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]
        config = XPINNConfig(continuity_weight=2.0, flux_weight=1.0)

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            config=config,
            rngs=nnx.Rngs(0),
        )

        total_loss = model.compute_interface_loss()
        continuity_loss = model.compute_continuity_loss()
        flux_loss = model.compute_flux_loss()

        expected = 2.0 * continuity_loss + 1.0 * flux_loss
        assert jnp.allclose(total_loss, expected)


class TestXPINNGradients:
    """Test XPINN gradient computation."""

    def test_gradient_computation(self):
        """Should compute gradients for training."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = XPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        y_target = jnp.array([[1.0], [2.0]])

        def loss_fn(model):
            y = model(x)
            data_loss = jnp.mean((y - y_target) ** 2)
            interface_loss = model.compute_interface_loss()
            return data_loss + interface_loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)

        assert jnp.isfinite(loss)
        # Grads should exist and be finite
        grad_leaves = jax.tree.leaves(grads)
        assert len(grad_leaves) > 0


class TestXPINN2D:
    """Test XPINN in 2D domain."""

    def test_2d_forward_pass(self):
        """Should handle 2D domain decomposition."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5], [0.0, 1.0]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0], [0.0, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]]),
                normal=jnp.array([1.0, 0.0]),
            )
        ]

        model = XPINN(
            input_dim=2,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[32, 16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25, 0.5], [0.75, 0.5]])
        y = model(x)

        assert y.shape == (2, 1)
        assert jnp.isfinite(y).all()
