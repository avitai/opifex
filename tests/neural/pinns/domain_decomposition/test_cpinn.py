"""Tests for Conservative PINNs (cPINNs).

TDD: These tests define the expected behavior for cPINNs with flux conservation.
"""

import jax.numpy as jnp
from flax import nnx


class TestCPINNConfig:
    """Test cPINN configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINNConfig

        config = CPINNConfig()
        assert config.flux_weight > 0
        assert config.continuity_weight > 0

    def test_custom_weights(self):
        """Should accept custom loss weights."""
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINNConfig

        config = CPINNConfig(flux_weight=2.0, continuity_weight=0.5)
        assert config.flux_weight == 2.0
        assert config.continuity_weight == 0.5


class TestFluxComputation:
    """Test flux computation for conservation."""

    def test_compute_flux_1d(self):
        """Should compute flux as gradient in 1D."""
        from opifex.neural.pinns.domain_decomposition.cpinn import compute_flux

        # Simple linear function u(x) = 2x, flux = du/dx = 2
        def u_fn(x):
            return 2.0 * x

        x = jnp.array([[0.5], [1.0]])
        flux = compute_flux(u_fn, x)

        assert flux.shape == x.shape
        assert jnp.allclose(flux, 2.0, atol=1e-5)

    def test_compute_flux_2d(self):
        """Should compute flux vector in 2D."""
        from opifex.neural.pinns.domain_decomposition.cpinn import compute_flux

        # u(x,y) = x^2 + y, flux = [2x, 1]
        def u_fn(x):
            return x[:, 0:1] ** 2 + x[:, 1:2]

        x = jnp.array([[1.0, 0.0], [2.0, 1.0]])
        flux = compute_flux(u_fn, x)

        assert flux.shape == (2, 2)  # (batch, dim)
        # At x=1, y=0: flux should be [2, 1]
        assert jnp.allclose(flux[0], jnp.array([2.0, 1.0]), atol=1e-5)


class TestConservativeInterface:
    """Test conservative interface conditions."""

    def test_flux_conservation_residual(self):
        """Should compute flux mismatch at interface."""
        from opifex.neural.pinns.domain_decomposition.cpinn import (
            compute_flux_conservation_residual,
        )

        # Two fluxes that should match
        flux_left = jnp.array([[1.0], [2.0]])
        flux_right = jnp.array([[1.0], [2.0]])
        normal = jnp.array([1.0])

        residual = compute_flux_conservation_residual(flux_left, flux_right, normal)
        assert jnp.allclose(residual, 0.0, atol=1e-6)

    def test_flux_conservation_with_mismatch(self):
        """Should detect flux mismatch."""
        from opifex.neural.pinns.domain_decomposition.cpinn import (
            compute_flux_conservation_residual,
        )

        flux_left = jnp.array([[1.0], [2.0]])
        flux_right = jnp.array([[2.0], [3.0]])  # Different
        normal = jnp.array([1.0])

        residual = compute_flux_conservation_residual(flux_left, flux_right, normal)
        assert residual > 0


class TestCPINN:
    """Test cPINN model."""

    def test_create_cpinn(self):
        """Should create cPINN with subdomains."""
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINN

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

        cpinn = CPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16, 16],
            rngs=nnx.Rngs(0),
        )

        assert cpinn is not None
        assert len(cpinn.subdomains) == 2

    def test_forward_pass(self):
        """Should compute forward pass."""
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINN

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

        cpinn = CPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        y = cpinn(x)

        assert y.shape == (2, 1)
        assert jnp.isfinite(y).all()

    def test_interface_loss(self):
        """Should compute conservative interface loss."""
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINN

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

        cpinn = CPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        loss = cpinn.compute_interface_loss()

        assert jnp.isfinite(loss)
        assert loss >= 0


class TestCPINNTraining:
    """Test cPINN in training context."""

    def test_gradient_computation(self):
        """Should compute gradients for training."""
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINN

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

        cpinn = CPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        y_target = jnp.array([[0.25], [0.75]])

        def loss_fn(model):
            y = model(x)
            return jnp.mean((y - y_target) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(cpinn)

        assert jnp.isfinite(loss)
        assert grads is not None

    def test_jit_compatible(self):
        """cPINN should be JIT compatible."""
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain
        from opifex.neural.pinns.domain_decomposition.cpinn import CPINN

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

        cpinn = CPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        @nnx.jit
        def forward(model, x):
            return model(x)

        x = jnp.array([[0.25], [0.75]])
        y = forward(cpinn, x)

        assert jnp.isfinite(y).all()
