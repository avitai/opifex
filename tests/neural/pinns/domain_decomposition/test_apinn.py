"""Tests for Augmented Physics-Informed Neural Networks (APINNs).

TDD: These tests define the expected behavior for APINNs with learnable gating.
"""

import jax.numpy as jnp
from flax import nnx


class TestAPINNConfig:
    """Test APINN configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINNConfig

        config = APINNConfig()
        assert config.temperature > 0
        assert config.gating_hidden_dims is not None

    def test_custom_temperature(self):
        """Should accept custom temperature."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINNConfig

        config = APINNConfig(temperature=0.5)
        assert config.temperature == 0.5


class TestGatingNetwork:
    """Test gating network for subdomain selection."""

    def test_create_gating_network(self):
        """Should create gating network."""
        from opifex.neural.pinns.domain_decomposition.apinn import GatingNetwork

        gating = GatingNetwork(
            input_dim=2,
            num_subdomains=3,
            hidden_dims=[16, 16],
            rngs=nnx.Rngs(0),
        )

        assert gating is not None

    def test_gating_output_shape(self):
        """Gating network should output weights for each subdomain."""
        from opifex.neural.pinns.domain_decomposition.apinn import GatingNetwork

        gating = GatingNetwork(
            input_dim=2,
            num_subdomains=3,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.5, 0.5], [0.2, 0.8]])
        weights = gating(x)

        assert weights.shape == (2, 3)  # (batch, num_subdomains)

    def test_gating_weights_sum_to_one(self):
        """Gating weights should sum to 1 (softmax)."""
        from opifex.neural.pinns.domain_decomposition.apinn import GatingNetwork

        gating = GatingNetwork(
            input_dim=1,
            num_subdomains=2,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        weights = gating(x)

        # Weights should sum to 1 for each point
        assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-5)


class TestAPINN:
    """Test APINN model."""

    def test_create_apinn(self):
        """Should create APINN with subdomains."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        apinn = APINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        assert apinn is not None
        assert len(apinn.subdomains) == 2
        assert apinn.gating_network is not None

    def test_forward_pass(self):
        """Should compute forward pass with learned gating."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        apinn = APINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        y = apinn(x)

        assert y.shape == (2, 1)
        assert jnp.isfinite(y).all()

    def test_gating_weights_accessible(self):
        """Should be able to access gating weights."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        apinn = APINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        weights = apinn.get_gating_weights(x)

        assert weights.shape == (2, 2)  # (batch, num_subdomains)
        assert jnp.allclose(jnp.sum(weights, axis=-1), 1.0, atol=1e-5)


class TestAPINNTemperature:
    """Test temperature-controlled gating."""

    def test_high_temperature_uniform(self):
        """High temperature should give more uniform weights."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN, APINNConfig
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        # High temperature
        config = APINNConfig(temperature=10.0)
        apinn = APINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            config=config,
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25]])
        weights = apinn.get_gating_weights(x)

        # With high temperature, weights should be closer to uniform
        assert jnp.abs(weights[0, 0] - weights[0, 1]) < 0.5

    def test_low_temperature_sharp(self):
        """Low temperature should give sharper weights."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN, APINNConfig
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        # Low temperature (sharp)
        config = APINNConfig(temperature=0.1)
        apinn = APINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[8],
            config=config,
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25]])
        weights = apinn.get_gating_weights(x)

        # With low temperature, one weight should dominate
        max_weight = jnp.max(weights)
        assert max_weight > 0.7


class TestAPINNTraining:
    """Test APINN in training context."""

    def test_gradient_computation(self):
        """Should compute gradients including gating network."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        apinn = APINN(
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

        loss, grads = nnx.value_and_grad(loss_fn)(apinn)

        assert jnp.isfinite(loss)
        assert grads is not None

    def test_jit_compatible(self):
        """APINN should be JIT compatible."""
        from opifex.neural.pinns.domain_decomposition.apinn import APINN
        from opifex.neural.pinns.domain_decomposition.base import Interface, Subdomain

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

        apinn = APINN(
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
        y = forward(apinn, x)

        assert jnp.isfinite(y).all()
