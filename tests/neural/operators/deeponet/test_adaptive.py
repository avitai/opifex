"""Test Adaptive DeepONet.

Test suite for Adaptive DeepONet with multi-resolution capabilities,
adaptive weights, and optional residual connections.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from tests.neural.operators.common.test_utils import TestEnvironmentManager

from opifex.neural.operators.deeponet.adaptive import AdaptiveDeepONet


class TestAdaptiveDeepONet:
    """Test Adaptive DeepONet with multi-resolution capabilities."""

    def setup_method(self):
        """Set up test environment with GPU/CPU compatibility."""
        self.env_manager = TestEnvironmentManager()
        self.platform = self.env_manager.get_current_platform()

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_adaptive_deeponet_initialization(self, rngs):
        """Test Adaptive DeepONet initialization with GPU/CPU compatibility."""
        # Dimensions for Adaptive DeepONet testing
        branch_input_dim = 32
        trunk_input_dim = 2
        base_latent_dim = 16
        num_resolution_levels = 3

        adaptive_deeponet = AdaptiveDeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            base_latent_dim=base_latent_dim,
            num_resolution_levels=num_resolution_levels,
            adaptive_latent_scaling=True,
            use_residual_connections=True,
            rngs=rngs,
        )

        # Verify initialization properties
        assert adaptive_deeponet.base_latent_dim == base_latent_dim
        assert adaptive_deeponet.num_resolution_levels == num_resolution_levels
        assert adaptive_deeponet.adaptive_latent_scaling
        assert adaptive_deeponet.use_residual_connections

        # Verify multi-resolution architecture
        assert len(adaptive_deeponet.branch_networks) == num_resolution_levels
        assert len(adaptive_deeponet.trunk_networks) == num_resolution_levels
        assert hasattr(adaptive_deeponet, "weight_predictor")
        assert hasattr(adaptive_deeponet, "residual_networks")

    def test_adaptive_deeponet_specific_resolution_level(self, rngs):
        """Test Adaptive DeepONet with specific resolution level and GPU/CPU compatibility."""
        # Dimensions for specific resolution level testing
        batch_size = 4
        branch_input_dim = 24
        trunk_input_dim = 1
        num_locations = 12
        base_latent_dim = 8
        num_resolution_levels = 3

        adaptive_deeponet = AdaptiveDeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            base_latent_dim=base_latent_dim,
            num_resolution_levels=num_resolution_levels,
            adaptive_latent_scaling=True,
            use_residual_connections=False,
            rngs=rngs,
        )

        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test specific resolution levels
        for level in range(num_resolution_levels):
            output = adaptive_deeponet(
                branch_input, trunk_input, resolution_level=level
            )
            expected_shape = (batch_size, num_locations)
            assert output.shape == expected_shape
            assert jnp.all(jnp.isfinite(output))

        # Verify different resolution levels produce different outputs
        output_level_0 = adaptive_deeponet(
            branch_input, trunk_input, resolution_level=0
        )
        output_level_1 = adaptive_deeponet(
            branch_input, trunk_input, resolution_level=1
        )
        assert not jnp.allclose(output_level_0, output_level_1, atol=1e-6)

    def test_adaptive_deeponet_all_levels_adaptive_weights(self, rngs):
        """Test Adaptive DeepONet using all levels with adaptive weights and GPU/CPU compatibility."""
        # Dimensions for adaptive weights testing
        batch_size = 3
        branch_input_dim = 20
        trunk_input_dim = 2
        num_locations = 10
        base_latent_dim = 12
        num_resolution_levels = 2

        adaptive_deeponet = AdaptiveDeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            base_latent_dim=base_latent_dim,
            num_resolution_levels=num_resolution_levels,
            adaptive_latent_scaling=True,
            use_residual_connections=True,
            rngs=rngs,
        )

        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test with adaptive weights
        output_adaptive = adaptive_deeponet(
            branch_input, trunk_input, adaptive_weights=True
        )
        expected_shape = (batch_size, num_locations)
        assert output_adaptive.shape == expected_shape
        assert jnp.all(jnp.isfinite(output_adaptive))

        # Test with uniform weights
        output_uniform = adaptive_deeponet(
            branch_input, trunk_input, adaptive_weights=False
        )
        assert output_uniform.shape == expected_shape
        assert jnp.all(jnp.isfinite(output_uniform))

        # Outputs should be different due to different weighting
        assert not jnp.allclose(output_adaptive, output_uniform, atol=1e-6)

        # Verify adaptive scaling is enabled
        assert adaptive_deeponet.adaptive_latent_scaling

    def test_adaptive_deeponet_no_residual_connections(self, rngs):
        """Test Adaptive DeepONet without residual connections and GPU/CPU compatibility."""
        # Dimensions for no residual connections testing
        batch_size = 2
        branch_input_dim = 16
        trunk_input_dim = 1
        num_locations = 8
        base_latent_dim = 8
        num_resolution_levels = 2

        adaptive_deeponet = AdaptiveDeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            base_latent_dim=base_latent_dim,
            num_resolution_levels=num_resolution_levels,
            adaptive_latent_scaling=False,  # Disable adaptive scaling
            use_residual_connections=False,
            rngs=rngs,
        )

        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test forward pass
        output = adaptive_deeponet(branch_input, trunk_input)

        # Verify output properties
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Verify residual connections are disabled
        assert not adaptive_deeponet.use_residual_connections
        assert not adaptive_deeponet.adaptive_latent_scaling

    def test_adaptive_deeponet_differentiability(self, rngs):
        """Test Adaptive DeepONet differentiability with GPU/CPU compatibility."""
        # Dimensions for differentiability testing
        branch_input_dim = 12
        trunk_input_dim = 1
        base_latent_dim = 6
        num_resolution_levels = 2
        batch_size = 2
        num_locations = 6

        adaptive_deeponet = AdaptiveDeepONet(
            branch_input_dim=branch_input_dim,
            trunk_input_dim=trunk_input_dim,
            base_latent_dim=base_latent_dim,
            num_resolution_levels=num_resolution_levels,
            adaptive_latent_scaling=True,
            use_residual_connections=True,
            rngs=rngs,
        )

        def loss_fn(model, branch_input, trunk_input):
            output = model(branch_input, trunk_input)
            return jnp.mean(output**2)

        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test gradient computation
        grads = nnx.grad(loss_fn)(adaptive_deeponet, branch_input, trunk_input)

        # Verify gradients exist
        assert grads is not None
        assert hasattr(grads, "weight_predictor")
        assert hasattr(grads, "branch_networks")
        assert hasattr(grads, "trunk_networks")

        # Check that gradients are not all zero
        grad_leaves = jax.tree_util.tree_leaves(grads)
        grad_norms = [
            jnp.linalg.norm(leaf) for leaf in grad_leaves if hasattr(leaf, "shape")
        ]
        assert len(grad_norms) > 0
        assert any(norm > 1e-8 for norm in grad_norms)

        # Test forward pass to ensure model works
        output = adaptive_deeponet(branch_input, trunk_input)
        assert output.shape == (batch_size, num_locations)
        assert jnp.all(jnp.isfinite(output))
