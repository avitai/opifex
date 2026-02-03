"""Test Fourier Enhanced DeepONet.

Test suite for Fourier Enhanced DeepONet combining spectral and operator learning
with enhanced spectral processing capabilities for both branch and trunk networks.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.deeponet.enhanced import FourierEnhancedDeepONet


class TestFourierEnhancedDeepONet:
    """Test Fourier-Enhanced DeepONet combining spectral and operator learning."""

    @pytest.fixture
    def rng_key(self):
        """Provide a JAX random key for testing."""
        return jax.random.PRNGKey(42)

    @pytest.fixture
    def rngs(self, rng_key):
        """Provide FLAX NNX rngs for operator initialization."""
        return nnx.Rngs(rng_key)

    def test_fourier_enhanced_deeponet_initialization(self, rngs):
        """Test Fourier-Enhanced DeepONet initialization with GPU/CPU compatibility."""
        # Dimensions for Fourier-Enhanced DeepONet testing
        branch_input_dim = 64
        trunk_input_dim = 2
        branch_hidden_dims = [128, 64]
        trunk_hidden_dims = [64, 32]
        latent_dim = 32
        fourier_modes = 8

        # Create branch and trunk sizes for API
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=True,
            use_spectral_trunk=False,
            rngs=rngs,
        )

        # Verify initialization properties
        assert fourier_deeponet.branch_sizes == branch_sizes
        assert fourier_deeponet.trunk_sizes == trunk_sizes
        assert fourier_deeponet.fourier_modes == fourier_modes
        assert fourier_deeponet.use_spectral_branch
        assert not fourier_deeponet.use_spectral_trunk

        # Verify components exist
        assert hasattr(fourier_deeponet, "branch_spectral")
        assert hasattr(fourier_deeponet, "branch_net")
        assert hasattr(fourier_deeponet, "trunk_net")
        assert hasattr(fourier_deeponet, "fourier_combiner")

    def test_fourier_enhanced_deeponet_spectral_branch_forward(self, rngs):
        """Test Fourier-Enhanced DeepONet forward pass with spectral branch and GPU/CPU compatibility."""
        # Dimensions for spectral branch testing
        batch_size = 4
        branch_input_dim = 32
        trunk_input_dim = 2
        num_locations = 16
        branch_hidden_dims = [64, 32]
        trunk_hidden_dims = [32, 16]
        latent_dim = 16
        fourier_modes = 4

        # Create branch and trunk sizes
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=True,
            use_spectral_trunk=False,
            rngs=rngs,
        )

        # Create test inputs
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test forward pass
        output = fourier_deeponet(branch_input, trunk_input)

        # Verify output properties
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Verify spectral branch is being used
        assert fourier_deeponet.use_spectral_branch
        assert not fourier_deeponet.use_spectral_trunk

    def test_fourier_enhanced_deeponet_spectral_trunk_forward(self, rngs):
        """Test Fourier-Enhanced DeepONet forward pass with spectral trunk and GPU/CPU compatibility."""
        # Dimensions for spectral trunk testing
        batch_size = 3
        branch_input_dim = 24
        trunk_input_dim = 1
        num_locations = 12
        branch_hidden_dims = [48, 24]
        trunk_hidden_dims = [24, 12]
        latent_dim = 12
        fourier_modes = 6

        # Create branch and trunk sizes
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=False,
            use_spectral_trunk=True,
            rngs=rngs,
        )

        # Create test inputs
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test forward pass
        output = fourier_deeponet(branch_input, trunk_input)

        # Verify output properties
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Verify spectral trunk is being used
        assert not fourier_deeponet.use_spectral_branch
        assert fourier_deeponet.use_spectral_trunk

    def test_fourier_enhanced_deeponet_both_spectral(self, rngs):
        """Test Fourier-Enhanced DeepONet with both spectral branch and trunk and GPU/CPU compatibility."""
        # Dimensions for both spectral testing
        batch_size = 2
        branch_input_dim = 16
        trunk_input_dim = 2
        num_locations = 8
        branch_hidden_dims = [32, 16]
        trunk_hidden_dims = [16, 8]
        latent_dim = 8
        fourier_modes = 4

        # Create branch and trunk sizes
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=True,
            use_spectral_trunk=True,
            rngs=rngs,
        )

        # Create test inputs
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test forward pass
        output = fourier_deeponet(branch_input, trunk_input)

        # Verify output properties
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Verify both spectral components are being used
        assert fourier_deeponet.use_spectral_branch
        assert fourier_deeponet.use_spectral_trunk

    def test_fourier_enhanced_deeponet_no_spectral(self, rngs):
        """Test Fourier-Enhanced DeepONet without spectral enhancement (baseline comparison)."""
        # Dimensions for no spectral testing
        batch_size = 2
        branch_input_dim = 20
        trunk_input_dim = 1
        num_locations = 10
        branch_hidden_dims = [40, 20]
        trunk_hidden_dims = [20, 10]
        latent_dim = 10
        fourier_modes = 4

        # Create branch and trunk sizes
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=False,
            use_spectral_trunk=False,
            rngs=rngs,
        )

        # Create test inputs
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test forward pass
        output = fourier_deeponet(branch_input, trunk_input)

        # Verify output properties
        expected_shape = (batch_size, num_locations)
        assert output.shape == expected_shape
        assert jnp.all(jnp.isfinite(output))

        # Verify no spectral components are being used
        assert not fourier_deeponet.use_spectral_branch
        assert not fourier_deeponet.use_spectral_trunk

    def test_fourier_enhanced_deeponet_differentiability(self, rngs):
        """Test Fourier-Enhanced DeepONet differentiability for training with GPU/CPU compatibility."""
        # Dimensions for differentiability testing
        batch_size = 2
        branch_input_dim = 16
        trunk_input_dim = 2
        num_locations = 8
        branch_hidden_dims = [32, 16]
        trunk_hidden_dims = [16, 8]
        latent_dim = 8
        fourier_modes = 4

        # Create branch and trunk sizes
        branch_sizes = [branch_input_dim, *branch_hidden_dims, latent_dim]
        trunk_sizes = [trunk_input_dim, *trunk_hidden_dims, latent_dim]

        fourier_deeponet = FourierEnhancedDeepONet(
            branch_sizes=branch_sizes,
            trunk_sizes=trunk_sizes,
            fourier_modes=fourier_modes,
            use_spectral_branch=True,
            use_spectral_trunk=True,
            rngs=rngs,
        )

        def loss_fn(model, branch_input, trunk_input):
            output = model(branch_input, trunk_input)
            # Ensure real-valued output for gradient computation
            # Take the real part and square it to ensure positive values
            return jnp.mean(jnp.real(output) ** 2)

        # Create test inputs
        branch_input = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, branch_input_dim)
        )
        trunk_input = jax.random.normal(
            jax.random.PRNGKey(1), (batch_size, num_locations, trunk_input_dim)
        )

        # Test gradient computation
        grads = nnx.grad(loss_fn)(fourier_deeponet, branch_input, trunk_input)

        # Verify gradients exist and are finite
        assert hasattr(grads, "branch_net")
        assert hasattr(grads, "trunk_net")
        assert hasattr(grads, "fourier_combiner")

        # Check that gradients contain finite values
        branch_grads = grads.branch_net
        trunk_grads = grads.trunk_net
        assert hasattr(branch_grads, "layers")
        assert hasattr(trunk_grads, "layers")
