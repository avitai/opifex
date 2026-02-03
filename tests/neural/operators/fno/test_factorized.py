"""Test Factorized Fourier Layer.

Test suite for FactorizedFourierLayer with Tucker and CP decomposition
for parameter reduction in Fourier Neural Operators.
"""

import jax
import jax.numpy as jnp
from flax import nnx

# Import TestEnvironmentManager from extracted location
from tests.neural.operators.common.test_utils import TestEnvironmentManager


class TestFactorizedFourierLayer:
    """Test FactorizedFourierLayer for tensor factorization capabilities."""

    def setup_method(self):
        """Set up test environment with GPU/CPU compatibility."""
        self.env_manager = TestEnvironmentManager()
        self.platform = self.env_manager.get_current_platform()
        print(f"Running FactorizedFourierLayer tests on {self.platform}")

    def test_factorized_layer_tucker_parameter_reduction(self):
        """Test Tucker factorization parameter reduction capabilities with GPU/CPU compatibility."""
        from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

        rngs = nnx.Rngs(42)

        # Dimensions for Tucker factorization testing
        in_channels = 64
        out_channels = 64
        modes = 32
        factorization_rank = 8  # Low rank for high compression

        # Create Tucker factorized layer
        tucker_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="tucker",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Verify initialization properties
        assert tucker_layer.factorization_type == "tucker"
        assert tucker_layer.factorization_rank == factorization_rank
        assert tucker_layer.in_channels == in_channels
        assert tucker_layer.out_channels == out_channels

        # Test parameter count analysis
        tucker_stats = tucker_layer.get_parameter_count()

        # Verify parameter reduction is significant
        assert tucker_stats["parameter_reduction"] > 0.5  # At least 50% reduction
        assert tucker_stats["compression_ratio"] < 0.5  # Compression ratio < 0.5
        assert tucker_stats["factorized_spectral"] > 0
        assert (
            tucker_stats["full_tensor_equivalent"] > tucker_stats["factorized_spectral"]
        )

        # Test that all expected keys are present
        expected_keys = {
            "factorized_spectral",
            "full_tensor_equivalent",
            "linear_layer",
            "total",
            "compression_ratio",
            "parameter_reduction",
        }
        assert set(tucker_stats.keys()) == expected_keys

        # Verify all statistics are finite and valid
        for key, value in tucker_stats.items():
            assert jnp.isfinite(value), f"{key} should be finite, got {value}"
            if key in ["compression_ratio", "parameter_reduction"]:
                assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"

    def test_factorized_layer_cp_parameter_reduction(self):
        """Test CP factorization parameter reduction capabilities with GPU/CPU compatibility."""
        from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

        rngs = nnx.Rngs(42)

        # Dimensions for CP factorization testing
        in_channels = 64
        out_channels = 64
        modes = 32
        factorization_rank = 16

        # Create CP factorized layer
        cp_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="cp",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Verify initialization properties
        assert cp_layer.factorization_type == "cp"
        assert cp_layer.factorization_rank == factorization_rank
        assert cp_layer.in_channels == in_channels
        assert cp_layer.out_channels == out_channels

        # Test parameter count analysis
        cp_stats = cp_layer.get_parameter_count()

        # Verify parameter reduction is significant
        assert cp_stats["parameter_reduction"] > 0.3  # At least 30% reduction
        assert cp_stats["compression_ratio"] < 0.7  # Compression ratio < 0.7
        assert cp_stats["factorized_spectral"] > 0
        assert cp_stats["full_tensor_equivalent"] > cp_stats["factorized_spectral"]

        # Verify all statistics are finite and valid
        for key, value in cp_stats.items():
            assert jnp.isfinite(value), f"{key} should be finite, got {value}"
            if key in ["compression_ratio", "parameter_reduction"]:
                assert 0 <= value <= 1, f"{key} should be between 0 and 1, got {value}"

    def test_factorized_layer_forward_pass_multi_dimensional(self):
        """Test factorized layer forward pass with 1D, 2D, and 3D inputs with GPU/CPU compatibility."""
        from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

        rngs = nnx.Rngs(42)

        # Dimensions for multi-dimensional testing
        in_channels = 64
        out_channels = 64
        modes = 32
        factorization_rank = 8

        # Create Tucker factorized layer for testing
        tucker_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="tucker",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Test 1D forward pass
        x_1d = jax.random.normal(jax.random.PRNGKey(0), (4, in_channels, 128))
        output_1d = tucker_layer(x_1d)
        assert output_1d.shape == (4, out_channels, 128)
        assert jnp.all(jnp.isfinite(output_1d))

        # Test 2D forward pass
        x_2d = jax.random.normal(jax.random.PRNGKey(1), (2, in_channels, 32, 32))
        output_2d = tucker_layer(x_2d)
        assert output_2d.shape == (2, out_channels, 32, 32)
        assert jnp.all(jnp.isfinite(output_2d))

        # Test 3D forward pass
        x_3d = jax.random.normal(jax.random.PRNGKey(2), (1, in_channels, 16, 16, 16))
        output_3d = tucker_layer(x_3d)
        assert output_3d.shape == (1, out_channels, 16, 16, 16)
        assert jnp.all(jnp.isfinite(output_3d))

    def test_factorized_layer_tucker_vs_cp_comparison(self):
        """Test comparison between Tucker and CP factorization methods with GPU/CPU compatibility."""
        from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

        rngs = nnx.Rngs(42)

        # Dimensions for comparison testing
        in_channels = 32
        out_channels = 32
        modes = 16
        factorization_rank = 8
        batch_size = 2
        spatial_dim = 64

        # Create both factorization types with same parameters
        tucker_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="tucker",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        cp_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="cp",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Compare parameter counts
        tucker_stats = tucker_layer.get_parameter_count()
        cp_stats = cp_layer.get_parameter_count()

        # Both should achieve parameter reduction
        assert tucker_stats["parameter_reduction"] > 0
        assert cp_stats["parameter_reduction"] > 0

        # Test forward pass with same input
        x = jax.random.normal(
            jax.random.PRNGKey(0), (batch_size, in_channels, spatial_dim)
        )

        tucker_output = tucker_layer(x)
        cp_output = cp_layer(x)

        # Both should produce valid outputs with same shape
        expected_shape = (batch_size, out_channels, spatial_dim)
        assert tucker_output.shape == expected_shape
        assert cp_output.shape == expected_shape
        assert jnp.all(jnp.isfinite(tucker_output))
        assert jnp.all(jnp.isfinite(cp_output))

        # Test that outputs are different (different factorizations)
        assert not jnp.allclose(tucker_output, cp_output, atol=1e-4)

        # Both should have reasonable magnitude
        assert jnp.abs(tucker_output).max() < 100
        assert jnp.abs(cp_output).max() < 100

    def test_factorized_layer_initialization_types(self):
        """Test different initialization schemes for factorized layers with GPU/CPU compatibility."""
        from opifex.neural.operators.fno.factorized import FactorizedFourierLayer

        rngs = nnx.Rngs(42)

        # Dimensions for initialization testing
        in_channels = 32
        out_channels = 32
        modes = 16
        factorization_rank = 8

        # Test Tucker initialization
        tucker_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="tucker",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Test CP initialization
        cp_layer = FactorizedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            factorization_type="cp",
            factorization_rank=factorization_rank,
            rngs=rngs,
        )

        # Test that layers have different factorization structures
        assert tucker_layer.factorization_type == "tucker"
        assert cp_layer.factorization_type == "cp"
        assert tucker_layer.factorization_rank == factorization_rank
        assert cp_layer.factorization_rank == factorization_rank

        # Test basic forward pass works for both
        x = jax.random.normal(jax.random.PRNGKey(0), (2, in_channels, 64))

        tucker_output = tucker_layer(x)
        cp_output = cp_layer(x)

        assert tucker_output.shape == (2, out_channels, 64)
        assert cp_output.shape == (2, out_channels, 64)
        assert jnp.all(jnp.isfinite(tucker_output))
        assert jnp.all(jnp.isfinite(cp_output))

        # Test parameter analysis
        tucker_stats = tucker_layer.get_parameter_count()
        cp_stats = cp_layer.get_parameter_count()

        # Both should report valid parameter statistics
        assert tucker_stats["factorized_spectral"] > 0
        assert cp_stats["factorized_spectral"] > 0
        assert tucker_stats["compression_ratio"] > 0
        assert cp_stats["compression_ratio"] > 0
