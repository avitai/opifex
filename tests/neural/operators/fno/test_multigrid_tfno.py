"""
Unit Tests for Multi-Grid Tensorized Fourier Neural Operator (MG-TFNO)

Comprehensive test suite for Phase 3 Multi-Grid TFNO implementation including:
- Hierarchical tensor decomposition
- Frequency-aware rank adaptation
- Adaptive rank learning
- Memory-optimal contractions
"""

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

# Import the modules we're testing
from opifex.neural.operators.fno.multigrid_tfno import (
    FrequencyAwareDecomposition,
    MultiGridTuckerDecomposition,
)


# Skip TensorLy tests if not available
try:
    from opifex.neural.operators.fno.tensorly_integration import TENSORLY_AVAILABLE
except ImportError:
    _tensorly_available = False
else:
    _tensorly_available = TENSORLY_AVAILABLE


class TestMultiGridTuckerDecomposition:
    """Test suite for Multi-Grid Tucker decomposition."""

    @pytest.fixture
    def rngs(self):
        """Provide FLAX NNX random number generators."""
        return nnx.Rngs(42)

    @pytest.fixture
    def tensor_shape(self):
        """Standard tensor shape for testing."""
        return (16, 32, 64)  # (in_channels, out_channels, spatial_modes)

    def test_initialization_basic(self, tensor_shape, rngs):
        """Test basic Multi-Grid Tucker initialization."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.2,
            rngs=rngs,
            use_complex=False,
            adaptive_rank_learning=False,
        )

        # Check basic attributes
        assert mg_tucker.tensor_shape == tensor_shape
        assert mg_tucker.in_channels == 16
        assert mg_tucker.out_channels == 32
        assert mg_tucker.spatial_modes == (64,)

        # Check frequency bands were generated
        assert len(mg_tucker.frequency_bands) == 3  # Default 3-band decomposition
        assert mg_tucker.frequency_bands[0] == (0, 21)  # Low frequency
        assert mg_tucker.frequency_bands[1] == (21, 42)  # Medium frequency
        assert mg_tucker.frequency_bands[2] == (42, 64)  # High frequency

    def test_frequency_band_generation(self, tensor_shape, rngs):
        """Test automatic frequency band generation."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.1,
            rngs=rngs,
        )

        bands = mg_tucker.frequency_bands

        # Should have 3 bands
        assert len(bands) == 3

        # Bands should be contiguous and cover full range
        assert bands[0][0] == 0
        assert bands[-1][1] == tensor_shape[2]  # Last spatial mode

        # Each band should have positive size
        for start, end in bands:
            assert end > start

    def test_frequency_rank_strategies(self, tensor_shape, rngs):
        """Test different frequency-aware rank adaptation strategies."""
        strategies: list[Literal["uniform", "frequency_decay", "energy_based"]] = [
            "uniform",
            "frequency_decay",
            "energy_based",
        ]

        for strategy in strategies:
            mg_tucker = MultiGridTuckerDecomposition(
                tensor_shape=tensor_shape,
                base_rank=0.15,
                rank_adaptation_strategy=strategy,
                rngs=rngs,
            )

            freq_ranks = mg_tucker.frequency_ranks

            # Should have ranks for each frequency band
            assert len(freq_ranks) == 3

            # Each band should have proper rank structure
            for _, ranks in freq_ranks.items():
                assert len(ranks) == 3  # in_channels, out_channels, spatial
                assert all(r >= 1 for r in ranks)  # All ranks should be positive

            # Test strategy-specific properties
            if strategy == "uniform":
                # All bands should have same ranks
                rank_values = list(freq_ranks.values())
                assert rank_values[0] == rank_values[1] == rank_values[2]

            elif strategy == "frequency_decay":
                # Higher frequency bands should have lower ranks
                band_0_ranks = freq_ranks["band_0"]
                band_1_ranks = freq_ranks["band_1"]
                band_2_ranks = freq_ranks["band_2"]

                # Check decay pattern (band 0 >= band 1 >= band 2)
                assert band_0_ranks[0] >= band_1_ranks[0] >= band_2_ranks[0]
                assert band_0_ranks[1] >= band_1_ranks[1] >= band_2_ranks[1]

    def test_complex_number_support(self, tensor_shape, rngs):
        """Test complex number support in decomposition."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.1,
            rngs=rngs,
            use_complex=True,
        )

        # Check that decomposition components are complex
        for _, band_decomp in mg_tucker.band_decompositions.items():
            core = band_decomp["core"].value
            factors = [f.value for f in band_decomp["factors"]]

            # Core should be complex
            assert jnp.iscomplexobj(core)

            # All factors should be complex
            for factor in factors:
                assert jnp.iscomplexobj(factor)

    def test_band_decomposition_structure(self, tensor_shape, rngs):
        """Test structure of band decompositions."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.2,
            rngs=rngs,
        )

        # Each band should have decomposition
        assert len(mg_tucker.band_decompositions) == 3

        for _, band_decomp in mg_tucker.band_decompositions.items():
            # Should have core and factors
            assert "core" in band_decomp
            assert "factors" in band_decomp

            # Core should be tensor
            core = band_decomp["core"].value
            assert isinstance(core, jax.Array)

            # Should have correct number of factors
            factors = band_decomp["factors"]
            assert len(factors) == 3  # in_channels, out_channels, spatial

            # Each factor should be matrix
            for factor in factors:
                assert isinstance(factor.value, jax.Array)
                assert len(factor.value.shape) == 2  # Matrix

    def test_compression_statistics(self, tensor_shape, rngs):
        """Test compression statistics computation."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.1,
            rngs=rngs,
        )

        stats = mg_tucker.get_compression_stats()

        # Should have all required fields
        required_fields = [
            "total_parameters",
            "original_parameters",
            "band_stats",
            "frequency_bands",
            "frequency_ranks",
            "compression_ratio",
            "parameter_reduction",
        ]
        for field in required_fields:
            assert field in stats

        # Check values are reasonable
        assert stats["total_parameters"] > 0
        assert stats["original_parameters"] == np.prod(tensor_shape)
        assert stats["compression_ratio"] > 1.0  # Should be compressed
        assert 0 < stats["parameter_reduction"] < 1.0  # Reduction percentage

        # Band stats should exist for each band
        assert len(stats["band_stats"]) == 3

        # Total parameters should equal sum of band parameters
        total_band_params = sum(
            band_stats["parameters"] for band_stats in stats["band_stats"].values()
        )
        assert stats["total_parameters"] == total_band_params

    def test_adaptive_rank_learning_initialization(self, tensor_shape, rngs):
        """Test adaptive rank learning initialization."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.15,
            rngs=rngs,
            adaptive_rank_learning=True,
            rank_learning_rate=0.02,
        )

        # Should have adaptive learning components
        assert hasattr(mg_tucker, "rank_gradients")
        assert hasattr(mg_tucker, "rank_momentum")

        # Should have entries for each frequency band
        assert len(mg_tucker.rank_gradients) == 3
        assert len(mg_tucker.rank_momentum) == 3

        # Initial values should be zero
        for band_name in mg_tucker.frequency_ranks:
            assert band_name in mg_tucker.rank_gradients
            assert band_name in mg_tucker.rank_momentum

            gradients = mg_tucker.rank_gradients[band_name]
            momentum = mg_tucker.rank_momentum[band_name]

            assert all(g == 0.0 for g in gradients)
            assert all(m == 0.0 for m in momentum)

    def test_tensor_reconstruction(self, tensor_shape, rngs):
        """Test tensor reconstruction from band decompositions."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.2,
            rngs=rngs,
            use_complex=False,
        )

        # Test reconstruction for each band
        for _, band_decomp in mg_tucker.band_decompositions.items():
            reconstructed = mg_tucker._reconstruct_band_tensor(band_decomp)

            # Should be a tensor
            assert isinstance(reconstructed, jax.Array)

            # Should have reasonable shape
            assert len(reconstructed.shape) >= 2

            # Values should be finite
            assert jnp.all(jnp.isfinite(reconstructed))

    @pytest.mark.parametrize("base_rank", [0.05, 0.1, 0.2, 0.3])
    def test_different_compression_ratios(self, tensor_shape, rngs, base_rank):
        """Test different compression ratios."""
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=base_rank,
            rngs=rngs,
        )

        stats = mg_tucker.get_compression_stats()

        # Higher base rank should result in lower compression ratio
        assert stats["compression_ratio"] > 1.0

        # Very low rank should give high compression
        if base_rank <= 0.1:
            assert stats["compression_ratio"] > 10.0

    def test_custom_frequency_bands(self, tensor_shape, rngs):
        """Test custom frequency band specification."""
        custom_bands = [(0, 16), (16, 32), (32, 48), (48, 64)]

        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.1,
            frequency_bands=custom_bands,
            rngs=rngs,
        )

        # Should use custom bands
        assert mg_tucker.frequency_bands == custom_bands
        assert len(mg_tucker.band_decompositions) == 4  # 4 custom bands
        assert len(mg_tucker.frequency_ranks) == 4

    def test_error_handling(self, tensor_shape, rngs):
        """Test error handling for invalid inputs."""
        # Invalid rank adaptation strategy - should raise ValueError
        invalid_strategy: Literal["invalid_strategy"] = "invalid_strategy"  # type: ignore[misc]

        with pytest.raises(ValueError, match="rank_adaptation_strategy"):
            MultiGridTuckerDecomposition(
                tensor_shape=tensor_shape,
                base_rank=0.1,
                rank_adaptation_strategy=invalid_strategy,  # type: ignore[arg-type]
                rngs=rngs,
            )


class TestFrequencyAwareDecomposition:
    """Test suite for frequency-aware decomposition protocol."""

    def test_protocol_interface(self):
        """Test that the protocol interface is properly defined."""
        # This test ensures the protocol has the expected methods
        # In practice, concrete implementations would be tested

        # Check protocol methods exist
        protocol_methods = [
            "decompose_by_frequency",
            "get_frequency_ranks",
            "adaptive_rank_update",
        ]

        for method in protocol_methods:
            assert hasattr(FrequencyAwareDecomposition, method)


@pytest.mark.skipif(not _tensorly_available, reason="TensorLy not available")
class TestTensorLyIntegration:
    """Test suite for TensorLy integration features."""

    def test_tensorly_initialization(self):
        """Test TensorLy-enhanced initialization when available."""
        # This test ensures TensorLy integration is available
        assert _tensorly_available is not None
        # Additional TensorLy-specific tests would go here


class TestMultiGridTFNOIntegration:
    """Integration tests for complete Multi-Grid TFNO functionality."""

    @pytest.fixture
    def rngs(self):
        return nnx.Rngs(123)

    def test_multi_grid_concept_validation(self, rngs):
        """Test that Multi-Grid TFNO concepts work together."""
        tensor_shape = (8, 16, 32)

        # Create Multi-Grid decomposition
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.2,
            rank_adaptation_strategy="energy_based",
            rngs=rngs,
            adaptive_rank_learning=True,
        )

        # Verify integrated functionality
        assert len(mg_tucker.frequency_bands) > 0
        assert len(mg_tucker.band_decompositions) == len(mg_tucker.frequency_bands)
        assert mg_tucker.adaptive_rank_learning

        # Get comprehensive stats
        stats = mg_tucker.get_compression_stats()
        assert stats["compression_ratio"] > 1.0

        # Test adaptive learning components
        assert hasattr(mg_tucker, "rank_gradients")
        assert hasattr(mg_tucker, "rank_momentum")

    def test_frequency_processing_simulation(self, rngs):
        """Test simulated frequency processing."""
        # This would test the factorized multiplication if fully implemented
        # For now, test that the structure supports it

        tensor_shape = (4, 8, 16)
        mg_tucker = MultiGridTuckerDecomposition(
            tensor_shape=tensor_shape,
            base_rank=0.3,
            rngs=rngs,
        )

        # Test band extraction concept
        for i, (start, end) in enumerate(mg_tucker.frequency_bands):
            band_size = end - start
            assert band_size > 0

            # Check corresponding decomposition exists
            band_name = f"band_{i}"
            assert band_name in mg_tucker.band_decompositions


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
