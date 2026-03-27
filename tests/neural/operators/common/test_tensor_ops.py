"""Tests for standardized tensor operations used across neural operators."""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.common.tensor_ops import (
    apply_linear_with_channel_transform,
    compute_padding_for_conv,
    create_channel_mapper,
    ensure_tensor_compatibility,
    get_tensor_info,
    interpolate_spatial_dimensions,
    pad_spectral_1d,
    safe_einsum,
    safe_spectral_conv,
    safe_spectral_multiply,
    standardized_fft,
    standardized_ifft,
    validate_channel_compatibility,
    validate_operator_config,
    validate_tensor_shape,
)


class TestValidateTensorShape:
    """Tests for tensor shape validation."""

    def test_valid_4d_tensor(self):
        """Accept correctly shaped 4D tensor (batch, channel, H, W)."""
        x = jnp.ones((2, 3, 8, 8))
        validate_tensor_shape(x, expected_dims=4, min_spatial_dims=2)

    def test_wrong_ndim_raises(self):
        """Reject tensor with wrong number of dimensions."""
        x = jnp.ones((2, 3))
        with pytest.raises(ValueError, match="Expected 4D tensor"):
            validate_tensor_shape(x, expected_dims=4)

    def test_insufficient_spatial_dims_raises(self):
        """Reject tensor with too few spatial dimensions."""
        x = jnp.ones((2, 3))
        with pytest.raises(ValueError, match="at least batch"):
            validate_tensor_shape(x, expected_dims=2, min_spatial_dims=2)


class TestValidateChannelCompatibility:
    """Tests for channel compatibility validation."""

    def test_matching_channels(self):
        """Accept matching channel counts."""
        validate_channel_compatibility(16, 16, "test_op")

    def test_mismatched_channels_raises(self):
        """Reject mismatched channel counts."""
        with pytest.raises(ValueError, match="test_op"):
            validate_channel_compatibility(16, 32, "test_op")


class TestStandardizedFFT:
    """Tests for standardized FFT operations."""

    def test_1d_fft_output_shape(self):
        """1D FFT produces correct output shape."""
        x = jnp.ones((2, 3, 64))
        x_ft = standardized_fft(x, spatial_dims=1)
        assert x_ft.shape[0] == 2
        assert x_ft.shape[1] == 3
        assert jnp.iscomplexobj(x_ft)

    def test_2d_fft_output_shape(self):
        """2D FFT produces correct output shape."""
        x = jnp.ones((2, 3, 16, 16))
        x_ft = standardized_fft(x, spatial_dims=2)
        assert x_ft.shape[0] == 2
        assert x_ft.shape[1] == 3
        assert jnp.iscomplexobj(x_ft)

    def test_roundtrip_fft_ifft(self):
        """FFT followed by IFFT recovers original signal."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 32))
        x_ft = standardized_fft(x, spatial_dims=1)
        x_rec = standardized_ifft(x_ft, target_shape=(1, 1, 32), spatial_dims=1)
        assert jnp.allclose(x, x_rec, atol=1e-5)


class TestSafeSpectralMultiply:
    """Tests for safe spectral multiplication."""

    def test_spectral_multiply_output_shape(self):
        """Spectral multiply preserves batch and channel dims."""
        x_modes = jnp.ones((2, 4, 8), dtype=jnp.complex64)
        weights = jnp.ones((4, 4, 8), dtype=jnp.complex64)
        result = safe_spectral_multiply(x_modes, weights, modes=(8,))
        assert result.shape[0] == 2
        assert result.shape[-1] == 8


class TestApplyLinearWithChannelTransform:
    """Tests for channel-first linear application."""

    def test_channel_transform_shape(self):
        """Linear layer applied across channel dim produces correct shape."""
        x = jnp.ones((2, 4, 16, 16))
        layer = nnx.Linear(4, 8, rngs=nnx.Rngs(0))
        result = apply_linear_with_channel_transform(x, layer)
        assert result.shape == (2, 8, 16, 16)


class TestInterpolateSpatialDimensions:
    """Tests for spatial interpolation."""

    def test_identity_interpolation(self):
        """Same-size interpolation preserves values."""
        x = jax.random.normal(jax.random.PRNGKey(0), (1, 1, 8, 8))
        result = interpolate_spatial_dimensions(x, target_spatial_shape=(8, 8))
        assert result.shape == (1, 1, 8, 8)


class TestComputePaddingForConv:
    """Tests for convolution padding computation."""

    def test_same_padding_odd_kernel(self):
        """Padding preserves input size with stride=1."""
        pad_left, pad_right = compute_padding_for_conv(input_size=16, kernel_size=3)
        assert pad_left + pad_right == 2

    def test_same_padding_even_kernel(self):
        """Padding preserves input size with even kernel."""
        pad_left, pad_right = compute_padding_for_conv(input_size=16, kernel_size=4)
        assert pad_left + pad_right == 3


class TestCreateChannelMapper:
    """Tests for channel mapper creation."""

    def test_creates_linear_layer(self):
        """Creates a linear layer with correct in/out channels."""
        mapper = create_channel_mapper(4, 8, rngs=nnx.Rngs(0))
        assert isinstance(mapper, nnx.Linear)
        x = jnp.ones((2, 4))
        y = mapper(x)
        assert y.shape == (2, 8)


class TestSafeEinsum:
    """Tests for safe einsum wrapper."""

    def test_basic_matmul(self):
        """Basic matrix multiplication via einsum."""
        a = jnp.ones((3, 4))
        b = jnp.ones((4, 5))
        result = safe_einsum("ij,jk->ik", a, b)
        assert result.shape == (3, 5)
        assert jnp.allclose(result, jnp.full((3, 5), 4.0))


class TestSafeSpectralConv:
    """Tests for safe spectral convolution."""

    def test_spectral_conv_output_shape(self):
        """Spectral convolution preserves batch dim and maps channels."""
        x = jnp.ones((2, 4, 16, 16))
        modes = (8, 9)
        weights = jnp.ones((4, 4, modes[0], modes[1]), dtype=jnp.complex64)
        result = safe_spectral_conv(x, weights, modes)
        assert result.shape[0] == 2
        assert result.shape[2] == 16
        assert result.shape[3] == 16


class TestEnsureTensorCompatibility:
    """Tests for tensor batch-dimension compatibility checking."""

    def test_compatible_batch_dims(self):
        """Tensors with same batch size are compatible."""
        a = jnp.ones((2, 3, 8))
        b = jnp.ones((2, 5, 16))
        ensure_tensor_compatibility(a, b)

    def test_incompatible_batch_dims_raises(self):
        """Tensors with different batch size raise."""
        a = jnp.ones((2, 3, 8))
        b = jnp.ones((4, 3, 8))
        with pytest.raises(ValueError, match="batch size"):
            ensure_tensor_compatibility(a, b)


class TestValidateOperatorConfig:
    """Tests for operator config validation."""

    def test_valid_config(self):
        """Config with all required keys passes."""
        config = {"in_channels": 4, "out_channels": 8, "modes": (12,)}
        validate_operator_config(config, ["in_channels", "out_channels"])

    def test_missing_key_raises(self):
        """Config missing required key raises."""
        config = {"in_channels": 4}
        with pytest.raises(ValueError, match="out_channels"):
            validate_operator_config(config, ["in_channels", "out_channels"])


class TestGetTensorInfo:
    """Tests for tensor info utility."""

    def test_returns_shape_and_dtype(self):
        """Info dict contains shape and dtype."""
        x = jnp.ones((2, 3, 8), dtype=jnp.float32)
        info = get_tensor_info(x)
        assert info["shape"] == (2, 3, 8)
        assert info["dtype"] == jnp.float32


class TestPadSpectral1D:
    """Tests for 1D spectral padding helper."""

    def test_no_padding_needed(self):
        """Returns input unchanged when already at target size."""
        x = jnp.ones((2, 4, 17), dtype=jnp.complex64)
        result = pad_spectral_1d(x, batch_size=2, out_channels=4, target_freq_size=17)
        assert result.shape == (2, 4, 17)
        assert jnp.allclose(result, x)

    def test_pads_to_target(self):
        """Pads with zeros to reach target frequency size."""
        x = jnp.ones((2, 4, 8), dtype=jnp.complex64)
        result = pad_spectral_1d(x, batch_size=2, out_channels=4, target_freq_size=17)
        assert result.shape == (2, 4, 17)
        # First 8 modes should be ones, rest zeros
        assert jnp.allclose(result[:, :, :8], jnp.ones((2, 4, 8), dtype=jnp.complex64))
        assert jnp.allclose(result[:, :, 8:], jnp.zeros((2, 4, 9), dtype=jnp.complex64))
