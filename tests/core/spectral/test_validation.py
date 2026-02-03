"""
Comprehensive tests for opifex.core.spectral.validation module.

This test suite provides comprehensive coverage for spectral operation validation
utilities including input validation, shape checking, and parameter validation.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.spectral.validation import (
    validate_axis_parameter,
    validate_fft_shape,
    validate_grid_spacing,
    validate_spatial_dims,
    validate_spectral_input,
)


class TestValidateSpectralInput:
    """Test spectral input validation."""

    def test_validate_spectral_input_valid_1d(self):
        """Test validation with valid 1D input."""
        x = jnp.ones((32,))
        validate_spectral_input(x, spatial_dims=1)  # Should not raise

    def test_validate_spectral_input_valid_2d(self):
        """Test validation with valid 2D input."""
        x = jnp.ones((32, 32))
        validate_spectral_input(x, spatial_dims=2)  # Should not raise

    def test_validate_spectral_input_valid_3d(self):
        """Test validation with valid 3D input."""
        x = jnp.ones((16, 16, 16))
        validate_spectral_input(x, spatial_dims=3)  # Should not raise

    def test_validate_spectral_input_with_batch_dimensions(self):
        """Test validation with batch and channel dimensions."""
        x = jnp.ones((4, 3, 32, 32))  # batch=4, channels=3, spatial=32x32
        validate_spectral_input(x, spatial_dims=2)  # Should not raise

    def test_validate_spectral_input_float32(self):
        """Test validation with float32 dtype."""
        x = jnp.ones((32,), dtype=jnp.float32)
        validate_spectral_input(x, spatial_dims=1)  # Should not raise

    def test_validate_spectral_input_float64(self):
        """Test validation with float64 dtype."""
        x = jnp.ones((32,), dtype=jnp.float64)
        validate_spectral_input(x, spatial_dims=1)  # Should not raise

    def test_validate_spectral_input_complex64(self):
        """Test validation with complex64 dtype."""
        x = jnp.ones((32,), dtype=jnp.complex64)
        validate_spectral_input(x, spatial_dims=1)  # Should not raise

    def test_validate_spectral_input_complex128(self):
        """Test validation with complex128 dtype."""
        x = jnp.ones((32,), dtype=jnp.complex128)
        validate_spectral_input(x, spatial_dims=1)  # Should not raise

    def test_validate_spectral_input_custom_min_size(self):
        """Test validation with custom minimum spatial size."""
        x = jnp.ones((8,))
        validate_spectral_input(
            x, spatial_dims=1, min_spatial_size=4
        )  # Should not raise

    def test_validate_spectral_input_not_jax_array(self):
        """Test validation with non-JAX array input."""
        x = np.ones((32,))  # NumPy array instead of JAX

        with pytest.raises(TypeError, match="Input must be a JAX array"):
            validate_spectral_input(x, spatial_dims=1)  # type: ignore[arg-type]

    def test_validate_spectral_input_insufficient_dimensions(self):
        """Test validation with insufficient dimensions."""
        x = jnp.ones((32,))  # 1D array

        with pytest.raises(
            ValueError, match=r"Input tensor has .* dimensions, but requires at least"
        ):
            validate_spectral_input(x, spatial_dims=2)  # Expecting 2D

    def test_validate_spectral_input_spatial_size_too_small(self):
        """Test validation with spatial dimensions too small."""
        x = jnp.ones((1,))  # Size 1, default min is 2

        with pytest.raises(
            ValueError,
            match=r"Spatial dimension .* has size .*, but minimum required size is",
        ):
            validate_spectral_input(x, spatial_dims=1)

    def test_validate_spectral_input_zero_sized_dimension(self):
        """Test validation with zero-sized dimension."""
        x = jnp.empty((0, 32))  # Zero-sized first dimension

        with pytest.raises(
            ValueError,
            match=r"Spatial dimension .* has size .*, but minimum required size is",
        ):
            validate_spectral_input(x, spatial_dims=2)

    def test_validate_spectral_input_invalid_dtype_int(self):
        """Test validation with integer dtype."""
        x = jnp.ones((32,), dtype=jnp.int32)

        with pytest.raises(
            ValueError, match=r"Input dtype .* is not supported for spectral operations"
        ):
            validate_spectral_input(x, spatial_dims=1)

    def test_validate_spectral_input_invalid_dtype_bool(self):
        """Test validation with boolean dtype."""
        x = jnp.ones((32,), dtype=jnp.bool_)

        with pytest.raises(
            ValueError, match=r"Input dtype .* is not supported for spectral operations"
        ):
            validate_spectral_input(x, spatial_dims=1)

    def test_validate_spectral_input_multiple_small_dimensions(self):
        """Test validation with multiple small spatial dimensions."""
        x = jnp.ones((1, 1))  # Both dimensions too small

        with pytest.raises(
            ValueError,
            match=r"Spatial dimension .* has size .*, but minimum required size is",
        ):
            validate_spectral_input(x, spatial_dims=2)

    def test_validate_spectral_input_insufficient_size_along_axis(self):
        """Test validation with insufficient size along axis."""
        # Single point in spatial dimension (insufficient for spectral analysis)
        x = jnp.ones((1,))  # Only 1 point, minimum is 2

        with pytest.raises(
            ValueError,
            match=r"Spatial dimension 0 has size 1.*minimum required size is 2",
        ):
            validate_spectral_input(x, spatial_dims=1)


class TestValidateSpatialDims:
    """Test spatial dimensions validation."""

    def test_validate_spatial_dims_valid_1(self):
        """Test validation with spatial_dims=1."""
        validate_spatial_dims(1)  # Should not raise

    def test_validate_spatial_dims_valid_2(self):
        """Test validation with spatial_dims=2."""
        validate_spatial_dims(2)  # Should not raise

    def test_validate_spatial_dims_valid_3(self):
        """Test validation with spatial_dims=3."""
        validate_spatial_dims(3)  # Should not raise

    def test_validate_spatial_dims_not_integer(self):
        """Test validation with non-integer spatial_dims."""
        with pytest.raises(TypeError, match="spatial_dims must be an integer"):
            validate_spatial_dims(2.5)  # type: ignore[arg-type]

    def test_validate_spatial_dims_string(self):
        """Test validation with string spatial_dims."""
        with pytest.raises(TypeError, match="spatial_dims must be an integer"):
            validate_spatial_dims("2")  # type: ignore[arg-type]

    def test_validate_spatial_dims_zero(self):
        """Test validation with spatial_dims=0."""
        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_spatial_dims(0)

    def test_validate_spatial_dims_negative(self):
        """Test validation with negative spatial_dims."""
        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_spatial_dims(-1)

    def test_validate_spatial_dims_too_large(self):
        """Test validation with spatial_dims too large."""
        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_spatial_dims(4)


class TestValidateFFTShape:
    """Test FFT shape validation."""

    def test_validate_fft_shape_1d_valid(self):
        """Test FFT shape validation for valid 1D case."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)  # rfft of size 32
        target_shape = (32,)
        validate_fft_shape(x_ft, target_shape, spatial_dims=1)  # Should not raise

    def test_validate_fft_shape_2d_valid(self):
        """Test FFT shape validation for valid 2D case."""
        x_ft = jnp.ones((32, 17), dtype=jnp.complex64)  # rfft of 32x32
        target_shape = (32, 32)
        validate_fft_shape(x_ft, target_shape, spatial_dims=2)  # Should not raise

    def test_validate_fft_shape_3d_valid(self):
        """Test FFT shape validation for valid 3D case."""
        x_ft = jnp.ones((16, 16, 9), dtype=jnp.complex64)  # rfft of 16x16x16
        target_shape = (16, 16, 16)
        validate_fft_shape(x_ft, target_shape, spatial_dims=3)  # Should not raise

    def test_validate_fft_shape_with_batch_dims(self):
        """Test FFT shape validation with batch dimensions."""
        x_ft = jnp.ones((4, 3, 32, 17), dtype=jnp.complex64)
        target_shape = (4, 3, 32, 32)
        validate_fft_shape(x_ft, target_shape, spatial_dims=2)  # Should not raise

    def test_validate_fft_shape_complex128(self):
        """Test FFT shape validation with complex128 dtype."""
        x_ft = jnp.ones((17,), dtype=jnp.complex128)
        target_shape = (32,)
        validate_fft_shape(x_ft, target_shape, spatial_dims=1)  # Should not raise

    def test_validate_fft_shape_not_jax_array(self):
        """Test FFT shape validation with non-JAX array."""
        x_ft = np.ones((17,), dtype=np.complex64)
        target_shape = (32,)

        with pytest.raises(TypeError, match="x_ft must be a JAX array"):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)  # type: ignore[arg-type]

    def test_validate_fft_shape_target_not_sequence(self):
        """Test FFT shape validation with non-sequence target_shape."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)
        target_shape = 32  # Not a sequence

        with pytest.raises(TypeError, match="target_shape must be a sequence"):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)  # type: ignore[arg-type]

    def test_validate_fft_shape_invalid_spatial_dims(self):
        """Test FFT shape validation with invalid spatial_dims."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)
        target_shape = (32,)

        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_fft_shape(x_ft, target_shape, spatial_dims=0)

    def test_validate_fft_shape_dimension_mismatch(self):
        """Test FFT shape validation with dimension mismatch."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)  # 1D
        target_shape = (32, 32)  # 2D

        with pytest.raises(
            ValueError,
            match=r"Target shape dimensionality .* doesn't match FFT tensor dimensionality",
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=2)

    def test_validate_fft_shape_not_complex(self):
        """Test FFT shape validation with non-complex dtype."""
        x_ft = jnp.ones((17,), dtype=jnp.float32)  # Real instead of complex
        target_shape = (32,)

        with pytest.raises(ValueError, match="FFT tensor must have complex dtype"):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)

    def test_validate_fft_shape_negative_target_size(self):
        """Test FFT shape validation with negative target size."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)
        target_shape = (-32,)  # Negative size

        with pytest.raises(
            ValueError, match=r"Target shape element .* must be a positive integer"
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)

    def test_validate_fft_shape_zero_target_size(self):
        """Test FFT shape validation with zero target size."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)
        target_shape = (0,)  # Zero size

        with pytest.raises(
            ValueError, match=r"Target shape element .* must be a positive integer"
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)

    def test_validate_fft_shape_float_target_size(self):
        """Test FFT shape validation with float target size."""
        x_ft = jnp.ones((17,), dtype=jnp.complex64)
        target_shape = (32.5,)  # Float instead of int

        with pytest.raises(
            ValueError, match=r"Target shape element .* must be a positive integer"
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)  # type: ignore[arg-type]

    def test_validate_fft_shape_incompatible_last_dim(self):
        """Test FFT shape validation with incompatible last dimension."""
        x_ft = jnp.ones((15,), dtype=jnp.complex64)  # Wrong size for rfft
        target_shape = (32,)  # Would expect 17 for rfft

        with pytest.raises(
            ValueError,
            match=r"FFT last spatial dimension .* incompatible with target size",
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=1)

    def test_validate_fft_shape_incompatible_other_dims(self):
        """Test FFT shape validation with incompatible non-last dimensions."""
        x_ft = jnp.ones((30, 17), dtype=jnp.complex64)  # Wrong first dimension
        target_shape = (32, 32)  # First dim should be 32

        with pytest.raises(
            ValueError,
            match=r"FFT spatial dimension .* has size .*, but target requires",
        ):
            validate_fft_shape(x_ft, target_shape, spatial_dims=2)


class TestValidateGridSpacing:
    """Test grid spacing validation."""

    def test_validate_grid_spacing_scalar_positive(self):
        """Test grid spacing validation with positive scalar."""
        dx = validate_grid_spacing(0.1, spatial_dims=1)
        assert dx == [0.1]

    def test_validate_grid_spacing_scalar_multiple_dims(self):
        """Test grid spacing validation with scalar for multiple dimensions."""
        dx = validate_grid_spacing(0.1, spatial_dims=3)
        assert dx == [0.1, 0.1, 0.1]

    def test_validate_grid_spacing_sequence_valid(self):
        """Test grid spacing validation with valid sequence."""
        dx = validate_grid_spacing([0.1, 0.2], spatial_dims=2)
        assert dx == [0.1, 0.2]

    def test_validate_grid_spacing_tuple_valid(self):
        """Test grid spacing validation with valid tuple."""
        dx = validate_grid_spacing((0.1, 0.2, 0.3), spatial_dims=3)
        assert dx == [0.1, 0.2, 0.3]

    def test_validate_grid_spacing_jax_array_valid(self):
        """Test grid spacing validation with JAX array."""
        dx_array = jnp.array([0.1, 0.2])

        # JAX arrays should be properly handled and converted
        result = validate_grid_spacing(dx_array, spatial_dims=2)
        assert len(result) == 2
        assert all(isinstance(x, float) for x in result)
        assert result[0] == pytest.approx(0.1, rel=1e-6)
        assert result[1] == pytest.approx(0.2, rel=1e-6)

    def test_validate_grid_spacing_integer_scalar(self):
        """Test grid spacing validation with integer scalar."""
        dx = validate_grid_spacing(1, spatial_dims=2)
        assert dx == [1.0, 1.0]

    def test_validate_grid_spacing_mixed_types(self):
        """Test grid spacing validation with mixed integer/float."""
        dx = validate_grid_spacing([1, 0.5], spatial_dims=2)
        assert dx == [1.0, 0.5]

    def test_validate_grid_spacing_invalid_spatial_dims(self):
        """Test grid spacing validation with invalid spatial_dims."""
        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_grid_spacing(0.1, spatial_dims=0)

    def test_validate_grid_spacing_negative_scalar(self):
        """Test grid spacing validation with negative scalar."""
        with pytest.raises(ValueError, match="Grid spacing must be positive"):
            validate_grid_spacing(-0.1, spatial_dims=1)

    def test_validate_grid_spacing_zero_scalar(self):
        """Test grid spacing validation with zero scalar."""
        with pytest.raises(ValueError, match="Grid spacing must be positive"):
            validate_grid_spacing(0.0, spatial_dims=1)

    def test_validate_grid_spacing_wrong_type(self):
        """Test grid spacing validation with wrong type."""
        with pytest.raises(TypeError, match="dx must be a scalar or sequence"):
            validate_grid_spacing("0.1", spatial_dims=1)  # type: ignore[arg-type]

    def test_validate_grid_spacing_wrong_length(self):
        """Test grid spacing validation with wrong sequence length."""
        with pytest.raises(
            ValueError, match=r"Grid spacing length .* doesn't match spatial_dims"
        ):
            validate_grid_spacing([0.1, 0.2], spatial_dims=1)

    def test_validate_grid_spacing_non_numeric_element(self):
        """Test grid spacing validation with non-numeric sequence element."""
        with pytest.raises(TypeError, match=r"dx element .* must be numeric"):
            validate_grid_spacing([0.1, "0.2"], spatial_dims=2)  # type: ignore[list-item]

    def test_validate_grid_spacing_negative_element(self):
        """Test grid spacing validation with negative sequence element."""
        with pytest.raises(ValueError, match=r"dx element .* must be positive"):
            validate_grid_spacing([0.1, -0.2], spatial_dims=2)

    def test_validate_grid_spacing_zero_element(self):
        """Test grid spacing validation with zero sequence element."""
        with pytest.raises(ValueError, match=r"dx element .* must be positive"):
            validate_grid_spacing([0.1, 0.0], spatial_dims=2)


class TestValidateAxisParameter:
    """Test axis parameter validation."""

    def test_validate_axis_parameter_none(self):
        """Test axis parameter validation with None (all axes)."""
        axes = validate_axis_parameter(None, spatial_dims=2)
        assert axes == [0, 1]  # Range from 0 to spatial_dims-1

    def test_validate_axis_parameter_single_positive(self):
        """Test axis parameter validation with single positive axis."""
        axes = validate_axis_parameter(1, spatial_dims=3)
        assert axes == [1]

    def test_validate_axis_parameter_single_negative(self):
        """Test axis parameter validation with single negative axis."""
        with pytest.raises(ValueError, match=r"Axis .* out of range"):
            validate_axis_parameter(-1, spatial_dims=2)

    def test_validate_axis_parameter_sequence_positive(self):
        """Test axis parameter validation with sequence of positive axes."""
        axes = validate_axis_parameter([1, 2], spatial_dims=3)
        assert axes == [1, 2]

    def test_validate_axis_parameter_sequence_negative(self):
        """Test axis parameter validation with sequence of negative axes."""
        with pytest.raises(ValueError, match=r"Axis element .* out of range"):
            validate_axis_parameter([-2, -1], spatial_dims=2)

    def test_validate_axis_parameter_sequence_mixed(self):
        """Test axis parameter validation with mixed positive/negative axes."""
        with pytest.raises(ValueError, match=r"Axis element .* out of range"):
            validate_axis_parameter([0, -1], spatial_dims=3)

    def test_validate_axis_parameter_tuple(self):
        """Test axis parameter validation with tuple."""
        axes = validate_axis_parameter((1, 2), spatial_dims=3)
        assert axes == [1, 2]

    def test_validate_axis_parameter_jax_array(self):
        """Test axis parameter validation with JAX array."""
        axis_array = jnp.array([1, 2])
        with pytest.raises(TypeError, match="unhashable type:"):
            validate_axis_parameter(axis_array, spatial_dims=3)  # type: ignore[arg-type]

    def test_validate_axis_parameter_invalid_spatial_dims(self):
        """Test axis parameter validation with invalid spatial_dims."""
        with pytest.raises(ValueError, match="spatial_dims must be 1, 2, or 3"):
            validate_axis_parameter(0, spatial_dims=-1)

    def test_validate_axis_parameter_wrong_type(self):
        """Test axis parameter validation with wrong type."""
        with pytest.raises(TypeError, match="axis must be an int, sequence, or None"):
            validate_axis_parameter("axis", spatial_dims=2)  # type: ignore[arg-type]

    def test_validate_axis_parameter_non_integer_element(self):
        """Test axis parameter validation with non-integer sequence element."""
        with pytest.raises(TypeError, match=r"axis element .* must be an integer"):
            validate_axis_parameter([1, 2.5], spatial_dims=3)  # type: ignore[list-item]

    def test_validate_axis_parameter_float_single(self):
        """Test axis parameter validation with float single axis."""
        with pytest.raises(TypeError, match="axis must be an int, sequence, or None"):
            validate_axis_parameter(1.5, spatial_dims=2)  # type: ignore[arg-type]

    def test_validate_axis_parameter_duplicate_axes(self):
        """Test axis parameter validation with duplicate axes."""
        with pytest.raises(ValueError, match="Duplicate axes found"):
            validate_axis_parameter([1, 1], spatial_dims=3)

    def test_validate_axis_parameter_equivalent_duplicates(self):
        """Test axis parameter validation with equivalent positive/negative axes."""
        # This test is actually invalid since negative axes are not supported
        with pytest.raises(ValueError, match=r"Axis element .* out of range"):
            validate_axis_parameter([1, -2], spatial_dims=3)


class TestIntegrationAndEdgeCases:
    """Test integration scenarios and edge cases."""

    def test_validation_pipeline_success(self):
        """Test complete validation pipeline with valid inputs."""
        # Create valid inputs
        x = jnp.ones((4, 3, 32, 32), dtype=jnp.float32)
        x_ft = jnp.ones((4, 3, 32, 17), dtype=jnp.complex64)
        target_shape = (4, 3, 32, 32)
        dx = [0.1, 0.1]
        spatial_dims = 2
        axis = None

        # All validations should pass
        validate_spectral_input(x, spatial_dims)
        validate_spatial_dims(spatial_dims)
        validate_fft_shape(x_ft, target_shape, spatial_dims)
        dx_validated = validate_grid_spacing(dx, spatial_dims)
        axes_validated = validate_axis_parameter(axis, spatial_dims)

        assert dx_validated == [0.1, 0.1]
        assert axes_validated == [0, 1]  # Corrected expectation

    def test_validation_with_minimal_sizes(self):
        """Test validation with minimal valid sizes."""
        # Minimal 1D case
        x = jnp.ones((2,))
        validate_spectral_input(x, spatial_dims=1, min_spatial_size=2)

        # Minimal 2D case
        x = jnp.ones((2, 2))
        validate_spectral_input(x, spatial_dims=2, min_spatial_size=2)

    def test_validation_with_large_inputs(self):
        """Test validation with large inputs (CI-friendly sizes)."""
        # Large 3D case - reduced from 1024^3 to 128^3 for CI memory limits
        # Still tests large input validation without exhausting CI resources
        x = jnp.ones((128, 128, 128))  # ~2M elements = ~8MB instead of 4GB
        validate_spectral_input(x, spatial_dims=3)

        # Corresponding FFT validation
        x_ft = jnp.ones((128, 128, 65), dtype=jnp.complex64)  # ~1M elements = ~8MB
        target_shape = (128, 128, 128)
        validate_fft_shape(x_ft, target_shape, spatial_dims=3)

    def test_validation_error_message_clarity(self):
        """Test that validation error messages are clear and informative."""
        x = jnp.ones((1,))  # Too small

        with pytest.raises(
            ValueError,
            match=r"Spatial dimension 0 has size 1.*minimum required size is 2",
        ):
            validate_spectral_input(x, spatial_dims=1)

    def test_validation_preserves_input_types(self):
        """Test that validation doesn't modify input types."""
        # Test that validation functions don't modify inputs
        x_original = jnp.ones((32,), dtype=jnp.float32)
        x_copy = x_original.copy()

        validate_spectral_input(x_original, spatial_dims=1)

        assert jnp.array_equal(x_original, x_copy)
        assert x_original.dtype == x_copy.dtype

    def test_validation_with_different_backends(self):
        """Test validation works with different JAX backends."""
        # This test ensures validation works regardless of JAX backend
        x = jnp.ones((32, 32))

        # Should work regardless of current backend configuration
        validate_spectral_input(x, spatial_dims=2)

        # Validate with complex input
        x_complex = jnp.ones((32, 32), dtype=jnp.complex128)
        validate_spectral_input(x_complex, spatial_dims=2)
