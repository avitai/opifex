"""Test suite for Fourier Continuation layers.

This module tests all Fourier continuation functionality including:
- Basic signal extension methods (periodic, symmetric, smooth, zero)
- Multi-dimensional signal extension
- Intelligent boundary handling with neural networks
- JAX transformations compatibility
- Error handling and edge cases
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.neural.operators.specialized.fourier_continuation import (
    create_continuation_pipeline,
    FourierBoundaryHandler,
    FourierContinuationExtender,
    PeriodicContinuation,
    SmoothContinuation,
    SymmetricContinuation,
)


class TestFourierContinuationExtender:
    """Test the core FourierContinuationExtender class."""

    def test_fourier_continuation_initialization(self):
        """Test basic initialization of FourierContinuationExtender."""
        extender = FourierContinuationExtender(
            extension_type="smooth",
            extension_length=16,
        )

        assert extender.extension_type == "smooth"
        assert extender.extension_length == 16

    def test_invalid_extension_parameters(self):
        """Test validation of extension parameters."""
        # Invalid extension length
        with pytest.raises(ValueError, match="extension_length must be positive"):
            FourierContinuationExtender(extension_length=-1)

        # Invalid taper width
        with pytest.raises(ValueError, match="taper_width must be in"):
            FourierContinuationExtender(taper_width=0.0)

        with pytest.raises(ValueError, match="taper_width must be in"):
            FourierContinuationExtender(taper_width=1.5)

        # Invalid smooth order
        with pytest.raises(ValueError, match="smooth_order must be positive"):
            FourierContinuationExtender(smooth_order=0)

    def test_periodic_extension_1d(self):
        """Test periodic signal extension in 1D."""
        extender = FourierContinuationExtender(
            extension_type="periodic",
            extension_length=8,
        )

        # Create a simple periodic signal
        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extender.extend_1d(signal)

        # Should have original length + 2 * extension_length
        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_symmetric_extension_1d(self):
        """Test symmetric signal extension in 1D."""
        extender = FourierContinuationExtender(
            extension_type="symmetric",
            extension_length=4,
        )

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extender.extend_1d(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

        # Check symmetric extension
        # Left extension should be flipped version of first part
        left_ext = extended[: extender.extension_length]
        expected_left = jnp.flip(signal[: extender.extension_length])
        assert jnp.allclose(left_ext, expected_left)

        # Right extension should be flipped version of last part
        right_ext = extended[-extender.extension_length :]
        expected_right = jnp.flip(signal[-extender.extension_length :])
        assert jnp.allclose(right_ext, expected_right)

    def test_zero_extension_1d(self):
        """Test zero-padding signal extension in 1D."""
        extender = FourierContinuationExtender(
            extension_type="zero",
            extension_length=6,
        )

        signal = jnp.array([1.0, 2.0, 3.0])
        extended = extender.extend_1d(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

        # Check zero padding
        left_zeros = extended[: extender.extension_length]
        right_zeros = extended[-extender.extension_length :]

        assert jnp.allclose(left_zeros, 0.0)
        assert jnp.allclose(right_zeros, 0.0)

        # Original signal should be preserved in the middle
        middle = extended[extender.extension_length : -extender.extension_length]
        assert jnp.allclose(middle, signal)

    def test_smooth_extension_1d(self):
        """Test smooth signal extension in 1D."""
        extender = FourierContinuationExtender(
            extension_type="smooth",
            extension_length=8,
        )

        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        extended = extender.extend_1d(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

        # Check that extension is smooth (no discontinuities)
        # The extended signal should start and end close to the boundary values
        boundary_tolerance = 0.1
        assert abs(extended[extender.extension_length] - signal[0]) < boundary_tolerance
        assert (
            abs(extended[-(extender.extension_length + 1)] - signal[-1])
            < boundary_tolerance
        )

    def test_extend_2d(self):
        """Test 2D signal extension."""
        extender = FourierContinuationExtender(
            extension_type="symmetric",
            extension_length=4,
        )

        # Create a 2D signal
        signal = jnp.ones((8, 6))
        extended = extender.extend_2d(signal)

        # Should extend both dimensions
        expected_shape = (
            signal.shape[0] + 2 * extender.extension_length,
            signal.shape[1] + 2 * extender.extension_length,
        )
        assert extended.shape == expected_shape

    def test_call_method_1d(self):
        """Test the __call__ method for 1D signals."""
        extender = FourierContinuationExtender(
            extension_type="periodic",
            extension_length=5,
        )

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extender(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_call_method_2d(self):
        """Test the __call__ method for 2D signals."""
        extender = FourierContinuationExtender(
            extension_type="smooth",
            extension_length=3,
        )

        signal = jnp.ones((5, 7))
        extended = extender(signal)

        expected_shape = (
            signal.shape[0] + 2 * extender.extension_length,
            signal.shape[1] + 2 * extender.extension_length,
        )
        assert extended.shape == expected_shape

    def test_call_method_with_custom_axes(self):
        """Test __call__ method with custom axes specification."""
        extender = FourierContinuationExtender(
            extension_type="zero",
            extension_length=4,
        )

        # 3D signal, extend only along last axis
        signal = jnp.ones((2, 3, 8))
        extended = extender(signal, axes=-1)

        expected_shape = (2, 3, 8 + 2 * extender.extension_length)
        assert extended.shape == expected_shape

    def test_invalid_extension_type(self):
        """Test handling of invalid extension types."""
        extender = FourierContinuationExtender(
            extension_type="smooth",  # Valid type for initialization
            extension_length=4,
        )

        # Manually change to invalid type to test error handling
        extender.extension_type = "invalid"

        signal = jnp.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Unknown extension type"):
            extender.extend_1d(signal)

    def test_invalid_input_dimensions(self):
        """Test handling of invalid input dimensions."""
        extender = FourierContinuationExtender(extension_length=4)

        # 0D input should raise error
        signal_0d = jnp.array(1.0)
        with pytest.raises(ValueError, match="Input must have at least 1 dimension"):
            extender(signal_0d)

    def test_batched_signals(self):
        """Test extension with batched signals."""
        extender = FourierContinuationExtender(
            extension_type="symmetric",
            extension_length=3,
        )

        # Batch of 1D signals
        batch_signal = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        extended = extender(batch_signal, axes=-1)

        expected_shape = (2, 3 + 2 * extender.extension_length)
        assert extended.shape == expected_shape


class TestSpecializedContinuationClasses:
    """Test specialized continuation classes."""

    def test_periodic_continuation(self):
        """Test PeriodicContinuation class."""
        extender = PeriodicContinuation(extension_length=8)

        assert extender.extension_type == "periodic"
        assert extender.extension_length == 8

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extender(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_symmetric_continuation(self):
        """Test SymmetricContinuation class."""
        extender = SymmetricContinuation(extension_length=6)

        assert extender.extension_type == "symmetric"
        assert extender.extension_length == 6

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extender(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_smooth_continuation(self):
        """Test SmoothContinuation class."""
        extender = SmoothContinuation(
            extension_length=10,
            taper_width=0.2,
            smooth_order=6,
        )

        assert extender.extension_type == "smooth"
        assert extender.extension_length == 10
        assert extender.taper_width == 0.2
        assert extender.smooth_order == 6

        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        extended = extender(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length


class TestFourierBoundaryHandler:
    """Test the intelligent FourierBoundaryHandler class."""

    def test_boundary_handler_initialization(self):
        """Test FourierBoundaryHandler initialization."""
        rngs = nnx.Rngs(0)
        handler = FourierBoundaryHandler(
            continuation_methods=("periodic", "symmetric", "smooth"),
            extension_length=16,
            rngs=rngs,
        )

        assert len(handler.continuation_methods) == 3
        assert handler.extension_length == 16
        assert len(handler.extenders) == 3

    def test_boundary_handler_forward_pass(self):
        """Test FourierBoundaryHandler forward pass."""
        rngs = nnx.Rngs(42)
        handler = FourierBoundaryHandler(
            continuation_methods=("periodic", "symmetric"),
            extension_length=8,
            rngs=rngs,
        )

        signal = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        extended = handler(signal)

        expected_length = len(signal) + 2 * handler.extension_length
        assert extended.shape[-1] == expected_length

    def test_boundary_handler_signal_features(self):
        """Test signal feature extraction."""
        rngs = nnx.Rngs(123)
        handler = FourierBoundaryHandler(
            continuation_methods=("periodic", "smooth"),
            extension_length=4,
            rngs=rngs,
        )

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        features = handler._extract_signal_features(signal)

        # Should extract 4 features: mean, std, boundary_grad, periodicity
        assert features.shape == (4,)
        assert jnp.isrealobj(features)  # JAX-native precision - features should be real

    def test_boundary_handler_with_different_precisions(self):
        """Test boundary handler with different numerical precisions."""
        rngs = nnx.Rngs(456)

        handler_1 = FourierBoundaryHandler(
            continuation_methods=("symmetric",),
            extension_length=6,
            rngs=rngs,
        )

        # Test with different extension length
        handler_2 = FourierBoundaryHandler(
            continuation_methods=("periodic",),
            extension_length=4,
            rngs=rngs,
        )

        signal = jnp.array([1.0, 2.0, 3.0])

        extended_1 = handler_1(signal)
        extended_2 = handler_2(signal)

        # Different handlers should produce different outputs
        assert extended_1.shape[-1] != extended_2.shape[-1]

    def test_boundary_handler_invalid_methods(self):
        """Test boundary handler with invalid continuation methods."""
        rngs = nnx.Rngs(789)

        with pytest.raises(ValueError, match="Invalid method"):
            FourierBoundaryHandler(
                continuation_methods=("invalid_method",),
                extension_length=4,
                rngs=rngs,
            )


class TestContinuationPipeline:
    """Test the create_continuation_pipeline factory function."""

    def test_create_intelligent_pipeline(self):
        """Test creating an intelligent continuation pipeline."""
        rngs = nnx.Rngs(100)
        pipeline = create_continuation_pipeline(
            methods=("periodic", "symmetric", "smooth"),
            extension_length=12,
            use_intelligent_handler=True,
            rngs=rngs,
        )

        assert isinstance(pipeline, FourierBoundaryHandler)
        assert pipeline.extension_length == 12

    def test_create_simple_pipeline(self):
        """Test creating a simple continuation pipeline."""
        pipeline = create_continuation_pipeline(
            methods=("smooth",),
            extension_length=8,
            use_intelligent_handler=False,
        )

        assert isinstance(pipeline, FourierContinuationExtender)
        assert pipeline.extension_type == "smooth"
        assert pipeline.extension_length == 8

    def test_create_pipeline_missing_rngs(self):
        """Test error when creating intelligent pipeline without rngs."""
        with pytest.raises(ValueError, match="rngs required"):
            create_continuation_pipeline(
                methods=("periodic", "symmetric"),
                use_intelligent_handler=True,
                rngs=None,
            )

    def test_create_pipeline_invalid_method(self):
        """Test error when creating pipeline with invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            create_continuation_pipeline(
                methods=("invalid_method",),
                use_intelligent_handler=False,
            )

    def test_create_pipeline_empty_methods(self):
        """Test creating pipeline with empty methods (should default to smooth)."""
        pipeline = create_continuation_pipeline(
            methods=(),
            use_intelligent_handler=False,
        )

        assert isinstance(pipeline, FourierContinuationExtender)
        assert pipeline.extension_type == "smooth"


class TestJAXTransformations:
    """Test JAX transformations compatibility."""

    def test_jit_compilation(self):
        """Test JIT compilation of continuation functions."""
        extender = FourierContinuationExtender(
            extension_type="symmetric",
            extension_length=4,
        )

        @jax.jit
        def extend_fn(signal):
            return extender(signal)

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = extend_fn(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_grad_computation(self):
        """Test gradient computation through continuation layers."""
        extender = FourierContinuationExtender(
            extension_type="smooth",
            extension_length=6,
        )

        def loss_fn(signal):
            extended = extender(signal)
            return jnp.sum(extended**2)

        signal = jnp.array([1.0, 2.0, 3.0])
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(signal)

        # Gradients should have the same shape as input
        assert grads.shape == signal.shape
        # Gradients should not be zero (signal affects the output)
        assert jnp.any(grads != 0)

    def test_vmap_compatibility(self):
        """Test vmap compatibility with continuation layers."""
        extender = FourierContinuationExtender(
            extension_type="periodic",
            extension_length=3,
        )

        # Batch of signals
        batch_signals = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        # Apply vmap
        vmapped_extend = jax.vmap(extender, in_axes=0)
        extended_batch = vmapped_extend(batch_signals)

        expected_shape = (
            batch_signals.shape[0],
            batch_signals.shape[1] + 2 * extender.extension_length,
        )
        assert extended_batch.shape == expected_shape

    def test_intelligent_handler_jit_compilation(self):
        """Test JIT compilation of intelligent boundary handler."""
        rngs = nnx.Rngs(200)
        handler = FourierBoundaryHandler(
            continuation_methods=("periodic", "symmetric"),
            extension_length=5,
            rngs=rngs,
        )

        @jax.jit
        def handler_fn(signal):
            return handler(signal)

        signal = jnp.array([1.0, 2.0, 3.0, 4.0])
        extended = handler_fn(signal)

        expected_length = len(signal) + 2 * handler.extension_length
        assert extended.shape[-1] == expected_length


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_short_signals(self):
        """Test extension of very short signals."""
        extender = FourierContinuationExtender(
            extension_type="symmetric",
            extension_length=2,
        )

        # Single-element signal
        signal = jnp.array([5.0])
        extended = extender(signal)

        expected_length = 1 + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_extension_longer_than_signal(self):
        """Test extension that is longer than the original signal."""
        extender = FourierContinuationExtender(
            extension_type="periodic",
            extension_length=10,  # Longer than signal
        )

        signal = jnp.array([1.0, 2.0, 3.0])  # Length 3
        extended = extender(signal)

        expected_length = len(signal) + 2 * extender.extension_length
        assert extended.shape[-1] == expected_length

    def test_zero_signal(self):
        """Test extension of zero signals."""
        extender = FourierContinuationExtender(
            extension_type="smooth",
            extension_length=5,
        )

        signal = jnp.zeros(4)
        extended = extender(signal)

        # All values should remain close to zero
        assert jnp.allclose(extended, 0.0, atol=1e-6)
