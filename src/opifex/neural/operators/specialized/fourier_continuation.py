"""Fourier Continuation Layers - Signal Extension and Boundary Handling.

This module implements Fourier continuation methods for extending signals beyond
their boundaries using various strategies. These are essential for neural operators
that need to handle boundary conditions in PDEs and signal processing.

Key Features:
- FourierContinuationExtender: Core signal extension with multiple boundary strategies
- PeriodicContinuation: Periodic boundary extension
- SymmetricContinuation: Symmetric/mirror boundary extension
- SmoothContinuation: Smooth tapering to zero at boundaries
- Multi-dimensional signal extension support
- Physics-informed boundary condition handling
"""

from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from flax import nnx


class FourierContinuationExtender(nnx.Module):
    """Fourier continuation-based signal extender.

    This layer extends signals beyond their boundaries using Fourier-based
    continuation methods, enabling proper handling of boundary conditions
    in neural operators for PDEs.

    Args:
        extension_type: Type of continuation ('periodic', 'symmetric', 'smooth', 'zero')
        extension_length: Number of points to extend on each side
        taper_width: Width of tapering region for smooth continuation
        smooth_order: Order of smoothness for smooth continuation
        precision: Numerical precision for computation
    """

    def __init__(
        self,
        extension_type: Literal["periodic", "symmetric", "smooth", "zero"] = "smooth",
        extension_length: int = 32,
        taper_width: float = 0.1,
        smooth_order: int = 8,
    ):
        super().__init__()

        self.extension_type = extension_type
        self.extension_length = extension_length
        self.taper_width = taper_width
        self.smooth_order = smooth_order

        # Validate parameters
        if extension_length <= 0:
            raise ValueError("extension_length must be positive")
        if not 0.0 < taper_width <= 1.0:
            raise ValueError("taper_width must be in (0, 1]")
        if smooth_order <= 0:
            raise ValueError("smooth_order must be positive")

    def _create_taper_window(self, length: int, taper_fraction: float) -> jnp.ndarray:
        """Create smooth tapering window.

        Args:
            length: Total length of the window
            taper_fraction: Fraction of length to use for tapering

        Returns:
            Tapering window array
        """
        taper_points = int(length * taper_fraction)
        if taper_points == 0:
            return jnp.ones(length)

        # Create smooth tapering function using raised cosine
        x = jnp.linspace(0, jnp.pi, taper_points)
        taper = 0.5 * (1.0 + jnp.cos(x))

        # Build full window
        window = jnp.ones(length)
        window = window.at[:taper_points].set(taper[::-1])  # Left taper
        return window.at[-taper_points:].set(taper)  # Right taper

    def _periodic_extension_1d(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extend signal periodically in 1D.

        Args:
            signal: Input signal of shape (length,) or (..., length)

        Returns:
            Extended signal
        """
        signal_length = signal.shape[-1]

        # For periodic extension, we want to extend by repeating the signal pattern
        # Left extension comes from the end of the signal
        # Right extension comes from the beginning of the signal

        if self.extension_length <= signal_length:
            # Simple case: extension fits within signal
            left_extension = signal[..., -self.extension_length :]
            right_extension = signal[..., : self.extension_length]
        else:
            # Extension is longer than signal, need to repeat pattern
            # Left extension
            n_full_repeats_left = self.extension_length // signal_length
            remainder_left = self.extension_length % signal_length

            if n_full_repeats_left > 0:
                repeated_left = jnp.tile(signal, n_full_repeats_left)
                if remainder_left > 0:
                    partial_left = signal[..., -remainder_left:]
                    left_extension = jnp.concatenate(
                        [repeated_left, partial_left], axis=-1
                    )
                else:
                    left_extension = repeated_left
            else:
                left_extension = signal[..., -remainder_left:]

            # Right extension
            n_full_repeats_right = self.extension_length // signal_length
            remainder_right = self.extension_length % signal_length

            if n_full_repeats_right > 0:
                repeated_right = jnp.tile(signal, n_full_repeats_right)
                if remainder_right > 0:
                    partial_right = signal[..., :remainder_right]
                    right_extension = jnp.concatenate(
                        [repeated_right, partial_right], axis=-1
                    )
                else:
                    right_extension = repeated_right
            else:
                right_extension = signal[..., :remainder_right]

        return jnp.concatenate([left_extension, signal, right_extension], axis=-1)

    def _symmetric_extension_1d(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extend signal symmetrically in 1D.

        Args:
            signal: Input signal of shape (length,) or (..., length)

        Returns:
            Extended signal
        """
        signal_length = signal.shape[-1]

        # Handle case where extension is longer than signal
        if self.extension_length >= signal_length:
            # Repeat and flip the entire signal as needed
            n_repeats = (self.extension_length // signal_length) + 1

            # Create left extension
            extended_signal = signal
            for i in range(n_repeats):
                if i % 2 == 0:
                    extended_signal = jnp.concatenate(
                        [jnp.flip(signal, axis=-1), extended_signal], axis=-1
                    )
                else:
                    extended_signal = jnp.concatenate(
                        [signal, extended_signal], axis=-1
                    )
            left_ext = extended_signal[
                ..., -(self.extension_length + signal_length) : -signal_length
            ]

            # Create right extension
            extended_signal = signal
            for i in range(n_repeats):
                if i % 2 == 0:
                    extended_signal = jnp.concatenate(
                        [extended_signal, jnp.flip(signal, axis=-1)], axis=-1
                    )
                else:
                    extended_signal = jnp.concatenate(
                        [extended_signal, signal], axis=-1
                    )
            right_ext = extended_signal[
                ..., signal_length : signal_length + self.extension_length
            ]
        else:
            # Simple case: extension is shorter than signal
            left_ext = jnp.flip(signal[..., : self.extension_length], axis=-1)
            right_ext = jnp.flip(signal[..., -self.extension_length :], axis=-1)

        return jnp.concatenate([left_ext, signal, right_ext], axis=-1)

    def _zero_extension_1d(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extend signal with zeros in 1D.

        Args:
            signal: Input signal of shape (length,) or (..., length)

        Returns:
            Extended signal
        """
        shape = signal.shape
        left_zeros = jnp.zeros((*shape[:-1], self.extension_length))
        right_zeros = jnp.zeros((*shape[:-1], self.extension_length))

        return jnp.concatenate([left_zeros, signal, right_zeros], axis=-1)

    def _smooth_extension_1d(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extend signal smoothly by tapering to zero.

        Args:
            signal: Input signal of shape (length,) or (..., length)

        Returns:
            Extended signal
        """
        # Create smooth extensions that taper to zero
        length = signal.shape[-1]

        # Get boundary values and derivatives for smooth matching
        left_val = signal[..., 0]
        right_val = signal[..., -1]

        # Estimate derivatives at boundaries
        if length >= 2:
            left_grad = signal[..., 1] - signal[..., 0]
            right_grad = signal[..., -1] - signal[..., -2]
        else:
            left_grad = jnp.zeros_like(left_val)
            right_grad = jnp.zeros_like(right_val)

        # Create smooth extension using polynomial interpolation
        x_ext = jnp.linspace(0, 1, self.extension_length)

        # Left extension (going backwards from signal start)
        poly_left = (
            left_val[..., None] * (1 - x_ext) ** 3
            + left_grad[..., None] * x_ext * (1 - x_ext) ** 2
        )
        left_ext = jnp.flip(poly_left, axis=-1)

        # Right extension (going forwards from signal end)
        poly_right = (
            right_val[..., None] * (1 - x_ext) ** 3
            + right_grad[..., None] * x_ext * (1 - x_ext) ** 2
        )

        return jnp.concatenate([left_ext, signal, poly_right], axis=-1)

    def extend_1d(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extend 1D signal using specified continuation method.

        Args:
            signal: Input signal to extend

        Returns:
            Extended signal
        """

        if self.extension_type == "periodic":
            return self._periodic_extension_1d(signal)
        if self.extension_type == "symmetric":
            return self._symmetric_extension_1d(signal)
        if self.extension_type == "zero":
            return self._zero_extension_1d(signal)
        if self.extension_type == "smooth":
            return self._smooth_extension_1d(signal)
        raise ValueError(f"Unknown extension type: {self.extension_type}")

    def extend_2d(
        self, signal: jnp.ndarray, axes: tuple[int, int] = (-2, -1)
    ) -> jnp.ndarray:
        """Extend 2D signal using specified continuation method.

        Args:
            signal: Input signal to extend, shape (..., height, width)
            axes: Axes along which to extend

        Returns:
            Extended signal
        """

        # Extend along first axis
        extended = jnp.apply_along_axis(self.extend_1d, axis=axes[0], arr=signal)

        # Extend along second axis
        return jnp.apply_along_axis(self.extend_1d, axis=axes[1], arr=extended)

    def __call__(
        self,
        x: jnp.ndarray,
        axes: int | tuple[int, ...] | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply Fourier continuation to input.

        Args:
            x: Input tensor to extend
            axes: Axes along which to extend (default: last axis for 1D, last 2 for 2D+)
            deterministic: Whether to use deterministic mode

        Returns:
            Extended tensor
        """
        if axes is None:
            if x.ndim == 1:
                axes = -1
            elif x.ndim >= 2:
                axes = (-2, -1)
            else:
                raise ValueError("Input must have at least 1 dimension")

        if isinstance(axes, int):
            # 1D extension
            return jnp.apply_along_axis(self.extend_1d, axis=axes, arr=x)
        if isinstance(axes, tuple) and len(axes) == 2:
            # 2D extension
            return self.extend_2d(x, axes)
        raise ValueError("Only 1D and 2D extensions are currently supported")


class PeriodicContinuation(FourierContinuationExtender):
    """Periodic continuation for signals with periodic boundary conditions."""

    def __init__(
        self,
        extension_length: int = 32,
    ):
        super().__init__(
            extension_type="periodic",
            extension_length=extension_length,
        )


class SymmetricContinuation(FourierContinuationExtender):
    """Symmetric continuation for signals with mirror boundary conditions."""

    def __init__(
        self,
        extension_length: int = 32,
    ):
        super().__init__(
            extension_type="symmetric",
            extension_length=extension_length,
        )


class SmoothContinuation(FourierContinuationExtender):
    """Smooth continuation for signals that taper to zero at boundaries."""

    def __init__(
        self,
        extension_length: int = 32,
        taper_width: float = 0.1,
        smooth_order: int = 8,
    ):
        super().__init__(
            extension_type="smooth",
            extension_length=extension_length,
            taper_width=taper_width,
            smooth_order=smooth_order,
        )


class FourierBoundaryHandler(nnx.Module):
    """Neural network layer for intelligent boundary handling.

    This layer learns optimal boundary conditions by combining multiple
    continuation strategies and adapting them based on signal characteristics.

    Args:
        continuation_methods: List of continuation methods to combine
        hidden_dim: Hidden dimension for the decision network
        rngs: Random number generator state
    """

    def __init__(
        self,
        continuation_methods: Sequence[str] = ("periodic", "symmetric", "smooth"),
        extension_length: int = 32,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        self.continuation_methods = continuation_methods
        self.extension_length = extension_length

        # Create continuation extenders for each method
        # Create continuation extenders for each method
        extenders = {}
        for method in continuation_methods:
            # Validate method is a valid extension type
            valid_methods = ["periodic", "symmetric", "smooth", "zero"]
            if method not in valid_methods:
                raise ValueError(
                    f"Invalid method {method}, must be one of {valid_methods}"
                )

            extenders[method] = FourierContinuationExtender(
                extension_type=method,  # type: ignore[arg-type]
                extension_length=extension_length,
            )
        self.extenders = nnx.Dict(extenders)

        # Neural network to decide on boundary strategy
        self.decision_network = nnx.Sequential(
            nnx.Linear(4, hidden_dim, rngs=rngs),  # 4 signal features
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(
                hidden_dim // 2,
                len(continuation_methods),
                rngs=rngs,
            ),
            nnx.softmax,  # Weights for combining methods
        )

    def _extract_signal_features(self, signal: jnp.ndarray) -> jnp.ndarray:
        """Extract features to characterize the signal for boundary decisions.

        Args:
            signal: Input signal

        Returns:
            Feature vector
        """
        # Extract statistical features from the signal
        # Features: mean, std, boundary gradient magnitude, periodicity estimate

        mean_val = jnp.mean(signal)
        std_val = jnp.std(signal)

        # Boundary gradient magnitude
        if signal.shape[-1] >= 2:
            left_grad = jnp.abs(signal[..., 1] - signal[..., 0])
            right_grad = jnp.abs(signal[..., -1] - signal[..., -2])
            boundary_grad = jnp.mean(jnp.array([left_grad, right_grad]))
        else:
            boundary_grad = jnp.array(0.0)

        # Simple periodicity estimate using autocorrelation
        if signal.shape[-1] >= 4:
            # Compare first and last quarters
            quarter_len = signal.shape[-1] // 4
            first_quarter = signal[..., :quarter_len]
            last_quarter = signal[..., -quarter_len:]
            periodicity = jnp.corrcoef(first_quarter.flatten(), last_quarter.flatten())[
                0, 1
            ]
            periodicity = jnp.where(jnp.isnan(periodicity), 0.0, periodicity)
        else:
            periodicity = jnp.array(0.0)

        return jnp.array(
            [mean_val, std_val, boundary_grad, periodicity],
        )

    def __call__(
        self,
        x: jnp.ndarray,
        axes: int | tuple[int, ...] | None = None,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """Apply intelligent boundary handling to input.

        Args:
            x: Input tensor to extend
            axes: Axes along which to extend
            deterministic: Whether to use deterministic mode

        Returns:
            Extended tensor with optimal boundary conditions
        """
        # Extract signal features for decision making
        features = self._extract_signal_features(x)

        # Get weights for different continuation methods
        method_weights = self.decision_network(features)

        # Apply each continuation method
        extended_results = []
        for method in self.continuation_methods:
            extended = self.extenders[method](x, axes, deterministic)
            extended_results.append(extended)

        # Ensure all results have the same shape before stacking
        if len(extended_results) > 1:
            # Get the expected shape from the first result
            target_shape = extended_results[0].shape

            # Verify all results have the same shape, pad if necessary
            for i in range(len(extended_results)):
                if extended_results[i].shape != target_shape:
                    # This should not happen if our extension methods are correct
                    # But add safety check
                    raise ValueError(
                        f"Extension method {method} returned shape "
                        f"{extended_results[i].shape}, expected {target_shape}"
                    )

        # Combine results using learned weights
        extended_signals = jnp.stack(extended_results, axis=0)

        # Reshape method_weights to broadcast correctly
        if x.ndim == 1:
            weights_shape = (len(self.continuation_methods), 1)
            method_weights_reshaped = method_weights.reshape(weights_shape)
        elif x.ndim == 2:
            weights_shape = (len(self.continuation_methods), 1, 1)
            method_weights_reshaped = method_weights.reshape(weights_shape)
        else:
            # For higher dimensions, use broadcasting instead of reshape
            shape_list = [len(self.continuation_methods)] + [1] * x.ndim
            method_weights_reshaped = method_weights.reshape(shape_list)

        return jnp.sum(method_weights_reshaped * extended_signals, axis=0)


def create_continuation_pipeline(
    methods: Sequence[str] = ("periodic", "symmetric", "smooth"),
    extension_length: int = 32,
    use_intelligent_handler: bool = True,
    *,
    rngs: nnx.Rngs | None = None,
) -> nnx.Module:
    """Create a complete Fourier continuation pipeline.

    Args:
        methods: Continuation methods to include
        extension_length: Length of extension on each side
        use_intelligent_handler: Whether to use intelligent boundary handler
        rngs: Random number generator state (required if use_intelligent_handler=True)

    Returns:
        Continuation pipeline module
    """
    if use_intelligent_handler:
        if rngs is None:
            raise ValueError("rngs required for intelligent boundary handler")
        return FourierBoundaryHandler(
            continuation_methods=methods,
            extension_length=extension_length,
            rngs=rngs,
        )
    # Return a simple continuation extender
    method = methods[0] if methods else "smooth"
    valid_methods = ["periodic", "symmetric", "smooth", "zero"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method {method}, must be one of {valid_methods}")

    return FourierContinuationExtender(
        extension_type=method,  # type: ignore[arg-type]
        extension_length=extension_length,
    )
