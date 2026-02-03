"""Spectral Normalization for Neural Operators.

This module provides spectral normalization techniques for stabilizing neural
operator training by controlling the Lipschitz constant of neural networks. Spectral
normalization normalizes the spectral norm (largest singular value) of weight matrices,
which is particularly important for neural operators working with PDEs where stability
is crucial.

Key Features:
- SpectralNorm: Core spectral normalization wrapper for any linear layer
- SpectralConvolution: Spectral normalized convolution layer
- SpectralLinear: Spectral normalized linear layer
- SpectralMultiHeadAttention: Spectral normalized attention for neural operators
- PowerIteration: Efficient power iteration algorithm for spectral norm estimation
- AdaptiveSpectralNorm: Adaptive spectral normalization with learnable bounds
- Utilities for creating spectral normalized neural operator architectures
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


class PowerIteration(nnx.Module):
    """Power iteration algorithm for estimating the spectral norm.

    Estimates the largest singular value efficiently using power iteration.
    This is the core algorithm used by spectral normalization to efficiently
    estimate the largest singular value of weight matrices without computing the
    full SVD.
    """

    def __init__(
        self,
        num_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize power iteration algorithm.

        Args:
            num_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators for initialization
        """
        self.num_iterations = num_iterations
        self.eps = eps

        # Initialize with dummy values - will be updated during first call
        self.u = nnx.Param(jnp.array([1.0]))
        self.v = nnx.Param(jnp.array([1.0]))

    def __call__(
        self, weight: jax.Array, training: bool = True
    ) -> tuple[jax.Array, jax.Array]:
        """Estimate spectral norm using power iteration.

        Args:
            weight: Weight matrix of shape (..., out_features, in_features)
            training: Whether in training mode (updates u, v vectors)

        Returns:
            Tuple of (spectral_norm, normalized_weight)
        """
        # Reshape weight to 2D matrix for SVD computation
        original_shape = weight.shape
        if len(original_shape) > 2:
            weight_2d = weight.reshape(-1, original_shape[-1])
        else:
            weight_2d = weight

        height, width = weight_2d.shape

        # Initialize u and v vectors if needed (JIT-compatible)
        # Instead of checking _initialized, we'll reinitialize if dimensions don't match
        # This avoids boolean conversion errors during JIT compilation

        u, v = self.u[...], self.v[...]

        # Ensure u and v have correct dimensions
        if u.shape[0] != height:
            u = jax.random.normal(nnx.Rngs(0).default(), (height,)) / jnp.sqrt(height)
        if v.shape[0] != width:
            v = jax.random.normal(nnx.Rngs(1).default(), (width,)) / jnp.sqrt(width)

        # Power iteration
        for _ in range(self.num_iterations):
            # v = W^T u / ||W^T u||
            v = weight_2d.T @ u
            v = v / (jnp.linalg.norm(v) + self.eps)

            # u = W v / ||W v||
            u = weight_2d @ v
            u = u / (jnp.linalg.norm(u) + self.eps)

        # Update stored vectors if training and not in JAX transformation
        if training:
            try:
                self.u[...] = u
                self.v[...] = v
            except Exception:
                # Skip updates if inside JAX transformation
                pass

        # Compute spectral norm: sigma = u^T W v
        spectral_norm = jnp.dot(u, weight_2d @ v)

        # Normalize weight by spectral norm
        normalized_weight = weight / (spectral_norm + self.eps)
        normalized_weight = normalized_weight.reshape(original_shape)

        return spectral_norm, normalized_weight


class SpectralNorm(nnx.Module):
    """Spectral normalization wrapper that can be applied to any linear layer.

    This wrapper normalizes the spectral norm of weight matrices to improve
    training stability and control the Lipschitz constant of neural networks.
    """

    def __init__(
        self,
        layer: nnx.Module,
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral normalization wrapper.

        Args:
            layer: The layer to apply spectral normalization to
            power_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators
        """
        self.layer = layer
        self.power_iter = PowerIteration(
            num_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, training: bool = True, **kwargs) -> jax.Array:
        """Apply spectral normalization and forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode
            **kwargs: Additional arguments passed to the layer

        Returns:
            Output tensor from the spectrally normalized layer
        """
        # Get the weight parameter
        if hasattr(self.layer, "kernel"):
            weight_name = "kernel"
        elif hasattr(self.layer, "weight"):
            weight_name = "weight"
        else:
            raise ValueError("Layer must have 'kernel' or 'weight' parameter")

        original_weight = getattr(self.layer, weight_name)[...]

        # Apply spectral normalization
        _, normalized_weight = self.power_iter(original_weight, training)

        # Temporarily set normalized weight
        original_value = getattr(self.layer, weight_name)[...]
        getattr(self.layer, weight_name)[...] = normalized_weight

        try:
            # Forward pass with normalized weight
            output = self.layer(x, **kwargs)  # type: ignore[operator]
        finally:
            # Restore original weight
            getattr(self.layer, weight_name)[...] = original_value

        return output


class SpectralLinear(nnx.Module):
    """Linear layer with built-in spectral normalization.

    This is a convenience class that combines a linear layer with spectral
    normalization for better performance and cleaner code.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral normalized linear layer.

        Args:
            in_features: Number of input features
            out_features: Number of output features
            use_bias: Whether to use bias
            power_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators
        """
        self.linear = nnx.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=use_bias,
            rngs=rngs,
        )

        self.power_iter = PowerIteration(
            num_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass with spectral normalization.

        Args:
            x: Input tensor of shape (..., in_features)
            training: Whether in training mode

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply spectral normalization to kernel
        _, normalized_kernel = self.power_iter(self.linear.kernel[...], training)

        # Manually compute linear transformation with normalized kernel
        # Flax Linear kernel shape is (in_features, out_features)
        y = x @ normalized_kernel

        # Add bias if present
        if self.linear.bias is not None:
            bias = self.linear.bias[...]
            y = y + bias

        return y


class SpectralNormalizedConv(nnx.Module):
    """Convolution layer with built-in spectral normalization.

    This applies spectral normalization to convolutional layers, which is
    particularly useful for neural operators working with spatial data.

    This class applies spectral normalization to regular spatial convolution
    operations to improve training stability.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int | Sequence[int],
        strides: int | Sequence[int] = 1,
        padding: str | int | Sequence[int] = "SAME",
        use_bias: bool = True,
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral normalized convolution layer.

        Args:
            in_features: Number of input channels
            out_features: Number of output channels
            kernel_size: Size of convolution kernel
            strides: Convolution strides
            padding: Padding mode
            use_bias: Whether to use bias
            power_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators
        """
        self.conv = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=use_bias,
            rngs=rngs,
        )

        self.power_iter = PowerIteration(
            num_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
        """Forward pass with spectral normalization.

        Args:
            x: Input tensor
            training: Whether in training mode

        Returns:
            Output tensor from spectrally normalized convolution
        """
        # Apply spectral normalization to kernel
        _, normalized_kernel = self.power_iter(self.conv.kernel[...], training)

        # Temporarily set normalized kernel
        original_kernel = self.conv.kernel[...]
        self.conv.kernel[...] = normalized_kernel

        try:
            # Forward pass with normalized kernel
            output = self.conv(x)
        finally:
            # Restore original kernel
            self.conv.kernel[...] = original_kernel

        return output


class AdaptiveSpectralNorm(nnx.Module):
    """Adaptive spectral normalization with learnable normalization bounds.

    This variant allows the network to learn the appropriate spectral norm
    bounds rather than fixing them to 1, which can be more flexible for
    different layers in neural operators.
    """

    def __init__(
        self,
        layer: nnx.Module,
        initial_bound: float = 1.0,
        learnable_bound: bool = True,
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize adaptive spectral normalization.

        Args:
            layer: The layer to apply spectral normalization to
            initial_bound: Initial spectral norm bound
            learnable_bound: Whether the bound is learnable
            power_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators
        """
        self.layer = layer
        self.learnable_bound = learnable_bound

        # Use Param for both cases, but only train it if learnable_bound is True
        self.bound = nnx.Param(jnp.array(initial_bound))

        self.power_iter = PowerIteration(
            num_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, training: bool = True, **kwargs) -> jax.Array:
        """Apply adaptive spectral normalization and forward pass.

        Args:
            x: Input tensor
            training: Whether in training mode
            **kwargs: Additional arguments passed to the layer

        Returns:
            Output tensor from the adaptively normalized layer
        """
        # Get the weight parameter
        if hasattr(self.layer, "kernel"):
            weight_name = "kernel"
        elif hasattr(self.layer, "weight"):
            weight_name = "weight"
        else:
            raise ValueError("Layer must have 'kernel' or 'weight' parameter")

        original_weight = getattr(self.layer, weight_name)[...]

        # Apply spectral normalization with adaptive bound
        _, normalized_weight = self.power_iter(original_weight, training)

        # Scale by learnable bound
        bound_value = self.bound[...]
        adaptive_weight = normalized_weight * jnp.maximum(
            bound_value, 0.1
        )  # Prevent collapse

        # Temporarily set adaptive weight
        original_value = getattr(self.layer, weight_name)[...]
        getattr(self.layer, weight_name)[...] = adaptive_weight

        try:
            # Forward pass with adaptive weight
            output = self.layer(x, **kwargs)  # type: ignore[operator]
        finally:
            # Restore original weight
            getattr(self.layer, weight_name)[...] = original_value

        return output


class SpectralMultiHeadAttention(nnx.Module):
    """Multi-head attention with spectral normalization for neural operators.

    This applies spectral normalization to all linear transformations in
    multi-head attention, which is particularly useful for transformer-based
    neural operators.
    """

    def __init__(
        self,
        num_heads: int,
        in_features: int,
        qkv_features: int | None = None,
        out_features: int | None = None,
        power_iterations: int = 1,
        eps: float = 1e-12,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize spectral normalized multi-head attention.

        Args:
            num_heads: Number of attention heads
            in_features: Input feature dimension
            qkv_features: Query/Key/Value feature dimension
            out_features: Output feature dimension
            power_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            rngs: Random number generators
        """
        if qkv_features is None:
            qkv_features = in_features
        if out_features is None:
            out_features = in_features

        self.num_heads = num_heads
        self.qkv_features = qkv_features
        self.head_dim = qkv_features // num_heads

        if qkv_features % num_heads != 0:
            raise ValueError("qkv_features must be divisible by num_heads")

        # Spectral normalized linear projections
        self.query_proj = SpectralLinear(
            in_features,
            qkv_features,
            power_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )
        self.key_proj = SpectralLinear(
            in_features,
            qkv_features,
            power_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )
        self.value_proj = SpectralLinear(
            in_features,
            qkv_features,
            power_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )
        self.out_proj = SpectralLinear(
            qkv_features,
            out_features,
            power_iterations=power_iterations,
            eps=eps,
            rngs=rngs,
        )

    def __call__(
        self,
        x: jax.Array,
        mask: jax.Array | None = None,
        training: bool = True,
    ) -> jax.Array:
        """Apply spectral normalized multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, features)
            mask: Optional attention mask
            training: Whether in training mode

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with spectral normalization
        q = self.query_proj(x, training=training)
        k = self.key_proj(x, training=training)
        v = self.value_proj(x, training=training)

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose to (batch, num_heads, seq_len, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # Scaled dot-product attention
        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_weights = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply mask if provided
        if mask is not None:
            attn_weights = jnp.where(mask, attn_weights, -1e9)

        attn_weights = jax.nn.softmax(attn_weights, axis=-1)

        # Apply attention to values
        out = jnp.einsum("bhqk,bhvd->bhqd", attn_weights, v)

        # Reshape back to (batch, seq_len, qkv_features)
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, self.qkv_features)

        # Final projection with spectral normalization
        return self.out_proj(out, training=training)


# SpectralNeuralOperator moved to FNO spectral module where it belongs architecturally
# Import it from the correct location if needed:
# from ..fno.spectral import SpectralNeuralOperator, create_spectral_neural_operator


def spectral_norm_summary(
    model: nnx.Module,
) -> dict[str, float | int | str]:
    """Compute summary statistics of spectral norms in a model.

    Args:
        model: Model containing spectral normalization layers

    Returns:
        Dictionary with spectral norm statistics
    """
    spectral_norms = []

    def collect_spectral_norms(obj):
        """Recursively collect spectral norms from any object."""
        # Check for spectral normalization layers
        spectral_types = (
            SpectralNorm,
            SpectralLinear,
            SpectralNormalizedConv,
            AdaptiveSpectralNorm,
        )

        if isinstance(obj, spectral_types) and hasattr(obj, "power_iter"):
            # Extract weight from layer - simplified logic
            weight = None
            for attr_name in ["linear", "conv", "layer"]:
                if hasattr(obj, attr_name):
                    layer = getattr(obj, attr_name)
                    weight = getattr(layer, "kernel", getattr(layer, "weight", None))
                    if weight is not None:
                        weight = weight.value
                        break

            if weight is not None:
                spectral_norm, _ = obj.power_iter(weight, training=False)
                spectral_norms.append(float(spectral_norm))

        # Recursively process containers and modules
        if isinstance(obj, (list, tuple)):
            for item in obj:
                collect_spectral_norms(item)
        elif hasattr(obj, "__dict__"):
            for attr_name in dir(obj):
                if not attr_name.startswith("_"):
                    try:
                        attr = getattr(obj, attr_name)
                        if isinstance(attr, (nnx.Module, list)):
                            collect_spectral_norms(attr)
                    except (AttributeError, TypeError):
                        pass

    # Start collection from the model
    collect_spectral_norms(model)

    if not spectral_norms:
        return {
            "message": "No spectral normalization layers found",
            "num_layers": 0,
        }

    return {
        "num_layers": len(spectral_norms),
        "mean_spectral_norm": float(jnp.mean(jnp.array(spectral_norms))),
        "max_spectral_norm": float(jnp.max(jnp.array(spectral_norms))),
        "min_spectral_norm": float(jnp.min(jnp.array(spectral_norms))),
        "std_spectral_norm": float(jnp.std(jnp.array(spectral_norms))),
    }
