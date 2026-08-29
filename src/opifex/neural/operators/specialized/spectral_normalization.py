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
from typing import Any

import flax.errors
import jax
import jax.numpy as jnp
from flax import nnx


def _matrix_shape(weight_shape: tuple[int, ...]) -> tuple[int, int]:
    """Return the 2D shape a weight is flattened to before power iteration.

    Mirrors the reshape in :meth:`PowerIteration.__call__`, so the stored vectors
    are sized exactly as that method expects them.

    Args:
        weight_shape: Shape of the weight to be normalized.

    Returns:
        Tuple of (height, width) of the flattened matrix.

    Raises:
        ValueError: If the weight has fewer than two dimensions.
    """
    if len(weight_shape) < 2:
        raise ValueError(f"Spectral normalization needs a matrix, got shape {weight_shape}")
    width = weight_shape[-1]
    height = 1
    for dim in weight_shape[:-1]:
        height *= dim
    return height, width


def _layer_weight_shape(layer: Any) -> tuple[int, ...] | None:
    """Return the shape of a layer's kernel/weight, or None when it exposes neither.

    Args:
        layer: The layer that will be spectrally normalized.

    Returns:
        The weight shape, or None when the layer has no usable kernel or weight.
    """
    weight = getattr(layer, "kernel", getattr(layer, "weight", None))
    value = getattr(weight, "value", None)
    shape = getattr(value, "shape", None)
    return tuple(shape) if shape is not None and len(shape) >= 2 else None


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
        use_running_average: bool = False,
        *,
        weight_shape: tuple[int, ...] | None = None,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize power iteration algorithm.

        Args:
            num_iterations: Number of power iteration steps
            eps: Small epsilon for numerical stability
            use_running_average: When True, reuse the stored vectors instead of
                writing the re-estimated ones back. ``nnx.Module.train()`` and
                ``nnx.Module.eval()`` set this recursively, as they do for
                ``nnx.BatchNorm``; the default matches nnx, which constructs
                modules in training mode.
            weight_shape: Shape of the weight this will normalize. Given it, the
                vectors start at the size they will be used at, so they persist
                across calls and the estimate sharpens. Omit it only when the
                weight cannot be known at construction, such as a wrapper around
                a layer that exposes no kernel.
            rngs: Random number generators for initialization
        """
        self.num_iterations = num_iterations
        self.eps = eps
        self.use_running_average = use_running_average

        # Size the vectors for the weight they will iterate on. Left as scalar
        # placeholders they never match, so __call__ re-draws them on every pass and
        # each estimate starts from a fresh random vector instead of the previous one.
        # Power iteration only sharpens by carrying its vectors forward, which is what
        # flax.nnx.SpectralNorm means by updating u "over time".
        if weight_shape is not None:
            height, width = _matrix_shape(weight_shape)
            self.u = nnx.Param(jax.random.normal(rngs.default(), (height,)) / jnp.sqrt(height))
            self.v = nnx.Param(jax.random.normal(rngs.default(), (width,)) / jnp.sqrt(width))
        else:
            self.u = nnx.Param(jnp.array([1.0]))
            self.v = nnx.Param(jnp.array([1.0]))

    def set_view(self, use_running_average: bool | None = None) -> None:
        """Class method used by ``nnx.view``.

        Args:
            use_running_average: if True, the stored vectors are reused instead
                of being re-estimated and written back.
        """
        if use_running_average is not None:
            self.use_running_average = use_running_average

    def __call__(self, weight: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Estimate spectral norm using power iteration.

        Whether the re-estimated vectors are written back is decided by
        ``use_running_average``, which ``train()`` and ``eval()`` set.

        Args:
            weight: Weight matrix of shape (..., out_features, in_features)

        Returns:
            Tuple of (spectral_norm, normalized_weight)
        """
        # Reshape weight to 2D matrix for SVD computation
        original_shape = weight.shape
        weight_2d = weight.reshape(-1, original_shape[-1]) if len(original_shape) > 2 else weight

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

        # Write the re-estimated vectors back unless the stored ones are in use.
        if not self.use_running_average:
            try:
                # Assign through .value, not u[...]: the vectors start as shape-(1,)
                # placeholders and are re-drawn above to (height,) / (width,), so the write
                # has to REPLACE the value. Variable.__setitem__ routes through
                # `.at[index].set()`, an in-place scatter that keeps the existing shape and
                # rejects the new one with ValueError.
                self.u.value = u
                self.v.value = v
            except (
                TypeError,
                jax.errors.TracerArrayConversionError,
                flax.errors.TraceContextError,
            ):
                # State mutation is invalid inside jit/grad traces; callers update outside
                # the transform when this is hit. flax raises TraceContextError for a write
                # to a Variable owned by an outer trace.
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
    ) -> None:
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
            weight_shape=_layer_weight_shape(self.layer),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply spectral normalization and forward pass.

        Args:
            x: Input tensor
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
        _, normalized_weight = self.power_iter(original_weight)

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
    ) -> None:
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
            weight_shape=_layer_weight_shape(self.linear),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with spectral normalization.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)
        """
        # Apply spectral normalization to kernel
        _, normalized_kernel = self.power_iter(self.linear.kernel[...])

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
    ) -> None:
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
            weight_shape=_layer_weight_shape(self.conv),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Forward pass with spectral normalization.

        Args:
            x: Input tensor

        Returns:
            Output tensor from spectrally normalized convolution
        """
        # Apply spectral normalization to kernel
        _, normalized_kernel = self.power_iter(self.conv.kernel[...])

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
    ) -> None:
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
            weight_shape=_layer_weight_shape(self.layer),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        """Apply adaptive spectral normalization and forward pass.

        Args:
            x: Input tensor
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
        _, normalized_weight = self.power_iter(original_weight)

        # Scale by learnable bound
        bound_value = self.bound[...]
        adaptive_weight = normalized_weight * jnp.maximum(bound_value, 0.1)  # Prevent collapse

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
    ) -> None:
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
    ) -> jax.Array:
        """Apply spectral normalized multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, features)
            mask: Optional attention mask

        Returns:
            Output tensor of shape (batch, seq_len, out_features)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with spectral normalization
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

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
        return self.out_proj(out)


def _extract_spectral_layer_weight(obj: Any) -> Any:
    """Return the kernel/weight value of a spectral layer's inner module, or None."""
    for attr_name in ("linear", "conv", "layer"):
        if hasattr(obj, attr_name):
            layer = getattr(obj, attr_name)
            weight = getattr(layer, "kernel", getattr(layer, "weight", None))
            if weight is not None:
                return weight.value
    return None


def _spectral_norm_children(obj: Any) -> list[Any]:
    """Return the child objects of obj to recurse into for spectral-norm collection."""
    if isinstance(obj, list | tuple):
        return list(obj)
    if not hasattr(obj, "__dict__"):
        return []
    children: list[Any] = []
    for attr_name in dir(obj):
        if attr_name.startswith("_"):
            continue
        try:
            attr = getattr(obj, attr_name)
        except (AttributeError, TypeError):
            continue
        if isinstance(attr, nnx.Module | list | tuple):
            children.append(attr)
    return children


def _collect_spectral_norms(obj: Any, spectral_norms: list[float]) -> None:
    """Recursively append spectral norms of spectral layers reachable from obj."""
    spectral_types = (
        SpectralNorm,
        SpectralLinear,
        SpectralNormalizedConv,
        AdaptiveSpectralNorm,
    )
    if isinstance(obj, spectral_types) and hasattr(obj, "power_iter"):
        weight = _extract_spectral_layer_weight(obj)
        if weight is not None:
            # A diagnostic must not mutate: nnx.view yields a copy with the stored
            # vectors in use, sharing the arrays and leaving `obj` untouched.
            frozen = nnx.view(obj.power_iter, use_running_average=True)
            spectral_norm, _ = frozen(weight)
            spectral_norms.append(float(spectral_norm))

    for child in _spectral_norm_children(obj):
        _collect_spectral_norms(child, spectral_norms)


def spectral_norm_summary(
    model: nnx.Module,
) -> dict[str, float | int | str]:
    """Compute summary statistics of spectral norms in a model.

    Args:
        model: Model containing spectral normalization layers

    Returns:
        Dictionary with spectral norm statistics
    """
    spectral_norms: list[float] = []
    _collect_spectral_norms(model, spectral_norms)

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
