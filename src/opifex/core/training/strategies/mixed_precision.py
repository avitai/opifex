#!/usr/bin/env python3
"""
Mixed Precision Training Infrastructure for Opifex JAX Neural Operators.

This module provides comprehensive mixed precision training support to enable
TensorCore utilization and improve performance based on profiling recommendations.

Key Features:
- Automatic mixed precision with bfloat16/float16 computation
- Float32 parameter storage for numerical stability
- Loss scaling and overflow detection
- Hardware-specific optimizations (GPU TensorCore, TPU bfloat16)
- Integration with existing Opifex training infrastructure
"""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.core.training.monitoring.metrics import TrainingMetrics
from opifex.training.basic_trainer import BasicTrainer, TrainingConfig


class MixedPrecisionConfig:
    """Configuration for mixed precision training."""

    def __init__(
        self,
        compute_dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.float32,
        loss_scale: float = 2**15,
        dynamic_loss_scaling: bool = True,
        loss_scale_factor: float = 2.0,
        min_loss_scale: float = 1.0,
        max_loss_scale: float = 2**24,
        overflow_check_frequency: int = 100,
        enable_tensorcore_alignment: bool = True,
    ):
        """Initialize mixed precision configuration.

        Args:
            compute_dtype: Data type for computations (bfloat16/float16)
            param_dtype: Data type for parameters (float32 for stability)
            loss_scale: Initial loss scaling factor
            dynamic_loss_scaling: Whether to use dynamic loss scaling
            loss_scale_factor: Factor to adjust loss scale
            min_loss_scale: Minimum loss scale value
            max_loss_scale: Maximum loss scale value
            overflow_check_frequency: How often to check for overflow
            enable_tensorcore_alignment: Align tensors for TensorCore usage
        """
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self.loss_scale = loss_scale
        self.dynamic_loss_scaling = dynamic_loss_scaling
        self.loss_scale_factor = loss_scale_factor
        self.min_loss_scale = min_loss_scale
        self.max_loss_scale = max_loss_scale
        self.overflow_check_frequency = overflow_check_frequency
        self.enable_tensorcore_alignment = enable_tensorcore_alignment

        # Hardware-specific optimizations
        self.backend = jax.default_backend()
        if self.backend == "gpu":
            # Use bfloat16 for GPU TensorCore optimization
            self.compute_dtype = jnp.bfloat16
        elif self.backend == "tpu":
            # TPU natively supports bfloat16
            self.compute_dtype = jnp.bfloat16
        else:
            # CPU fallback to float16
            self.compute_dtype = jnp.float16


class MixedPrecisionState:
    """State for mixed precision training."""

    def __init__(self, loss_scale: float, overflow_count: int = 0):
        self.loss_scale = loss_scale
        self.overflow_count = overflow_count
        self.step_count = 0


def create_mixed_precision_policy(config: MixedPrecisionConfig) -> Callable:
    """Create mixed precision policy function."""

    def mixed_precision_policy(x: jax.Array) -> jax.Array:
        """Apply mixed precision policy to input tensor."""
        if x.dtype == config.param_dtype:
            # Convert parameters to compute dtype for forward pass
            return x.astype(config.compute_dtype)
        return x

    return mixed_precision_policy


def align_for_tensorcore(x: jax.Array, alignment: int = 8) -> jax.Array:
    """Align tensor dimensions for TensorCore utilization.

    Args:
        x: Input tensor
        alignment: Alignment requirement (8, 16, or 32)

    Returns:
        Aligned tensor with padded dimensions
    """
    if not isinstance(alignment, int) or alignment not in [8, 16, 32]:
        raise ValueError("Alignment must be 8, 16, or 32")

    # Pad last two dimensions to be multiples of alignment
    shape = x.shape
    if len(shape) < 2:
        return x

    # Calculate padding for last two dimensions
    pad_h = (alignment - (shape[-2] % alignment)) % alignment
    pad_w = (alignment - (shape[-1] % alignment)) % alignment

    if pad_h == 0 and pad_w == 0:
        return x

    # Create padding specification
    pad_width = [(0, 0)] * (len(shape) - 2) + [(0, pad_h), (0, pad_w)]

    return jnp.pad(x, pad_width, mode="constant", constant_values=0)


def check_for_overflow(grads: Any) -> bool:
    """Check if gradients contain NaN or Inf values."""

    def is_finite(x):
        return jnp.all(jnp.isfinite(x))

    finite_checks = jax.tree.map(is_finite, grads)
    return not jax.tree.reduce(lambda a, b: a and b, finite_checks, True)


def scale_gradients(grads: Any, loss_scale: float) -> Any:
    """Scale gradients by loss scale factor."""
    return jax.tree.map(lambda g: g / loss_scale, grads)


def update_loss_scale(
    mp_state: MixedPrecisionState,
    has_overflow: bool,
    config: MixedPrecisionConfig,
) -> MixedPrecisionState:
    """Update loss scale based on overflow detection."""
    if not config.dynamic_loss_scaling:
        return mp_state

    if has_overflow:
        # Reduce loss scale on overflow
        new_loss_scale = max(
            mp_state.loss_scale / config.loss_scale_factor, config.min_loss_scale
        )
        new_overflow_count = mp_state.overflow_count + 1
    else:
        # Increase loss scale periodically if no overflow
        if mp_state.step_count % config.overflow_check_frequency == 0:
            new_loss_scale = min(
                mp_state.loss_scale * config.loss_scale_factor, config.max_loss_scale
            )
        else:
            new_loss_scale = mp_state.loss_scale
        new_overflow_count = 0

    return MixedPrecisionState(
        loss_scale=new_loss_scale,
        overflow_count=new_overflow_count,
    )


class MixedPrecisionTrainer(BasicTrainer):
    """Mixed precision trainer for neural operators."""

    def __init__(
        self,
        model: nnx.Module,
        config: TrainingConfig,
        mp_config: MixedPrecisionConfig | None = None,
    ):
        """Initialize mixed precision trainer.

        Args:
            model: Neural operator model
            config: Training configuration
            mp_config: Mixed precision configuration
        """
        super().__init__(model, config)

        self.mp_config = mp_config or MixedPrecisionConfig()
        self.mp_state = MixedPrecisionState(self.mp_config.loss_scale)

        # Create mixed precision policy
        self.precision_policy = create_mixed_precision_policy(self.mp_config)

        # Convert model to mixed precision
        self._convert_model_to_mixed_precision()

        import logging

        logger = logging.getLogger(__name__)
        logger.info("🔧 Mixed Precision Training Initialized:")
        logger.info(f"   • Compute dtype: {self.mp_config.compute_dtype}")
        logger.info(f"   • Parameter dtype: {self.mp_config.param_dtype}")
        logger.info(f"   • Backend: {self.mp_config.backend}")
        logger.info(
            f"   • TensorCore alignment: {self.mp_config.enable_tensorcore_alignment}"
        )

    def _convert_model_to_mixed_precision(self):
        """Convert model parameters to mixed precision format."""
        # Keep parameters in float32 for stability
        # Computations will be done in lower precision
        # Model parameters stay in float32

    def _prepare_batch(
        self, batch: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        """Prepare batch with mixed precision and TensorCore alignment."""
        x, y = batch

        # Convert to compute dtype
        x = x.astype(self.mp_config.compute_dtype)
        y = y.astype(self.mp_config.compute_dtype)

        # Align for TensorCore if enabled
        if self.mp_config.enable_tensorcore_alignment:
            x = align_for_tensorcore(x)
            y = align_for_tensorcore(y)

        return x, y

    def _compute_loss_and_grads(
        self,
        model: nnx.Module,
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[float, Any]:
        """Compute loss and gradients with mixed precision."""
        x, y = self._prepare_batch(batch)

        def loss_fn(model_state):
            # Apply mixed precision policy to model
            model_mp = jax.tree.map(self.precision_policy, model_state)

            # Forward pass in mixed precision
            pred = model_mp(x)

            # Compute loss (scale for mixed precision)
            loss = jnp.mean((pred - y) ** 2)
            scaled_loss = loss * self.mp_state.loss_scale

            return scaled_loss, loss

        # Compute gradients
        (_, actual_loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            nnx.state(model)
        )

        # Check for overflow
        has_overflow = check_for_overflow(grads)

        if has_overflow:
            # Skip update on overflow
            grads = jax.tree.map(jnp.zeros_like, grads)
        else:
            # Unscale gradients
            grads = scale_gradients(grads, self.mp_state.loss_scale)

        # Update loss scale
        self.mp_state = update_loss_scale(self.mp_state, has_overflow, self.mp_config)
        self.mp_state.step_count += 1

        return actual_loss, grads

    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        batch: tuple[jax.Array, jax.Array],
    ) -> tuple[float, TrainingMetrics]:
        """Single training step with mixed precision."""
        # Compute loss and gradients
        loss, grads = self._compute_loss_and_grads(model, batch)

        # Update model parameters (in float32)
        optimizer.update(model, grads)

        # Create metrics
        metrics = TrainingMetrics()
        metrics.train_losses.append(float(loss))

        return loss, metrics

    def get_mixed_precision_stats(self) -> dict[str, Any]:
        """Get mixed precision training statistics."""
        return {
            "loss_scale": self.mp_state.loss_scale,
            "overflow_count": self.mp_state.overflow_count,
            "step_count": self.mp_state.step_count,
            "compute_dtype": str(self.mp_config.compute_dtype),
            "param_dtype": str(self.mp_config.param_dtype),
            "backend": self.mp_config.backend,
        }


# Utility functions for mixed precision optimization
def create_mixed_precision_optimizer(
    learning_rate: float,
    config: MixedPrecisionConfig,
) -> optax.GradientTransformation:
    """Create optimizer optimized for mixed precision training."""
    _ = config  # Reserved for future use

    # Use AdamW with mixed precision optimizations
    base_optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=1e-4,
        eps=1e-4,  # Larger epsilon for numerical stability
    )

    # Add gradient clipping for stability
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        base_optimizer,
    )


def optimize_batch_size_for_hardware(
    base_batch_size: int,
    backend: str | None = None,
) -> int:
    """Optimize batch size for hardware TensorCore utilization."""
    if backend is None:
        backend = jax.default_backend()

    if backend == "gpu":
        # GPU TensorCore optimization - prefer multiples of 8, 16, 32
        # Based on profiling results showing optimal batch size > 200-298
        optimal_sizes = [256, 512, 1024]
        for size in optimal_sizes:
            if size >= base_batch_size:
                return size
        return max(optimal_sizes)

    if backend == "tpu":
        # TPU optimization - prefer multiples of 8
        return ((base_batch_size + 7) // 8) * 8

    # CPU - no specific alignment requirements
    return base_batch_size


# Export main components
__all__ = [
    "MixedPrecisionConfig",
    "MixedPrecisionState",
    "MixedPrecisionTrainer",
    "align_for_tensorcore",
    "create_mixed_precision_optimizer",
    "create_mixed_precision_policy",
    "optimize_batch_size_for_hardware",
]
