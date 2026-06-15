"""Optimizer creation and configuration for Opifex framework.

This module provides a centralized, DRY approach to creating and configuring
Optax optimizers with common patterns like learning rate schedules and
gradient clipping.

Following strict TDD principles - all functions are implemented to pass
the tests defined in test_optimizers.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import optax


@dataclass(frozen=True, slots=True, kw_only=True)
class OptimizerConfig:
    """Configuration for optimizer creation.

    This dataclass centralizes all optimizer configuration options,
    eliminating the need for scattered parameter dictionaries across
    the codebase.
    """

    # Basic optimizer settings
    optimizer_type: str = "adam"
    learning_rate: float = 1e-3

    # Adam/AdamW specific
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0

    # SGD specific
    momentum: float = 0.0

    # RMSprop specific
    decay: float = 0.9

    # Learning rate schedule
    schedule_type: str | None = None
    decay_steps: int | None = None
    alpha: float = 0.1  # For cosine decay
    transition_steps: int | None = None
    decay_rate: float = 0.96  # For exponential decay
    end_value: float | None = None  # For linear decay
    boundaries_and_values: tuple[list[int], list[float]] | None = None  # For step
    peak_value: float | None = None  # For warmup_cosine
    warmup_steps: int | None = None  # For warmup_cosine

    # Gradient clipping
    gradient_clip: float | None = None
    clip_type: str = "by_global_norm"  # or "by_value"
    max_value: float | None = None  # For clip_by_value


def create_adam(
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
) -> optax.GradientTransformation:
    """Create Adam optimizer.

    Args:
        learning_rate: Learning rate
        b1: Exponential decay rate for the first moment estimates
        b2: Exponential decay rate for the second moment estimates
        eps: Small constant for numerical stability

    Returns:
        Adam optimizer
    """
    return optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)


def create_adamw(
    learning_rate: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Create AdamW optimizer with weight decay.

    Args:
        learning_rate: Learning rate
        b1: Exponential decay rate for the first moment estimates
        b2: Exponential decay rate for the second moment estimates
        eps: Small constant for numerical stability
        weight_decay: Weight decay coefficient

    Returns:
        AdamW optimizer
    """
    return optax.adamw(
        learning_rate=learning_rate, b1=b1, b2=b2, eps=eps, weight_decay=weight_decay
    )


def create_sgd(
    learning_rate: float = 1e-2,
    momentum: float = 0.0,
) -> optax.GradientTransformation:
    """Create SGD optimizer with optional momentum.

    Args:
        learning_rate: Learning rate
        momentum: Momentum coefficient

    Returns:
        SGD optimizer
    """
    return optax.sgd(learning_rate=learning_rate, momentum=momentum)


def create_rmsprop(
    learning_rate: float = 1e-3,
    eps: float = 1e-8,
    decay: float = 0.9,
) -> optax.GradientTransformation:
    """Create RMSprop optimizer.

    Args:
        learning_rate: Learning rate
        eps: Small constant for numerical stability
        decay: Decay rate for moving average

    Returns:
        RMSprop optimizer
    """
    return optax.rmsprop(learning_rate=learning_rate, decay=decay, eps=eps)


def _constant_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Constant learning-rate schedule."""
    return optax.constant_schedule(config.learning_rate)


def _cosine_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Cosine-decay schedule (``decay_steps`` defaults to 1000)."""
    decay_steps = config.decay_steps if config.decay_steps is not None else 1000
    return optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=decay_steps, alpha=config.alpha
    )


def _exponential_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Exponential-decay schedule (``transition_steps`` defaults to 1000)."""
    transition_steps = config.transition_steps if config.transition_steps is not None else 1000
    return optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=transition_steps,
        decay_rate=config.decay_rate,
    )


def _linear_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Linear schedule to ``end_value`` (defaults to 10% of the initial value)."""
    transition_steps = config.transition_steps if config.transition_steps is not None else 1000
    end_value = config.end_value if config.end_value is not None else config.learning_rate * 0.1
    return optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=end_value,
        transition_steps=transition_steps,
    )


def _step_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Piecewise-constant (step) schedule with a default decaying staircase."""
    if config.boundaries_and_values is None:
        boundaries, values = (
            [100, 200],
            [config.learning_rate, config.learning_rate * 0.1, config.learning_rate * 0.01],
        )
    else:
        boundaries, values = config.boundaries_and_values
    scales = (
        {
            boundary: values[i + 1] / values[i]
            for i, boundary in enumerate(boundaries)
            if i + 1 < len(values)
        }
        if len(values) > 1
        else {}
    )
    return optax.piecewise_constant_schedule(
        init_value=values[0] if values else config.learning_rate,
        boundaries_and_scales=scales,
    )


def _warmup_cosine_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Linear warmup followed by cosine decay."""
    return optax.warmup_cosine_decay_schedule(
        init_value=config.learning_rate,
        peak_value=config.peak_value if config.peak_value is not None else config.learning_rate,
        warmup_steps=config.warmup_steps if config.warmup_steps is not None else 100,
        decay_steps=config.decay_steps if config.decay_steps is not None else 1000,
    )


def _as_float32_schedule(schedule: optax.Schedule) -> optax.Schedule:
    """Wrap a schedule so it returns float32 (stable optax updates under x64)."""

    def wrapped_schedule(count):
        return jnp.asarray(schedule(count), dtype=jnp.float32)

    return wrapped_schedule


_SCHEDULE_BUILDERS = {
    "constant": _constant_schedule,
    "cosine": _cosine_schedule,
    "exponential": _exponential_schedule,
    "linear": _linear_schedule,
    "step": _step_schedule,
    "warmup_cosine": _warmup_cosine_schedule,
}


def create_schedule(config: OptimizerConfig) -> optax.Schedule:
    """Create a learning-rate schedule from an optimizer configuration.

    The schedule kind and its parameters are read from ``config`` (see
    :class:`OptimizerConfig`); ``config.learning_rate`` is the schedule's
    initial value. The output is cast to float32 for stable optax updates
    under x64.

    Args:
        config: Optimizer configuration carrying ``schedule_type`` and the
            associated schedule parameters.

    Returns:
        A float32 ``optax.Schedule``.

    Raises:
        ValueError: If ``config.schedule_type`` is unknown.
    """
    builder = _SCHEDULE_BUILDERS.get(config.schedule_type or "")
    if builder is None:
        raise ValueError(f"Unknown schedule type: {config.schedule_type}")
    return _as_float32_schedule(builder(config))


def with_gradient_clipping(
    optimizer: optax.GradientTransformation,
    max_norm: float | None = None,
    clip_type: str = "by_global_norm",
    max_value: float | None = None,
) -> optax.GradientTransformation:
    """Add gradient clipping to an optimizer.

    Args:
        optimizer: Base optimizer
        max_norm: Maximum gradient norm (for global norm clipping)
        clip_type: Type of clipping ("by_global_norm" or "by_value")
        max_value: Maximum absolute value (for value clipping)

    Returns:
        Optimizer with gradient clipping
    """
    if clip_type == "by_global_norm":
        if max_norm is None:
            max_norm = 1.0
        return optax.chain(optax.clip_by_global_norm(max_norm), optimizer)

    if clip_type == "by_value":
        if max_value is None:
            max_value = 1.0
        return optax.chain(optax.clip(max_value), optimizer)

    # Default to global norm clipping
    if max_norm is None:
        max_norm = 1.0
    return optax.chain(optax.clip_by_global_norm(max_norm), optimizer)


def with_schedule(
    optimizer: optax.GradientTransformation,
    schedule: optax.Schedule,
) -> optax.GradientTransformation:
    """Apply learning rate schedule to an optimizer.

    Args:
        optimizer: Base optimizer
        schedule: Learning rate schedule

    Returns:
        Optimizer with learning rate schedule
    """
    return optax.chain(optax.scale_by_schedule(schedule), optimizer)


def create_optimizer(config: OptimizerConfig) -> optax.GradientTransformation:
    """Create optimizer from configuration.

    This is the main entry point for creating optimizers. It handles:
    1. Creating the base optimizer
    2. Adding gradient clipping if specified
    3. Adding learning rate schedule if specified

    Args:
        config: Optimizer configuration

    Returns:
        Configured optimizer

    Raises:
        ValueError: If optimizer_type is unknown
    """
    # Create base optimizer
    if config.optimizer_type == "adam":
        base_optimizer = create_adam(
            learning_rate=config.learning_rate,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
        )
    elif config.optimizer_type == "adamw":
        base_optimizer = create_adamw(
            learning_rate=config.learning_rate,
            b1=config.b1,
            b2=config.b2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer_type == "sgd":
        base_optimizer = create_sgd(
            learning_rate=config.learning_rate,
            momentum=config.momentum,
        )
    elif config.optimizer_type == "rmsprop":
        base_optimizer = create_rmsprop(
            learning_rate=config.learning_rate,
            eps=config.eps,
            decay=config.decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")

    # Build transformation chain
    transformations: list[optax.GradientTransformation] = []

    # Add gradient clipping first (if specified)
    if config.gradient_clip is not None:
        if config.clip_type == "by_global_norm":
            transformations.append(optax.clip_by_global_norm(config.gradient_clip))
        elif config.clip_type == "by_value":
            max_val = config.max_value if config.max_value is not None else config.gradient_clip
            transformations.append(optax.clip(max_val))

    # Add schedule (if specified)
    if config.schedule_type is not None:
        schedule = create_schedule(config)
        transformations.append(optax.scale_by_schedule(schedule))

    # Add base optimizer
    transformations.append(base_optimizer)

    # Chain all transformations
    if len(transformations) > 1:
        return optax.chain(*transformations)
    return base_optimizer


__all__ = [
    "OptimizerConfig",
    "create_adam",
    "create_adamw",
    "create_optimizer",
    "create_rmsprop",
    "create_schedule",
    "create_sgd",
    "with_gradient_clipping",
    "with_schedule",
]
