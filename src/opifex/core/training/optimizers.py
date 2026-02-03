"""Optimizer creation and configuration for Opifex framework.

This module provides a centralized, DRY approach to creating and configuring
Optax optimizers with common patterns like learning rate schedules and
gradient clipping.

Following strict TDD principles - all functions are implemented to pass
the tests defined in test_optimizers.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import optax


@dataclass
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


def create_schedule(  # noqa: PLR0912
    schedule_type: str,
    init_value: float = 1e-3,
    decay_steps: int | None = None,
    alpha: float = 0.1,
    transition_steps: int | None = None,
    decay_rate: float = 0.96,
    end_value: float | None = None,
    boundaries_and_values: tuple[list[int], list[float]] | None = None,
    peak_value: float | None = None,
    warmup_steps: int | None = None,
) -> optax.Schedule:
    """Create learning rate schedule.

    Args:
        schedule_type: Type of schedule ("constant", "cosine", "exponential",
            "linear", "step", "warmup_cosine")
        init_value: Initial learning rate value
        decay_steps: Number of steps for decay (cosine)
        alpha: Minimum learning rate multiplier (cosine)
        transition_steps: Steps between rate changes (exponential, linear)
        decay_rate: Decay rate (exponential)
        end_value: Final learning rate value (linear)
        boundaries_and_values: (boundaries, values) for step schedule
        peak_value: Peak learning rate after warmup
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate schedule function

    Raises:
        ValueError: If schedule_type is unknown
    """
    if schedule_type == "constant":
        return optax.constant_schedule(init_value)

    if schedule_type == "cosine":
        if decay_steps is None:
            decay_steps = 1000
        return optax.cosine_decay_schedule(
            init_value=init_value, decay_steps=decay_steps, alpha=alpha
        )

    if schedule_type == "exponential":
        if transition_steps is None:
            transition_steps = 1000
        return optax.exponential_decay(
            init_value=init_value,
            transition_steps=transition_steps,
            decay_rate=decay_rate,
        )

    if schedule_type == "linear":
        if transition_steps is None:
            transition_steps = 1000
        if end_value is None:
            end_value = init_value * 0.1
        return optax.linear_schedule(
            init_value=init_value,
            end_value=end_value,
            transition_steps=transition_steps,
        )

    if schedule_type == "step":
        if boundaries_and_values is None:
            # Default step schedule
            boundaries_and_values = (
                [100, 200],
                [init_value, init_value * 0.1, init_value * 0.01],
            )
        boundaries, values = boundaries_and_values
        return optax.piecewise_constant_schedule(
            init_value=values[0] if values else init_value,
            boundaries_and_scales={
                b: values[i + 1] / values[i]
                for i, b in enumerate(boundaries)
                if i + 1 < len(values)
            }
            if len(values) > 1
            else {},
        )

    if schedule_type == "warmup_cosine":
        if warmup_steps is None:
            warmup_steps = 100
        if peak_value is None:
            peak_value = init_value
        if decay_steps is None:
            decay_steps = 1000

        return optax.warmup_cosine_decay_schedule(
            init_value=init_value,
            peak_value=peak_value,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
        )

    raise ValueError(f"Unknown schedule type: {schedule_type}")


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
            max_val = (
                config.max_value
                if config.max_value is not None
                else config.gradient_clip
            )
            transformations.append(optax.clip(max_val))

    # Add schedule (if specified)
    if config.schedule_type is not None:
        schedule = create_schedule(
            schedule_type=config.schedule_type,
            init_value=config.learning_rate,
            decay_steps=config.decay_steps,
            alpha=config.alpha,
            transition_steps=config.transition_steps,
            decay_rate=config.decay_rate,
            end_value=config.end_value,
            boundaries_and_values=config.boundaries_and_values,
            peak_value=config.peak_value,
            warmup_steps=config.warmup_steps,
        )
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
