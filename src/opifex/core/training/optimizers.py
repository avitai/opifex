"""From ``OptimizationConfig`` to the optimizer substrax builds.

:class:`~opifex.core.training.config.OptimizationConfig` is the one owner of the optimizer's
settings. :func:`optimizer_spec` maps it onto :class:`substrax.optim.OptimizerConfig`, with the
configured schedule as the optimizer's learning rate, so a decaying schedule reaches the update
rather than being cancelled by the base optimizer's normalisation; :func:`create_schedule`
builds that schedule. The optimizer itself comes from :func:`substrax.optim.create_optimizer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import optax
from substrax.optim import MOMENTUM_TYPES, OPTIMIZER_TYPES, OptimizerConfig


if TYPE_CHECKING:
    from collections.abc import Callable

    from opifex.core.training.config import OptimizationConfig


def _constant_schedule(config: OptimizationConfig) -> optax.Schedule:
    """Constant learning-rate schedule."""
    return optax.constant_schedule(config.learning_rate)


def _cosine_schedule(config: OptimizationConfig) -> optax.Schedule:
    """Cosine-decay schedule (``decay_steps`` defaults to 1000)."""
    decay_steps = config.decay_steps if config.decay_steps is not None else 1000
    return optax.cosine_decay_schedule(
        init_value=config.learning_rate, decay_steps=decay_steps, alpha=config.alpha
    )


def _exponential_schedule(config: OptimizationConfig) -> optax.Schedule:
    """Exponential-decay schedule (``transition_steps`` defaults to 1000)."""
    transition_steps = config.transition_steps if config.transition_steps is not None else 1000
    return optax.exponential_decay(
        init_value=config.learning_rate,
        transition_steps=transition_steps,
        decay_rate=config.decay_rate,
    )


def _linear_schedule(config: OptimizationConfig) -> optax.Schedule:
    """Linear schedule to ``end_value`` (defaults to 10% of the initial value)."""
    transition_steps = config.transition_steps if config.transition_steps is not None else 1000
    end_value = config.end_value if config.end_value is not None else config.learning_rate * 0.1
    return optax.linear_schedule(
        init_value=config.learning_rate,
        end_value=end_value,
        transition_steps=transition_steps,
    )


def _step_schedule(config: OptimizationConfig) -> optax.Schedule:
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


def _warmup_cosine_schedule(config: OptimizationConfig) -> optax.Schedule:
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


_SCHEDULE_BUILDERS: dict[str, Callable[[OptimizationConfig], optax.Schedule]] = {
    "constant": _constant_schedule,
    "cosine": _cosine_schedule,
    "exponential": _exponential_schedule,
    "linear": _linear_schedule,
    "step": _step_schedule,
    "warmup_cosine": _warmup_cosine_schedule,
}


def create_schedule(config: OptimizationConfig) -> optax.Schedule:
    """Create a learning-rate schedule from an optimization configuration.

    The schedule kind and its parameters are read from ``config`` (see
    :class:`~opifex.core.training.config.OptimizationConfig`); ``config.learning_rate`` is the
    schedule's initial value. The output is cast to float32 for stable optax updates under x64.

    Args:
        config: Optimization configuration carrying ``schedule_type`` and the associated
            schedule parameters.

    Returns:
        A float32 ``optax.Schedule``.

    Raises:
        ValueError: If ``config.schedule_type`` is unknown.
    """
    builder = _SCHEDULE_BUILDERS.get(config.schedule_type or "")
    if builder is None:
        raise ValueError(f"Unknown schedule type: {config.schedule_type}")
    return _as_float32_schedule(builder(config))


def optimizer_spec(config: OptimizationConfig) -> OptimizerConfig:
    """Map an optimization configuration onto substrax's optimizer specification.

    The configured schedule, when there is one, becomes the optimizer's learning rate.
    ``momentum`` reaches only the optimizers that take it (``sgd``, ``rmsprop``); substrax
    refuses a weight decay on an optimizer without decoupled decay and both clip fields at once.

    Args:
        config: The optimization configuration owned by the training configuration.

    Returns:
        The specification :func:`substrax.optim.create_optimizer` builds.

    Raises:
        ValueError: If ``config.optimizer`` is not one substrax builds.
    """
    if config.optimizer not in OPTIMIZER_TYPES:
        raise ValueError(
            f"Unknown optimizer type: {config.optimizer!r}; one of {', '.join(OPTIMIZER_TYPES)}"
        )
    learning_rate: float | optax.Schedule = (
        create_schedule(config) if config.schedule_type is not None else config.learning_rate
    )
    return OptimizerConfig(
        optimizer_type=config.optimizer,  # type: ignore[arg-type]
        learning_rate=learning_rate,
        b1=config.beta1,
        b2=config.beta2,
        eps=config.eps,
        momentum=config.momentum if config.optimizer in MOMENTUM_TYPES else None,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        gradient_clip_value=config.gradient_clip_value,
    )


__all__ = ["create_schedule", "optimizer_spec"]
