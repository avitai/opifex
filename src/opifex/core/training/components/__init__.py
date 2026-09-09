"""Centralized training components for flexible composition.

This module provides the single source of truth for all reusable training
components, consolidating patterns from across the codebase following DRY principles.

Components included:
- TrainingComponent: Base class for all training components
- CheckpointComponent: Checkpoint management
- MixedPrecisionComponent: Mixed precision training
- RecoveryComponent: Error recovery and stability management

Following strict TDD - implementation designed to pass full test suite.

Author: Opifex Framework Team
Date: October 2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx
from flax.training.dynamic_scale import DynamicScale


if TYPE_CHECKING:
    from collections.abc import Callable

from opifex.core.training.components.lifecycle import TrainingComponent


class CheckpointComponent(TrainingComponent):
    """Component for checkpoint management.

    Handles saving and restoring model checkpoints during training.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize checkpoint component.

        Args:
            config: Configuration including checkpoint_dir, save_frequency, max_to_keep
        """
        super().__init__(config)
        self.checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        self.save_frequency = self.config.get("save_frequency", 100)
        self.max_to_keep = self.config.get("max_to_keep", 5)
        self._checkpoints: list[dict[str, Any]] = []

    def setup(self, model: nnx.Module, training_state: Any) -> None:  # noqa: ARG002 - training-component lifecycle interface
        """Setup checkpoint directory.

        Args:
            model: The neural network model
            training_state: Current training state

        Raises:
            PermissionError: If checkpoint directory cannot be created
        """
        # Create checkpoint directory if it doesn't exist
        try:
            Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise PermissionError(
                f"Cannot create checkpoint directory: {self.checkpoint_dir}"
            ) from e

    def step(self, model: nnx.Module, training_state: Any) -> dict[str, Any] | None:
        """Save checkpoint if at save frequency.

        Args:
            model: The neural network model
            training_state: Current training state

        Returns:
            Dict with checkpoint info if saved, None otherwise
        """
        step = getattr(training_state, "step", 0)

        if step % self.save_frequency == 0 and step > 0:
            # Save checkpoint
            checkpoint = {
                "step": step,
                "model_state": nnx.state(model),
                "training_state": training_state,
            }

            self._checkpoints.append(checkpoint)

            # Maintain max_to_keep limit
            if len(self._checkpoints) > self.max_to_keep:
                self._checkpoints.pop(0)

            return {"checkpoint_saved": True, "step": step}

        return None

    def restore_checkpoint(self, step: int) -> dict[str, Any]:
        """Restore checkpoint from specific step.

        Args:
            step: Step number to restore

        Returns:
            Checkpoint dict if found

        Raises:
            ValueError: If checkpoint not found for step
        """
        for checkpoint in self._checkpoints:
            if checkpoint["step"] == step:
                return checkpoint

        raise ValueError(f"No checkpoint found for step {step}")

    def cleanup(self) -> None:
        """Clear checkpoint memory."""
        self._checkpoints.clear()


class MixedPrecisionComponent(TrainingComponent):
    """Mixed precision training over flax's dynamic loss scaling.

    The component casts inputs to the compute dtype (``create_precision_policy``)
    and differentiates through ``flax.training.dynamic_scale.DynamicScale``, which
    scales the loss, unscales the gradients, reports whether the step was finite,
    backs the scale off on a non-finite step and grows it after
    ``growth_interval`` finite ones. Configuration keys: ``compute_dtype``,
    ``param_dtype``, ``loss_scale`` (initial), ``dynamic_loss_scaling``,
    ``growth_factor``, ``backoff_factor``, ``growth_interval``, ``min_loss_scale``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize mixed precision component.

        Args:
            config: Configuration including compute_dtype, param_dtype, loss_scale

        Raises:
            TypeError: If a dtype is given as a string rather than a JAX dtype
        """
        super().__init__(config)
        compute_dtype = self.config.get("compute_dtype", jnp.bfloat16)
        param_dtype = self.config.get("param_dtype", jnp.float32)
        for dtype in (compute_dtype, param_dtype):
            if isinstance(dtype, str):
                raise TypeError(f"Invalid dtype: {dtype}. Must be a JAX dtype.")
        self.compute_dtype = compute_dtype
        self.param_dtype = param_dtype
        self.initial_loss_scale = float(self.config.get("loss_scale", 2**15))
        self.dynamic_loss_scaling = bool(self.config.get("dynamic_loss_scaling", True))
        self.dynamic_scale = self._fresh_dynamic_scale()
        self.overflow_count = 0
        self.step_count = 0

    def _fresh_dynamic_scale(self) -> DynamicScale:
        """A DynamicScale at the initial scale; static scaling neither grows nor backs off."""
        if not self.dynamic_loss_scaling:
            return DynamicScale(
                growth_factor=1.0,
                backoff_factor=1.0,
                growth_interval=1,
                scale=self.initial_loss_scale,
                minimum_scale=self.initial_loss_scale,
            )
        return DynamicScale(
            growth_factor=float(self.config.get("growth_factor", 2.0)),
            backoff_factor=float(self.config.get("backoff_factor", 0.5)),
            growth_interval=int(self.config.get("growth_interval", 100)),
            scale=self.initial_loss_scale,
            minimum_scale=float(self.config.get("min_loss_scale", 1.0)),
        )

    @property
    def loss_scale(self) -> float:
        """The current loss scale."""
        return float(self.dynamic_scale.scale)

    def setup(self, model: nnx.Module, training_state: Any) -> None:  # noqa: ARG002 - training-component lifecycle interface
        """Reset the loss scale and the counters for a new run.

        Args:
            model: The neural network model
            training_state: Current training state
        """
        self.dynamic_scale = self._fresh_dynamic_scale()
        self.overflow_count = 0
        self.step_count = 0

    def create_precision_policy(self) -> Callable[[jax.Array], jax.Array]:
        """Create the cast that moves parameter-dtype arrays to the compute dtype.

        Returns:
            Callable policy function for mixed precision
        """

        def policy(x: jax.Array) -> jax.Array:
            """Apply mixed precision policy to tensor."""
            if x.dtype == self.param_dtype:
                return x.astype(self.compute_dtype)
            return x

        return policy

    def value_and_grad(
        self, loss_fn: Callable[..., jax.Array]
    ) -> Callable[..., tuple[jax.Array, jax.Array, Any]]:
        """Differentiate ``loss_fn(model, *args)`` under dynamic loss scaling.

        The returned callable evaluates the scaled loss, unscales the gradients
        and updates the component's scale and counters; it returns
        ``(is_finite, loss, grads)``. Callers skip the optimizer update when
        ``is_finite`` is false.

        Args:
            loss_fn: ``(model, *args) -> scalar loss``.

        Returns:
            A callable with the same arguments as ``loss_fn``.
        """

        def step(model: nnx.Module, *args: Any) -> tuple[jax.Array, jax.Array, Any]:
            graphdef, state = nnx.split(model)

            def pure_loss(pure_state: Any) -> jax.Array:
                return loss_fn(nnx.merge(graphdef, pure_state), *args)

            self.dynamic_scale, is_finite, loss, grads = self.dynamic_scale.value_and_grad(
                pure_loss
            )(state)
            self.step_count += 1
            if not bool(is_finite):
                self.overflow_count += 1
            return is_finite, loss, grads

        return step

    def get_mixed_precision_stats(self) -> dict[str, Any]:
        """Report the loss scale, the dtypes and the step counters."""
        return {
            "loss_scale": self.loss_scale,
            "overflow_count": self.overflow_count,
            "step_count": self.step_count,
            "compute_dtype": str(self.compute_dtype),
            "param_dtype": str(self.param_dtype),
        }


class FlexibleOptimizerFactory(TrainingComponent):
    """Factory for creating and managing sophisticated optimizers.

    Uses the centralized opifex.core.training.optimizers module.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize optimizer factory.

        Args:
            config: Configuration for optimizer creation and scheduling
        """
        super().__init__(config)

        # Import here to avoid circular dependency
        from opifex.core.training.optimizers import create_optimizer, OptimizerConfig

        # Convert dict config to OptimizerConfig
        self.optimizer_config = OptimizerConfig(
            optimizer_type=self.config.get("optimizer_type", "adam"),
            learning_rate=self.config.get("learning_rate", 1e-3),
            weight_decay=self.config.get("weight_decay", 0.0),
            b1=self.config.get("beta1", 0.9),
            b2=self.config.get("beta2", 0.999),
            eps=self.config.get("eps", 1e-8),
            momentum=self.config.get("momentum", 0.0),
            schedule_type=self.config.get("schedule_type")
            if self.config.get("use_schedule", True)
            else None,
            decay_steps=self.config.get("total_steps", 10000),
            alpha=self.config.get("cosine_alpha", 0.1),
            transition_steps=self.config.get("decay_steps", 1000),
            decay_rate=self.config.get("decay_rate", 0.95),
            gradient_clip=self.config.get("grad_clip"),
        )
        self._create_optimizer = create_optimizer

    def create_optimizer(self, model: nnx.Module):  # noqa: ARG002 - optimizer-factory interface receives model
        """Create optimizer with optional scheduling.

        Args:
            model: The neural network model

        Returns:
            Configured optimizer
        """
        # Use centralized optimizer creation
        return self._create_optimizer(self.optimizer_config)


def _get_recovery_base():
    """Lazy import to avoid circular dependency."""
    from opifex.core.training.components.recovery import ErrorRecoveryManager

    return ErrorRecoveryManager


class RecoveryComponent(TrainingComponent):
    """Component for error recovery and stability management.

    Delegates all recovery logic to :class:`ErrorRecoveryManager` (DRY).
    This class provides the component-pattern interface while the actual
    implementation lives in ``recovery.py``.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize recovery component by delegating to ErrorRecoveryManager."""
        super().__init__(config)
        _cls = _get_recovery_base()
        self._delegate = _cls(config)

    @property
    def last_stable_state(self) -> Any:
        """Last known stable training state."""
        return self._delegate.last_stable_state

    @last_stable_state.setter
    def last_stable_state(self, value: Any) -> None:
        self._delegate.last_stable_state = value

    @property
    def recovery_attempts(self) -> int:
        """Number of recovery attempts since last stable state."""
        return self._delegate.recovery_attempts

    @recovery_attempts.setter
    def recovery_attempts(self, value: int) -> None:
        self._delegate.recovery_attempts = value

    def setup(self, model: nnx.Module, training_state: Any) -> None:
        """Initialize with stable state."""
        self._delegate.setup(model, training_state)

    def check_stability(
        self,
        loss: float,
        grads: Any,
        training_state: Any,
    ) -> tuple[bool, str | None]:
        """Check if training is stable."""
        return self._delegate.check_training_stability(loss, grads, training_state)

    def apply_gradient_clipping(self, grads: Any) -> Any:
        """Apply gradient clipping for stability."""
        return self._delegate.apply_gradient_clipping(grads)

    def recover_from_instability(self, issue_type: str, training_state: Any) -> Any:
        """Attempt recovery from training instability."""
        return self._delegate.recover_from_instability(issue_type, training_state)

    def update_stable_state(self, training_state: Any) -> None:
        """Update the last known stable training state."""
        self._delegate.update_stable_state(training_state)


__all__ = [
    "CheckpointComponent",
    "MixedPrecisionComponent",
    "RecoveryComponent",
    "TrainingComponent",
]
