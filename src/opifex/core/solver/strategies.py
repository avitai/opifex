"""Training Strategies and Components for Unified Solver.

This module defines the protocols and concrete implementations for:
1. TrainingCallback: Modular units that hook into the training loop.
2. TrainingStrategy: High-level configurations that compose components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import optax
from flax import nnx

from opifex.core.training.components.base import BaseCallback, TrainingCallback


@runtime_checkable
class TrainingStrategy(Protocol):
    """Protocol for high-level training strategies.

    A strategy is a factory or configurator that sets up the training process,
    potentially returning a list of components to attach.
    """

    def configure(self, config: Any) -> Any:
        """Modify or validate the training configuration."""
        ...

    def create_components(self) -> list[TrainingCallback]:
        """Create and return the list of components for this strategy."""
        ...


class AdaptiveLossBalancing(BaseCallback, nnx.Module):
    """Adaptive Loss Balancing Strategy (e.g., GradNorm-style).

    Adjusts weights of different loss components during training to balance
    learning rates of different tasks (e.g., PDE residual vs Boundary condition).
    """

    def __init__(self, loss_keys: list[str], alpha: float = 0.5) -> None:
        self.alpha = alpha  # Smoothing parameter
        # Internal state using nnx.Dict for JIT compatibility
        # Eagerly initialize variables for known keys
        self.weights = nnx.Dict({key: nnx.Variable(jnp.array(1.0)) for key in loss_keys})

    def update(self, losses: dict[str, Any]) -> None:
        """Update weights based on current losses using Inverse Magnitude Balancing.

        Balances weights such that w_i * L_i ~ 1.
        """
        epsilon = 1e-8
        for key, loss_val in losses.items():
            # Target weight is inverse of loss magnitude
            w_target = 1.0 / (jnp.abs(loss_val) + epsilon)

            # Check existence using hasattr for safe nnx.Dict access
            if hasattr(self.weights, key):
                # Smooth update
                old_w = self.weights[key].value
                new_w = old_w * self.alpha + w_target * (1 - self.alpha)
                self.weights[key].value = new_w
            # else: Unknown keys are ignored to maintain static structure for JIT

    def on_batch_end(self, batch: int, state: Any, loss: Any, metrics: dict[str, Any]) -> None:
        """Hook to update weights after batch."""
        # Extract individual loss components from metrics if available
        relevant_losses = {k: v for k, v in metrics.items() if k.startswith("loss_")}
        if relevant_losses:
            self.update(relevant_losses)


@dataclass(frozen=True, slots=True, kw_only=True)
class CurriculumRegularization(BaseCallback):
    """Curriculum-weighting callback for a bound loss-term weight.

    Deterministically anneals a loss-term weight along a linear schedule over
    training epochs (curriculum weighting; Krishnapriyan et al. 2021,
    "Characterizing possible failure modes in physics-informed neural
    networks"). ``target_weight`` is the ``nnx.Variable`` the trainer's
    ``loss_fn`` reads (for example an entry of ``Trainer._constraint_weights``);
    :meth:`on_epoch_begin` updates its ``.value`` in place between jitted steps,
    and the next ``nnx.jit`` train step observes the new value without
    recompilation (reference semantics preserve the shared Variable). The
    schedule delegates to :func:`optax.linear_schedule`.
    """

    target_weight: nnx.Variable
    start_val: float = 0.0
    end_val: float = 1.0
    total_epochs: int = 100

    def get_value(self, epoch: int) -> float:
        """Return the scheduled weight for ``epoch`` (linear, clamped at the ends)."""
        schedule = optax.linear_schedule(self.start_val, self.end_val, self.total_epochs)
        return float(jnp.asarray(schedule(epoch)))

    def on_epoch_begin(self, epoch: int, state: Any) -> None:
        """Anneal the bound loss-term weight in place for the upcoming epoch."""
        self.target_weight.value = jnp.asarray(
            self.get_value(epoch), dtype=self.target_weight.value.dtype
        )
