"""Generic Cascade Trainer for Multilevel Training.

This module implements a generic trainer that orchestrates training across a
hierarchy of models (coarse to fine). It handles:
1. Model switching.
2. Parameter prolongation (transfer from coarse to fine).
3. Optimizer state resizing (transfer of momentum).

References:
    - Survey Section 8.2: Multilevel Training
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from opifex.training.multilevel.multilevel_adam import MultilevelAdam


# Protocol for prolongate function: (coarse_model, fine_model) -> fine_model
ProlongateFn = Callable[[nnx.Module, nnx.Module], nnx.Module]

# Protocol for optim state transition: (old_val, new_shape) -> new_val
StateTransitionFn = Callable[[Any, tuple[int, ...]], Any]


class CascadeTrainer:
    """Manages training across a hierarchy of models."""

    def __init__(
        self,
        hierarchy: Sequence[nnx.Module],
        optimizer: MultilevelAdam,
        prolongate_fn: ProlongateFn,
        state_transition_fn: StateTransitionFn | None = None,
    ):
        """Initialize Cascade Trainer.

        Args:
            hierarchy: Sequence of models from coarse to fine.
            optimizer: MultilevelAdam optimizer instance.
            prolongate_fn: Function to transfer parameters from coarse to fine model.
            state_transition_fn: Function to resize optimizer state. If None,
                                 uses a default zero-padding strategy.
        """
        if not hierarchy:
            raise ValueError("Hierarchy must contain at least one model.")

        self.hierarchy = hierarchy
        self.optimizer = optimizer
        self.prolongate_fn = prolongate_fn
        self.state_transition_fn = state_transition_fn or self._default_state_transition

        self.current_level = 0

        # Initialize optimizer with the first (coarsest) model
        self.optimizer.init(self.get_current_model())

    def get_current_model(self) -> nnx.Module:
        """Return the model at the current level."""
        return self.hierarchy[self.current_level]

    def advance_level(self) -> bool:
        """Advance to the next finer level.

        Performs:
        1. Parameter prolongation from current -> next model.
        2. Optimizer state resizing.
        3. Switching current model pointer.

        Returns:
            True if advanced, False if already at finest level.
        """
        if self.current_level >= len(self.hierarchy) - 1:
            return False

        coarse_model = self.hierarchy[self.current_level]
        fine_model = self.hierarchy[self.current_level + 1]

        # 1. Prolongate parameters
        self.prolongate_fn(coarse_model, fine_model)

        # 2. Resize optimizer state
        self.optimizer.resize_state(fine_model, self.state_transition_fn)

        # 3. Advance pointer
        self.current_level += 1

        return True

    @staticmethod
    def _default_state_transition(old_val: Any, new_shape: tuple[int, ...]) -> Any:
        """Default strategy: Zero pad or truncate to match new shape."""
        # This is a heuristic default.
        # Ideally user provides specific logic (e.g. interpolating momentum).
        # Here we assume extending with zeros is safe (resetting momentum for
        # new parts).

        if not hasattr(old_val, "shape") or not hasattr(old_val, "dtype"):
            return old_val

        new_val = jnp.zeros(new_shape, dtype=old_val.dtype)

        # Copy overlapping slice
        slices = tuple(
            slice(0, min(o, n)) for o, n in zip(old_val.shape, new_shape, strict=False)
        )
        return new_val.at[slices].set(old_val[slices])
