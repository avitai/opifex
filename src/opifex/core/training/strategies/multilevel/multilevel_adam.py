"""Multilevel Adam Optimizer.

This module implements an Adam optimizer wrapper capable of resizing its internal
state (moments) when the model architecture changes (e.g., during prolongation
in multilevel training).

References:
    - Survey Section 8.2: Multilevel Training
    - Gratton et al. (2025): Recursive bound-constrained AdaGrad...
      (Concept of state transfer)
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import jax
import optax
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    # Transition function signature: (old_value, new_shape) -> new_value
    TransitionFn = Callable[[jax.Array, tuple[int, ...]], jax.Array]


class MultilevelAdam:
    """Adam optimizer with state resizing capabilities for multilevel training.

    This optimizer wraps optax.adam but adds a `resize_state` method that allows
    transferring optimizer state (momentum estimates) when the model parameters
    change shape (e.g., restricted or prolongated).
    """

    def __init__(
        self,
        learning_rate: float | optax.Schedule = 1e-3,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        eps_root: float = 0.0,
    ):
        """Initialize Multilevel Adam.

        Args:
            learning_rate: Learning rate or schedule.
            b1: Exponential decay rate for first moment.
            b2: Exponential decay rate for second moment.
            eps: Term added to denominator to improve numerical stability.
            eps_root: Term added to denominator inside square-root.
        """
        self.tx = optax.adam(
            learning_rate=learning_rate,
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
        )
        self.opt_state = None

    def init(self, model: nnx.Module):
        """Initialize optimizer state for the given model."""
        _, params = nnx.split(model, nnx.Param)
        self.opt_state = self.tx.init(params)  # type: ignore # noqa: PGH003

    def update(self, model: nnx.Module, grads: Any):
        """Perform an optimization step.

        Args:
            model: The model to update (in-place).
            grads: Gradients matching the model structure.
        """
        _, params = nnx.split(model, nnx.Param)

        if self.opt_state is None:
            self.init(model)

        updates, self.opt_state = self.tx.update(  # type: ignore # noqa: PGH003
            grads,
            self.opt_state,
            params,  # type: ignore # noqa: PGH003
        )
        nnx.update(model, optax.apply_updates(params, updates))  # type: ignore # noqa: PGH003

    def resize_state(self, new_model: nnx.Module, transition_fn: TransitionFn):
        """Resize optimizer state to match a new model structure.

        This allows preserving momentum history across levels in multilevel training.

        Args:
            new_model: The new model (e.g., finer grid).
            transition_fn: Function to map old state tensor to new shape.
                           Signature: (old_tensor, new_shape) -> new_tensor
        """
        if self.opt_state is None:
            return

        _, new_params = nnx.split(new_model, nnx.Param)

        # Inspect optax.adam state structure
        # Standard optax.adam returns a chain: (ScaleByAdamState, EmptyState)
        # ScaleByAdamState has attributes: count, mu, nu
        # We need to resize mu and nu. count is scalar (global).

        # We try to unpack it. If structure differs (e.g. newer optax), we might fail.
        # This implementation assumes the standard optax.adam structure.

        adam_state = self.opt_state[0]  # type: ignore # noqa: PGH003
        # Check if it has mu and nu
        if not (hasattr(adam_state, "mu") and hasattr(adam_state, "nu")):
            # Fallback: re-init if structure is unknown
            # But let's log or warn? For now, re-init to be safe if compatible.
            self.init(new_model)
            return

        # Resize mu
        new_mu = jax.tree.map(
            lambda mu, p: transition_fn(mu, p.shape) if hasattr(p, "shape") else mu,
            adam_state.mu,  # type: ignore # noqa: PGH003
            new_params,
        )

        # Resize nu
        new_nu = jax.tree.map(
            lambda nu, p: transition_fn(nu, p.shape) if hasattr(p, "shape") else nu,
            adam_state.nu,  # type: ignore # noqa: PGH003
            new_params,
        )

        # Reconstruct state
        # Assume optax.scale_by_adam.ScaleByAdamState is a NamedTuple or dataclass
        # We can replace fields.
        if hasattr(adam_state, "_replace"):
            new_adam_state = adam_state._replace(mu=new_mu, nu=new_nu)  # type: ignore # noqa: PGH003
        else:
            # Fallback for generic object
            # This relies on implementation details of optax, which uses
            # NamedTuples usually
            from optax._src.transform import ScaleByAdamState

            new_adam_state = ScaleByAdamState(
                count=adam_state.count,  # type: ignore[attr-defined]
                mu=new_mu,
                nu=new_nu,  # type: ignore # noqa: PGH003
            )

        # Update self.opt_state
        # opt_state is a tuple (ScaleByAdamState, EmptyState)
        # We assume index 0 is Adam.
        new_opt_state_list = list(self.opt_state)  # type: ignore # noqa: PGH003
        new_opt_state_list[0] = new_adam_state
        self.opt_state = tuple(new_opt_state_list)
