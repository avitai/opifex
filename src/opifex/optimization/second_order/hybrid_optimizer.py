"""Hybrid Adam→L-BFGS optimizer for physics-informed training.

This module implements a hybrid optimization strategy that starts with
Adam for initial exploration and switches to L-BFGS for efficient
convergence once the loss landscape becomes smooth.

Design Rationale (from Survey Section 7.4):
    "L-BFGS is more effective in later stages when loss varies smoothly."

The hybrid approach combines:
    - Adam's robustness in noisy, high-curvature early optimization
    - L-BFGS's superior convergence in smooth regions near optima

Key Features:
    - Multiple switching criteria (epoch, loss variance, gradient norm)
    - Loss history tracking for variance-based switching
    - Full JAX/JIT compatibility
    - Works with FLAX NNX models

References:
    - Survey: arXiv:2601.10222v1 Section 7.4
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax

from opifex.optimization.second_order.config import (
    HybridOptimizerConfig,
    LinesearchType,
    SwitchCriterion,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float, PyTree


class HybridOptimizerState(NamedTuple):
    """State for hybrid Adam→L-BFGS optimizer.

    Attributes:
        adam_state: Optax state for Adam optimizer
        lbfgs_state: Optax state for L-BFGS optimizer
        step_count: Total optimization steps taken
        loss_history: Recent loss values for variance computation
        using_lbfgs: Whether currently using L-BFGS (vs Adam)
        switched: Whether the switch to L-BFGS has occurred
    """

    adam_state: optax.OptState
    lbfgs_state: optax.OptState
    step_count: int
    loss_history: Float[Array, ...]
    using_lbfgs: bool
    switched: bool


@dataclass
class HybridOptimizer:
    """Hybrid Adam→L-BFGS optimizer.

    This optimizer starts with Adam and switches to L-BFGS based on
    configurable criteria. The transition is designed to leverage
    Adam's robustness in early training and L-BFGS's efficiency
    for final convergence.

    Attributes:
        config: Hybrid optimizer configuration
        adam: Adam optimizer instance
        lbfgs: L-BFGS optimizer instance

    Example:
        >>> config = HybridOptimizerConfig(first_order_steps=1000)
        >>> optimizer = HybridOptimizer(config)
        >>> state = optimizer.init(params)
        >>> # Training loop
        >>> for step in range(num_steps):
        ...     loss, grads = loss_and_grad_fn(params)
        ...     updates, state = optimizer.update(grads, state, params, loss=loss)
        ...     params = optax.apply_updates(params, updates)
    """

    config: HybridOptimizerConfig

    def __post_init__(self):
        """Initialize Adam and L-BFGS optimizers."""
        # Create Adam optimizer
        self.adam = optax.adam(
            learning_rate=self.config.adam_learning_rate,
            b1=self.config.adam_b1,
            b2=self.config.adam_b2,
        )

        # Create L-BFGS optimizer
        lbfgs_config = self.config.lbfgs_config
        if lbfgs_config.linesearch == LinesearchType.ZOOM:
            linesearch = optax.scale_by_zoom_linesearch(
                max_linesearch_steps=lbfgs_config.max_linesearch_steps,
            )
        else:
            linesearch = optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=lbfgs_config.max_linesearch_steps,
            )

        self.lbfgs = optax.lbfgs(
            memory_size=lbfgs_config.memory_size,
            scale_init_precond=lbfgs_config.scale_init_precond,
            linesearch=linesearch,
        )

    def init(self, params: PyTree) -> HybridOptimizerState:
        """Initialize optimizer state.

        Args:
            params: Model parameters (PyTree)

        Returns:
            Initial optimizer state
        """
        adam_state = self.adam.init(params)
        lbfgs_state = self.lbfgs.init(params)
        loss_history = jnp.full(self.config.loss_history_window, jnp.inf)

        return HybridOptimizerState(
            adam_state=adam_state,
            lbfgs_state=lbfgs_state,
            step_count=0,
            loss_history=loss_history,
            using_lbfgs=False,
            switched=False,
        )

    def update(
        self,
        grads: PyTree,
        state: HybridOptimizerState,
        params: PyTree,
        *,
        loss: Float[Array, ""] | None = None,
        value: Float[Array, ""] | None = None,
        grad: PyTree | None = None,
        value_fn: Callable[[PyTree], Float[Array, ""]] | None = None,
    ) -> tuple[PyTree, HybridOptimizerState]:
        """Compute parameter updates.

        This method handles the switching logic and delegates to either
        Adam or L-BFGS depending on the current state.

        Args:
            grads: Parameter gradients
            state: Current optimizer state
            params: Current parameters (needed for L-BFGS)
            loss: Current loss value (for variance-based switching)
            value: Alias for loss (for optax L-BFGS compatibility)
            grad: Alias for grads (for optax L-BFGS compatibility)
            value_fn: Loss function (needed for L-BFGS line search)

        Returns:
            Tuple of (updates, new_state)
        """
        # Handle value/grad aliases for L-BFGS compatibility
        if loss is None and value is not None:
            loss = value
        if grad is not None:
            grads = grad

        # Update loss history
        new_loss_history = self._update_loss_history(state.loss_history, loss)

        # Check if we should switch to L-BFGS
        should_switch = self._should_switch(state, new_loss_history, grads)

        # Determine if we're using L-BFGS this step
        using_lbfgs = state.switched or (should_switch and not state.switched)
        switched = state.switched or should_switch

        if using_lbfgs:
            # Use L-BFGS
            updates, new_lbfgs_state = self.lbfgs.update(
                grads,
                state.lbfgs_state,
                params,
                value=loss,
                grad=grads,
                value_fn=value_fn,
            )
            new_adam_state = state.adam_state
        else:
            # Use Adam
            updates, new_adam_state = self.adam.update(grads, state.adam_state, params)
            new_lbfgs_state = state.lbfgs_state

        new_state = HybridOptimizerState(
            adam_state=new_adam_state,
            lbfgs_state=new_lbfgs_state,
            step_count=state.step_count + 1,
            loss_history=new_loss_history,
            using_lbfgs=using_lbfgs,
            switched=switched,
        )

        return updates, new_state

    def _update_loss_history(
        self,
        history: Float[Array, ...],
        loss: Float[Array, ""] | None,
    ) -> Float[Array, ...]:
        """Update rolling loss history.

        Args:
            history: Current loss history
            loss: New loss value (or None)

        Returns:
            Updated loss history
        """
        if loss is None:
            return history

        # Roll history and add new loss
        new_history = jnp.roll(history, -1)
        return new_history.at[-1].set(loss)

    def _should_switch(  # noqa: PLR0911
        self,
        state: HybridOptimizerState,
        loss_history: Float[Array, ...],
        grads: PyTree,
    ) -> bool:
        """Determine if we should switch from Adam to L-BFGS.

        Args:
            state: Current optimizer state
            loss_history: Updated loss history
            grads: Current gradients

        Returns:
            True if should switch to L-BFGS
        """
        # Already switched
        if state.switched:
            return False

        # Not enough steps yet
        if state.step_count < self.config.first_order_steps:
            return False

        criterion = self.config.switch_criterion

        if criterion == SwitchCriterion.EPOCH:
            # Simply switch after first_order_steps
            return True

        if criterion == SwitchCriterion.LOSS_VARIANCE:
            # Switch when loss variance is below threshold
            # Only use finite values in history
            finite_mask = jnp.isfinite(loss_history)
            if not jnp.any(finite_mask):
                return False
            finite_losses = jnp.where(finite_mask, loss_history, 0.0)
            count = jnp.sum(finite_mask)
            if count < 2:
                return False
            variance = jnp.var(finite_losses, where=finite_mask)
            return float(variance) < self.config.loss_variance_threshold

        if criterion == SwitchCriterion.GRADIENT_NORM:
            # Switch when gradient norm is below threshold
            grad_norm = _compute_grad_norm(grads)
            return float(grad_norm) < self.config.gradient_norm_threshold

        if criterion == SwitchCriterion.RELATIVE_IMPROVEMENT:
            # Switch when relative improvement is below threshold
            finite_mask = jnp.isfinite(loss_history)
            if not jnp.any(finite_mask):
                return False
            first_loss = loss_history[0]
            last_loss = loss_history[-1]
            if not (jnp.isfinite(first_loss) and jnp.isfinite(last_loss)):
                return False
            rel_improvement = jnp.abs(first_loss - last_loss) / (
                jnp.abs(first_loss) + 1e-8
            )
            return float(rel_improvement) < self.config.relative_improvement_threshold

        return False

    @property
    def is_using_lbfgs(self) -> bool:
        """Check if optimizer is currently using L-BFGS.

        Note: This is a convenience property. For actual state, check
        the HybridOptimizerState.using_lbfgs field.
        """
        # This property doesn't make sense without state
        # It's here for API compatibility but should use state.using_lbfgs
        raise RuntimeError(
            "Use state.using_lbfgs to check current optimizer mode. "
            "This property requires state which is not available."
        )


def _compute_grad_norm(grads: PyTree) -> Float[Array, ""]:
    """Compute global L2 norm of gradients.

    Args:
        grads: Gradient PyTree

    Returns:
        L2 norm of flattened gradients
    """
    leaves = jax.tree.leaves(grads)
    squared_norms = [jnp.sum(g**2) for g in leaves]
    return jnp.sqrt(sum(squared_norms))
