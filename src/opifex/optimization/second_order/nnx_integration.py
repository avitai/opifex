"""FLAX NNX integration for second-order optimizers.

This module provides wrapper classes that make it easy to use second-order
optimizers (L-BFGS, hybrid Adam→L-BFGS) with FLAX NNX models.

Design Philosophy:
    - Hide the complexity of nnx.split/merge from users
    - Provide familiar step() interface similar to nnx.Optimizer
    - Support both pure L-BFGS and hybrid optimization strategies

Key Classes:
    - NNXSecondOrderOptimizer: L-BFGS optimizer for NNX models
    - NNXHybridOptimizer: Hybrid Adam→L-BFGS for NNX models
    - create_nnx_lbfgs_optimizer: Factory function for L-BFGS

References:
    - FLAX NNX documentation: https://flax.readthedocs.io/en/latest/nnx/
    - optax L-BFGS requires functional API with value_and_grad_from_state
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import optax
from flax import nnx

from opifex.optimization.second_order.config import (
    HybridOptimizerConfig,
    LBFGSConfig,
    LinesearchType,
    SwitchCriterion,
)
from opifex.optimization.second_order.hybrid_optimizer import (
    HybridOptimizer,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from jaxtyping import Array, Float


def create_nnx_lbfgs_optimizer(
    model: nnx.Module,
    config: LBFGSConfig | None = None,
) -> NNXSecondOrderOptimizer:
    """Create L-BFGS optimizer for NNX model.

    Factory function that creates an NNXSecondOrderOptimizer configured
    with L-BFGS.

    Args:
        model: FLAX NNX model to optimize
        config: L-BFGS configuration. Uses defaults if None.

    Returns:
        Configured NNXSecondOrderOptimizer instance.

    Example:
        >>> model = MyModel(rngs=nnx.Rngs(0))
        >>> optimizer = create_nnx_lbfgs_optimizer(model)
        >>> for _ in range(100):
        ...     loss = optimizer.step(loss_fn)
    """
    return NNXSecondOrderOptimizer(model, config)


class NNXSecondOrderOptimizer:
    """L-BFGS optimizer wrapper for FLAX NNX models.

    This class wraps optax.lbfgs to work seamlessly with NNX models.
    It handles the split/merge operations internally and provides a
    simple step() interface.

    Attributes:
        model: The NNX model being optimized
        config: L-BFGS configuration

    Example:
        >>> model = MyModel(rngs=nnx.Rngs(0))
        >>> optimizer = NNXSecondOrderOptimizer(model)
        >>> for _ in range(100):
        ...     loss = optimizer.step(lambda m: compute_loss(m, data))
    """

    def __init__(
        self,
        model: nnx.Module,
        config: LBFGSConfig | None = None,
    ):
        """Initialize the optimizer.

        Args:
            model: FLAX NNX model to optimize
            config: L-BFGS configuration. Uses defaults if None.
        """
        self.model = model
        self.config = config or LBFGSConfig()

        # Create optax L-BFGS optimizer
        if self.config.linesearch == LinesearchType.ZOOM:
            linesearch = optax.scale_by_zoom_linesearch(
                max_linesearch_steps=self.config.max_linesearch_steps,
            )
        else:
            linesearch = optax.scale_by_backtracking_linesearch(
                max_backtracking_steps=self.config.max_linesearch_steps,
            )

        self._optimizer = optax.lbfgs(
            memory_size=self.config.memory_size,
            scale_init_precond=self.config.scale_init_precond,
            linesearch=linesearch,
        )

        # Initialize optimizer state with model parameters
        self._graphdef, self._params = nnx.split(model, nnx.Param)
        self._opt_state = self._optimizer.init(
            self._params  # pyright: ignore[reportArgumentType]
        )

    def step(
        self,
        loss_fn: Callable[[nnx.Module], Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Perform one optimization step.

        Args:
            loss_fn: Function that takes model and returns scalar loss.

        Returns:
            Loss value at current parameters (before update).
        """

        # Create functional loss that works with params
        def functional_loss(params):
            model = nnx.merge(self._graphdef, params)
            return loss_fn(model)

        # Get value and grad using optax's L-BFGS compatible function
        value, grad = optax.value_and_grad_from_state(functional_loss)(
            self._params, state=self._opt_state
        )

        # Update parameters
        updates, self._opt_state = self._optimizer.update(
            grad,
            self._opt_state,
            self._params,
            value=value,
            grad=grad,
            value_fn=functional_loss,
        )
        self._params = optax.apply_updates(
            self._params,  # pyright: ignore[reportArgumentType]
            updates,
        )

        # Update the model in place
        nnx.update(self.model, self._params)

        return value  # pyright: ignore[reportReturnType]


class NNXHybridOptimizer:
    """Hybrid Adam→L-BFGS optimizer wrapper for FLAX NNX models.

    This class wraps the HybridOptimizer to work seamlessly with NNX models.
    It starts with Adam for initial exploration and switches to L-BFGS
    for efficient final convergence.

    Attributes:
        model: The NNX model being optimized
        config: Hybrid optimizer configuration
        is_using_lbfgs: Whether currently using L-BFGS (vs Adam)
        step_count: Total optimization steps taken

    Example:
        >>> model = MyModel(rngs=nnx.Rngs(0))
        >>> optimizer = NNXHybridOptimizer(model)
        >>> for _ in range(1000):
        ...     loss = optimizer.step(lambda m: compute_loss(m, data))
        >>> print(f"Switched to L-BFGS: {optimizer.is_using_lbfgs}")
    """

    def __init__(
        self,
        model: nnx.Module,
        config: HybridOptimizerConfig | None = None,
    ):
        """Initialize the hybrid optimizer.

        Args:
            model: FLAX NNX model to optimize
            config: Hybrid optimizer configuration. Uses defaults if None.
        """
        self.model = model
        self.config = config or HybridOptimizerConfig()

        # Handle string criterion for convenience
        if isinstance(self.config.switch_criterion, str):
            criterion_map = {
                "epoch": SwitchCriterion.EPOCH,
                "loss_variance": SwitchCriterion.LOSS_VARIANCE,
                "gradient_norm": SwitchCriterion.GRADIENT_NORM,
                "relative_improvement": SwitchCriterion.RELATIVE_IMPROVEMENT,
            }
            # Create new config with proper enum - need to handle frozen dataclass
            new_config_dict = {
                "first_order_steps": self.config.first_order_steps,
                "switch_criterion": criterion_map.get(
                    self.config.switch_criterion, SwitchCriterion.LOSS_VARIANCE
                ),
                "loss_variance_threshold": self.config.loss_variance_threshold,
                "loss_history_window": self.config.loss_history_window,
                "gradient_norm_threshold": self.config.gradient_norm_threshold,
                "relative_improvement_threshold": (
                    self.config.relative_improvement_threshold
                ),
                "adam_learning_rate": self.config.adam_learning_rate,
                "adam_b1": self.config.adam_b1,
                "adam_b2": self.config.adam_b2,
                "lbfgs_config": self.config.lbfgs_config,
            }
            self.config = HybridOptimizerConfig(**new_config_dict)

        # Create hybrid optimizer
        self._hybrid = HybridOptimizer(self.config)

        # Initialize with model parameters
        self._graphdef, self._params = nnx.split(model, nnx.Param)
        self._state = self._hybrid.init(self._params)

    @property
    def is_using_lbfgs(self) -> bool:
        """Check if optimizer is currently using L-BFGS."""
        return self._state.using_lbfgs

    @property
    def step_count(self) -> int:
        """Get total optimization steps taken."""
        return self._state.step_count

    def step(
        self,
        loss_fn: Callable[[nnx.Module], Float[Array, ""]],
    ) -> Float[Array, ""]:
        """Perform one optimization step.

        Args:
            loss_fn: Function that takes model and returns scalar loss.

        Returns:
            Loss value at current parameters (before update).
        """

        # Create functional loss that works with params
        def functional_loss(params):
            model = nnx.merge(self._graphdef, params)
            return loss_fn(model)

        # Compute loss and gradients
        loss, grads = jax.value_and_grad(functional_loss)(self._params)

        # Update using hybrid optimizer
        updates, self._state = self._hybrid.update(
            grads,
            self._state,
            self._params,  # pyright: ignore[reportArgumentType]
            loss=loss,
            value_fn=functional_loss,
        )
        self._params = optax.apply_updates(
            self._params,  # pyright: ignore[reportArgumentType]
            updates,
        )

        # Update the model in place
        nnx.update(self.model, self._params)

        return loss
