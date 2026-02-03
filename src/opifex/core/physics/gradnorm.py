"""GradNorm: Gradient Normalization for Multi-Task Learning.

This module implements GradNorm, which automatically balances the
contribution of different loss terms based on gradient magnitudes.

Key Features:
    - Automatic loss weight adaptation
    - Balances training rates across tasks
    - Prevents gradient domination by any single loss

The key insight is that losses with larger gradients tend to dominate
training. GradNorm adjusts weights to equalize the gradient contributions.

From Survey Section 2.2.2:
    L_grad = Σᵢ |‖γᵢ∇_θR̂ᵢ‖ - Ḡ × rᵢ^ζ|

References:
    - Chen et al. (2018): GradNorm: Gradient Normalization for
      Adaptive Loss Balancing in Deep Multitask Networks
    - Survey Section 2.2.2: Loss Weighting Strategies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jaxtyping import Array, Float


@dataclass(frozen=True)
class GradNormConfig:
    """Configuration for GradNorm balancing.

    Attributes:
        alpha: Asymmetry parameter (ζ in the paper). Controls how much
               to penalize tasks with different training rates.
               alpha=0: Equal weighting for all tasks
               alpha>0: Stronger gradient for tasks training slower
        learning_rate: Learning rate for weight updates
        update_frequency: How often to update weights (in training steps)
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight
    """

    alpha: float = 1.5
    learning_rate: float = 0.01
    update_frequency: int = 1
    min_weight: float = 0.01
    max_weight: float = 100.0


def compute_inverse_training_rates(
    current_losses: Float[Array, ...],
    initial_losses: Float[Array, ...],
) -> Float[Array, ...]:
    """Compute relative inverse training rates.

    The inverse training rate for task i is:
        r_i = L_i(t) / L_i(0)

    We normalize so the mean is 1:
        r̃_i = r_i / mean(r)

    Args:
        current_losses: Current loss values for each task
        initial_losses: Initial loss values for each task

    Returns:
        Relative inverse training rates (mean normalized to 1)
    """
    # Compute raw training rates (how much loss has decreased)
    rates = current_losses / (initial_losses + 1e-10)

    # Normalize so mean is 1
    mean_rate = jnp.mean(rates)
    return rates / (mean_rate + 1e-10)


def compute_gradient_norms(
    model: nnx.Module,
    loss_fns: Sequence[Callable[[nnx.Module], Float[Array, ""]]],
) -> Float[Array, ...]:
    """Compute gradient norms for each loss function.

    Args:
        model: The neural network model
        loss_fns: List of loss functions, each taking model and returning scalar

    Returns:
        Array of gradient norms for each loss
    """
    norms = []

    for loss_fn in loss_fns:
        # Compute gradient
        _, grads = nnx.value_and_grad(loss_fn)(model)

        # Compute L2 norm of gradient
        grad_leaves = jax.tree.leaves(grads)
        total_norm_sq = sum(jnp.sum(g**2) for g in grad_leaves)
        norm = jnp.sqrt(total_norm_sq)
        norms.append(norm)

    return jnp.array(norms)


class GradNormBalancer(nnx.Module):
    """GradNorm balancer for multi-task learning.

    This module maintains learnable weights for each loss component
    and updates them to balance gradient contributions.

    Attributes:
        config: GradNorm configuration
        log_weights: Learnable log-weights (exp to get actual weights)

    Example:
        >>> balancer = GradNormBalancer(num_losses=3, rngs=nnx.Rngs(0))
        >>> losses = jnp.array([data_loss, pde_loss, boundary_loss])
        >>> weighted_loss = balancer.compute_weighted_loss(losses)
    """

    def __init__(
        self,
        num_losses: int,
        config: GradNormConfig | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize GradNorm balancer.

        Args:
            num_losses: Number of loss components to balance
            config: GradNorm configuration
            rngs: Random number generators
        """
        self.config = config or GradNormConfig()
        self.num_losses = num_losses

        # Initialize log-weights to 0 (weights = 1)
        self.log_weights = nnx.Param(jnp.zeros(num_losses))

        # Track initial losses
        self._initial_losses: Float[Array, ...] | None = None

    @property
    def weights(self) -> Float[Array, ...]:
        """Get current weights (exponentiated and clipped)."""
        raw_weights = jnp.exp(self.log_weights[...])
        return jnp.clip(raw_weights, self.config.min_weight, self.config.max_weight)

    def compute_weighted_loss(
        self,
        losses: Float[Array, ...],
    ) -> Float[Array, ""]:
        """Compute weighted sum of losses.

        Args:
            losses: Individual loss values

        Returns:
            Weighted sum of losses
        """
        return jnp.sum(self.weights * losses)

    def compute_gradnorm_loss(
        self,
        grad_norms: Float[Array, ...],
        losses: Float[Array, ...],
        initial_losses: Float[Array, ...],
    ) -> Float[Array, ""]:
        """Compute GradNorm balancing loss.

        The GradNorm loss encourages:
            ‖w_i ∇L_i‖ ≈ Ḡ × r_i^α

        where Ḡ is the average gradient norm and r_i is the relative
        inverse training rate.

        Args:
            grad_norms: Gradient norms for each loss
            losses: Current loss values
            initial_losses: Initial loss values

        Returns:
            GradNorm loss for weight updates
        """
        weights = self.weights

        # Weighted gradient norms
        weighted_grad_norms = weights * grad_norms

        # Average gradient norm
        g_bar = jnp.mean(weighted_grad_norms)

        # Relative inverse training rates
        relative_rates = compute_inverse_training_rates(losses, initial_losses)

        # Target gradient norms: G_bar * r_i^alpha
        targets = g_bar * jnp.power(relative_rates, self.config.alpha)

        # GradNorm loss: sum of |weighted_norm - target|
        return jnp.sum(jnp.abs(weighted_grad_norms - targets))

    def update_weights(
        self,
        grad_norms: Float[Array, ...],
        losses: Float[Array, ...],
        initial_losses: Float[Array, ...],
    ) -> None:
        """Update weights based on gradient norms.

        This updates the log_weights to minimize the GradNorm loss.

        Args:
            grad_norms: Gradient norms for each loss
            losses: Current loss values
            initial_losses: Initial loss values
        """

        # Compute gradient of GradNorm loss w.r.t. log_weights
        def gradnorm_fn(log_weights_val):
            weights = jnp.exp(log_weights_val)
            weights = jnp.clip(weights, self.config.min_weight, self.config.max_weight)

            weighted_grad_norms = weights * grad_norms
            g_bar = jnp.mean(weighted_grad_norms)

            relative_rates = compute_inverse_training_rates(losses, initial_losses)
            targets = g_bar * jnp.power(relative_rates, self.config.alpha)

            return jnp.sum(jnp.abs(weighted_grad_norms - targets))

        grad = jax.grad(gradnorm_fn)(self.log_weights[...])

        # Update log_weights with gradient descent
        new_log_weights = self.log_weights[...] - self.config.learning_rate * grad
        self.log_weights[...] = new_log_weights

    def set_initial_losses(self, losses: Float[Array, ...]) -> None:
        """Set initial losses for training rate computation.

        Args:
            losses: Initial loss values
        """
        self._initial_losses = losses

    def get_initial_losses(self) -> Float[Array, ...] | None:
        """Get stored initial losses.

        Returns:
            Initial losses if set, None otherwise
        """
        return self._initial_losses
