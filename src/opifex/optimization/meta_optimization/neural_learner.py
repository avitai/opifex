"""Learn-to-optimize (L2O) meta-learning engine.

This module implements learn-to-optimize algorithms that use neural networks
to learn optimization strategies from data. The meta-network learns to
predict good parameter updates based on gradient information and
optimization history.

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.neural.base import StandardMLP


if TYPE_CHECKING:
    from collections.abc import Callable


class LearnToOptimize(nnx.Module):
    """Learn-to-optimize (L2O) meta-learning system.

    This class implements learn-to-optimize algorithms that use neural networks
    to learn optimization strategies from data. The meta-network learns to
    predict good parameter updates based on gradient information and
    optimization history.

    Attributes:
        meta_network: Neural network for learning optimization rules
        base_optimizer: Base optimization algorithm
        meta_learning_rate: Learning rate for meta-network training
        unroll_steps: Number of unrolling steps for meta-gradient computation
        adaptive_step_size: Enable adaptive step size learning
        quantum_aware: Enable quantum-specific adaptations
        scf_integration: Enable SCF convergence acceleration
    """

    def __init__(
        self,
        meta_network_layers: list[int] | None = None,
        base_optimizer: str = "adam",
        meta_learning_rate: float = 1e-4,
        unroll_steps: int = 20,
        adaptive_step_size: bool = False,
        quantum_aware: bool = False,
        scf_integration: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize Learn-to-Optimize meta-optimizer.

        Args:
            meta_network_layers: Architecture of meta-network
            base_optimizer: Base optimizer to enhance
            meta_learning_rate: Learning rate for meta-network training
            unroll_steps: Number of unroll steps for meta-gradients
            adaptive_step_size: Enable adaptive step size learning
            quantum_aware: Enable quantum-specific optimizations
            scf_integration: Enable SCF convergence acceleration
            rngs: Random number generators for initialization
        """
        if meta_network_layers is None:
            meta_network_layers = [128, 64, 32]

        super().__init__()

        self.meta_network_layers = meta_network_layers
        self.base_optimizer = base_optimizer
        self.meta_learning_rate = meta_learning_rate
        self.unroll_steps = unroll_steps
        self.adaptive_step_size = adaptive_step_size
        self.quantum_aware = quantum_aware
        self.scf_integration = scf_integration

        # Meta-network for learning optimization rules
        # Input: [gradient, previous_updates, loss_history]
        # Output: [parameter_update] or [parameter_update, step_size]
        output_dim = (
            meta_network_layers[0]
            if not adaptive_step_size
            else meta_network_layers[0] + 1
        )

        layers = [*meta_network_layers, output_dim]
        self.meta_network = StandardMLP(layers, rngs=rngs)

        # Meta-optimizer for training the meta-network
        self.meta_optimizer = nnx.Optimizer(
            self.meta_network, optax.adam(meta_learning_rate), wrt=nnx.Param
        )

    def compute_update(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
        loss_history: jax.Array | None = None,
    ) -> jax.Array:
        """Compute parameter update using meta-network.

        Args:
            gradient: Current gradient
            previous_updates: History of previous updates
            loss_history: History of loss values

        Returns:
            Predicted parameter update
        """
        # Prepare input features for meta-network
        input_features = self._prepare_meta_input(
            gradient, previous_updates, loss_history
        )

        # Get meta-network prediction
        meta_output = self.meta_network(input_features)

        if self.adaptive_step_size:
            # Split output into update direction and step size
            update_direction = meta_output[:-1]
            step_size = jnp.abs(meta_output[-1])  # Ensure positive step size
            parameter_update = step_size * update_direction
        else:
            parameter_update = meta_output

        return parameter_update

    def _prepare_meta_input(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
        loss_history: jax.Array | None = None,
    ) -> jax.Array:
        """Prepare input features for meta-network."""
        # Normalize gradient
        grad_norm = jnp.linalg.norm(gradient)
        normalized_grad = gradient / (grad_norm + 1e-8)

        # Features from previous updates
        if previous_updates.size > 0:
            avg_update = jnp.mean(previous_updates, axis=0)
            update_variance = jnp.var(previous_updates, axis=0)
        else:
            avg_update = jnp.zeros_like(gradient)
            update_variance = jnp.zeros_like(gradient)

        # Combine features
        features = jnp.concatenate(
            [
                normalized_grad,
                avg_update,
                update_variance,
                jnp.array([grad_norm]),  # Include gradient norm as scalar feature
            ]
        )

        # Pad or truncate to match meta-network input size
        target_size = self.meta_network_layers[0]  # Use actual network input size
        if len(features) > target_size:
            features = features[:target_size]
        elif len(features) < target_size:
            padding = jnp.zeros(target_size - len(features))
            features = jnp.concatenate([features, padding])

        return features

    def compute_meta_gradients(
        self,
        loss_fn: Callable[[jax.Array], jax.Array],
        initial_params: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute meta-gradients for meta-network training.

        Args:
            loss_fn: Loss function for optimization problem
            initial_params: Initial parameters for optimization

        Returns:
            Meta-gradients for meta-network parameters
        """

        def meta_loss_fn(meta_params):
            # Create temporary meta-network with correct architecture
            output_size = initial_params.size
            layers = [*self.meta_network_layers, output_size]
            temp_meta_network = StandardMLP(layers, rngs=nnx.Rngs(42))
            nnx.update(temp_meta_network, meta_params)

            # Simulate optimization trajectory using current meta-network
            params = initial_params
            total_loss = 0.0
            previous_updates = jnp.zeros((0, output_size))

            for _step in range(self.unroll_steps):
                # Compute gradient
                gradient = jax.grad(loss_fn)(params)

                # Get meta-network update
                input_features = self._prepare_meta_input(gradient, previous_updates)
                update = temp_meta_network(input_features)

                # Ensure update has correct shape - truncate or pad to match params
                if update.size > initial_params.size:
                    update = update[: initial_params.size]
                elif update.size < initial_params.size:
                    padding = jnp.zeros(initial_params.size - update.size)
                    update = jnp.concatenate([update, padding])
                update = update.reshape(initial_params.shape)

                # Apply update
                params = params - update

                # Accumulate loss
                step_loss = loss_fn(params)
                total_loss += step_loss

                # Update history
                previous_updates = jnp.concatenate(
                    [previous_updates, update.flatten().reshape(1, -1)], axis=0
                )
                if previous_updates.shape[0] > 5:  # Keep only recent history
                    previous_updates = previous_updates[-5:]

            return total_loss / self.unroll_steps

        # Compute meta-gradients
        meta_params = nnx.state(self.meta_network, nnx.Param)
        meta_grads = jax.grad(meta_loss_fn)(meta_params)

        # Convert State to dictionary for compatibility - properly handle JAX types
        return jax.tree.map(lambda x: x, dict(meta_grads))

    def compute_adaptive_update(
        self,
        gradient: jax.Array,
        previous_updates: jax.Array,
    ) -> jax.Array:
        """Compute adaptive parameter update."""
        # Prepare meta-network input
        input_features = self._prepare_meta_input(gradient, previous_updates)

        # Get adaptive update from meta-network
        return self.meta_network(input_features)

    def compute_quantum_update(
        self,
        orbital_params: jax.Array,
        scf_history: jax.Array,
    ) -> jax.Array:
        """Compute quantum-aware parameter update.

        Args:
            orbital_params: Orbital coefficient parameters
            scf_history: SCF convergence history

        Returns:
            Quantum-adapted parameter update
        """
        if not self.quantum_aware:
            return jnp.zeros_like(orbital_params)

        # Simplified quantum adaptation
        # In practice, this would use sophisticated quantum mechanical insights

        # SCF convergence-based adaptation
        scf_trend = (
            jnp.diff(scf_history[-5:]) if len(scf_history) > 1 else jnp.array([0.0])
        )
        scf_acceleration = jnp.mean(scf_trend)

        # Orbital-based features
        orbital_norm = jnp.linalg.norm(orbital_params)
        orbital_features = jnp.array([orbital_norm, scf_acceleration])

        # Pad features to match meta-network input
        padded_features = jnp.zeros(128)
        padded_features = padded_features.at[: len(orbital_features)].set(
            orbital_features
        )

        # Get quantum adaptation from meta-network
        quantum_update = self.meta_network(padded_features)

        # Reshape to match orbital parameters
        return quantum_update[: orbital_params.size].reshape(orbital_params.shape)


__all__ = ["LearnToOptimize"]
