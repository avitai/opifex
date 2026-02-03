"""Warm-starting strategies for optimization acceleration.

This module implements various warm-starting strategies to accelerate
optimization by leveraging information from previous optimizations
or similar problems.

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


class WarmStartingStrategy:
    """Warm-starting strategies for optimization acceleration.

    This class implements various warm-starting strategies to accelerate
    optimization by leveraging information from previous optimizations
    or similar problems.

    Attributes:
        strategy_type: Type of warm-starting strategy
        similarity_threshold: Threshold for problem similarity
        adaptation_steps: Steps for parameter adaptation
        memory_size: Size of optimization memory
        adaptation_ratio: Ratio for optimizer state adaptation
    """

    def __init__(
        self,
        strategy_type: str = "parameter_transfer",
        similarity_threshold: float = 0.8,
        adaptation_steps: int = 5,
        memory_size: int = 10,
        adaptation_ratio: float = 0.9,
        similarity_metric: str = "cosine",
        min_similarity: float = 0.7,
    ):
        """Initialize warm-starting strategy.

        Args:
            strategy_type: Strategy type
                ('parameter_transfer', 'optimizer_state_transfer',
                 'molecular_similarity')
            similarity_threshold: Threshold for considering problems similar
            adaptation_steps: Number of adaptation steps
            memory_size: Maximum number of previous optimizations to remember
            adaptation_ratio: Ratio for adapting previous states
            similarity_metric: Metric for similarity computation
            min_similarity: Minimum similarity for warm-starting
        """
        self.strategy_type = strategy_type
        self.similarity_threshold = similarity_threshold
        self.adaptation_steps = adaptation_steps
        self.memory_size = memory_size
        self.adaptation_ratio = adaptation_ratio
        self.similarity_metric = similarity_metric
        self.min_similarity = min_similarity

        # Memory for previous optimizations
        self._parameter_memory = []
        self._problem_features_memory = []
        self._optimizer_state_memory = []

    def get_warm_start_params(
        self, previous_params: jax.Array, current_problem_features: jax.Array
    ) -> jax.Array:
        """Get warm-start parameters based on parameter transfer.

        Args:
            previous_params: Parameters from previous optimization
            current_problem_features: Features of current problem

        Returns:
            Warm-start parameters for current problem
        """
        if self.strategy_type == "parameter_transfer":
            # Simple parameter transfer with optional adaptation
            adapted_params = previous_params * self.adaptation_ratio

            # Add small random perturbation for exploration
            noise = 0.1 * jax.random.normal(
                jax.random.PRNGKey(42), previous_params.shape
            )
            return adapted_params + noise

        # Default: return parameters as-is
        return previous_params

    def adapt_optimizer_state(
        self, previous_opt_state: dict[str, Any]
    ) -> dict[str, Any]:
        """Adapt optimizer state for warm-starting.

        Args:
            previous_opt_state: Previous optimizer state

        Returns:
            Adapted optimizer state
        """
        adapted_state = {}

        for key, value in previous_opt_state.items():
            if key == "step":
                # Reset step count but keep some history
                adapted_state[key] = jnp.array(max(0, int(value * 0.1)))
            elif isinstance(value, jax.Array):
                # Scale momentum/variance terms
                adapted_state[key] = value * self.adaptation_ratio
            else:
                # Keep other state elements as-is
                adapted_state[key] = value

        return adapted_state

    def get_molecular_warm_start(
        self,
        previous_fingerprints: jax.Array,
        previous_params: jax.Array,
        current_fingerprint: jax.Array,
    ) -> jax.Array:
        """Get warm-start parameters based on molecular similarity.

        Args:
            previous_fingerprints: Fingerprints of previous molecules
            previous_params: Parameters for previous molecules
            current_fingerprint: Fingerprint of current molecule

        Returns:
            Warm-start parameters based on most similar molecule
        """
        # Compute similarities
        if self.similarity_metric == "cosine":
            similarities = jnp.dot(previous_fingerprints, current_fingerprint) / (
                jnp.linalg.norm(previous_fingerprints, axis=1)
                * jnp.linalg.norm(current_fingerprint)
            )
        elif self.similarity_metric == "euclidean":
            distances = jnp.linalg.norm(
                previous_fingerprints - current_fingerprint, axis=1
            )
            similarities = 1.0 / (1.0 + distances)
        else:
            # Default to uniform similarity
            similarities = jnp.ones(len(previous_fingerprints))

        # Find most similar molecule
        most_similar_idx = jnp.argmax(similarities)
        max_similarity = similarities[most_similar_idx]

        if max_similarity > self.min_similarity:
            # Use parameters from most similar molecule
            return previous_params[most_similar_idx]
        # No similar molecule found, return average parameters
        return jnp.mean(previous_params, axis=0)


__all__ = ["WarmStartingStrategy"]
