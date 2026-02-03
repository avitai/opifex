"""Multi-Objective Learn-to-Optimize (L2O) Framework.

This module implements neural network-based multi-objective optimization algorithms
that can simultaneously optimize multiple conflicting objectives using learned
strategies.

Key Features:
- Multi-objective optimization with neural Pareto frontier approximation
- Learned scalarization strategies for objective combination
- Multi-objective MAML adaptation for different objective combinations
- Performance indicators (hypervolume, spread, convergence metrics)
- Integration with existing L2O framework for enhanced optimization
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from flax import nnx


if TYPE_CHECKING:
    from collections.abc import Callable

    from flax.nnx import Rngs

    from .l2o_engine import L2OEngine


@dataclass
class MultiObjectiveConfig:
    """Configuration for multi-objective L2O optimization.

    This configuration defines parameters for simultaneously optimizing multiple
    conflicting objectives using neural network-based strategies.
    """

    num_objectives: int = 2
    pareto_points_target: int = 100
    scalarization_strategy: str = "learned"  # "learned", "weighted_sum", "chebyshev"
    diversity_pressure: float = 0.1
    convergence_tolerance: float = 1e-6
    max_pareto_iterations: int = 500
    hypervolume_reference_point: list[float] | None = None
    adaptive_weights: bool = True
    dominated_solution_filtering: bool = True

    def __post_init__(self):
        """Validate multi-objective configuration parameters."""
        if self.num_objectives < 2:
            raise ValueError(
                "Multi-objective optimization requires at least 2 objectives"
            )
        if self.pareto_points_target <= 0:
            raise ValueError("Pareto points target must be positive")
        valid_strategies = ["learned", "weighted_sum", "chebyshev", "achievement"]
        if self.scalarization_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid scalarization strategy: {self.scalarization_strategy}. "
                f"Must be one of {valid_strategies}"
            )
        if (
            self.hypervolume_reference_point is not None
            and len(self.hypervolume_reference_point) != self.num_objectives
        ):
            raise ValueError(
                "Hypervolume reference point must have same dimension as objectives"
            )


class ParetoFrontierOptimizer(nnx.Module):
    """Neural Pareto frontier approximation for multi-objective optimization.

    This optimizer learns to approximate the Pareto frontier using neural networks
    and can generate diverse solutions along the frontier efficiently.
    """

    def __init__(
        self,
        config: MultiObjectiveConfig,
        problem_dimension: int,
        *,
        rngs: Rngs,
    ):
        """Initialize Pareto frontier optimizer.

        Args:
            config: Multi-objective configuration
            problem_dimension: Dimension of optimization problem
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.problem_dimension = problem_dimension

        # Neural network for Pareto frontier approximation
        self.frontier_network = nnx.Sequential(
            nnx.Linear(self.config.num_objectives, 128, rngs=rngs),
            nnx.gelu,
            nnx.Linear(128, 128, rngs=rngs),
            nnx.gelu,
            nnx.Linear(128, problem_dimension, rngs=rngs),
        )

        # Reference points for hypervolume calculation
        if config.hypervolume_reference_point is not None:
            self.reference_point = jnp.array(config.hypervolume_reference_point)
        else:
            # Default reference point for minimization (should dominate all solutions)
            # Use positive values that are likely to be dominated by actual objective
            # values
            self.reference_point = jnp.ones(config.num_objectives) * 10.0

    def generate_pareto_solutions(
        self,
        objective_functions: list[Callable[[jax.Array], jax.Array]],
        preference_vectors: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Generate diverse solutions along the Pareto frontier.

        Args:
            objective_functions: List of objective functions to optimize
            preference_vectors: Optional preference vectors for guided search

        Returns:
            Tuple of (solutions, objective_values)
        """
        if preference_vectors is None:
            # Generate uniform preference vectors
            preference_vectors = self._generate_uniform_preferences()

        # Vectorized computation using JAX vmap for efficient batch processing
        def process_single_preference(preference, key):
            """Process single preference vector and return solution with objectives."""
            # Generate solution from frontier network
            solution = self.frontier_network(preference)

            # Evaluate all objectives for this solution
            objectives = jnp.array([obj_fn(solution) for obj_fn in objective_functions])

            # Handle invalid objectives with conditional fallback using JAX control flow
            is_invalid = jnp.any(jnp.isnan(objectives)) | jnp.any(jnp.isinf(objectives))

            def use_fallback():
                fallback_solution = (
                    jax.random.normal(key, (self.problem_dimension,)) * 0.5
                )
                fallback_objectives = jnp.array(
                    [obj_fn(fallback_solution) for obj_fn in objective_functions]
                )
                return fallback_solution, fallback_objectives

            def use_original():
                return solution, objectives

            # Use JAX conditional to select between original and fallback
            final_solution, final_objectives = jax.lax.cond(
                is_invalid, use_fallback, use_original
            )

            return final_solution, final_objectives

        # Generate keys for each preference vector for deterministic randomness
        keys = jax.random.split(jax.random.PRNGKey(42), len(preference_vectors))

        # Vectorized processing of all preference vectors using vmap
        solutions, objective_values = jax.vmap(process_single_preference)(
            preference_vectors, keys
        )

        # Filter dominated solutions if enabled
        if self.config.dominated_solution_filtering:
            non_dominated_mask = self._identify_non_dominated_solutions(
                objective_values
            )
            solutions = solutions[non_dominated_mask]
            objective_values = objective_values[non_dominated_mask]

        return solutions, objective_values

    def optimize_pareto_frontier(
        self,
        objective_functions: list[Callable[[jax.Array], jax.Array]],
        constraint_function: Callable[[jax.Array], jax.Array] | None = None,
    ) -> dict[str, Any]:
        """Optimize the neural network to better approximate Pareto frontier.

        Args:
            objective_functions: List of objective functions
            constraint_function: Optional constraint function

        Returns:
            Optimization results and metrics
        """

        def pareto_loss_fn(model):
            """Loss function for Pareto frontier approximation."""

            # Generate solutions and evaluate objectives
            preference_vectors = self._generate_uniform_preferences()

            def compute_single_preference_loss(preference):
                """Compute loss for a single preference vector."""
                solution = model(preference)

                # Check constraints if provided
                if constraint_function is not None:
                    constraint_violation = constraint_function(solution)
                    has_violation = jnp.any(constraint_violation > 0)
                    constraint_penalty = 1000.0 * jnp.sum(
                        jnp.maximum(0, constraint_violation)
                    )
                else:
                    has_violation = False
                    constraint_penalty = 0.0

                # Evaluate objectives
                objectives = jnp.array(
                    [obj_fn(solution) for obj_fn in objective_functions]
                )

                # Multi-objective loss: combination of objectives with diversity
                scalarized_loss = self._scalarize_objectives(objectives, preference)
                diversity_term = self._compute_diversity_loss(
                    solution, preference_vectors
                )

                objective_loss = (
                    scalarized_loss + self.config.diversity_pressure * diversity_term
                )

                # Use JAX conditional to handle constraint violations
                total_loss = jax.lax.cond(
                    has_violation, lambda: constraint_penalty, lambda: objective_loss
                )

                # Return (loss, is_valid) tuple
                is_valid = ~has_violation
                return total_loss, is_valid

            # Vectorized computation across all preference vectors
            vmap_results = jax.vmap(compute_single_preference_loss)(preference_vectors)
            losses, validity_mask = vmap_results
            # Ensure losses is treated as an array
            losses = jnp.asarray(losses)

            # Compute mean loss only over valid solutions
            num_valid = jnp.sum(validity_mask)
            valid_losses = jnp.where(validity_mask, losses, 0.0)

            return jnp.sum(valid_losses) / jnp.maximum(1, num_valid)

        # Optimize frontier network using JAX
        optimizer = nnx.Optimizer(
            self.frontier_network, optax.adam(1e-3), wrt=nnx.Param
        )

        best_loss = float("inf")
        iteration_count = 0
        for _iteration in range(self.config.max_pareto_iterations):
            loss_value, grads = nnx.value_and_grad(pareto_loss_fn)(
                self.frontier_network
            )

            # Update network parameters using NNX API
            optimizer.update(self.frontier_network, grads)

            best_loss = min(best_loss, loss_value)
            iteration_count += 1

            # Early stopping check
            if loss_value < self.config.convergence_tolerance:
                break

        return {
            "final_loss": float(best_loss),
            "iterations": iteration_count,
            "converged": loss_value < self.config.convergence_tolerance,
        }

    def _generate_uniform_preferences(self) -> jax.Array:
        """Generate uniform preference vectors for diverse Pareto solutions."""
        num_points = self.config.pareto_points_target

        if self.config.num_objectives == 2:
            # For 2 objectives, use simple linear spacing
            weights = jnp.linspace(0, 1, num_points)
            preferences = jnp.column_stack([weights, 1 - weights])
        else:
            # For >2 objectives, use Dirichlet distribution sampling
            key = jax.random.PRNGKey(42)  # Fixed seed for reproducibility
            alpha = jnp.ones(self.config.num_objectives)
            preferences = jax.random.dirichlet(key, alpha, shape=(num_points,))

        return preferences

    def _identify_non_dominated_solutions(
        self, objective_values: jax.Array
    ) -> jax.Array:
        """Identify non-dominated solutions using vectorized operations."""
        num_solutions = objective_values.shape[0]

        def check_domination_for_solution(i):
            """Check if solution i is dominated by any other solution."""
            # Compare solution i against all other solutions
            solution_i = objective_values[i]

            # Vectorized comparison: for each j, check if j dominates i
            def dominates_i(j):
                solution_j = objective_values[j]
                # j dominates i if: all(j <= i) AND any(j < i) AND i != j
                all_leq = jnp.all(solution_j <= solution_i)
                any_less = jnp.any(solution_j < solution_i)
                not_same = i != j
                return all_leq & any_less & not_same

            # Check domination against all solutions
            solution_indices = jnp.arange(num_solutions)
            dominated_by_any = jnp.any(jax.vmap(dominates_i)(solution_indices))

            return ~dominated_by_any  # Return True if NOT dominated

        # Vectorized computation across all solutions
        return jax.vmap(check_domination_for_solution)(jnp.arange(num_solutions))

    def _scalarize_objectives(
        self, objectives: jax.Array, preference: jax.Array
    ) -> jax.Array:
        """Convert multi-objective problem to single objective using scalarization."""
        if self.config.scalarization_strategy == "weighted_sum":
            return jnp.dot(preference, objectives)
        if self.config.scalarization_strategy == "chebyshev":
            return jnp.max(preference * objectives)
        if self.config.scalarization_strategy == "achievement":
            # Achievement scalarizing function
            return jnp.max(preference * objectives) + 0.01 * jnp.sum(
                preference * objectives
            )
        # "learned"
        # For now, use weighted sum - in practice, this would be a learned function
        return jnp.dot(preference, objectives)

    def _compute_diversity_loss(
        self, solution: jax.Array, all_preferences: jax.Array
    ) -> jax.Array:
        """Compute diversity loss to encourage spread along Pareto frontier."""
        # Sample subset of preferences for efficiency
        subset_preferences = all_preferences[:10]

        # Vectorized generation of other solutions
        other_solutions = jax.vmap(self.frontier_network)(subset_preferences)

        # Vectorized distance computation
        def compute_distance_to_solution(other_sol):
            return jnp.linalg.norm(solution - other_sol)

        distances = jax.vmap(compute_distance_to_solution)(other_solutions)

        # Encourage minimum distance to other solutions
        min_distance = jnp.min(
            distances + 1e-8
        )  # Add small epsilon to avoid division by zero
        return 1.0 / min_distance


class ObjectiveScalarizer(nnx.Module):
    """Learned scalarization strategies for multi-objective optimization.

    This module learns optimal ways to combine multiple objectives into single
    objectives for efficient optimization using neural networks.
    """

    def __init__(
        self,
        config: MultiObjectiveConfig,
        problem_features_dim: int,
        *,
        rngs: Rngs,
    ):
        """Initialize objective scalarizer.

        Args:
            config: Multi-objective configuration
            problem_features_dim: Dimension of problem feature vectors
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.features_dim = problem_features_dim

        # Neural network for learning scalarization weights
        self.weight_network = nnx.Sequential(
            nnx.Linear(problem_features_dim, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, config.num_objectives, rngs=rngs),
            nnx.softmax,  # Ensure weights sum to 1
        )

    def learn_scalarization_weights(
        self,
        problem_features: jax.Array,
        objective_values_history: jax.Array,
        performance_feedback: jax.Array,
    ) -> jax.Array:
        """Learn optimal scalarization weights based on problem characteristics.

        Args:
            problem_features: Features describing the optimization problem
            objective_values_history: History of objective values achieved
            performance_feedback: Feedback on solution quality

        Returns:
            Learned scalarization weights
        """
        learned_weights = self.weight_network(problem_features)

        # Adaptive adjustment based on performance feedback
        if self.config.adaptive_weights:
            # Simple adaptation: increase weights for objectives with poor performance
            poor_performance_mask = performance_feedback < jnp.mean(
                performance_feedback
            )
            adaptation_factor = 1.1

            adapted_weights = jnp.where(
                poor_performance_mask,
                learned_weights * adaptation_factor,
                learned_weights,
            )

            # Renormalize to ensure sum to 1
            return adapted_weights / jnp.sum(adapted_weights)

        return learned_weights

    def scalarize_objectives(
        self,
        objectives: jax.Array,
        weights: jax.Array,
        strategy: str | None = None,
    ) -> jax.Array:
        """Convert multiple objectives to single scalar value.

        Args:
            objectives: Array of objective values
            weights: Scalarization weights
            strategy: Scalarization strategy override

        Returns:
            Scalar objective value
        """
        if strategy is None:
            strategy = self.config.scalarization_strategy

        if strategy == "weighted_sum":
            return jnp.dot(weights, objectives)
        if strategy == "chebyshev":
            return jnp.max(weights * objectives)
        if strategy == "achievement":
            return jnp.max(weights * objectives) + 0.01 * jnp.sum(weights * objectives)
        # "learned" or other
        # Use weighted sum as base, but could be extended with learned functions
        return jnp.dot(weights, objectives)


class PerformanceIndicators:
    """Performance indicators for multi-objective optimization quality assessment.

    This class provides various metrics to evaluate the quality of Pareto frontier
    approximations and multi-objective solutions.
    """

    @staticmethod
    def compute_hypervolume(
        pareto_front: jax.Array,
        reference_point: jax.Array,
    ) -> jax.Array:
        """Compute hypervolume indicator for Pareto front quality.

        Args:
            pareto_front: Array of Pareto optimal solutions (objectives)
            reference_point: Reference point for hypervolume calculation

        Returns:
            Hypervolume value
        """
        # Handle empty pareto front
        if len(pareto_front) == 0:
            return jnp.array(0.0)

        # Simplified hypervolume calculation for 2D case
        if pareto_front.shape[1] == 2:
            # Sort by first objective
            sorted_front = pareto_front[jnp.argsort(pareto_front[:, 0])]

            # Calculate area under the curve with proper bounds checking
            hypervolume = 0.0
            for i in range(len(sorted_front) - 1):
                width = sorted_front[i + 1, 0] - sorted_front[i, 0]
                height = reference_point[1] - sorted_front[i, 1]
                # Only add positive contributions
                area_contribution = width * jnp.maximum(0.0, height)
                hypervolume += area_contribution

            # Add contribution from last point
            if len(sorted_front) > 0:
                last_height = reference_point[1] - sorted_front[-1, 1]
                # Use a small width for the final segment
                final_width = 0.1  # Small default width
                hypervolume += final_width * jnp.maximum(0.0, last_height)

            # Ensure non-negative result
            return jnp.maximum(0.0, hypervolume)

        # For higher dimensions, use approximation with robust bounds checking
        dominated_volume = 0.0
        for solution in pareto_front:
            # Volume of box from solution to reference point
            dimensions_diff = reference_point - solution
            # Only include positive dimensions
            positive_dims = jnp.maximum(0.0, dimensions_diff)
            box_volume = jnp.prod(positive_dims)
            # Prevent NaN by checking for valid box volumes
            box_volume = jnp.where(jnp.isfinite(box_volume), box_volume, 0.0)
            dominated_volume += box_volume

        # Ensure result is finite and non-negative
        result = jnp.maximum(0.0, dominated_volume)
        return jnp.where(jnp.isfinite(result), result, 0.0)

    @staticmethod
    def compute_spread_indicator(pareto_front: jax.Array) -> jax.Array:
        """Compute spread (diversity) indicator for Pareto front.

        Args:
            pareto_front: Array of Pareto optimal solutions

        Returns:
            Spread indicator value
        """
        if len(pareto_front) < 2:
            return jnp.array(0.0)

        # Vectorized pairwise distance computation
        def compute_distances_from_point(i):
            """Compute distances from point i to all subsequent points."""
            point_i = pareto_front[i]
            # Only compute distances to points j > i to avoid duplicates
            subsequent_points = pareto_front[i + 1 :]

            def distance_to_point(point_j):
                return jnp.linalg.norm(point_i - point_j)

            return jax.vmap(distance_to_point)(subsequent_points)

        # Compute all pairwise distances efficiently
        num_points = len(pareto_front)
        all_distances = []

        for i in range(num_points - 1):
            distances_from_i = compute_distances_from_point(i)
            all_distances.append(distances_from_i)

        # Concatenate all distances
        distances = jnp.concatenate(all_distances) if all_distances else jnp.array([])

        # Spread is the standard deviation of distances
        return jnp.std(distances) if len(distances) > 0 else jnp.array(0.0)

    @staticmethod
    def compute_convergence_indicator(
        pareto_front: jax.Array,
        true_pareto_front: jax.Array | None = None,
    ) -> jax.Array:
        """Compute convergence indicator measuring closeness to true Pareto front.

        Args:
            pareto_front: Approximated Pareto front
            true_pareto_front: True Pareto front (if known)

        Returns:
            Convergence indicator value
        """
        if true_pareto_front is None:
            # Without true front, use internal convergence measure
            # Measure how well-distributed the solutions are
            if len(pareto_front) < 2:
                return jnp.array(0.0)

            # Use standard deviation of objective values as proxy
            return jnp.mean(jnp.std(pareto_front, axis=0))

        # With true front, compute minimum distances using vectorized operations
        def compute_min_distance_to_true_front(approx_point):
            """Compute minimum distance from approx_point to true Pareto front."""

            def distance_to_true_point(true_point):
                return jnp.linalg.norm(approx_point - true_point)

            # Vectorized distance computation to all true front points
            distances_to_true = jax.vmap(distance_to_true_point)(true_pareto_front)
            return jnp.min(distances_to_true)

        # Vectorized computation across all approximated points
        min_distances = jax.vmap(compute_min_distance_to_true_front)(pareto_front)

        return jnp.mean(min_distances)


class MultiObjectiveL2OEngine(nnx.Module):
    """Core multi-objective L2O optimization engine.

    This engine integrates Pareto frontier optimization, learned scalarization,
    and performance assessment for comprehensive multi-objective optimization.
    """

    def __init__(
        self,
        config: MultiObjectiveConfig,
        l2o_engine: L2OEngine,
        problem_dimension: int,
        *,
        rngs: Rngs,
    ):
        """Initialize multi-objective L2O engine.

        Args:
            config: Multi-objective configuration
            l2o_engine: Base L2O engine for single-objective optimization
            problem_dimension: Dimension of optimization problems
            rngs: Random number generators
        """
        super().__init__()

        self.config = config
        self.l2o_engine = l2o_engine
        self.problem_dimension = problem_dimension

        # Initialize components
        self.pareto_optimizer = ParetoFrontierOptimizer(
            config, problem_dimension, rngs=rngs
        )

        # Problem features dimension based on problem characteristics
        features_dim = problem_dimension * 2  # Problem params + derived features
        self.scalarizer = ObjectiveScalarizer(config, features_dim, rngs=rngs)

        self.performance_indicators = PerformanceIndicators()

    def solve_multi_objective_problem(
        self,
        objective_functions: list[Callable[[jax.Array], jax.Array]],
        problem_features: jax.Array,
        constraint_function: Callable[[jax.Array], jax.Array] | None = None,
        true_pareto_front: jax.Array | None = None,
    ) -> dict[str, Any]:
        """Solve multi-objective optimization problem using L2O strategies.

        Args:
            objective_functions: List of objective functions to optimize
            problem_features: Features characterizing the problem
            constraint_function: Optional constraint function
            true_pareto_front: Optional true Pareto front for evaluation

        Returns:
            Comprehensive optimization results and metrics
        """
        start_time = time.time()

        # Step 1: Optimize Pareto frontier approximation
        pareto_training_results = self.pareto_optimizer.optimize_pareto_frontier(
            objective_functions, constraint_function
        )

        # Step 2: Generate diverse Pareto solutions
        pareto_solutions, pareto_objectives = (
            self.pareto_optimizer.generate_pareto_solutions(objective_functions)
        )

        # Step 3: Learn scalarization weights
        performance_feedback = jnp.ones(self.config.num_objectives)  # Placeholder
        learned_weights = self.scalarizer.learn_scalarization_weights(
            problem_features, pareto_objectives, performance_feedback
        )

        # Step 4: Compute performance indicators
        hypervolume = self.performance_indicators.compute_hypervolume(
            pareto_objectives, self.pareto_optimizer.reference_point
        )
        spread = self.performance_indicators.compute_spread_indicator(pareto_objectives)
        convergence = self.performance_indicators.compute_convergence_indicator(
            pareto_objectives, true_pareto_front
        )

        solve_time = time.time() - start_time

        return {
            "pareto_solutions": pareto_solutions,
            "pareto_objectives": pareto_objectives,
            "learned_weights": learned_weights,
            "hypervolume": float(hypervolume),
            "spread": float(spread),
            "convergence": float(convergence),
            "num_pareto_points": len(pareto_solutions),
            "solve_time": solve_time,
            "pareto_training_converged": pareto_training_results["converged"].item(),
            "pareto_training_iterations": pareto_training_results["iterations"],
        }

    def solve_with_preference(
        self,
        objective_functions: list[Callable[[jax.Array], jax.Array]],
        preference_vector: jax.Array,
        problem_features: jax.Array,
    ) -> tuple[jax.Array, dict[str, Any]]:
        """Solve multi-objective problem with user preference vector.

        Args:
            objective_functions: List of objective functions
            preference_vector: User preference weights for objectives
            problem_features: Problem characterization features

        Returns:
            Solution and optimization metrics
        """
        # Generate solution based on preference
        solution = self.pareto_optimizer.frontier_network(preference_vector)

        # Evaluate objectives
        objectives = jnp.array([obj_fn(solution) for obj_fn in objective_functions])

        # Scalarize objectives using preference
        scalarized_value = self.scalarizer.scalarize_objectives(
            objectives, preference_vector
        )

        metrics = {
            "objectives": objectives,
            "scalarized_value": float(scalarized_value),
            "preference_vector": preference_vector,
        }

        return solution, metrics
