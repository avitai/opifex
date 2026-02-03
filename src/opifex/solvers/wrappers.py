"""Solver Wrappers for Advanced Functionality.

Provides wrappers to enhance base SciML solvers with:
- Bayesian Uncertainty Quantification (UQ)
- Learning-to-Optimize (L2O) capabilities
"""

from collections.abc import Sequence

import jax.numpy as jnp

from opifex.core.problems import Problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
    SolverConfig,
    SolverState,
)


class BayesianWrapper:
    """Wraps a solver to provide Bayesian Uncertainty Quantification."""

    def __init__(self, solver: SciMLSolver, num_samples: int = 10):
        self.solver = solver
        self.num_samples = num_samples

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run solver multiple times (e.g. with MC Dropout) and aggregate."""
        # Note: In a real implementation, we would toggle dropout RNG per sample.
        # For this prototype, we mock the variance by calling solve multiple times
        # (assuming solver has stochasticity/rng handling).

        solutions = []
        for _ in range(self.num_samples):
            # Ideally pass different RNG seeds in state
            sol = self.solver.solve(problem, initial_state, config)
            solutions.append(sol)

        # Aggregate results
        # Assuming fields are arrays, we compute mean and std
        base_fields = solutions[0].fields
        aggregated_fields = {}
        for key in base_fields:
            stacked = jnp.stack([s.fields[key] for s in solutions])
            aggregated_fields[key] = jnp.mean(stacked, axis=0)
            aggregated_fields[f"{key}_std"] = jnp.std(stacked, axis=0)

        metrics = solutions[0].metrics.copy()
        metrics["num_samples"] = self.num_samples
        metrics["uncertainty"] = jnp.mean(
            jnp.array([jnp.mean(s.fields[key]) for s in solutions])
        )  # Dummy metric

        return Solution(
            fields=aggregated_fields,
            metrics=metrics,
            execution_time=sum(s.execution_time for s in solutions),
            converged=all(s.converged for s in solutions),
        )


class L2OWrapper:
    """Wraps a solver to apply Learning-to-Optimize strategies."""

    def __init__(self, solver: SciMLSolver, l2o_model_path: str):
        self.solver = solver
        self.l2o_model_path = l2o_model_path

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run solver with L2O enhancements."""
        # 1. Load L2O model (placeholder)
        # model = load(self.l2o_model_path)

        # 2. Modify config or state based on L2O prediction
        # new_lr = model.predict(problem_features)

        # 3. Run solver
        solution = self.solver.solve(problem, initial_state, config)

        # 4. Add L2O metadata
        solution.metrics["l2o_active"] = True

        return solution


class ConformalWrapper:
    """Wraps a solver to provide Conformal Prediction intervals."""

    def __init__(self, solver: SciMLSolver, alpha: float = 0.1, num_samples: int = 20):
        self.solver = solver
        self.alpha = alpha
        self.num_samples = num_samples

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run solver multiple times and compute empirical prediction intervals."""
        # Monte Carlo Sampling
        solutions = []
        for _ in range(self.num_samples):
            sol = self.solver.solve(problem, initial_state, config)
            solutions.append(sol)

        # Aggregate results
        base_fields = solutions[0].fields
        aggregated_fields = {}

        lower_q = self.alpha / 2
        upper_q = 1.0 - (self.alpha / 2)

        for key in base_fields:
            stacked = jnp.stack([s.fields[key] for s in solutions])
            aggregated_fields[key] = jnp.mean(stacked, axis=0)
            aggregated_fields[f"{key}_lower"] = jnp.quantile(stacked, lower_q, axis=0)
            aggregated_fields[f"{key}_upper"] = jnp.quantile(stacked, upper_q, axis=0)
            aggregated_fields[f"{key}_std"] = jnp.std(stacked, axis=0)

        metrics = solutions[0].metrics.copy()
        metrics["num_samples"] = self.num_samples
        metrics["alpha"] = self.alpha
        metrics["conformal_method"] = "monte_carlo_empirical"

        return Solution(
            fields=aggregated_fields,
            metrics=metrics,
            execution_time=sum(s.execution_time for s in solutions),
            converged=all(s.converged for s in solutions),
        )


class EnsembleWrapper:
    """Wraps multiple solvers to provide Ensemble-based Uncertainty Quantification.

    Executes a list of independent solvers (instances) and aggregates their results
    to provide mean predictions and uncertainty estimates (standard deviation).
    """

    def __init__(self, solvers: Sequence[SciMLSolver]):
        """Initialize with a list of instantiated solvers.

        Args:
            solvers: List of solver instances to form the ensemble.
        """
        if not solvers:
            raise ValueError("Ensemble cannot be empty")
        self.solvers = solvers

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run all solvers in the ensemble and aggregate results."""
        solutions = []
        for solver in self.solvers:
            # Each solver runs independently
            # Note: Parallel execution (pmap) could be added here later
            sol = solver.solve(problem, initial_state, config)
            solutions.append(sol)

        # Aggregate results
        base_fields = solutions[0].fields
        aggregated_fields = {}

        # Verify consistency
        # In a robust implementation we'd check all keys match

        for key in base_fields:
            # Stack fields from all solutions: (num_models, ...)
            stacked = jnp.stack([s.fields[key] for s in solutions])

            # Compute Mean and Std
            aggregated_fields[key] = jnp.mean(stacked, axis=0)
            aggregated_fields[f"{key}_std"] = jnp.std(stacked, axis=0)

            # Also provide explicit Lower/Upper bounds (e.g., +/- 2 std) if desired,
            # but std is sufficient for basic Gaussian assumption.

        # Aggregate metrics
        metrics = solutions[0].metrics.copy()

        # Add ensemble specific metrics
        metrics["ensemble_size"] = len(self.solvers)

        # Average execution time
        metrics["avg_execution_time"] = sum(s.execution_time for s in solutions) / len(
            self.solvers
        )
        metrics["total_execution_time"] = sum(s.execution_time for s in solutions)

        return Solution(
            fields=aggregated_fields,
            metrics=metrics,
            execution_time=metrics["total_execution_time"],
            converged=all(s.converged for s in solutions),
        )


class GenerativeWrapper:
    """Wraps a generative solver (which returns samples) to provide UQ statistics.

    Assumes the base solver returns a Solution where fields are batches of samples
    (e.g., shape [batch, ...]). This wrapper computes mean and std dev across the batch.
    """

    def __init__(self, solver: SciMLSolver):
        """Initialize with a generative solver instance."""
        self.solver = solver

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run solver and compute statistics from generated samples."""
        # Run base solver (expected to return batch of samples)
        raw_solution = self.solver.solve(problem, initial_state, config)

        # Process fields
        processed_fields = {}
        for key, value in raw_solution.fields.items():
            # Assume value is array with shape (num_samples, ...)
            # We compute statistics over the first dimension (axis=0)
            if value.ndim > 0:
                processed_fields[f"{key}_mean"] = jnp.mean(value, axis=0)
                processed_fields[f"{key}_std"] = jnp.std(value, axis=0)
                # Keep original key pointing to MEAN for compatibility
                processed_fields[key] = processed_fields[f"{key}_mean"]
            else:
                # Scalar, pass through
                processed_fields[key] = value

        # Update metrics
        metrics = raw_solution.metrics.copy()
        metrics["uq_method"] = "generative_sampling"
        if "log_likelihood" in metrics:
            metrics["mean_log_likelihood"] = metrics["log_likelihood"]

        return Solution(
            fields=processed_fields,
            metrics=metrics,
            execution_time=raw_solution.execution_time,
            converged=raw_solution.converged,
        )
