"""Tests for HybridSolver implementation (TDD).

HybridSolver combines a Neural Solver (e.g. PINN/Operator) with a Classical
Numerical Solver (e.g. Diffrax, custom RK4).
This allows for 'Physics-Informed with Correction' or 'Residual Learning'.

Goal: Verify HybridSolver(neural_solver, classical_solver) orchestrates them correctly.
"""

import jax.numpy as jnp

from opifex.core.problems import create_ode_problem
from opifex.core.solver.interface import Solution
from opifex.solvers.hybrid import HybridSolver  # Will fail


class MockClassicalSolver:
    """Mock classical solver (e.g. RK4)."""

    def solve(self, problem, initial_state=None, config=None):
        # Return a known solution
        return Solution(
            fields={"u": jnp.ones((10, 1)) * 0.5},
            metrics={"error": 0.0},
            execution_time=0.1,
            converged=True,
        )


class MockNeuralSolver:
    """Mock neural solver."""

    def solve(self, problem, initial_state=None, config=None):
        return Solution(
            fields={"u": jnp.ones((10, 1)) * 0.2},  # Different value
            metrics={"loss": 0.1},
            execution_time=0.1,
            converged=True,
        )


def test_hybrid_solver_mix():
    """Test HybridSolver mixes solutions from both solvers."""
    # Scenario: Hybrid solution = alpha * Classical + (1-alpha) * Neural
    # Or: Classical + Neural (Residual)

    # Let's assume the HybridSolver takes a strategy:
    # "additive": u = u_classical + u_neural
    # "correction": u_neural learns error of u_classical

    classical = MockClassicalSolver()
    neural = MockNeuralSolver()

    # We define HybridSolver to run both and combine.
    # For TDD, let's start with a simple "Additive" composition.

    solver = HybridSolver(
        classical_solver=classical, neural_solver=neural, mode="additive"
    )

    # Create valid problem
    # Create valid problem
    problem = create_ode_problem(
        (0.0, 1.0), lambda t, y: -y, initial_conditions={"y": 1.0}
    )

    solution = solver.solve(problem)

    # Expect u = 0.5 (classical) + 0.2 (neural) = 0.7
    assert jnp.allclose(solution.fields["u"], 0.7)
    assert solution.metrics["hybrid_error"] is not None  # Check metrics combined
