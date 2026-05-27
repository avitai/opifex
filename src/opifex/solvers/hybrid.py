"""Hybrid Solver Implementation.

Combines classical and neural solvers for physics-informed correction
or residual learning.
"""

from typing import Literal

from opifex.core.problems import Problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
    SolverConfig,
    SolverState,
)


class HybridSolver:
    """Combines two solvers to produce a hybrid solution.

    Only the ``"additive"`` mode is implemented. The previous
    ``"correction"`` value was silently aliased to additive — that
    POLA-violating shadow has been removed (Rule 0 + Rule 6: fail fast
    on unsupported configurations instead of returning a misleading
    result).
    """

    def __init__(
        self,
        classical_solver: SciMLSolver,
        neural_solver: SciMLSolver,
        mode: Literal["additive"] = "additive",
    ) -> None:
        if mode != "additive":
            raise ValueError(
                f"HybridSolver only supports mode='additive'; got {mode!r}. "
                "The 'correction' mode is not yet implemented."
            )
        self.classical = classical_solver
        self.neural = neural_solver
        self.mode = mode

    def solve(
        self,
        problem: Problem,
        initial_state: SolverState | None = None,
        config: SolverConfig | None = None,
    ) -> Solution:
        """Run both solvers and combine results."""
        config = config or SolverConfig()
        state = initial_state or SolverState()

        # 1. Run Classical Solver
        sol_classical = self.classical.solve(problem, state, config)

        # 2. Run Neural Solver
        # (In a real residual scenario, we might modify the problem passed to Neural)
        sol_neural = self.neural.solve(problem, state, config)

        # 3. Combine
        combined_fields = {}
        for key in sol_classical.fields:
            if key in sol_neural.fields:
                val_c = sol_classical.fields[key]
                val_n = sol_neural.fields[key]

                combined_fields[key] = val_c + val_n

        # Merge metrics
        metrics = {
            **{f"classical_{k}": v for k, v in sol_classical.metrics.items()},
            **{f"neural_{k}": v for k, v in sol_neural.metrics.items()},
            "hybrid_error": 0.0,  # Placeholder
        }

        return Solution(
            fields=combined_fields,
            metrics=metrics,
            execution_time=sol_classical.execution_time + sol_neural.execution_time,
            converged=sol_classical.converged and sol_neural.converged,
        )
