"""Hybrid Solver Implementation.

Combines classical and neural solvers for physics-informed correction
or residual learning.
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp

from opifex.core.problems import Problem
from opifex.core.solver.interface import (
    SciMLSolver,
    Solution,
    SolverConfig,
    SolverState,
)


def relative_field_discrepancy(classical_field: jax.Array, neural_field: jax.Array) -> jax.Array:
    """Relative L2 discrepancy between two field arrays.

    Computes ``||classical - neural|| / (||classical|| + eps)``, a non-negative
    measure of how far the neural prediction departs from the classical one on
    a shared field. It is well defined even when the classical field is zero.

    Args:
        classical_field: Field array from the classical solver.
        neural_field: Field array from the neural solver.

    Returns:
        Non-negative relative discrepancy as a scalar array.
    """
    difference_norm = jnp.linalg.norm(classical_field - neural_field)
    reference_norm = jnp.linalg.norm(classical_field) + 1e-12
    return difference_norm / reference_norm


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

    @staticmethod
    def _relative_discrepancy(classical_field: Any, neural_field: Any) -> float:
        """Relative L2 discrepancy between a classical and neural field prediction.

        Computes ``||classical - neural|| / (||classical|| + eps)``. This
        measures how much the neural solver's prediction departs from
        the classical one on a shared field (the magnitude of the correction the
        hybrid adds), and is well defined even when the classical field is zero.

        Args:
            classical_field: Field array from the classical solver.
            neural_field: Field array from the neural solver.

        Returns:
            Non-negative relative discrepancy.
        """
        return float(
            relative_field_discrepancy(jnp.asarray(classical_field), jnp.asarray(neural_field))
        )

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

        # 3. Combine and measure the classical/neural discrepancy.
        combined_fields = {}
        discrepancies: list[float] = []
        for key in sol_classical.fields:
            if key in sol_neural.fields:
                val_c = sol_classical.fields[key]
                val_n = sol_neural.fields[key]

                combined_fields[key] = val_c + val_n
                discrepancies.append(self._relative_discrepancy(val_c, val_n))

        # ``hybrid_error`` is the mean relative L2 discrepancy between the
        # classical and neural field predictions -- a measured quantity
        # computed from the two solutions. It
        # quantifies how much the two solvers disagree on the shared fields
        # (i.e. the magnitude of the neural correction relative to the classical
        # solution). It is ``0.0`` only when there are no shared fields.
        hybrid_error = float(sum(discrepancies) / len(discrepancies)) if discrepancies else 0.0

        # Merge metrics
        metrics = {
            **{f"classical_{k}": v for k, v in sol_classical.metrics.items()},
            **{f"neural_{k}": v for k, v in sol_neural.metrics.items()},
            "hybrid_error": hybrid_error,
        }

        return Solution(
            fields=combined_fields,
            metrics=metrics,
            execution_time=sol_classical.execution_time + sol_neural.execution_time,
            converged=sol_classical.converged and sol_neural.converged,
        )
