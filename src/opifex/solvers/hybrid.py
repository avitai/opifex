"""Combining a classical and a neural solver into one.

The combination is additive: the neural field is a correction added to the classical one,
and the discrepancy between them is reported as a metric so a caller can see how large
that correction is.

Everything stays in arrays. Reducing the discrepancy to a Python float would sync the
device mid-solve, once per shared field, and would make the combined solver impossible to
jit, map or differentiate -- which is the whole reason for combining two solvers that can.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from opifex.core.problems import Problem
from opifex.core.solver.interface import Solution, Solver
from opifex.core.solver.status import combine_statuses


def relative_field_discrepancy(classical_field: jax.Array, neural_field: jax.Array) -> jax.Array:
    """Relative L2 discrepancy between two field arrays.

    Computes ``||classical - neural|| / (||classical|| + eps)``, a non-negative measure of
    how far the neural prediction departs from the classical one on a shared field. It is
    well defined even when the classical field is zero.

    Args:
        classical_field: Field array from the classical solver.
        neural_field: Field array from the neural solver.

    Returns:
        Non-negative relative discrepancy as a scalar array.
    """
    difference_norm = jnp.linalg.norm(classical_field - neural_field)
    reference_norm = jnp.linalg.norm(classical_field) + 1e-12
    return difference_norm / reference_norm


def additive_hybrid(classical: Solver, neural: Solver) -> Solver:
    """A solver that adds a neural correction to a classical solution.

    Fields present in both solutions are summed; a field present in only one is dropped,
    since there is nothing to correct it with. The mean relative discrepancy over the
    shared fields is reported as the ``hybrid_error`` metric.

    Args:
        classical: The solver producing the base solution.
        neural: The solver producing the correction.

    Returns:
        A solver combining the two.
    """

    def solve(problem: Problem) -> Solution:
        base = classical(problem)
        correction = neural(problem)

        shared = [name for name in base.fields if name in correction.fields]
        combined = {name: base.fields[name] + correction.fields[name] for name in shared}
        discrepancies = [
            relative_field_discrepancy(
                jnp.asarray(base.fields[name]), jnp.asarray(correction.fields[name])
            )
            for name in shared
        ]
        # Zero when the two solvers share no field: there is no disagreement to report
        # rather than an undefined one.
        hybrid_error = jnp.mean(jnp.stack(discrepancies)) if discrepancies else jnp.asarray(0.0)

        metrics = {
            **{f"classical_{name}": value for name, value in base.metrics.items()},
            **{f"neural_{name}": value for name, value in correction.metrics.items()},
            "hybrid_error": hybrid_error,
        }
        return Solution(
            fields=combined,
            metrics=metrics,
            status=combine_statuses(base.status, correction.status),
            stats={
                **{f"classical_{name}": value for name, value in base.stats.items()},
                **{f"neural_{name}": value for name, value in correction.stats.items()},
            },
        )

    return solve


HybridSolver: Callable[[Solver, Solver], Solver] = additive_hybrid
"""Alias for :func:`additive_hybrid`, naming the combination rather than the rule."""


__all__ = ["HybridSolver", "additive_hybrid", "relative_field_discrepancy"]
