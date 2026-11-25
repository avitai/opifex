"""The solve protocol: problems, methods, and the solutions they produce.

A :class:`Problem` states what to solve, a :class:`Method` how, and :func:`solve` applies
one to the other after checking that they belong together. Time-stepping methods build on
:func:`integrate`; every result is a :class:`Solution` whose ``status`` says how it went.
"""

from opifex.core.solver.implicit import (
    call_unary_or_binary,
    cast_inexact_pytree,
    dense_tangent_solve,
    gmres_tangent_solve,
    guard_unconverged_tangents,
    infer_inexact_dtype,
    loop_status,
    lsmr_tangent_solve,
    TangentSolverFn,
)
from opifex.core.solver.integrate import FINAL_STATE, integrate, Snapshots
from opifex.core.solver.loop import bounded_while_loop, unvmap_any
from opifex.core.solver.lsmr import lsmr
from opifex.core.solver.problem import IncompatibleProblemError, Method, Problem, solve
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import is_successful, Status


__all__ = [
    "FINAL_STATE",
    "IncompatibleProblemError",
    "Method",
    "Problem",
    "Snapshots",
    "Solution",
    "Status",
    "TangentSolverFn",
    "bounded_while_loop",
    "call_unary_or_binary",
    "cast_inexact_pytree",
    "dense_tangent_solve",
    "gmres_tangent_solve",
    "guard_unconverged_tangents",
    "infer_inexact_dtype",
    "integrate",
    "is_successful",
    "loop_status",
    "lsmr",
    "lsmr_tangent_solve",
    "solve",
    "unvmap_any",
]
