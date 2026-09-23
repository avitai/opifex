"""The solve protocol: problems, methods, and the solutions they produce.

A :class:`Problem` states what to solve, a :class:`Method` how, and :func:`solve` applies
one to the other after checking that they belong together. Time-stepping methods build on
:func:`integrate`; every result is a :class:`Solution` whose ``status`` says how it went.
"""

from opifex.core.solver.integrate import FINAL_STATE, integrate, Snapshots
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
    "integrate",
    "is_successful",
    "solve",
]
