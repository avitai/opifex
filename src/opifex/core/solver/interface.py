"""What a solver is.

A solver is a callable from a problem to a :class:`~opifex.core.solver.solution.Solution`,
and both are pytrees. That is the whole interface: no base class to inherit, no protocol
to satisfy beyond the signature. It is what lets the classical numerics -- free functions
over pytrees -- be solvers without giving up a transform.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from opifex.core.problems import Problem
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import Status


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverConfig:
    """Configuration for the solver.

    Held by the solver rather than passed through a transform: every field is a Python
    scalar, so this is static data and belongs in a closure or a static argument.
    """

    max_iterations: int = 1000
    tolerance: float = 1e-4
    verbose: bool = True
    # Can hold a TrainingConfig if using the Trainer
    training_config: Any | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverState:
    """State of the solver (parameters, optimizer state, etc.)."""

    params: Any | None = None
    optim_state: Any | None = None
    step: int = 0
    rng_key: Any | None = None


Solver = Callable[[Problem], Solution]
"""A solver: a callable from a problem to a solution.

Both sides are pytrees, so a solver composes with every transform. Configuration is held
by the solver rather than passed at call time, which keeps static data out of the traced
signature and lets a solver be built once and reused.

An alias rather than a protocol, on purpose. A ``runtime_checkable`` protocol checks only
that an attribute of the right *name* exists -- a class whose ``solve()`` takes no
arguments satisfies one -- while a ``Callable`` alias is checked against the signature.
"""


@runtime_checkable
class SupportsSolve(Protocol):
    """An object exposing a solver as a method, for callers that prefer objects.

    Prefer :data:`Solver`. ``isinstance`` against this confirms only that the name
    ``solve`` is present, never the signature.
    """

    def solve(self, problem: Problem) -> Solution:
        """Solve the problem and return the solution.

        Args:
            problem: The problem instance.

        Returns:
            The solution.
        """
        ...


__all__ = [
    "Solver",
    "SolverConfig",
    "SolverState",
    "Status",
    "SupportsSolve",
]
