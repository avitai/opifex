"""What a solver is, and what it returns.

A solver is a callable from a problem to a solution, and both are pytrees. That is the
whole interface: no base class to inherit, no protocol to satisfy beyond the signature.
It is what lets the classical numerics -- free functions over pytrees -- be solvers
without giving up a transform.

A ``Solution`` holds arrays and nothing else, so it can be built *inside* ``jit`` and
returned through it. Two kinds of field would prevent that and are deliberately absent.
Wall-clock time cannot be read from traced code: the clock is evaluated once at trace
time and baked in, so a reported duration would be the first compilation's for every
later call. A Python ``bool`` cannot be produced from a traced comparison at all, and on
a frozen dataclass it rides in the static metadata that forms the jit cache key, so each
outcome would compile its own program. Both belong to whoever ran the solve; see
:mod:`opifex.core.solver.report`.

Outcome is a traced status code rather than a flag; :mod:`opifex.core.solver.status` has
the reasoning.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import jax
from flax import struct

from opifex.core.problems import Problem
from opifex.core.solver.status import is_successful, Status


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


class Solution(struct.PyTreeNode, kw_only=True):
    """What a solve produces: arrays, and a code saying how it went.

    Every field is pytree data, so a ``Solution`` survives ``jit``, ``vmap``, ``grad`` and
    ``scan``. Under ``vmap`` the status carries one outcome per batch element, which a
    single flag could not express.

    Keep ``stats`` to scalar reductions. Integer counters cost the backward pass nothing,
    since an integer output has no tangent, whereas a per-step history stacked over a scan
    is residual traffic proportional to the trajectory length.

    Attributes:
        fields: The solution fields, keyed by name.
        metrics: Scalar diagnostics, such as a final loss.
        status: Why the solve stopped; an ``int32`` array, batched under ``vmap``.
        stats: Counted work, such as the number of steps taken.
        auxiliary_data: Further payloads, such as uncertainty summaries. Arrays only: a
            Python object here becomes a pytree leaf and breaks the transform.
    """

    fields: dict[str, Any]
    metrics: dict[str, Any]
    status: jax.Array
    stats: dict[str, Any] = struct.field(default_factory=dict)
    auxiliary_data: dict[str, Any] = struct.field(default_factory=dict)

    @property
    def is_converged(self) -> jax.Array:
        """Whether the status is one of the successful outcomes.

        A boolean **array**, not a Python ``bool``: under ``vmap`` it carries one answer
        per element, and it cannot drive a Python ``if`` inside traced code.
        """
        return is_successful(self.status)


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
    "Solution",
    "Solver",
    "SolverConfig",
    "SolverState",
    "Status",
    "SupportsSolve",
]
