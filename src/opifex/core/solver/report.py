"""Host-side facts about a solve: how long it took, and what to tell a person.

These are the things a :class:`~opifex.core.solver.interface.Solution` cannot hold. A
clock read from traced code is evaluated once, at trace time, and baked in as a constant,
so the duration reported on the hundredth call is the first compilation's. Text cannot be
produced inside a transform at all. Both are properties of *running* a solve rather than
of its result, so they are assembled here, once, at the boundary.

Timing a JAX computation needs an explicit synchronisation: dispatch is asynchronous, so
a stopwatch around an unsynchronised call measures the time to enqueue work rather than
to do it. :func:`timed_solve` synchronises deliberately, and takes the smallest of
several runs, because the minimum is the estimate least polluted by whatever else the
machine was doing.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax

from opifex.core.solver.interface import Solution
from opifex.core.solver.status import message


@dataclass(frozen=True, slots=True, kw_only=True)
class SolveReport:
    """What running a solve cost, and how it went, in host terms.

    Attributes:
        solution: The traced result the solve produced.
        execution_time: Seconds of wall clock, measured with a deliberate sync.
        converged: Whether every element succeeded. A Python bool, because it is read on
            the host; inside a transform, use ``solution.is_converged``.
        reason: The human-readable outcome, one entry per batch element.
        extra: Anything else the caller wants to carry alongside.
    """

    solution: Solution
    execution_time: float
    converged: bool
    reason: str | list[str]
    extra: dict[str, Any] = field(default_factory=dict)


def report(solution: Solution, execution_time: float = 0.0, **extra: Any) -> SolveReport:
    """Read a solution's outcome on the host.

    Forces a synchronisation, because it converts a traced status into a Python bool.

    Args:
        solution: The result of a solve.
        execution_time: Seconds the solve took, if measured.
        **extra: Anything else to carry alongside.

    Returns:
        The host-side report.
    """
    return SolveReport(
        solution=solution,
        execution_time=execution_time,
        converged=bool(jax.numpy.all(solution.is_converged)),
        reason=message(solution.status),
        extra=dict(extra),
    )


def timed_solve(solve: Callable[[], Solution], repeats: int = 1, **extra: Any) -> SolveReport:
    """Run a solve, timing it honestly, and report the outcome.

    The first run is discarded: it pays for compilation, which is not what a caller timing
    a solve wants to measure. Each timed run is synchronised before the clock is read, and
    the smallest is kept.

    Args:
        solve: A no-argument callable performing the solve.
        repeats: How many timed runs to take the minimum over.
        **extra: Anything else to carry into the report.

    Returns:
        The host-side report, carrying the last solution and the best time.

    Raises:
        ValueError: If ``repeats`` is below one, which would leave nothing to time.
    """
    if repeats < 1:
        msg = f"repeats must be at least one run, got {repeats}"
        raise ValueError(msg)

    solution = solve()
    jax.block_until_ready(solution)

    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        solution = solve()
        jax.block_until_ready(solution)
        best = min(best, time.perf_counter() - start)

    return report(solution, execution_time=best, **extra)


__all__ = ["SolveReport", "report", "timed_solve"]
