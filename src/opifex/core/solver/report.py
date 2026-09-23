"""Host-side facts about a solve: how long it took, and what to tell a person.

These are the things a :class:`~opifex.core.solver.interface.Solution` cannot hold. A
clock read from traced code is evaluated once, at trace time, and baked in as a constant,
so the duration reported on the hundredth call is the first compilation's. Text cannot be
produced inside a transform at all. Both are properties of *running* a solve rather than
of its result, so they are assembled here, once, at the boundary.

The measurement itself is :func:`calibrax.profiling.time_calls`, which owns benchmark
timing across the ecosystem: it discards warm-up calls, synchronises each timed call
before the clock stops -- dispatch is asynchronous, so an unsynchronised stopwatch
measures the time to *enqueue* work -- and reports a median rather than a mean, which one
slow call moves. This module adds only the solve-specific part: turning a traced status
into a sentence.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import jax
from calibrax.profiling import time_calls

from opifex.core.solver.interface import Solution
from opifex.core.solver.status import message


@dataclass(frozen=True, slots=True, kw_only=True)
class SolveReport:
    """What running a solve cost, and how it went, in host terms.

    Attributes:
        solution: The traced result the solve produced.
        execution_time: Median seconds per run, measured with a deliberate sync.
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


def timed_solve(
    solve: Callable[[], Solution],
    *,
    warmup: int = 1,
    iterations: int = 1,
    **extra: Any,
) -> SolveReport:
    """Run a solve, time it honestly, and report the outcome.

    Args:
        solve: A no-argument callable performing the solve.
        warmup: Calls made and discarded first; these absorb compilation, a cost no
            later caller pays.
        iterations: Timed calls to take the median over.
        **extra: Anything else to carry into the report.

    Returns:
        The host-side report, carrying the last solution and the median time.
    """
    produced: list[Solution] = []

    def run() -> Solution:
        solution = solve()
        produced.append(solution)
        return solution

    timing = time_calls(run, warmup=warmup, iterations=iterations)
    return report(produced[-1], execution_time=timing.median_sec, **extra)


__all__ = ["SolveReport", "report", "timed_solve"]
