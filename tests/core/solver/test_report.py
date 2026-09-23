"""Host-side facts about a solve: the clock, and the text.

Neither can come from traced code. A clock read inside a transform is evaluated once,
at trace time, and baked in as a constant, so the duration reported on the hundredth
call is the first compilation's. Text cannot be produced inside a transform at all.

The measurement itself belongs to :func:`calibrax.profiling.time_calls`, which owns
benchmark timing across the ecosystem; what is tested here is the part opifex adds --
turning a traced status into a sentence, and carrying the measured median alongside it.
"""

import time

import jax.numpy as jnp

from opifex.core.solver.interface import Solution
from opifex.core.solver.report import report, timed_solve
from opifex.core.solver.status import Status


def _solution(status: Status = Status.SUCCESS) -> Solution:
    return Solution(
        fields={"u": jnp.zeros(4)},
        metrics={},
        status=jnp.asarray(int(status), jnp.int32),
    )


class TestTheTimingIsMeasuredNotAssumed:
    """The reported duration has to come from running the thing."""

    def test_the_reported_time_reflects_the_work_done(self) -> None:
        # A solve that sleeps must report more than one that does not. This is the whole
        # claim: the number is a measurement, not a constant captured once.
        def slow() -> Solution:
            time.sleep(0.01)
            return _solution()

        quick_report = timed_solve(_solution, iterations=3)
        slow_report = timed_solve(slow, iterations=3)

        assert slow_report.execution_time > quick_report.execution_time
        assert slow_report.execution_time >= 0.01

    def test_the_warm_up_calls_are_not_counted(self) -> None:
        calls = {"count": 0}

        def counted() -> Solution:
            calls["count"] += 1
            return _solution()

        timed_solve(counted, warmup=2, iterations=3)

        assert calls["count"] == 5

    def test_it_reports_the_solution_the_last_timed_call_produced(self) -> None:
        statuses = [Status.MAX_ITERS, Status.MAX_ITERS, Status.SUCCESS]

        def varying() -> Solution:
            return _solution(statuses.pop(0) if statuses else Status.SUCCESS)

        produced = timed_solve(varying, warmup=0, iterations=3)

        assert produced.converged is True


class TestTheReport:
    """Reading a traced outcome on the host, once, at the boundary."""

    def test_it_converts_the_status_to_a_python_bool(self) -> None:
        produced = report(_solution())

        assert produced.converged is True
        assert produced.reason == ""

    def test_a_failure_carries_its_reason_as_text(self) -> None:
        produced = report(_solution(Status.MAX_ITERS))

        assert produced.converged is False
        assert "budget" in produced.reason

    def test_a_batch_is_converged_only_if_every_element_is(self) -> None:
        batched = Solution(
            fields={"u": jnp.zeros((3, 2))},
            metrics={},
            status=jnp.asarray([Status.SUCCESS, Status.MAX_ITERS, Status.SUCCESS], jnp.int32),
        )

        produced = report(batched)

        assert produced.converged is False
        assert len(produced.reason) == 3

    def test_timed_solve_reports_both_the_outcome_and_the_cost(self) -> None:
        produced = timed_solve(_solution, iterations=2, label="probe")

        assert produced.converged is True
        assert produced.execution_time >= 0.0
        assert produced.extra["label"] == "probe"
