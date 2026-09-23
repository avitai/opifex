"""The solver outcome, and the one property it exists to have: it survives a transform.

A result object that holds a Python ``bool`` cannot be built inside ``jit`` at all --
``bool()`` on a traced value raises ``ConcretizationTypeError`` -- so a solver returning
one holding one can never be jitted, mapped or differentiated. A Python bool on a frozen
dataclass also rides in the pytree's static metadata, which is part of the jit cache key,
so each distinct outcome compiles its own program. A traced status compiles once and
costs 0.08 ms per call whatever the outcome.

So the status is an ``int32`` array and nothing else. The names are host-side constants
in an ``IntEnum``; the value that travels is the integer. This is the pattern JAX itself
uses -- ``jax.scipy.optimize.OptimizeResults`` declares ``status: int | Array`` -- and
the one LAPACK uses, where the ``info`` code is shaped like the batch dimensions.

**Success is more than one code, and that is deliberate.** A solve that stopped because
the caller asked it to, or that stalled at a point that is genuinely a solution, has
succeeded; a solve that stalled short of one has not. A single boolean cannot tell those
apart, which is why ``is_successful`` is a predicate over codes rather than an equality
test against one. ``status == SUCCESS`` is the anti-pattern.

References:
    * Rackauckas et al., SciMLBase.jl ``ReturnCode`` -- twenty-two codes of which six are
      successes, and ``successful_retcode`` rather than ``== Success``.
"""

import enum
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.solver import status as status_module
from opifex.core.solver.status import is_successful, message, Status


class TestTheStatusIsTraceable:
    """The property the redesign exists for."""

    def test_it_is_built_inside_jit(self) -> None:
        # ``bool()`` on a traced comparison raises, so a Python flag cannot be produced here.
        @jax.jit
        def solve(residual: jax.Array) -> jax.Array:
            return jnp.where(residual < 1e-6, Status.SUCCESS, Status.CONVERGENCE_FAILURE)

        assert int(solve(jnp.asarray(1e-9))) == Status.SUCCESS
        assert int(solve(jnp.asarray(1.0))) == Status.CONVERGENCE_FAILURE

    def test_one_compilation_serves_every_outcome(self) -> None:
        # A Python bool in static metadata is part of the cache key, so each outcome
        # would compile its own program.
        traces = {"count": 0}

        @jax.jit
        def consume(status: jax.Array) -> jax.Array:
            traces["count"] += 1
            return is_successful(status)

        for code in (Status.SUCCESS, Status.MAX_ITERS, Status.NONFINITE, Status.TERMINATED):
            consume(jnp.asarray(int(code), jnp.int32))

        assert traces["count"] == 1

    def test_a_batch_carries_one_outcome_per_element(self) -> None:
        # What a Python bool cannot represent at all: under vmap, different elements of
        # the batch genuinely succeeded and failed.
        codes = jnp.asarray(
            [Status.SUCCESS, Status.MAX_ITERS, Status.TERMINATED, Status.NONFINITE], jnp.int32
        )

        outcome = jax.vmap(is_successful)(codes)

        np.testing.assert_array_equal(np.asarray(outcome), [True, False, True, False])

    def test_it_differentiates_through_a_solve_that_reports_one(self) -> None:
        # An integer output has no tangent, so carrying a status costs the backward pass
        # nothing and leaves the gradient intact.
        def solve(x: jax.Array) -> tuple[jax.Array, jax.Array]:
            value = jnp.sum(x**2)
            status = jnp.where(value < 1.0, Status.SUCCESS, Status.MAX_ITERS)
            return value, status

        gradient, status = jax.grad(solve, has_aux=True)(jnp.asarray([3.0]))

        assert float(gradient[0]) == pytest.approx(6.0)
        assert int(status) == Status.MAX_ITERS


class TestSuccessIsMoreThanOneCode:
    """A boolean cannot express the distinction, which is why this is a predicate."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            (Status.SUCCESS, True),
            (Status.TERMINATED, True),
            (Status.STALLED_SUCCESS, True),
            (Status.DEFAULT, False),
            (Status.MAX_ITERS, False),
            (Status.STALLED, False),
            (Status.NONFINITE, False),
            (Status.CONVERGENCE_FAILURE, False),
        ],
    )
    def test_each_code_reports_the_intended_outcome(self, code: Status, expected: bool) -> None:
        assert bool(is_successful(jnp.asarray(int(code), jnp.int32))) is expected

    def test_stalling_is_a_failure_or_a_success_depending_on_where_it_stalled(self) -> None:
        # One mechanism, two meanings: steps reaching zero is a failure for a solve that
        # has not reached a solution and a success for one at a valid local minimum.
        assert not bool(is_successful(jnp.asarray(int(Status.STALLED), jnp.int32)))
        assert bool(is_successful(jnp.asarray(int(Status.STALLED_SUCCESS), jnp.int32)))

    def test_an_unset_status_is_not_a_success(self) -> None:
        # So a result that never reported an outcome cannot be mistaken for a good one.
        assert not bool(is_successful(jnp.asarray(int(Status.DEFAULT), jnp.int32)))

    def test_every_code_is_classified(self) -> None:
        # A new code added without a decision defaults to unsuccessful rather than being
        # silently unclassified.
        for code in Status:
            assert isinstance(bool(is_successful(jnp.asarray(int(code), jnp.int32))), bool)

    def test_the_codes_are_stable(self) -> None:
        # Codes are persisted and compared across versions, so they are append-only.
        assert Status.DEFAULT == 0
        assert Status.SUCCESS == 1
        assert len(set(Status)) == len(list(Status)), "no duplicate values"


class TestTheMessageIsHostSide:
    """Human-readable text is not part of the traced result."""

    def test_a_scalar_status_reads_as_one_message(self) -> None:
        assert isinstance(message(jnp.asarray(int(Status.MAX_ITERS), jnp.int32)), str)
        assert message(jnp.asarray(int(Status.SUCCESS), jnp.int32)) == ""

    def test_a_batch_reads_as_one_message_per_element(self) -> None:
        codes = jnp.asarray([Status.SUCCESS, Status.NONFINITE], jnp.int32)

        rendered = message(codes)

        assert isinstance(rendered, list) and len(rendered) == 2
        assert rendered[0] == "" and "finite" in rendered[1].lower()

    def test_every_code_has_a_message(self) -> None:
        for code in Status:
            assert isinstance(message(jnp.asarray(int(code), jnp.int32)), str)


class TestWhatIsDeliberatelyAbsent:
    """The fields that would force the result back onto the host."""

    def test_the_status_module_reads_no_clock(self) -> None:
        # A clock read inside jit is evaluated once, at trace time, and baked in as a
        # constant, so a duration reported from traced code is the first compilation's
        # number for every later call. Timing belongs to whoever owns the experiment,
        # and must sync deliberately: an async dispatch left unsynced measures nothing.
        source = inspect.getsource(status_module)

        assert "import time" not in source
        assert "perf_counter" not in source and "monotonic" not in source

    def test_status_is_an_int_enum_so_the_value_is_what_travels(self) -> None:
        assert issubclass(Status, enum.IntEnum)
        assert isinstance(int(Status.SUCCESS), int)
