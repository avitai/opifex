"""Combining a classical and a neural solver.

A solver is a callable from a problem to a solution, so the doubles below are functions
rather than objects with a ``solve`` method. That is the point of the interface: anything
with the right signature composes, including the classical numerics, which are free
functions over pytrees.

The combination stays traced, which is what most of these tests are for. A combined solver
that reduced its discrepancy metric to a Python float would sync the device once per
shared field and could not be jitted, mapped or differentiated -- defeating the reason for
combining two solvers that can.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.problems import create_ode_problem
from opifex.core.solver.interface import Solution
from opifex.core.solver.status import Status
from opifex.solvers.hybrid import additive_hybrid


def _solver(value: float, *, status: Status = Status.SUCCESS, name: str = "loss"):
    """A solver returning one constant field, for composing in the tests below."""

    def solve(problem: object) -> Solution:
        del problem
        return Solution(
            fields={"u": jnp.ones((10, 1)) * value},
            metrics={name: jnp.asarray(0.0)},
            status=jnp.asarray(int(status), jnp.int32),
        )

    return solve


def _problem():
    return create_ode_problem((0.0, 1.0), lambda t, y: -y, initial_conditions={"y": 1.0})


class TestTheCombination:
    """What the hybrid produces from its two parts."""

    def test_the_fields_add(self) -> None:
        hybrid = additive_hybrid(_solver(0.5), _solver(0.2))

        solution = hybrid(_problem())

        assert jnp.allclose(solution.fields["u"], 0.7)

    def test_the_metric_is_the_relative_discrepancy(self) -> None:
        # Classical 0.5 against neural 0.2, both constant, so the relative L2 discrepancy
        # is ||0.5 - 0.2|| / ||0.5|| = 0.6.
        hybrid = additive_hybrid(_solver(0.5), _solver(0.2))

        solution = hybrid(_problem())

        assert float(solution.metrics["hybrid_error"]) == pytest.approx(0.6, abs=1e-5)

    def test_the_metric_is_zero_when_the_solvers_agree(self) -> None:
        hybrid = additive_hybrid(_solver(0.5), _solver(0.5))

        solution = hybrid(_problem())

        assert float(solution.metrics["hybrid_error"]) == pytest.approx(0.0, abs=1e-6)

    def test_metrics_from_both_parts_are_kept_apart(self) -> None:
        hybrid = additive_hybrid(_solver(0.5, name="error"), _solver(0.2, name="loss"))

        solution = hybrid(_problem())

        assert "classical_error" in solution.metrics
        assert "neural_loss" in solution.metrics


class TestTheCombinedOutcome:
    """A combined solve succeeds only if both parts did."""

    def test_two_successes_combine_to_a_success(self) -> None:
        hybrid = additive_hybrid(_solver(0.5), _solver(0.2))

        assert bool(hybrid(_problem()).is_converged)

    def test_a_failing_part_carries_its_reason_through(self) -> None:
        # The failing code is kept rather than reduced to a flag, so a caller can see why
        # the combination is not usable.
        hybrid = additive_hybrid(_solver(0.5), _solver(0.2, status=Status.MAX_ITERS))

        solution = hybrid(_problem())

        assert int(solution.status) == Status.MAX_ITERS
        assert not bool(solution.is_converged)

    def test_the_earlier_failure_takes_precedence(self) -> None:
        hybrid = additive_hybrid(
            _solver(0.5, status=Status.NONFINITE), _solver(0.2, status=Status.MAX_ITERS)
        )

        assert int(hybrid(_problem()).status) == Status.NONFINITE


class TestItSurvivesATransform:
    """The property the callable interface exists to allow."""

    def test_it_runs_inside_jit(self) -> None:
        hybrid = additive_hybrid(_solver(0.5), _solver(0.2))
        problem = _problem()

        solution = jax.jit(lambda: hybrid(problem))()

        assert jnp.allclose(solution.fields["u"], 0.7)

    def test_it_batches_with_one_outcome_per_element(self) -> None:
        problem = _problem()

        def scaled(scale: jax.Array) -> Solution:
            hybrid = additive_hybrid(
                lambda _: Solution(
                    fields={"u": jnp.ones((4,)) * scale},
                    metrics={},
                    status=jnp.where(scale < 1.0, Status.SUCCESS, Status.MAX_ITERS),
                ),
                lambda _: Solution(
                    fields={"u": jnp.ones((4,)) * 0.1},
                    metrics={},
                    status=jnp.asarray(int(Status.SUCCESS), jnp.int32),
                ),
            )
            return hybrid(problem)

        batched = jax.vmap(scaled)(jnp.asarray([0.5, 2.0, 0.25]))

        np.testing.assert_array_equal(np.asarray(batched.is_converged), [True, False, True])

    def test_it_differentiates(self) -> None:
        problem = _problem()

        def total(scale: jax.Array) -> jax.Array:
            hybrid = additive_hybrid(
                lambda _: Solution(
                    fields={"u": jnp.ones((4,)) * scale},
                    metrics={},
                    status=jnp.asarray(int(Status.SUCCESS), jnp.int32),
                ),
                lambda _: Solution(
                    fields={"u": jnp.ones((4,)) * 0.1},
                    metrics={},
                    status=jnp.asarray(int(Status.SUCCESS), jnp.int32),
                ),
            )
            return jnp.sum(hybrid(problem).fields["u"])

        assert float(jax.grad(total)(jnp.asarray(2.0))) == pytest.approx(4.0)
