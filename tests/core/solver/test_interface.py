"""The solver interface: a callable from a problem to a solution, both pytrees.

The properties asserted here are the ones that decide whether the classical numerics can
be solvers at all. A ``Solution`` must be constructible *inside* a transform, must batch
under ``vmap`` with one outcome per element, and must differentiate. A host value in any
field would forfeit all three.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.solver.interface import Solution, SupportsSolve
from opifex.core.solver.status import Status


def _solution(status: Status = Status.SUCCESS, value: float = 1.0) -> Solution:
    return Solution(
        fields={"u": jnp.asarray([value])},
        metrics={"loss": jnp.asarray(0.1)},
        status=jnp.asarray(int(status), jnp.int32),
    )


class TestASolutionSurvivesATransform:
    """The property the interface exists to guarantee."""

    def test_it_is_built_inside_jit_and_returned_through_it(self) -> None:
        @jax.jit
        def solve(value: jax.Array) -> Solution:
            return Solution(
                fields={"u": value * 2.0},
                metrics={"loss": jnp.sum(value)},
                status=jnp.where(jnp.sum(value) < 10.0, Status.SUCCESS, Status.MAX_ITERS),
            )

        produced = solve(jnp.asarray([1.0, 2.0]))

        assert float(produced.fields["u"][0]) == pytest.approx(2.0)
        assert int(produced.status) == Status.SUCCESS

    def test_a_batch_carries_one_outcome_per_element(self) -> None:
        @jax.vmap
        def solve(value: jax.Array) -> Solution:
            return Solution(
                fields={"u": value},
                metrics={},
                status=jnp.where(value < 5.0, Status.SUCCESS, Status.MAX_ITERS),
            )

        produced = solve(jnp.asarray([1.0, 9.0, 2.0]))

        np.testing.assert_array_equal(np.asarray(produced.is_converged), [True, False, True])

    def test_it_differentiates_through_the_fields(self) -> None:
        def loss(value: jax.Array) -> jax.Array:
            produced = Solution(
                fields={"u": value**2},
                metrics={},
                status=jnp.asarray(int(Status.SUCCESS), jnp.int32),
            )
            return jnp.sum(produced.fields["u"])

        assert float(jax.grad(loss)(jnp.asarray([3.0]))[0]) == pytest.approx(6.0)

    def test_every_leaf_is_an_array(self) -> None:
        # A Python scalar among the leaves would be traced; one in the metadata would be
        # part of the cache key. Either way the result stops behaving like data.
        leaves = jax.tree.leaves(_solution())

        assert leaves and all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_it_is_frozen(self) -> None:
        produced = _solution()

        with pytest.raises((AttributeError, TypeError)):
            produced.status = jnp.asarray(0, jnp.int32)  # type: ignore[misc]


class TestConvergenceIsAnArray:
    """``is_converged`` answers per element, so it cannot drive a Python branch."""

    def test_a_successful_status_reports_converged(self) -> None:
        assert bool(_solution(Status.SUCCESS).is_converged)

    def test_a_failed_status_does_not(self) -> None:
        assert not bool(_solution(Status.MAX_ITERS).is_converged)

    def test_an_unset_status_does_not(self) -> None:
        assert not bool(_solution(Status.DEFAULT).is_converged)


class TestTheProtocolCheckIsNameOnly:
    """Why the interface is a callable alias rather than a protocol."""

    def test_isinstance_accepts_a_wrong_signature(self) -> None:
        # A runtime_checkable protocol checks that the attribute exists, not what it
        # takes. This is the reason `Solver` is a Callable alias: a type checker compares
        # signatures, where isinstance cannot.
        class WrongSignature:
            def solve(self) -> None:  # takes no problem at all
                return None

        assert isinstance(WrongSignature(), SupportsSolve)

    def test_isinstance_rejects_a_missing_method(self) -> None:
        class NotASolver:
            pass

        assert not isinstance(NotASolver(), SupportsSolve)
