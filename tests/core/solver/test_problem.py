"""The solve protocol: a problem, a method that solves its family, and ``solve``.

The oracle throughout is exponential decay, ``u' = -r u``, whose solution
``u0 exp(-r t)`` is known in closed form. The properties asserted are the ones the
protocol exists for: a method declares the family it solves, a mismatched pairing is
refused by name, and a problem and a method both cross ``jit``, ``vmap`` and ``grad``
-- compiling once per method configuration, never once per problem instance.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct

from opifex.core.solver import IncompatibleProblemError, Method, Problem, Solution, solve
from opifex.core.solver.status import Status


class DecayProblem(Problem):
    """``u' = -rate u`` from ``initial`` up to ``time_final``."""

    initial: jax.Array
    rate: jax.Array
    time_final: jax.Array


class StiffDecayProblem(DecayProblem):
    """A member of the decay family, so every decay method solves it."""


class OscillatorProblem(Problem):
    """A different family: ``u'' = -omega^2 u``."""

    initial: jax.Array
    omega: jax.Array


def _succeeded(fields: dict[str, jax.Array]) -> Solution:
    return Solution(fields=fields, metrics={}, status=jnp.asarray(Status.SUCCESS, jnp.int32))


class ExactDecay(Method[DecayProblem]):
    """The closed-form solution."""

    def solve(self, problem: DecayProblem) -> Solution:
        return _succeeded({"u": problem.initial * jnp.exp(-problem.rate * problem.time_final)})


class ScaledExactDecay(ExactDecay):
    """A concrete method's subclass: it inherits the family it solves."""

    scale: float = struct.field(pytree_node=False, default=1.0)

    def solve(self, problem: DecayProblem) -> Solution:
        exact = problem.initial * jnp.exp(-problem.rate * problem.time_final)
        return _succeeded({"u": self.scale * exact})


class Unbound[P: Problem](Method[P]):
    """Generic over its problem: declares no family until specialised."""

    def solve(self, problem: P) -> Solution:
        return _succeeded({})


def _decay(initial: float = 2.0, rate: float = 0.5, time_final: float = 1.0) -> DecayProblem:
    return DecayProblem(
        initial=jnp.asarray(initial), rate=jnp.asarray(rate), time_final=jnp.asarray(time_final)
    )


class TestAMethodDeclaresItsFamily:
    """The family comes from the generic argument, stated once."""

    def test_the_family_is_the_generic_argument(self) -> None:
        assert ExactDecay.problem_type is DecayProblem

    def test_a_subclass_of_a_concrete_method_inherits_it(self) -> None:
        assert ScaledExactDecay.problem_type is DecayProblem

    def test_a_generic_method_declares_none(self) -> None:
        assert Unbound.problem_type is None

    def test_a_method_without_solve_cannot_be_built(self) -> None:
        class Incomplete(Method[DecayProblem]):
            pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # pyright: ignore[reportAbstractUsage]


class TestSolveChecksThePairing:
    """A method is applied only to the family it solves."""

    def test_it_returns_what_the_method_returns(self) -> None:
        result = solve(_decay(), ExactDecay())
        np.testing.assert_allclose(float(result.fields["u"]), 2.0 * np.exp(-0.5), rtol=1e-6)
        assert bool(result.is_converged)

    def test_a_member_of_the_family_is_accepted(self) -> None:
        problem = StiffDecayProblem(
            initial=jnp.asarray(1.0), rate=jnp.asarray(50.0), time_final=jnp.asarray(0.1)
        )
        np.testing.assert_allclose(
            float(solve(problem, ExactDecay()).fields["u"]), np.exp(-5.0), rtol=1e-6
        )

    def test_another_family_is_refused_by_name(self) -> None:
        problem = OscillatorProblem(initial=jnp.asarray(1.0), omega=jnp.asarray(2.0))
        with pytest.raises(IncompatibleProblemError) as raised:
            solve(problem, ExactDecay())  # pyright: ignore[reportArgumentType]
        message = str(raised.value)
        assert "ExactDecay" in message
        assert "DecayProblem" in message
        assert "OscillatorProblem" in message

    def test_the_refusal_is_a_type_error(self) -> None:
        assert issubclass(IncompatibleProblemError, TypeError)

    def test_a_method_declaring_no_family_is_refused(self) -> None:
        with pytest.raises(IncompatibleProblemError, match="Unbound"):
            solve(_decay(), Unbound())


class TestProblemsAndMethodsAreValues:
    """Frozen pytrees: data traced, configuration static."""

    def test_a_problem_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            _decay().rate = jnp.asarray(1.0)  # pyright: ignore[reportAttributeAccessIssue]

    def test_a_method_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            ScaledExactDecay(scale=2.0).scale = 3.0  # pyright: ignore[reportAttributeAccessIssue]

    def test_problem_data_are_leaves_and_configuration_is_not(self) -> None:
        assert len(jax.tree.leaves(_decay())) == 3
        assert jax.tree.leaves(ScaledExactDecay(scale=2.0)) == []


class TestTheProtocolCrossesTransforms:
    """One compile per method configuration; problems batch and differentiate."""

    def test_jit_compiles_once_across_problem_instances(self) -> None:
        traces: list[int] = []

        @jax.jit
        def run(problem: DecayProblem, method: ScaledExactDecay) -> jax.Array:
            traces.append(1)
            return solve(problem, method).fields["u"]

        run(_decay(initial=1.0), ScaledExactDecay())
        run(_decay(initial=3.0, rate=0.1), ScaledExactDecay())
        assert len(traces) == 1
        run(_decay(), ScaledExactDecay(scale=2.0))
        assert len(traces) == 2

    def test_vmap_over_a_batch_of_problems(self) -> None:
        rates = jnp.asarray([0.1, 0.5, 2.0])
        batch = DecayProblem(initial=jnp.ones(3), rate=rates, time_final=jnp.full(3, 2.0))
        values = jax.vmap(lambda problem: solve(problem, ExactDecay()).fields["u"])(batch)
        np.testing.assert_allclose(np.asarray(values), np.exp(-2.0 * np.asarray(rates)), rtol=1e-6)

    def test_grad_with_respect_to_problem_data(self) -> None:
        def final(rate: jax.Array) -> jax.Array:
            problem = DecayProblem(initial=jnp.asarray(2.0), rate=rate, time_final=jnp.asarray(1.5))
            return solve(problem, ExactDecay()).fields["u"]

        expected = -1.5 * 2.0 * np.exp(-0.5 * 1.5)
        np.testing.assert_allclose(float(jax.grad(final)(jnp.asarray(0.5))), expected, rtol=1e-6)

    def test_the_status_batches_under_vmap(self) -> None:
        batch = DecayProblem(initial=jnp.ones(4), rate=jnp.ones(4), time_final=jnp.ones(4))
        statuses = jax.vmap(lambda problem: solve(problem, ExactDecay()).status)(batch)
        assert statuses.shape == (4,)
        assert statuses.dtype == jnp.int32
