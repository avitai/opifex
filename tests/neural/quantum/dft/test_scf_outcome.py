"""How an SCF solve reports what happened, and why that outcome is an array.

The forward solve has two paths -- Anderson/DIIS self-consistency and direct energy
minimisation -- and both end in a backend solve that reports whether it converged. A
result that reduced that report to a Python ``bool`` could not be built inside a
transform at all, and on a frozen dataclass it would ride in the static metadata that
forms the jit cache key, so a converged and a stalled solve would compile separate
programs.

So the outcome travels as the same ``int32`` status code the rest of opifex uses
(:mod:`opifex.core.solver.status`), converted at one boundary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.solver.status import Status
from opifex.neural.quantum.dft._energy import _SOLVER_STATUS, solver_status
from opifex.neural.quantum.dft.scf import SCFResult, SCFSolver


_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067


def _h2_system() -> MolecularSystem:
    """H2 at 0.74 Angstrom on the z-axis -- the cheapest closed-shell system."""
    return MolecularSystem(
        atomic_numbers=jnp.array([1, 1]),
        positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * _BOHR_PER_ANGSTROM]]),
        basis_set="sto-3g",
    )


class TestTheStatusTranslation:
    """The backend reports an enumeration; opifex reports a code."""

    def test_a_successful_solve_is_a_success(self) -> None:
        assert int(solver_status(optx.RESULTS.successful)) == Status.SUCCESS

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [
            ("nonlinear_max_steps_reached", Status.MAX_ITERS),
            ("nonlinear_divergence", Status.CONVERGENCE_FAILURE),
            ("nonfinite", Status.NONFINITE),
            ("nonfinite_input", Status.NONFINITE),
            ("singular", Status.LINEAR_SOLVE_FAILED),
            ("max_steps_reached", Status.LINEAR_SOLVE_FAILED),
        ],
    )
    def test_each_failure_keeps_its_reason(self, outcome: str, expected: Status) -> None:
        # Reduced to a flag these would be indistinguishable, and a caller could not tell
        # "the budget ran out" from "the iteration diverged".
        assert int(solver_status(getattr(optx.RESULTS, outcome))) == expected

    def test_every_backend_outcome_is_translated(self) -> None:
        # The control on the table: an outcome with no entry falls through to DEFAULT,
        # which is not a success and carries no reason. A backend release adding an
        # outcome fails here rather than silently reporting nothing -- the count is what
        # catches the addition, since a name nobody listed cannot be looked up.
        untranslated = [
            name
            for name, _ in _SOLVER_STATUS
            if int(solver_status(getattr(optx.RESULTS, name))) == Status.DEFAULT
        ]

        assert untranslated == []
        assert len(_SOLVER_STATUS) == len(optx.RESULTS)

    def test_it_is_a_traced_array(self) -> None:
        status = solver_status(optx.RESULTS.successful)

        assert isinstance(status, jax.Array)
        assert status.dtype == jnp.int32


class TestTheResultIsAPytree:
    """``SCFResult`` holds arrays, so it survives the transforms the solver is built on."""

    def test_every_leaf_is_an_array(self) -> None:
        result = SCFSolver(_h2_system(), max_iterations=8, convergence_tolerance=1e-4).solve()
        leaves = jax.tree.leaves(result)

        assert leaves and all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_the_outcome_is_a_leaf_rather_than_metadata(self) -> None:
        # Checking only that the fields are arrays would not catch this: a static field
        # holding an int32 array is still an array, and `is_converged` would still
        # return one. What separates the two is whether the outcome is a *leaf*.
        result = SCFSolver(_h2_system(), max_iterations=8, convergence_tolerance=1e-4).solve()
        leaves = jax.tree.leaves(result)

        assert any(leaf is result.status for leaf in leaves)
        assert any(leaf is result.n_iterations for leaf in leaves)
        assert isinstance(result.is_converged, jax.Array)


class TestTheReportedOutcomeIsMeasured:
    """Both paths report the solve that actually ran, never a constant."""

    def test_a_converged_diis_solve_reports_success(self) -> None:
        with jax.enable_x64(True):
            result = SCFSolver(_h2_system()).solve()

        assert bool(result.is_converged)
        assert int(result.status) == Status.SUCCESS

    def test_a_converged_direct_solve_reports_success(self) -> None:
        with jax.enable_x64(True):
            result = SCFSolver(_h2_system(), mode="direct").solve()

        assert bool(result.is_converged)

    def test_a_starved_direct_solve_does_not_claim_convergence(self) -> None:
        # The direct path used to discard the minimiser's outcome and report
        # `converged=True` unconditionally: at two steps it stops 0.03 Ha above the
        # fixed point and still called itself converged.
        with jax.enable_x64(True):
            starved = SCFSolver(_h2_system(), mode="direct", max_iterations=2).solve()
            converged = SCFSolver(_h2_system(), mode="direct").solve()

        assert not bool(starved.is_converged)
        assert int(starved.status) == Status.MAX_ITERS
        assert float(starved.total_energy) > float(converged.total_energy) + 1e-3

    def test_a_direct_solve_reports_the_steps_it_took(self) -> None:
        # Not the iteration budget: a converged solve that stopped early must say so.
        with jax.enable_x64(True):
            result = SCFSolver(_h2_system(), mode="direct", max_iterations=100).solve()

        assert 0 < int(result.n_iterations) < 100


class TestItSurvivesATransform:
    """What holding the outcome as data buys: the solve runs inside a transform.

    The solver is built outside and closed over, since the AO basis is chosen from
    concrete atomic numbers and is static by construction; the geometry, density and
    outcome are the traced parts.
    """

    def test_a_solve_is_returned_through_jit(self) -> None:
        with jax.enable_x64(True):
            solver = SCFSolver(_h2_system())
            eager = solver.solve()
            compiled = jax.jit(solver.solve)()

        assert float(compiled.total_energy) == pytest.approx(float(eager.total_energy), abs=1e-12)
        assert int(compiled.status) == int(eager.status)

    def test_a_failing_solve_is_also_returned_through_jit(self) -> None:
        # The case a Python bool made impossible: `bool()` on a traced comparison raises
        # rather than yielding False, so a failing solve could not leave a transform.
        with jax.enable_x64(True):
            solver = SCFSolver(_h2_system(), mode="direct", max_iterations=2)
            compiled = jax.jit(solver.solve)()

        assert int(compiled.status) == Status.MAX_ITERS

    def test_both_outcomes_share_one_structure(self) -> None:
        # Different treedefs would mean a compilation per outcome for any function
        # taking a result as an argument.
        with jax.enable_x64(True):
            converged = SCFSolver(_h2_system(), mode="direct").solve()
            starved = SCFSolver(_h2_system(), mode="direct", max_iterations=2).solve()

        assert jax.tree.structure(converged) == jax.tree.structure(starved)

    def test_a_result_is_consumed_by_compiled_code_without_recompiling(self) -> None:
        traces = {"count": 0}

        @jax.jit
        def energy_above(result: SCFResult, floor: jax.Array) -> jax.Array:
            traces["count"] += 1
            return jnp.where(result.is_converged, result.total_energy - floor, jnp.inf)

        with jax.enable_x64(True):
            floor = jnp.asarray(-2.0)
            energy_above(SCFSolver(_h2_system(), mode="direct").solve(), floor)
            energy_above(SCFSolver(_h2_system(), mode="direct", max_iterations=2).solve(), floor)

        assert traces["count"] == 1


class TestTheToleranceIsReachableInThePrecisionThatRuns:
    """A tolerance below the working precision's resolution is not a tolerance.

    The default 1e-8 is meaningful in float64 and sits an order of magnitude under
    float32's epsilon of 1.19e-07, where the residual cannot reach it however long the
    iteration runs. Left unchecked the solve burns its whole budget and reports
    MAX_ITERS, which reads as a hard problem rather than an impossible request.

    SciPy sets the precedent: ``brentq`` refuses an ``rtol`` below ``4 * eps`` with
    "rtol too small" rather than pursuing it.
    """

    def test_the_default_tolerance_is_refused_at_default_precision(self) -> None:
        solver = SCFSolver(_h2_system())

        with pytest.raises(ValueError, match="below the float32 resolution"):
            solver.solve()

    def test_the_same_tolerance_is_accepted_under_x64(self) -> None:
        with jax.enable_x64(True):
            result = SCFSolver(_h2_system()).solve()

        assert int(result.status) == Status.SUCCESS

    def test_a_reachable_tolerance_is_accepted_at_default_precision(self) -> None:
        result = SCFSolver(_h2_system(), convergence_tolerance=1e-4).solve()

        assert bool(jnp.isfinite(result.total_energy))
        assert float(result.total_energy) == pytest.approx(-1.1212060, abs=1e-4)

    def test_the_message_names_the_floor_and_what_to_do(self) -> None:
        solver = SCFSolver(_h2_system())

        with pytest.raises(ValueError, match="convergence_tolerance") as raised:
            solver.solve()

        message = str(raised.value)
        assert "1e-08" in message
        assert "enable_x64" in message
