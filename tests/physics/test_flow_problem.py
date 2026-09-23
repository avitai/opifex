"""The incompressible-flow problem: what it bundles, and what it refuses.

The bundle earns its place by holding the one condition no single object can check.
A ``StaggeredGrid`` knows its own resolution and boundary; a ``WallVelocity`` knows its
own. Nothing verifies that the two agree, and a mismatch is not a shape error at
construction -- it surfaces later as a broadcast failure inside a traced step, where the
grid and the walls are already several frames apart.

What the bundle deliberately does *not* re-check is the singular pressure system. Under
all-Neumann boundaries the operator has a constant null mode, and the projection already
fixes the gauge by dropping it, guarded on both sides of the division so reverse mode
never meets a division by zero. A ``validate`` that re-asked that question would be
strictly weaker than the structure that already answers it: it would run on the host,
could not see traced data, and would duplicate an invariant the numerics enforce by
construction.

Validation lives in a classmethod rather than ``__post_init__`` because a
``struct.dataclass`` runs ``__post_init__`` on **every unflatten** -- once at
construction, again on each pytree round trip, including inside ``jit``. Checks placed
there would re-run per transform boundary.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import StaggeredGrid
from opifex.fields.staggered_diffusion import WallVelocity
from opifex.physics.flow_problem import IncompressibleFlowProblem


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))


def _problem(cells: int = 8, **overrides: object) -> IncompressibleFlowProblem:
    defaults: dict[str, object] = {
        "initial_velocity": StaggeredGrid.zeros((cells, cells), BOX, Extrapolation.ZERO),
        "viscosity": jnp.asarray(0.01),
        "total_time": jnp.asarray(0.1),
        "walls": WallVelocity.zeros((cells, cells), Extrapolation.ZERO),
        "num_steps": 4,
    }
    return IncompressibleFlowProblem.create(**(defaults | overrides))  # type: ignore[arg-type]


class TestTheSplitBetweenDataAndMetadata:
    """Which fields are traced decides both what differentiates and what recompiles."""

    def test_the_physics_is_traced_so_it_differentiates(self) -> None:
        problem = _problem()

        def final_energy(viscosity: jax.Array) -> jax.Array:
            carried = problem.replace(viscosity=viscosity)
            return jnp.sum(carried.viscosity**2) + jnp.sum(carried.total_time)

        assert float(jax.grad(final_energy)(jnp.asarray(0.01))) == pytest.approx(0.02)

    def test_the_step_count_is_static_so_it_can_size_a_scan(self) -> None:
        # A traced step count cannot set a scan length, and a traced resolution cannot
        # size an array, so both belong in the treedef.
        problem = _problem()

        assert isinstance(problem.num_steps, int)
        assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(problem)), (
            "every leaf must be an array; a Python int among them would be traced"
        )

    def test_changing_the_physics_does_not_recompile(self) -> None:
        traces = {"count": 0}

        @jax.jit
        def run(problem: IncompressibleFlowProblem) -> jax.Array:
            traces["count"] += 1
            return problem.viscosity * 2.0

        for value in (0.01, 0.02, 0.05):
            run(_problem(viscosity=jnp.asarray(value)))

        assert traces["count"] == 1

    def test_changing_the_step_count_does_recompile(self) -> None:
        # The other half of the same fact: a static field is part of the cache key. This
        # is the cost of making it static, and it is why only shape-setting fields are.
        traces = {"count": 0}

        @jax.jit
        def run(problem: IncompressibleFlowProblem) -> jax.Array:
            traces["count"] += 1
            return problem.viscosity * 2.0

        run(_problem(num_steps=4))
        run(_problem(num_steps=8))

        assert traces["count"] == 2

    def test_it_is_frozen(self) -> None:
        problem = _problem()

        with pytest.raises((AttributeError, TypeError)):
            problem.num_steps = 16  # type: ignore[misc]


class TestItRefusesAnInconsistentBundle:
    """The cross-object check that no single object can make."""

    def test_walls_built_for_another_resolution_are_refused(self) -> None:
        with pytest.raises(ValueError, match="resolution"):
            _problem(walls=WallVelocity.zeros((16, 16), Extrapolation.ZERO))

    def test_walls_built_for_another_boundary_are_refused(self) -> None:
        with pytest.raises(ValueError, match=r"boundary|extrapolation"):
            _problem(walls=WallVelocity.zeros((8, 8), Extrapolation.PERIODIC))

    def test_a_boundary_the_layer_does_not_carry_is_refused(self) -> None:
        with pytest.raises(ValueError, match="does not carry"):
            _problem(
                initial_velocity=StaggeredGrid.zeros((8, 8), BOX, Extrapolation.NEUMANN),
                walls=None,
            )

    def test_a_step_count_below_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="step"):
            _problem(num_steps=0)

    def test_a_negative_viscosity_is_refused(self) -> None:
        # Negative viscosity is anti-diffusion: the scheme is unconditionally unstable and
        # returns NaN rather than raising, so the refusal has to happen here.
        with pytest.raises(ValueError, match="viscosity"):
            _problem(viscosity=jnp.asarray(-0.01))

    def test_a_consistent_bundle_is_accepted(self) -> None:
        # The positive control: without it every refusal above could be a constructor
        # that rejects everything.
        problem = _problem()

        assert problem.num_steps == 4
        assert problem.initial_velocity.resolution == (8, 8)

    def test_walls_are_optional(self) -> None:
        problem = _problem(walls=None)

        assert problem.walls is None


class TestValidationDoesNotRunPerTransform:
    """``__post_init__`` would re-run on every unflatten, including inside ``jit``."""

    def test_crossing_a_jit_boundary_does_not_revalidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        problem = _problem()
        calls = {"count": 0}
        original = IncompressibleFlowProblem.create

        def counted(**kwargs: object) -> IncompressibleFlowProblem:
            calls["count"] += 1
            return original(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(IncompressibleFlowProblem, "create", counted)

        jax.jit(lambda carried: carried.viscosity)(problem)
        jax.tree.unflatten(*reversed(jax.tree.flatten(problem)))

        assert calls["count"] == 0, "unflatten must not re-enter the validating constructor"
        # The control: the patch is live, so a zero above means the path was not taken
        # rather than that the counter was never wired up.
        _problem()
        assert calls["count"] == 1
