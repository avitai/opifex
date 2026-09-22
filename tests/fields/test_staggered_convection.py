"""Convection on the staggered layout, and the one property it is chosen for.

Written in divergence form with arithmetic-mean interpolation (Harlow, Welch 1965), the
convective operator is **skew-symmetric about a discretely divergence-free transporting
velocity**. Skew means it produces no energy: the inviscid scheme neither damps nor
amplifies, so whatever energy drift a simulation shows is the time integrator's and
vanishes as the step shrinks, rather than a spatial error that no amount of refinement
removes. First-order upwind, by contrast, is almost pure dissipation.

The qualifier is the whole of it, and it is asserted here with a negative control: the
proof holds *about a divergence-free field and only there*. Fed a transporting velocity
that is not discretely solenoidal, the same operator produces energy at order one. This
is why a projection method must project at every Runge-Kutta stage rather than once per
step -- a stage that starts from an unprojected field is outside the regime the property
covers.

References:
    * Harlow, Welch 1965 -- *Numerical calculation of time-dependent viscous incompressible
      flow of fluid with free surface*, Phys. Fluids 8(12), 2182.
    * Verstappen, Veldman 2003 -- *Symmetry-preserving discretization of turbulent flow*,
      J. Comput. Phys. 187(1), 343.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_convection import convect
from opifex.fields.staggered_pressure import project


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
RESOLUTION = (8, 8)


def _random(resolution, seed):
    shapes = StaggeredGrid.component_shapes(resolution, Extrapolation.PERIODIC)
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    return StaggeredGrid(
        tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)),
        BOX,
        Extrapolation.PERIODIC,
        resolution,
    )


def _relative_divergence(projected: StaggeredGrid, before: StaggeredGrid) -> float:
    """Residual divergence as a fraction of what the field came in with."""
    return float(jnp.linalg.norm(divergence(projected)) / jnp.linalg.norm(divergence(before)))


def _flatten(field: StaggeredGrid) -> jax.Array:
    return jnp.concatenate([component.ravel() for component in field.components])


def _unflatten(flat: jax.Array, resolution, extrapolation) -> StaggeredGrid:
    pieces, offset = [], 0
    for shape in StaggeredGrid.component_shapes(resolution, extrapolation):
        size = int(np.prod(shape))
        pieces.append(flat[offset : offset + size].reshape(shape))
        offset += size
    return StaggeredGrid(tuple(pieces), BOX, extrapolation, resolution)


def _convection_matrix(velocity: StaggeredGrid) -> np.ndarray:
    """The matrix of ``w -> convect(w, velocity)``, the transporting field held fixed."""
    size = int(_flatten(velocity).size)
    columns = []
    for index in range(size):
        basis = _unflatten(
            jnp.zeros(size).at[index].set(1.0), velocity.resolution, velocity.extrapolation
        )
        columns.append(np.asarray(_flatten(convect(basis, velocity))))
    return np.stack(columns, axis=1)


class TestSkewSymmetry:
    """No energy production -- but only about a divergence-free transporting velocity."""

    def test_it_is_skew_about_a_divergence_free_velocity(self) -> None:
        raw = _random(RESOLUTION, seed=0)
        solenoidal, _ = project(raw)
        # The precondition belongs in the units the projection guarantees: it leaves a
        # fixed fraction of the incoming divergence, not a fixed absolute norm, and the
        # incoming norm here is ~123 because differencing random noise divides by dx.
        assert _relative_divergence(solenoidal, raw) < 1e-6

        matrix = _convection_matrix(solenoidal)

        asymmetry = np.abs(matrix + matrix.T).max() / np.abs(matrix).max()
        assert asymmetry <= 1e-5

    def test_it_is_not_skew_about_a_field_with_divergence(self) -> None:
        # The negative control. Without it, an operator that is skew for the trivial
        # reason of being zero, or a test fixture that is accidentally solenoidal, would
        # pass the assertion above and prove nothing.
        divergent = _random(RESOLUTION, seed=1)
        assert float(jnp.linalg.norm(divergence(divergent))) > 1.0

        matrix = _convection_matrix(divergent)

        asymmetry = np.abs(matrix + matrix.T).max() / np.abs(matrix).max()
        assert asymmetry > 0.1

    def test_the_energy_it_produces_is_zero(self) -> None:
        # The same statement as a quadratic form, which is what actually matters to a
        # simulation: <w, C w> is the rate energy enters through the convective term.
        raw = _random(RESOLUTION, seed=2)
        solenoidal, _ = project(raw)
        assert _relative_divergence(solenoidal, raw) < 1e-6
        transported = _random(RESOLUTION, seed=3)

        produced = _flatten(transported) @ _flatten(convect(transported, solenoidal))
        scale = jnp.sum(jnp.abs(_flatten(transported) * _flatten(convect(transported, solenoidal))))

        assert abs(float(produced) / float(scale)) <= 1e-5


class TestTransforms:
    """jit, vmap and reverse-mode grad, all three."""

    def test_it_traces_once_under_jit(self) -> None:
        velocity = _random(RESOLUTION, seed=4)
        traces = {"count": 0}

        def counted(field: StaggeredGrid) -> StaggeredGrid:
            traces["count"] += 1
            return convect(field, field)

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(velocity)

        assert traces["count"] == 1

    def test_it_differentiates_in_reverse_mode(self) -> None:
        velocity = _random(RESOLUTION, seed=5)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, Extrapolation.PERIODIC, RESOLUTION)
            return jnp.sum(_flatten(convect(field, field)) ** 2)

        grads = jax.grad(loss)(velocity.components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    def test_it_maps_over_a_batch(self) -> None:
        velocity = _random(RESOLUTION, seed=6)
        batch = tuple(jnp.stack([c, c]) for c in velocity.components)

        def run(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, Extrapolation.PERIODIC, RESOLUTION)
            return convect(field, field).components[0]

        mapped = jax.vmap(run)(batch)

        assert mapped.shape == (2, *velocity.components[0].shape)


class TestBoundariesNotYetCarried:
    """What the operator refuses, and why it refuses rather than approximating."""

    @pytest.mark.parametrize("extrapolation", [Extrapolation.ZERO, Extrapolation.NEUMANN])
    def test_a_non_periodic_boundary_is_refused(self, extrapolation: Extrapolation) -> None:
        field = StaggeredGrid.zeros(RESOLUTION, BOX, extrapolation)

        with pytest.raises(ValueError, match="periodic"):
            convect(field, field)
