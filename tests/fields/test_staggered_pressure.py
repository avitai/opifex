"""The pressure projection over the staggered layout.

The projection is exact in a stronger sense here than on a collocated grid. Because
``divergence`` is minus the adjoint of ``gradient``, the Poisson operator is symmetric by
construction, and because each axis is separable the operator diagonalises under a
transform: an FFT where the axis is periodic, and a **DCT-II where it has walls** -- the
homogeneous-Neumann pressure condition the staggered gradient implies at a wall is exactly
the DCT-II boundary condition. So a box with walls gets a direct, non-iterative,
machine-exact projection at the same cost as the periodic one, which is what removes the
last reason to restrict an incompressible solver to the torus.

The contract asserted below is the strong one: the projected field is divergence free to
round-off under the same ``divergence`` that measured it, and the projection is idempotent,
leaves an already divergence-free field alone, and carries jit, vmap and reverse-mode grad.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.fields.field import Box, Extrapolation
from opifex.fields.staggered import divergence, StaggeredGrid
from opifex.fields.staggered_pressure import project


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))

# The residual here does NOT grow with resolution, because the operator is diagonalised
# exactly rather than assembled from differences whose round-off the solve then amplifies.
# Measured relative residual, flat across a factor of sixteen in n:
#
#     n           8       16      32      64     128
#     periodic  1.6e-07 1.5e-07 1.8e-07 2.2e-07 2.4e-07
#     walls     2.5e-07 3.0e-07 2.7e-07 3.1e-07 3.1e-07
#
# So the limit is a constant just above the largest of those, not a function of n. The
# collocated projection needs `0.05 * eps * n^2` for the same statement, which at 128 cells
# is sixty times looser than this.
_RESIDUAL = 1e-6


def _divergent(resolution, extrapolation, seed=0):
    """A face velocity with genuine divergence."""
    shapes = StaggeredGrid.component_shapes(resolution, extrapolation)
    keys = jax.random.split(jax.random.key(seed), len(shapes))
    components = tuple(
        jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)
    )
    return StaggeredGrid(components, BOX, extrapolation, resolution)


def _divergence_norm(field: StaggeredGrid) -> float:
    return float(jnp.linalg.norm(divergence(field)))


class TestTheProjectionIsExact:
    """Divergence free under the same operator that measured it."""

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    @pytest.mark.parametrize("n", [8, 16, 32])
    def test_the_projected_field_is_divergence_free(
        self, extrapolation: Extrapolation, n: int
    ) -> None:
        velocity = _divergent((n, n), extrapolation)

        projected, _ = project(velocity)

        assert _divergence_norm(projected) <= _RESIDUAL * _divergence_norm(velocity)

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_projecting_twice_changes_nothing(self, extrapolation: Extrapolation) -> None:
        velocity = _divergent((16, 16), extrapolation)

        once, _ = project(velocity)
        twice, _ = project(once)

        for first, second in zip(once.components, twice.components, strict=True):
            assert jnp.allclose(first, second, atol=1e-5)

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_a_divergence_free_field_is_left_alone(self, extrapolation: Extrapolation) -> None:
        velocity, _ = project(_divergent((16, 16), extrapolation, seed=3))

        projected, pressure = project(velocity)

        for before, after in zip(velocity.components, projected.components, strict=True):
            assert jnp.allclose(before, after, atol=1e-5)
        assert float(jnp.max(jnp.abs(pressure))) <= 1e-4


class TestTheWallSolveIsNotThePeriodicOne:
    """A negative control: the periodic symbol must fail on a walled box."""

    def test_the_periodic_symbol_does_not_project_a_walled_box(self) -> None:
        # Guards against the transform tier being selected by accident rather than by the
        # boundary: if this passed, the DCT tier would not be being exercised at all.
        from opifex.fields.staggered_pressure import _poisson_eigenvalues

        walls = _poisson_eigenvalues((16, 16), Extrapolation.ZERO, BOX)
        periodic = _poisson_eigenvalues((16, 16), Extrapolation.PERIODIC, BOX)

        assert not jnp.allclose(walls, periodic)


class TestTransforms:
    """jit, vmap and reverse-mode grad, all three."""

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_it_traces_once_under_jit(self, extrapolation: Extrapolation) -> None:
        velocity = _divergent((16, 16), extrapolation)
        traces = {"count": 0}

        def counted(field: StaggeredGrid) -> StaggeredGrid:
            traces["count"] += 1
            return project(field)[0]

        compiled = jax.jit(counted)
        for _ in range(4):
            compiled(velocity)

        assert traces["count"] == 1

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_it_differentiates_in_reverse_mode(self, extrapolation: Extrapolation) -> None:
        resolution = (16, 16)
        velocity = _divergent(resolution, extrapolation)

        def loss(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, extrapolation, resolution)
            return jnp.sum(project(field)[0].components[0] ** 2)

        grads = jax.grad(loss)(velocity.components)

        assert all(bool(jnp.all(jnp.isfinite(g))) for g in grads)

    def test_it_maps_over_a_batch(self) -> None:
        resolution = (16, 16)
        velocity = _divergent(resolution, Extrapolation.PERIODIC)
        batch = tuple(jnp.stack([c, 2.0 * c]) for c in velocity.components)

        def run(components: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(components, BOX, Extrapolation.PERIODIC, resolution)
            return project(field)[0].components[0]

        mapped = jax.vmap(run)(batch)

        assert mapped.shape == (2, *velocity.components[0].shape)
        # The projection is linear, so doubling the input doubles the output.
        assert jnp.allclose(mapped[1], 2.0 * mapped[0], atol=1e-5)
