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
from tests.fields.staggered_support import (
    gradient_transform_agreement,
    gradient_versus_central_difference,
    JIT_GRADIENT_ULP,
    TAYLOR_RATE_FLOOR,
    taylor_remainder_rates,
    TAYLOR_RESIDUAL_FLOOR,
    VMAP_GRADIENT_ULP,
)


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

#: Central-difference step for the gradient check. Large on purpose: the projection is
#: linear, so a squared-norm loss of it is quadratic, its third derivative vanishes, and
#: a central difference has no truncation term. The gap is pure roundoff and so falls as
#: the step grows -- measured 1.2e-12, 9.6e-11, 4.2e-09, 3.2e-08 at h = 1e-2, 1e-4, 1e-6,
#: 1e-8. There is no U-shaped minimum to find.
CENTRAL_DIFFERENCE_STEP = 1e-2

#: Budget for that comparison. The measured worst case over both boundaries and seeds 0
#: and 5 is 2.668e-12, so this leaves about four hundred times headroom while still
#: refusing a wrong adjoint, which misses by O(1).
GRADIENT_TOLERANCE = 1e-9


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
    @pytest.mark.parametrize("seed", [0, 5])
    def test_the_gradient_matches_a_central_difference(
        self, extrapolation: Extrapolation, seed: int
    ) -> None:
        """The gradient of a solve must be the derivative, not merely finite.

        This was ``all(isfinite(g))`` alone, which a wholly wrong gradient satisfies: a
        ``stop_gradient`` placed inside a solver leaves the primal bit-identical, breaks
        the adjoint, and passes it. The reference below calls only the primal, so it is
        independent of the adjoint it checks -- which matters most here, because this is
        a gradient *through a linear solve*.

        **The step is 1e-2, not the 1e-4 the other operators take, and that is a
        property of this operator.** The projection is linear, so this squared-norm loss
        is quadratic, so its third derivative vanishes and a central difference carries
        *no truncation error at all*. The entire gap is roundoff, which falls as the
        step grows: measured in float64 the relative gap runs 1.2e-12, 9.6e-11, 4.2e-09,
        3.2e-08 over h = 1e-2, 1e-4, 1e-6, 1e-8 -- monotone, with no U-shaped minimum.
        Using 1e-4 here would throw away two orders of accuracy for nothing.
        """
        resolution = (16, 16)
        velocity = _divergent(resolution, extrapolation, seed=seed)

        with jax.enable_x64(True):
            components = tuple(c.astype(jnp.float64) for c in velocity.components)

            def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
                field = StaggeredGrid(parts, BOX, extrapolation, resolution)
                return jnp.sum(project(field)[0].components[0] ** 2)

            relative = gradient_versus_central_difference(
                loss, components, step=CENTRAL_DIFFERENCE_STEP
            )

        assert relative <= GRADIENT_TOLERANCE, (
            f"reverse mode and a central difference disagree by a relative {relative:.3e}; "
            f"the measured worst case across both boundaries and seeds is 2.7e-12, so the "
            f"{GRADIENT_TOLERANCE:.0e} budget is not what is failing here"
        )

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_the_taylor_remainder_converges_at_second_order(
        self, extrapolation: Extrapolation
    ) -> None:
        """Adjoint verification whose pass criterion is a rate, not a tolerance.

        Follows pyadjoint's ``taylor_test``, the standard check in the PDE-constrained
        optimisation literature: the remainder
        ``|J(m + eps h) - J(m) - eps <grad J, h>|`` is O(eps^2) for a correct gradient
        and O(eps) for a wrong one, so halving ``eps`` quarters the former and halves
        the latter. Rate 2 against rate 1, dimensionless, with nothing to tune -- which
        matters most for a gradient through a solve, where there is no closed form to
        compare against.

        Measured: residuals 2.49e-05, 6.23e-06, 1.56e-06, 3.89e-07, rates all 2.000.
        """
        resolution = (16, 16)
        velocity = _divergent(resolution, extrapolation)

        with jax.enable_x64(True):
            components = tuple(c.astype(jnp.float64) for c in velocity.components)

            def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
                field = StaggeredGrid(parts, BOX, extrapolation, resolution)
                return jnp.sum(project(field)[0].components[0] ** 2)

            residuals, rates = taylor_remainder_rates(loss, components)

        assert min(residuals) > TAYLOR_RESIDUAL_FLOOR, (
            f"smallest remainder {min(residuals):.3e} is at the rounding floor, so the "
            "rates describe noise"
        )
        assert min(rates) > TAYLOR_RATE_FLOOR, (
            f"Taylor rates {[f'{r:.3f}' for r in rates]} from residuals "
            f"{[f'{r:.3e}' for r in residuals]}; a rate near 1 is a wrong gradient"
        )

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    def test_the_gradient_survives_jit_and_vmap(self, extrapolation: Extrapolation) -> None:
        """Three XLA programs, not one -- and a gradient through a solve in each.

        ``test_it_traces_once_under_jit`` and ``test_it_maps_over_a_batch`` cover the
        primal only. ``vmap`` of this gradient had no coverage anywhere, and it is what a
        batched flow-control problem runs.
        """
        resolution = (16, 16)
        velocity = _divergent(resolution, extrapolation)

        def loss(parts: tuple[jax.Array, ...]) -> jax.Array:
            field = StaggeredGrid(parts, BOX, extrapolation, resolution)
            return jnp.sum(project(field)[0].components[0] ** 2)

        jit_gap, vmap_gap = gradient_transform_agreement(loss, velocity.components)

        assert jit_gap <= JIT_GRADIENT_ULP, f"jit(grad) differs from grad by {jit_gap:.2f} ULP"
        assert vmap_gap <= VMAP_GRADIENT_ULP, (
            f"vmap(grad) differs from a loop by {vmap_gap:.2f} ULP; this measured exactly "
            "zero on both the CPU and GPU backends"
        )

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


class TestAnisotropicSpacing:
    """The Poisson symbol must use each axis's own cell size.

    ``_axis_eigenvalues`` divides by ``spacing**2`` and is called with ``spacing[axis]``.
    Every other case in this module uses a square resolution in the unit square, so
    ``dx`` equals ``dy`` and ``spacing[axis]`` is numerically identical to
    ``spacing[0]`` -- the one subscript that makes the solve correct is unmeasured.

    Measured: replacing it with ``spacing[0]`` leaves the relative divergence below
    *unchanged at 1.609e-07* on the unit box, and moves it to **3.765e-01** once the box
    is twice as tall. That is six orders, against a clean value that barely moves with
    aspect ratio (1.6e-07, 2.0e-07, 2.1e-07 at aspect 1, 2, 4).

    The invariant is this module's own and needs no reference values: a projection that
    does not leave the field divergence free is not a projection. A wrong symbol still
    produces *a* field and still differentiates, so nothing else here would notice.
    """

    @pytest.mark.parametrize("extrapolation", [Extrapolation.PERIODIC, Extrapolation.ZERO])
    @pytest.mark.parametrize("aspect", [2.0, 4.0])
    def test_the_projection_is_divergence_free_on_an_anisotropic_grid(
        self, extrapolation: Extrapolation, aspect: float
    ) -> None:
        resolution = (16, 16)
        box = Box(lower=(0.0, 0.0), upper=(1.0, aspect))
        shapes = StaggeredGrid.component_shapes(resolution, extrapolation)
        keys = jax.random.split(jax.random.key(0), len(shapes))
        velocity = StaggeredGrid(
            tuple(jax.random.normal(key, shape) for key, shape in zip(keys, shapes, strict=True)),
            box,
            extrapolation,
            resolution,
        )
        spacing = velocity.dx
        assert float(spacing[0]) != float(spacing[1]), (
            "an isotropic box makes this case vacuous: it is the only thing here that "
            "distinguishes spacing[axis] from spacing[0]"
        )

        projected, _ = project(velocity)

        relative = _divergence_norm(projected) / _divergence_norm(velocity)
        assert relative <= _RESIDUAL, (
            f"relative divergence {relative:.3e} on an aspect-{aspect:g} box; a per-axis "
            "spacing defect reads 3.8e-01 here and is invisible on a square box"
        )
