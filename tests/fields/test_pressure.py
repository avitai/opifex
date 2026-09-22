"""Incompressible projection: the pressure solve and the field it returns.

A projection is only as good as the operator it inverts. Both solvers here invert
``divergence(gradient(.))`` -- the exact composition the projection then applies -- so the
contract every test below states is the strong one: the projected field is divergence free
under the same ``divergence`` that measured it, to solver accuracy rather than by some margin.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.fields.field import Box, CenteredGrid, Extrapolation
from opifex.fields.operations import divergence, gradient
from opifex.fields.pressure import pressure_solve_lsmr, pressure_solve_spectral


BOX = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))


def _velocity(n: int, extrapolation: Extrapolation = Extrapolation.PERIODIC) -> CenteredGrid:
    """A smooth field with genuine divergence, and more than one active wavenumber."""
    coords = CenteredGrid(jnp.zeros((n, n)), BOX).cell_centers()
    x, y = coords[..., 0], coords[..., 1]
    vx = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y) + 0.3 * jnp.sin(4 * jnp.pi * x)
    vy = jnp.cos(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
    return CenteredGrid(jnp.stack([vx, vy], axis=-1), BOX, extrapolation)


def _solenoidal(n: int) -> CenteredGrid:
    """A field that is already divergence free: the curl of a stream function."""
    coords = CenteredGrid(jnp.zeros((n, n)), BOX).cell_centers()
    x, y = coords[..., 0], coords[..., 1]
    vx = -jnp.sin(2 * jnp.pi * y)
    vy = jnp.sin(2 * jnp.pi * x)
    return CenteredGrid(jnp.stack([vx, vy], axis=-1), BOX, Extrapolation.PERIODIC)


# A projection applies two differences, each dividing by dx, so float32 round-off in the
# residual grows like eps * n^2 and a single number cannot bound it across resolutions.
# Measured relative residual over eps * n^2, which is flat and so is the right model:
#
#     n            16       32       64      128
#     spectral  0.0108   0.0071   0.0099   0.0071
#     lsmr      0.0495   0.0244   0.0282   0.0397
#
# Each limit is five times the largest measured for that solver. A flat limit wide enough
# for n = 128 would be eighty times the value it exists to catch at n = 16. The iterative
# solve floors higher because it stops at a fixed iteration count rather than inverting the
# symbol exactly.
_EPSILON = float(jnp.finfo(jnp.float32).eps)


def _spectral_limit(n: int) -> float:
    """The relative residual divergence the exact symbol solve cannot go below."""
    return 0.05 * _EPSILON * n * n


def _iterative_limit(n: int) -> float:
    """The relative residual divergence the fixed-budget least-squares solve reaches."""
    return 0.25 * _EPSILON * n * n


def _divergence_norm(field: CenteredGrid) -> float:
    return float(jnp.linalg.norm(divergence(field).values))


class TestSpectralPressureSolve:
    """The FFT solve, which inverts the operator's symbol exactly."""

    @pytest.mark.parametrize("n", [16, 32, 64])
    def test_the_projected_field_is_divergence_free(self, n: int) -> None:
        velocity = _velocity(n)

        projected, _ = pressure_solve_spectral(velocity)

        assert _divergence_norm(projected) <= _spectral_limit(n) * _divergence_norm(velocity)

    def test_the_pressure_solves_the_operator_the_projection_applies(self) -> None:
        # Operator consistency: p solves divergence(gradient(p)) = divergence(v), not some
        # other discrete Laplacian that merely approximates it.
        velocity = _velocity(32)

        _, pressure = pressure_solve_spectral(velocity)

        residual = divergence(gradient(pressure)).values - divergence(velocity).values
        assert float(jnp.linalg.norm(residual)) <= _spectral_limit(32) * _divergence_norm(velocity)

    def test_a_divergence_free_field_is_left_alone(self) -> None:
        velocity = _solenoidal(32)

        projected, pressure = pressure_solve_spectral(velocity)

        # Neither component varies along the axis it is differentiated on, so the
        # divergence is a subtraction of identical values and is zero exactly, not nearly.
        assert jnp.array_equal(projected.values, velocity.values)
        assert jnp.array_equal(pressure.values, jnp.zeros_like(pressure.values))

    def test_it_refuses_boundaries_it_cannot_represent(self) -> None:
        with pytest.raises(ValueError, match="periodic"):
            pressure_solve_spectral(_velocity(16, Extrapolation.NEUMANN))

    def test_it_traces_under_jit(self) -> None:
        velocity = _velocity(16)

        projected, _ = jax.jit(pressure_solve_spectral)(velocity)

        assert _divergence_norm(projected) <= _spectral_limit(16) * _divergence_norm(velocity)

    def test_it_maps_over_a_batch_of_fields(self) -> None:
        velocity = _velocity(16)
        batch = jnp.stack([velocity.values, 2.0 * velocity.values])

        def project(values: jax.Array) -> jax.Array:
            field = CenteredGrid(values, BOX, Extrapolation.PERIODIC)
            return pressure_solve_spectral(field)[0].values

        projected = jax.vmap(project)(batch)

        assert projected.shape == batch.shape
        # The projection is linear, so doubling the input doubles the output.
        assert jnp.allclose(projected[1], 2.0 * projected[0], atol=1e-5)


class TestLeastSquaresPressureSolve:
    """The matrix-free solve, which carries any boundary the operators support."""

    @pytest.mark.parametrize(
        "extrapolation",
        [Extrapolation.PERIODIC, Extrapolation.NEUMANN, Extrapolation.ZERO],
    )
    def test_the_projected_field_is_divergence_free(self, extrapolation: Extrapolation) -> None:
        velocity = _velocity(32, extrapolation)

        projected, _ = pressure_solve_lsmr(velocity)

        assert _divergence_norm(projected) <= _iterative_limit(32) * _divergence_norm(velocity)

    def test_it_agrees_with_the_spectral_solve_on_a_periodic_field(self) -> None:
        velocity = _velocity(32)

        from_cg, _ = pressure_solve_lsmr(velocity)
        from_fft, _ = pressure_solve_spectral(velocity)

        assert jnp.allclose(from_cg.values, from_fft.values, atol=1e-5)

    def test_a_divergence_free_field_is_left_alone(self) -> None:
        velocity = _solenoidal(32)

        projected, pressure = pressure_solve_lsmr(velocity)

        assert jnp.array_equal(projected.values, velocity.values)
        assert jnp.array_equal(pressure.values, jnp.zeros_like(pressure.values))

    def test_more_iterations_leave_less_divergence(self) -> None:
        velocity = _velocity(32, Extrapolation.NEUMANN)

        short, _ = pressure_solve_lsmr(velocity, num_matvecs=20)
        long, _ = pressure_solve_lsmr(velocity, num_matvecs=500)

        assert _divergence_norm(long) < 0.01 * _divergence_norm(short)

    def test_it_traces_under_jit(self) -> None:
        velocity = _velocity(16)

        projected, _ = jax.jit(pressure_solve_lsmr)(velocity)

        assert _divergence_norm(projected) <= _iterative_limit(16) * _divergence_norm(velocity)
