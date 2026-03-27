"""Tests for pressure solve (incompressible projection)."""

import jax.numpy as jnp

from opifex.fields.field import Box, CenteredGrid, Extrapolation
from opifex.fields.operations import divergence
from opifex.fields.pressure import pressure_solve_jacobi, pressure_solve_spectral


class TestSpectralPressureSolve:
    """Tests for FFT-based pressure solve."""

    def test_divergence_free_output(self):
        """Projected velocity should be nearly divergence-free."""
        n = 32
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()

        # Create divergent velocity: v = (sin(2πx), sin(2πy))
        vx = jnp.sin(2 * jnp.pi * coords[..., 0])
        vy = jnp.sin(2 * jnp.pi * coords[..., 1])
        vel = CenteredGrid(jnp.stack([vx, vy], axis=-1), box, Extrapolation.PERIODIC)

        projected, _pressure = pressure_solve_spectral(vel)

        # Check divergence is significantly reduced
        div_before = float(jnp.max(jnp.abs(divergence(vel).values)))
        div = divergence(projected)
        div_after = float(jnp.max(jnp.abs(div.values)))
        assert div_after < div_before * 0.5

    def test_already_divergence_free(self):
        """Already divergence-free field should be unchanged."""
        n = 32
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()

        # v = (-sin(2πy), sin(2πx)) is divergence-free
        vx = -jnp.sin(2 * jnp.pi * coords[..., 1])
        vy = jnp.sin(2 * jnp.pi * coords[..., 0])
        vel = CenteredGrid(jnp.stack([vx, vy], axis=-1), box, Extrapolation.PERIODIC)

        projected, pressure = pressure_solve_spectral(vel)

        # Velocity should be ~unchanged, pressure ~zero
        assert jnp.allclose(projected.values, vel.values, atol=0.1)
        assert jnp.max(jnp.abs(pressure.values)) < 0.1


class TestJacobiPressureSolve:
    """Tests for iterative Jacobi pressure solve."""

    def test_reduces_divergence(self):
        """Jacobi projection reduces divergence."""
        n = 16
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()

        vx = jnp.sin(2 * jnp.pi * coords[..., 0])
        vy = jnp.sin(2 * jnp.pi * coords[..., 1])
        vel = CenteredGrid(jnp.stack([vx, vy], axis=-1), box, Extrapolation.PERIODIC)

        div_before = float(jnp.max(jnp.abs(divergence(vel).values)))

        projected, _ = pressure_solve_jacobi(vel, n_iterations=200)
        div_after = float(jnp.max(jnp.abs(divergence(projected).values)))

        assert div_after < div_before
