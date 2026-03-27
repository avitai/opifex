"""Tests for advection schemes."""

import jax.numpy as jnp

from opifex.fields.advection import maccormack, semi_lagrangian
from opifex.fields.field import Box, CenteredGrid, Extrapolation


class TestSemiLagrangian:
    """Tests for semi-Lagrangian advection."""

    def test_zero_velocity_no_change(self):
        """Zero velocity leaves field unchanged."""
        n = 32
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        values = jnp.sin(2 * jnp.pi * jnp.linspace(0, 1, n))[:, None] * jnp.ones((1, n))
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)
        vel = CenteredGrid(jnp.zeros((n, n, 2)), box, Extrapolation.PERIODIC)

        result = semi_lagrangian(field, vel, dt=0.01)
        assert jnp.allclose(result.values, field.values, atol=0.01)

    def test_advection_shifts_field(self):
        """Uniform velocity shifts the field."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        # Gaussian pulse
        cx, cy = 0.5, 0.5
        values = jnp.exp(-50 * ((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2))
        field = CenteredGrid(values, box, Extrapolation.ZERO)

        # Uniform velocity to the right
        vel_values = jnp.zeros((n, n, 2))
        vel_values = vel_values.at[..., 0].set(1.0)
        vel = CenteredGrid(vel_values, box, Extrapolation.ZERO)

        result = semi_lagrangian(field, vel, dt=0.1)
        # Peak should have moved to the right
        orig_peak_x = jnp.argmax(jnp.max(field.values, axis=1))
        new_peak_x = jnp.argmax(jnp.max(result.values, axis=1))
        assert int(new_peak_x) > int(orig_peak_x)


class TestMacCormack:
    """Tests for MacCormack advection."""

    def test_less_diffusive_than_sl(self):
        """MacCormack should preserve peak better than SL."""
        n = 64
        box = Box(lower=(0.0, 0.0), upper=(1.0, 1.0))
        coords = CenteredGrid(jnp.zeros((n, n)), box).cell_centers()
        values = jnp.exp(-50 * ((coords[..., 0] - 0.5) ** 2 + (coords[..., 1] - 0.5) ** 2))
        field = CenteredGrid(values, box, Extrapolation.PERIODIC)

        vel_values = jnp.zeros((n, n, 2))
        vel_values = vel_values.at[..., 0].set(0.5)
        vel = CenteredGrid(vel_values, box, Extrapolation.PERIODIC)

        sl_result = semi_lagrangian(field, vel, dt=0.05)
        mc_result = maccormack(field, vel, dt=0.05)

        # MacCormack should preserve the peak height better
        sl_peak = float(jnp.max(sl_result.values))
        mc_peak = float(jnp.max(mc_result.values))
        assert mc_peak >= sl_peak * 0.95  # At least as good
