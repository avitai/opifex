"""Characterization tests for ConservationLawEnforcer dispatch (Task 12.3.11 B).

``conservation.py`` is the single source of truth for conservation-law
violation functions. ``ConservationLawEnforcer`` previously reimplemented
"simplified/for demonstration" bodies inline.

Only ``_compute_symmetry_conservation`` has an exact equivalent in
``conservation.py`` (``symmetry_violation``): both compute the reflection
residual ``mean((f - flip(f))**2)``. The other laws (mass / momentum /
energy / particle_number / charge) use a *different* single-state
self-consistency formulation with no two-argument ``(y_pred, y_true)``
counterpart in ``conservation.py``, so they are retained unchanged.

These tests pin:
  * symmetry == the VETTED ``conservation.symmetry_violation`` result
    (tolerance gating now applied -- a deliberate correctness fix), and
  * the remaining five laws keep their existing numeric behaviour.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.core.physics import conservation as conservation_module
from opifex.core.physics.losses import ConservationLawEnforcer


class TestSymmetryDispatch:
    """Symmetry must match conservation.symmetry_violation exactly."""

    def test_symmetric_field_matches_vetted(self) -> None:
        """A perfectly symmetric field yields the vetted (zero) violation."""
        enforcer = ConservationLawEnforcer(["symmetry"], tolerance=1e-6)
        field = jnp.array([1.0, 2.0, 3.0, 2.0, 1.0])

        residual = enforcer.compute_residual("symmetry", field)
        vetted = conservation_module.symmetry_violation(field, tolerance=1e-6)

        assert jnp.allclose(residual, vetted)
        assert jnp.allclose(residual, 0.0, atol=1e-6)

    def test_asymmetric_field_matches_vetted(self) -> None:
        """An asymmetric field matches the vetted violation value."""
        enforcer = ConservationLawEnforcer(["symmetry"], tolerance=1e-6)
        field = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])

        residual = enforcer.compute_residual("symmetry", field)
        vetted = conservation_module.symmetry_violation(field, tolerance=1e-6)

        assert jnp.allclose(residual, vetted)
        # Strong asymmetry sits well above tolerance, so it passes through.
        assert float(residual) > 0.1

    def test_batched_field_matches_vetted(self) -> None:
        """Batched 2D fields dispatch with identical results."""
        enforcer = ConservationLawEnforcer(["symmetry"], tolerance=1e-6)
        x = jnp.linspace(-1.0, 1.0, 32)
        field = jnp.tile(jnp.exp(-(x**2)), (64, 1))

        residual = enforcer.compute_residual("symmetry", field)
        vetted = conservation_module.symmetry_violation(field, tolerance=1e-6)

        assert jnp.allclose(residual, vetted)

    def test_tolerance_gating_is_honoured(self) -> None:
        """Sub-tolerance asymmetry is gated to zero by the vetted function.

        This pins the deliberate correctness change: the previous inline body
        returned the raw (tiny) residual; the vetted function zeroes any value
        below ``tolerance``.
        """
        enforcer = ConservationLawEnforcer(["symmetry"], tolerance=1e-3)
        # Asymmetry magnitude ~5e-9, far below tolerance 1e-3.
        field = jnp.array([1.0, 1.0, 1.0, 1.0 + 1e-4])

        residual = enforcer.compute_residual("symmetry", field)
        vetted = conservation_module.symmetry_violation(field, tolerance=1e-3)

        assert jnp.allclose(residual, vetted)
        assert float(residual) == 0.0


class TestRetainedLawsUnchanged:
    """The five laws without a conservation.py equivalent keep their values."""

    def test_mass_value_unchanged(self) -> None:
        """Mass residual stays mean(abs(v))**2."""
        enforcer = ConservationLawEnforcer(["mass"], tolerance=1e-6)
        velocity = jnp.array([0.1, 0.2, 0.3])

        residual = enforcer.compute_residual("mass", velocity, None)

        expected = jnp.mean(jnp.abs(velocity)) ** 2
        assert jnp.allclose(residual, expected)

    def test_momentum_value_unchanged(self) -> None:
        """Momentum residual stays mean(v**2)."""
        enforcer = ConservationLawEnforcer(["momentum"], tolerance=1e-6)
        velocity = jnp.array([1.0, 2.0, 3.0])

        residual = enforcer.compute_residual("momentum", velocity, None)

        assert jnp.allclose(residual, jnp.mean(velocity**2))

    def test_energy_value_unchanged(self) -> None:
        """Energy residual stays the Hamiltonian-variance formulation."""
        enforcer = ConservationLawEnforcer(["energy"], tolerance=1e-6)
        q = jnp.array([1.0, 0.0])
        p = jnp.array([0.0, 1.0])

        residual = enforcer.compute_residual("energy", (q, p))

        kinetic = jnp.mean(p**2) / 2
        potential = jnp.mean(q**2) / 2
        expected = jnp.var(kinetic + potential)
        assert jnp.allclose(residual, expected)

    def test_particle_number_value_unchanged(self) -> None:
        """Particle-number residual stays var(trace(density_matrix))."""
        enforcer = ConservationLawEnforcer(["particle_number"], tolerance=1e-6)
        density_matrix = jax.random.normal(jax.random.PRNGKey(0), (8, 4, 4))

        residual = enforcer.compute_residual("particle_number", density_matrix)

        expected = jnp.var(jnp.trace(density_matrix, axis1=-2, axis2=-1))
        assert jnp.allclose(residual, expected)

    def test_charge_value_unchanged(self) -> None:
        """Charge residual stays var(abs(trace(density_matrix)))."""
        enforcer = ConservationLawEnforcer(["charge"], tolerance=1e-6)
        density_matrix = jax.random.normal(jax.random.PRNGKey(1), (8, 4, 4))

        residual = enforcer.compute_residual("charge", density_matrix)

        expected = jnp.var(jnp.abs(jnp.trace(density_matrix, axis1=-2, axis2=-1)))
        assert jnp.allclose(residual, expected)

    def test_unknown_law_returns_zero(self) -> None:
        """Unknown laws still return zero."""
        enforcer = ConservationLawEnforcer(["unknown"], tolerance=1e-6)

        residual = enforcer.compute_residual("unknown", jnp.array([1.0, 2.0]))

        assert jnp.allclose(residual, 0.0)
