"""Analytical tests for the pseudo-spectral PDE steppers.

Ground truth comes from each PDE's conservation laws, the exact KdV soliton, and
the formal fourth-order accuracy of ETDRK4 -- not from the implementation. Burgers
conserves mass and dissipates energy; KdV conserves mass and the L2 invariant and
translates a soliton at a speed set by its amplitude; Kuramoto-Sivashinsky is
chaotic but remains bounded and (with the zero-mode fix) conserves mass.
"""

import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.spectral.steppers import (
    solve_burgers_spectral,
    solve_kdv_spectral,
    solve_kuramoto_sivashinsky_spectral,
)


def _smooth_periodic(num_points: int, domain: float, seed: int = 0) -> jax.Array:
    """A smooth, strictly periodic field built from a few low Fourier modes."""
    x = np.linspace(0.0, domain, num_points, endpoint=False)
    rng = np.random.default_rng(seed)
    field = np.zeros(num_points)
    for mode in range(1, 5):
        phase = rng.uniform(0, 2 * np.pi)
        field += rng.uniform(0.3, 1.0) * np.sin(2 * np.pi * mode * x / domain + phase)
    return jnp.asarray(field)


class TestBurgersSpectral:
    """Viscous Burgers: mass conserved, energy dissipated, ETDRK4 4th order."""

    def test_mass_is_conserved(self) -> None:
        u0 = _smooth_periodic(128, 1.0, seed=1)
        traj = solve_burgers_spectral(
            u0, 0.02, domain_extent=1.0, time_final=0.5, num_steps=300, num_snapshots=4
        )
        masses = np.asarray(traj.sum(axis=-1))
        assert np.allclose(masses, masses[0], atol=1e-5)

    def test_energy_dissipates(self) -> None:
        u0 = _smooth_periodic(128, 1.0, seed=2)
        traj = solve_burgers_spectral(
            u0, 0.05, domain_extent=1.0, time_final=0.5, num_steps=300, num_snapshots=4
        )
        energy = np.asarray((traj**2).sum(axis=-1))
        assert np.all(np.diff(energy) <= 1e-8)
        assert energy[-1] < energy[0]

    def test_fourth_order_temporal_convergence(self) -> None:
        """Halving ``dt`` cuts the error by ~16x (ETDRK4 is fourth order).

        Measured in a smooth, well-resolved regime so the temporal error -- not an
        under-resolved shock -- dominates the difference from the fine reference.
        """
        x = np.linspace(0.0, 1.0, 256, endpoint=False)
        u0 = jnp.asarray(np.sin(2.0 * np.pi * x))

        def final(num_steps: int) -> jax.Array:
            return solve_burgers_spectral(
                u0, 0.1, domain_extent=1.0, time_final=0.2, num_steps=num_steps
            )[-1]

        reference = final(8000)
        err_coarse = float(jnp.max(jnp.abs(final(50) - reference)))
        err_fine = float(jnp.max(jnp.abs(final(100) - reference)))
        order = np.log2(err_coarse / err_fine)
        assert order > 3.5

    def test_jit_and_vmap(self) -> None:
        batch = jnp.stack([_smooth_periodic(64, 1.0, seed=s) for s in range(3)])
        solve = lambda ic: solve_burgers_spectral(
            ic, 0.05, domain_extent=1.0, time_final=0.3, num_steps=100
        )
        out = jax.jit(jax.vmap(solve))(batch)
        assert out.shape == (3, 2, 64)
        assert bool(jnp.all(jnp.isfinite(out)))

    def test_matches_exact_cole_hopf_periodic_solution(self) -> None:
        """Closed-form periodic Cole-Hopf oracle ``u = -2 * nu * phi_x / phi`` where
        ``phi_t = nu * phi_xx`` converges at 4th-order ETDRK4 slope against exact truth."""
        from calibrax.analysis import scaling_fit

        num_points = 128
        domain = 1.0
        viscosity = 0.05
        time_final = 0.2
        x = np.linspace(0.0, domain, num_points, endpoint=False)
        k1 = 2.0 * np.pi / domain
        k2 = 4.0 * np.pi / domain

        def cole_hopf_exact(t: float) -> np.ndarray:
            c1 = 0.4 * np.exp(-viscosity * (k1**2) * t)
            c2 = 0.15 * np.exp(-viscosity * (k2**2) * t)
            phi = 1.0 + c1 * np.cos(k1 * x) + c2 * np.sin(k2 * x)
            phi_x = -c1 * k1 * np.sin(k1 * x) + c2 * k2 * np.cos(k2 * x)
            return -2.0 * viscosity * (phi_x / phi)

        with jax.enable_x64(True):
            u0_64 = jnp.asarray(cole_hopf_exact(0.0), dtype=jnp.float64)
            u_final_exact = jnp.asarray(cole_hopf_exact(time_final), dtype=jnp.float64)
            step_grid = [50, 100, 200]
            errs_64: list[float] = []
            for n_steps in step_grid:
                ours_64 = solve_burgers_spectral(
                    u0_64,
                    viscosity,
                    domain_extent=domain,
                    time_final=time_final,
                    num_steps=n_steps,
                    num_snapshots=1,
                )[-1]
                assert ours_64.dtype == jnp.float64
                errs_64.append(float(jnp.max(jnp.abs(ours_64 - u_final_exact))))

            assert errs_64[2] < errs_64[1] < errs_64[0] < float(jnp.max(jnp.abs(u_final_exact)))
            fit = scaling_fit([float(n) for n in step_grid], errs_64)
            assert abs((-fit.exponent) - 4.0) < 0.2

        with jax.enable_x64(False):
            u0_32 = jnp.asarray(cole_hopf_exact(0.0), dtype=jnp.float32)
            ours_32 = solve_burgers_spectral(
                u0_32,
                viscosity,
                domain_extent=domain,
                time_final=time_final,
                num_steps=200,
                num_snapshots=1,
            )[-1]
            assert ours_32.dtype == jnp.float32
            round_32 = 200.0 * 64.0 * float(jnp.finfo(jnp.float32).eps)
            assert float(jnp.max(jnp.abs(ours_32 - u_final_exact.astype(jnp.float32)))) <= (
                errs_64[2] + round_32
            )


class TestKdVSpectral:
    """KdV conserves mass and the L2 invariant; a soliton moves at speed ~ c."""

    def _soliton(self, num_points: int, domain: float, c: float, x0: float) -> jax.Array:
        # u_t + 6 u u_x + u_xxx = 0 admits u = (c/2) sech^2(sqrt(c)/2 (x - c t - x0)).
        x = np.linspace(0.0, domain, num_points, endpoint=False)
        return jnp.asarray((c / 2.0) / np.cosh(np.sqrt(c) / 2.0 * (x - x0)) ** 2)

    def test_mass_and_l2_invariants_conserved(self) -> None:
        u0 = self._soliton(256, 20.0, c=1.0, x0=20.0 / 3.0)
        traj = solve_kdv_spectral(
            u0, domain_extent=20.0, time_final=2.0, num_steps=8000, num_snapshots=4
        )
        mass = np.asarray(traj.sum(axis=-1))
        l2 = np.asarray((traj**2).sum(axis=-1))
        assert np.allclose(mass, mass[0], atol=1e-7)
        assert np.allclose(l2, l2[0], rtol=1e-3)

    def test_single_soliton_translates_at_amplitude_speed(self) -> None:
        # A unit-amplitude soliton (width ~2) is well resolved on dx = 20/256.
        num_points, domain, c, t = 256, 20.0, 1.0, 2.0
        u0 = self._soliton(num_points, domain, c, x0=domain / 3.0)
        final = solve_kdv_spectral(u0, domain_extent=domain, time_final=t, num_steps=8000)[-1]
        dx = domain / num_points
        peak_shift = (int(np.argmax(np.asarray(final))) - int(np.argmax(np.asarray(u0)))) * dx
        # KdV soliton speed equals its amplitude parameter c.
        assert abs(peak_shift - c * t) < 5 * dx
        # Shape is preserved (amplitude essentially unchanged).
        assert abs(float(final.max()) - float(u0.max())) < 0.05 * float(u0.max())


class TestKuramotoSivashinskySpectral:
    """KS is chaotic but stays bounded; the zero-mode fix conserves mass."""

    def test_remains_bounded_through_chaos(self) -> None:
        u0 = 0.1 * _smooth_periodic(256, 32.0 * np.pi, seed=5)
        traj = solve_kuramoto_sivashinsky_spectral(
            u0, domain_extent=32.0 * np.pi, time_final=40.0, num_steps=2000, num_snapshots=5
        )
        assert bool(jnp.all(jnp.isfinite(traj)))
        # Saturated chaos: amplitude is O(1), not blowing up or decaying to zero.
        assert 0.1 < float(jnp.max(jnp.abs(traj[-1]))) < 10.0

    def test_mass_is_conserved(self) -> None:
        u0 = 0.1 * _smooth_periodic(256, 32.0 * np.pi, seed=6)
        traj = solve_kuramoto_sivashinsky_spectral(
            u0, domain_extent=32.0 * np.pi, time_final=20.0, num_steps=1000, num_snapshots=4
        )
        masses = np.asarray(traj.sum(axis=-1))
        assert np.allclose(masses, masses[0], atol=1e-4)
