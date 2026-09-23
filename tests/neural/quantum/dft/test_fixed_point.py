"""Tests for the Anderson-accelerated fixed-point solver.

Validates convergence on a stiff scalar map, the exact implicit-function-theorem
gradient through :func:`optimistix.fixed_point`, and robust convergence of the
water/LDA Kohn-Sham density where plain Roothaan iteration diverges.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from opifex.neural.quantum.dft._fixed_point import AndersonAcceleration


def test_converges_scalar_fixed_point() -> None:
    """Recovers the cos fixed point ``x = cos(x)`` to high precision."""
    with jax.enable_x64(True):
        solver = AndersonAcceleration(rtol=1e-12, atol=1e-12, history_size=5)
        solution = optx.fixed_point(
            lambda x, _: jnp.cos(x), solver, jnp.array(0.0), max_steps=64, throw=False
        )
    assert float(solution.value) == pytest.approx(0.7390851332151607, abs=1e-9)


def test_converges_oscillatory_map() -> None:
    """Anderson stabilises an oscillatory map ``f(x) = sqrt(2) cos(x)``.

    The derivative at the fixed point has magnitude ``> 0`` and the iteration
    spirals; Anderson mixing damps the oscillation to convergence.
    """
    with jax.enable_x64(True):

        def stiff(x: jax.Array, _: None) -> jax.Array:
            return jnp.sqrt(2.0) * jnp.cos(x)

        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=6)
        solution = optx.fixed_point(stiff, solver, jnp.array(0.5), max_steps=64, throw=False)
        residual = float(jnp.abs(stiff(solution.value, None) - solution.value))
    assert residual < 1e-9


def test_implicit_gradient_is_exact() -> None:
    r"""IFT gradient through Anderson matches the analytic ``dy*/dtheta``.

    For ``f(y, theta) = 0.5 y + theta`` the fixed point is ``y* = 2 theta`` so
    ``dy*/dtheta = 2``, independent of the forward solver. Confirms that using
    Anderson as the forward iterator leaves the :class:`optimistix.ImplicitAdjoint`
    gradient exact.
    """
    with jax.enable_x64(True):

        def solve(theta: jax.Array) -> jax.Array:
            solver = AndersonAcceleration(rtol=1e-12, atol=1e-12, history_size=4)
            return optx.fixed_point(
                lambda y, t: 0.5 * y + t, solver, jnp.array(0.0), args=theta, max_steps=64
            ).value

        value = solve(jnp.array(1.5))
        gradient = jax.grad(solve)(jnp.array(1.5))
    assert float(value) == pytest.approx(3.0, abs=1e-9)
    assert float(gradient) == pytest.approx(2.0, abs=1e-9)


def test_mixing_coefficients_sum_to_one() -> None:
    """The Pulay/DIIS coefficients are an affine combination (sum to one)."""
    with jax.enable_x64(True):
        solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
        residuals = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]], dtype=jnp.float64)
        coefficients = solver._mixing_coefficients(residuals, jnp.array(3))
    assert float(jnp.sum(coefficients)) == pytest.approx(1.0, abs=1e-10)
    # The unfilled fourth slot receives zero weight.
    assert float(coefficients[3]) == pytest.approx(0.0, abs=1e-12)


def test_is_jit_and_vmap_compatible() -> None:
    """The full solve jits and vmaps over a batch of fixed-point problems."""
    with jax.enable_x64(True):

        def solve(theta: jax.Array) -> jax.Array:
            solver = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=4)
            return optx.fixed_point(
                lambda y, t: 0.5 * y + t, solver, jnp.array(0.0), args=theta, max_steps=64
            ).value

        thetas = jnp.array([1.0, 2.0, 3.0])
        batched = jax.jit(jax.vmap(solve))(thetas)
    assert jnp.allclose(batched, 2.0 * thetas, atol=1e-9)


@pytest.mark.slow
def test_converges_water_lda_density() -> None:
    """Anderson converges the water/LDA SCF density that plain Roothaan cannot.

    Plain Roothaan iteration stalls at a density residual of ~0.9 for water/LDA;
    Anderson drives it below 1e-9 in a few iterations.
    """
    from opifex.core.quantum.molecular_system import MolecularSystem
    from opifex.neural.quantum.dft._energy import _density_from_fock, _scf_step
    from opifex.neural.quantum.dft.grid import build_molecular_grid_traceable
    from opifex.neural.quantum.dft.scf import SCFSolver

    with jax.enable_x64(True):
        positions = jnp.array([[0.0, 0.0, 0.0], [0.0, 1.43, 1.11], [0.0, -1.43, 1.11]])
        system = MolecularSystem(
            atomic_numbers=jnp.array([8, 1, 1]), positions=positions, basis_set="sto-3g"
        )
        grid = build_molecular_grid_traceable(system, n_radial=8, n_theta=6, n_phi=8)
        solver_obj = SCFSolver(system, grid_template=grid)
        integrals = solver_obj._integrals(positions)
        functional = solver_obj._functional
        n_occupied = solver_obj._n_occupied

        initial = _density_from_fock(
            integrals.core_hamiltonian, integrals.orthogonaliser, n_occupied
        )[0]
        anderson = AndersonAcceleration(rtol=1e-10, atol=1e-10, history_size=6)
        solution = optx.fixed_point(
            lambda density, _: _scf_step(density, integrals, functional, n_occupied, None),
            anderson,
            initial,
            max_steps=100,
            throw=False,
        )
        converged = solution.value
        residual = jnp.max(
            jnp.abs(_scf_step(converged, integrals, functional, n_occupied, None) - converged)
        )
    assert float(residual) < 1e-9


class TestTheMixingSolveAtDefaultPrecision:
    """The Tikhonov scale has to be measured in the arithmetic that runs.

    Every test above enables x64, so the default-precision path was never exercised.
    In float32 the residuals of a converging SCF reach the arithmetic's own noise floor
    (~1e-7) while the tolerance is still unmet, and the Gram matrix of near-identical
    noise vectors is numerically singular: a stabiliser too small to perturb it leaves
    the mixing solve to return non-finite coefficients.
    """

    @staticmethod
    def _rank_deficient_history(dtype: jnp.dtype) -> jax.Array:
        """Six residuals differing only at the last representable digit."""
        base = jnp.linspace(0.1, 0.9, 16).astype(dtype)
        noise = jnp.finfo(dtype).eps * jnp.arange(1, 7, dtype=dtype)[:, None]
        return (base[None, :] * (1.0 + noise)).astype(dtype)

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_coefficients_stay_finite_on_a_rank_deficient_history(self, dtype: jnp.dtype) -> None:
        with jax.enable_x64(True):
            solver = AndersonAcceleration(rtol=1e-8, atol=1e-8, history_size=6)
            residuals = self._rank_deficient_history(dtype)

            coefficients = solver._mixing_coefficients(residuals, jnp.array(6))

        assert bool(jnp.all(jnp.isfinite(coefficients)))
        assert float(jnp.sum(coefficients)) == pytest.approx(1.0, abs=1e-5)

    def test_the_scale_comes_from_the_residuals_not_the_empty_slots(self) -> None:
        # Unfilled slots are masked into the Gram as the identity, whose entries are 1
        # while a converging residual's are ~1e-12. A scale taken from the padded trace
        # therefore regularises by many times the data while the history fills.
        #
        # The two residuals are orthogonal with very different norms, so the exact
        # coefficients are c ~ (1/a, 1/b) normalised -- far from equal. A stabiliser
        # much larger than both drives them to (0.5, 0.5) instead, which is what a
        # padded scale produces and what equal-magnitude residuals would hide.
        with jax.enable_x64(True):
            solver = AndersonAcceleration(rtol=1e-8, atol=1e-8, history_size=6)
            residuals = jnp.zeros((6, 4), jnp.float64)
            residuals = residuals.at[0, 0].set(1e-6).at[1, 1].set(1e-7)

            coefficients = solver._mixing_coefficients(residuals, jnp.array(2))

        # a = 1e-12, b = 1e-14: c = (1/a, 1/b) / (1/a + 1/b) = (1/101, 100/101).
        assert float(coefficients[0]) == pytest.approx(1.0 / 101.0, rel=1e-3)
        assert float(coefficients[1]) == pytest.approx(100.0 / 101.0, rel=1e-3)
        assert float(jnp.sum(coefficients[2:])) == pytest.approx(0.0, abs=1e-12)

    def test_an_oscillatory_map_converges_without_x64(self) -> None:
        # x = sqrt(2) cos(x), whose root is 0.8900586117066737 to float64 precision
        # (Brent, residual 4e-16). At default precision the old fixed stabiliser left
        # this solve returning NaN after nine steps.
        def stiff(x: jax.Array, _: None) -> jax.Array:
            return jnp.sqrt(2.0) * jnp.cos(x)

        solver = AndersonAcceleration(rtol=1e-5, atol=1e-6, history_size=5)

        solution = optx.fixed_point(
            stiff, solver, jnp.array(0.0, jnp.float32), max_steps=128, throw=False
        )
        residual = float(jnp.abs(stiff(solution.value, None) - solution.value))

        assert float(solution.value) == pytest.approx(0.8900586117066737, abs=1e-5)
        assert residual < 1e-5

    def test_the_scf_density_converges_without_x64(self) -> None:
        # The default-precision H2 solve: the Fock build itself is accurate to seven
        # digits in float32, so a non-finite energy can only come from the mixing.
        from opifex.core.quantum.molecular_system import MolecularSystem
        from opifex.neural.quantum.dft.scf import SCFSolver

        bohr_per_angstrom = 1.0 / 0.52917721067
        system = MolecularSystem(
            atomic_numbers=jnp.array([1, 1]),
            positions=jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * bohr_per_angstrom]]),
            basis_set="sto-3g",
        )

        result = SCFSolver(system, convergence_tolerance=1e-6).solve()

        assert bool(jnp.isfinite(result.total_energy))
        assert float(result.total_energy) == pytest.approx(-1.1212060, abs=1e-4)
