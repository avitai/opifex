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
