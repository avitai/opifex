"""JAX-transform regression tests for the quantum-problem energy API.

These pin two properties of :class:`ElectronicStructureProblem`, which is backed
by the real restricted Kohn-Sham SCF solver
(:class:`opifex.neural.quantum.dft.SCFSolver`):

* :attr:`MolecularSystem.n_electrons` is *static* structural metadata (a plain
  Python ``int``), so it can size arrays / drive control flow under ``jax.jit``
  instead of degrading to a tracer.
* :meth:`ElectronicStructureProblem._energy_from_positions` -- the converged
  Kohn-Sham total energy as a pure function of the nuclear coordinates -- is
  ``jit`` / ``grad`` / ``vmap`` compatible, and :meth:`compute_forces` returns
  exactly its negative gradient (``F = -dE/dR``).

The SCF-backed tests build the AO basis / integrals eagerly (so they are slow
and float64) and are marked ``slow``; they run on H2 in the STO-3G minimal basis
(the only basis the integral backend ships), whose LDA energy is validated
against PySCF in the dedicated DFT test suite.

References: Szabo & Ostlund, *Modern Quantum Chemistry*, Ch. 3 (the
energy/force relation ``F = -dE/dR``); JAX static-shape rules in
``memory-bank/guides/jax_guide.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

from opifex.core.problems import create_molecular_system, create_neural_dft_problem


if TYPE_CHECKING:
    from opifex.core.problems import ElectronicStructureProblem
    from opifex.core.quantum.molecular_system import MolecularSystem


def _hydrogen_molecule() -> MolecularSystem:
    """Build an H2 molecule (closed shell, 2 electrons) at ~0.74 Angstrom."""
    return create_molecular_system([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))])


def _h2_problem() -> ElectronicStructureProblem:
    """An electronic-structure problem for H2 with its SCF solver pre-built.

    The solver (and its AO basis / grid template) is constructed eagerly so the
    subsequent ``jit`` / ``grad`` / ``vmap`` only traces the nuclear positions.
    """
    problem = create_neural_dft_problem(molecular_system=_hydrogen_molecule())
    _ = problem.scf_solver  # eagerly build the basis/grid before tracing
    return problem


class TestMolecularSystemElectronCountStatic:
    """``n_electrons`` is static structural metadata, not a tracer."""

    def test_n_electrons_is_python_int(self) -> None:
        """A neutral hydrogen atom reports exactly one electron as a Python int."""
        hydrogen = create_molecular_system([("H", (0.0, 0.0, 0.0))])
        result = hydrogen.n_electrons
        assert isinstance(result, int)
        assert result == 1

    def test_charged_ion_reduces_electron_count(self) -> None:
        """A +1 helium cation has one fewer electron than neutral helium."""
        helium_cation = create_molecular_system([("He", (0.0, 0.0, 0.0))], charge=1)
        assert helium_cation.n_electrons == 1

    def test_n_electrons_is_static_under_jit(self) -> None:
        """``n_electrons`` stays a static int usable as an array shape under jit."""
        lithium = create_molecular_system([("Li", (0.0, 0.0, 0.0))])

        @jax.jit
        def build_with_count() -> jax.Array:
            # Sizing an array with ``n_electrons`` only works if it is static.
            return jnp.zeros((lithium.n_electrons,))

        result = build_with_count()
        assert result.shape == (3,)  # lithium has three electrons


@pytest.mark.slow
class TestQuantumProblemEnergyTransforms:
    """The Kohn-Sham energy API is jit/grad/vmap compatible (real SCF, H2)."""

    def test_compute_energy_eager_returns_bound_python_float(self) -> None:
        """Eager ``compute_energy`` returns a concrete, bound Python float.

        The H2 LDA/STO-3G total energy is about -1.12 Hartree (validated against
        PySCF in the dedicated DFT suite); here we only pin that the problem API
        returns a concrete, finite, bound-state float.
        """
        with jax.enable_x64(True):
            energy = _h2_problem().compute_energy()
        assert isinstance(energy, float)
        assert energy < 0.0  # bound state
        assert -1.5 < energy < -0.5  # in the H2 LDA/STO-3G ball-park

    def test_energy_is_jit_compatible(self) -> None:
        """The differentiable energy traces under ``jax.jit`` to a finite value."""
        with jax.enable_x64(True):
            problem = _h2_problem()
            positions = problem.molecular_system.positions
            energy = jax.jit(problem._energy_from_positions)(positions)
            assert bool(jnp.isfinite(energy))
            assert float(energy) < 0.0

    def test_compute_energy_matches_differentiable_energy_function(self) -> None:
        """``compute_energy`` returns the value the forces are derived from.

        ``compute_forces`` differentiates ``_energy_from_positions``; for the
        energy and forces to satisfy ``F = -dE/dR`` they must come from one
        energy function. This pins ``compute_energy`` onto that same routine.
        """
        with jax.enable_x64(True):
            problem = _h2_problem()
            reported = problem.compute_energy()
            differentiable = float(
                problem._energy_from_positions(problem.molecular_system.positions)
            )
        # Both come from ``_energy_from_positions``; two separate evaluations of
        # the SCF energy can reassociate float reductions by ~1 ULP, so the
        # contract is numerical agreement, not bitwise identity.
        assert reported == pytest.approx(differentiable, abs=1e-10)

    def test_energy_supports_vmap_over_geometries(self) -> None:
        """The differentiable energy maps over a batch of geometries with ``vmap``."""
        with jax.enable_x64(True):
            problem = _h2_problem()
            base = problem.molecular_system.positions
            batch = base[None, :, :] + jnp.linspace(-0.05, 0.05, 3)[:, None, None]
            energies = jax.vmap(problem._energy_from_positions)(batch)
            assert energies.shape == (3,)
            assert bool(jnp.all(jnp.isfinite(energies)))

    def test_forces_equal_negative_energy_gradient(self) -> None:
        """Forces are exactly ``-dE/dR`` of the energy the problem reports."""
        with jax.enable_x64(True):
            problem = _h2_problem()
            positions = problem.molecular_system.positions
            forces = problem.compute_forces()
            grad_energy = jax.grad(problem._energy_from_positions)(positions)
            assert forces.shape == grad_energy.shape
            assert bool(jnp.allclose(forces, -grad_energy))
