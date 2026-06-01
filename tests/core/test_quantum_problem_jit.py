"""JAX-transform regression tests for the quantum-problem energy API.

These pin the behaviour fixed after the 2026-05-31 quantum-JIT failure:

* :attr:`MolecularSystem.n_electrons` is *static* structural metadata (a plain
  Python ``int``), so it can size arrays / drive control flow under ``jax.jit``
  instead of degrading to a tracer (the original ``min(n_electrons * 2, 50)``
  ``TracerBoolConversionError``).
* :meth:`ElectronicStructureProblem.compute_energy` is evaluated through the same
  differentiable routine that :meth:`compute_forces` differentiates, so the
  energy and the forces are mutually consistent (``F = -dE/dR``) and the public
  quantum-problem API is ``jit`` / ``grad`` / ``vmap`` compatible.

References: Szabo & Ostlund, *Modern Quantum Chemistry*, Ch. 3 (the
energy/force relation ``F = -dE/dR``); JAX static-shape rules in
``memory-bank/guides/jax_guide.md``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.core.problems import create_molecular_system, create_neural_dft_problem


if TYPE_CHECKING:
    from opifex.core.quantum.molecular_system import MolecularSystem


def _water() -> MolecularSystem:
    """Build a water molecule (3 atoms) for the energy/force tests."""
    return create_molecular_system(
        [("O", (0.0, 0.0, 0.0)), ("H", (0.76, 0.59, 0.0)), ("H", (-0.76, 0.59, 0.0))]
    )


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


class TestQuantumProblemEnergyTransforms:
    """The quantum-problem energy API is jit/grad/vmap compatible."""

    def test_compute_energy_is_jit_compatible(self) -> None:
        """``compute_energy`` traces under ``jax.jit`` and returns a finite bound state."""
        problem = create_neural_dft_problem(
            molecular_system=create_molecular_system([("H", (0.0, 0.0, 0.0))]),
            grid_level=1,
        )
        energy = jax.jit(problem.compute_energy)()
        assert jnp.isfinite(energy)
        assert float(energy) < 0.0  # bound state

    def test_compute_energy_eager_returns_python_float(self) -> None:
        """In eager execution ``compute_energy`` returns a concrete Python float."""
        problem = create_neural_dft_problem(
            molecular_system=create_molecular_system([("H", (0.0, 0.0, 0.0))]),
            grid_level=1,
        )
        energy = problem.compute_energy()
        assert isinstance(energy, float)
        assert energy < 0.0

    def test_compute_energy_matches_differentiable_energy_function(self) -> None:
        """``compute_energy`` returns the same value the forces are derived from.

        ``compute_forces`` differentiates ``_energy_from_positions``; for the
        energy and forces to satisfy ``F = -dE/dR`` they must come from one
        energy function. This pins ``compute_energy`` onto that same routine.
        """
        problem = create_neural_dft_problem(molecular_system=_water(), grid_level=1)
        reported = problem.compute_energy()
        differentiable = float(problem._energy_from_positions(problem.molecular_system.positions))
        assert reported == differentiable

    def test_energy_supports_vmap_over_geometries(self) -> None:
        """The differentiable energy maps over a batch of geometries with ``vmap``."""
        problem = create_neural_dft_problem(molecular_system=_water(), grid_level=1)
        base = problem.molecular_system.positions
        batch = base[None, :, :] + jnp.linspace(-0.1, 0.1, 4)[:, None, None]
        energies = jax.vmap(problem._energy_from_positions)(batch)
        assert energies.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(energies)))

    def test_forces_equal_negative_energy_gradient(self) -> None:
        """Forces are exactly ``-dE/dR`` of the energy the problem reports."""
        problem = create_neural_dft_problem(molecular_system=_water(), grid_level=1)
        forces = problem.compute_forces()
        grad_energy = jax.grad(problem._energy_from_positions)(problem.molecular_system.positions)
        assert forces.shape == grad_energy.shape
        assert bool(jnp.allclose(forces, -grad_energy))
