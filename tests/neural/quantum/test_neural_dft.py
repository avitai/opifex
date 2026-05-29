"""Tests for the classical-SCF fallback path of :class:`NeuralDFT`.

These tests exercise ``NeuralDFT.compute_energy`` when the neural SCF solver
is disabled (``use_neural_scf=False``), forcing the classical Roothaan SCF
branch in ``_solve_classical_scf``. That branch diagonalises the real
``n_basis x n_basis`` Kohn-Sham matrix, occupies the lowest orbitals, mixes
densities and monitors the band-structure energy, materialising its
per-iteration trace with ``jnp.array`` (``jax.Array`` is an abstract type, not
a constructor, so calling it would raise ``TypeError``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.neural.quantum.neural_dft import DFTResult, NeuralDFT


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Provide FLAX NNX rngs for component initialization."""
    return nnx.Rngs(jax.random.PRNGKey(0))


@pytest.fixture
def h2_system() -> MolecularSystem:
    """Minimal two-atom, two-electron hydrogen molecule in Bohr."""
    positions = jnp.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    atomic_numbers = jnp.array([1, 1])
    return MolecularSystem(positions=positions, atomic_numbers=atomic_numbers)


def test_compute_energy_classical_scf_fallback(
    rngs: nnx.Rngs, h2_system: MolecularSystem
) -> None:
    """``compute_energy`` must run the classical Roothaan SCF branch.

    With ``use_neural_scf=False`` the neural SCF solver is ``None`` and
    ``compute_energy`` takes the classical else-branch, which diagonalises the
    real ``n_basis x n_basis`` Kohn-Sham matrix (not the scalar trace) and runs
    a converging SCF loop. Two historical defects are exercised here:

    * ``jnp.linalg.eigh`` was previously called on the *scalar* trace of the
      Hamiltonian (shape ``()``), raising ``TypeError`` before any density
      update could run.
    * the inner result used ``jax.Array(convergence_history)``; ``jax.Array``
      is an abstract type, not a constructor, so it raised ``TypeError``.
    """
    max_scf_iterations = 3
    neural_dft = NeuralDFT(
        grid_size=8,
        max_scf_iterations=max_scf_iterations,
        xc_functional_type="lda",
        use_neural_scf=False,
        rngs=rngs,
    )

    assert neural_dft.neural_scf_solver is None

    result = neural_dft.compute_energy(h2_system)

    assert isinstance(result, DFTResult)
    # The convergence trace records one band-structure energy per iteration.
    assert isinstance(result.convergence_history, jax.Array)
    assert result.convergence_history.shape[0] == result.iterations
    assert 1 <= result.iterations <= max_scf_iterations
    # Every recorded SCF energy, the final density, and the total energy must be
    # finite — i.e. the loop produced real physics, not NaNs.
    assert bool(jnp.all(jnp.isfinite(result.convergence_history)))
    assert bool(jnp.all(jnp.isfinite(result.final_density)))
    assert jnp.isfinite(result.total_energy)
