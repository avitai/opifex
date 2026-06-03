r"""Restricted Kohn-Sham (RKS) self-consistent-field solver.

A closed-shell RKS/LDA SCF driver built on the native McMurchie-Davidson
integral backend (:class:`~opifex.core.quantum.backend.JaxGaussianBackend`) and
the LDA exchange-correlation functional (:mod:`opifex.neural.quantum.dft.xc`).

The Kohn-Sham equations are solved by the standard symmetric-orthogonalisation
fixed-point iteration with DIIS acceleration:

#. Orthogonalise with Lowdin's :math:`S^{-1/2}`.
#. Build the Fock matrix :math:`F(D) = h_\text{core} + J[D] + V_{xc}[D]` where
   the Coulomb matrix :math:`J_{\mu\nu} = \sum_{\lambda\sigma}
   (\mu\nu|\lambda\sigma) D_{\lambda\sigma}` comes from the ERIs and
   :math:`V_{xc}` is the LDA potential evaluated on a real molecular grid.
#. DIIS-extrapolate the Fock matrix from the error
   :math:`e = F D S - S D F` (Pulay).
#. Solve :math:`F' C' = C' \varepsilon` in the orthonormal basis, back-transform,
   occupy the lowest :math:`n_\text{occ}` orbitals, form
   :math:`D = 2 C_\text{occ} C_\text{occ}^\top`.

The reported total energy is the proper Kohn-Sham energy

.. math::
    E = \operatorname{Tr}[D\,h_\text{core}]
      + \tfrac12 \operatorname{Tr}[D\,J[D]]
      + E_{xc} + E_{nn},

i.e. one-electron + Hartree (Coulomb) + exchange-correlation + nuclear repulsion,
*not* the band energy :math:`2\sum_i \varepsilon_i`.

References
----------
* P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) -- DIIS.
* R. G. Parr, W. Yang, *Density-Functional Theory of Atoms and Molecules*,
  Oxford (1989), Ch. 7 -- the Kohn-Sham total-energy expression.
* A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996), Ch. 3 --
  Roothaan equations and Lowdin symmetric orthogonalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from opifex.core.quantum.backend import JaxGaussianBackend
from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.neural.quantum.dft.grid import build_molecular_grid, MolecularGrid
from opifex.neural.quantum.dft.xc import (
    lda_energy_density,
    lda_exchange_correlation_potential,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class SCFResult:
    """Outcome of a restricted Kohn-Sham SCF calculation.

    Attributes:
        total_energy: Converged Kohn-Sham total energy (Hartree).
        orbital_energies: Molecular-orbital eigenvalues [Shape: (n_ao,)].
        density_matrix: Converged AO density matrix [Shape: (n_ao, n_ao)].
        coefficients: MO coefficients [Shape: (n_ao, n_ao)].
        n_iterations: Number of SCF iterations performed.
        converged: Whether the density change fell below the tolerance.
    """

    total_energy: Array
    orbital_energies: Array
    density_matrix: Array
    coefficients: Array
    n_iterations: int
    converged: bool


def _symmetric_orthogonaliser(overlap: Array) -> Array:
    r"""Lowdin :math:`S^{-1/2}` via eigendecomposition of the overlap matrix."""
    eigenvalues, eigenvectors = jnp.linalg.eigh(overlap)
    return eigenvectors @ jnp.diag(eigenvalues ** (-0.5)) @ eigenvectors.T


def _coulomb_matrix(eri: Array, density: Array) -> Array:
    r"""Coulomb matrix ``J_{mn} = sum_{ls} (mn|ls) D_{ls}``."""
    return jnp.einsum("mnls,ls->mn", eri, density)


@dataclass(frozen=True, slots=True, kw_only=True)
class _GridData:
    """Precomputed grid quantities reused every SCF iteration."""

    grid: MolecularGrid
    ao_values: Array  # (n_points, n_ao)


def _build_xc(density: Array, grid_data: _GridData) -> tuple[Array, Array]:
    r"""Exchange-correlation energy and matrix on the molecular grid.

    The electron density at each grid point is
    :math:`\rho(r) = \sum_{\mu\nu} D_{\mu\nu}\phi_\mu(r)\phi_\nu(r)`; the XC matrix
    is :math:`V^{xc}_{\mu\nu} = \sum_g w_g v_{xc}(\rho_g)\phi_\mu(r_g)\phi_\nu(r_g)`
    and the XC energy is :math:`E_{xc} = \sum_g w_g \rho_g \varepsilon_{xc}(\rho_g)`.

    Returns:
        A pair ``(energy_xc, matrix_xc)``.
    """
    ao = grid_data.ao_values
    weights = grid_data.grid.weights
    rho = jnp.einsum("gm,mn,gn->g", ao, density, ao)
    rho = jnp.clip(rho, 0.0, None)

    energy_per_particle = lda_energy_density(rho)
    energy_xc = jnp.sum(weights * rho * energy_per_particle)

    potential = lda_exchange_correlation_potential(rho)
    weighted = weights * potential
    matrix_xc = jnp.einsum("g,gm,gn->mn", weighted, ao, ao)
    return energy_xc, matrix_xc


def _density_from_fock(
    fock: Array, orthogonaliser: Array, n_occupied: int
) -> tuple[Array, Array, Array]:
    """Solve ``FC=SCe`` in the orthonormal basis and build the closed-shell density.

    Returns:
        ``(density, coefficients, orbital_energies)``.
    """
    fock_orthonormal = orthogonaliser.T @ fock @ orthogonaliser
    orbital_energies, orthonormal_coeffs = jnp.linalg.eigh(fock_orthonormal)
    coefficients = orthogonaliser @ orthonormal_coeffs
    occupied = coefficients[:, :n_occupied]
    density = 2.0 * occupied @ occupied.T
    return density, coefficients, orbital_energies


def _diis_extrapolate(fock_history: list[Array], error_history: list[Array]) -> Array:
    """Pulay DIIS extrapolation of the Fock matrix from stored error vectors."""
    n = len(fock_history)
    b_matrix = jnp.zeros((n + 1, n + 1))
    for i in range(n):
        for j in range(n):
            b_matrix = b_matrix.at[i, j].set(jnp.sum(error_history[i] * error_history[j]))
    b_matrix = b_matrix.at[n, :n].set(-1.0)
    b_matrix = b_matrix.at[:n, n].set(-1.0)
    rhs = jnp.zeros(n + 1).at[n].set(-1.0)
    coefficients = jnp.linalg.solve(b_matrix, rhs)[:n]
    extrapolated = jnp.zeros_like(fock_history[0])
    for i in range(n):
        extrapolated = extrapolated + coefficients[i] * fock_history[i]
    return extrapolated


class SCFSolver:
    """Restricted Kohn-Sham (RKS) LDA self-consistent-field solver.

    Args:
        system: The molecular system to solve.
        basis: The AO basis (defaults to STO-3G built from the system).
        max_iterations: Maximum SCF iterations.
        convergence_tolerance: RMS density-change convergence threshold.
        diis_space: Number of Fock/error matrices retained for DIIS.
    """

    def __init__(
        self,
        system: MolecularSystem,
        basis: AtomicOrbitalBasis | None = None,
        *,
        max_iterations: int = 100,
        convergence_tolerance: float = 1.0e-8,
        diis_space: int = 8,
    ) -> None:
        """Initialise the solver and its integral backend."""
        if system.multiplicity != 1:
            raise ValueError("SCFSolver supports closed-shell (multiplicity 1) systems only")
        self._system = system
        self._basis = basis or AtomicOrbitalBasis.from_molecular_system(system)
        self._backend = JaxGaussianBackend(system, self._basis)
        self._max_iterations = max_iterations
        self._convergence_tolerance = convergence_tolerance
        self._diis_space = diis_space
        self._n_occupied = system.n_electrons // 2
        if system.n_electrons % 2 != 0:
            raise ValueError("Closed-shell RKS requires an even electron count")

    def _grid_data(self) -> _GridData:
        """Build the molecular grid and cache the AO values on it."""
        grid = build_molecular_grid(self._system)
        ao_values = self._basis.evaluate(grid.points)
        return _GridData(grid=grid, ao_values=ao_values)

    def build_fock(
        self,
        density: Array,
        core_hamiltonian: Array,
        eri: Array,
        grid_data: _GridData,
    ) -> tuple[Array, Array, Array]:
        """Build ``F = h_core + J + V_xc`` and return the energy components.

        Returns:
            ``(fock, coulomb, xc_matrix)`` plus, via :meth:`energy`, the scalar
            energy contributions.
        """
        coulomb = _coulomb_matrix(eri, density)
        _, xc_matrix = _build_xc(density, grid_data)
        fock = core_hamiltonian + coulomb + xc_matrix
        return fock, coulomb, xc_matrix

    def _total_energy(
        self,
        density: Array,
        core_hamiltonian: Array,
        coulomb: Array,
        energy_xc: Array,
        nuclear_repulsion: Array,
    ) -> Array:
        """Proper Kohn-Sham total energy from the converged quantities."""
        one_electron = jnp.sum(density * core_hamiltonian)
        hartree = 0.5 * jnp.sum(density * coulomb)
        return one_electron + hartree + energy_xc + nuclear_repulsion

    def solve(self) -> SCFResult:
        """Run the RKS SCF iteration to convergence.

        Returns:
            The :class:`SCFResult` with the converged total energy and orbitals.
        """
        overlap = self._backend.overlap()
        core_hamiltonian = self._backend.core_hamiltonian()
        eri = self._backend.electron_repulsion()
        nuclear_repulsion = self._backend.nuclear_repulsion()
        orthogonaliser = _symmetric_orthogonaliser(overlap)
        grid_data = self._grid_data()

        # Core-Hamiltonian initial guess.
        density, coefficients, orbital_energies = _density_from_fock(
            core_hamiltonian, orthogonaliser, self._n_occupied
        )

        fock_history: list[Array] = []
        error_history: list[Array] = []
        total_energy = jnp.asarray(0.0)
        converged = False
        iterations_run = 0

        for iteration in range(1, self._max_iterations + 1):
            iterations_run = iteration

            # Build the Fock matrix and the energy at the current density (J and
            # XC are computed exactly once per iteration).
            coulomb = _coulomb_matrix(eri, density)
            energy_xc, xc_matrix = _build_xc(density, grid_data)
            fock = core_hamiltonian + coulomb + xc_matrix
            total_energy = self._total_energy(
                density, core_hamiltonian, coulomb, energy_xc, nuclear_repulsion
            )

            # Pulay DIIS extrapolation from the commutator error e = FDS - SDF.
            error = fock @ density @ overlap - overlap @ density @ fock
            fock_history.append(fock)
            error_history.append(error)
            if len(fock_history) > self._diis_space:
                fock_history.pop(0)
                error_history.pop(0)
            extrapolated = (
                _diis_extrapolate(fock_history, error_history) if len(fock_history) > 1 else fock
            )

            new_density, coefficients, orbital_energies = _density_from_fock(
                extrapolated, orthogonaliser, self._n_occupied
            )

            density_change = jnp.sqrt(jnp.mean((new_density - density) ** 2))
            density = new_density
            if float(density_change) < self._convergence_tolerance:
                converged = True
                break

        return SCFResult(
            total_energy=total_energy,
            orbital_energies=orbital_energies,
            density_matrix=density,
            coefficients=coefficients,
            n_iterations=iterations_run,
            converged=converged,
        )


__all__ = [
    "SCFResult",
    "SCFSolver",
]
