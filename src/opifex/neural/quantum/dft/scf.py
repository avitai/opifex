r"""Restricted Kohn-Sham (RKS) self-consistent-field solver.

A closed-shell RKS driver built on the native McMurchie-Davidson integral backend
(:class:`~opifex.core.quantum.backend.JaxGaussianBackend`) and the
exchange-correlation functionals in :mod:`opifex.neural.quantum.dft.xc` (LDA
Slater+VWN5 and the PBE GGA).

Forward SCF
-----------
The Kohn-Sham equations are solved by symmetric-orthogonalisation fixed-point
iteration with DIIS acceleration:

#. Orthogonalise with Lowdin's :math:`S^{-1/2}`.
#. Build the Fock matrix :math:`F(D) = h_\text{core} + J[D] + V_{xc}[D]` with the
   Coulomb matrix :math:`J_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma)
   D_{\lambda\sigma}` and the LDA/GGA :math:`V_{xc}` on a real molecular grid.
#. DIIS-extrapolate the Fock matrix from the error :math:`e = F D S - S D F`.
#. Solve :math:`F' C' = C' \varepsilon`, back-transform, occupy the lowest
   :math:`n_\text{occ}` orbitals, form :math:`D = 2 C_\text{occ} C_\text{occ}^\top`.

A direct-minimisation (SCF-free) mode is available behind the same interface:
the Kohn-Sham energy is minimised directly over a QR-orthonormalised coefficient
matrix (jrystal / DWD, arXiv:2411.05033) -- intended for the learned-XC path.

Differentiable energy and analytic forces
-----------------------------------------
:meth:`SCFSolver.energy_from_positions` returns the converged total energy as a
pure, differentiable function of the nuclear coordinates: the integrals, grid and
XC matrix are rebuilt from ``positions`` and the self-consistent density is found
as an implicit fixed point (:mod:`opifex.neural.quantum.dft._energy`). Optimistix's
:class:`~optimistix.ImplicitAdjoint` differentiates the converged fixed point by
the implicit function theorem, so :meth:`SCFSolver.compute_forces` /
:meth:`SCFSolver.energy_and_forces` -- the analytic forces
:math:`F = -\partial E/\partial R` from :func:`jax.grad` -- are exact and avoid
backprop through the SCF iterations (the PySCFAD rationale, Zhang & Chan 2022).

The reported total energy is the proper Kohn-Sham energy
:math:`E = \operatorname{Tr}[D\,h_\text{core}] + \tfrac12 \operatorname{Tr}[D\,J]
+ E_{xc} + E_{nn}`.

References
----------
* P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) -- DIIS.
* X. Zhang, G. K.-L. Chan, *J. Chem. Phys.* **157**, 204801 (2022),
  arXiv:2207.13836 -- implicit differentiation of the SCF fixed point (PySCFAD).
* L. Y. Yao et al., arXiv:2411.05033 (jrystal / DWD) -- direct minimisation.
* R. G. Parr, W. Yang, *Density-Functional Theory of Atoms and Molecules*,
  Oxford (1989), Ch. 7 -- the Kohn-Sham total-energy expression.
* A. Szabo, N. S. Ostlund, *Modern Quantum Chemistry*, Dover (1996), Ch. 3 --
  Roothaan equations and Lowdin symmetric orthogonalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
from jax import Array

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.neural.quantum.dft._energy import (
    _density_from_fock as _density_from_fock_impl,
    _Integrals,
    assemble_integrals,
    build_fock,
    converged_density_direct,
    converged_density_implicit,
    converged_density_unrolled,
    Functional,
    total_energy,
)
from opifex.neural.quantum.dft.grid import (
    build_molecular_grid_traceable,
    MolecularGridTemplate,
)


class SolverMode(StrEnum):
    """How the self-consistent density is found."""

    DIIS = "diis"
    DIRECT = "direct"


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
    """Restricted Kohn-Sham (RKS) self-consistent-field solver.

    Args:
        system: The molecular system to solve.
        basis: The AO basis (defaults to STO-3G built from the system).
        functional: The exchange-correlation functional (``"lda"`` or ``"pbe"``).
        mode: ``"diis"`` for the DIIS SCF or ``"direct"`` for direct minimisation.
        max_iterations: Maximum SCF / fixed-point / minimisation iterations.
        convergence_tolerance: RMS density-change convergence threshold.
        diis_space: Number of Fock/error matrices retained for DIIS.
    """

    def __init__(
        self,
        system: MolecularSystem,
        basis: AtomicOrbitalBasis | None = None,
        *,
        functional: Functional | str = Functional.LDA,
        mode: SolverMode | str = SolverMode.DIIS,
        max_iterations: int = 100,
        convergence_tolerance: float = 1.0e-8,
        diis_space: int = 8,
    ) -> None:
        """Initialise the solver, its integral backend and the grid template."""
        if system.multiplicity != 1:
            raise ValueError("SCFSolver supports closed-shell (multiplicity 1) systems only")
        if system.n_electrons % 2 != 0:
            raise ValueError("Closed-shell RKS requires an even electron count")
        self._system = system
        self._basis = basis or AtomicOrbitalBasis.from_molecular_system(system)
        self._functional = Functional(functional)
        self._mode = SolverMode(mode)
        self._max_iterations = max_iterations
        self._convergence_tolerance = convergence_tolerance
        self._diis_space = diis_space
        self._n_occupied = system.n_electrons // 2
        self._grid_template: MolecularGridTemplate = build_molecular_grid_traceable(system)

    @property
    def functional(self) -> Functional:
        """The exchange-correlation functional in use."""
        return self._functional

    def _integrals(self, positions: Array) -> _Integrals:
        """Assemble the position-dependent integrals/grid for ``positions``."""
        return assemble_integrals(
            positions, self._system, self._basis, self._grid_template, self._functional
        )

    def energy_from_positions(self, positions: Array, *, differentiable: str = "implicit") -> Array:
        """Converged Kohn-Sham total energy as a function of nuclear positions.

        The self-consistent density is found as an *implicit fixed point* of the
        Roothaan step regardless of the forward :class:`SolverMode` (direct
        minimisation and the DIIS/fixed-point iteration converge to the same
        Kohn-Sham density). Differentiating the implicit fixed point gives exact,
        memory-cheap gradients via the implicit function theorem and avoids the
        gauge-singular Hessian of the direct-minimisation parametrisation, so
        :meth:`compute_forces` is robust for both modes.

        Args:
            positions: Nuclear positions in Bohr [Shape: (n_atoms, 3)].
            differentiable: ``"implicit"`` (default) finds the density as an
                implicit fixed point (IFT gradient); ``"unroll"`` runs a fixed
                number of differentiable SCF steps (gradient cross-check).

        Returns:
            The scalar converged total energy (Hartree).
        """
        integrals = self._integrals(positions)
        if differentiable == "unroll":
            density = converged_density_unrolled(
                integrals,
                self._functional,
                self._n_occupied,
                n_steps=self._max_iterations,
            )
        else:
            density = converged_density_implicit(
                integrals,
                self._functional,
                self._n_occupied,
                tolerance=self._convergence_tolerance,
                max_steps=self._max_iterations,
            )
        return total_energy(density, integrals, self._functional)

    def energy(self) -> Array:
        """Converged total energy at the system's nuclear geometry."""
        return self.energy_from_positions(self._system.positions)

    def compute_forces(self, positions: Array | None = None) -> Array:
        r"""Analytic nuclear forces :math:`F = -\partial E/\partial R`.

        Computed by :func:`jax.grad` of the implicit-diff total energy with
        respect to the nuclear coordinates.

        Args:
            positions: Geometry to evaluate at (defaults to the system geometry).

        Returns:
            Forces in Hartree/Bohr [Shape: (n_atoms, 3)].
        """
        where = self._system.positions if positions is None else positions
        gradient = jax.grad(self.energy_from_positions)(where)
        return -gradient

    def energy_and_forces(self, positions: Array | None = None) -> tuple[Array, Array]:
        r"""Converged total energy and the analytic forces :math:`-\partial E/\partial R`.

        Args:
            positions: Geometry to evaluate at (defaults to the system geometry).

        Returns:
            A pair ``(energy, forces)`` with ``forces`` in Hartree/Bohr.
        """
        where = self._system.positions if positions is None else positions
        energy, gradient = jax.value_and_grad(self.energy_from_positions)(where)
        return energy, -gradient

    def solve(self) -> SCFResult:
        """Run the forward SCF (DIIS or direct minimisation) to convergence.

        Returns:
            The :class:`SCFResult` with the converged total energy and orbitals.
        """
        if self._mode is SolverMode.DIRECT:
            return self._solve_direct()
        return self._solve_diis()

    def _solve_direct(self) -> SCFResult:
        """Direct-minimisation forward solve (SCF-free path)."""
        integrals = self._integrals(self._system.positions)
        density = converged_density_direct(
            integrals,
            self._functional,
            self._n_occupied,
            self._basis.n_atomic_orbitals,
            tolerance=self._convergence_tolerance,
            max_steps=self._max_iterations,
        )
        fock, _, _ = build_fock(density, integrals, self._functional)
        density, coefficients, orbital_energies = _density_from_fock_impl(
            fock, integrals.orthogonaliser, self._n_occupied
        )
        energy = total_energy(density, integrals, self._functional)
        return SCFResult(
            total_energy=energy,
            orbital_energies=orbital_energies,
            density_matrix=density,
            coefficients=coefficients,
            n_iterations=self._max_iterations,
            converged=True,
        )

    def _solve_diis(self) -> SCFResult:
        """DIIS forward solve at the system's nuclear geometry."""
        integrals = self._integrals(self._system.positions)
        overlap = integrals.overlap
        orthogonaliser = integrals.orthogonaliser

        density, coefficients, orbital_energies = _density_from_fock_impl(
            integrals.core_hamiltonian, orthogonaliser, self._n_occupied
        )

        fock_history: list[Array] = []
        error_history: list[Array] = []
        energy = jnp.asarray(0.0)
        converged = False
        iterations_run = 0

        for iteration in range(1, self._max_iterations + 1):
            iterations_run = iteration
            fock, _, _ = build_fock(density, integrals, self._functional)
            energy = total_energy(density, integrals, self._functional)

            error = fock @ density @ overlap - overlap @ density @ fock
            fock_history.append(fock)
            error_history.append(error)
            if len(fock_history) > self._diis_space:
                fock_history.pop(0)
                error_history.pop(0)
            extrapolated = (
                _diis_extrapolate(fock_history, error_history) if len(fock_history) > 1 else fock
            )

            new_density, coefficients, orbital_energies = _density_from_fock_impl(
                extrapolated, orthogonaliser, self._n_occupied
            )
            density_change = jnp.sqrt(jnp.mean((new_density - density) ** 2))
            density = new_density
            if float(density_change) < self._convergence_tolerance:
                converged = True
                break

        return SCFResult(
            total_energy=energy,
            orbital_energies=orbital_energies,
            density_matrix=density,
            coefficients=coefficients,
            n_iterations=iterations_run,
            converged=converged,
        )


__all__ = [
    "Functional",
    "SCFResult",
    "SCFSolver",
    "SolverMode",
]
