r"""Restricted Kohn-Sham (RKS) self-consistent-field solver.

A closed-shell RKS driver built on the native McMurchie-Davidson integral backend
(:class:`~opifex.core.quantum.backend.JaxGaussianBackend`) and the
exchange-correlation functionals in :mod:`opifex.neural.quantum.dft.xc` (LDA
Slater+VWN5 and the PBE GGA).

Forward SCF
-----------
The Kohn-Sham equations are solved by symmetric-orthogonalisation fixed-point
iteration with Anderson acceleration (Pulay DIIS on the density residual --
:class:`~opifex.neural.quantum.dft._fixed_point.AndersonAcceleration`):

#. Orthogonalise with Lowdin's :math:`S^{-1/2}`.
#. Build the Fock matrix :math:`F(D) = h_\text{core} + J[D] + V_{xc}[D]` with the
   Coulomb matrix :math:`J_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu|\lambda\sigma)
   D_{\lambda\sigma}` and the LDA/GGA :math:`V_{xc}` on a real molecular grid.
#. Solve :math:`F' C' = C' \varepsilon`, back-transform, occupy the lowest
   :math:`n_\text{occ}` orbitals, form the Roothaan step
   :math:`D' = 2 C_\text{occ} C_\text{occ}^\top`.
#. Anderson-mix a short history of densities to converge the residual
   :math:`D' - D`; plain Roothaan iteration charge-sloshes and stalls.

The forward solve and the differentiable energy path share this one
fixed-point engine, so both are jit-compatible and converge identically.

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
* P. Pulay, *Chem. Phys. Lett.* **73**, 393 (1980) -- DIIS; D. G. Anderson,
  *J. ACM* **12**, 547 (1965) -- Anderson acceleration (the density-space DIIS
  used here).
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
from flax import nnx  # noqa: TC002
from jax import Array
from optimistix import RESULTS

from opifex.core.quantum.basis import AtomicOrbitalBasis
from opifex.core.quantum.molecular_system import MolecularSystem  # noqa: TC001
from opifex.neural.quantum.dft._energy import (
    _density_from_fock as _density_from_fock_impl,
    _Integrals,
    _symmetric_orthogonaliser,
    assemble_integrals,
    build_fock,
    converged_density_direct,
    converged_density_implicit,
    converged_density_unrolled,
    Functional,
    NeuralXCSpec,
    scf_fixed_point,
    total_energy,
)
from opifex.neural.quantum.dft.grid import (
    build_molecular_grid_traceable,
    MolecularGridTemplate,
)
from opifex.neural.quantum.neural_xc import NeuralXCFunctional  # noqa: TC001


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


class SCFSolver:
    """Restricted Kohn-Sham (RKS) self-consistent-field solver.

    Args:
        system: The molecular system to solve.
        basis: The AO basis (defaults to STO-3G built from the system).
        functional: The exchange-correlation functional
            (``"lda"``, ``"pbe"`` or ``"neural"``).
        mode: ``"diis"`` for the Anderson-accelerated self-consistent SCF (Pulay
            DIIS on the density residual) or ``"direct"`` for direct minimisation.
        neural_functional: A learned XC functional; required (and selects the
            ``"neural"`` functional) when ``functional == "neural"``.
        grid_template: A pre-built molecular-grid template; defaults to the
            standard Becke grid for ``system``. Supplying a coarser template
            trades XC-integration accuracy for speed (e.g. in tests).
        max_iterations: Maximum SCF / fixed-point / minimisation iterations.
        convergence_tolerance: RMS density-change convergence threshold.
    """

    def __init__(
        self,
        system: MolecularSystem,
        basis: AtomicOrbitalBasis | None = None,
        *,
        functional: Functional | str = Functional.LDA,
        mode: SolverMode | str = SolverMode.DIIS,
        neural_functional: NeuralXCFunctional | None = None,
        grid_template: MolecularGridTemplate | None = None,
        max_iterations: int = 100,
        convergence_tolerance: float = 1.0e-8,
    ) -> None:
        """Initialise the solver, its integral backend and the grid template."""
        if system.multiplicity != 1:
            raise ValueError("SCFSolver supports closed-shell (multiplicity 1) systems only")
        if system.n_electrons % 2 != 0:
            raise ValueError("Closed-shell RKS requires an even electron count")
        self._system = system
        self._basis = basis or AtomicOrbitalBasis.from_molecular_system(system)
        # A learned functional selects the NEURAL path; an explicit functional
        # without a neural module must not be NEURAL.
        if neural_functional is not None:
            self._functional = Functional.NEURAL
        else:
            self._functional = Functional(functional)
        if self._functional is Functional.NEURAL and neural_functional is None:
            raise ValueError("functional='neural' requires a neural_functional")
        self._neural_spec: NeuralXCSpec | None = (
            NeuralXCSpec.from_functional(neural_functional)
            if neural_functional is not None
            else None
        )
        self._mode = SolverMode(mode)
        self._max_iterations = max_iterations
        self._convergence_tolerance = convergence_tolerance
        self._n_occupied = system.n_electrons // 2
        self._grid_template: MolecularGridTemplate = (
            grid_template if grid_template is not None else build_molecular_grid_traceable(system)
        )

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
        return self._energy_from_positions(
            positions, self._neural_spec, differentiable=differentiable
        )

    def _energy_from_positions(
        self,
        positions: Array,
        neural_spec: NeuralXCSpec | None,
        *,
        differentiable: str = "implicit",
    ) -> Array:
        """Converged total energy for ``positions`` with an explicit XC spec.

        Threading ``neural_spec`` explicitly (rather than only reading
        ``self._neural_spec``) lets the learned-XC training loop differentiate
        the converged energy with respect to the *traced* parameter state.
        """
        integrals = self._integrals(positions)
        if differentiable == "unroll":
            density = converged_density_unrolled(
                integrals,
                self._functional,
                self._n_occupied,
                n_steps=self._max_iterations,
                neural_spec=neural_spec,
            )
        else:
            density = converged_density_implicit(
                integrals,
                self._functional,
                self._n_occupied,
                tolerance=self._convergence_tolerance,
                max_steps=self._max_iterations,
                neural_spec=neural_spec,
            )
        return total_energy(density, integrals, self._functional, neural_spec)

    def energy_from_state(self, state: nnx.State, positions: Array | None = None) -> Array:
        """Converged total energy as a differentiable function of the XC state.

        The entry point for learned-XC training: ``jax.grad`` of this with
        respect to ``state`` gives the exact ``dE/dtheta`` through the
        implicit-diff SCF (the implicit function theorem differentiates the
        converged fixed point, not the iterations).

        Args:
            state: The neural XC parameter state (an ``nnx.State`` pytree, as
                produced by :func:`flax.nnx.split`).
            positions: Geometry to evaluate at (defaults to the system geometry).

        Returns:
            The scalar converged total energy (Hartree).
        """
        if self._neural_spec is None:
            raise ValueError("energy_from_state requires a neural functional")
        spec = NeuralXCSpec(graphdef=self._neural_spec.graphdef, state=state)
        where = self._system.positions if positions is None else positions
        return self._energy_from_positions(where, spec)

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

    def solve(self, *, initial_density: Array | None = None) -> SCFResult:
        """Run the forward SCF (DIIS or direct minimisation) to convergence.

        Args:
            initial_density: Optional closed-shell density to seed the
                Anderson/DIIS iteration. A high-quality guess (e.g. reconstructed
                from a neural-network predicted Fock via :func:`density_from_fock`)
                reaches the fixed point in fewer iterations; the converged result
                is unchanged. Ignored by the direct-minimisation mode, which is
                seeded internally.

        Returns:
            The :class:`SCFResult` with the converged total energy and orbitals.
        """
        if self._mode is SolverMode.DIRECT:
            return self._solve_direct()
        return self._solve_self_consistent(initial_density=initial_density)

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
            neural_spec=self._neural_spec,
        )
        fock, _, _ = build_fock(density, integrals, self._functional, self._neural_spec)
        density, coefficients, orbital_energies = _density_from_fock_impl(
            fock, integrals.orthogonaliser, self._n_occupied
        )
        energy = total_energy(density, integrals, self._functional, self._neural_spec)
        return SCFResult(
            total_energy=energy,
            orbital_energies=orbital_energies,
            density_matrix=density,
            coefficients=coefficients,
            n_iterations=self._max_iterations,
            converged=True,
        )

    def _solve_self_consistent(self, *, initial_density: Array | None = None) -> SCFResult:
        """Anderson-accelerated self-consistent forward solve.

        Drives the Kohn-Sham density to the fixed point with the same
        :class:`~opifex.neural.quantum.dft._fixed_point.AndersonAcceleration`
        engine as the differentiable energy path (Anderson mixing is Pulay DIIS
        applied to the density residual), then reconstructs the orbitals and
        total energy from the converged density.
        """
        integrals = self._integrals(self._system.positions)
        solution = scf_fixed_point(
            integrals,
            self._functional,
            self._n_occupied,
            tolerance=self._convergence_tolerance,
            max_steps=self._max_iterations,
            neural_spec=self._neural_spec,
            initial_density=initial_density,
        )
        fock, _, _ = build_fock(solution.value, integrals, self._functional, self._neural_spec)
        density, coefficients, orbital_energies = _density_from_fock_impl(
            fock, integrals.orthogonaliser, self._n_occupied
        )
        energy = total_energy(density, integrals, self._functional, self._neural_spec)
        return SCFResult(
            total_energy=energy,
            orbital_energies=orbital_energies,
            density_matrix=density,
            coefficients=coefficients,
            n_iterations=int(solution.stats["num_steps"]),
            converged=bool(solution.result == RESULTS.successful),
        )


def density_from_fock(fock: Array, overlap: Array, n_occupied: int) -> Array:
    """Closed-shell density from a Fock matrix by solving ``FC = SCe``.

    Reconstructs an initial-guess density from a Fock matrix (such as one
    predicted by a neural Hamiltonian model) in the same AO basis as ``overlap``:
    it Lowdin-orthogonalises with ``S^{-1/2}``, diagonalises the orthonormal Fock,
    back-transforms the lowest ``n_occupied`` orbitals and forms
    ``D = 2 C_occ C_occ^T``. Pair the result with
    :meth:`SCFSolver.solve(initial_density=...)<SCFSolver.solve>` to seed the SCF.

    Args:
        fock: The Fock matrix ``(n_ao, n_ao)``.
        overlap: The AO overlap matrix ``S`` ``(n_ao, n_ao)`` in the same basis.
        n_occupied: Number of doubly-occupied orbitals (electrons // 2).

    Returns:
        The closed-shell density matrix ``(n_ao, n_ao)``.
    """
    orthogonaliser = _symmetric_orthogonaliser(overlap)
    density, _, _ = _density_from_fock_impl(fock, orthogonaliser, n_occupied)
    return density


__all__ = [
    "Functional",
    "SCFResult",
    "SCFSolver",
    "SolverMode",
    "density_from_fock",
]
