"""Measure SCF iteration reduction from a high-quality initial guess.

A neural Hamiltonian model that predicts a Fock matrix close to the
self-consistent one can seed the SCF with a near-converged density, so the
Anderson/DIIS iteration reaches the fixed point in fewer steps than the default
core-Hamiltonian guess. This module quantifies that reduction: it runs the same
:class:`~opifex.neural.quantum.dft.scf.SCFSolver` from the default guess
(baseline) and from a supplied ``initial_density`` (guided) and reports the
iteration counts, having checked that both reach the same converged energy.

The guess must be a closed-shell density in the solver's own AO basis. Use
:func:`~opifex.neural.quantum.dft.scf.density_from_fock` to turn a predicted Fock
(in that basis) into a density first. Wiring a QH9-trained spherical def2-SVP
B3LYP predictor additionally requires a matching spherical-def2-SVP solver path;
that basis bridge is tracked separately and is not assumed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from opifex.core.quantum._spherical import apply_matrix, build_block_transform
from opifex.neural.quantum.dft.scf import density_from_fock


if TYPE_CHECKING:
    from jax import Array

    from opifex.neural.quantum.dft.scf import SCFSolver


@dataclass(frozen=True, slots=True, kw_only=True)
class SCFAccelerationResult:
    """Outcome of comparing a guided SCF solve against the default guess.

    Attributes:
        baseline_iterations: SCF cycles from the default core-Hamiltonian guess.
        guided_iterations: SCF cycles from the supplied initial density.
        energy_hartree: The converged total energy (Hartree); identical for both.
        converged: Whether the guided solve reached the convergence tolerance.
    """

    baseline_iterations: int
    guided_iterations: int
    energy_hartree: float
    converged: bool

    @property
    def iteration_reduction(self) -> int:
        """Number of SCF cycles saved by the guided initial guess."""
        return self.baseline_iterations - self.guided_iterations


def measure_scf_acceleration(
    solver: SCFSolver,
    initial_density: Array,
    *,
    energy_tolerance: float = 1.0e-6,
) -> SCFAccelerationResult:
    """Compare a guided SCF solve against the default-guess baseline.

    Runs ``solver`` once from the default core-Hamiltonian guess and once from
    ``initial_density``, then reports the iteration counts. The two solves must
    reach the same converged energy (the seed only changes the path, not the
    fixed point); a mismatch beyond ``energy_tolerance`` indicates an
    inconsistent guess (e.g. a density in the wrong AO basis) and raises.

    Args:
        solver: The configured SCF solver (Anderson/DIIS mode).
        initial_density: Closed-shell density seed in the solver's AO basis.
        energy_tolerance: Maximum allowed energy difference (Hartree) between the
            baseline and guided solves.

    Returns:
        The :class:`SCFAccelerationResult` with both iteration counts.

    Raises:
        ValueError: If the guided solve converges to a different energy than the
            baseline (a sign the seed is inconsistent with the solver's basis).
    """
    baseline = solver.solve()
    guided = solver.solve(initial_density=initial_density)
    energy_difference = abs(float(guided.total_energy) - float(baseline.total_energy))
    if energy_difference > energy_tolerance:
        raise ValueError(
            "guided solve converged to a different energy than the baseline "
            f"(|dE| = {energy_difference:.3e} Ha > {energy_tolerance:.1e}); the "
            "initial density is likely inconsistent with the solver's AO basis."
        )
    return SCFAccelerationResult(
        baseline_iterations=baseline.n_iterations,
        guided_iterations=guided.n_iterations,
        energy_hartree=float(guided.total_energy),
        converged=guided.converged,
    )


def spherical_fock_to_cartesian_density(
    spherical_fock: Array,
    cartesian_overlap: Array,
    angular_momenta: tuple[int, ...],
    n_occupied: int,
) -> Array:
    r"""Build a Cartesian SCF seed density from a spherical-basis Fock matrix.

    Bridges the predictor's *spherical* def2-SVP Fock (the standard
    ``2l+1``-per-shell basis) to an initial density in the SCF's *Cartesian*
    basis (``(l+1)(l+2)/2`` per shell -- e.g. 6 d components, with the extra
    contaminant). With the validated Cartesian->spherical block transform ``T``
    (:func:`~opifex.core.quantum._spherical.build_block_transform`, columns in the
    spherical AO order), the Cartesian overlap is mapped to spherical
    (``S_sph = T^T S_cart T``), the closed-shell density is solved there
    (:func:`~opifex.neural.quantum.dft.scf.density_from_fock`), and embedded back
    as ``D_cart = T D_sph T^T``. This congruence preserves the electron count
    ``Tr(D_cart S_cart) = 2 n_occ`` and overlap-metric idempotency
    ``D_cart S_cart D_cart = 2 D_cart`` exactly, so ``D_cart`` is a valid
    closed-shell seed for :meth:`SCFSolver.solve(initial_density=...)<...solve>`.

    The seed lives in the spherical subspace of the Cartesian basis (the d
    contaminant starts at zero and the SCF relaxes it), so it is an approximate
    guess, not the exact Cartesian fixed point. ``spherical_fock`` must be in the
    same spherical AO order as ``T``'s columns; a QH9-predictor Fock (the
    ``pyscf_def2svp`` p-order) needs
    :func:`~opifex.neural.quantum.hamiltonian.qh9_eval.to_pyscf_internal_ordering`
    applied first.

    Args:
        spherical_fock: The Fock matrix in the spherical AO basis
            ``(n_sph, n_sph)``.
        cartesian_overlap: The SCF's Cartesian AO overlap ``(n_cart, n_cart)``.
        angular_momenta: The angular momentum ``l`` of each shell, in AO order
            (the SCF basis's ``shell.angular_momentum`` sequence).
        n_occupied: Number of doubly-occupied orbitals (electrons // 2).

    Returns:
        The Cartesian closed-shell seed density ``(n_cart, n_cart)``.
    """
    transform = build_block_transform(angular_momenta)
    spherical_overlap = apply_matrix(transform, cartesian_overlap)
    spherical_density = density_from_fock(spherical_fock, spherical_overlap, n_occupied)
    return transform @ spherical_density @ transform.T


__all__ = [
    "SCFAccelerationResult",
    "measure_scf_acceleration",
    "spherical_fock_to_cartesian_density",
]
