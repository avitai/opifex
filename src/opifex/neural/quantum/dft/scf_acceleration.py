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


__all__ = [
    "SCFAccelerationResult",
    "measure_scf_acceleration",
]
