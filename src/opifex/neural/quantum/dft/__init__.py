"""Differentiable Kohn-Sham density-functional theory in native JAX.

Public surface:

* :class:`~opifex.neural.quantum.dft.scf.SCFSolver` -- restricted Kohn-Sham
  (RKS) self-consistent-field driver (LDA, PBE and the learned ``neural``
  functional; DIIS and direct-minimisation modes) with
  implicit-differentiation analytic nuclear forces and exact
  exchange-correlation-parameter gradients, plus its :class:`SCFResult`, the
  :class:`Functional` and :class:`SolverMode` enums.
* LDA and PBE exchange-correlation primitives in
  :mod:`opifex.neural.quantum.dft.xc`.
* The Becke-partitioned molecular quadrature grid (eager and the
  position-traceable template) in :mod:`opifex.neural.quantum.dft.grid`.
"""

from opifex.neural.quantum.dft.grid import (
    build_molecular_grid,
    build_molecular_grid_traceable,
    MolecularGrid,
    MolecularGridTemplate,
)
from opifex.neural.quantum.dft.scf import (
    density_from_fock,
    Functional,
    SCFResult,
    SCFSolver,
    SolverMode,
)
from opifex.neural.quantum.dft.scf_acceleration import (
    measure_scf_acceleration,
    SCFAccelerationResult,
)
from opifex.neural.quantum.dft.xc import (
    lda_energy_density,
    lda_exchange_correlation_potential,
    pbe_correlation_energy_density,
    pbe_energy_density,
    pbe_exchange_correlation_potential,
    pbe_exchange_energy_density,
    pw92_correlation_energy_density,
    slater_exchange_energy_density,
    vwn_correlation_energy_density,
)


__all__ = [
    "Functional",
    "MolecularGrid",
    "MolecularGridTemplate",
    "SCFAccelerationResult",
    "SCFResult",
    "SCFSolver",
    "SolverMode",
    "build_molecular_grid",
    "build_molecular_grid_traceable",
    "density_from_fock",
    "lda_energy_density",
    "lda_exchange_correlation_potential",
    "measure_scf_acceleration",
    "pbe_correlation_energy_density",
    "pbe_energy_density",
    "pbe_exchange_correlation_potential",
    "pbe_exchange_energy_density",
    "pw92_correlation_energy_density",
    "slater_exchange_energy_density",
    "vwn_correlation_energy_density",
]
