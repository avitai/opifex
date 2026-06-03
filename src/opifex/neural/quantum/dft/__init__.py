"""Differentiable Kohn-Sham density-functional theory in native JAX.

Public surface:

* :class:`~opifex.neural.quantum.dft.scf.SCFSolver` -- restricted Kohn-Sham
  (RKS) LDA self-consistent-field driver and its :class:`SCFResult`.
* LDA exchange-correlation primitives in
  :mod:`opifex.neural.quantum.dft.xc`.
* The Becke-partitioned molecular quadrature grid in
  :mod:`opifex.neural.quantum.dft.grid`.
"""

from opifex.neural.quantum.dft.grid import build_molecular_grid, MolecularGrid
from opifex.neural.quantum.dft.scf import SCFResult, SCFSolver
from opifex.neural.quantum.dft.xc import (
    lda_energy_density,
    lda_exchange_correlation_potential,
    slater_exchange_energy_density,
    vwn_correlation_energy_density,
)


__all__ = [
    "MolecularGrid",
    "SCFResult",
    "SCFSolver",
    "build_molecular_grid",
    "lda_energy_density",
    "lda_exchange_correlation_potential",
    "slater_exchange_energy_density",
    "vwn_correlation_energy_density",
]
