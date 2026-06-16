"""Spectral methods for Opifex physics.

Two families live here:

- **General pseudo-spectral PDE solving** — a fourth-order exponential
  time-differencing (ETDRK4) integrator for semilinear PDEs ``u_t = L u + N(u)``
  (:mod:`~opifex.physics.spectral.etdrk`), its Fourier operators and nonlinear
  terms (:mod:`~opifex.physics.spectral.semilinear`), and ready-made solvers for
  Burgers, Kuramoto-Sivashinsky and Korteweg-de Vries
  (:mod:`~opifex.physics.spectral.steppers`).
- **Quantum spectral operators** — Fourier momentum/kinetic-energy operators
  (:mod:`~opifex.physics.spectral.quantum_spectral`).
"""

from opifex.physics.spectral.etdrk import (
    etdrk4_coefficients,
    etdrk4_step,
    ETDRK4Coefficients,
    integrate_etdrk4,
)
from opifex.physics.spectral.quantum_spectral import (
    spectral_gradient,
    spectral_kinetic_energy,
    spectral_momentum,
    spectral_second_derivative,
)
from opifex.physics.spectral.semilinear import (
    convection_nonlinearity,
    dealias_mask,
    first_derivative_operator,
    gradient_norm_nonlinearity,
    laplace_operator,
    rfft_wavenumbers,
    third_derivative_operator,
)
from opifex.physics.spectral.steppers import (
    solve_burgers_spectral,
    solve_kdv_spectral,
    solve_kuramoto_sivashinsky_spectral,
)


__all__ = [
    "ETDRK4Coefficients",
    "convection_nonlinearity",
    "dealias_mask",
    "etdrk4_coefficients",
    "etdrk4_step",
    "first_derivative_operator",
    "gradient_norm_nonlinearity",
    "integrate_etdrk4",
    "laplace_operator",
    "rfft_wavenumbers",
    "solve_burgers_spectral",
    "solve_kdv_spectral",
    "solve_kuramoto_sivashinsky_spectral",
    "spectral_gradient",
    "spectral_kinetic_energy",
    "spectral_momentum",
    "spectral_second_derivative",
    "third_derivative_operator",
]
