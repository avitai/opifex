"""Multi-fidelity emulation (Task 11.3).

Two canonical multi-fidelity surrogate families plus the multi-fidelity
acquisition pillar (MUMBO):

* :mod:`opifex.uncertainty.multi_fidelity.linear` — Kennedy & O'Hagan
  2000 AR(1) linear multi-fidelity GP (slice 33). Models each fidelity
  as ``f_i(x) = rho_i f_{i-1}(x) + delta_i(x)`` where each ``delta_i``
  is an independent GP.
* (Planned) :mod:`opifex.uncertainty.multi_fidelity.nonlinear` —
  Perdikaris, Raissi, Damianou, Lawrence, Karniadakis 2017 NARGP
  non-linear multi-fidelity emulator.
* (Planned) :mod:`opifex.uncertainty.multi_fidelity.acquisition` —
  Moss, Leslie, Rayson 2020 MUMBO max-value entropy acquisition for
  multi-fidelity Bayesian optimisation.

References
----------
* Kennedy, O'Hagan 2000 — *Predicting the output from a complex
  computer code when fast approximations are available*, Biometrika.
* Perdikaris, Raissi, Damianou, Lawrence, Karniadakis 2017 — *Nonlinear
  information fusion algorithms for data-efficient multi-fidelity
  modelling*, Proc. R. Soc. A.
* Moss, Leslie, Rayson 2020 — *MUMBO: MUlti-task Max-value Bayesian
  Optimisation*, ECML-PKDD.
"""

from __future__ import annotations

from opifex.uncertainty.multi_fidelity.linear import (
    fit_linear_multi_fidelity_gp,
    linear_multi_fidelity_kernel,
    LinearMultiFidelityGPState,
    predict_linear_multi_fidelity_gp,
)
from opifex.uncertainty.multi_fidelity.nonlinear import (
    fit_nonlinear_multi_fidelity_gp,
    NonLinearMultiFidelityGPState,
    predict_nonlinear_multi_fidelity_gp,
)


__all__ = [
    "LinearMultiFidelityGPState",
    "NonLinearMultiFidelityGPState",
    "fit_linear_multi_fidelity_gp",
    "fit_nonlinear_multi_fidelity_gp",
    "linear_multi_fidelity_kernel",
    "predict_linear_multi_fidelity_gp",
    "predict_nonlinear_multi_fidelity_gp",
]
