"""Exact and approximate Gaussian-Process regression for opifex.

Phase 11 Task 11.1 ships the foundational *exact conjugate-Gaussian*
GP regression surface on top of the Task 6.3 adapter specs at
:mod:`opifex.uncertainty.adapters.gp`. Subsequent slices will add deep
kernels, multi-output ICM/LCM, RFF approximations, natural-gradient
SVGP, heteroscedastic likelihoods, OAK kernels, and CARMA/Celerite/SHO
state-space kernels per the Phase-11 plan.

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; Algorithm 2.1 §2.2 (PRIMARY for
  ``exact``).
"""

from __future__ import annotations

from opifex.uncertainty.gp.exact import (
    ExactGPState,
    fit_exact_gp,
    predict_exact_gp,
    rbf_kernel,
)


__all__ = [
    "ExactGPState",
    "fit_exact_gp",
    "predict_exact_gp",
    "rbf_kernel",
]
