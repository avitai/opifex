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
    exact_gp_loocv_log_predictive,
    ExactGPState,
    fit_exact_gp,
    fit_heteroscedastic_exact_gp,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.gp.kernels import (
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    multi_output_icm_kernel,
    multi_output_lcm_kernel,
)
from opifex.uncertainty.gp.laplace_classification import (
    BernoulliLaplaceGPState,
    fit_bernoulli_laplace_gp,
    predict_bernoulli_laplace_gp,
)
from opifex.uncertainty.gp.rff import (
    fit_rff_gp,
    predict_rff_gp,
    rbf_random_fourier_features,
    RFFGPState,
)


__all__ = [
    "BernoulliLaplaceGPState",
    "ExactGPState",
    "RFFGPState",
    "exact_gp_loocv_log_predictive",
    "fit_bernoulli_laplace_gp",
    "fit_exact_gp",
    "fit_heteroscedastic_exact_gp",
    "fit_rff_gp",
    "matern12_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "multi_output_icm_kernel",
    "multi_output_lcm_kernel",
    "predict_bernoulli_laplace_gp",
    "predict_exact_gp",
    "predict_rff_gp",
    "rbf_kernel",
    "rbf_random_fourier_features",
]
