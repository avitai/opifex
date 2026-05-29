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
    additive_kernel,
    constrained_rbf_kernel,
    damped_oscillator_kernel,
    deep_kernel,
    graph_diffusion_kernel,
    matern12_kernel,
    matern32_kernel,
    matern52_kernel,
    multi_output_icm_kernel,
    multi_output_lcm_kernel,
    orthogonal_additive_kernel,
)
from opifex.uncertainty.gp.laplace import (
    fit_laplace_gp,
    LaplaceGPState,
    predict_laplace_latent_moments,
)
from opifex.uncertainty.gp.laplace_classification import (
    fit_bernoulli_laplace_gp,
    predict_bernoulli_laplace_gp,
)
from opifex.uncertainty.gp.laplace_likelihoods import (
    fit_beta_laplace_gp,
    fit_poisson_laplace_gp,
    fit_studentst_laplace_gp,
    predict_beta_laplace_gp,
    predict_poisson_laplace_gp,
    predict_studentst_laplace_gp,
)
from opifex.uncertainty.gp.rff import (
    fit_rff_gp,
    predict_rff_gp,
    rbf_random_fourier_features,
    RFFGPState,
)
from opifex.uncertainty.gp.svgp import (
    fit_svgp,
    predict_svgp,
    svgp_collapsed_elbo,
    SVGPState,
)


__all__ = [
    "ExactGPState",
    "LaplaceGPState",
    "RFFGPState",
    "SVGPState",
    "additive_kernel",
    "constrained_rbf_kernel",
    "damped_oscillator_kernel",
    "deep_kernel",
    "exact_gp_loocv_log_predictive",
    "fit_bernoulli_laplace_gp",
    "fit_beta_laplace_gp",
    "fit_exact_gp",
    "fit_heteroscedastic_exact_gp",
    "fit_laplace_gp",
    "fit_poisson_laplace_gp",
    "fit_rff_gp",
    "fit_studentst_laplace_gp",
    "fit_svgp",
    "graph_diffusion_kernel",
    "matern12_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "multi_output_icm_kernel",
    "multi_output_lcm_kernel",
    "orthogonal_additive_kernel",
    "predict_bernoulli_laplace_gp",
    "predict_beta_laplace_gp",
    "predict_exact_gp",
    "predict_laplace_latent_moments",
    "predict_poisson_laplace_gp",
    "predict_rff_gp",
    "predict_studentst_laplace_gp",
    "predict_svgp",
    "rbf_kernel",
    "rbf_random_fourier_features",
    "svgp_collapsed_elbo",
]
