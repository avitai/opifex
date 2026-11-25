"""Exact and approximate Gaussian-Process regression for opifex.

Provides exact conjugate-Gaussian GP regression, sparse variational GPs
(Titsias collapsed SVGP and Hensman natural-gradient SVGP), deep kernels,
multi-output ICM/LCM, random Fourier features, heteroscedastic likelihoods,
orthogonal additive kernels (OAK), and CARMA/Celerite/SHO state-space kernels
on top of the adapter specs at :mod:`opifex.uncertainty.adapters.gp`.

References:
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; Algorithm 2.1 §2.2 (PRIMARY for
  ``exact``).
"""

from __future__ import annotations

from opifex.uncertainty.gp._validation import PositiveScalar
from opifex.uncertainty.gp.exact import (
    exact_gp_loocv_log_predictive,
    ExactGPState,
    fit_exact_gp,
    fit_heteroscedastic_exact_gp,
    kernel_diagonal,
    predict_exact_gp,
    rbf_kernel,
)
from opifex.uncertainty.gp.kernels import (
    additive_kernel,
    carma_kernel,
    celerite_complex_kernel,
    constrained_rbf_kernel,
    damped_oscillator_kernel,
    deep_kernel,
    graph_diffusion_kernel,
    kernel_sum,
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
from opifex.uncertainty.gp.quasisep import (
    fit_quasisep_sho_gp,
    predict_quasisep_sho_gp,
    QuasisepGPState,
)
from opifex.uncertainty.gp.quasisep_carma import (
    fit_quasisep_carma_gp,
    predict_quasisep_carma_gp,
    QuasisepCarmaGPState,
)
from opifex.uncertainty.gp.rff import (
    fit_rff_gp,
    predict_rff_gp,
    rbf_random_fourier_features,
    RFFGPState,
)
from opifex.uncertainty.gp.spatiotemporal import (
    fit_spatiotemporal_vgp,
    predict_spatiotemporal_vgp,
    separable_spatiotemporal_kernel,
    spatiotemporal_vgp_log_marginal,
    SpatioTemporalGPState,
)
from opifex.uncertainty.gp.svgp import (
    fit_svgp,
    predict_svgp,
    svgp_collapsed_elbo,
    SVGPState,
)
from opifex.uncertainty.gp.svgp_stochastic import (
    bernoulli_log_likelihood,
    init_stochastic_svgp_state,
    natural_gradient_step,
    poisson_log_likelihood,
    predict_stochastic_svgp,
    stochastic_svgp_elbo,
    StochasticSVGPState,
)


__all__ = [
    "ExactGPState",
    "LaplaceGPState",
    "PositiveScalar",
    "QuasisepCarmaGPState",
    "QuasisepGPState",
    "RFFGPState",
    "SVGPState",
    "SpatioTemporalGPState",
    "StochasticSVGPState",
    "additive_kernel",
    "bernoulli_log_likelihood",
    "carma_kernel",
    "celerite_complex_kernel",
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
    "fit_quasisep_carma_gp",
    "fit_quasisep_sho_gp",
    "fit_rff_gp",
    "fit_spatiotemporal_vgp",
    "fit_studentst_laplace_gp",
    "fit_svgp",
    "graph_diffusion_kernel",
    "init_stochastic_svgp_state",
    "kernel_diagonal",
    "kernel_sum",
    "matern12_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "multi_output_icm_kernel",
    "multi_output_lcm_kernel",
    "natural_gradient_step",
    "orthogonal_additive_kernel",
    "poisson_log_likelihood",
    "predict_bernoulli_laplace_gp",
    "predict_beta_laplace_gp",
    "predict_exact_gp",
    "predict_laplace_latent_moments",
    "predict_poisson_laplace_gp",
    "predict_quasisep_carma_gp",
    "predict_quasisep_sho_gp",
    "predict_rff_gp",
    "predict_spatiotemporal_vgp",
    "predict_stochastic_svgp",
    "predict_studentst_laplace_gp",
    "predict_svgp",
    "rbf_kernel",
    "rbf_random_fourier_features",
    "separable_spatiotemporal_kernel",
    "spatiotemporal_vgp_log_marginal",
    "stochastic_svgp_elbo",
    "svgp_collapsed_elbo",
]
