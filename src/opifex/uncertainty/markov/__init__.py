"""Non-Gaussian inference on Markov GPs (Task 11.2).

Builds on the linear-Gaussian state-space layer in
:mod:`opifex.uncertainty.statespace` and the per-observation
``LikelihoodComponentsFn`` interface introduced for non-conjugate GP
inference in :mod:`opifex.uncertainty.gp.laplace` (Task 11.1 D5).

References
----------
* Sarkka 2013 — *Bayesian Filtering and Smoothing*, CUP (state-space
  GPs + iterated extended Kalman smoothing).
* Wilkinson, Solin, Adam 2020+ — ``bayesnewton`` (primary reference
  for the inference algorithm catalogue: PEP / VI / Laplace /
  Posterior Linearisation / Newton).
* Solin, Hensman, Turner 2018 — *Infinite-horizon Gaussian
  processes*, NeurIPS (steady-state Kalman variants — deferred).
"""

from __future__ import annotations

from opifex.uncertainty.markov.markov_laplace import (
    fit_markov_laplace_gp,
    MarkovLaplaceGPState,
    predict_markov_laplace_gp,
)
from opifex.uncertainty.markov.markov_laplace_likelihoods import (
    fit_bernoulli_markov_laplace_gp,
    fit_beta_markov_laplace_gp,
    fit_gaussian_markov_laplace_gp,
    fit_poisson_markov_laplace_gp,
    fit_studentst_markov_laplace_gp,
    predict_bernoulli_markov_laplace_gp,
    predict_beta_markov_laplace_gp,
    predict_gaussian_markov_laplace_gp,
    predict_poisson_markov_laplace_gp,
    predict_studentst_markov_laplace_gp,
)
from opifex.uncertainty.markov.markov_pep import (
    fit_markov_pep_gp,
    MarkovPEPGPState,
    predict_markov_pep_gp,
)
from opifex.uncertainty.markov.markov_pep_likelihoods import (
    fit_bernoulli_markov_pep_gp,
    fit_beta_markov_pep_gp,
    fit_gaussian_markov_pep_gp,
    fit_poisson_markov_pep_gp,
    fit_studentst_markov_pep_gp,
    predict_bernoulli_markov_pep_gp,
    predict_beta_markov_pep_gp,
    predict_gaussian_markov_pep_gp,
    predict_poisson_markov_pep_gp,
    predict_studentst_markov_pep_gp,
)
from opifex.uncertainty.markov.markov_pl import (
    fit_markov_pl_gp,
    MarkovPLGPState,
    predict_markov_pl_gp,
)
from opifex.uncertainty.markov.markov_pl_likelihoods import (
    fit_bernoulli_markov_pl_gp,
    fit_beta_markov_pl_gp,
    fit_gaussian_markov_pl_gp,
    fit_poisson_markov_pl_gp,
    fit_studentst_markov_pl_gp,
    predict_bernoulli_markov_pl_gp,
    predict_beta_markov_pl_gp,
    predict_gaussian_markov_pl_gp,
    predict_poisson_markov_pl_gp,
    predict_studentst_markov_pl_gp,
)
from opifex.uncertainty.markov.markov_vi import (
    fit_markov_vi_gp,
    MarkovVIGPState,
    predict_markov_vi_gp,
)
from opifex.uncertainty.markov.markov_vi_likelihoods import (
    fit_bernoulli_markov_vi_gp,
    fit_beta_markov_vi_gp,
    fit_gaussian_markov_vi_gp,
    fit_poisson_markov_vi_gp,
    fit_studentst_markov_vi_gp,
    predict_bernoulli_markov_vi_gp,
    predict_beta_markov_vi_gp,
    predict_gaussian_markov_vi_gp,
    predict_poisson_markov_vi_gp,
    predict_studentst_markov_vi_gp,
)


__all__ = [
    "MarkovLaplaceGPState",
    "MarkovPEPGPState",
    "MarkovPLGPState",
    "MarkovVIGPState",
    "fit_bernoulli_markov_laplace_gp",
    "fit_bernoulli_markov_pep_gp",
    "fit_bernoulli_markov_pl_gp",
    "fit_bernoulli_markov_vi_gp",
    "fit_beta_markov_laplace_gp",
    "fit_beta_markov_pep_gp",
    "fit_beta_markov_pl_gp",
    "fit_beta_markov_vi_gp",
    "fit_gaussian_markov_laplace_gp",
    "fit_gaussian_markov_pep_gp",
    "fit_gaussian_markov_pl_gp",
    "fit_gaussian_markov_vi_gp",
    "fit_markov_laplace_gp",
    "fit_markov_pep_gp",
    "fit_markov_pl_gp",
    "fit_markov_vi_gp",
    "fit_poisson_markov_laplace_gp",
    "fit_poisson_markov_pep_gp",
    "fit_poisson_markov_pl_gp",
    "fit_poisson_markov_vi_gp",
    "fit_studentst_markov_laplace_gp",
    "fit_studentst_markov_pep_gp",
    "fit_studentst_markov_pl_gp",
    "fit_studentst_markov_vi_gp",
    "predict_bernoulli_markov_laplace_gp",
    "predict_bernoulli_markov_pep_gp",
    "predict_bernoulli_markov_pl_gp",
    "predict_bernoulli_markov_vi_gp",
    "predict_beta_markov_laplace_gp",
    "predict_beta_markov_pep_gp",
    "predict_beta_markov_pl_gp",
    "predict_beta_markov_vi_gp",
    "predict_gaussian_markov_laplace_gp",
    "predict_gaussian_markov_pep_gp",
    "predict_gaussian_markov_pl_gp",
    "predict_gaussian_markov_vi_gp",
    "predict_markov_laplace_gp",
    "predict_markov_pep_gp",
    "predict_markov_pl_gp",
    "predict_markov_vi_gp",
    "predict_poisson_markov_laplace_gp",
    "predict_poisson_markov_pep_gp",
    "predict_poisson_markov_pl_gp",
    "predict_poisson_markov_vi_gp",
    "predict_studentst_markov_laplace_gp",
    "predict_studentst_markov_pep_gp",
    "predict_studentst_markov_pl_gp",
    "predict_studentst_markov_vi_gp",
]
