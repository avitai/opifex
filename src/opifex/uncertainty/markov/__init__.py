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


__all__ = [
    "MarkovLaplaceGPState",
    "fit_markov_laplace_gp",
    "predict_markov_laplace_gp",
]
