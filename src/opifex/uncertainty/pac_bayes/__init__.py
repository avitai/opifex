"""PAC-Bayes objective + certificate subsystem.

Modules:

* :mod:`opifex.uncertainty.pac_bayes.bounds` — pure JAX bound formulas
  (McAllester, Catoni, quadratic / kl-inversion).
* :mod:`opifex.uncertainty.pac_bayes.objectives` —
  :func:`pac_bayes_kl_objective` differentiable training objective.
* :mod:`opifex.uncertainty.pac_bayes.certificates` —
  :class:`PACBayesCertificate` typed result and the
  :func:`pac_bayes_certificate` driver.

All bound formulas delegate KL computation to
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` (which itself
delegates to Artifex ``gaussian_kl_divergence`` for the standard-normal prior),
so the KL formula lives in exactly one place across the stack.

Canonical references (read-only):

* Dziugaite & Roy (2017) — ``arXiv:1703.11008``.
* Pérez-Ortiz et al. (JMLR v22) — empirical PAC-Bayes for deep networks.
* Alquier (2024) — ``arXiv:2110.11216`` (survey).
"""

from __future__ import annotations

from opifex.uncertainty.pac_bayes.bounds import (
    catoni_bound,
    kl_bernoulli,
    mcallester_bound,
    quadratic_bound,
)
from opifex.uncertainty.pac_bayes.certificates import (
    pac_bayes_certificate,
    PACBayesCertificate,
)
from opifex.uncertainty.pac_bayes.objectives import pac_bayes_kl_objective


__all__ = [
    "PACBayesCertificate",
    "catoni_bound",
    "kl_bernoulli",
    "mcallester_bound",
    "pac_bayes_certificate",
    "pac_bayes_kl_objective",
    "quadratic_bound",
]
