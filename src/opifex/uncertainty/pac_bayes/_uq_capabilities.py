"""UQ capability declarations for the PAC-Bayes subpackage.

Static, module-level constants, collected into
:data:`opifex.capabilities.UQ_CAPABILITIES`; importing this module changes no global state.

Single registered surface:

* ``pac_bayes:certificate`` — the :func:`pac_bayes_certificate` driver
  + accompanying bound formulas (McAllester, Catoni, quadratic) and the
  :func:`pac_bayes_kl_objective` differentiable training objective.
  Advertises :attr:`supports_pac_bayes_certificate=True` and
  :attr:`default_strategy=DefaultStrategy.PAC_BAYES`.

References:
* Dziugaite & Roy (2017) — ``arXiv:1703.11008``.
* Pérez-Ortiz et al. (JMLR v22) — empirical PAC-Bayes for deep networks.
* Alquier (2024) — ``arXiv:2110.11216`` (survey).
"""

from __future__ import annotations

from opifex.core.capabilities import DefaultStrategy, UQCapability


_PAC_BAYES_CERTIFICATE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    supports_pac_bayes_certificate=True,
    default_strategy=DefaultStrategy.PAC_BAYES,
    source_package="opifex",
    notes=(
        "PAC-Bayes certificate driver + bound formulas (McAllester, "
        "Catoni, quadratic / kl-inversion) and the pac_bayes_kl_objective "
        "differentiable training objective. KL computation delegates to "
        "opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl "
        "(Artifex gaussian_kl_divergence under the hood). References: "
        "Dziugaite & Roy (arXiv:1703.11008); Pérez-Ortiz et al. (JMLR v22); "
        "Alquier survey (arXiv:2110.11216)."
    ),
)


PAC_BAYES_CAPABILITIES: dict[str, UQCapability] = {
    "pac_bayes:certificate": _PAC_BAYES_CERTIFICATE_CAPABILITY,
}


__all__ = ["PAC_BAYES_CAPABILITIES"]
