"""UQ capability declarations for the PAC-Bayes subpackage (Task 8.1).

Static, module-level constants — no import-time mutable side effects.
The :data:`PAC_BAYES_CAPABILITIES` table is consumed by the PAC-Bayes
subpackage's ``__init__`` to seed the singleton :class:`UQRegistry`.

Plan reference: ``08-phase-pac-bayes-sbi-active-stochastic-fields.md``
lines 755-790 — Task 8.5 flips the per-subsystem capability flag for
the PAC-Bayes certificate driver and its bound formulas.

Single registered surface:

* ``pac_bayes:certificate`` — the :func:`pac_bayes_certificate` driver
  + accompanying bound formulas (McAllester, Catoni, quadratic) and the
  :func:`pac_bayes_kl_objective` differentiable training objective.
  Advertises :attr:`supports_pac_bayes_certificate=True` and
  :attr:`default_strategy=DefaultStrategy.PAC_BAYES`.

Canonical references (read-only):

* Dziugaite & Roy (2017) — ``arXiv:1703.11008``.
* Pérez-Ortiz et al. (JMLR v22) — empirical PAC-Bayes for deep networks.
* Alquier (2024) — ``arXiv:2110.11216`` (survey).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


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
