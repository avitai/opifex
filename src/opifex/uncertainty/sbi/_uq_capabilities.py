"""UQ capability declarations for the SBI subpackage (Task 8.2).

Static, module-level constants — no import-time mutable side effects.
The :data:`SBI_CAPABILITIES` table is consumed by the SBI subpackage's
``__init__`` to seed the singleton :class:`UQRegistry` and by capability
introspection tests.

Each of the three estimator surfaces (NPE / NLE / NRE) is an
inverse-problem-capable surface, so the plan's exit criterion mandates:

* ``supports_likelihood_free=True``
* ``default_strategy=DefaultStrategy.LIKELIHOOD_FREE_SBI``
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_NPE_CAPABILITY = UQCapability(
    native_nnx_module=True,
    supports_likelihood_free=True,
    default_strategy=DefaultStrategy.LIKELIHOOD_FREE_SBI,
    source_package="opifex",
    notes=(
        "NeuralPosteriorEstimator (NPE) — fits q(theta|x) with Artifex "
        "ConditionalRealNVP (Greenberg+ 2019, arXiv:1905.07488)."
    ),
)

_NLE_CAPABILITY = UQCapability(
    native_nnx_module=True,
    supports_likelihood_free=True,
    default_strategy=DefaultStrategy.LIKELIHOOD_FREE_SBI,
    source_package="opifex",
    notes=(
        "NeuralLikelihoodEstimator (NLE) — fits q(x|theta) with Artifex "
        "ConditionalRealNVP and runs posterior MCMC via BlackJAXBackend "
        "(Papamakarios+ 2019, arXiv:1805.07226)."
    ),
)

_NRE_CAPABILITY = UQCapability(
    native_nnx_module=True,
    supports_likelihood_free=True,
    default_strategy=DefaultStrategy.LIKELIHOOD_FREE_SBI,
    source_package="opifex",
    notes=(
        "NeuralRatioEstimator (NRE) — trains a contrastive classifier for "
        "log p(x|theta)/p(x) and runs posterior MCMC via BlackJAXBackend "
        "(Hermans+ 2020, arXiv:1903.04057)."
    ),
)


SBI_CAPABILITIES: dict[str, UQCapability] = {
    "sbi:npe": _NPE_CAPABILITY,
    "sbi:nle": _NLE_CAPABILITY,
    "sbi:nre": _NRE_CAPABILITY,
}


__all__ = ["SBI_CAPABILITIES"]
