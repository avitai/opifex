"""UQ capability declaration for the Bayesian-quadrature subpackage (Task 7.2).

Static, module-level constant — no import-time mutable side effects beyond
the constant itself (Rule 13).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_QUADRATURE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.BAYESIAN_QUADRATURE,
    source_package="emukit",
    notes=(
        "Bayesian-quadrature primitives — Bayesian Monte Carlo, vanilla "
        "BQ, WSABI-L, SOBER, Frank-Wolfe BQ — plus typed integral "
        "estimates and Gaussian / Lebesgue measure containers. Pure JAX "
        "kernels citing emukit (Rasmussen & Ghahramani NeurIPS 2003; "
        "Briol et al. Statistical Science 2019)."
    ),
)


QUADRATURE_CAPABILITIES: dict[str, UQCapability] = {
    "subpackage:quadrature": _QUADRATURE_CAPABILITY,
}


__all__ = ["QUADRATURE_CAPABILITIES"]
