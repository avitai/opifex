"""UQ capability declaration for the state-space subpackage (Task 7.2).

Static, module-level constant — no import-time mutable side effects beyond
the constant itself (Rule 13).
"""

from __future__ import annotations

from opifex.uncertainty.registry import DefaultStrategy, UQCapability


_STATESPACE_CAPABILITY = UQCapability(
    native_jax_kernel=True,
    default_strategy=DefaultStrategy.STATE_SPACE_FILTERING,
    source_package="opifex",
    notes=(
        "State-space math primitives — Kalman filter / smoother "
        "(sequential and parallel-scan), square-root Kalman, LTI-SDE "
        "discretization, diagonal EK1, Compute-Aware Kalman "
        "Filter / Smoother (CAKF / CAKS) per Pförtner et al. "
        "arXiv:2306.07879 (also arXiv:2405.08971), and state-space "
        "kernels (Matern 1/2 / 3/2 / 5/2 / 7/2, Periodic, Cosine, "
        "QuasiPeriodicMatern12). Pure JAX kernels."
    ),
)


STATESPACE_CAPABILITIES: dict[str, UQCapability] = {
    "subpackage:statespace": _STATESPACE_CAPABILITY,
}


__all__ = ["STATESPACE_CAPABILITIES"]
